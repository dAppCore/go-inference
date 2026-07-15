// SPDX-Licence-Identifier: EUPL-1.2

package qwenmoe_test

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	_ "dappco.re/go/inference/model/qwenmoe"
	"dappco.re/go/inference/model/safetensors"
)

type seededWeights struct{ state uint32 }

func (s *seededWeights) values(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		s.state = 1664525*s.state + 1013904223
		out[i] = float32(int32(s.state>>24)-128) / 512
	}
	return out
}

func f32Tensor(values []float32, shape ...int) safetensors.Tensor {
	data := make([]byte, len(values)*4)
	for i, value := range values {
		bits := math.Float32bits(value)
		data[4*i], data[4*i+1], data[4*i+2], data[4*i+3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
	}
	return safetensors.Tensor{Dtype: "F32", Shape: shape, Data: data}
}

func tinyQwen2MoEWeights() (map[string]safetensors.Tensor, []float32) {
	const hidden, vocab, expertFF, sharedFF, heads, kvHeads, headDim, experts = 8, 32, 6, 10, 2, 1, 4, 4
	s := seededWeights{state: 0x2a3e1234}
	router := s.values(experts * hidden)
	tensors := map[string]safetensors.Tensor{
		"model.embed_tokens.weight":                         f32Tensor(s.values(vocab*hidden), vocab, hidden),
		"model.norm.weight":                                 f32Tensor(s.values(hidden), hidden),
		"lm_head.weight":                                    f32Tensor(s.values(vocab*hidden), vocab, hidden),
		"model.layers.0.input_layernorm.weight":             f32Tensor(s.values(hidden), hidden),
		"model.layers.0.post_attention_layernorm.weight":    f32Tensor(s.values(hidden), hidden),
		"model.layers.0.self_attn.q_proj.weight":            f32Tensor(s.values(heads*headDim*hidden), heads*headDim, hidden),
		"model.layers.0.self_attn.k_proj.weight":            f32Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.v_proj.weight":            f32Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.o_proj.weight":            f32Tensor(s.values(hidden*heads*headDim), hidden, heads*headDim),
		"model.layers.0.mlp.gate.weight":                    f32Tensor(router, experts, hidden),
		"model.layers.0.mlp.shared_expert.gate_proj.weight": f32Tensor(s.values(sharedFF*hidden), sharedFF, hidden),
		"model.layers.0.mlp.shared_expert.up_proj.weight":   f32Tensor(s.values(sharedFF*hidden), sharedFF, hidden),
		"model.layers.0.mlp.shared_expert.down_proj.weight": f32Tensor(s.values(hidden*sharedFF), hidden, sharedFF),
		"model.layers.0.mlp.shared_expert_gate.weight":      f32Tensor(s.values(hidden), 1, hidden),
	}
	for expert := range experts {
		prefix := core.Sprintf("model.layers.0.mlp.experts.%d.", expert)
		tensors[prefix+"gate_proj.weight"] = f32Tensor(s.values(expertFF*hidden), expertFF, hidden)
		tensors[prefix+"up_proj.weight"] = f32Tensor(s.values(expertFF*hidden), expertFF, hidden)
		tensors[prefix+"down_proj.weight"] = f32Tensor(s.values(hidden*expertFF), hidden, expertFF)
	}
	return tensors, router
}

func TestQwen2MoEForward_Good(t *testing.T) {
	const hidden, experts, topK = 8, 4, 2
	tensors, router := tinyQwen2MoEWeights()
	config := []byte(`{"model_type":"qwen2_moe","hidden_size":8,"intermediate_size":10,"moe_intermediate_size":6,"shared_expert_intermediate_size":10,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"num_experts":4,"num_experts_per_tok":2,"vocab_size":32,"norm_topk_prob":false,"tie_word_embeddings":false}`)
	spec, ok := model.LookupArch("qwen2_moe")
	if !ok {
		t.Fatal("qwen2_moe architecture not registered")
	}
	parsed, err := spec.Parse(config)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	arch, err := parsed.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	assembled, err := composed.LoadComposedWithArch(tensors, config, arch)
	if err != nil {
		t.Fatalf("assemble: %v", err)
	}
	moe, ok := assembled.Layers[0].MLP.(*composed.MoEMLP)
	if !ok {
		t.Fatalf("layer FFN = %T, want *composed.MoEMLP", assembled.Layers[0].MLP)
	}
	if moe.TopK != topK || moe.NormTopKProb || moe.Shared == nil || len(moe.SharedGate) != hidden {
		t.Fatalf("MoE receipt = type %T top-k %d normalise %v shared %v shared gate %d", assembled.Layers[0].MLP, moe.TopK, moe.NormTopKProb, moe.Shared != nil, len(moe.SharedGate))
	}
	tm, err := spec.Composed(tensors, config)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	tokens := []int32{1, 7, 19, 23, 5, 29}
	inputs := make([][]byte, len(tokens))
	distributions := map[[topK]int]bool{}
	for i, token := range tokens {
		inputs[i], err = tm.Embed(token)
		if err != nil {
			t.Fatalf("embed %d: %v", token, err)
		}
		fill := make([]float32, hidden)
		for d := range fill {
			fill[d] = float32((i+1)*(d+3)%11-5) / 7
		}
		best := [topK]int{-1, -1}
		bestScore := [topK]float32{-1e30, -1e30}
		for expert := range experts {
			var score float32
			for d := range hidden {
				score += fill[d] * router[expert*hidden+d]
			}
			if score > bestScore[0] {
				best[1], bestScore[1] = best[0], bestScore[0]
				best[0], bestScore[0] = expert, score
			} else if score > bestScore[1] {
				best[1], bestScore[1] = expert, score
			}
		}
		distributions[best] = true
	}
	if len(distributions) < 2 {
		t.Fatalf("varied fills selected %d router distributions, want at least 2", len(distributions))
	}
	output, err := tm.DecodeForward(inputs)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(output) != len(tokens) || len(output[0]) != hidden*2 || string(output[0]) == string(output[len(output)-1]) {
		t.Fatalf("seeded forward receipt = rows %d bytes %d varied %v", len(output), len(output[0]), string(output[0]) != string(output[len(output)-1]))
	}
	t.Logf("router distribution receipt: %d distinct top-%d expert pairs from %d varied fills", len(distributions), topK, len(tokens))
}

func TestQwen3MoEForward_Good(t *testing.T) {
	const hidden = 8
	tensors, _ := tinyQwen2MoEWeights()
	delete(tensors, "model.layers.0.mlp.shared_expert.gate_proj.weight")
	delete(tensors, "model.layers.0.mlp.shared_expert.up_proj.weight")
	delete(tensors, "model.layers.0.mlp.shared_expert.down_proj.weight")
	delete(tensors, "model.layers.0.mlp.shared_expert_gate.weight")
	config := []byte(`{"model_type":"qwen3_moe","hidden_size":8,"intermediate_size":10,"moe_intermediate_size":6,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"num_experts":4,"num_experts_per_tok":2,"vocab_size":32,"norm_topk_prob":true,"tie_word_embeddings":false}`)
	spec, ok := model.LookupArch("qwen3_moe")
	if !ok {
		t.Fatal("qwen3_moe architecture not registered")
	}
	parsed, err := spec.Parse(config)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	arch, err := parsed.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	assembled, err := composed.LoadComposedWithArch(tensors, config, arch)
	if err != nil {
		t.Fatalf("assemble: %v", err)
	}
	moe, ok := assembled.Layers[0].MLP.(*composed.MoEMLP)
	if !ok {
		t.Fatalf("layer FFN = %T, want *composed.MoEMLP", assembled.Layers[0].MLP)
	}
	if !moe.NormTopKProb || moe.Shared != nil || len(moe.SharedGate) != 0 {
		t.Fatalf("Qwen3-MoE policy = normalise %v shared %v gate %d", moe.NormTopKProb, moe.Shared != nil, len(moe.SharedGate))
	}
	tm, err := spec.Composed(tensors, config)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	input, err := tm.Embed(11)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	output, err := tm.DecodeForward([][]byte{input})
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(output) != 1 || len(output[0]) != hidden*2 {
		t.Fatalf("forward shape = [%d,%d], want [1,%d]", len(output), len(output[0]), hidden*2)
	}
}
