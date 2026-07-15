// SPDX-Licence-Identifier: EUPL-1.2

package olmoe_test

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	_ "dappco.re/go/inference/model/arch/allenai/olmoe"
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

func tinyOLMoEWeights() map[string]safetensors.Tensor {
	const hidden, vocab, expertFF, heads, kvHeads, headDim, experts = 8, 32, 12, 2, 1, 4, 4
	s := seededWeights{state: 0x01e0e123}
	embed := s.values(vocab * hidden)
	router := s.values(experts * hidden)
	tensors := map[string]safetensors.Tensor{
		"model.embed_tokens.weight":                      f32Tensor(embed, vocab, hidden),
		"model.norm.weight":                              f32Tensor(s.values(hidden), hidden),
		"lm_head.weight":                                 f32Tensor(s.values(vocab*hidden), vocab, hidden),
		"model.layers.0.input_layernorm.weight":          f32Tensor(s.values(hidden), hidden),
		"model.layers.0.post_attention_layernorm.weight": f32Tensor(s.values(hidden), hidden),
		"model.layers.0.self_attn.q_proj.weight":         f32Tensor(s.values(heads*headDim*hidden), heads*headDim, hidden),
		"model.layers.0.self_attn.k_proj.weight":         f32Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.v_proj.weight":         f32Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.o_proj.weight":         f32Tensor(s.values(hidden*heads*headDim), hidden, heads*headDim),
		"model.layers.0.self_attn.q_norm.weight":         f32Tensor(s.values(headDim), headDim),
		"model.layers.0.self_attn.k_norm.weight":         f32Tensor(s.values(headDim), headDim),
		"model.layers.0.mlp.gate.weight":                 f32Tensor(router, experts, hidden),
	}
	for expert := range experts {
		prefix := core.Sprintf("model.layers.0.mlp.experts.%d", expert)
		tensors[prefix+".gate_proj.weight"] = f32Tensor(s.values(expertFF*hidden), expertFF, hidden)
		tensors[prefix+".down_proj.weight"] = f32Tensor(s.values(hidden*expertFF), hidden, expertFF)
		tensors[prefix+".up_proj.weight"] = f32Tensor(s.values(expertFF*hidden), expertFF, hidden)
	}
	return tensors
}

func TestTinyOLMoEForward_Good(t *testing.T) {
	const hidden = 8
	tensors := tinyOLMoEWeights()
	config := []byte(`{"model_type":"olmoe","hidden_size":8,"intermediate_size":12,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"num_experts":4,"num_experts_per_tok":2,"vocab_size":32,"rms_norm_eps":1e-5,"rope_theta":10000,"norm_topk_prob":false,"tie_word_embeddings":false}`)
	spec, ok := model.LookupArch("olmoe")
	if !ok {
		t.Fatal("OLMoE architecture not registered")
	}
	policy := model.Arch{Experts: 4, TopK: 2, MoEGating: model.MoEGatingSoftmax, NormaliseMoETopK: false, SharedExperts: 0}
	cm, err := composed.LoadComposedWithArch(tensors, config, policy)
	if err != nil {
		t.Fatalf("inspect tiny OLMoE: %v", err)
	}
	moe, ok := cm.Layers[0].MLP.(*composed.MoEMLP)
	if !ok {
		t.Fatalf("layer 0 FFN = %T, want *composed.MoEMLP", cm.Layers[0].MLP)
	}
	if moe.TopK != 2 || moe.NormTopKProb || moe.Shared != nil {
		t.Fatalf("router parameters = top-k %d normalise %v shared %v, want 2 false nil", moe.TopK, moe.NormTopKProb, moe.Shared != nil)
	}
	tm, err := spec.Composed(tensors, config)
	if err != nil {
		t.Fatalf("load tiny OLMoE: %v", err)
	}
	tokens := []int32{1, 7, 19, 23}
	inputs := make([][]byte, len(tokens))
	for i, token := range tokens {
		inputs[i], err = tm.Embed(token)
		if err != nil {
			t.Fatalf("embed token %d: %v", token, err)
		}
	}
	output, err := tm.DecodeForward(inputs)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(output) != len(tokens) || len(output[0]) != hidden*2 {
		t.Fatalf("forward shape = [%d,%d], want [%d,%d bf16 bytes]", len(output), len(output[0]), len(tokens), hidden*2)
	}
	if string(output[0]) == string(output[len(output)-1]) {
		t.Fatal("varied seeded tokens produced identical first and last hidden rows")
	}
}
