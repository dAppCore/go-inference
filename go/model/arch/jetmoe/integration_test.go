// SPDX-Licence-Identifier: EUPL-1.2

package jetmoe

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/safetensors"
)

type seededValues struct{ state uint32 }

func (s *seededValues) values(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		s.state = 1664525*s.state + 1013904223
		out[i] = float32(int32(s.state>>24)-128) / 256
	}
	return out
}

func jetF32Tensor(values []float32, shape ...int) safetensors.Tensor {
	data := make([]byte, len(values)*4)
	for i, value := range values {
		bits := math.Float32bits(value)
		data[4*i], data[4*i+1], data[4*i+2], data[4*i+3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
	}
	return safetensors.Tensor{Dtype: "F32", Shape: shape, Data: data}
}

func tinyJetMoEFFN() (map[string]safetensors.Tensor, Config, []byte) {
	const hidden, ff, experts, vocab, heads, kvHeads, headDim = 8, 12, 4, 32, 2, 1, 4
	s := seededValues{state: 0x0e7a0e}
	tensors := map[string]safetensors.Tensor{
		"model.embed_tokens.weight":                      jetF32Tensor(s.values(vocab*hidden), vocab, hidden),
		"model.norm.weight":                              jetF32Tensor(s.values(hidden), hidden),
		"lm_head.weight":                                 jetF32Tensor(s.values(vocab*hidden), vocab, hidden),
		"model.layers.0.input_layernorm.weight":          jetF32Tensor(s.values(hidden), hidden),
		"model.layers.0.post_attention_layernorm.weight": jetF32Tensor(s.values(hidden), hidden),
		"model.layers.0.mlp.input_linear.weight":         jetF32Tensor(s.values(experts*2*ff*hidden), experts, 2*ff, hidden),
		"model.layers.0.mlp.output_linear.weight":        jetF32Tensor(s.values(experts*hidden*ff), experts, hidden, ff),
		"model.layers.0.mlp.router.layer.weight":         jetF32Tensor(s.values(experts*hidden), experts, hidden),
		"model.layers.0.self_attn.q_proj.weight":         jetF32Tensor(s.values(heads*headDim*hidden), heads*headDim, hidden),
		"model.layers.0.self_attn.k_proj.weight":         jetF32Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.v_proj.weight":         jetF32Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.o_proj.weight":         jetF32Tensor(s.values(hidden*heads*headDim), hidden, heads*headDim),
	}
	cfg := Config{ModelType: "jetmoe", HiddenSize: hidden, FFNHiddenSize: ff, NumHiddenLayers: 1, NumAttentionHeads: heads, NumKeyValueHeads: kvHeads, KVChannels: headDim, MoENumExperts: experts, MoETopK: 2, VocabSize: vocab, RMSNormEps: 1e-5, RopeTheta: 10_000, RotaryPercent: 1}
	config := []byte(`{"model_type":"jetmoe","hidden_size":8,"ffn_hidden_size":12,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"kv_channels":4,"moe_num_experts":4,"moe_top_k":2,"vocab_size":32,"rms_norm_eps":1e-5,"rope_theta":10000,"rotary_percent":1}`)
	return tensors, cfg, config
}

func TestTinyJetMoEFFNForward_Good(t *testing.T) {
	tensors, cfg, config := tinyJetMoEFFN()
	adapted, err := adaptFFNWeights(tensors, cfg)
	if err != nil {
		t.Fatal(err)
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatal(err)
	}
	cm, err := composed.LoadComposedWithArch(adapted, config, arch)
	if err != nil {
		t.Fatal(err)
	}
	moe, ok := cm.Layers[0].MLP.(*composed.MoEMLP)
	if !ok || moe.TopK != 2 || !moe.NormTopKProb || moe.Shared != nil {
		t.Fatalf("JetMoE FFN = %T top-k %d normalise %v shared %v", cm.Layers[0].MLP, moe.TopK, moe.NormTopKProb, moe.Shared != nil)
	}
	routerSets := map[[2]int]bool{}
	for token := 1; token <= 12; token++ {
		x := cm.Embed[token*cfg.HiddenSize : (token+1)*cfg.HiddenSize]
		best, second := -1, -1
		for expert := 0; expert < cfg.MoENumExperts; expert++ {
			var score float32
			for d := 0; d < cfg.HiddenSize; d++ {
				score += x[d] * moe.Router[expert*cfg.HiddenSize+d]
			}
			if best < 0 || score > routerScore(x, moe.Router, best, cfg.HiddenSize) {
				second, best = best, expert
			} else if second < 0 || score > routerScore(x, moe.Router, second, cfg.HiddenSize) {
				second = expert
			}
		}
		routerSets[[2]int{best, second}] = true
	}
	if len(routerSets) < 2 {
		t.Fatalf("router distribution receipt = %v, want at least two top-2 selections", routerSets)
	}
	tm := composed.NewTokenModel(cm)
	tokens := []int32{1, 7, 19, 23}
	inputs := make([][]byte, len(tokens))
	for i, token := range tokens {
		inputs[i], err = tm.Embed(token)
		if err != nil {
			t.Fatal(err)
		}
	}
	output, err := tm.DecodeForward(inputs)
	if err != nil {
		t.Fatal(err)
	}
	if len(output) != len(tokens) || string(output[0]) == string(output[len(output)-1]) {
		t.Fatalf("seeded forward rows = %d varied=%v", len(output), string(output[0]) != string(output[len(output)-1]))
	}
	t.Logf("JetMoE FFN receipt: %d distinct top-2 routes across 12 tokens; varied 4-token forward", len(routerSets))
}

func routerScore(x, router []float32, expert, hidden int) float32 {
	var score float32
	for d := 0; d < hidden; d++ {
		score += x[d] * router[expert*hidden+d]
	}
	return score
}

func TestTinyJetMoEFFNForward_Bad(t *testing.T) {
	tensors, cfg, _ := tinyJetMoEFFN()
	delete(tensors, "model.layers.0.mlp.router.layer.weight")
	if _, err := adaptFFNWeights(tensors, cfg); err == nil {
		t.Fatal("missing router accepted")
	}
}

func TestTinyJetMoEFFNForward_Ugly(t *testing.T) {
	tensors, cfg, _ := tinyJetMoEFFN()
	bad := tensors["model.layers.0.mlp.output_linear.weight"]
	bad.Data = bad.Data[:len(bad.Data)-4]
	tensors["model.layers.0.mlp.output_linear.weight"] = bad
	if _, err := adaptFFNWeights(tensors, cfg); err == nil {
		t.Fatal("truncated expert tensor accepted")
	}
}
