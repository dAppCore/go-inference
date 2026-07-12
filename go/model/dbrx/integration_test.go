// SPDX-Licence-Identifier: EUPL-1.2

package dbrx

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/safetensors"
)

type seededWeights struct{ state uint32 }

func (s *seededWeights) values(n int, fill float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		s.state = 1664525*s.state + 1013904223
		out[i] = fill + float32(int32(s.state>>24)-128)/512
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

func tinyDBRXWeights(fill float32) map[string]safetensors.Tensor {
	const hidden, vocab, expertFF, heads, kvHeads, headDim, experts = 8, 32, 12, 2, 1, 4, 4
	s := seededWeights{state: 0xd8b01234}
	ones := make([]float32, hidden)
	for i := range ones {
		ones[i] = 1
	}
	return map[string]safetensors.Tensor{
		"transformer.wte.weight": f32Tensor(s.values(vocab*hidden, fill), vocab, hidden), "transformer.norm_f.weight": f32Tensor(ones, hidden),
		"lm_head.weight": f32Tensor(s.values(vocab*hidden, fill), vocab, hidden),
		"transformer.blocks.0.norm_attn_norm.norm_1.weight": f32Tensor(ones, hidden), "transformer.blocks.0.norm_attn_norm.norm_2.weight": f32Tensor(ones, hidden),
		"transformer.blocks.0.norm_attn_norm.attn.Wqkv.weight":     f32Tensor(s.values((heads+2*kvHeads)*headDim*hidden, fill), (heads+2*kvHeads)*headDim, hidden),
		"transformer.blocks.0.norm_attn_norm.attn.out_proj.weight": f32Tensor(s.values(hidden*hidden, fill), hidden, hidden),
		"transformer.blocks.0.ffn.router.layer.weight":             f32Tensor(s.values(experts*hidden, fill), experts, hidden),
		"transformer.blocks.0.ffn.experts.mlp.w1":                  f32Tensor(s.values(experts*expertFF*hidden, fill), experts, expertFF, hidden),
		"transformer.blocks.0.ffn.experts.mlp.v1":                  f32Tensor(s.values(experts*expertFF*hidden, -fill), experts, expertFF, hidden),
		"transformer.blocks.0.ffn.experts.mlp.w2":                  f32Tensor(s.values(experts*hidden*expertFF, fill), experts, hidden, expertFF),
	}
}

func TestTinyDBRXForward_Good(t *testing.T) {
	config := []byte(`{"model_type":"dbrx","d_model":8,"n_heads":2,"n_layers":1,"vocab_size":32,"attn_config":{"kv_n_heads":1,"rope_theta":10000},"ffn_config":{"ffn_hidden_size":12,"moe_num_experts":4,"moe_top_k":2}}`)
	spec, ok := model.LookupArch("dbrx")
	if !ok {
		t.Fatal("DBRX architecture not registered")
	}
	var cfg Config
	if !core.JSONUnmarshal(config, &cfg).OK {
		t.Fatal("tiny config parse")
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	cm, err := composed.LoadComposedWithArch(NormalizeWeights(tinyDBRXWeights(0.02), cfg), loaderJSON(cfg), arch)
	if err != nil {
		t.Fatalf("inspect tiny DBRX: %v", err)
	}
	moe, ok := cm.Layers[0].MLP.(*composed.MoEMLP)
	if !ok || moe.TopK != 2 || moe.NormTopKProb || moe.Shared != nil {
		t.Fatalf("router receipt = %T top-k %d normalise %v", cm.Layers[0].MLP, moe.TopK, moe.NormTopKProb)
	}
	tm, err := spec.Composed(tinyDBRXWeights(0.02), config)
	if err != nil {
		t.Fatalf("load tiny DBRX: %v", err)
	}
	inputs := make([][]byte, 4)
	for i, token := range []int32{1, 7, 19, 23} {
		inputs[i], err = tm.Embed(token)
		if err != nil {
			t.Fatalf("embed %d: %v", token, err)
		}
	}
	output, err := tm.DecodeForward(inputs)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(output) != 4 || len(output[0]) != 16 || string(output[0]) == string(output[3]) {
		t.Fatalf("varied-fill forward receipt shape/variation failed")
	}
	other, err := spec.Composed(tinyDBRXWeights(-0.03), config)
	if err != nil {
		t.Fatalf("load second fill: %v", err)
	}
	input, _ := other.Embed(7)
	second, err := other.DecodeForward([][]byte{input})
	if err != nil {
		t.Fatalf("second forward: %v", err)
	}
	if string(second[0]) == string(output[1]) {
		t.Fatal("different seeded fills produced identical router/forward receipt")
	}
}
