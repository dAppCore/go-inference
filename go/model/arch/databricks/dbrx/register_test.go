// SPDX-Licence-Identifier: EUPL-1.2

package dbrx

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

func coreJSONUnmarshal(data []byte, value any) bool { return core.JSONUnmarshal(data, value).OK }

func TestRegister_NormalizeWeights_Good(t *testing.T) {
	// Representative entries copied from the published 29 KB INDEX only (the pack is 263 GB):
	// https://huggingface.co/databricks/dbrx-instruct/blob/main/model.safetensors.index.json
	cfg := Config{DModel: 8, Heads: 2, Layers: 1, Attention: AttentionConfig{KVHeads: 1}, FFN: FFNConfig{HiddenSize: 6, Experts: 2, TopK: 1}}
	in := map[string]safetensors.Tensor{
		"transformer.wte.weight": {Shape: []int{16, 8}}, "transformer.norm_f.weight": {Shape: []int{8}},
		"transformer.blocks.0.norm_attn_norm.attn.Wqkv.weight":     {Dtype: "F32", Shape: []int{16, 8}, Data: make([]byte, 16*8*4)},
		"transformer.blocks.0.norm_attn_norm.attn.out_proj.weight": {Shape: []int{8, 8}},
		"transformer.blocks.0.norm_attn_norm.norm_1.weight":        {Shape: []int{8}}, "transformer.blocks.0.norm_attn_norm.norm_2.weight": {Shape: []int{8}},
		"transformer.blocks.0.ffn.router.layer.weight": {Shape: []int{2, 8}},
		"transformer.blocks.0.ffn.experts.mlp.w1":      {Dtype: "F32", Shape: []int{2 * 6, 8}, Data: make([]byte, 2*6*8*4)},
		"transformer.blocks.0.ffn.experts.mlp.v1":      {Dtype: "F32", Shape: []int{2 * 6, 8}, Data: make([]byte, 2*6*8*4)},
		"transformer.blocks.0.ffn.experts.mlp.w2":      {Dtype: "F32", Shape: []int{2 * 6, 8}, Data: make([]byte, 2*6*8*4)},
	}
	got := NormalizeWeights(in, cfg)
	for _, name := range []string{"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.v_proj.weight", "model.layers.0.mlp.gate.weight", "model.layers.0.mlp.experts.1.up_proj.weight", "model.layers.0.mlp.experts.1.down_proj.weight"} {
		if _, ok := got[name]; !ok {
			t.Fatalf("normalised tensor %q absent", name)
		}
	}
	if shape := got["model.layers.0.mlp.experts.1.down_proj.weight"].Shape; len(shape) != 2 || shape[0] != 8 || shape[1] != 6 {
		t.Fatalf("transposed packed w2 shape = %v", shape)
	}
}

func TestRegister_NormalizeWeights_Bad(t *testing.T) {
	got := NormalizeWeights(map[string]safetensors.Tensor{"transformer.blocks.0.ffn.experts.mlp.w1": {Shape: []int{3, 6, 8}}}, Config{Layers: 1, FFN: FFNConfig{Experts: 2}})
	if _, ok := got["model.layers.0.mlp.experts.0.gate_proj.weight"]; ok {
		t.Fatal("malformed packed experts aliased")
	}
}

func TestRegister_NormalizeWeights_Ugly(t *testing.T) {
	if got := NormalizeWeights(nil, Config{}); len(got) != 0 {
		t.Fatalf("empty map produced %d aliases", len(got))
	}
}

func TestDBRXRegistered_Good(t *testing.T) {
	spec, ok := model.LookupArch("dbrx")
	if !ok || spec.Parse == nil {
		t.Fatalf("DBRX registration = found %v spec %+v", ok, spec)
	}
}

func TestDBRXRegistered_Bad(t *testing.T) {
	spec, _ := model.LookupArch("dbrx")
	if _, err := spec.Parse([]byte("{")); err == nil {
		t.Fatal("malformed config accepted")
	}
}

func TestDBRXRegistered_Ugly(t *testing.T) {
	spec, _ := model.LookupArch("dbrx")
	if _, err := spec.Parse([]byte("{")); err == nil {
		t.Fatal("malformed config accepted")
	}
}
