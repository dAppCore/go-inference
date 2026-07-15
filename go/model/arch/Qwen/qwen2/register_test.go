// SPDX-Licence-Identifier: EUPL-1.2

package qwen2

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestRegister_WeightIndex_Good pins the mapping evidenced by Qwen/Qwen2.5-7B's index:
// https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/model.safetensors.index.json
// Qwen2 is llama/mistral-shaped with QKV projection biases, not qwen3-shaped QK norm.
func TestRegister_WeightIndex_Good(t *testing.T) {
	data := core.ReadFile(core.PathJoin("testdata", "qwen-qwen2.5-7b-model.safetensors.index.json"))
	if !data.OK {
		t.Fatal("read Qwen2.5 weight index fixture")
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if result := core.JSONUnmarshal(data.Value.([]byte), &index); !result.OK {
		t.Fatal("parse Qwen2.5 weight index fixture")
	}
	for _, name := range []string{
		"model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.bias",
		"model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.k_proj.bias",
		"model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.v_proj.bias",
		"model.layers.0.post_attention_layernorm.weight",
	} {
		if _, ok := index.WeightMap[name]; !ok {
			t.Fatalf("public index missing expected Qwen2 weight %q", name)
		}
	}
	for _, absent := range []string{"model.layers.0.self_attn.q_norm.weight", "model.layers.0.self_attn.k_norm.weight"} {
		if _, ok := index.WeightMap[absent]; ok {
			t.Fatalf("public Qwen2 index unexpectedly contains Qwen3 weight %q", absent)
		}
	}
	spec, ok := model.LookupArch("qwen2")
	if !ok {
		t.Fatal("qwen2 not registered")
	}
	if spec.Weights.MLPNorm != ".post_attention_layernorm.weight" || spec.Weights.PostAttnNorm != "" {
		t.Fatalf("Qwen2 norm mapping = %+v", spec.Weights)
	}
	if spec.Weights.QNorm != "" || spec.Weights.KNorm != "" {
		t.Fatalf("Qwen2 incorrectly inherited qwen3 QK norms: %q %q", spec.Weights.QNorm, spec.Weights.KNorm)
	}
}

func TestRegister_MoE_Bad(t *testing.T) {
	if _, ok := model.LookupArch("qwen2_moe"); ok {
		t.Fatal("Qwen2-MoE must remain out of the dense Qwen2 registration")
	}
}

func TestRegister_Empty_Ugly(t *testing.T) {
	if _, ok := model.LookupArch(""); ok {
		t.Fatal("empty model type unexpectedly registered")
	}
}
