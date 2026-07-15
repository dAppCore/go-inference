// SPDX-Licence-Identifier: EUPL-1.2

package olmo

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func indexKeys(t *testing.T, name string) map[string]string {
	t.Helper()
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if r := core.JSONUnmarshal(fixture(t, name), &index); !r.OK {
		t.Fatalf("parse %s: %v", name, r.Error())
	}
	return index.WeightMap
}

// TestRegister_OLMoWeightIndex_Good pins roles against allenai/OLMo-7B-0724-hf.
// Source: https://huggingface.co/allenai/OLMo-7B-0724-hf/blob/main/model.safetensors.index.json
func TestRegister_OLMoWeightIndex_Good(t *testing.T) {
	keys := indexKeys(t, "allenai-olmo-7b-0724-hf-model.safetensors.index.json")
	for _, name := range []string{"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate_proj.weight"} {
		if _, ok := keys[name]; !ok {
			t.Fatalf("OLMo index missing %q", name)
		}
	}
	spec, ok := model.LookupArch("olmo")
	if !ok {
		t.Fatal("olmo not registered")
	}
	if spec.Weights.AttnNorm != "" || spec.Weights.MLPNorm != "" || spec.Weights.FinalNorm != "" || spec.Weights.QNorm != "" || spec.Weights.KNorm != "" {
		t.Fatalf("OLMo non-parametric norms mapped as weights: %+v", spec.Weights)
	}
}

// TestRegister_OLMo2WeightIndex_Good pins roles against allenai/OLMo-2-1124-7B.
// Source: https://huggingface.co/allenai/OLMo-2-1124-7B/blob/main/model.safetensors.index.json
func TestRegister_OLMo2WeightIndex_Good(t *testing.T) {
	keys := indexKeys(t, "allenai-olmo-2-1124-7b-model.safetensors.index.json")
	for _, name := range []string{
		"model.norm.weight", "model.layers.0.post_attention_layernorm.weight",
		"model.layers.0.post_feedforward_layernorm.weight",
		"model.layers.0.self_attn.q_norm.weight", "model.layers.0.self_attn.k_norm.weight",
	} {
		if _, ok := keys[name]; !ok {
			t.Fatalf("OLMo2 index missing %q", name)
		}
	}
	spec, ok := model.LookupArch("olmo2")
	if !ok {
		t.Fatal("olmo2 not registered")
	}
	if spec.Weights.AttnNorm != "" || spec.Weights.MLPNorm != "" || spec.Weights.PostAttnNorm != ".post_attention_layernorm.weight" || spec.Weights.PostFFNorm != ".post_feedforward_layernorm.weight" {
		t.Fatalf("OLMo2 post-norm mapping = %+v", spec.Weights)
	}
}

func TestRegister_Bad(t *testing.T) {
	if _, ok := model.LookupArch("hf_olmo"); ok {
		t.Fatal("legacy remote-code hf_olmo unexpectedly registered")
	}
}

func TestRegister_Ugly(t *testing.T) {
	if _, ok := model.LookupArch(""); ok {
		t.Fatal("empty model type unexpectedly registered")
	}
}
