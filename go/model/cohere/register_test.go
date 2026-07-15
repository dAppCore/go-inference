// SPDX-Licence-Identifier: EUPL-1.2

package cohere

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func TestCohereRegistered_Good(t *testing.T) {
	for _, mt := range []string{"cohere", "cohere2"} {
		if spec, ok := model.LookupArch(mt); !ok || spec.Parse == nil {
			t.Fatalf("model_type %q is not registered", mt)
		}
	}
}

func TestCohereWeightNames_Good(t *testing.T) {
	// Extracted weight maps from the public safetensors indexes only:
	// https://huggingface.co/CohereLabs/c4ai-command-r-v01/blob/main/model.safetensors.index.json
	// https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024/blob/main/model.safetensors.index.json
	for _, file := range []string{"CohereLabs-c4ai-command-r-v01-model.safetensors.index.json", "CohereLabs-c4ai-command-r7b-12-2024-model.safetensors.index.json"} {
		r := core.ReadFile(core.PathJoin("testdata", file))
		if !r.OK {
			t.Fatalf("read %s", file)
		}
		var index struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if decoded := core.JSONUnmarshal(r.Value.([]byte), &index); !decoded.OK {
			t.Fatalf("parse %s", file)
		}
		for _, name := range []string{"model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.down_proj.weight", "model.norm.weight"} {
			if _, ok := index.WeightMap[name]; !ok {
				t.Fatalf("%s absent from %s", name, file)
			}
		}
	}
	spec, _ := model.LookupArch("cohere")
	if spec.Weights.AttnNorm != ".input_layernorm.weight" || spec.Weights.MLPNorm != "" || spec.Weights.QNorm != ".self_attn.q_norm.weight" {
		t.Fatalf("Cohere weight names = %+v", spec.Weights)
	}
}

func TestCohereRegistered_Bad(t *testing.T) {
	if _, ok := model.LookupArch("CohereForCausalLM"); ok {
		t.Fatal("architecture class registered as model_type")
	}
}

func TestCohereRegistered_Ugly(t *testing.T) {
	if _, ok := model.LookupArch(""); ok {
		t.Fatal("empty model_type registered")
	}
}
