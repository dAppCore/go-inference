// SPDX-Licence-Identifier: EUPL-1.2

package granite

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestRegister_WeightIndex_Good pins the Llama-shaped dense mapping from:
// https://huggingface.co/ibm-granite/granite-3.3-2b-base/blob/main/model.safetensors.index.json
func TestRegister_WeightIndex_Good(t *testing.T) {
	data := core.ReadFile(core.PathJoin("testdata", "ibm-granite-granite-3.3-2b-base-model.safetensors.index.json"))
	if !data.OK {
		t.Fatal("read Granite weight index fixture")
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if result := core.JSONUnmarshal(data.Value.([]byte), &index); !result.OK {
		t.Fatal("parse Granite weight index fixture")
	}
	for _, name := range []string{"model.embed_tokens.weight", "model.norm.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate_proj.weight"} {
		if _, ok := index.WeightMap[name]; !ok {
			t.Fatalf("public Granite index missing %q", name)
		}
	}
	spec, ok := model.LookupArch("granite")
	if !ok {
		t.Fatal("granite not registered")
	}
	if spec.Weights.MLPNorm != ".post_attention_layernorm.weight" || spec.Weights.PostAttnNorm != "" || spec.Weights.PostFFNorm != "" {
		t.Fatalf("Granite weight mapping = %+v", spec.Weights)
	}
}

func TestRegister_MoEHybrid_Bad(t *testing.T) {
	for _, modelType := range []string{"granitemoe", "granitemoehybrid"} {
		if _, ok := model.LookupArch(modelType); ok {
			t.Fatalf("out-of-scope %s registered", modelType)
		}
	}
}

func TestRegister_Empty_Ugly(t *testing.T) {
	if _, ok := model.LookupArch(""); ok {
		t.Fatal("empty model type unexpectedly registered")
	}
}
