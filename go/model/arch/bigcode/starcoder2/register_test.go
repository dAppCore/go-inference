// SPDX-Licence-Identifier: EUPL-1.2

package starcoder2

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestRegister_WeightIndex_Good pins StarCoder2's tensor roles against its public sharded index.
// Source: https://huggingface.co/bigcode/starcoder2-7b/blob/main/model.safetensors.index.json
// The 3B repository currently publishes one unsharded model.safetensors and therefore has no index.
func TestRegister_WeightIndex_Good(t *testing.T) {
	data := core.ReadFile(core.PathJoin("testdata", "bigcode-starcoder2-7b-model.safetensors.index.json"))
	if !data.OK {
		t.Fatal("read StarCoder2 weight-index fixture")
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if r := core.JSONUnmarshal(data.Value.([]byte), &index); !r.OK {
		t.Fatal("parse StarCoder2 weight-index fixture")
	}
	for _, name := range []string{
		"model.embed_tokens.weight", "model.norm.weight",
		"model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight",
		"model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.o_proj.weight",
		"model.layers.0.mlp.c_fc.weight", "model.layers.0.mlp.c_proj.weight",
	} {
		if _, ok := index.WeightMap[name]; !ok {
			t.Fatalf("public index missing StarCoder2 weight %q", name)
		}
	}
	spec, ok := model.LookupArch("starcoder2")
	if !ok {
		t.Fatal("starcoder2 not registered")
	}
	if spec.Weights.LayerPrefix != "model.layers.%d" || spec.Weights.Gate != ".mlp.c_fc" || spec.Weights.Up != ".mlp.c_fc" || spec.Weights.Down != ".mlp.c_proj" {
		t.Fatalf("StarCoder2 weight names = %+v", spec.Weights)
	}
}

// TestRegister_ArchitectureClass_Bad rejects a class name as a model_type alias.
func TestRegister_ArchitectureClass_Bad(t *testing.T) {
	if _, ok := model.LookupArch("Starcoder2ForCausalLM"); ok {
		t.Fatal("architecture class registered as model_type")
	}
}

// TestRegister_CodeGen_Ugly keeps CodeGen out until an indexed safetensors layout proves its mapping.
func TestRegister_CodeGen_Ugly(t *testing.T) {
	if _, ok := model.LookupArch("codegen"); ok {
		t.Fatal("CodeGen registered without an indexed safetensors weight-map receipt")
	}
}
