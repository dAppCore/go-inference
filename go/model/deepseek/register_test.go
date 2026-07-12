// SPDX-Licence-Identifier: EUPL-1.2

package deepseek

import (
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

func TestWeightNames_Good(t *testing.T) {
	w := WeightNames()
	if w.Q != ".self_attn.q_proj.weight" || w.QA != ".self_attn.q_a_proj.weight" || w.QANorm != ".self_attn.q_a_layernorm.weight" || w.QB != ".self_attn.q_b_proj.weight" ||
		w.KVA != ".self_attn.kv_a_proj_with_mqa.weight" || w.KVB != ".self_attn.kv_b_proj.weight" ||
		w.KVANorm != ".self_attn.kv_a_layernorm.weight" || w.O != ".self_attn.o_proj.weight" ||
		w.Router != ".mlp.gate.weight" || w.ExpertGate != ".mlp.experts.%d.gate_proj.weight" {
		t.Fatalf("DeepSeek names = %+v", w)
	}
}

func TestDeepSeekRegistered_Good(t *testing.T) {
	for _, modelType := range []string{"deepseek_v2", "deepseek_v3"} {
		spec, ok := model.LookupArch(modelType)
		if !ok || spec.Composed == nil || spec.Parse == nil {
			t.Fatalf("%s registration = found %v spec %+v", modelType, ok, spec)
		}
		if _, err := spec.Composed(map[string]safetensors.Tensor{}, []byte(`{"model_type":"deepseek_v2"}`)); err == nil {
			t.Fatalf("%s composed hook accepted MLA without an MLA mixer", modelType)
		}
	}
}

func TestParse_Bad(t *testing.T) {
	spec, _ := model.LookupArch("deepseek_v2")
	if _, err := spec.Parse([]byte("{")); err == nil {
		t.Fatal("malformed config accepted")
	}
}

func TestParse_Good(t *testing.T) {
	spec, _ := model.LookupArch("deepseek_v2")
	if _, err := spec.Parse([]byte(`{"hidden_size":8}`)); err != nil {
		t.Fatalf("valid JSON rejected: %v", err)
	}
}
