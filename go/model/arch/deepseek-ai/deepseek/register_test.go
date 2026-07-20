// SPDX-Licence-Identifier: EUPL-1.2

package deepseek

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func TestRegister_WeightNames_Good(t *testing.T) {
	w := WeightNames()
	if w.Q != ".self_attn.q_proj.weight" || w.QA != ".self_attn.q_a_proj.weight" || w.QANorm != ".self_attn.q_a_layernorm.weight" || w.QB != ".self_attn.q_b_proj.weight" ||
		w.KVA != ".self_attn.kv_a_proj_with_mqa.weight" || w.KVB != ".self_attn.kv_b_proj.weight" ||
		w.KVANorm != ".self_attn.kv_a_layernorm.weight" || w.O != ".self_attn.o_proj.weight" ||
		w.Router != ".mlp.gate.weight" || w.ExpertGate != ".mlp.experts.%d.gate_proj.weight" {
		t.Fatalf("DeepSeek names = %+v", w)
	}
}

// TestRegister_WeightNames_Bad guards against a copy-paste alias collision:
// the router role must never share a tensor-name template with an expert role.
func TestRegister_WeightNames_Bad(t *testing.T) {
	w := WeightNames()
	if w.Router == w.ExpertGate || w.Router == w.SharedGate {
		t.Fatalf("router alias collides with an expert role: %+v", w)
	}
}

// TestRegister_WeightNames_Ugly proves the routed/shared expert split: routed
// experts are %d-templated (one tensor per expert index) while shared
// experts are fixed names (always-active, no index) — a dropped template on
// one or a spurious one on the other would silently break the split.
func TestRegister_WeightNames_Ugly(t *testing.T) {
	w := WeightNames()
	if core.Contains(w.SharedGate, "%d") {
		t.Fatalf("shared-expert name unexpectedly templated: %q", w.SharedGate)
	}
	if !core.Contains(w.ExpertGate, "%d") {
		t.Fatalf("routed-expert name missing its %%d template: %q", w.ExpertGate)
	}
}

func TestDeepSeekRegistered_Good(t *testing.T) {
	for _, modelType := range []string{"deepseek_v2", "deepseek_v3"} {
		spec, ok := model.LookupArch(modelType)
		if !ok || spec.Parse == nil {
			t.Fatalf("%s registration = found %v spec %+v", modelType, ok, spec)
		}
		ac, err := spec.Parse([]byte(`{"model_type":"` + modelType + `","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"vocab_size":32,"kv_lora_rank":4,"qk_rope_head_dim":2,"qk_nope_head_dim":2,"v_head_dim":2}`))
		if err != nil {
			t.Fatalf("%s Parse: %v", modelType, err)
		}
		if _, aerr := ac.Arch(); aerr == nil {
			t.Fatalf("%s Arch() accepted MLA without an MLA attention implementation", modelType)
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
