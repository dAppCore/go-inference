// SPDX-Licence-Identifier: EUPL-1.2

package builtin_test

import (
	"testing"

	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/builtin"
)

// TestBuiltinRegistersComposed is the blank-import round-trip: importing builtin must carry the composed
// hybrids' registration into the reactive loader, so a serve binary resolves qwen3_5 / qwen3_5_moe /
// qwen3_next by model_type through the ArchSpec.Composed hook — no engine/metal blank-import required for
// the registration to be present.
func TestBuiltinRegistersComposed(t *testing.T) {
	for _, mt := range []string{"qwen3_5", "qwen3_5_moe", "qwen3_next"} {
		spec, ok := model.LookupArch(mt)
		if !ok {
			t.Fatalf("model_type %q not registered — builtin did not carry the composed init()", mt)
		}
		if spec.Composed == nil {
			t.Fatalf("model_type %q registered without a Composed hook", mt)
		}
	}
	// A standard transformer arch resolves through the same import, without a Composed hook — the
	// contrast that confirms Composed is the hybrid-only routing signal.
	if spec, ok := model.LookupArch("gemma4"); !ok || spec.Composed != nil {
		t.Fatalf("gemma4 = (spec.Composed==nil? %v, registered? %v), want a registered transformer arch with no Composed hook", spec.Composed == nil, ok)
	}
}

func TestBuiltinRegistersLlama(t *testing.T) {
	spec, ok := model.LookupArch("llama")
	if !ok {
		t.Fatal("model_type llama not registered through builtin")
	}
	if spec.Composed != nil {
		t.Fatal("llama unexpectedly registered as a composed architecture")
	}
}

func TestBuiltinRegistersGranite(t *testing.T) {
	if _, ok := model.LookupArch("granite"); !ok {
		t.Fatal("model_type granite not registered through builtin")
	}
	for _, modelType := range []string{"granitemoe", "granitemoehybrid"} {
		if _, ok := model.LookupArch(modelType); ok {
			t.Fatalf("out-of-scope %s registered through builtin", modelType)
		}
	}
}

func TestBuiltinRegistersQwen2(t *testing.T) {
	spec, ok := model.LookupArch("qwen2")
	if !ok {
		t.Fatal("model_type qwen2 not registered through builtin")
	}
	if spec.Composed != nil {
		t.Fatal("qwen2 unexpectedly registered as a composed architecture")
	}
}

func TestBuiltinRegistersOLMo(t *testing.T) {
	for _, mt := range []string{"olmo", "olmo2"} {
		if spec, ok := model.LookupArch(mt); !ok || spec.Composed != nil {
			t.Fatalf("model_type %q = registered %v composed %v", mt, ok, spec.Composed != nil)
		}
	}
}

func TestBuiltinRegistersPhi(t *testing.T) {
	for _, mt := range []string{"phi", "phi3"} {
		if spec, ok := model.LookupArch(mt); !ok || spec.Composed != nil {
			t.Fatalf("model_type %q = registered %v composed %v", mt, ok, spec.Composed != nil)
		}
	}
}

func TestBuiltinRegistersGPT2Families(t *testing.T) {
	for _, mt := range []string{"gpt2", "gpt_bigcode", "starcoder"} {
		if _, ok := model.LookupArch(mt); !ok {
			t.Fatalf("model_type %q not registered through builtin", mt)
		}
	}
}

func TestBuiltinRegistersOPT(t *testing.T) {
	if _, ok := model.LookupArch("opt"); !ok {
		t.Fatal("model_type opt not registered through builtin")
	}
}

func TestBuiltinRegistersStarCoder2(t *testing.T) {
	if _, ok := model.LookupArch("starcoder2"); !ok {
		t.Fatal("model_type starcoder2 not registered through builtin")
	}
}

func TestBuiltinRegistersOpportunisticDenseFamilies(t *testing.T) {
	for _, mt := range []string{"mpt", "stablelm", "smollm3"} {
		if spec, ok := model.LookupArch(mt); !ok || spec.Composed != nil {
			t.Fatalf("model_type %q = registered %v composed %v", mt, ok, spec.Composed != nil)
		}
	}
}

func TestBuiltinRegistersMoEFamilies(t *testing.T) {
	for _, mt := range []string{"mixtral", "deepseek_v2", "deepseek_v3"} {
		spec, ok := model.LookupArch(mt)
		if !ok || spec.Composed == nil {
			t.Fatalf("model_type %q = registered %v composed %v", mt, ok, spec.Composed != nil)
		}
	}
}
