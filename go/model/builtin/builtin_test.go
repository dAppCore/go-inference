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
	if spec, ok := model.LookupArch("granitemoe"); !ok || spec.Composed == nil {
		t.Fatalf("granitemoe = registered %v composed %v, want composed architecture", ok, spec.Composed != nil)
	}
	if _, ok := model.LookupArch("granitemoehybrid"); ok {
		t.Fatal("out-of-scope granitemoehybrid registered through builtin")
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
	for _, mt := range []string{"mixtral", "deepseek_v2", "deepseek_v3", "olmoe", "qwen2_moe", "qwen3_moe"} {
		spec, ok := model.LookupArch(mt)
		if !ok || spec.Composed == nil {
			t.Fatalf("model_type %q = registered %v composed %v", mt, ok, spec.Composed != nil)
		}
	}
}

// TestBuiltinRegistersOCRArches confirms the blank-import wiring alone (no other package import)
// carries all three OCR vision-language arches' registrations: deepseek_vl_v2 (DeepSeek-OCR),
// dots_ocr / dots_ocr_1_5 (DOTS-OCR), and glm_ocr / glm_ocr_text (GLM-OCR). Each is recognised
// with both a Parse and a Composed refusal hook — a checkpoint resolves to a named architecture,
// not "unknown model architecture" — though neither loader path implements the forward yet (see
// each package's own Registered_Good test for the refusal-message contract).
func TestBuiltinRegistersOCRArches(t *testing.T) {
	for _, mt := range []string{"deepseek_vl_v2", "dots_ocr", "dots_ocr_1_5", "glm_ocr", "glm_ocr_text"} {
		spec, ok := model.LookupArch(mt)
		if !ok {
			t.Fatalf("model_type %q not registered through builtin — an OCR arch must be recognised, not fall to \"unknown model architecture\"", mt)
		}
		if spec.Parse == nil || spec.Composed == nil {
			t.Fatalf("model_type %q registered without both a Parse and a Composed refusal hook", mt)
		}
	}
}

// TestBuiltinRegistersLlama4 pins the llama4 wiring through the blank-import:
// both model_type spellings resolve with the Composed hook the arch registers
// (llama4 runs host-side through the composed loader). The gap this closes was
// real — the arch package existed with registration and tests but no builtin
// import, so a serve binary reported "unknown model architecture" for llama4.
func TestBuiltinRegistersLlama4(t *testing.T) {
	for _, mt := range []string{"llama4", "llama4_text"} {
		spec, ok := model.LookupArch(mt)
		if !ok || spec.Composed == nil {
			t.Fatalf("model_type %q = registered %v composed %v, want a composed llama4 arch through builtin", mt, ok, spec.Composed != nil)
		}
	}
}

func TestBuiltinRegistersOpenAI(t *testing.T) {
	for _, mt := range []string{"openai_privacy_filter", "whisper", "gpt_oss"} {
		spec, ok := model.LookupArch(mt)
		if !ok || spec.Composed != nil {
			t.Fatalf("model_type %q = registered %v composed %v, want a registered Parse-based arch through builtin", mt, ok, spec.Composed != nil)
		}
		if spec.Parse == nil {
			t.Fatalf("model_type %q registered through builtin without a Parse func", mt)
		}
	}
}
