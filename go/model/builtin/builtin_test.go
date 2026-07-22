// SPDX-Licence-Identifier: EUPL-1.2

package builtin_test

import (
	"testing"

	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/builtin"
)

// TestBuiltinRegistersQwenHybrids is the blank-import round-trip: importing builtin must carry the
// qwen35 registration into the reactive loader, so a serve binary resolves qwen3_5 / qwen3_5_moe /
// qwen3_next by model_type through the factory ArchSpec — no engine blank-import required for the
// registration to be present. (The retired composed engine's generic "composed"/"hybrid" ids died
// with it, #50 — they must NOT resolve.)
func TestBuiltinRegistersQwenHybrids(t *testing.T) {
	for _, mt := range []string{"qwen3_5", "qwen3_5_moe", "qwen3_next"} {
		spec, ok := model.LookupArch(mt)
		if !ok {
			t.Fatalf("model_type %q not registered — builtin did not carry the qwen35 init()", mt)
		}
		if spec.Parse == nil {
			t.Fatalf("model_type %q registered without a Parse hook", mt)
		}
	}
	for _, mt := range []string{"composed", "hybrid"} {
		if _, ok := model.LookupArch(mt); ok {
			t.Fatalf("generic model_type %q still registered — it should have died with the composed engine (#50)", mt)
		}
	}
	if _, ok := model.LookupArch("gemma4"); !ok {
		t.Fatal("gemma4 not registered — want the standard transformer arch through the same import")
	}
}

func TestBuiltinRegistersLlama(t *testing.T) {
	if _, ok := model.LookupArch("llama"); !ok {
		t.Fatal("model_type llama not registered through builtin")
	}
}

func TestBuiltinRegistersGranite(t *testing.T) {
	if _, ok := model.LookupArch("granite"); !ok {
		t.Fatal("model_type granite not registered through builtin")
	}
	if spec, ok := model.LookupArch("granitemoe"); !ok || spec.Parse == nil {
		t.Fatalf("granitemoe = registered %v parse %v, want the factory MoE arch", ok, spec.Parse != nil)
	}
	if _, ok := model.LookupArch("granitemoehybrid"); ok {
		t.Fatal("out-of-scope granitemoehybrid registered through builtin")
	}
}

func TestBuiltinRegistersQwen2(t *testing.T) {
	if _, ok := model.LookupArch("qwen2"); !ok {
		t.Fatal("model_type qwen2 not registered through builtin")
	}
}

func TestBuiltinRegistersOLMo(t *testing.T) {
	for _, mt := range []string{"olmo", "olmo2"} {
		if _, ok := model.LookupArch(mt); !ok {
			t.Fatalf("model_type %q not registered through builtin", mt)
		}
	}
}

func TestBuiltinRegistersPhi(t *testing.T) {
	for _, mt := range []string{"phi", "phi3"} {
		if _, ok := model.LookupArch(mt); !ok {
			t.Fatalf("model_type %q not registered through builtin", mt)
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
		if _, ok := model.LookupArch(mt); !ok {
			t.Fatalf("model_type %q not registered through builtin", mt)
		}
	}
}

func TestBuiltinRegistersMoEFamilies(t *testing.T) {
	for _, mt := range []string{"mixtral", "deepseek_v2", "deepseek_v3", "olmoe", "qwen2_moe", "qwen3_moe"} {
		spec, ok := model.LookupArch(mt)
		if !ok || spec.Parse == nil {
			t.Fatalf("model_type %q = registered %v parse %v", mt, ok, spec.Parse != nil)
		}
	}
}

// TestBuiltinRegistersOCRArches confirms the blank-import wiring alone (no other package import)
// carries all three OCR vision-language arches' registrations: deepseek_vl_v2 (DeepSeek-OCR),
// dots_ocr / dots_ocr_1_5 (DOTS-OCR), and glm_ocr / glm_ocr_text (GLM-OCR). Each is recognised
// with a Parse hook — a checkpoint resolves to a named architecture, not "unknown model
// architecture" — though the forward is not implemented yet (see each package's own
// Registered_Good test for the refusal-message contract).
func TestBuiltinRegistersOCRArches(t *testing.T) {
	for _, mt := range []string{"deepseek_vl_v2", "dots_ocr", "dots_ocr_1_5", "glm_ocr", "glm_ocr_text"} {
		spec, ok := model.LookupArch(mt)
		if !ok {
			t.Fatalf("model_type %q not registered through builtin — an OCR arch must be recognised, not fall to \"unknown model architecture\"", mt)
		}
		if spec.Parse == nil {
			t.Fatalf("model_type %q registered without a Parse hook", mt)
		}
	}
}

// TestBuiltinRegistersLlama4 pins the llama4 wiring through the blank-import: both model_type
// spellings resolve with the factory spec the arch registers. The gap this closes was real — the
// arch package existed with registration and tests but no builtin import, so a serve binary
// reported "unknown model architecture" for llama4.
func TestBuiltinRegistersLlama4(t *testing.T) {
	for _, mt := range []string{"llama4", "llama4_text"} {
		spec, ok := model.LookupArch(mt)
		if !ok || spec.Parse == nil {
			t.Fatalf("model_type %q = registered %v parse %v, want the factory llama4 arch through builtin", mt, ok, spec.Parse != nil)
		}
	}
}

func TestBuiltinRegistersOpenAI(t *testing.T) {
	for _, mt := range []string{"openai_privacy_filter", "whisper", "gpt_oss"} {
		spec, ok := model.LookupArch(mt)
		if !ok {
			t.Fatalf("model_type %q not registered through builtin", mt)
		}
		if spec.Parse == nil {
			t.Fatalf("model_type %q registered through builtin without a Parse func", mt)
		}
	}
}
