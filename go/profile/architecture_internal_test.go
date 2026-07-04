// SPDX-Licence-Identifier: EUPL-1.2

// Internal parity tests for the byte-walk compactArchitectureNameInto
// helper introduced in W11-E. The hot-path zero-alloc variant MUST
// produce bit-exact output against the heap-allocating fallback
// (which preserves the pre-W11E core.Lower + core.Replace semantics)
// for every architecture name the package ever resolves.

package profile

import "strings"

import "testing"

func TestCompactArchitectureNameInto_ParityWithFallback(t *testing.T) {
	cases := []string{
		"",
		"gemma2",
		"Gemma3ForCausalLM",
		"Gemma4ForConditionalGeneration",
		"Gemma4TextForCausalLM",
		"Gemma4AssistantForCausalLM",
		"LlamaForCausalLM",
		"Qwen2ForCausalLM",
		"Qwen2.5ForCausalLM",
		"Qwen2_5ForCausalLM",
		"Qwen3ForCausalLM",
		"Qwen3NextForCausalLM",
		"Qwen3_5ForConditionalGeneration",
		"Qwen3.5ForConditionalGeneration",
		"Qwen3_6ForConditionalGeneration",
		"Qwen3.6ForConditionalGeneration",
		"Qwen3_5MoeForConditionalGeneration",
		"Qwen3.5MoeForConditionalGeneration",
		"Qwen3_6MoeForConditionalGeneration",
		"Qwen3.6MoeForConditionalGeneration",
		"Qwen3MoeForCausalLM",
		"MiniMaxM2ForCausalLM",
		"MistralForCausalLM",
		"MixtralForCausalLM",
		"PhiForCausalLM",
		"Phi3ForCausalLM",
		"Phi4ForCausalLM",
		"DeepseekV3ForCausalLM",
		"DeepSeekV3ForCausalLM",
		"DeepseekR1ForCausalLM",
		"GptOssForCausalLM",
		"GPTOSSForCausalLM",
		"KimiForCausalLM",
		"MoonshotForCausalLM",
		"GlmForCausalLM",
		"ChatGLMForConditionalGeneration",
		"HermesForCausalLM",
		"GraniteForCausalLM",
		"BertModel",
		"BertForMaskedLM",
		"BertForSequenceClassification",
		"RobertaForSequenceClassification",
		"XLMRobertaForSequenceClassification",
		"DebertaV2ForSequenceClassification",
		"qwen-3.5",
		"qwen_3_5",
		"qwen3.5",
		"qwen35",
		"qwen36",
		"gpt_oss_model",
		"bert-cross-encoder",
		"foo_bar-baz.qux",
		"already_lowercase_with_dots.and-dashes",
	}
	var buf [maxArchitectureNameBytes]byte
	for _, in := range cases {
		got := compactArchitectureNameInto(buf[:], in)
		want := compactArchitectureNameFallback(in)
		if got != want {
			t.Errorf("compactArchitectureNameInto(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestCompactArchitectureNameInto_FallbackOnOverflow(t *testing.T) {
	// Input longer than the stack buffer must fall back cleanly to
	// the heap-stable helper — no panic, identical output.
	var long strings.Builder
	for range maxArchitectureNameBytes + 1 {
		long.WriteString("x")
	}
	var buf [maxArchitectureNameBytes]byte
	got := compactArchitectureNameInto(buf[:], long.String())
	want := compactArchitectureNameFallback(long.String())
	if got != want {
		t.Fatalf("overflow fallback diverged: got %q want %q", got, want)
	}
}

func TestCompactArchitectureNameInto_FallbackOnNonASCII(t *testing.T) {
	// Non-ASCII byte must trigger fallback, preserving Lower-via-
	// Unicode-table semantics.
	in := "Café-Gemma3"
	var buf [maxArchitectureNameBytes]byte
	got := compactArchitectureNameInto(buf[:], in)
	want := compactArchitectureNameFallback(in)
	if got != want {
		t.Fatalf("non-ASCII fallback diverged: got %q want %q", got, want)
	}
}

// TestArchitectureInternal_NormalizeArchitecture_Good locks the canonical
// architecture-alias contract. profile.NormalizeArchitecture is the single
// source of truth the memory, gguf, model, and minimax packages now share
// (each previously carried its own drifted copy — gguf/minimax had frozen
// "qwen3_5" at the old "qwen3_next" id), so the alias map and the
// lowercase/trim/'-'.'→'_' normalisation are pinned here.
func TestArchitectureInternal_NormalizeArchitecture_Good(t *testing.T) {
	cases := map[string]string{
		"qwen3_5":             "qwen3_6", // the corrected fold — was "qwen3_next" in the stale copies
		"qwen3.6":             "qwen3_6", // dot folds to underscore
		"qwen3_5_text":        "qwen3_6",
		"qwen3_5_moe":         "qwen3_6_moe",
		"qwen2.5":             "qwen2",
		"MiniMax-M2":          "minimax_m2", // dash folds + lowercased
		"  bert ":             "bert",       // surrounding whitespace trimmed
		"bert_cross_encoder":  "bert_rerank",
		"bert_model":          "bert",
		"phi3":                "phi",
		"moonshot":            "kimi", // kimi alias
		"gemma4_unified":      "gemma4_unified",
		"gemma4_unified_text": "gemma4_text",
		"unknown-arch":        "unknown_arch", // unknown passes through normalised
	}
	for in, want := range cases {
		if got := NormalizeArchitecture(in); got != want {
			t.Fatalf("NormalizeArchitecture(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestArchitectureInternal_ArchitectureFromTransformersName_Good locks the HF
// class-name → canonical-id contract. profile.ArchitectureFromTransformersName
// is the single source of truth the gguf, model, and hf packages now share;
// their previous copies had drifted — gguf lost the qwen3_6 arms and hf could
// never return "gemma4_assistant" (a dead caller check in hf). The two
// previously-lost cases are pinned here.
func TestArchitectureInternal_ArchitectureFromTransformersName_Good(t *testing.T) {
	cases := map[string]string{
		"Gemma4ForConditionalGeneration":        "gemma4", // multimodal → base loader, not text-only
		"Gemma4UnifiedForConditionalGeneration": "gemma4_unified",
		"Gemma4MultimodalForCausalLM":           "gemma4",
		"Gemma4VisionForCausalLM":               "gemma4",
		"Gemma4ForCausalLM":                     "gemma4_text",      // text/causal → text loader
		"Gemma4AssistantForCausalLM":            "gemma4_assistant", // was unreachable in hf/gguf
		"Gemma3ForCausalLM":                     "gemma3",
		"Gemma2ForCausalLM":                     "gemma2",
		"Qwen3ForCausalLM":                      "qwen3",
		"Qwen3MoeForCausalLM":                   "qwen3_moe",
		"Qwen3NextForCausalLM":                  "qwen3_next",
		"Qwen3_6ForConditionalGeneration":       "qwen3_6", // was unreachable in gguf/hf
		"Qwen3.6ForConditionalGeneration":       "qwen3_6",
		"Qwen3_6MoeForConditionalGeneration":    "qwen3_6_moe",
		"Qwen2ForCausalLM":                      "qwen2",
		"LlamaForCausalLM":                      "llama",
		"MiniMaxM2ForCausalLM":                  "minimax_m2",
		"MixtralForCausalLM":                    "mixtral",
		"MistralForCausalLM":                    "mistral",
		"Phi3ForCausalLM":                       "phi",
		"DeepseekV3ForCausalLM":                 "deepseek",
		"GptOssForCausalLM":                     "gpt_oss",
		"KimiForCausalLM":                       "kimi",
		"MoonshotForCausalLM":                   "kimi", // moonshot alias
		"HermesForCausalLM":                     "hermes",
		"GraniteForCausalLM":                    "granite",
		"GlmForCausalLM":                        "glm",
		"BertModel":                             "bert",
		"BertForSequenceClassification":         "bert_rerank",
		"RobertaForSequenceClassification":      "bert_rerank",
		"UnknownForCausalLM":                    "",
	}
	for in, want := range cases {
		if got := ArchitectureFromTransformersName(in); got != want {
			t.Fatalf("ArchitectureFromTransformersName(%q) = %q, want %q", in, got, want)
		}
	}
}
