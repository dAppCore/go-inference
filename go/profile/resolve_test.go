// SPDX-Licence-Identifier: EUPL-1.2

package profile_test

import (
	"testing"

	"dappco.re/go/inference/profile"
)

// TestResolve_ResolveArchitecture_Good pins the full config-probe →
// registered-id resolution the loader depends on. It is the single home for the
// resolution ORDER (top-level model_type, then a declared text-tower, then the
// architectures fallback) plus the two family refinements that used to live as
// name-branches in the metal loader: a Gemma-4 multimodal wrapper resolves to
// its declared text tower, and a BERT encoder whose architectures name a
// cross-encoder resolves to the rerank variant. Every case mirrors a behaviour
// the metal probeModelType tests already pin, so this guards exactness as the
// knowledge moves into the registry.
func TestResolve_ResolveArchitecture_Good(t *testing.T) {
	cases := []struct {
		name      string
		modelType string
		textTower string
		archs     []string
		want      string
	}{
		// Top-level model_type, canonicalised through NormalizeArchitecture.
		{"qwen2.5 alias", "qwen2.5", "", []string{"Qwen2.5ForCausalLM"}, "qwen2"},
		{"qwen3.5 → 3.6", "qwen3_5", "", []string{"Qwen3_5ForConditionalGeneration"}, "qwen3_6"},
		{"qwen3.5 moe", "qwen3_5_moe", "", []string{"Qwen3_5MoeForConditionalGeneration"}, "qwen3_6_moe"},
		{"qwen3_5 model_type only", "qwen3_5", "", nil, "qwen3_6"},
		// Text-tower fallback when there is no top-level model_type.
		{"text_config qwen", "", "qwen3_5_text", []string{"Qwen3_5ForConditionalGeneration"}, "qwen3_6"},
		// Architectures fallback (no model_type, no text tower).
		{"arch mistral", "", "", []string{"MistralForCausalLM"}, "mistral"},
		{"arch hermes", "", "", []string{"HermesForCausalLM"}, "hermes"},
		{"arch granite", "", "", []string{"GraniteForCausalLM"}, "granite"},
		{"arch phi3", "", "", []string{"Phi3ForCausalLM"}, "phi"},
		{"arch glm", "", "", []string{"GlmForCausalLM"}, "glm"},
		{"arch qwen3 moe", "", "", []string{"Qwen3MoeForCausalLM"}, "qwen3_moe"},
		{"arch qwen3 next", "", "", []string{"Qwen3NextForCausalLM"}, "qwen3_next"},
		{"arch minimax", "", "", []string{"MiniMaxM2ForCausalLM"}, "minimax_m2"},
		// Gemma-4 multimodal wrapper resolves to its declared text tower.
		{"gemma4 multimodal → text", "gemma4", "gemma4_text", []string{"Gemma4ForConditionalGeneration"}, "gemma4_text"},
		// gemma4_unified is its own canonical 12B multimodal id (no text-tower refinement).
		{"gemma4_unified stays unified", "gemma4_unified", "gemma4_unified_text", []string{"Gemma4UnifiedForConditionalGeneration"}, "gemma4_unified"},
		// The unified text tower normalises to gemma4_text.
		{"gemma4_unified_text → text", "gemma4_unified_text", "", []string{"Gemma4TextForCausalLM"}, "gemma4_text"},
		// BERT encoder vs cross-encoder, distinguished only by architectures.
		{"bert plain", "bert", "", []string{"BertModel"}, "bert"},
		{"bert rerank", "bert", "", []string{"BertForSequenceClassification"}, "bert_rerank"},
		{"bert rerank xlm", "bert", "", []string{"XLMRobertaForSequenceClassification"}, "bert_rerank"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := profile.ResolveArchitecture(tc.modelType, tc.textTower, tc.archs)
			if got != tc.want {
				t.Fatalf("ResolveArchitecture(%q, %q, %v) = %q, want %q", tc.modelType, tc.textTower, tc.archs, got, tc.want)
			}
		})
	}
}

// TestResolve_ResolveArchitecture_Bad pins the non-resolving paths: when none of
// the three config signals names a recognised architecture, ResolveArchitecture
// returns the empty string so the loader reports an unknown model rather than
// dispatching on a guess.
func TestResolve_ResolveArchitecture_Bad(t *testing.T) {
	cases := []struct {
		name      string
		modelType string
		textTower string
		archs     []string
	}{
		{"all empty", "", "", nil},
		{"unknown arch class", "", "", []string{"SomethingForCausalLM"}},
		{"empty arch slice", "", "", []string{}},
		{"only unknown archs", "", "", []string{"NotAModelClass", "AlsoUnknown"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := profile.ResolveArchitecture(tc.modelType, tc.textTower, tc.archs); got != "" {
				t.Fatalf("ResolveArchitecture(%q, %q, %v) = %q, want empty", tc.modelType, tc.textTower, tc.archs, got)
			}
		})
	}
}

// TestResolve_ResolveArchitecture_Ugly pins the refinement-boundary edges: a
// Gemma-4 wrapper whose text_config does not name its declared tower keeps the
// wrapper id (no spurious refinement), and an unknown top-level model_type is
// returned in normalised form even when later signals are present — the
// authoritative-first ORDER is not bypassed by a recognisable fallback.
func TestResolve_ResolveArchitecture_Ugly(t *testing.T) {
	if got := profile.ResolveArchitecture("gemma4", "", []string{"Gemma4ForConditionalGeneration"}); got != "gemma4" {
		t.Fatalf("ResolveArchitecture(gemma4, no tower) = %q, want gemma4 (no refinement)", got)
	}
	if got := profile.ResolveArchitecture("gemma4", "qwen3", []string{"Gemma4ForConditionalGeneration"}); got != "gemma4" {
		t.Fatalf("ResolveArchitecture(gemma4, mismatched tower) = %q, want gemma4 (refinement rejected)", got)
	}
	// An unrecognised top-level model_type wins over a resolvable architectures
	// entry — authoritative-first, returned in normalised form, never the arch.
	if got := profile.ResolveArchitecture("Totally-Unknown.Thing", "", []string{"MistralForCausalLM"}); got != "totally_unknown_thing" {
		t.Fatalf("ResolveArchitecture(unknown model_type) = %q, want normalised pass-through", got)
	}
}
