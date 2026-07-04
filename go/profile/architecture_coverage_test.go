// SPDX-Licence-Identifier: EUPL-1.2

package profile_test

import (
	"testing"

	prof "dappco.re/go/inference/profile"
)

// These tests close the residual statement-coverage gaps in architecture.go that
// the per-function _Good/_Bad/_Ugly trios leave open. Each targets one specific
// uncovered branch; the assertion pins the documented behaviour of that branch,
// not just its execution, so a regression in the branch fails the test rather
// than silently dropping coverage.

// TestArchitectureCoverage_LookupArchitectureProfileRef_WhitespaceMiss exercises
// the ArchitectureID-returns-empty arm of LookupArchitectureProfileRef. A
// whitespace-only value is non-empty (so it passes the empty short-circuit and
// the direct-index probe both miss), but ArchitectureID trims it to "" — the
// resolver must then report a clean miss rather than indexing on an empty id.
func TestArchitectureCoverage_LookupArchitectureProfileRef_WhitespaceMiss(t *testing.T) {
	for _, value := range []string{"   ", "\t", "\n  \t"} {
		t.Run(value, func(t *testing.T) {
			ref, ok := prof.LookupArchitectureProfileRef(value)
			if ok || ref != nil {
				t.Fatalf("prof.LookupArchitectureProfileRef(%q) = %+v, %v; want nil, false", value, ref, ok)
			}
		})
	}
}

// TestArchitectureCoverage_ArchitectureID_NormalizesToRerank exercises the early
// "normalized == bert_rerank" return in ArchitectureID. "bert_cross_encoder" is
// not a Transformers class name (so ArchitectureFromTransformersName misses) but
// NormalizeArchitecture folds it to "bert_rerank", which ArchitectureID returns
// directly without entering the compact-substring switch.
func TestArchitectureCoverage_ArchitectureID_NormalizesToRerank(t *testing.T) {
	if got := prof.ArchitectureFromTransformersName("bert_cross_encoder"); got != "" {
		t.Fatalf("precondition: ArchitectureFromTransformersName(bert_cross_encoder) = %q, want empty so ArchitectureID reaches the normalize arm", got)
	}
	if got := prof.ArchitectureID("bert_cross_encoder"); got != "bert_rerank" {
		t.Fatalf("prof.ArchitectureID(bert_cross_encoder) = %q, want bert_rerank", got)
	}
}

// TestArchitectureCoverage_ChatTemplateName_UnknownNormalizedFamily exercises the
// final switch in ChatTemplateName — the fallback for an architecture string that
// is NOT a registered profile but whose NormalizeArchitecture form names a bare
// template family. "gemma", "qwen", and the llama aliases are not profile ids
// (the registry keys gemma2/gemma3/.../llama), so they fall through the
// registry-ref branch into the normalized-name switch.
func TestArchitectureCoverage_ChatTemplateName_UnknownNormalizedFamily(t *testing.T) {
	cases := map[string]string{
		"gemma":  "gemma",
		"qwen":   "qwen",
		"llama3": "llama",
		"llama4": "llama",
	}
	for in, want := range cases {
		t.Run(in, func(t *testing.T) {
			// Precondition: these are not registered profile ids, so the
			// registry-ref branch must miss and the normalized switch runs.
			if _, ok := prof.LookupArchitectureProfileRef(in); ok {
				t.Skipf("%q is a registered profile id; the fallback switch is unreachable for it", in)
			}
			if got := prof.ChatTemplateName(in); got != want {
				t.Fatalf("prof.ChatTemplateName(%q) = %q, want %q", in, got, want)
			}
		})
	}
}

// TestArchitectureCoverage_CanonicalWeightName_SurvivesUnrooted exercises the
// final "return trimmed, true" in CanonicalWeightName: a registered architecture
// whose checkpoint rules do not skip the tensor and do not re-root it under
// "model." returns the (wrapper-stripped) name unchanged. A plain family with no
// weight rules (qwen2) and a gemma4 tensor outside every model-prefix
// (lm_head.weight) both land here.
func TestArchitectureCoverage_CanonicalWeightName_SurvivesUnrooted(t *testing.T) {
	cases := []struct {
		architecture string
		name         string
		want         string
	}{
		// qwen2 declares no wrapper/skip/model prefixes, so every name passes
		// through unchanged with ok=true.
		{"qwen2", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight"},
		{"qwen2", "lm_head.weight", "lm_head.weight"},
		// gemma4 strips its wrapper prefixes; "lm_head" is neither skipped nor a
		// model-prefix, so the stripped name returns unchanged (already model.-rooted).
		{"gemma4", "model.lm_head.weight", "lm_head.weight"},
	}
	for _, tc := range cases {
		t.Run(tc.architecture+"/"+tc.name, func(t *testing.T) {
			got, ok := prof.CanonicalWeightName(tc.architecture, tc.name)
			if !ok || got != tc.want {
				t.Fatalf("prof.CanonicalWeightName(%q, %q) = %q, %v; want %q, true", tc.architecture, tc.name, got, ok, tc.want)
			}
		})
	}
}

// TestArchitectureCoverage_NormalizeArchitecture_SingletonFamilyArms exercises the
// alias arms of NormalizeArchitecture that the existing _Good table omits —
// mixtral, mistral, and the deepseek family (bare id plus the _v3/_r1 aliases).
// Each must fold to its canonical id rather than the normalised pass-through.
func TestArchitectureCoverage_NormalizeArchitecture_SingletonFamilyArms(t *testing.T) {
	cases := map[string]string{
		"mixtral":       "mixtral",
		"Mixtral":       "mixtral",
		"mistral":       "mistral",
		"Mistral":       "mistral",
		"deepseek":      "deepseek",
		"deepseek_v3":   "deepseek",
		"deepseek_r1":   "deepseek",
		"DeepSeek-V3":   "deepseek",
		"deepseek-r1":   "deepseek",
		"gpt_oss":       "gpt_oss",
		"gpt_oss_model": "gpt_oss",
		"gptoss":        "gpt_oss",
	}
	for in, want := range cases {
		t.Run(in, func(t *testing.T) {
			if got := prof.NormalizeArchitecture(in); got != want {
				t.Fatalf("prof.NormalizeArchitecture(%q) = %q, want %q", in, got, want)
			}
		})
	}
}
