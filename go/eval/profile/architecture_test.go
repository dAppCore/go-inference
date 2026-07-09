// SPDX-Licence-Identifier: EUPL-1.2

package profile_test

import (
	"testing"

	prof "dappco.re/go/inference/eval/profile"
)

func requireExactLoRATargets(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("LoRATargets = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("LoRATargets = %v, want %v", got, want)
		}
	}
}

// --- BuiltinArchitectureProfiles ------------------------------------------

// TestArchitecture_BuiltinArchitectureProfiles_Good pins the metadata-only
// target list: every profile carries a unique non-empty ID and the broad
// feature-parity families are all present.
func TestArchitecture_BuiltinArchitectureProfiles_Good(t *testing.T) {
	profiles := prof.BuiltinArchitectureProfiles()
	if len(profiles) < 12 {
		t.Fatalf("prof.BuiltinArchitectureProfiles len = %d, want broad feature-parity target list", len(profiles))
	}
	seen := map[string]bool{}
	for _, profile := range profiles {
		if profile.ID == "" {
			t.Fatalf("profile missing ID: %+v", profile)
		}
		if seen[profile.ID] {
			t.Fatalf("duplicate profile ID %q", profile.ID)
		}
		seen[profile.ID] = true
	}
	for _, id := range []string{"gemma4_text", "gemma4_unified", "gemma4_assistant", "qwen2", "qwen3_next", "qwen3_6", "qwen3_6_moe", "qwen3_moe", "minimax_m2", "mixtral", "deepseek", "gpt_oss", "bert", "bert_rerank"} {
		if !seen[id] {
			t.Fatalf("missing builtin architecture profile %q", id)
		}
	}
}

// TestArchitecture_BuiltinArchitectureProfiles_Bad pins the defensive deep-clone
// contract: mutating a returned profile's slice field must not corrupt the
// shared registry, so a fresh call sees the original aliases.
func TestArchitecture_BuiltinArchitectureProfiles_Bad(t *testing.T) {
	profiles := prof.BuiltinArchitectureProfiles()
	var idx = -1
	for i := range profiles {
		if len(profiles[i].Aliases) > 0 {
			idx = i
			break
		}
	}
	if idx < 0 {
		t.Fatal("BuiltinArchitectureProfiles: expected at least one profile with aliases")
	}
	id := profiles[idx].ID
	original := profiles[idx].Aliases[0]
	profiles[idx].Aliases[0] = "mutated-alias"
	for _, fresh := range prof.BuiltinArchitectureProfiles() {
		if fresh.ID == id {
			if fresh.Aliases[0] == "mutated-alias" {
				t.Fatalf("BuiltinArchitectureProfiles returned aliased Aliases backing array for %q", id)
			}
			if fresh.Aliases[0] != original {
				t.Fatalf("BuiltinArchitectureProfiles[%q].Aliases[0] = %q, want stable %q", id, fresh.Aliases[0], original)
			}
		}
	}
}

// TestArchitecture_BuiltinArchitectureProfiles_Ugly pins per-call independence
// at the element-scalar level: mutating the Family of a returned profile must
// not leak into a fresh call, and the fresh slice keeps a stable length (no
// aliasing of the backing array between calls).
func TestArchitecture_BuiltinArchitectureProfiles_Ugly(t *testing.T) {
	first := prof.BuiltinArchitectureProfiles()
	if len(first) < 2 {
		t.Fatalf("BuiltinArchitectureProfiles len = %d, want at least 2 to test element independence", len(first))
	}
	want := len(first)
	id := first[1].ID
	original := first[1].Family
	first[1].Family = "mutated-family"
	second := prof.BuiltinArchitectureProfiles()
	if len(second) != want {
		t.Fatalf("BuiltinArchitectureProfiles len = %d, want stable %d", len(second), want)
	}
	if second[1].ID != id || second[1].Family != original {
		t.Fatalf("BuiltinArchitectureProfiles[1] = {%q, Family=%q}, want stable {%q, %q}", second[1].ID, second[1].Family, id, original)
	}
}

// TestArchitecture_BuiltinArchitectureProfiles_BatchArenaMatchesSingleClone is a
// regression lock on BuiltinArchitectureProfiles' batch clone: every returned
// profile packs its ~11 clone-managed []string fields into ONE shared arena
// (profileStringFieldLen sizes it, cloneArchitectureProfileInto/sliceFromArena
// carve each profile's slice out of it in field order). Those two functions'
// field lists must stay in exact sync — a field added to one but not the other
// silently under- or over-sizes the shared arena, and because sliceFromArena
// truncates on a short arena instead of panicking, the corruption is SILENT and
// lands on whichever profiles happen to be built after the deficit compounds
// (verified by injecting exactly this bug during audit: profileStringFieldLen
// omitting one field kept the entire suite green). LookupArchitectureProfile
// is a genuinely independent code path for this purpose — it clones a single
// profile into its own exactly-sized arena, never sharing capacity with any
// other profile — so cross-checking every batch-cloned profile against its
// single-clone sibling catches an arena-sizing drift that no per-call-isolation
// or scalar-field test (the Bad/Ugly siblings above) would ever observe.
func TestArchitecture_BuiltinArchitectureProfiles_BatchArenaMatchesSingleClone(t *testing.T) {
	batch := prof.BuiltinArchitectureProfiles()
	if len(batch) < 12 {
		t.Fatalf("BuiltinArchitectureProfiles len = %d, want the full registry", len(batch))
	}
	for _, got := range batch {
		want, ok := prof.LookupArchitectureProfile(got.ID)
		if !ok {
			t.Fatalf("LookupArchitectureProfile(%q) ok = false, want the same profile the batch returned", got.ID)
		}
		requireExactLoRATargets(t, got.LoRATargets, want.LoRATargets)
		requireExactLoRATargets(t, got.LoRADefaultTargets, want.LoRADefaultTargets)
		requireExactLoRATargets(t, got.LoRAExtendedTargets, want.LoRAExtendedTargets)
		requireExactLoRATargets(t, got.WeightWrapperPrefixes, want.WeightWrapperPrefixes)
		requireExactLoRATargets(t, got.WeightSkipPrefixes, want.WeightSkipPrefixes)
		requireExactLoRATargets(t, got.WeightSkipSubstrings, want.WeightSkipSubstrings)
		requireExactLoRATargets(t, got.WeightModelPrefixes, want.WeightModelPrefixes)
		requireExactLoRATargets(t, got.QuantizationHints, want.QuantizationHints)
		requireExactLoRATargets(t, got.CacheHints, want.CacheHints)
		requireExactLoRATargets(t, got.Notes, want.Notes)
		requireExactLoRATargets(t, got.Aliases, want.Aliases)
	}
}

// --- LookupArchitectureProfile --------------------------------------------

// TestArchitecture_LookupArchitectureProfile_Good pins the config-name →
// built-in-profile resolution across every family: a Transformers class name or
// a config model_type resolves to the right id, parser, and feature flags.
func TestArchitecture_LookupArchitectureProfile_Good(t *testing.T) {
	cases := []struct {
		name       string
		input      string
		wantID     string
		wantParser string
		wantMoE    bool
		wantEmbed  bool
		wantNative bool
	}{
		{name: "minimax", input: "MiniMaxM2ForCausalLM", wantID: "minimax_m2", wantParser: "minimax", wantMoE: true, wantNative: true},
		{name: "mixtral", input: "MixtralForCausalLM", wantID: "mixtral", wantParser: "mistral", wantMoE: true, wantNative: true},
		{name: "mistral", input: "mistral", wantID: "mistral", wantParser: "mistral", wantNative: true},
		{name: "hermes", input: "HermesForCausalLM", wantID: "hermes", wantParser: "hermes", wantNative: true},
		{name: "granite", input: "GraniteForCausalLM", wantID: "granite", wantParser: "granite", wantNative: true},
		{name: "phi", input: "Phi3ForCausalLM", wantID: "phi", wantParser: "generic", wantNative: true},
		{name: "glm", input: "GlmForCausalLM", wantID: "glm", wantParser: "glm", wantNative: true},
		{name: "kimi", input: "KimiForCausalLM", wantID: "kimi", wantParser: "kimi", wantMoE: true, wantNative: true},
		{name: "deepseek", input: "DeepseekV3ForCausalLM", wantID: "deepseek", wantParser: "deepseek-r1", wantMoE: true, wantNative: true},
		{name: "gptoss", input: "GptOssForCausalLM", wantID: "gpt_oss", wantParser: "gpt-oss", wantMoE: true, wantNative: true},
		{name: "bert", input: "BertModel", wantID: "bert", wantParser: "generic", wantEmbed: true, wantNative: true},
		{name: "bert-rerank", input: "BertForSequenceClassification", wantID: "bert_rerank", wantParser: "generic", wantNative: true},
		{name: "qwen-native", input: "qwen3", wantID: "qwen3", wantParser: "qwen", wantNative: true},
		{name: "qwen3-moe", input: "Qwen3MoeForCausalLM", wantID: "qwen3_moe", wantParser: "qwen", wantMoE: true, wantNative: true},
		{name: "qwen2-5-native", input: "Qwen2.5ForCausalLM", wantID: "qwen2", wantParser: "qwen", wantNative: true},
		{name: "gemma4-unified", input: "Gemma4UnifiedForConditionalGeneration", wantID: "gemma4_unified", wantParser: "gemma", wantNative: true},
		{name: "gemma4-assistant", input: "gemma4_assistant", wantID: "gemma4_assistant", wantParser: "gemma", wantNative: true},
		{name: "qwen36-dense", input: "Qwen3_5ForConditionalGeneration", wantID: "qwen3_6", wantParser: "qwen", wantNative: true},
		{name: "qwen36-moe", input: "Qwen3_5MoeForConditionalGeneration", wantID: "qwen3_6_moe", wantParser: "qwen", wantMoE: true, wantNative: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, ok := prof.LookupArchitectureProfile(tc.input)
			if !ok {
				t.Fatalf("prof.LookupArchitectureProfile(%q) ok = false", tc.input)
			}
			if p.ID != tc.wantID || p.ParserID != tc.wantParser {
				t.Fatalf("profile = %+v, want id %q parser %q", p, tc.wantID, tc.wantParser)
			}
			if p.MoE != tc.wantMoE || p.Embeddings != tc.wantEmbed || p.NativeRuntime != tc.wantNative {
				t.Fatalf("profile flags = moe:%v embeddings:%v native:%v, want %v/%v/%v", p.MoE, p.Embeddings, p.NativeRuntime, tc.wantMoE, tc.wantEmbed, tc.wantNative)
			}
			if tc.name == "bert-rerank" && !p.Rerank {
				t.Fatalf("profile = %+v, want rerank profile", p)
			}
			if tc.name == "gemma4-assistant" && (p.Generation || p.Chat || p.RequiresChatTemplate) {
				t.Fatalf("profile = %+v, want attached native drafter without standalone chat/generation", p)
			}
			if tc.name == "minimax" && (p.Generation || p.Chat || !p.MoE) {
				t.Fatalf("profile = %+v, want staged native MiniMax M2 loader without standalone generation", p)
			}
			if tc.name == "qwen36-dense" && (p.Generation || p.Chat || p.MoE) {
				t.Fatalf("profile = %+v, want staged native Qwen3.6 loader without standalone generation/chat or MoE", p)
			}
			if tc.name == "qwen3-moe" && (p.Generation || p.Chat || !p.MoE) {
				t.Fatalf("profile = %+v, want staged native Qwen3 MoE loader without standalone generation/chat", p)
			}
			if tc.name == "mixtral" && (p.Generation || p.Chat || !p.MoE) {
				t.Fatalf("profile = %+v, want staged native mixtral loader without standalone generation/chat", p)
			}
			if tc.name == "deepseek" && (p.Generation || p.Chat || !p.MoE) {
				t.Fatalf("profile = %+v, want staged native deepseek loader without standalone generation/chat", p)
			}
			if tc.name == "gptoss" && (p.Generation || p.Chat || !p.MoE) {
				t.Fatalf("profile = %+v, want staged native gpt_oss loader without standalone generation/chat", p)
			}
			if tc.name == "kimi" && (p.Generation || p.Chat || !p.MoE) {
				t.Fatalf("profile = %+v, want staged native kimi loader without standalone generation/chat", p)
			}
			if tc.name == "qwen36-moe" && (p.Generation || p.Chat || !p.MoE) {
				t.Fatalf("profile = %+v, want staged native Qwen3.6 MoE loader without standalone generation/chat", p)
			}
		})
	}
}

// TestArchitecture_LookupArchitectureProfile_Bad pins the miss path: a value
// that names no registered family yields ok=false and the zero profile, so the
// loader branches on ok rather than dispatching on a guessed id.
func TestArchitecture_LookupArchitectureProfile_Bad(t *testing.T) {
	for _, input := range []string{"nonexistent_family", "NotAModelForCausalLM"} {
		p, ok := prof.LookupArchitectureProfile(input)
		if ok {
			t.Fatalf("prof.LookupArchitectureProfile(%q) ok = true, want false", input)
		}
		if p.ID != "" {
			t.Fatalf("prof.LookupArchitectureProfile(%q) = %+v, want zero profile", input, p)
		}
	}
}

// TestArchitecture_LookupArchitectureProfile_Ugly pins the defensive-clone
// promise the doc-comment makes: the returned profile's slice fields are
// independent of the registry, so a caller may mutate them without corrupting a
// later lookup.
func TestArchitecture_LookupArchitectureProfile_Ugly(t *testing.T) {
	first, ok := prof.LookupArchitectureProfile("gemma4")
	if !ok || len(first.LoRATargets) == 0 {
		t.Fatalf("prof.LookupArchitectureProfile(gemma4) ok=%v targets=%v, want a populated profile", ok, first.LoRATargets)
	}
	original := first.LoRATargets[0]
	first.LoRATargets[0] = "mutated-target"
	second, _ := prof.LookupArchitectureProfile("gemma4")
	if second.LoRATargets[0] == "mutated-target" {
		t.Fatal("prof.LookupArchitectureProfile returned aliased LoRATargets backing array")
	}
	if second.LoRATargets[0] != original {
		t.Fatalf("LookupArchitectureProfile(gemma4).LoRATargets[0] = %q, want stable %q", second.LoRATargets[0], original)
	}
}

// --- LookupArchitectureProfileRef -----------------------------------------

// TestArchitecture_LookupArchitectureProfileRef_Good pins the hot-path pointer
// resolver: a canonical id and a Transformers class name both resolve to the
// shared registry entry with the expected id.
func TestArchitecture_LookupArchitectureProfileRef_Good(t *testing.T) {
	cases := map[string]string{
		"gemma4":                                "gemma4",
		"qwen3_moe":                             "qwen3_moe",
		"MiniMaxM2ForCausalLM":                  "minimax_m2",
		"Gemma4UnifiedForConditionalGeneration": "gemma4_unified",
		"bert":                                  "bert",
	}
	for input, wantID := range cases {
		t.Run(input, func(t *testing.T) {
			ref, ok := prof.LookupArchitectureProfileRef(input)
			if !ok {
				t.Fatalf("prof.LookupArchitectureProfileRef(%q) ok = false", input)
			}
			if ref == nil || ref.ID != wantID {
				t.Fatalf("prof.LookupArchitectureProfileRef(%q) = %+v, want id %q", input, ref, wantID)
			}
		})
	}
}

// TestArchitecture_LookupArchitectureProfileRef_Bad pins the miss path: an
// unregistered value yields a nil pointer and ok=false, so a hot-path caller
// never dereferences a guessed entry.
func TestArchitecture_LookupArchitectureProfileRef_Bad(t *testing.T) {
	ref, ok := prof.LookupArchitectureProfileRef("nonexistent_family")
	if ok {
		t.Fatal("prof.LookupArchitectureProfileRef(nonexistent) ok = true, want false")
	}
	if ref != nil {
		t.Fatalf("prof.LookupArchitectureProfileRef(nonexistent) = %+v, want nil", ref)
	}
}

// TestArchitecture_LookupArchitectureProfileRef_Ugly pins the empty-value
// short-circuit: an empty string returns nil/false before the resolver pipeline
// runs, and the two lookups agree (the ref id equals the cloned lookup id).
func TestArchitecture_LookupArchitectureProfileRef_Ugly(t *testing.T) {
	if ref, ok := prof.LookupArchitectureProfileRef(""); ok || ref != nil {
		t.Fatalf("prof.LookupArchitectureProfileRef(\"\") = %+v, %v; want nil, false", ref, ok)
	}
	// A canonical id resolves the same via Ref and the cloning Lookup.
	ref, okRef := prof.LookupArchitectureProfileRef("gemma4")
	clone, okClone := prof.LookupArchitectureProfile("gemma4")
	if !okRef || !okClone || ref.ID != clone.ID {
		t.Fatalf("Ref/Lookup disagree: ref=%+v (%v), clone id=%q (%v)", ref, okRef, clone.ID, okClone)
	}
}

// --- ArchitectureID -------------------------------------------------------

// TestArchitecture_ArchitectureID_Good pins the compact-substring fallback arm —
// the path a config model_type takes when it carries a family fragment but is
// not a clean Transformers class name and is not a direct alias. Each input
// folds, compacts, and matches one family substring; the order is authoritative
// (moe/next arms before bare qwen3, rerank class names before bare bert).
func TestArchitecture_ArchitectureID_Good(t *testing.T) {
	cases := map[string]string{
		"my_qwen35moe_v2":                    "qwen3_6_moe",
		"custom-qwen3.6":                     "qwen3_6",
		"qwen3moe_local":                     "qwen3_moe",
		"qwen3next_x":                        "qwen3_next",
		"minimaxm2_q4":                       "minimax_m2",
		"mixtral_local":                      "mixtral",
		"my-mistral":                         "mistral",
		"deepseek_local":                     "deepseek",
		"gptoss_x":                           "gpt_oss",
		"phi_local":                          "phi",
		"DebertaV2ForSequenceClassification": "bert_rerank",
		"bert_local":                         "bert",
	}
	for in, want := range cases {
		t.Run(in, func(t *testing.T) {
			if got := prof.ArchitectureID(in); got != want {
				t.Fatalf("prof.ArchitectureID(%q) = %q, want %q", in, got, want)
			}
		})
	}
}

// TestArchitecture_ArchitectureID_Bad pins the empty-id path: an input that
// names no family is returned in its normalised form rather than guessed at —
// it does not resolve to any registered id.
func TestArchitecture_ArchitectureID_Bad(t *testing.T) {
	if got := prof.ArchitectureID("Totally-Unknown.Thing"); got != "totally_unknown_thing" {
		t.Fatalf("prof.ArchitectureID(unknown) = %q, want normalised pass-through", got)
	}
	if _, ok := prof.LookupArchitectureProfileRef(prof.ArchitectureID("Totally-Unknown.Thing")); ok {
		t.Fatal("ArchitectureID(unknown) resolved to a registered profile, want unregistered")
	}
}

// TestArchitecture_ArchitectureID_Ugly pins the boundary inputs: empty and
// whitespace-only values yield the empty id after trim.
func TestArchitecture_ArchitectureID_Ugly(t *testing.T) {
	for _, in := range []string{"", "  ", "\t\n"} {
		if got := prof.ArchitectureID(in); got != "" {
			t.Fatalf("prof.ArchitectureID(%q) = %q, want empty after trim", in, got)
		}
	}
}

// --- IsGemma4TargetArchitecture -------------------------------------------

// TestArchitecture_IsGemma4TargetArchitecture_Good pins the target-family
// membership: the gemma4 text/unified target ids and their Transformers class
// names all report true.
func TestArchitecture_IsGemma4TargetArchitecture_Good(t *testing.T) {
	for _, architecture := range []string{
		"gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text",
		"Gemma4ForConditionalGeneration", "Gemma4UnifiedForConditionalGeneration",
		"Gemma4ForCausalLM", "Gemma4TextForCausalLM",
	} {
		t.Run(architecture, func(t *testing.T) {
			if !prof.IsGemma4TargetArchitecture(architecture) {
				t.Fatalf("prof.IsGemma4TargetArchitecture(%q) = false, want true", architecture)
			}
		})
	}
}

// TestArchitecture_IsGemma4TargetArchitecture_Bad pins the exclusions: the
// attached drafter and non-gemma families report false even though the drafter
// is a gemma family member.
func TestArchitecture_IsGemma4TargetArchitecture_Bad(t *testing.T) {
	for _, architecture := range []string{
		"gemma4_assistant", "Gemma4AssistantForCausalLM", "gemma3", "qwen3", "llama",
	} {
		t.Run(architecture, func(t *testing.T) {
			if prof.IsGemma4TargetArchitecture(architecture) {
				t.Fatalf("prof.IsGemma4TargetArchitecture(%q) = true, want false", architecture)
			}
		})
	}
}

// TestArchitecture_IsGemma4TargetArchitecture_Ugly pins the empty-input edge:
// an empty architecture is not a target.
func TestArchitecture_IsGemma4TargetArchitecture_Ugly(t *testing.T) {
	if prof.IsGemma4TargetArchitecture("") {
		t.Fatal("prof.IsGemma4TargetArchitecture(\"\") = true, want false for empty")
	}
}

// --- IsGemma4LargeVariant --------------------------------------------------

// TestArchitecture_IsGemma4LargeVariant_Good pins the large-variant predicate: a
// gemma4 target with at least 16 attention heads takes the large-variant
// suppressor path.
func TestArchitecture_IsGemma4LargeVariant_Good(t *testing.T) {
	for _, architecture := range []string{"Gemma4ForConditionalGeneration", "gemma4_unified_text", "gemma4_text"} {
		t.Run(architecture, func(t *testing.T) {
			if !prof.IsGemma4LargeVariant(architecture, 16) {
				t.Fatalf("prof.IsGemma4LargeVariant(%q, 16) = false, want true", architecture)
			}
		})
	}
}

// TestArchitecture_IsGemma4LargeVariant_Bad pins the exclusions: a small head
// count, the attached drafter, and a non-gemma family all report false.
func TestArchitecture_IsGemma4LargeVariant_Bad(t *testing.T) {
	cases := []struct {
		architecture string
		heads        int
	}{
		{"gemma4_text", 8},
		{"Gemma4AssistantForCausalLM", 16},
		{"qwen3", 16},
		{"gemma4_text", 0},
	}
	for _, tc := range cases {
		if prof.IsGemma4LargeVariant(tc.architecture, tc.heads) {
			t.Fatalf("prof.IsGemma4LargeVariant(%q, %d) = true, want false", tc.architecture, tc.heads)
		}
	}
}

// TestArchitecture_IsGemma4LargeVariant_Ugly pins the head-count boundary: 16 is
// large, 15 is not, for the same target architecture.
func TestArchitecture_IsGemma4LargeVariant_Ugly(t *testing.T) {
	if !prof.IsGemma4LargeVariant("gemma4_text", 16) {
		t.Fatal("prof.IsGemma4LargeVariant(gemma4_text, 16) = false, want true at boundary")
	}
	if prof.IsGemma4LargeVariant("gemma4_text", 15) {
		t.Fatal("prof.IsGemma4LargeVariant(gemma4_text, 15) = true, want false below boundary")
	}
}

// --- DefaultThinkingEnabled ------------------------------------------------

// TestArchitecture_DefaultThinkingEnabled_Good pins the thinking default: the
// gemma4 family renders its chat prompt with reasoning on by default.
func TestArchitecture_DefaultThinkingEnabled_Good(t *testing.T) {
	for _, architecture := range []string{"gemma4", "gemma4_text", "Gemma4ForConditionalGeneration", "gemma4_unified"} {
		t.Run(architecture, func(t *testing.T) {
			if !prof.DefaultThinkingEnabled(architecture) {
				t.Fatalf("prof.DefaultThinkingEnabled(%q) = false, want true", architecture)
			}
		})
	}
}

// TestArchitecture_DefaultThinkingEnabled_Bad pins the families that do not
// default to thinking — a plain qwen/llama renders without reasoning on.
func TestArchitecture_DefaultThinkingEnabled_Bad(t *testing.T) {
	for _, architecture := range []string{"qwen3", "llama", "mistral", "gemma3"} {
		t.Run(architecture, func(t *testing.T) {
			if prof.DefaultThinkingEnabled(architecture) {
				t.Fatalf("prof.DefaultThinkingEnabled(%q) = true, want false", architecture)
			}
		})
	}
}

// TestArchitecture_DefaultThinkingEnabled_Ugly pins the miss branch: empty,
// whitespace, and unknown architectures report the safe default (false) rather
// than panicking or guessing.
func TestArchitecture_DefaultThinkingEnabled_Ugly(t *testing.T) {
	for _, architecture := range []string{"", "  ", "nonexistent_family"} {
		if prof.DefaultThinkingEnabled(architecture) {
			t.Fatalf("prof.DefaultThinkingEnabled(%q) = true, want false for unknown", architecture)
		}
	}
}

// --- AttachedOnlyArchitecture ----------------------------------------------

// TestArchitecture_AttachedOnlyArchitecture_Good pins the attached-only flag:
// the gemma4 assistant drafter can only load attached to a target.
func TestArchitecture_AttachedOnlyArchitecture_Good(t *testing.T) {
	for _, architecture := range []string{"gemma4_assistant", "Gemma4AssistantForCausalLM"} {
		t.Run(architecture, func(t *testing.T) {
			if !prof.AttachedOnlyArchitecture(architecture) {
				t.Fatalf("prof.AttachedOnlyArchitecture(%q) = false, want true", architecture)
			}
		})
	}
}

// TestArchitecture_AttachedOnlyArchitecture_Bad pins the standalone families:
// gemma4 and other targets report false, so a normal load is not rejected.
func TestArchitecture_AttachedOnlyArchitecture_Bad(t *testing.T) {
	for _, architecture := range []string{"gemma4", "gemma4_text", "qwen3", "bert"} {
		t.Run(architecture, func(t *testing.T) {
			if prof.AttachedOnlyArchitecture(architecture) {
				t.Fatalf("prof.AttachedOnlyArchitecture(%q) = true, want false", architecture)
			}
		})
	}
}

// TestArchitecture_AttachedOnlyArchitecture_Ugly pins the miss branch: empty,
// whitespace, and unknown architectures report the safe default (false).
func TestArchitecture_AttachedOnlyArchitecture_Ugly(t *testing.T) {
	for _, architecture := range []string{"", "  ", "nonexistent_family"} {
		if prof.AttachedOnlyArchitecture(architecture) {
			t.Fatalf("prof.AttachedOnlyArchitecture(%q) = true, want false for unknown", architecture)
		}
	}
}

// --- ChatTemplateName ------------------------------------------------------

// TestArchitecture_ChatTemplateName_Good pins the advertised template ids: the
// gemma4 family advertises its own template, gemma3 the gemma template, a qwen
// id the qwen default, and a llama alias the llama template.
func TestArchitecture_ChatTemplateName_Good(t *testing.T) {
	cases := map[string]string{
		"Gemma4ForConditionalGeneration": "gemma4",
		"gemma4_unified_text":            "gemma4",
		"Gemma3ForCausalLM":              "gemma",
		"qwen3_6_moe":                    "qwen",
		"llama3":                         "llama",
	}
	for architecture, want := range cases {
		t.Run(architecture, func(t *testing.T) {
			if got := prof.ChatTemplateName(architecture); got != want {
				t.Fatalf("prof.ChatTemplateName(%q) = %q, want %q", architecture, got, want)
			}
		})
	}
}

// TestArchitecture_ChatTemplateName_Bad pins the families that advertise no
// template id: the attached drafter and the staged MoE loaders return the
// empty string (or their own id) rather than a chat template they cannot render.
func TestArchitecture_ChatTemplateName_Bad(t *testing.T) {
	for _, architecture := range []string{"Gemma4AssistantForCausalLM", "MiniMaxM2ForCausalLM", "DeepseekV3ForCausalLM"} {
		t.Run(architecture, func(t *testing.T) {
			if got := prof.ChatTemplateName(architecture); got != "" {
				t.Fatalf("prof.ChatTemplateName(%q) = %q, want empty for non-chat staged loader", architecture, got)
			}
		})
	}
}

// TestArchitecture_ChatTemplateName_Ugly pins the unknown/empty edges: an
// unregistered name and the empty string both yield no template id.
func TestArchitecture_ChatTemplateName_Ugly(t *testing.T) {
	for _, architecture := range []string{"unknown", "", "  "} {
		if got := prof.ChatTemplateName(architecture); got != "" {
			t.Fatalf("prof.ChatTemplateName(%q) = %q, want empty", architecture, got)
		}
	}
}

// --- DefaultLoRATargets ----------------------------------------------------

// TestArchitecture_DefaultLoRATargets_Good pins the registered narrow default
// LoRA set for the gemma4 family across its id and class-name aliases.
func TestArchitecture_DefaultLoRATargets_Good(t *testing.T) {
	want := []string{"q_proj", "v_proj", "o_proj"}
	for _, architecture := range []string{
		"gemma4", "gemma4_text", "gemma4_unified",
		"Gemma4ForConditionalGeneration", "Gemma4UnifiedForConditionalGeneration",
	} {
		t.Run(architecture, func(t *testing.T) {
			requireExactLoRATargets(t, prof.DefaultLoRATargets(architecture), want)
		})
	}
}

// TestArchitecture_DefaultLoRATargets_Bad pins the unknown-family path: an
// unregistered architecture yields nil rather than a guessed target set.
func TestArchitecture_DefaultLoRATargets_Bad(t *testing.T) {
	if got := prof.DefaultLoRATargets("nonexistent_family"); got != nil {
		t.Fatalf("prof.DefaultLoRATargets(nonexistent) = %v, want nil", got)
	}
	// The attached drafter declares no LoRA defaults.
	if got := prof.DefaultLoRATargets("gemma4_assistant"); len(got) != 0 {
		t.Fatalf("prof.DefaultLoRATargets(gemma4_assistant) = %v, want none for the attached drafter", got)
	}
}

// TestArchitecture_DefaultLoRATargets_Ugly pins the copy contract: the returned
// slice is a copy, so mutating it must not corrupt the registry's defaults.
func TestArchitecture_DefaultLoRATargets_Ugly(t *testing.T) {
	want := []string{"q_proj", "v_proj", "o_proj"}
	prof.DefaultLoRATargets("gemma4")[0] = "mutated"
	requireExactLoRATargets(t, prof.DefaultLoRATargets("gemma4"), want)
}

// --- LoRATargetPath --------------------------------------------------------

// TestArchitecture_LoRATargetPath_Good pins the key → projection-path
// canonicalisation for the gemma4 family — both the bare key and its already-
// qualified form resolve to the same projection path.
func TestArchitecture_LoRATargetPath_Good(t *testing.T) {
	cases := []struct {
		key      string
		wantPath string
	}{
		{"q_proj", "self_attn.q_proj"},
		{"self_attn.q_proj", "self_attn.q_proj"},
		{"gate_proj", "mlp.gate_proj"},
		{"mlp.up_proj", "mlp.up_proj"},
		{"router.proj", "router.proj"},
		{"per_layer_input_gate", "per_layer_input_gate"},
	}
	for _, tc := range cases {
		t.Run(tc.key, func(t *testing.T) {
			path, ok := prof.LoRATargetPath("gemma4", tc.key)
			if !ok || path != tc.wantPath {
				t.Fatalf("prof.LoRATargetPath(gemma4, %q) = %q, %v; want %q, true", tc.key, path, ok, tc.wantPath)
			}
		})
	}
}

// TestArchitecture_LoRATargetPath_Bad pins the unknown-architecture path: a
// family the registry does not carry yields ok=false and no path.
func TestArchitecture_LoRATargetPath_Bad(t *testing.T) {
	path, ok := prof.LoRATargetPath("nonexistent_family", "q_proj")
	if ok || path != "" {
		t.Fatalf("prof.LoRATargetPath(nonexistent, q_proj) = %q, %v; want \"\", false", path, ok)
	}
}

// TestArchitecture_LoRATargetPath_Ugly pins the unknown-key edge: a key the
// family does not register (a vision tower projection on gemma4) yields
// ok=false rather than a guessed path.
func TestArchitecture_LoRATargetPath_Ugly(t *testing.T) {
	if path, ok := prof.LoRATargetPath("gemma4", "vision_tower.q_proj"); ok || path != "" {
		t.Fatalf("prof.LoRATargetPath(gemma4, vision_tower.q_proj) = %q, %v; want \"\", false", path, ok)
	}
}

// --- SafeLoRATarget --------------------------------------------------------

// TestArchitecture_SafeLoRATarget_Good pins the safe-by-default set: the
// attention and MLP projections resolve to known paths outside the family's
// opt-in extended set, so they are safe to enable by default.
func TestArchitecture_SafeLoRATarget_Good(t *testing.T) {
	for _, key := range []string{"q_proj", "self_attn.q_proj", "gate_proj", "mlp.up_proj"} {
		t.Run(key, func(t *testing.T) {
			if !prof.SafeLoRATarget("gemma4", key) {
				t.Fatalf("prof.SafeLoRATarget(gemma4, %q) = false, want true", key)
			}
		})
	}
}

// TestArchitecture_SafeLoRATarget_Bad pins the opt-in extended targets: a key
// that resolves to a path in the family's extended set is not safe by default.
func TestArchitecture_SafeLoRATarget_Bad(t *testing.T) {
	for _, key := range []string{"router.proj", "per_layer_input_gate", "per_layer_projection"} {
		t.Run(key, func(t *testing.T) {
			if prof.SafeLoRATarget("gemma4", key) {
				t.Fatalf("prof.SafeLoRATarget(gemma4, %q) = true, want false for extended target", key)
			}
		})
	}
}

// TestArchitecture_SafeLoRATarget_Ugly pins the miss edges: an unknown
// architecture and an unknown key both report not-safe rather than guessing.
func TestArchitecture_SafeLoRATarget_Ugly(t *testing.T) {
	if prof.SafeLoRATarget("nonexistent_family", "q_proj") {
		t.Fatal("prof.SafeLoRATarget(nonexistent, q_proj) = true, want false")
	}
	if prof.SafeLoRATarget("gemma4", "vision_tower.q_proj") {
		t.Fatal("prof.SafeLoRATarget(gemma4, vision_tower.q_proj) = true, want false for unknown key")
	}
}

// --- CanonicalWeightName ---------------------------------------------------

// TestArchitecture_CanonicalWeightName_Good pins the text-tensor canonicalisation
// for the gemma4 family: wrapper prefixes are stripped and text tensors are
// re-rooted under "model.".
func TestArchitecture_CanonicalWeightName_Good(t *testing.T) {
	cases := []struct {
		name string
		want string
	}{
		{"language_model.model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight"},
		{"model.language_model.model.model.layers.1.mlp.down_proj.scales", "model.layers.1.mlp.down_proj.scales"},
		{"model.layers.2.self_attn.o_proj.weight", "model.layers.2.self_attn.o_proj.weight"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := prof.CanonicalWeightName("gemma4", tc.name)
			if !ok || got != tc.want {
				t.Fatalf("prof.CanonicalWeightName(gemma4, %q) = %q, %v; want %q, true", tc.name, got, ok, tc.want)
			}
		})
	}
}

// TestArchitecture_CanonicalWeightName_Bad pins the dropped-tensor path: non-text
// helper tensors (vision/audio towers, quant min/max sidecars) return ok=false
// so the loader skips them.
func TestArchitecture_CanonicalWeightName_Bad(t *testing.T) {
	for _, name := range []string{
		"language_model.model.layers.0.self_attn.q_proj.input_max",
		"model.vision_tower.patch_embedding.weight",
		"language_model.embed_audio.embedding_projection.weight",
	} {
		t.Run(name, func(t *testing.T) {
			got, ok := prof.CanonicalWeightName("gemma4", name)
			if ok || got != "" {
				t.Fatalf("prof.CanonicalWeightName(gemma4, %q) = %q, %v; want \"\", false", name, got, ok)
			}
		})
	}
}

// TestArchitecture_CanonicalWeightName_Ugly pins the unknown-architecture edge:
// a family with no weight rules passes the name through unchanged with ok=true,
// so the engine names no family.
func TestArchitecture_CanonicalWeightName_Ugly(t *testing.T) {
	got, ok := prof.CanonicalWeightName("nonexistent_family", "model.layers.0.weight")
	if !ok || got != "model.layers.0.weight" {
		t.Fatalf("prof.CanonicalWeightName(nonexistent, ...) = %q, %v; want pass-through, true", got, ok)
	}
}

// --- TrimWeightWrapperPrefix -----------------------------------------------

// TestArchitecture_TrimWeightWrapperPrefix_Good pins the single-wrapper strip:
// one declared gemma4 wrapper prefix is removed and ok reports the match.
func TestArchitecture_TrimWeightWrapperPrefix_Good(t *testing.T) {
	cases := []struct {
		name string
		want string
	}{
		{"language_model.model.layers.0", "layers.0"},
		{"model.embed_tokens.weight", "embed_tokens.weight"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := prof.TrimWeightWrapperPrefix("gemma4", tc.name)
			if !ok || got != tc.want {
				t.Fatalf("prof.TrimWeightWrapperPrefix(gemma4, %q) = %q, %v; want %q, true", tc.name, got, ok, tc.want)
			}
		})
	}
}

// TestArchitecture_TrimWeightWrapperPrefix_Bad pins the unknown-architecture
// no-op: a family the registry does not carry returns the name unchanged and
// ok=false.
func TestArchitecture_TrimWeightWrapperPrefix_Bad(t *testing.T) {
	got, ok := prof.TrimWeightWrapperPrefix("nonexistent_family", "model.layers.0")
	if ok || got != "model.layers.0" {
		t.Fatalf("prof.TrimWeightWrapperPrefix(nonexistent) = %q, %v; want model.layers.0, false", got, ok)
	}
}

// TestArchitecture_TrimWeightWrapperPrefix_Ugly pins the no-prefix-match edge: a
// gemma4 name that carries none of the declared wrapper prefixes is returned
// unchanged with ok=false.
func TestArchitecture_TrimWeightWrapperPrefix_Ugly(t *testing.T) {
	got, ok := prof.TrimWeightWrapperPrefix("gemma4", "unrelated.tensor.name")
	if ok || got != "unrelated.tensor.name" {
		t.Fatalf("prof.TrimWeightWrapperPrefix(gemma4, unmatched) = %q, %v; want unchanged, false", got, ok)
	}
}

// --- ArchitectureIDs -------------------------------------------------------

// TestArchitecture_ArchitectureIDs_Good pins the enumerated id list: it leads
// with the registry order and contains the broad family set, every id non-empty.
func TestArchitecture_ArchitectureIDs_Good(t *testing.T) {
	ids := prof.ArchitectureIDs()
	if len(ids) < 12 {
		t.Fatalf("prof.ArchitectureIDs len = %d, want broad family list", len(ids))
	}
	if ids[0] != "gemma2" || ids[1] != "gemma3" || ids[2] != "gemma3_text" {
		t.Fatalf("prof.ArchitectureIDs head = %v, want stable registry order", ids[:3])
	}
	index := map[string]bool{}
	for _, id := range ids {
		if id == "" {
			t.Fatal("prof.ArchitectureIDs contained an empty id")
		}
		index[id] = true
	}
	for _, want := range []string{"gemma4_text", "qwen3_moe", "minimax_m2", "bert", "bert_rerank"} {
		if !index[want] {
			t.Fatalf("prof.ArchitectureIDs missing %q", want)
		}
	}
}

// TestArchitecture_ArchitectureIDs_Bad pins that the list carries no duplicate
// ids — every enumerated family appears exactly once.
func TestArchitecture_ArchitectureIDs_Bad(t *testing.T) {
	seen := map[string]bool{}
	for _, id := range prof.ArchitectureIDs() {
		if seen[id] {
			t.Fatalf("prof.ArchitectureIDs returned duplicate id %q", id)
		}
		seen[id] = true
	}
}

// TestArchitecture_ArchitectureIDs_Ugly pins the registry round-trip: every id
// ArchitectureIDs enumerates resolves back to a profile with that exact id, so
// the list never names an unresolvable id.
func TestArchitecture_ArchitectureIDs_Ugly(t *testing.T) {
	for _, id := range prof.ArchitectureIDs() {
		ref, ok := prof.LookupArchitectureProfileRef(id)
		if !ok || ref.ID != id {
			t.Fatalf("ArchitectureIDs id %q did not round-trip: ref=%+v ok=%v", id, ref, ok)
		}
	}
}

// --- NormalizeArchitecture -------------------------------------------------

// TestArchitecture_NormalizeArchitecture_Good pins the canonical alias contract:
// dotted/dashed/cased aliases fold and map to their canonical id. This is the
// single source of truth the memory, gguf, model, and minimax packages share.
func TestArchitecture_NormalizeArchitecture_Good(t *testing.T) {
	cases := map[string]string{
		"qwen3_5":             "qwen3_6",
		"qwen3.6":             "qwen3_6",
		"qwen3_5_text":        "qwen3_6",
		"qwen3_5_moe":         "qwen3_6_moe",
		"qwen2.5":             "qwen2",
		"MiniMax-M2":          "minimax_m2",
		"  bert ":             "bert",
		"bert_cross_encoder":  "bert_rerank",
		"bert_model":          "bert",
		"phi3":                "phi",
		"moonshot":            "kimi",
		"gemma4_unified":      "gemma4_unified",
		"gemma4_unified_text": "gemma4_text",
	}
	for in, want := range cases {
		t.Run(in, func(t *testing.T) {
			if got := prof.NormalizeArchitecture(in); got != want {
				t.Fatalf("prof.NormalizeArchitecture(%q) = %q, want %q", in, got, want)
			}
		})
	}
}

// TestArchitecture_NormalizeArchitecture_Bad pins the pass-through path: an
// unknown value is returned in its normalised (lowercased, '-'/'.'-folded) form
// rather than mapped to a guessed canonical id.
func TestArchitecture_NormalizeArchitecture_Bad(t *testing.T) {
	cases := map[string]string{
		"unknown-arch":      "unknown_arch",
		"Some-New.Arch":     "some_new_arch",
		"Totally-Unrelated": "totally_unrelated",
	}
	for in, want := range cases {
		t.Run(in, func(t *testing.T) {
			if got := prof.NormalizeArchitecture(in); got != want {
				t.Fatalf("prof.NormalizeArchitecture(%q) = %q, want normalised pass-through %q", in, got, want)
			}
		})
	}
}

// TestArchitecture_NormalizeArchitecture_Ugly pins the boundary inputs: a
// non-ASCII value takes the heap-stable fallback (semantics identical to the
// ASCII path), and a whitespace-only value trims to empty.
func TestArchitecture_NormalizeArchitecture_Ugly(t *testing.T) {
	if got := prof.NormalizeArchitecture("Café-Gemma3"); got != "café_gemma3" {
		t.Fatalf("prof.NormalizeArchitecture(non-ASCII) = %q, want café_gemma3", got)
	}
	if got := prof.NormalizeArchitecture("   "); got != "" {
		t.Fatalf("prof.NormalizeArchitecture(whitespace) = %q, want empty after trim", got)
	}
}

// --- ArchitectureFromTransformersName --------------------------------------

// TestArchitecture_ArchitectureFromTransformersName_Good pins the HF class-name →
// canonical-id contract — the single source of truth the gguf, model, and hf
// packages share. The two previously-lost arms (qwen3_6, gemma4_assistant) are
// pinned here.
func TestArchitecture_ArchitectureFromTransformersName_Good(t *testing.T) {
	cases := map[string]string{
		"Gemma4ForConditionalGeneration":        "gemma4",
		"Gemma4UnifiedForConditionalGeneration": "gemma4_unified",
		"Gemma4MultimodalForCausalLM":           "gemma4",
		"Gemma4VisionForCausalLM":               "gemma4",
		"Gemma4ForCausalLM":                     "gemma4_text",
		"Gemma4AssistantForCausalLM":            "gemma4_assistant",
		"Gemma3ForCausalLM":                     "gemma3",
		"Gemma2ForCausalLM":                     "gemma2",
		"Qwen3ForCausalLM":                      "qwen3",
		"Qwen3MoeForCausalLM":                   "qwen3_moe",
		"Qwen3NextForCausalLM":                  "qwen3_next",
		"Qwen3_6ForConditionalGeneration":       "qwen3_6",
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
		"MoonshotForCausalLM":                   "kimi",
		"HermesForCausalLM":                     "hermes",
		"GraniteForCausalLM":                    "granite",
		"GlmForCausalLM":                        "glm",
		"BertModel":                             "bert",
		"BertForSequenceClassification":         "bert_rerank",
		"RobertaForSequenceClassification":      "bert_rerank",
	}
	for in, want := range cases {
		t.Run(in, func(t *testing.T) {
			if got := prof.ArchitectureFromTransformersName(in); got != want {
				t.Fatalf("prof.ArchitectureFromTransformersName(%q) = %q, want %q", in, got, want)
			}
		})
	}
}

// TestArchitecture_ArchitectureFromTransformersName_Bad pins the no-match path: a
// class name that names no known family returns the empty string, so the
// resolver falls through to its next signal.
func TestArchitecture_ArchitectureFromTransformersName_Bad(t *testing.T) {
	for _, in := range []string{"UnknownForCausalLM", "SomethingForCausalLM", "NotAModelClass"} {
		t.Run(in, func(t *testing.T) {
			if got := prof.ArchitectureFromTransformersName(in); got != "" {
				t.Fatalf("prof.ArchitectureFromTransformersName(%q) = %q, want empty", in, got)
			}
		})
	}
}

// TestArchitecture_ArchitectureFromTransformersName_Ugly pins the substring
// precedence at the boundary: a sequence-classification BERT class resolves to
// the rerank id (probed before bare bert), and the empty string yields the
// empty id.
func TestArchitecture_ArchitectureFromTransformersName_Ugly(t *testing.T) {
	if got := prof.ArchitectureFromTransformersName("XLMRobertaForSequenceClassification"); got != "bert_rerank" {
		t.Fatalf("ArchitectureFromTransformersName(xlm-roberta-seq-cls) = %q, want bert_rerank (precedence over bert)", got)
	}
	if got := prof.ArchitectureFromTransformersName(""); got != "" {
		t.Fatalf("ArchitectureFromTransformersName(\"\") = %q, want empty", got)
	}
}
