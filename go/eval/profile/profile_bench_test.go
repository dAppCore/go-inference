// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the profile package — BuiltinAlgorithmProfiles,
// LookupAlgorithmProfile, AlgorithmCapabilities (the algorithm side),
// plus BuiltinArchitectureProfiles, LookupArchitectureProfile,
// ArchitectureID, ArchitectureIDs (the architecture side).
//
// Per AX-11 — these surfaces are touched on every CapabilityReport()
// call (algorithm capabilities is appended), on every model-load
// architecture-resolution path (LookupArchitectureProfile /
// ArchitectureID), and on every profile clone/list. Cold-start latency
// budget flows through them.
//
// Run:    go test -bench='BenchmarkProfile' -benchmem -run='^$' ./go/profile

package profile_test

import (
	"testing"

	"dappco.re/go/inference"
	prof "dappco.re/go/inference/eval/profile"
)

// Sinks defeat compiler DCE.
var (
	profileBenchSinkAlgorithms     []prof.AlgorithmProfile
	profileBenchSinkAlgorithm      prof.AlgorithmProfile
	profileBenchSinkAlgorithmOK    bool
	profileBenchSinkCapabilities   []inference.Capability
	profileBenchSinkArchitectures  []prof.ModelArchitectureProfile
	profileBenchSinkArchitecture   prof.ModelArchitectureProfile
	profileBenchSinkArchitectureRP *prof.ModelArchitectureProfile
	profileBenchSinkArchOK         bool
	profileBenchSinkArchIDs        []string
	profileBenchSinkArchID         string
	profileBenchSinkWeightName     string
	profileBenchSinkLoRATargets    []string
	profileBenchSinkNormalized     string
	profileBenchSinkResolved       string
)

// --- BuiltinAlgorithmProfiles ---
// Full-list clone of the 14-entry built-in algorithm matrix. Fires
// once per CapabilityReport via AlgorithmCapabilities.

func BenchmarkProfile_BuiltinAlgorithmProfiles(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkAlgorithms = prof.BuiltinAlgorithmProfiles()
	}
}

// --- LookupAlgorithmProfile ---
// Linear scan over the built-in list — hits early (first entry),
// late (deep in list), and miss-path.

func BenchmarkProfile_LookupAlgorithmProfile_EarlyHit(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkAlgorithm, profileBenchSinkAlgorithmOK = prof.LookupAlgorithmProfile(inference.CapabilityScheduler)
	}
}

func BenchmarkProfile_LookupAlgorithmProfile_LateHit(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkAlgorithm, profileBenchSinkAlgorithmOK = prof.LookupAlgorithmProfile(inference.CapabilityCacheDisk)
	}
}

func BenchmarkProfile_LookupAlgorithmProfile_Miss(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkAlgorithm, profileBenchSinkAlgorithmOK = prof.LookupAlgorithmProfile(inference.CapabilityID("not-a-real-cap"))
	}
}

// --- AlgorithmCapabilities ---
// Fires on every CapabilityReport — produces the inference.Capability
// slice consumed by the metalCapabilityReport.

func BenchmarkProfile_AlgorithmCapabilities(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkCapabilities = prof.AlgorithmCapabilities()
	}
}

// --- BuiltinArchitectureProfiles ---
// Deep clone of the architecture matrix.

func BenchmarkProfile_BuiltinArchitectureProfiles(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitectures = prof.BuiltinArchitectureProfiles()
	}
}

// --- LookupArchitectureProfile ---

func BenchmarkProfile_LookupArchitectureProfile_Native(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitecture, profileBenchSinkArchOK = prof.LookupArchitectureProfile("qwen3")
	}
}

// Transformers-name path — exercises architectureFromTransformersName.
func BenchmarkProfile_LookupArchitectureProfile_TransformersName(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitecture, profileBenchSinkArchOK = prof.LookupArchitectureProfile("Qwen3ForCausalLM")
	}
}

// Alias path — exercises the second-pass alias scan.
func BenchmarkProfile_LookupArchitectureProfile_Alias(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitecture, profileBenchSinkArchOK = prof.LookupArchitectureProfile("MiniMaxM2ForCausalLM")
	}
}

func BenchmarkProfile_LookupArchitectureProfile_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitecture, profileBenchSinkArchOK = prof.LookupArchitectureProfile("")
	}
}

// --- LookupArchitectureProfileRef ---
// Pointer-into-static-table form used by read-only callers (planFit,
// archSupported, archNativeRuntime, tuningRuntimeForArchitecture,
// memory.NewPlan, model.pack inspectors). Should be zero-alloc.

func BenchmarkProfile_LookupArchitectureProfileRef_Native(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitectureRP, profileBenchSinkArchOK = prof.LookupArchitectureProfileRef("qwen3")
	}
}

func BenchmarkProfile_LookupArchitectureProfileRef_TransformersName(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitectureRP, profileBenchSinkArchOK = prof.LookupArchitectureProfileRef("Qwen3ForCausalLM")
	}
}

func BenchmarkProfile_LookupArchitectureProfileRef_Alias(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitectureRP, profileBenchSinkArchOK = prof.LookupArchitectureProfileRef("MiniMaxM2ForCausalLM")
	}
}

func BenchmarkProfile_LookupArchitectureProfileRef_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitectureRP, profileBenchSinkArchOK = prof.LookupArchitectureProfileRef("")
	}
}

// --- ArchitectureID ---
// Hot path during model-load — resolves Transformers names back to
// internal architecture IDs.

func BenchmarkProfile_ArchitectureID_TransformersName(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchID = prof.ArchitectureID("Gemma4ForConditionalGeneration")
	}
}

func BenchmarkProfile_ArchitectureID_Direct(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchID = prof.ArchitectureID("qwen3")
	}
}

func BenchmarkProfile_ArchitectureID_Normalised(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchID = prof.ArchitectureID("qwen-3.5")
	}
}

func BenchmarkProfile_ArchitectureID_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchID = prof.ArchitectureID("")
	}
}

// --- ArchitectureIDs ---
// Slice clone of the full architecture-ID list.

func BenchmarkProfile_ArchitectureIDs(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchIDs = prof.ArchitectureIDs()
	}
}

// --- CanonicalWeightName ---
// The per-tensor weight-name sanitiser fired by the gemma4 loader on every
// checkpoint tensor (go/pkg/metal/model/gemma4/weights.go) — the hottest
// production caller of this package, not a per-load report surface. Three
// shapes: a re-rooted tensor (the lone intrinsic alloc, "model."+trimmed, a new
// string value the loader consumes), a wrapper-strip-only tensor (sub-string of
// the input, zero-alloc), and an unknown architecture (pass-through, zero-alloc).

func BenchmarkProfile_CanonicalWeightName_Rerooted(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkWeightName, profileBenchSinkArchOK = prof.CanonicalWeightName("gemma4", "model.language_model.model.layers.0.self_attn.q_proj.weight")
	}
}

func BenchmarkProfile_CanonicalWeightName_WrapperStrip(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Strips the "model." wrapper but matches no WeightModelPrefix, so the
		// result is a sub-string of the input (TrimPrefix), not a re-root.
		profileBenchSinkWeightName, profileBenchSinkArchOK = prof.CanonicalWeightName("gemma4", "model.lm_head.weight")
	}
}

func BenchmarkProfile_CanonicalWeightName_Unknown(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkWeightName, profileBenchSinkArchOK = prof.CanonicalWeightName("not-a-real-arch", "model.layers.0.weight")
	}
}

// --- DefaultLoRATargets ---
// Defensive clone of a family's narrow default LoRA target set — resolved once
// per adapter setup (gemma4/policy.go), a cold path. The lone alloc is the
// contract clone protecting the registry singleton from caller mutation.

func BenchmarkProfile_DefaultLoRATargets(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkLoRATargets = prof.DefaultLoRATargets("gemma4")
	}
}

// --- NormalizeArchitecture ---
// The fold-to-'_' canonicaliser under ArchitectureID / LookupArchitectureProfile
// and called directly by the gguf/hf/model config-probe paths on a config's
// model_type. Its zero-alloc shape (foldArchitectureKeyInto + unsafe.String over
// a stack buffer) holds for an already-canonical input — whether it matches an
// alias arm (the common model-load case) or falls through unchanged. The lone
// intrinsic alloc is the default branch's heap-stable copy of a *folded* miss:
// an unknown architecture carrying caps/'-'/'.' (e.g. "not-a-real-arch") has no
// canonical home, so the canonicalised bytes are copied off the stack before
// return. That alloc is the package's one non-obvious floor — these benches pin
// it. Production never feeds this path on a hot loop (config model_types are
// already canonical; the weight loader passes the "gemma4" literal), so the cost
// stays where it belongs: an unknown-arch diagnostic edge, not a per-token cost.

// Already-canonical input that matches an alias arm — returns a literal, no alloc.
func BenchmarkProfile_NormalizeArchitecture_CanonicalHit(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkNormalized = prof.NormalizeArchitecture("minimax_m2")
	}
}

// Already-folded miss — no fold, default returns the original value, no alloc.
func BenchmarkProfile_NormalizeArchitecture_CanonicalMiss(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkNormalized = prof.NormalizeArchitecture("not_a_real_arch")
	}
}

// Foldable miss — caps/'-' fold to '_', no alias arm matches, so the default
// branch heap-copies the folded bytes. This is the lone intrinsic alloc.
func BenchmarkProfile_NormalizeArchitecture_FoldableMiss(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkNormalized = prof.NormalizeArchitecture("not-a-real-arch")
	}
}

// --- ArchitectureFromTransformersName ---
// Maps a HuggingFace transformers class name to a canonical id via the
// zero-alloc compactArchitectureNameInto stack-buffer scan. Both the contains-
// prefix arm (qwen3_moe) and the miss path should be 0 alloc.

func BenchmarkProfile_ArchitectureFromTransformersName_Hit(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkNormalized = prof.ArchitectureFromTransformersName("Qwen3MoeForCausalLM")
	}
}

func BenchmarkProfile_ArchitectureFromTransformersName_Miss(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkNormalized = prof.ArchitectureFromTransformersName("NotARealForCausalLM")
	}
}

// --- ResolveArchitecture ---
// The config-signal resolver fired once per model load (pkg/metal/model.go) over
// model_type, text_config.model_type, and the architectures class list. The
// multimodal-wrapper text-tower refinement (the Gemma-4 path) walks ref lookups,
// all zero-alloc; the only alloc would come from a model_type that is a foldable
// miss, which a real config never is.

func BenchmarkProfile_ResolveArchitecture_TextTower(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkResolved = prof.ResolveArchitecture("gemma4", "gemma4_text", []string{"Gemma4ForConditionalGeneration"})
	}
}

func BenchmarkProfile_ResolveArchitecture_ArchitecturesOnly(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkResolved = prof.ResolveArchitecture("", "", []string{"Qwen3MoeForCausalLM"})
	}
}
