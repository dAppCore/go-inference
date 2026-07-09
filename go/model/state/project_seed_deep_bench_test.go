// SPDX-Licence-Identifier: EUPL-1.2

// Deeper benchmarks for the project-seed durable-checkpoint primitives.
// Per AX-11 — the existing project_seed_bench_test.go covers the main
// constructor + per-session paths. These benches drill into the
// CheckWakeCompatibility partial-mismatch matrix (one reason at a time
// matters, since the report carries them as a slice), the URI-join
// helper (joinURI is on the hot construction path), and the
// PlanContinuation sleep-request assembly that does the heaviest
// per-seed work.
//
// Run:    go test -bench='BenchmarkProjectSeedDeep' -benchmem -run='^$' ./state

package state

import "testing"

// Sinks defeat compiler DCE. Distinct names per state-package bench file.
var (
	psDeepSinkSeed   ProjectSeed
	psDeepSinkPlan   ProjectSeedContinuationPlan
	psDeepSinkReport WakeCompatibilityReport
	psDeepSinkString string
	psDeepSinkMap    map[string]string
)

// --- CheckWakeCompatibility partial-mismatch matrix ---
// One mismatch reason at a time exercises the comparator without other
// branches polluting the per-call cost.

func BenchmarkProjectSeedDeep_CheckCompat_ModelHashMismatch(b *testing.B) {
	bundle := Bundle{
		Model:        ModelIdentity{Hash: "model-a", Architecture: "gemma4", NumLayers: 28, QuantBits: 4, ContextLength: 4096},
		Tokenizer:    TokenizerIdentity{Hash: "tok-a"},
		Adapter:      AdapterIdentity{Hash: "adapter-a"},
		Runtime:      RuntimeIdentity{Backend: "metal"},
		PromptTokens: 2048,
	}
	req := WakeRequest{
		Model:     ModelIdentity{Hash: "model-X", Architecture: "gemma4", NumLayers: 28, QuantBits: 4, ContextLength: 8192},
		Tokenizer: TokenizerIdentity{Hash: "tok-a"},
		Adapter:   AdapterIdentity{Hash: "adapter-a"},
		Runtime:   RuntimeIdentity{Backend: "metal"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		psDeepSinkReport = CheckWakeCompatibility(bundle, req)
	}
}

func BenchmarkProjectSeedDeep_CheckCompat_TokenizerMismatch(b *testing.B) {
	bundle := Bundle{
		Model:     ModelIdentity{Hash: "m"},
		Tokenizer: TokenizerIdentity{Hash: "tok-a", ChatTemplate: "chat-a"},
		Adapter:   AdapterIdentity{Hash: "adapter-a"},
		Runtime:   RuntimeIdentity{Backend: "metal"},
	}
	req := WakeRequest{
		Model:     ModelIdentity{Hash: "m"},
		Tokenizer: TokenizerIdentity{Hash: "tok-b", ChatTemplate: "chat-b"},
		Adapter:   AdapterIdentity{Hash: "adapter-a"},
		Runtime:   RuntimeIdentity{Backend: "metal"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		psDeepSinkReport = CheckWakeCompatibility(bundle, req)
	}
}

func BenchmarkProjectSeedDeep_CheckCompat_AdapterMissing(b *testing.B) {
	bundle := Bundle{
		Model:     ModelIdentity{Hash: "m"},
		Tokenizer: TokenizerIdentity{Hash: "tok-a"},
		Adapter:   AdapterIdentity{Hash: "adapter-a", Rank: 8},
		Runtime:   RuntimeIdentity{Backend: "metal"},
	}
	req := WakeRequest{
		Model:     ModelIdentity{Hash: "m"},
		Tokenizer: TokenizerIdentity{Hash: "tok-a"},
		Adapter:   AdapterIdentity{}, // missing — exercises the bundleActive && !reqActive branch
		Runtime:   RuntimeIdentity{Backend: "metal"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		psDeepSinkReport = CheckWakeCompatibility(bundle, req)
	}
}

func BenchmarkProjectSeedDeep_CheckCompat_AdapterUnexpected(b *testing.B) {
	bundle := Bundle{
		Model:     ModelIdentity{Hash: "m"},
		Tokenizer: TokenizerIdentity{Hash: "tok-a"},
		Adapter:   AdapterIdentity{},
		Runtime:   RuntimeIdentity{Backend: "metal"},
	}
	req := WakeRequest{
		Model:     ModelIdentity{Hash: "m"},
		Tokenizer: TokenizerIdentity{Hash: "tok-a"},
		Adapter:   AdapterIdentity{Hash: "adapter-x", Rank: 8},
		Runtime:   RuntimeIdentity{Backend: "metal"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		psDeepSinkReport = CheckWakeCompatibility(bundle, req)
	}
}

func BenchmarkProjectSeedDeep_CheckCompat_AdapterRankMismatch(b *testing.B) {
	bundle := Bundle{
		Model:     ModelIdentity{Hash: "m"},
		Tokenizer: TokenizerIdentity{Hash: "tok-a"},
		Adapter:   AdapterIdentity{Hash: "adapter-a", Rank: 8, Path: "/a"},
		Runtime:   RuntimeIdentity{Backend: "metal"},
	}
	req := WakeRequest{
		Model:     ModelIdentity{Hash: "m"},
		Tokenizer: TokenizerIdentity{Hash: "tok-a"},
		Adapter:   AdapterIdentity{Hash: "adapter-a", Rank: 16, Path: "/a"},
		Runtime:   RuntimeIdentity{Backend: "metal"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		psDeepSinkReport = CheckWakeCompatibility(bundle, req)
	}
}

func BenchmarkProjectSeedDeep_CheckCompat_RuntimeBackendChange(b *testing.B) {
	// Runtime mismatches emit Warnings, not Reasons — the report stays
	// Compatible:true but carries telemetry.
	bundle := Bundle{
		Model:     ModelIdentity{Hash: "m"},
		Tokenizer: TokenizerIdentity{Hash: "tok-a"},
		Adapter:   AdapterIdentity{Hash: "adapter-a"},
		Runtime:   RuntimeIdentity{Backend: "metal", CacheMode: "paged"},
	}
	req := WakeRequest{
		Model:     ModelIdentity{Hash: "m"},
		Tokenizer: TokenizerIdentity{Hash: "tok-a"},
		Adapter:   AdapterIdentity{Hash: "adapter-a"},
		Runtime:   RuntimeIdentity{Backend: "rocm", CacheMode: "paged-q4"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		psDeepSinkReport = CheckWakeCompatibility(bundle, req)
	}
}

func BenchmarkProjectSeedDeep_CheckCompat_ContextTooSmall(b *testing.B) {
	bundle := Bundle{
		Model:           ModelIdentity{Hash: "m", ContextLength: 4096},
		PromptTokens:    2048,
		GeneratedTokens: 2048,
	}
	req := WakeRequest{
		Model: ModelIdentity{Hash: "m", ContextLength: 1024},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		psDeepSinkReport = CheckWakeCompatibility(bundle, req)
	}
}

// --- PlanContinuation with custom URIs ---
// PlanContinuation defaults the entry/bundle/index URIs from the seed
// when not provided. These benches exercise the override branch where
// the consumer supplies explicit URIs.

func BenchmarkProjectSeedDeep_PlanContinuation_CustomURIs(b *testing.B) {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
	})
	opts := ProjectSeedContinuationOptions{
		Mode:      ProjectSeedStateCheckpoint,
		Store:     "store",
		EntryURI:  "state://override/entry",
		BundleURI: "state://override/entry/bundle",
		IndexURI:  "state://override/entry/index",
		Title:     "override-title",
		Model:     ModelIdentity{ID: "gemma4"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		psDeepSinkPlan = seed.PlanContinuation(opts)
	}
}

func BenchmarkProjectSeedDeep_PlanContinuation_WithParent(b *testing.B) {
	// Parent ref provided — the sleepRequest assembly walks
	// core.FirstNonBlank for entry/bundle/index URIs.
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
	})
	opts := ProjectSeedContinuationOptions{
		Mode:  ProjectSeedStateCheckpoint,
		Store: "store",
		Parent: WakeResult{
			Entry: Ref{
				URI:       "state://parent/entry",
				BundleURI: "state://parent/bundle",
				IndexURI:  "state://parent/index",
			},
		},
		Model: ModelIdentity{ID: "gemma4"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		psDeepSinkPlan = seed.PlanContinuation(opts)
	}
}

// --- NewProjectSeed with mixed defaults ---
// One or two URIs supplied, rest defaulted. Exercises the per-field
// core.FirstNonBlank + joinURI fallback paths in the constructor.

func BenchmarkProjectSeedDeep_NewProjectSeed_PartialURIs(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		psDeepSinkSeed = NewProjectSeed(ProjectSeedOptions{
			BaseURI:   "state://lthn/projects",
			ProjectID: "core/go-mlx",
			EntryURI:  "state://override/entry",
			// BundleURI + IndexURI left empty so the defaults run.
		})
	}
}

func BenchmarkProjectSeedDeep_NewProjectSeed_AllURIs(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		psDeepSinkSeed = NewProjectSeed(ProjectSeedOptions{
			BaseURI:   "state://lthn/projects",
			ProjectID: "core/go-mlx",
			EntryURI:  "state://lthn/projects/core/go-mlx/seed",
			BundleURI: "state://lthn/projects/core/go-mlx/seed/bundle",
			IndexURI:  "state://lthn/projects/core/go-mlx/seed/index",
			Title:     "core/go-mlx seed",
		})
	}
}

// --- WakeRequest with mixed label shapes ---
// labels-only-from-seed vs labels-only-from-opts vs both — the
// merge path's allocator behaviour depends on the empty case.

func BenchmarkProjectSeedDeep_WakeRequest_LabelsSeedOnly(b *testing.B) {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
		Labels:    labelsMap(8),
	})
	opts := ProjectSeedWakeOptions{
		Store: "store",
		Model: ModelIdentity{ID: "gemma4"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		seed.WakeRequest(opts)
	}
}

func BenchmarkProjectSeedDeep_WakeRequest_LabelsOptsOnly(b *testing.B) {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
	})
	opts := ProjectSeedWakeOptions{
		Store:  "store",
		Model:  ModelIdentity{ID: "gemma4"},
		Labels: labelsMap(8),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		seed.WakeRequest(opts)
	}
}

func BenchmarkProjectSeedDeep_WakeRequest_NoLabels(b *testing.B) {
	// Both sides empty — the merge helper takes the early-out path
	// and returns nil without allocating.
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
	})
	opts := ProjectSeedWakeOptions{
		Store: "store",
		Model: ModelIdentity{ID: "gemma4"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		seed.WakeRequest(opts)
	}
}
