// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the project-seed durable-checkpoint primitives.
// Per AX-11 — ProjectSeed is the per-project root; NewProjectSeed
// fires per workspace entry, WakeRequest / PlanContinuation fire per
// session boundary, and CheckWakeCompatibility fires before every
// model-state restore. The Labels / Metadata maps are the per-call
// allocation drivers; both shapes are benched here.
//
// Run:    go test -bench='Benchmark' -benchmem -run='^$' ./state

package state

import "testing"

// Sinks defeat compiler DCE. Distinct names per state-package bench file.
var (
	projectSeedSinkSeed   ProjectSeed
	projectSeedSinkWake   WakeRequest
	projectSeedSinkPlan   ProjectSeedContinuationPlan
	projectSeedSinkReport WakeCompatibilityReport
)

// labelsMap builds a deterministic map of n distinct entries for
// benching map-merge + clone shapes. Each key is unique so the bench
// reflects the real per-entry map cost, not collision dedup.
func labelsMap(n int) map[string]string {
	out := make(map[string]string, n)
	for i := range n {
		out[labelsKey(i)] = labelsValue(i)
	}
	return out
}

func labelsKey(i int) string {
	// Inline base-36 digits keep the key short + unique without
	// pulling core.Sprintf onto the hot fixture path.
	const digits = "0123456789abcdefghijklmnopqrstuvwxyz"
	if i < 36 {
		return "k" + string(digits[i])
	}
	return "k" + string(digits[i/36]) + string(digits[i%36])
}

func labelsValue(i int) string {
	const digits = "0123456789abcdefghijklmnopqrstuvwxyz"
	if i < 36 {
		return "v" + string(digits[i])
	}
	return "v" + string(digits[i/36]) + string(digits[i%36])
}

// --- NewProjectSeed (per-workspace entry — sets defaults) ---

func BenchmarkProjectSeed_NewProjectSeed_Minimal(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		projectSeedSinkSeed = NewProjectSeed(ProjectSeedOptions{
			BaseURI:   "state://lthn/projects",
			ProjectID: "core/go-mlx",
		})
	}
}

func BenchmarkProjectSeed_NewProjectSeed_Defaulted(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// All URIs left empty so the default-fill branch runs.
		projectSeedSinkSeed = NewProjectSeed(ProjectSeedOptions{
			ProjectID: "core/go-mlx",
		})
	}
}

func BenchmarkProjectSeed_NewProjectSeed_Labels_10(b *testing.B) {
	labels := labelsMap(10)
	metadata := labelsMap(10)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		projectSeedSinkSeed = NewProjectSeed(ProjectSeedOptions{
			BaseURI:   "state://lthn/projects",
			ProjectID: "core/go-mlx",
			Labels:    labels,
			Metadata:  metadata,
		})
	}
}

func BenchmarkProjectSeed_NewProjectSeed_Labels_100(b *testing.B) {
	labels := labelsMap(100)
	metadata := labelsMap(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		projectSeedSinkSeed = NewProjectSeed(ProjectSeedOptions{
			BaseURI:   "state://lthn/projects",
			ProjectID: "core/go-mlx",
			Labels:    labels,
			Metadata:  metadata,
		})
	}
}

// --- WakeRequest (per session boot) ---

func BenchmarkProjectSeed_WakeRequest_Minimal(b *testing.B) {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
	})
	opts := ProjectSeedWakeOptions{
		Store:     "store",
		Model:     ModelIdentity{ID: "gemma4", Hash: "model-a"},
		Tokenizer: TokenizerIdentity{Hash: "tok-a"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		projectSeedSinkWake = seed.WakeRequest(opts)
	}
}

func BenchmarkProjectSeed_WakeRequest_Labels_10(b *testing.B) {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
		Labels:    labelsMap(10),
	})
	opts := ProjectSeedWakeOptions{
		Store:  "store",
		Model:  ModelIdentity{ID: "gemma4"},
		Labels: labelsMap(10),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		projectSeedSinkWake = seed.WakeRequest(opts)
	}
}

func BenchmarkProjectSeed_WakeRequest_Labels_100(b *testing.B) {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
		Labels:    labelsMap(100),
	})
	opts := ProjectSeedWakeOptions{
		Store:  "store",
		Model:  ModelIdentity{ID: "gemma4"},
		Labels: labelsMap(100),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		projectSeedSinkWake = seed.WakeRequest(opts)
	}
}

// --- PlanContinuation (per session end — selects sleep shape) ---

func BenchmarkProjectSeed_PlanContinuation_StateCheckpoint(b *testing.B) {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
	})
	opts := ProjectSeedContinuationOptions{
		Mode:  ProjectSeedStateCheckpoint,
		Store: "store",
		Model: ModelIdentity{ID: "gemma4"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		projectSeedSinkPlan = seed.PlanContinuation(opts)
	}
}

func BenchmarkProjectSeed_PlanContinuation_ReuseCurrent(b *testing.B) {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
	})
	opts := ProjectSeedContinuationOptions{Mode: ProjectSeedReuseCurrent}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		projectSeedSinkPlan = seed.PlanContinuation(opts)
	}
}

func BenchmarkProjectSeed_PlanContinuation_SummaryWindow(b *testing.B) {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
	})
	opts := ProjectSeedContinuationOptions{Mode: ProjectSeedSummaryWindow}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		projectSeedSinkPlan = seed.PlanContinuation(opts)
	}
}

func BenchmarkProjectSeed_PlanContinuation_Hybrid(b *testing.B) {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
	})
	opts := ProjectSeedContinuationOptions{
		Mode:  ProjectSeedHybrid,
		Store: "store",
		Model: ModelIdentity{ID: "gemma4"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		projectSeedSinkPlan = seed.PlanContinuation(opts)
	}
}

func BenchmarkProjectSeed_PlanContinuation_Labels_100(b *testing.B) {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
		Labels:    labelsMap(100),
		Metadata:  labelsMap(100),
	})
	opts := ProjectSeedContinuationOptions{
		Mode:     ProjectSeedStateCheckpoint,
		Store:    "store",
		Model:    ModelIdentity{ID: "gemma4"},
		Labels:   labelsMap(100),
		Metadata: labelsMap(100),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		projectSeedSinkPlan = seed.PlanContinuation(opts)
	}
}

// --- CheckWakeCompatibility (per restore — gates the wake) ---

func BenchmarkProjectSeed_CheckWakeCompatibility_Compatible(b *testing.B) {
	bundle := Bundle{
		Model:        ModelIdentity{Hash: "model-a", Architecture: "gemma4_text", NumLayers: 28, QuantBits: 4, ContextLength: 4096},
		Tokenizer:    TokenizerIdentity{Hash: "tok-a", ChatTemplate: "chat-a"},
		Adapter:      AdapterIdentity{Hash: "adapter-a", Rank: 8},
		Runtime:      RuntimeIdentity{Backend: "metal", CacheMode: "paged-q8"},
		PromptTokens: 2048,
	}
	req := WakeRequest{
		Model:     ModelIdentity{Hash: "model-a", Architecture: "gemma4_text", NumLayers: 28, QuantBits: 4, ContextLength: 8192},
		Tokenizer: TokenizerIdentity{Hash: "tok-a", ChatTemplate: "chat-a"},
		Adapter:   AdapterIdentity{Hash: "adapter-a", Rank: 8},
		Runtime:   RuntimeIdentity{Backend: "metal", CacheMode: "paged-q8"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		projectSeedSinkReport = CheckWakeCompatibility(bundle, req)
	}
}

func BenchmarkProjectSeed_CheckWakeCompatibility_Incompatible(b *testing.B) {
	bundle := Bundle{
		Model:        ModelIdentity{Hash: "model-a", Architecture: "gemma4_text", NumLayers: 28, QuantBits: 4, ContextLength: 4096},
		Tokenizer:    TokenizerIdentity{Hash: "tok-a", ChatTemplate: "chat-a"},
		Adapter:      AdapterIdentity{Hash: "adapter-a", Rank: 8},
		Runtime:      RuntimeIdentity{Backend: "metal", CacheMode: "paged-q8"},
		PromptTokens: 2048,
	}
	req := WakeRequest{
		Model:     ModelIdentity{Hash: "model-b", Architecture: "qwen3", NumLayers: 28, QuantBits: 8, ContextLength: 1024},
		Tokenizer: TokenizerIdentity{Hash: "tok-b", ChatTemplate: "chat-b"},
		Adapter:   AdapterIdentity{},
		Runtime:   RuntimeIdentity{Backend: "rocm", CacheMode: "paged-q4"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		projectSeedSinkReport = CheckWakeCompatibility(bundle, req)
	}
}

func BenchmarkProjectSeed_CheckWakeCompatibility_Skip(b *testing.B) {
	bundle := Bundle{Model: ModelIdentity{Hash: "model-a"}}
	req := WakeRequest{SkipCompatibilityCheck: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		projectSeedSinkReport = CheckWakeCompatibility(bundle, req)
	}
}
