// SPDX-Licence-Identifier: EUPL-1.2

package ai

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// AX-11 baseline benchmarks for PlanDifferentialLoad and friends.
//
// PlanDifferentialLoad fires on every model-load decision — every time
// an agent or research workflow stages a base/fine-tune pair. The
// helper predicates (modelIdentityEmpty, adapterIdentityEmpty,
// sameModelIdentity) fire inside the planning loop and on every route
// resolution; they govern the floor of the planning surface.
//
// Run:
//   go test -bench=. -benchmem -benchtime=300ms ./ai/...

// Sinks.
var (
	dlBenchSinkResult core.Result
	dlBenchSinkBool   bool
)

// --- fixtures ---

func benchModelIdentity() inference.ModelIdentity {
	return inference.ModelIdentity{
		Path:         "/models/gemma3-1b",
		Hash:         "sha256:abc123def456",
		Architecture: "gemma3",
	}
}

func benchAdapterIdentity() inference.AdapterIdentity {
	return inference.AdapterIdentity{
		Path:   "/adapters/cladius-lora",
		Hash:   "sha256:deadbeef",
		Format: "safetensors",
	}
}

// --- PlanDifferentialLoad — per-model-load planning entry ---

func BenchmarkDifferentialLoader_PlanDifferentialLoad_BaseOnly(b *testing.B) {
	req := DifferentialLoadRequest{Base: benchModelIdentity()}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dlBenchSinkResult = PlanDifferentialLoad(req)
	}
}

func BenchmarkDifferentialLoader_PlanDifferentialLoad_ReuseAdapter(b *testing.B) {
	req := DifferentialLoadRequest{
		Base:    benchModelIdentity(),
		Adapter: benchAdapterIdentity(),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dlBenchSinkResult = PlanDifferentialLoad(req)
	}
}

func BenchmarkDifferentialLoader_PlanDifferentialLoad_Compare(b *testing.B) {
	tuned := benchModelIdentity()
	tuned.Hash = "sha256:tunedhash"
	req := DifferentialLoadRequest{
		Base:  benchModelIdentity(),
		Tuned: tuned,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dlBenchSinkResult = PlanDifferentialLoad(req)
	}
}

// --- modelIdentityEmpty / adapterIdentityEmpty — predicates inside the loop ---

func BenchmarkDifferentialLoader_modelIdentityEmpty_Full(b *testing.B) {
	model := benchModelIdentity()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dlBenchSinkBool = modelIdentityEmpty(model)
	}
}

func BenchmarkDifferentialLoader_modelIdentityEmpty_Empty(b *testing.B) {
	model := inference.ModelIdentity{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dlBenchSinkBool = modelIdentityEmpty(model)
	}
}

func BenchmarkDifferentialLoader_sameModelIdentity_Same(b *testing.B) {
	left := benchModelIdentity()
	right := benchModelIdentity()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dlBenchSinkBool = sameModelIdentity(left, right)
	}
}

func BenchmarkDifferentialLoader_sameModelIdentity_Different(b *testing.B) {
	left := benchModelIdentity()
	right := benchModelIdentity()
	right.Hash = "sha256:differenthash"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dlBenchSinkBool = sameModelIdentity(left, right)
	}
}

// --- AX-11 alloc-budget gates ---

// TestAllocBudget_DifferentialLoader_modelIdentityEmpty locks the
// per-call predicate. Fires inside the planning loop on every
// PlanDifferentialLoad — must stay at zero allocs.
func TestAllocBudget_DifferentialLoader_modelIdentityEmpty(t *testing.T) {
	model := benchModelIdentity()

	// Behavioural lock — full identity is not empty.
	if modelIdentityEmpty(model) {
		t.Fatalf("modelIdentityEmpty incorrectly reported full identity as empty")
	}
	if !modelIdentityEmpty(inference.ModelIdentity{}) {
		t.Fatalf("modelIdentityEmpty failed to detect empty identity")
	}

	avg := testing.AllocsPerRun(5, func() {
		dlBenchSinkBool = modelIdentityEmpty(model)
	})
	// Ceiling: 0 — pure string trim + comparison. core.Trim on a
	// non-whitespace string is alloc-free (returns input substring).
	const budget = 0.0
	if avg > budget {
		t.Fatalf("modelIdentityEmpty alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"Fires inside every PlanDifferentialLoad — per-load floor.",
			avg, budget)
	}
}

// TestAllocBudget_DifferentialLoader_sameModelIdentity locks the
// per-call identity comparison.
func TestAllocBudget_DifferentialLoader_sameModelIdentity(t *testing.T) {
	left := benchModelIdentity()
	right := benchModelIdentity()

	// Behavioural lock — identical identities match by hash.
	if !sameModelIdentity(left, right) {
		t.Fatalf("sameModelIdentity failed on identical identities")
	}
	differentRight := right
	differentRight.Hash = "sha256:different"
	if sameModelIdentity(left, differentRight) {
		t.Fatalf("sameModelIdentity matched on different hashes")
	}

	avg := testing.AllocsPerRun(5, func() {
		dlBenchSinkBool = sameModelIdentity(left, right)
	})
	// Ceiling: 0 — modelIdentityEmpty calls + string compares only.
	const budget = 0.0
	if avg > budget {
		t.Fatalf("sameModelIdentity alloc budget exceeded: %.1f allocs/call (budget=%.0f)",
			avg, budget)
	}
}
