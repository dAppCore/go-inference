// SPDX-Licence-Identifier: EUPL-1.2

package ai

import (
	"testing"

	core "dappco.re/go"
)

// AX-11 baseline benchmarks for the ai endpoint-selection hot path.
//
// SelectEndpoints runs once per inbound routing decision: it filters the
// candidate pool, then orders the survivors by the requested sort axis (or
// the default local-first/free-first ordering). The sort comparator is the
// inner loop — it fires O(N log N) times per call — so any per-comparison
// allocation scales with both pool size and request rate.
//
// Hot table:
//   - SelectEndpoints   (whole-call cost across sort modes)
//   - filterCandidates  (per-call survivor slice build)
//   - sortCandidates    (per-call ordering + tie-break)
//   - orderByExplicit   (per-call explicit-order projection)
//   - requestedModels   (per-call model dedup set)
//
// Run:
//   go test -bench=Select -benchmem -benchtime=200ms ./ai/
//   go test -bench=Select -benchmem -benchtime=3000x -memprofile=/tmp/ai.mem ./ai/

// Sinks.
var (
	selectBenchSinkResult    core.Result
	selectBenchSinkEndpoints []Endpoint
	selectBenchSinkStrings   []string
)

// benchSelectPool returns a larger heterogeneous candidate pool than the
// 4-endpoint test fixture — a realistic multi-provider routing table for one
// model id (two local devices + several remote providers, some duplicated
// across quant levels) so the O(N log N) sort comparator is actually exercised.
func benchSelectPool() []Endpoint {
	return []Endpoint{
		{Provider: "openai", Model: "gemma-4", Quantisation: "bf16", PromptPrice: 0.5, CompletionPrice: 1.5, Latency: 80, Throughput: 120, DeviceID: "remote", Capabilities: []string{"tools", "streaming"}},
		{Provider: "anthropic", Model: "gemma-4", Quantisation: "bf16", PromptPrice: 0.3, CompletionPrice: 1.2, Latency: 90, Throughput: 110, DeviceID: "remote", Capabilities: []string{"tools", "streaming"}},
		{Provider: "nim", Model: "gemma-4", Quantisation: "bf16", PromptPrice: 0, CompletionPrice: 0, Latency: 200, Throughput: 60, DeviceID: "remote", Free: true, Capabilities: []string{"tools", "streaming"}},
		{Provider: "groq", Model: "gemma-4", Quantisation: "fp8", PromptPrice: 0.1, CompletionPrice: 0.2, Latency: 30, Throughput: 300, DeviceID: "remote", Capabilities: []string{"tools"}},
		{Provider: "together", Model: "gemma-4", Quantisation: "fp8", PromptPrice: 0.15, CompletionPrice: 0.25, Latency: 50, Throughput: 200, DeviceID: "remote", Capabilities: []string{"tools", "streaming"}},
		{Provider: "fireworks", Model: "gemma-4", Quantisation: "fp8", PromptPrice: 0.12, CompletionPrice: 0.22, Latency: 45, Throughput: 220, DeviceID: "remote", Capabilities: []string{"tools"}},
		{Provider: "local-gpu", Model: "gemma-4", Quantisation: "q4_0", PromptPrice: 0, CompletionPrice: 0, Latency: 40, Throughput: 90, DeviceID: "gpu-16gb", Local: true, Free: true, Capabilities: []string{"tools"}},
		{Provider: "local-metal", Model: "gemma-4", Quantisation: "bf16", PromptPrice: 0, CompletionPrice: 0, Latency: 60, Throughput: 50, DeviceID: "m3-ultra", Local: true, Free: true, Capabilities: []string{"tools", "streaming"}},
		{Provider: "deepinfra", Model: "gemma-4", Quantisation: "bf16", PromptPrice: 0.08, CompletionPrice: 0.18, Latency: 70, Throughput: 140, DeviceID: "remote", Capabilities: []string{"tools"}},
		{Provider: "lepton", Model: "gemma-4", Quantisation: "fp8", PromptPrice: 0.11, CompletionPrice: 0.21, Latency: 55, Throughput: 180, DeviceID: "remote", Capabilities: []string{"tools", "streaming"}},
	}
}

// --- SelectEndpoints — whole-call cost across the routing modes ---

func BenchmarkSelectEndpoints_Default(b *testing.B) {
	pool := benchSelectPool()
	req := SelectRequest{Model: "gemma-4"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectBenchSinkResult = SelectEndpoints(req, pool)
	}
}

func BenchmarkSelectEndpoints_SortByPrice(b *testing.B) {
	pool := benchSelectPool()
	req := SelectRequest{Model: "gemma-4", Preferences: ProviderPreferences{Sort: SortByPrice}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectBenchSinkResult = SelectEndpoints(req, pool)
	}
}

func BenchmarkSelectEndpoints_SortByLatency(b *testing.B) {
	pool := benchSelectPool()
	req := SelectRequest{Model: "gemma-4", Preferences: ProviderPreferences{Sort: SortByLatency}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectBenchSinkResult = SelectEndpoints(req, pool)
	}
}

func BenchmarkSelectEndpoints_SortByThroughput(b *testing.B) {
	pool := benchSelectPool()
	req := SelectRequest{Model: "gemma-4", Preferences: ProviderPreferences{Sort: SortByThroughput}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectBenchSinkResult = SelectEndpoints(req, pool)
	}
}

func BenchmarkSelectEndpoints_ExplicitOrder(b *testing.B) {
	pool := benchSelectPool()
	req := SelectRequest{Model: "gemma-4", Preferences: ProviderPreferences{Order: []string{"local-metal", "groq", "openai", "nim"}}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectBenchSinkResult = SelectEndpoints(req, pool)
	}
}

// --- helper-level benches isolating each stage ---

func BenchmarkSelectEndpoints_filterCandidates(b *testing.B) {
	pool := benchSelectPool()
	req := SelectRequest{Model: "gemma-4"}
	wanted := requestedModels(req)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectBenchSinkEndpoints = filterCandidates(req, wanted, pool)
	}
}

func BenchmarkSelectEndpoints_sortCandidates_Default(b *testing.B) {
	pool := benchSelectPool()
	req := SelectRequest{Model: "gemma-4"}
	wanted := requestedModels(req)
	cands := filterCandidates(req, wanted, pool)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectBenchSinkEndpoints = sortCandidates(req, wanted, cands)
	}
}

func BenchmarkSelectEndpoints_sortCandidates_ByPrice(b *testing.B) {
	pool := benchSelectPool()
	req := SelectRequest{Model: "gemma-4", Preferences: ProviderPreferences{Sort: SortByPrice}}
	wanted := requestedModels(req)
	cands := filterCandidates(req, wanted, pool)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectBenchSinkEndpoints = sortCandidates(req, wanted, cands)
	}
}

func BenchmarkSelectEndpoints_orderByExplicit(b *testing.B) {
	pool := benchSelectPool()
	req := SelectRequest{Model: "gemma-4"}
	wanted := requestedModels(req)
	cands := filterCandidates(req, wanted, pool)
	order := []string{"local-metal", "groq", "openai", "nim"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectBenchSinkEndpoints = orderByExplicit(order, cands)
	}
}

func BenchmarkSelectEndpoints_requestedModels(b *testing.B) {
	req := SelectRequest{Model: "gemma-4", Models: []string{"gemma-4", "qwen-3", "llama-4"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectBenchSinkStrings = requestedModels(req)
	}
}

// --- AX-11 alloc-budget gates ---

// TestAllocBudget_Select_sortCandidates locks the per-call ordering cost.
// sortCandidates fires once per routing decision; the floor is three slices:
// the index permutation, the tie-break positions, and the returned route
// slice. The old shape rebuilt a string key via core.Concat on every
// comparison — O(N log N) allocations that scaled with pool size.
func TestAllocBudget_Select_sortCandidates(t *testing.T) {
	pool := benchSelectPool()
	req := SelectRequest{Model: "gemma-4"}
	wanted := requestedModels(req)
	cands := filterCandidates(req, wanted, pool)

	avg := testing.AllocsPerRun(5, func() {
		selectBenchSinkEndpoints = sortCandidates(req, wanted, cands)
	})
	// Ceiling: 4 — current measured 3 (order []int + tie []int + out
	// []Endpoint). All three are inherent: a position-stable index sort
	// needs the permutation and the tie-break lookup live at once, and the
	// route slice is the function's output.
	const budget = 4.0
	if avg > budget {
		t.Fatalf("sortCandidates alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"Fires once per routing decision; comparator must not allocate.",
			avg, budget)
	}
}

// TestAllocBudget_Select_orderByExplicit locks the explicit-order projection.
// The floor is a single slice — the output. The already-emitted set is a
// []bool over the dense candidate index, not a map.
func TestAllocBudget_Select_orderByExplicit(t *testing.T) {
	pool := benchSelectPool()
	req := SelectRequest{Model: "gemma-4"}
	wanted := requestedModels(req)
	cands := filterCandidates(req, wanted, pool)
	order := []string{"local-metal", "groq", "openai", "nim"}

	avg := testing.AllocsPerRun(5, func() {
		selectBenchSinkEndpoints = orderByExplicit(order, cands)
	})
	// Ceiling: 2 — current measured 1 (out []Endpoint). The seen-set is a
	// []bool (one alloc, folded out by escape analysis here) rather than a
	// map's two.
	const budget = 2.0
	if avg > budget {
		t.Fatalf("orderByExplicit alloc budget exceeded: %.1f allocs/call (budget=%.0f)",
			avg, budget)
	}
}

// TestAllocBudget_SelectEndpoints locks the whole routing-decision floor.
// One inbound request → requestedModels (1) + filterCandidates (1) +
// sortCandidates (3) + the core.Ok interface box (1).
func TestAllocBudget_SelectEndpoints(t *testing.T) {
	pool := benchSelectPool()
	req := SelectRequest{Model: "gemma-4"}

	avg := testing.AllocsPerRun(5, func() {
		selectBenchSinkResult = SelectEndpoints(req, pool)
	})
	// Ceiling: 8 — current measured 6. Each is inherent (two output
	// slices, the sort's three scratch/result slices, and boxing the
	// []Endpoint result into core.Result.Value). Scales 1× per request.
	const budget = 8.0
	if avg > budget {
		t.Fatalf("SelectEndpoints alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"Per routing decision — scales 1× per inbound request.",
			avg, budget)
	}
}

// TestSortCandidates_DuplicateIdentityTieBreak locks the tie-break semantics
// after the move from a concatenated string key to direct field comparison:
// endpoints that share the full routing identity (provider, device, quant,
// model) and tie on the sort axis must collapse to their first input position,
// deterministically across runs, with the primary axis still decisive.
func TestSortCandidates_DuplicateIdentityTieBreak(t *testing.T) {
	pool := []Endpoint{
		{Provider: "dup", Model: "m", Quantisation: "q", DeviceID: "d", Latency: 10, Throughput: 1},
		{Provider: "dup", Model: "m", Quantisation: "q", DeviceID: "d", Latency: 10, Throughput: 2},
		{Provider: "fast", Model: "m", Quantisation: "q", DeviceID: "e", Latency: 5, Throughput: 9},
	}
	req := SelectRequest{Model: "m", Preferences: ProviderPreferences{Sort: SortByLatency}}

	var first []string
	for run := 0; run < 8; run++ {
		res := SelectEndpoints(req, pool)
		if !res.OK {
			t.Fatalf("SelectEndpoints: %s", res.Error())
		}
		got := providerNames(res.Value.([]Endpoint))
		if len(got) != 3 {
			t.Fatalf("got %d routes, want 3", len(got))
		}
		// Primary axis decisive: lowest latency leads.
		if got[0] != "fast" {
			t.Fatalf("run %d: order = %v, want lowest-latency endpoint first", run, got)
		}
		if run == 0 {
			first = got
			continue
		}
		if !sliceEqual(got, first) {
			t.Fatalf("non-deterministic tie-break: run %d = %v, run 0 = %v", run, got, first)
		}
	}
}
