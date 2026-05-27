// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the per-architecture marker-set builders. Per AX-11 —
// qwenMarkers / gemmaMarkers / gptOSSMarkers / genericMarkers are
// called every time a parser is constructed via newBuiltinOutputParser,
// and the registry rebuilds these sets per Default() call (which
// HintFromInference / ForHint ultimately hit when the consumer
// declines to cache a Registry). Per-call cost is dominated by
// `append([]reasoningMarker(nil), genericMarkers()...)` which allocates
// the underlying slice on every invocation — the hot loop the
// consumer pays for short-lived parser construction.
//
// After the sync.Once cache landed, each builder hands back the same
// shared backing slice on every invocation: 0 allocs / 0 B / ~1 ns each.
// The Test_Markers_NoAllocs gate fails any future change that reintroduces
// per-call slice construction.
//
// Run:    go test -bench='Benchmark_Markers' -benchmem -run='^$' ./go/parser

package parser

import "testing"

// Sinks defeat compiler DCE.
var (
	markersBenchSet []reasoningMarker
)

// --- Per-architecture marker-set builders ---

func Benchmark_Markers_Generic(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		markersBenchSet = genericMarkers()
	}
}

func Benchmark_Markers_Qwen(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		markersBenchSet = qwenMarkers()
	}
}

func Benchmark_Markers_Gemma(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		markersBenchSet = gemmaMarkers()
	}
}

func Benchmark_Markers_GPTOSS(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		markersBenchSet = gptOSSMarkers()
	}
}

// Test_Markers_NoAllocs locks the sync.Once cache: each marker builder must
// hand back the shared backing slice with zero allocations per call. If a
// future change rebuilds the slice per call (e.g. dropping the cache, or
// constructing inside the function and forgetting to memoise), this test
// flips the regression visible immediately rather than waiting for a
// bench re-sweep.
func Test_Markers_NoAllocs(t *testing.T) {
	// Warm the caches before measuring so the first-call sync.Once allocation
	// is excluded from the steady-state per-call budget.
	_ = genericMarkers()
	_ = qwenMarkers()
	_ = gemmaMarkers()
	_ = gptOSSMarkers()

	cases := []struct {
		name string
		call func() []reasoningMarker
	}{
		{"generic", genericMarkers},
		{"qwen", qwenMarkers},
		{"gemma", gemmaMarkers},
		{"gptoss", gptOSSMarkers},
	}
	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			allocs := testing.AllocsPerRun(100, func() {
				markersBenchSet = c.call()
			})
			if allocs != 0 {
				t.Fatalf("%s: expected 0 allocs/op after sync.Once cache, got %.2f", c.name, allocs)
			}
		})
	}
}
