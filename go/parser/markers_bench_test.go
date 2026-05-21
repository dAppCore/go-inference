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
