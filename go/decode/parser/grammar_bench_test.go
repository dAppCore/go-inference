// SPDX-Licence-Identifier: EUPL-1.2

// Allocation contracts for the reasoning-marker grammar (grammar.go). Both
// exported entry points sit on the streaming reasoning path: IsReasoningChannel
// fires once per `<|channel>NAME` open the gpt-oss/gemma4 lanes emit, and
// PairedReasoningMarkers hands the openai extractor + genericMarkers their span
// table. Neither may allocate — the table is a package-owned read-only view and
// the channel test is a bare switch — so these benches pin both to zero.
//
// Run: go test -bench=. -benchmem -run='^$' ./parser/
package parser

import "testing"

// Package sinks defeat dead-code elimination so the benchmarked work survives.
var (
	sinkPairedMarkers []PairedMarker
	sinkReasoningBool bool
)

// PairedReasoningMarkers returns the shared span table by view — no copy, no
// per-call slice header allocation.
func Benchmark_Grammar_PairedReasoningMarkers(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkPairedMarkers = PairedReasoningMarkers()
	}
}

// HIT: a reasoning channel name ("analysis" is gpt-oss harmony's reasoning
// channel) resolves through the switch to true with no allocation.
func Benchmark_Grammar_IsReasoningChannel_Hit(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkReasoningBool = IsReasoningChannel("analysis")
	}
}

// MISS: a content channel name ("final") falls through every case to false —
// the dominant streaming outcome once the visible answer begins.
func Benchmark_Grammar_IsReasoningChannel_Miss(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkReasoningBool = IsReasoningChannel("final")
	}
}
