// SPDX-Licence-Identifier: EUPL-1.2

package slurs

import core "dappco.re/go"

// Benchmarks use placeholder tokens, never real slurs — same discipline as the
// tests. The matcher carries a handful of placeholder terms so the inner term
// loop is exercised (Default's catalogue is seeded empty, so it would never
// iterate). Fixtures cover the request-path shapes: clean lowercase ASCII (the
// fold/Lower no-op fast path), mixed case (forces a Lower copy), a directed
// hit, an l33t-folded hit (forces a Replace copy), and a self-reference window.

var benchMatcher = New([]string{"fooslur", "barslur", "bazslur", "quxslur"})

const (
	benchClean   = "the quick brown fox jumps over the lazy dog and then runs away"
	benchMixed   = "The Quick Brown Fox Jumps Over The Lazy Dog And Then Runs Away"
	benchHitText = "you are an absolute fooslur and everyone here knows it"
	benchLeet    = "what an utter f00slur that person really is to say such things"
	benchSelf    = "i am a fooslur and i am proud to call myself one today"
)

// Package sinks — defeat dead-code elimination of the benchmarked results.
var (
	sinkHit  bool
	sinkTerm string
)

// benchDefault is the production matcher — the Snider-curated catalogue, which
// seeds empty (catalogue.go). Detect calls Match on this every served turn, so
// the empty-catalogue path is the live per-turn shape.
var benchDefault = Default()

func BenchmarkMatcher_Match_Clean(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	var hit bool
	var term string
	for i := 0; i < b.N; i++ {
		hit, term = benchMatcher.Match(benchClean)
	}
	sinkHit, sinkTerm = hit, term
}

func BenchmarkMatcher_Match_Mixedcase(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	var hit bool
	var term string
	for i := 0; i < b.N; i++ {
		hit, term = benchMatcher.Match(benchMixed)
	}
	sinkHit, sinkTerm = hit, term
}

func BenchmarkMatcher_Match_Hit(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	var hit bool
	var term string
	for i := 0; i < b.N; i++ {
		hit, term = benchMatcher.Match(benchHitText)
	}
	sinkHit, sinkTerm = hit, term
}

func BenchmarkMatcher_Match_Leet(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	var hit bool
	var term string
	for i := 0; i < b.N; i++ {
		hit, term = benchMatcher.Match(benchLeet)
	}
	sinkHit, sinkTerm = hit, term
}

func BenchmarkMatcher_Match_SelfRef(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	var hit bool
	var term string
	for i := 0; i < b.N; i++ {
		hit, term = benchMatcher.Match(benchSelf)
	}
	sinkHit, sinkTerm = hit, term
}

// BenchmarkMatcher_Match_EmptyCatalogue is the live production shape: the
// curated catalogue seeds empty, so Detect's per-turn Match hits the
// no-catalogue guard and never folds or walks.
func BenchmarkMatcher_Match_EmptyCatalogue(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	var hit bool
	var term string
	for i := 0; i < b.N; i++ {
		hit, term = benchDefault.Match(benchClean)
	}
	sinkHit, sinkTerm = hit, term
}
