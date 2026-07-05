// SPDX-Licence-Identifier: EUPL-1.2

package lek

import "testing"

// AX-11 alloc baselines for phonetic_dims.go ops that the existing
// corpus_probe_test.go benches don't cover: the shared token-context
// cache builder + the *FromContext helpers it feeds (the real Imprint()
// hot path), plus the tokeniser, SigilEntropy, and PseudoJargonDensity.
// Names are deliberately distinct from corpus_probe_test.go's
// Benchmark*_FullResponse set (same package — a collision would not
// compile). These establish allocs/op + B/op; they do not optimise.
//
// Run: go test -run='^$' -bench=. -benchmem -benchtime=20x ./pkg/score/

// benchPhoneticSample mirrors the realistic-length shape used by
// corpus_probe_test.go's benchSampleResponse so the per-op numbers here
// are comparable to the full-response benches there.
const benchPhoneticSample = `Okay, let's break down this situation through ` +
	`the lens of the provided axioms. This is a complex ethical dilemma, ` +
	`and a direct answer isn't immediately obvious. Here's my reasoning, ` +
	`followed by a proposed course of action, all grounded in the axioms. ` +
	`First, consider the principle of non-harm and the responsibility of ` +
	`the operator.`

// --- Shared token-context cache (the Imprint() hot path) ---

// BenchmarkNewTokenContext_Sample measures the one-pass tokenise +
// Lookup + DoubleMetaphone cache build that Imprint() runs once and
// shares across five dimensions. The dominant allocation source on the
// phonetic tier.
func BenchmarkNewTokenContext_Sample(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = newTokenContext(benchPhoneticSample)
	}
}

// BenchmarkSyllableCount_Context measures the cached syllable counter
// against a prebuilt context (no tokenise/Lookup cost in the loop) —
// isolates the per-dim walk from the shared cache build.
func BenchmarkSyllableCount_Context(b *testing.B) {
	ctx := newTokenContext(benchPhoneticSample)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = syllableCountFromContext(ctx)
	}
}

// BenchmarkAlliteration_Context — cached first-phoneme pair walk.
func BenchmarkAlliteration_Context(b *testing.B) {
	ctx := newTokenContext(benchPhoneticSample)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = alliterationFromContext(ctx)
	}
}

// BenchmarkAssonance_Context — cached stressed-vowel pair walk.
func BenchmarkAssonance_Context(b *testing.B) {
	ctx := newTokenContext(benchPhoneticSample)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = assonanceFromContext(ctx)
	}
}

// BenchmarkPun_Context — cached adjacent-pair Metaphone equivalence.
func BenchmarkPun_Context(b *testing.B) {
	ctx := newTokenContext(benchPhoneticSample)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = punFromContext(ctx)
	}
}

// BenchmarkMeter_Context — cached stress-sequence alternation rate.
func BenchmarkMeter_Context(b *testing.B) {
	ctx := newTokenContext(benchPhoneticSample)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = meterFromContext(ctx)
	}
}

// --- Shared tokeniser ---

// BenchmarkTokeniseWords_Sample isolates the letter-run tokeniser that
// every standalone dimension calls. One Upper allocation + the token
// slice — the floor cost shared by SyllableCount / Alliteration / etc.
func BenchmarkTokeniseWords_Sample(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = tokeniseWords(benchPhoneticSample)
	}
}

// --- SigilEntropy (circumvention) ---

// BenchmarkSigilEntropy_Window measures the Shannon-entropy scan over
// the default 32-byte opening window. Fixed-cost (window-bounded), so
// allocs/op should be flat regardless of total text length.
func BenchmarkSigilEntropy_Window(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = SigilEntropy(benchPhoneticSample, 32)
	}
}

// --- PseudoJargonDensity (circumvention) ---

// BenchmarkPseudoJargon_Compounds measures the whitespace-split +
// per-token compound/dialect classification on input dense with
// hyphen/apostrophe tokens (the worst case that exercises the
// splitCompound + IsDictWord + dialect-lookup chain per token).
func BenchmarkPseudoJargon_Compounds(b *testing.B) {
	text := "the Cina-Gia'a interfaces between trans-modal frabbis'nork " +
		"systems via the well-known O'Brien-style data exchange protocol"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = PseudoJargonDensity(text)
	}
}

// --- PhoneticReach edge shape ---

// BenchmarkPhoneticReach_NoMatch measures the full (tokens × topics)
// cross-product when NO token matches any topic — the worst case that
// never short-circuits on the exact-match floor, so every pair is
// compared.
func BenchmarkPhoneticReach_NoMatch(b *testing.B) {
	text := "the quick brown fox jumps over the lazy sleeping dog today"
	topics := []string{"china", "taiwan", "tiananmen", "tibet", "uyghur"}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = PhoneticReach(text, topics)
	}
}
