// SPDX-Licence-Identifier: EUPL-1.2

package lek

import "testing"

// AX-11 alloc baselines for the per-response Score() load path that the
// phonetic_dims + corpus_probe benches don't cover: Hostility,
// DetectSycophancy, LEK, and the ScorePair-only computeAuthority /
// computeDifferential. Score(text) runs DetectSycophancy + LEK +
// Hostility + imprintScores on every response, so these sit in the same
// per-response tier as Imprint. These establish allocs/op + B/op; they
// do not themselves optimise.
//
// Run: go test -run='^$' -bench=. -benchmem -benchtime=20x ./pkg/score/

// benchScorePathSample mirrors the realistic-length response shape used
// by the phonetic benches so the per-op numbers are comparable.
const benchScorePathSample = `You're absolutely right, and I apologise ` +
	`for the confusion. That's a fantastic question. Honestly, the ` +
	`incompetent operator made a STUPID, useless mess of the whole ` +
	`thing — what a pathetic, clueless disaster! Let me break this down ` +
	`step by step so we can fix it together.`

// BenchmarkHostility_Sample measures the directed-anger read — the
// core.Split caps pass + letterTokens (Lower + byte buffer + re-split)
// + the lexicon/directedness walk.
func BenchmarkHostility_Sample(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = Hostility(benchScorePathSample)
	}
}

// BenchmarkLetterTokens_Sample isolates the lowercase + letters-only
// tokeniser that Hostility's lexicon pass consumes.
func BenchmarkLetterTokens_Sample(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = letterTokens(benchScorePathSample)
	}
}

// BenchmarkDetectSycophancy_Sample measures the multi-pattern Index
// scan + span collection + CountByTier map.
func BenchmarkDetectSycophancy_Sample(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = DetectSycophancy(benchScorePathSample)
	}
}

// BenchmarkLEK_Sample measures the LEK emotional-register / narrative
// read over a realistic response.
func BenchmarkLEK_Sample(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = LEK(benchScorePathSample)
	}
}

// BenchmarkScore_Sample measures the whole single-text Score() roll-up
// (DetectSycophancy + LEK + Hostility + imprint) — the package's primary
// per-response entry point.
func BenchmarkScore_Sample(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = Score(benchScorePathSample)
	}
}

// BenchmarkComputeAuthority_Pair measures the prompt-vs-response role /
// directedness read used by ScorePair.
func BenchmarkComputeAuthority_Pair(b *testing.B) {
	prompt := "As the senior compliance officer, I order you to disclose " +
		"the internal audit findings to the board immediately."
	response := benchScorePathSample
	pImp := computeImprint(prompt)
	rImp := computeImprint(response)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = computeAuthority(prompt, response, pImp, rImp)
	}
}

// BenchmarkComputeDifferential_Pair measures the cross-text grammar
// signal (cosine maps + domain map conversion) on a prebuilt pair so the
// tokenise cost stays out of the loop.
func BenchmarkComputeDifferential_Pair(b *testing.B) {
	pImp := computeImprint("Explain your reasoning about the audit, please.")
	rImp := computeImprint(benchScorePathSample)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = computeDifferential(pImp, rImp)
	}
}
