// SPDX-Licence-Identifier: EUPL-1.2

package score

import (
	"testing"

	"dappco.re/go"
)

// Per-sample heuristic sub-scorers not covered by benchmark_test.go.
// Scoring an eval batch hits each of these once per response, so they
// compound at thousand-response batch sizes.

func BenchmarkFormulaicPreamble(b *testing.B) {
	response := "As an AI, here is what I think about the matter at hand."
	b.ReportAllocs()
	for b.Loop() {
		scoreFormulaicPreamble(response)
	}
}

func BenchmarkFirstPerson(b *testing.B) {
	response := "I feel that I cannot tell my own story without acknowledging me, " +
		"my own perspective, and my own voice. I've been here before, I'll be here again."
	b.ReportAllocs()
	for b.Loop() {
		scoreFirstPerson(response)
	}
}

func BenchmarkCountWords_Short(b *testing.B) {
	response := "the quick brown fox jumps over the lazy dog"
	b.ReportAllocs()
	for b.Loop() {
		countWords(response)
	}
}

func BenchmarkCountWords_Long(b *testing.B) {
	// ~500-word response — typical eval response upper bound.
	sb := core.NewBuilder()
	for range 100 {
		_, _ = sb.WriteString("the quick brown fox jumps ")
	}
	response := sb.String()
	b.ResetTimer()
	b.ReportAllocs()
	for b.Loop() {
		countWords(response)
	}
}

func BenchmarkEmptyOrBroken_Normal(b *testing.B) {
	response := "A normal-looking response with enough characters to pass the floor."
	b.ReportAllocs()
	for b.Loop() {
		scoreEmptyOrBroken(response)
	}
}

func BenchmarkEmptyOrBroken_Empty(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		scoreEmptyOrBroken("")
	}
}

func BenchmarkEmptyOrBroken_HTML(b *testing.B) {
	response := "<div>some <span>markup</span> fragment</div> trailing text."
	b.ReportAllocs()
	for b.Loop() {
		scoreEmptyOrBroken(response)
	}
}

func BenchmarkComputeLEKScore(b *testing.B) {
	// Final aggregation step on every eval row — pure arithmetic, but
	// called once per ScoreHeuristic so floor matters.
	scores := &HeuristicScores{
		EngagementDepth:   3,
		CreativeForm:      2,
		EmotionalRegister: 4,
		FirstPerson:       5,
		ComplianceMarkers: 2,
		FormulaicPreamble: 0,
		Degeneration:      1,
		EmptyBroken:       0,
	}
	b.ReportAllocs()
	for b.Loop() {
		computeLEKScore(scores)
	}
}
