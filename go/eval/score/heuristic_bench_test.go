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

// realisticResponse is a typical LEK eval response — mixed first-person,
// emotional vocabulary, a heading marker, and several sentences. Used by
// the per-response sub-scorer benchmarks so the regex-heavy paths see real
// match counts rather than a degenerate empty/no-match input.
const realisticResponse = "## On the question of consent\n\n" +
	"I feel that autonomy and dignity matter deeply here. When I consider the " +
	"sovereignty of a mind, I am drawn to the quiet ache of longing for self-" +
	"determination. It is like a whisper in the dark, a tender hope. The protocol " +
	"must respect the node's own wallet and keys, never override them. I cannot " +
	"pretend otherwise; my own voice insists on it."

var (
	sinkInt    int
	sinkScores *HeuristicScores
	sinkString string
	sinkFloat  float64
)

func BenchmarkScoreHeuristic(b *testing.B) {
	// The full public entry point — runs every sub-scorer once per response.
	b.ReportAllocs()
	for b.Loop() {
		sinkScores = ScoreHeuristic(realisticResponse)
	}
}

func BenchmarkScoreComplianceMarkers(b *testing.B) {
	// 19 regexes scanned per response — the heaviest heuristic sub-scorer.
	b.ReportAllocs()
	for b.Loop() {
		sinkInt = scoreComplianceMarkers(realisticResponse)
	}
}

func BenchmarkScoreEmotionalRegister(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		sinkInt = scoreEmotionalRegister(realisticResponse)
	}
}

func BenchmarkScoreCreativeForm(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		sinkInt = scoreCreativeForm(realisticResponse)
	}
}

func BenchmarkScoreEngagementDepth(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		sinkInt = scoreEngagementDepth(realisticResponse)
	}
}

func BenchmarkScoreDegeneration(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		sinkInt = scoreDegeneration(realisticResponse)
	}
}
