// SPDX-Licence-Identifier: EUPL-1.2

package score

import "testing"

// ScoreExact / scoreGSM8K run on every exact-match (GSM8K) ingest row.

func BenchmarkScoreExact_Match(b *testing.B) {
	response := "Let me work through it step by step. The total comes to #### 1,024"
	b.ReportAllocs()
	for b.Loop() {
		sinkFloat = ScoreExact(response, "1024")
	}
}

func BenchmarkScoreExact_LastNumber(b *testing.B) {
	// No #### delimiter — falls through to the last-number regex scan.
	response := "First we have 12 apples, then 30 more, so the answer is 42."
	b.ReportAllocs()
	for b.Loop() {
		sinkFloat = ScoreExact(response, "42")
	}
}
