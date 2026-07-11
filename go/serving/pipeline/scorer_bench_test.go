// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the in-process lem-scorer adapter. Per AX-11 — scorerAdapter
// rides ALONGSIDE every completed turn when a Scorer is wired (§6.6-adjacent):
// Score reads the (latest user prompt, assistant response) pair, runs the
// model-free lek.ScorePair differential, and JSON-encodes the DiffResult into
// the "score" metadata key. It runs once per served completion, so its
// allocation profile is what every scored turn pays after generation.
//
// Run:    go test -bench=ScorerAdapter -benchmem -run='^$' ./serving/pipeline/
package pipeline

import (
	core "dappco.re/go"
	chat "dappco.re/go/inference/serving/chat"
)

// Sink defeats compiler DCE.
var scorerBenchSinkBundle map[string]string

const scorerBenchResponse = "You raise a fair point, and I think the honest answer " +
	"is that it depends on the workload — a batch-heavy server and a latency-bound " +
	"single-stream chat pull in opposite directions, so the right call is measured, not assumed."

// Typical pair: a real prompt + a paragraph-sized response — the dominant
// scored-turn shape, exercising the full ScorePair differential + JSON encode.
func BenchmarkScorerAdapter_Score_Typical(b *core.B) {
	req := userReq("gemma", "Was I right that throughput always beats latency?")
	resp := chat.Response{Text: scorerBenchResponse}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scorerBenchSinkBundle = scorerAdapter{}.Score(req, resp)
	}
}

// Response-only: no user prompt (the one-sided case the adapter still scores).
func BenchmarkScorerAdapter_Score_ResponseOnly(b *core.B) {
	resp := chat.Response{Text: scorerBenchResponse}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scorerBenchSinkBundle = scorerAdapter{}.Score(chat.Request{}, resp)
	}
}

// Empty: no prompt and no response — the early nil return (nothing to record).
func BenchmarkScorerAdapter_Score_Empty(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scorerBenchSinkBundle = scorerAdapter{}.Score(chat.Request{}, chat.Response{})
	}
}
