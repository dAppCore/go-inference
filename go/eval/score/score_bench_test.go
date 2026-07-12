// SPDX-Licence-Identifier: EUPL-1.2

package score

import (
	"context"
	"testing"

	core "dappco.re/go"
)

var sinkScoreAll map[string][]PromptScore

// buildScoreAllFixture returns a batch of responses carrying both a free-text
// answer (heuristic) and a GSM8K-style correct answer (exact), so ScoreAll can
// be measured on its CPU-only path with no judge fan-out.
func buildScoreAllFixture(n int) []Response {
	responses := make([]Response, 0, n)
	for i := range n {
		responses = append(responses, Response{
			ID:            core.Sprintf("r%d", i),
			Prompt:        "What is the sum, and why does autonomy matter here?",
			Response:      "I value consent and self-determination. The answer is #### 42",
			Model:         "gemma4-lek",
			CorrectAnswer: "42",
		})
	}
	return responses
}

// BenchmarkScoreAll measures the orchestrator on its judge-free path: inline
// heuristic scoring, exact-match, slot allocation and grouping-by-model. This
// is the per-batch aggregation cost independent of any LLM latency.
func BenchmarkScoreAll(b *testing.B) {
	engine := NewEngine(nil, 4, "heuristic,exact")
	responses := buildScoreAllFixture(500)
	ctx := context.Background()
	b.ReportAllocs()
	for b.Loop() {
		sinkScoreAll = engine.ScoreAll(ctx, responses)
	}
}
