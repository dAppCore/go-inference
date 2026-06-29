// SPDX-Licence-Identifier: EUPL-1.2

package experiments_test

import (
	"testing"

	"dappco.re/go/inference/experiments"
)

// benchEvalExample is the (example, output) fixture the evaluator benchmarks
// score — a reference answer the output can hit or miss.
func benchEvalExample() experiments.Example {
	return experiments.Example{
		ID:        "ex-1",
		DatasetID: "ds",
		Reference: map[string]any{"answer": "context-dependent"},
	}
}

func BenchmarkExactMatch_Eval(b *testing.B) {
	ev := experiments.ExactMatch()
	ex := benchEvalExample()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchKey, benchScore, benchComment = ev.Eval(ex, "context-dependent")
	}
}

func BenchmarkContains_Eval(b *testing.B) {
	ev := experiments.Contains()
	ex := benchEvalExample()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchKey, benchScore, benchComment = ev.Eval(ex, "the answer is context-dependent really")
	}
}

func BenchmarkRegexp_Eval(b *testing.B) {
	r := experiments.Regexp(`\bcontext-\w+\b`)
	if !r.OK {
		b.Fatalf("compile: %v", r.Error())
	}
	ev := r.Value.(experiments.Evaluator)
	ex := benchEvalExample()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchKey, benchScore, benchComment = ev.Eval(ex, "the answer is context-dependent really")
	}
}

func BenchmarkLengthScore_Eval(b *testing.B) {
	r := experiments.LengthScore(17)
	if !r.OK {
		b.Fatalf("construct: %v", r.Error())
	}
	ev := r.Value.(experiments.Evaluator)
	ex := benchEvalExample()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchKey, benchScore, benchComment = ev.Eval(ex, "context-dependent")
	}
}
