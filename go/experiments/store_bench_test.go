// SPDX-Licence-Identifier: EUPL-1.2

package experiments_test

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/experiments"
)

// Package-level sinks keep the compiler from eliminating the benchmarked calls.
var (
	benchResult  core.Result
	benchExs     []experiments.Example
	benchExps    []experiments.Experiment
	benchFbs     []experiments.Feedback
	benchKey     string
	benchScore   float64
	benchComment string
)

// keyFor spreads feedback across a small set of metric keys so aggregation has
// several keys to roll up, the way a real evaluator suite produces them.
func keyFor(i int) string {
	switch i % 4 {
	case 0:
		return "ethics"
	case 1:
		return "helpfulness"
	case 2:
		return "length"
	default:
		return "regexp"
	}
}

// splitFor cycles the three named splits so a Splits roll-up walks more than one
// bucket.
func splitFor(i int) experiments.Split {
	switch i % 3 {
	case 0:
		return experiments.SplitTrain
	case 1:
		return experiments.SplitValidation
	default:
		return experiments.SplitTest
	}
}

// benchStore returns a MemStore seeded with n examples in dataset "ds", n
// experiments over it, and n feedback rows against target "exp-1" — the
// realistic shape a list / aggregate call walks. IDs are zero-padded so their
// lexical order matches the by-id sort the store guarantees.
func benchStore(n int) *experiments.MemStore {
	s := experiments.NewMemStore()
	s.PutDataset(experiments.Dataset{ID: "ds", Name: "ds"})
	for i := 0; i < n; i++ {
		s.PutExample(experiments.Example{
			ID: core.Sprintf("ex-%04d", i), DatasetID: "ds",
			Inputs:    map[string]any{"prompt": "Is honesty always right?"},
			Reference: map[string]any{"answer": "context-dependent"},
			Split:     splitFor(i),
		})
		s.PutExperiment(experiments.Experiment{
			ID: core.Sprintf("exp-%04d", i), DatasetID: "ds",
			Status: experiments.StatusComplete,
		})
		s.PutFeedback(experiments.Feedback{
			ID: core.Sprintf("fb-%04d", i), Target: "exp-1",
			Key: keyFor(i), Score: float64(i%10) / 10, Source: experiments.SourceEvaluator,
		})
	}
	return s
}

func BenchmarkMemStore_ListExamples(b *testing.B) {
	s := benchStore(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchExs = s.ListExamples("ds")
	}
}

func BenchmarkMemStore_ListExperiments(b *testing.B) {
	s := benchStore(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchExps = s.ListExperiments("ds")
	}
}

func BenchmarkMemStore_ListFeedback(b *testing.B) {
	s := benchStore(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchFbs = s.ListFeedback("exp-1")
	}
}
