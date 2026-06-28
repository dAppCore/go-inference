package classify

import (
	"context"
	"testing"

	"dappco.re/go"
	"dappco.re/go/inference"
)

// Package-level sinks keep the compiler from eliminating benchmarked work.
var (
	benchStringSink string
	benchBoolSink   bool
	benchResultSink core.Result
	benchStatsSink  *ClassifyStats
	benchErrSink    error
)

// fixedTechModel classifies every prompt as "technical" — a realistic
// single-token classifier stand-in with no inference cost of its own.
func fixedTechModel() *mockModel {
	return &mockModel{
		classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i := range prompts {
				results[i] = inference.ClassifyResult{Token: inference.Token{Text: "technical"}}
			}
			return results, nil
		},
	}
}

func BenchmarkMapTokenToDomain(b *testing.B) {
	// Lowercase fragments are the common case: the model emits "technical",
	// "tech", "cre", etc. Mixed-case exercises core.Lower's allocation path.
	cases := []string{"technical", "Creative"}
	for _, tok := range cases {
		b.Run(tok, func(b *testing.B) {
			b.ReportAllocs()
			var s string
			for i := 0; i < b.N; i++ {
				s = mapTokenToDomain(tok)
			}
			benchStringSink = s
		})
	}
}

func BenchmarkClassifyCorpus(b *testing.B) {
	// 16 realistic JSONL records → two default-size (8) batches per run.
	var sb core.Builder
	for i := 0; i < 16; i++ {
		sb.WriteString(`{"seed_id":"`)
		sb.WriteString(core.Sprintf("%d", i))
		sb.WriteString(`","domain":"general","prompt":"Delete the file and rebuild the project"}` + "\n")
	}
	input := sb.String()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stats, err := ClassifyCorpus(context.Background(), fixedTechModel(),
			core.NewReader(input), core.NewBuffer())
		benchStatsSink = stats
		benchErrSink = err
	}
}
