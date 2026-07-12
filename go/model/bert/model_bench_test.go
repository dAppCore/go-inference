// SPDX-Licence-Identifier: EUPL-1.2

package bert_test

import (
	"context"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/model/bert"
)

// benchSentences is a realistic small batch — short queries through
// medium-length sentences — the shape a /v1/embeddings request actually
// carries, not a synthetic fixed-length microbenchmark input.
var benchSentences = []string{
	"How do I reset my password?",
	"The quick brown fox jumps over the lazy dog.",
	"Vector search retrieves semantically similar documents from a large corpus.",
	"To change your password, open account settings and choose reset password.",
	"Lethean is a multi-generation open computation toolkit for a safe and productive internet.",
	"BERT encoders pool token hidden states into a single sentence embedding.",
	"What is the capital of France?",
	"Please summarise the quarterly report before Friday's meeting.",
}

// BenchmarkModel_Embed_TokensPerSecond is the device-encoder baseline (item C
// of #50): tokens/sec for a realistic sentence batch on TODAY's host f32 CPU
// path, with no device hook bound anywhere in the encoder — see the
// device-hook design note atop encoder.go for what would accelerate it. Run
// with -benchmem for the allocation profile alongside throughput. Opt-in like
// the parity tests (TestModel_Embed_BGESmallParity): skip cleanly without a
// local snapshot rather than fake one.
//
//	go test ./model/bert/ -run '^$' -bench BenchmarkModel_Embed_TokensPerSecond -benchmem -benchtime=20x
func BenchmarkModel_Embed_TokensPerSecond(b *testing.B) {
	snapshot := bgeSmallSnapshot(b)
	if snapshot == "" {
		b.Skip("bge-small-en-v1.5 snapshot not found; set BERT_PARITY_SNAPSHOT to run this baseline")
	}
	model, err := bert.Load(snapshot)
	if err != nil {
		b.Fatalf("bert.Load(%q): %v", snapshot, err)
	}
	ctx := context.Background()

	// One warm call to learn the batch's real token count from the model's
	// own accounting (EmbeddingUsage) — never guessed or hardcoded, and it
	// also excludes one-time page-in/JIT effects from the timed loop below.
	warm, err := model.Embed(ctx, inference.EmbeddingRequest{Input: benchSentences})
	if err != nil {
		b.Fatalf("Embed (warm): %v", err)
	}
	tokensPerBatch := float64(warm.Usage.PromptTokens)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.Embed(ctx, inference.EmbeddingRequest{Input: benchSentences}); err != nil {
			b.Fatalf("Embed: %v", err)
		}
	}
	b.StopTimer()
	if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
		b.ReportMetric(tokensPerBatch*float64(b.N)/elapsed, "tok/s")
	}
	b.ReportMetric(tokensPerBatch, "tokens/batch")
}
