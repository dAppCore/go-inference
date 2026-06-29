// SPDX-Licence-Identifier: EUPL-1.2

// Allocation contracts for the embed request-shaping surface (AX-11).
// Each public function gets a perf bench with realistic embedding/rerank
// fixtures. The fake backends here are deliberately allocation-free per call
// (pre-built vectors returned as windows; a reused Scored scratch) so each
// bench line isolates the embed function's OWN buffering from the backend's
// vector maths — the thing this package can actually control.
//
// Run:    go test -bench=. -benchmem -run='^$' ./embed/

package embed_test

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/embed"
)

// benchDim is a realistic embedding width — BERT-base and many
// sentence-transformer models emit 768-d float32 vectors.
const benchDim = 768

// benchEmbedder returns pre-built corpus vectors as consecutive windows, so
// Embed itself allocates nothing. EmbedBatched copies only the per-vector
// slice headers into its own buffer, so the bench measures that buffer alone.
type benchEmbedder struct {
	vecs [][]float32
	off  int
}

func (e *benchEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := e.vecs[e.off : e.off+len(texts)]
	e.off += len(texts)
	return out, nil
}

func newBenchEmbedder(n int) *benchEmbedder {
	vecs := make([][]float32, n)
	backing := make([]float32, n*benchDim)
	for i := range vecs {
		v := backing[i*benchDim : (i+1)*benchDim : (i+1)*benchDim]
		for j := range v {
			v[j] = float32(i*benchDim+j) * 0.001
		}
		vecs[i] = v
	}
	return &benchEmbedder{vecs: vecs}
}

// benchReranker refills a reused Scored scratch each call (allocation-free),
// so RerankTopK's bench line shows only the sort machinery's allocations.
type benchReranker struct {
	scratch []embed.Scored
	scores  []float64
}

func (r *benchReranker) Rerank(_ context.Context, _ string, docs []string) ([]embed.Scored, error) {
	out := r.scratch[:len(docs)]
	for i := range docs {
		out[i] = embed.Scored{Index: i, Score: r.scores[i]}
	}
	return out, nil
}

func newBenchReranker(n int) *benchReranker {
	scores := make([]float64, n)
	for i := range scores {
		// Pseudo-random in [0,1) with deliberate ties to exercise the
		// stable-sort Index tiebreak.
		scores[i] = float64((i*2654435761)%100) / 100.0
	}
	return &benchReranker{scratch: make([]embed.Scored, n), scores: scores}
}

func makeBenchTexts(n int) []string {
	texts := make([]string, n)
	for i := range texts {
		// Content is irrelevant — the lean fakes key off length/index, not text.
		texts[i] = "document"
	}
	return texts
}

// Package sinks defeat dead-code elimination.
var (
	sinkVecs   [][]float32
	sinkScored []embed.Scored
	sinkFloat  float64
)

// --- EmbedBatched ---

func BenchmarkEmbedBatched(b *core.B) {
	const n = 256
	ctx := context.Background()
	e := newBenchEmbedder(n)
	texts := makeBenchTexts(n)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.off = 0
		v, err := embed.EmbedBatched(ctx, e, texts, 32)
		if err != nil {
			b.Fatal(err)
		}
		sinkVecs = v
	}
}

// --- RerankTopK ---

func BenchmarkRerankTopK(b *core.B) {
	const n = 100
	ctx := context.Background()
	r := newBenchReranker(n)
	docs := makeBenchTexts(n)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s, err := embed.RerankTopK(ctx, r, "query", docs, 10)
		if err != nil {
			b.Fatal(err)
		}
		sinkScored = s
	}
}

// --- Cosine ---

func BenchmarkCosine(b *core.B) {
	a := make([]float32, benchDim)
	c := make([]float32, benchDim)
	for i := range a {
		a[i] = float32(i) * 0.001
		c[i] = float32(benchDim-i) * 0.001
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkFloat = embed.Cosine(a, c)
	}
}
