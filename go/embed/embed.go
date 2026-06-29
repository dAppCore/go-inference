// SPDX-Licence-Identifier: EUPL-1.2

// Package embed shapes embedding and rerank requests for the serving
// surface (RFC §6.8). It is pure request-shaping over injected interfaces —
// the real backends are go-mlx's bert / bert_rerank on-device or a remote
// provider; this package never does the model maths, only the batching,
// top-k selection, and a cosine helper for callers that rerank locally by
// embedding.
//
//	// Embed a large corpus in fixed-size batches through one loaded model:
//	vecs, err := embed.EmbedBatched(ctx, embedder, docs, 32)
//
//	// Rerank and keep the best three:
//	top, err := embed.RerankTopK(ctx, reranker, query, docs, 3)
package embed

import (
	"cmp"
	"context"
	"math"
	"slices"

	core "dappco.re/go"
)

// Embedder turns texts into vectors. Implemented by go-mlx's bert model on
// device or a remote embedder; faked in tests. One call embeds one batch — the
// returned slice is aligned to the input (vector i is texts[i]).
//
//	vecs, err := embedder.Embed(ctx, []string{"hello", "world"})
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// Reranker scores documents against a query. Implemented by go-mlx's
// bert_rerank on device or a remote reranker; faked in tests. Each Scored
// carries the document's original index so the caller can map back after a
// reorder.
//
//	scored, err := reranker.Rerank(ctx, "how do I reset?", docs)
type Reranker interface {
	Rerank(ctx context.Context, query string, docs []string) ([]Scored, error)
}

// Scored is one reranked document — its position in the original docs slice
// and the reranker's relevance score (higher = more relevant).
type Scored struct {
	Index int     `json:"index"`
	Score float64 `json:"score"`
}

// EmbedBatched splits texts into batches of batchSize, embeds each batch
// through embedder, and concatenates the vectors back in input order. Use it
// to push a large corpus through a single loaded model without exceeding a
// backend's per-call limit; on the local path this maps onto go-mlx's
// BatchGenerate (RFC §6.3). A batch error surfaces immediately — no partial
// result is returned.
//
//	vecs, err := embed.EmbedBatched(ctx, embedder, docs, 32)
//	// len(vecs) == len(docs); vecs[i] is the embedding of docs[i].
func EmbedBatched(ctx context.Context, embedder Embedder, texts []string, batchSize int) ([][]float32, error) {
	if embedder == nil {
		return nil, core.E("embed", "embed batched: nil embedder", nil)
	}
	if batchSize <= 0 {
		return nil, core.E("embed", "embed batched: batch size must be positive", nil)
	}
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += batchSize {
		end := start + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		vecs, err := embedder.Embed(ctx, texts[start:end])
		if err != nil {
			return nil, core.E("embed", "embed batched: batch failed", err)
		}
		out = append(out, vecs...)
	}
	return out, nil
}

// RerankTopK reranks docs against query and returns the top k by score
// descending. Ties hold the original input order (stable). k larger than the
// document count clamps to all docs; k <= 0 or empty docs return an empty
// slice without consulting the reranker for a top slice.
//
//	top, err := embed.RerankTopK(ctx, reranker, "reset password", docs, 5)
//	for _, s := range top { use(docs[s.Index], s.Score) }
func RerankTopK(ctx context.Context, reranker Reranker, query string, docs []string, k int) ([]Scored, error) {
	if reranker == nil {
		return nil, core.E("embed", "rerank top-k: nil reranker", nil)
	}
	if k <= 0 || len(docs) == 0 {
		return []Scored{}, nil
	}

	scored, err := reranker.Rerank(ctx, query, docs)
	if err != nil {
		return nil, core.E("embed", "rerank top-k: rerank failed", err)
	}

	// Highest score first; equal scores keep their original document order.
	slices.SortStableFunc(scored, func(a, b Scored) int {
		if a.Score != b.Score {
			if a.Score > b.Score {
				return -1
			}
			return 1
		}
		return cmp.Compare(a.Index, b.Index)
	})

	if k > len(scored) {
		k = len(scored)
	}
	return scored[:k], nil
}

// Cosine is the cosine similarity of two equal-length vectors — 1.0 for the
// same direction, 0 for orthogonal, -1.0 for opposite. It guards against a
// length mismatch and a zero-magnitude vector by returning 0 rather than
// panicking or producing NaN, so a caller can rerank locally by embedding
// without pre-checking every pair.
//
//	score := embed.Cosine(queryVec, docVec)
func Cosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		av, bv := float64(a[i]), float64(b[i])
		dot += av * bv
		normA += av * av
		normB += bv * bv
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
