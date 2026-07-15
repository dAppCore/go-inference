// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"math"
	"sort"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func rocmReferenceMeanPoolEmbedding(tokens [][]float32, normalize bool) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, core.E("rocm.Embedding.ReferenceMeanPool", "token embeddings are required", nil)
	}
	dim := len(tokens[0])
	if dim == 0 {
		return nil, core.E("rocm.Embedding.ReferenceMeanPool", "embedding dimension must be positive", nil)
	}
	out := make([]float32, dim)
	for i, token := range tokens {
		if len(token) != dim {
			return nil, core.E("rocm.Embedding.ReferenceMeanPool", core.Sprintf("token %d dimension %d does not match %d", i, len(token), dim), nil)
		}
		for j, value := range token {
			out[j] += value
		}
	}
	scale := float32(1) / float32(len(tokens))
	for i := range out {
		out[i] *= scale
	}
	if normalize {
		return rocmReferenceL2Normalize(out)
	}
	return out, nil
}

func rocmReferenceCosineSimilarity(left, right []float32) (float64, error) {
	if len(left) == 0 || len(left) != len(right) {
		return 0, core.E("rocm.Rerank.ReferenceCosine", "vectors must be non-empty and equal length", nil)
	}
	dot := float64(0)
	leftNorm := float64(0)
	rightNorm := float64(0)
	for i := range left {
		l := float64(left[i])
		r := float64(right[i])
		dot += l * r
		leftNorm += l * l
		rightNorm += r * r
	}
	if leftNorm == 0 || rightNorm == 0 {
		return 0, core.E("rocm.Rerank.ReferenceCosine", "zero vector cannot be scored", nil)
	}
	return dot / (math.Sqrt(leftNorm) * math.Sqrt(rightNorm)), nil
}

func rocmReferenceRerank(query []float32, documents [][]float32, texts []string, topN int) ([]inference.RerankScore, error) {
	if len(documents) == 0 {
		return nil, core.E("rocm.Rerank.Reference", "documents are required", nil)
	}
	if len(texts) != 0 && len(texts) != len(documents) {
		return nil, core.E("rocm.Rerank.Reference", "document text count must match document vectors", nil)
	}
	results := make([]inference.RerankScore, len(documents))
	for i, document := range documents {
		score, err := rocmReferenceCosineSimilarity(query, document)
		if err != nil {
			return nil, err
		}
		results[i] = inference.RerankScore{Index: i, Score: score}
		if len(texts) > 0 {
			results[i].Text = texts[i]
		}
	}
	sort.SliceStable(results, func(i, j int) bool {
		if results[i].Score == results[j].Score {
			return results[i].Index < results[j].Index
		}
		return results[i].Score > results[j].Score
	})
	if topN > 0 && topN < len(results) {
		results = results[:topN]
	}
	return results, nil
}

func rocmReferenceL2Normalize(vector []float32) ([]float32, error) {
	norm := float64(0)
	for _, value := range vector {
		norm += float64(value * value)
	}
	if norm == 0 {
		return nil, core.E("rocm.Embedding.ReferenceNormalize", "zero vector cannot be normalized", nil)
	}
	out := make([]float32, len(vector))
	scale := float32(1 / math.Sqrt(norm))
	for i, value := range vector {
		out[i] = value * scale
	}
	return out, nil
}
