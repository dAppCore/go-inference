// SPDX-Licence-Identifier: EUPL-1.2

package embed

import (
	"context"
	"slices"

	core "dappco.re/go"
)

// fakeEmbedder records every batch it was handed and returns a deterministic
// one-dimensional vector per input text (its rune length as a float). Lets the
// batching logic be exercised without a live go-mlx bert model.
//
//	e := &fakeEmbedder{}
//	vecs, _ := EmbedBatched(ctx, e, []string{"a", "bb"}, 1)
type fakeEmbedder struct {
	batches [][]string // every batch, in call order — proves order + split
	err     error      // non-nil → every Embed call fails (a batch error)
	failOn  string     // non-empty → fail only the batch containing this text
}

func (e *fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	// Copy the slice — the caller's batch sub-slice must not alias our record.
	batch := make([]string, len(texts))
	copy(batch, texts)
	e.batches = append(e.batches, batch)

	if e.err != nil {
		return nil, e.err
	}
	if e.failOn != "" {
		if slices.Contains(texts, e.failOn) {
			return nil, core.E("embed", "fake batch failure", nil)
		}
	}

	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = []float32{float32(len([]rune(t)))}
	}
	return out, nil
}

// fakeReranker scores each doc by a fixed lookup, defaulting to 0. Index is the
// doc's position in the input — RerankTopK must sort on score, not index.
type fakeReranker struct {
	scores map[string]float64 // doc text → score
	err    error
}

func (r *fakeReranker) Rerank(_ context.Context, _ string, docs []string) ([]Scored, error) {
	if r.err != nil {
		return nil, r.err
	}
	out := make([]Scored, len(docs))
	for i, d := range docs {
		out[i] = Scored{Index: i, Score: r.scores[d]}
	}
	return out, nil
}

// --- EmbedBatched ---

func TestEmbed_EmbedBatched_Good(t *core.T) {
	e := &fakeEmbedder{}
	texts := []string{"a", "bb", "ccc", "dddd", "eeeee"}

	got, err := EmbedBatched(context.Background(), e, texts, 2)
	core.AssertNoError(t, err)

	// One vector per input, in input order — vector value is rune length.
	core.AssertLen(t, got, 5)
	want := [][]float32{{1}, {2}, {3}, {4}, {5}}
	core.AssertEqual(t, want, got)

	// 5 texts at batchSize 2 → batches of [2,2,1], in order.
	core.AssertLen(t, e.batches, 3)
	core.AssertEqual(t, []string{"a", "bb"}, e.batches[0])
	core.AssertEqual(t, []string{"ccc", "dddd"}, e.batches[1])
	core.AssertEqual(t, []string{"eeeee"}, e.batches[2])
}

func TestEmbed_EmbedBatched_Bad(t *core.T) {
	// A batch error surfaces — the whole call fails, no partial vectors.
	e := &fakeEmbedder{failOn: "ccc"}
	texts := []string{"a", "bb", "ccc", "dddd"}

	got, err := EmbedBatched(context.Background(), e, texts, 2)
	core.AssertError(t, err)
	core.AssertNil(t, got)

	// nil embedder is a programming error, not a runtime one — guarded.
	_, err = EmbedBatched(context.Background(), nil, texts, 2)
	core.AssertError(t, err)

	// A non-positive batch size is rejected rather than looping forever.
	_, err = EmbedBatched(context.Background(), e, texts, 0)
	core.AssertError(t, err)
}

func TestEmbed_EmbedBatched_Ugly(t *core.T) {
	// Batch size larger than the input → one batch, all texts, order kept.
	e := &fakeEmbedder{}
	texts := []string{"x", "yy", "zzz"}
	got, err := EmbedBatched(context.Background(), e, texts, 100)
	core.AssertNoError(t, err)
	core.AssertEqual(t, [][]float32{{1}, {2}, {3}}, got)
	core.AssertLen(t, e.batches, 1)
	core.AssertEqual(t, texts, e.batches[0])

	// Empty input → empty result, embedder never called.
	empty := &fakeEmbedder{}
	out, err := EmbedBatched(context.Background(), empty, nil, 4)
	core.AssertNoError(t, err)
	core.AssertLen(t, out, 0)
	core.AssertLen(t, empty.batches, 0)
}

// --- RerankTopK ---

func TestEmbed_RerankTopK_Good(t *core.T) {
	r := &fakeReranker{scores: map[string]float64{
		"alpha":   0.10,
		"bravo":   0.90,
		"charlie": 0.50,
		"delta":   0.70,
	}}
	docs := []string{"alpha", "bravo", "charlie", "delta"}

	got, err := RerankTopK(context.Background(), r, "q", docs, 2)
	core.AssertNoError(t, err)
	core.AssertLen(t, got, 2)

	// Top-2 by score descending: bravo(0.90) then delta(0.70).
	core.AssertEqual(t, 1, got[0].Index) // bravo is docs[1]
	core.AssertInDelta(t, 0.90, got[0].Score, 1e-9)
	core.AssertEqual(t, 3, got[1].Index) // delta is docs[3]
	core.AssertInDelta(t, 0.70, got[1].Score, 1e-9)
}

func TestEmbed_RerankTopK_Bad(t *core.T) {
	r := &fakeReranker{err: core.E("embed", "reranker down", nil)}
	got, err := RerankTopK(context.Background(), r, "q", []string{"a", "b"}, 1)
	core.AssertError(t, err)
	core.AssertNil(t, got)

	// nil reranker → guarded error, no panic.
	_, err = RerankTopK(context.Background(), nil, "q", []string{"a"}, 1)
	core.AssertError(t, err)
}

func TestEmbed_RerankTopK_Ugly(t *core.T) {
	r := &fakeReranker{scores: map[string]float64{"a": 0.5, "b": 0.5, "c": 0.5}}
	docs := []string{"a", "b", "c"}

	// k larger than the doc count → clamp to all docs, no out-of-range.
	got, err := RerankTopK(context.Background(), r, "q", docs, 99)
	core.AssertNoError(t, err)
	core.AssertLen(t, got, 3)

	// Ties (all 0.5) keep original input order — stable sort by Index.
	core.AssertEqual(t, 0, got[0].Index)
	core.AssertEqual(t, 1, got[1].Index)
	core.AssertEqual(t, 2, got[2].Index)

	// k <= 0 → empty result (asked for nothing), not an error.
	none, err := RerankTopK(context.Background(), r, "q", docs, 0)
	core.AssertNoError(t, err)
	core.AssertLen(t, none, 0)

	// Empty docs → empty result, reranker never consulted for a top slice.
	empty, err := RerankTopK(context.Background(), r, "q", nil, 3)
	core.AssertNoError(t, err)
	core.AssertLen(t, empty, 0)
}

// --- Cosine ---

func TestEmbed_Cosine_Good(t *core.T) {
	// Identical direction → 1.0.
	core.AssertInDelta(t, 1.0, Cosine([]float32{1, 0}, []float32{2, 0}), 1e-9)

	// 45° between (1,0) and (1,1) → cos = 1/√2.
	core.AssertInDelta(t, 0.7071067811865476, Cosine([]float32{1, 0}, []float32{1, 1}), 1e-9)
}

func TestEmbed_Cosine_Bad(t *core.T) {
	// Length mismatch is undefined → 0 (the guard), never a panic.
	core.AssertEqual(t, 0.0, Cosine([]float32{1, 2, 3}, []float32{1, 2}))

	// Two empty vectors → 0, no divide-by-zero.
	core.AssertEqual(t, 0.0, Cosine(nil, nil))
}

func TestEmbed_Cosine_Ugly(t *core.T) {
	// Orthogonal vectors → 0.
	core.AssertInDelta(t, 0.0, Cosine([]float32{1, 0}, []float32{0, 1}), 1e-9)

	// Opposite direction → -1.
	core.AssertInDelta(t, -1.0, Cosine([]float32{1, 0}, []float32{-1, 0}), 1e-9)

	// A zero vector against a real one → 0 (guarded magnitude), not NaN.
	core.AssertEqual(t, 0.0, Cosine([]float32{0, 0}, []float32{1, 1}))
}
