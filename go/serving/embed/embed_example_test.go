// SPDX-Licence-Identifier: EUPL-1.2

package embed

import (
	"context"

	core "dappco.re/go"
)

// ExampleEmbedBatched demonstrates embedding a corpus in fixed-size batches
// through one loaded model, concatenating the vectors back in input order.
func ExampleEmbedBatched() {
	e := &fakeEmbedder{}
	vecs, err := EmbedBatched(context.Background(), e, []string{"a", "bb", "ccc"}, 2)
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(len(vecs))
	core.Println(len(e.batches))
	// Output:
	// 3
	// 2
}

// ExampleRerankTopK demonstrates reranking documents against a query and
// keeping only the best-scoring k, highest first.
func ExampleRerankTopK() {
	r := &fakeReranker{scores: map[string]float64{
		"alpha": 0.1,
		"bravo": 0.9,
	}}
	top, err := RerankTopK(context.Background(), r, "q", []string{"alpha", "bravo"}, 1)
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(top[0].Index)
	core.Println(top[0].Score)
	// Output:
	// 1
	// 0.9
}

// ExampleCosine demonstrates the cosine similarity of two vectors.
func ExampleCosine() {
	core.Println(Cosine([]float32{1, 0}, []float32{1, 0}))
	// Output:
	// 1
}
