// SPDX-Licence-Identifier: EUPL-1.2

package bert_test

import (
	"context"
	"os"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/arch/bert"
)

// exampleModel loads the same tiny synthetic on-disk snapshot model_test.go's triplets
// exercise — a real (if minimal) BertModel, not a stub — sized so Example output is stable.
// Uses buildSyntheticSnapshot directly (no *testing.T — Example functions don't get one):
// bert.Load fully buffers config/vocab/weights, so the snapshot directory is safe to remove
// immediately once Load returns.
func exampleModel() *bert.Model {
	dir, err := os.MkdirTemp("", "bert-example-*")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)
	if err := buildSyntheticSnapshot(dir, 20); err != nil {
		panic(err)
	}
	m, err := bert.Load(dir)
	if err != nil {
		panic(err)
	}
	return m
}

func ExampleLoad() {
	m := exampleModel()
	core.Println(m.Config().HiddenSize)
	// Output: 8
}

func ExampleModel_Config() {
	m := exampleModel()
	core.Println(m.Config().HiddenSize, m.Config().NumHiddenLayers)
	// Output: 8 1
}

func ExampleModel_Pooling() {
	m := exampleModel()
	core.Println(m.Pooling())
	// Output: mean
}

func ExampleModel_Normalises() {
	m := exampleModel()
	core.Println(m.Normalises())
	// Output: true
}

func ExampleModel_Embed() {
	m := exampleModel()
	res, err := m.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"the quick fox"}})
	core.Println(err == nil, len(res.Vectors), len(res.Vectors[0]))
	// Output: true 1 8
}

func ExampleModel_Rerank() {
	m := exampleModel()
	res, err := m.Rerank(context.Background(), inference.RerankRequest{
		Query:     "reset password",
		Documents: []string{"the quick fox", "reset password"},
	})
	core.Println(err == nil, len(res.Results))
	// Output: true 2
}
