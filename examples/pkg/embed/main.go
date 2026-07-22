// SPDX-Licence-Identifier: EUPL-1.2

// Text embeddings via the host BERT encoder. model/bert is a pure-CPU
// float32 forward pass — it never touches the GPU registry the chat/eval
// examples blank-import (no metallib, no engine backend needed), so this
// example has no `_ "dappco.re/go/inference/examples/internal/engine"` line.
//
//	go run ./pkg/embed -model ~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/<rev>
//
// Fetch a snapshot first:
//
//	hf download BAAI/bge-small-en-v1.5
package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"

	"dappco.re/go/inference"
	"dappco.re/go/inference/model/arch/bert"
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "sentence-transformers snapshot directory (config.json, vocab.txt, model.safetensors)")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a bge-small-en-v1.5 snapshot directory")
		os.Exit(2)
	}

	// bert.Load and Model.Embed return plain Go errors, not core.Result — the
	// host encoder path never crosses the engine registry that LoadModel/Chat
	// report through.
	m, err := bert.Load(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}

	sentences := []string{
		"The lighthouse keeper trimmed the wick before nightfall.",
		"A keeper must watch the lamp through every storm.",
		"The bakery on the corner sells sourdough on Fridays.",
	}
	res, err := m.Embed(context.Background(), inference.EmbeddingRequest{Input: sentences})
	if err != nil {
		fmt.Fprintln(os.Stderr, "embed:", err)
		os.Exit(1)
	}

	fmt.Printf("%d vectors, %d dims each\n", len(res.Vectors), len(res.Vectors[0]))
	fmt.Printf("cosine(0,1) = %.4f  (both about lighthouse keeping)\n", cosine(res.Vectors[0], res.Vectors[1]))
	fmt.Printf("cosine(0,2) = %.4f  (unrelated topic)\n", cosine(res.Vectors[0], res.Vectors[2]))
}

// cosine is the standard vector similarity: the dot product over the product
// of magnitudes, in [-1, 1] for non-zero vectors.
func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
