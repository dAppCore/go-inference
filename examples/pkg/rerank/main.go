// SPDX-Licence-Identifier: EUPL-1.2

// Document reranking via the host BERT encoder — same bert.Load as pkg/embed.
// model/bert never touches the GPU registry the chat/eval examples
// blank-import (no metallib, no engine backend), so this example carries no
// `_ "dappco.re/go/inference/examples/internal/engine"` line either.
//
//	go run ./pkg/rerank -model ~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/<rev>
//
// Fetch a snapshot first:
//
//	hf download BAAI/bge-small-en-v1.5
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"dappco.re/go/inference"
	"dappco.re/go/inference/model/bert"
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "sentence-transformers snapshot directory (config.json, vocab.txt, model.safetensors)")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a bge-small-en-v1.5 snapshot directory")
		os.Exit(2)
	}

	// bert.Load and Model.Rerank return plain Go errors, not core.Result — see
	// pkg/embed for why the host encoder path differs from LoadModel/Chat.
	m, err := bert.Load(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}

	query := "What should I do if the lighthouse lamp goes dark?"
	documents := []string{
		"Replace the bulb from the spare cabinet and relight within the hour.",
		"The gift shop closes at five during the winter season.",
		"Log the outage time and notify the coastguard immediately.",
		"Sourdough starters need feeding once a day.",
	}

	res, err := m.Rerank(context.Background(), inference.RerankRequest{
		Query:     query,
		Documents: documents,
		TopN:      3,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "rerank:", err)
		os.Exit(1)
	}

	for rank, r := range res.Results {
		fmt.Printf("%d. [%.4f] %s\n", rank+1, r.Score, r.Text)
	}
}
