// SPDX-Licence-Identifier: EUPL-1.2

// Batch generation: BatchGenerate runs full autoregressive decoding for every
// prompt in one call (parallel, unlike Classify's single-forward-pass fast
// path — see pkg/classify). The Result carries one inference.BatchResult per
// prompt, each with its own Tokens and a per-prompt Err so one bad prompt
// (cancelled context, OOM) doesn't fail the whole batch.
//
//	go run ./pkg/batch -model ~/models/gemma-4-e2b-it-4bit
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"dappco.re/go/inference"
	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "model snapshot directory (config.json + *.safetensors)")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a model snapshot directory")
		os.Exit(2)
	}

	prompts := []string{
		"Name one duty of a lighthouse keeper.",
		"Name one hazard of the open sea.",
		"Name one tool a navigator relies on.",
	}

	r := inference.LoadModel(*model)
	if !r.OK {
		fmt.Fprintln(os.Stderr, "load:", r.Value)
		os.Exit(1)
	}
	m := r.Value.(inference.TextModel)
	defer m.Close()

	br := m.BatchGenerate(context.Background(), prompts, inference.WithMaxTokens(64))
	if !br.OK {
		fmt.Fprintln(os.Stderr, "batch generate:", br.Value)
		os.Exit(1)
	}
	results := br.Value.([]inference.BatchResult)

	for i, res := range results {
		if res.Err != nil {
			fmt.Printf("%d: %s -> error: %v\n", i, prompts[i], res.Err)
			continue
		}
		var text strings.Builder
		for _, tok := range res.Tokens {
			text.WriteString(tok.Text)
		}
		fmt.Printf("%d: %s -> %s\n", i, prompts[i], strings.TrimSpace(text.String()))
	}
}
