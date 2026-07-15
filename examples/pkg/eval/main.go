// SPDX-Licence-Identifier: EUPL-1.2

// A minimal eval harness on the Classify fast path: batched prefill-only
// inference — each prompt gets ONE forward pass and the sampled token at the
// last position is the model's one-token answer. That makes label tasks
// (sentiment, topic, yes/no) cheap enough to run inside a test suite.
//
//	go run ./pkg/eval -model ~/models/gemma-4-e2b-it-4bit
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

	cases := []struct{ review, want string }{
		{"The lighthouse tour was magical — our guide knew every stone.", "positive"},
		{"Waited an hour in the rain and the lamp room was shut.", "negative"},
		{"Best coastal walk of the year; the kids loved the foghorn demo.", "positive"},
		{"Overpriced, overcrowded, and the gift shop ate my card twice.", "negative"},
	}

	// One prompt per case, framed so the answer is the very next token.
	prompts := make([]string, len(cases))
	for i, c := range cases {
		prompts[i] = "Review: " + c.review + "\nAnswer with exactly one word, positive or negative.\nSentiment:"
	}

	r := inference.LoadModel(*model)
	if !r.OK {
		fmt.Fprintln(os.Stderr, "load:", r.Value)
		os.Exit(1)
	}
	m := r.Value.(inference.TextModel)
	defer m.Close()

	// Classify runs the whole batch in one call; add WithLogits() to also get
	// the raw vocab logits per prompt (calibration, margin analysis).
	cr := m.Classify(context.Background(), prompts)
	if !cr.OK {
		fmt.Fprintln(os.Stderr, "classify:", cr.Value)
		os.Exit(1)
	}
	results := cr.Value.([]inference.ClassifyResult)

	correct := 0
	for i, res := range results {
		got := strings.TrimSpace(strings.ToLower(res.Token.Text))
		ok := strings.HasPrefix(cases[i].want, got) && got != ""
		if ok {
			correct++
		}
		fmt.Printf("%-8s want %-8s %v\n", got, cases[i].want, ok)
	}
	fmt.Printf("accuracy %d/%d\n", correct, len(cases))
}
