// SPDX-Licence-Identifier: EUPL-1.2

// Stop control: WithStopTokens halts generation the instant a listed token ID
// is sampled, WithSuppressTokens removes an ID from the distribution
// entirely, and WithMinTokensBeforeStop delays stop-token matching until n
// tokens have been emitted. Token IDs are model-specific, so this example
// obtains one honestly — from a baseline generation's own Token.ID stream —
// rather than guessing a value.
//
//	go run ./pkg/chat/stop -model ~/models/gemma-4-e2b-it-4bit
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"dappco.re/go/inference"
	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "model snapshot directory (config.json + *.safetensors)")
	prompt := flag.String("prompt", "Count from one to five, one number per line.", "user message")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a model snapshot directory")
		os.Exit(2)
	}

	r := inference.LoadModel(*model)
	if !r.OK {
		fmt.Fprintln(os.Stderr, "load:", r.Value)
		os.Exit(1)
	}
	m := r.Value.(inference.TextModel)
	defer m.Close()

	off := false
	msgs := []inference.Message{{Role: "user", Content: *prompt}}
	// Greedy (temperature 0) so the captured token ID is reproducible run to run.
	base := []inference.GenerateOption{
		inference.WithMaxTokens(32),
		inference.WithTemperature(0),
		inference.WithEnableThinking(&off),
	}

	// Baseline: capture the first token's ID honestly from the stream.
	var first inference.Token
	n := 0
	for tok := range m.Chat(context.Background(), msgs, base...) {
		if n == 0 {
			first = tok
		}
		n++
	}
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}
	fmt.Printf("baseline first token: %q (id %d)\n", first.Text, first.ID)

	// WithSuppressTokens removes that ID from the distribution — the greedy
	// pick moves to whatever ranks second.
	var suppressed string
	for tok := range m.Chat(context.Background(), msgs, append(base, inference.WithSuppressTokens(first.ID))...) {
		suppressed += tok.Text
	}
	fmt.Printf("with id %d suppressed: %q\n", first.ID, suppressed)

	// WithStopTokens on that same ID halts generation the instant it would be
	// sampled — since it is the greedy first token, output is empty.
	var stopped string
	for tok := range m.Chat(context.Background(), msgs, append(base, inference.WithStopTokens(first.ID))...) {
		stopped += tok.Text
	}
	fmt.Printf("with id %d as a stop token: %q (empty: stop fires before any token is emitted)\n", first.ID, stopped)

	// WithMinTokensBeforeStop masks the same stop token until n tokens have
	// been emitted, giving the model a chance to say something first.
	var delayed string
	minOpts := append(base, inference.WithStopTokens(first.ID), inference.WithMinTokensBeforeStop(3))
	for tok := range m.Chat(context.Background(), msgs, minOpts...) {
		delayed += tok.Text
	}
	fmt.Printf("with min-tokens-before-stop 3: %q\n", delayed)
}
