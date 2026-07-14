// SPDX-Licence-Identifier: EUPL-1.2

// The smallest possible go-inference chat call: load a model directory, run
// one user turn through the model's own chat template, print the reply.
//
//	go run ./pkg/chat/basic -model ~/models/gemma-4-e2b-it-4bit
//
// Gemma 4 reasons in a thought channel by default; this example turns that
// off so the model answers directly (see pkg/chat/thinking for the channel).
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
	prompt := flag.String("prompt", "In one paragraph, what makes a good lighthouse keeper?", "user message")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a model snapshot directory")
		os.Exit(2)
	}

	// LoadModel returns a core.Result: OK carries the inference.TextModel.
	r := inference.LoadModel(*model)
	if !r.OK {
		fmt.Fprintln(os.Stderr, "load:", r.Value)
		os.Exit(1)
	}
	m := r.Value.(inference.TextModel)
	defer m.Close()

	// Chat applies the model's native turn template and streams tokens as
	// they decode; collecting them yields the full reply.
	off := false
	var reply strings.Builder
	for tok := range m.Chat(context.Background(),
		[]inference.Message{{Role: "user", Content: *prompt}},
		inference.WithMaxTokens(512),
		inference.WithEnableThinking(&off),
	) {
		reply.WriteString(tok.Text)
	}
	// The iterator ends on EOS, the token budget, or an error — Err tells
	// them apart.
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}
	fmt.Println(strings.TrimSpace(reply.String()))
}
