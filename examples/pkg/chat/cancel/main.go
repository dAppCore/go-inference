// SPDX-Licence-Identifier: EUPL-1.2

// Cancelling generation: two different mechanisms, easy to conflate.
// context.WithCancel stops the ENGINE — the backend observes ctx.Done() and
// tears down mid-generation. Breaking the range loop stops the CONSUMER —
// the iterator simply isn't asked for another token, and the engine notices
// on its next step and winds down; no context involved.
//
//	go run ./pkg/chat/cancel -model ~/models/gemma-4-e2b-it-4bit
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"dappco.re/go/inference"
	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "model snapshot directory (config.json + *.safetensors)")
	prompt := flag.String("prompt", "Tell me a very long story about a lighthouse keeper.", "user message")
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

	// Mechanism 1: cancel the context from another goroutine after ~500ms.
	// The engine itself observes ctx.Done() and stops generating.
	fmt.Println("--- context cancel after 500ms ---")
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(500 * time.Millisecond)
		cancel()
	}()
	n := 0
	for tok := range m.Chat(ctx, msgs, inference.WithMaxTokens(4096), inference.WithEnableThinking(&off)) {
		fmt.Print(tok.Text)
		n++
	}
	fmt.Printf("\n(%d tokens before ctx cancel fired)\n", n)
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value) // expected: context cancelled
	}

	// Mechanism 2: break the range loop after N tokens. No context involved —
	// the iterator simply isn't asked for another value.
	fmt.Println("--- range break after 10 tokens ---")
	n = 0
	for tok := range m.Chat(context.Background(), msgs, inference.WithMaxTokens(4096), inference.WithEnableThinking(&off)) {
		fmt.Print(tok.Text)
		n++
		if n >= 10 {
			break
		}
	}
	fmt.Printf("\n(stopped after %d tokens by breaking the loop)\n", n)
}
