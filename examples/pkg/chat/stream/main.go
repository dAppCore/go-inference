// SPDX-Licence-Identifier: EUPL-1.2

// Token streaming: print each token the moment it decodes — the shape a
// terminal UI or SSE endpoint wants. The Chat iterator IS the stream; there
// is no separate streaming API to opt into.
//
//	go run ./pkg/chat/stream -model ~/models/gemma-4-e2b-it-4bit
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
	prompt := flag.String("prompt", "Tell me a short story about a lighthouse keeper.", "user message")
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

	// WithTemperature(0.7) samples; 0 would be deterministic greedy.
	// Breaking out of the range loop cancels generation cleanly — the
	// context does the same from another goroutine.
	off := false
	for tok := range m.Chat(context.Background(),
		[]inference.Message{{Role: "user", Content: *prompt}},
		inference.WithMaxTokens(512),
		inference.WithTemperature(0.7),
		inference.WithEnableThinking(&off),
	) {
		fmt.Print(tok.Text) // stdout is unbuffered in Go — tokens appear live
	}
	fmt.Println()
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}

	// Every completed call leaves throughput counters behind.
	fmt.Fprintf(os.Stderr, "\n[%.1f tok/s decode]\n", m.Metrics().DecodeTokensPerSec)
}
