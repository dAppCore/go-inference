// SPDX-Licence-Identifier: EUPL-1.2

// Multi-turn chat: go-inference is stateless per call — there is no
// server-side session. Each Chat call resends the FULL message history; the
// model only "remembers" earlier turns because they are still in the slice
// you pass it.
//
//	go run ./pkg/chat/multiturn -model ~/models/gemma-4-e2b-it-4bit
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

	r := inference.LoadModel(*model)
	if !r.OK {
		fmt.Fprintln(os.Stderr, "load:", r.Value)
		os.Exit(1)
	}
	m := r.Value.(inference.TextModel)
	defer m.Close()

	// The whole history so far: system framing, one finished exchange, and a
	// fresh user turn. Nothing is cached between calls — this slice IS the
	// model's memory.
	off := false
	history := []inference.Message{
		{Role: "system", Content: "You are a terse lighthouse keeper."},
		{Role: "user", Content: "What is the first thing you do at dusk?"},
		{Role: "assistant", Content: "Light the lamp."},
		{Role: "user", Content: "And if the lamp fails?"},
	}
	var reply strings.Builder
	for tok := range m.Chat(context.Background(), history, inference.WithMaxTokens(256), inference.WithEnableThinking(&off)) {
		reply.WriteString(tok.Text)
	}
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}
	answer := strings.TrimSpace(reply.String())
	fmt.Println(answer)

	// Continuing the conversation means appending the assistant's reply and
	// the next user turn, then resending the whole thing again.
	history = append(history,
		inference.Message{Role: "assistant", Content: answer},
		inference.Message{Role: "user", Content: "How quickly can you fix it?"},
	)
	reply.Reset()
	for tok := range m.Chat(context.Background(), history, inference.WithMaxTokens(256), inference.WithEnableThinking(&off)) {
		reply.WriteString(tok.Text)
	}
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}
	fmt.Println(strings.TrimSpace(reply.String()))
}
