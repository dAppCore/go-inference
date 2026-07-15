// SPDX-Licence-Identifier: EUPL-1.2

// Thinking budget: WithThinkingBudget caps how many tokens a reasoning model
// (thinking ON, the Gemma 4 default) may spend inside its thought channel
// before the backend forces the channel closed and moves to the visible
// answer. GenerateMetrics.ThinkingBudgetForced reports whether that forced
// close actually happened on this generation.
//
//	go run ./pkg/chat/budget -model ~/models/gemma-4-e2b-it-4bit
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
	prompt := flag.String("prompt", "A farmer has 17 sheep. All but 9 run away. How many are left?", "user message")
	budget := flag.Int("budget", 32, "maximum tokens spent in the thought channel")
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

	// Thinking defaults ON — no WithEnableThinking needed. WithThinkingBudget
	// alone caps the reasoning spend; the raw stream still carries whatever
	// channel markers the model emits (see pkg/chat/thinking for splitting
	// them out).
	for tok := range m.Chat(context.Background(),
		[]inference.Message{{Role: "user", Content: *prompt}},
		inference.WithMaxTokens(2048),
		inference.WithThinkingBudget(*budget),
	) {
		fmt.Print(tok.Text)
	}
	fmt.Println()
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}

	if m.Metrics().ThinkingBudgetForced {
		fmt.Fprintf(os.Stderr, "[thinking budget of %d forced the channel closed]\n", *budget)
	} else {
		fmt.Fprintf(os.Stderr, "[reasoning finished within the %d token budget on its own]\n", *budget)
	}
}
