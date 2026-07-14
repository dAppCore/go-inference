// SPDX-Licence-Identifier: EUPL-1.2

// Sampling control: Temperature/TopK/TopP/MinP shape the distribution a
// token is drawn from; WithSeed pins the draw itself. Same seed plus the
// same sampling settings on the same prompt reproduces the same completion —
// this example runs it twice and compares.
//
//	go run ./pkg/chat/sampling -model ~/models/gemma-4-e2b-it-4bit
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
	prompt := flag.String("prompt", "Name a colour, one word only.", "user message")
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
	opts := []inference.GenerateOption{
		inference.WithMaxTokens(16),
		inference.WithTemperature(0.9),
		inference.WithTopK(40),
		inference.WithTopP(0.9),
		inference.WithMinP(0.05),
		inference.WithSeed(42),
		inference.WithEnableThinking(&off),
	}

	run := func() string {
		var out strings.Builder
		for tok := range m.Chat(context.Background(), []inference.Message{{Role: "user", Content: *prompt}}, opts...) {
			out.WriteString(tok.Text)
		}
		if er := m.Err(); !er.OK {
			fmt.Fprintln(os.Stderr, "generate:", er.Value)
			os.Exit(1)
		}
		return strings.TrimSpace(out.String())
	}

	first := run()
	second := run()
	fmt.Println("run 1:", first)
	fmt.Println("run 2:", second)
	if first == second {
		fmt.Println("MATCH")
	} else {
		fmt.Println("DIFFER")
	}
}
