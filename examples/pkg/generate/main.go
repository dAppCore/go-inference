// SPDX-Licence-Identifier: EUPL-1.2

// Raw completion versus chat: Generate continues the prompt text verbatim —
// no turn markers, no system/user framing, just the model predicting what
// comes next. Chat (see pkg/chat/basic) wraps the same call in the model's
// native chat template before it ever reaches the engine. Point this at a
// base/pretrained snapshot for the clearest contrast; an instruction-tuned
// model will still continue raw text, just less fluently.
//
//	go run ./pkg/generate -model ~/models/gemma-4-e2b-4bit
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
	prompt := flag.String("prompt", "The capital of France is", "text to continue — no chat template applied")
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

	// Generate feeds *prompt straight to the tokeniser — the string printed
	// below is exactly what the model saw, with the reply appended after it.
	var reply strings.Builder
	for tok := range m.Generate(context.Background(), *prompt, inference.WithMaxTokens(32), inference.WithTemperature(0)) {
		reply.WriteString(tok.Text)
	}
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}
	fmt.Println(*prompt + reply.String())
}
