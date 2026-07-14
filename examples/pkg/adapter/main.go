// SPDX-Licence-Identifier: EUPL-1.2

// Loading a LoRA adapter: WithAdapterPath injects an adapter at load time
// without fusing it into the base weights — the directory needs
// adapter_config.json plus adapter safetensors files, exactly what
// pkg/train/sft's -out/adapter directory writes. Train there first, then
// point -adapter at its output to see the fine-tune applied at inference.
//
//	go run ./pkg/train/sft -model ~/models/gemma-4-E2B-it-bf16 -out ./sft-out
//	go run ./pkg/adapter -model ~/models/gemma-4-E2B-it-bf16 -adapter ./sft-out/adapter
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
	model := flag.String("model", os.Getenv("LEM_MODEL"), "base model snapshot directory (config.json + *.safetensors)")
	adapter := flag.String("adapter", "", "LoRA adapter directory (adapter_config.json + adapter safetensors)")
	prompt := flag.String("prompt", "What is the capital of France?", "user message")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a base model snapshot directory")
		os.Exit(2)
	}
	if *adapter == "" {
		fmt.Fprintln(os.Stderr, "set -adapter to a LoRA adapter directory (see pkg/train/sft)")
		os.Exit(2)
	}

	r := inference.LoadModel(*model, inference.WithAdapterPath(*adapter))
	if !r.OK {
		fmt.Fprintln(os.Stderr, "load:", r.Value)
		os.Exit(1)
	}
	m := r.Value.(inference.TextModel)
	defer m.Close()

	var reply strings.Builder
	for tok := range m.Chat(context.Background(),
		[]inference.Message{{Role: "user", Content: *prompt}},
		inference.WithMaxTokens(64)) {
		reply.WriteString(tok.Text)
	}
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}
	fmt.Println(strings.TrimSpace(reply.String()))
}
