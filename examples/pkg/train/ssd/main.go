// SPDX-Licence-Identifier: EUPL-1.2

// Self-distillation sampling (SSD): sample the FROZEN base model over a
// prompt set and capture every self-output at birth into a trace. SSD does
// NOT train — the trace (ssd-captures.jsonl) is the deliverable; a later
// curation step picks rows from it and a separate SFT run (pkg/train/sft)
// teaches them back.
//
//	go run ./pkg/train/ssd -model ~/models/gemma-4-E2B-it-bf16 -out ./ssd-out
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"dappco.re/go/inference/train"

	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "model snapshot directory (config.json + *.safetensors)")
	out := flag.String("out", "ssd-out", "directory for the prompt set + captured trace")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a model snapshot directory")
		os.Exit(2)
	}

	// A prompt set is a JSONL file of {"prompt": ...} rows (the dataset
	// loader also accepts text/messages/instruction shapes). Real runs point
	// -data at a curated set; this example writes a tiny one to stay
	// self-contained.
	if err := os.MkdirAll(*out, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	data := filepath.Join(*out, "prompts.jsonl")
	rows := []string{
		`{"prompt": "Explain why the sky is blue in two sentences."}`,
		`{"prompt": "Write a haiku about a lighthouse in a storm."}`,
		`{"prompt": "What does a git rebase do, in plain words?"}`,
	}
	if err := os.WriteFile(data, []byte(strings.Join(rows, "\n")+"\n"), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// RunSSDCommand = load + sample + capture. Score is nil, so this runs
	// capture-only (birth-scoring is a driver-supplied hook); the trace lands
	// beside the checkpoints as ssd-captures.jsonl with {step,prompt,text}
	// rows — every return, pre-filter.
	err := train.RunSSDCommand(context.Background(), train.SSDCommandConfig{
		ModelPath:       *model,
		DataPath:        data,
		CheckpointDir:   *out,
		SampleMaxTokens: 128,
		SampleTemp:      0.9, // sampling variety is the point — never 0 here
		SampleTopK:      64,
		SampleTopP:      0.95,
		Out:             os.Stdout,
		Log:             os.Stderr,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "ssd:", err)
		os.Exit(1)
	}
}
