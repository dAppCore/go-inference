// SPDX-Licence-Identifier: EUPL-1.2

// LoRA supervised fine-tuning: teach the model {"messages": [...]} assistant
// turns from a JSONL set. Train on a bf16 snapshot (the quantised 4-bit
// snapshots are for serving); the run writes an adapter you apply at load
// time rather than merging into the base weights.
//
//	go run ./pkg/train/sft -model ~/models/gemma-4-E2B-it-bf16 -out ./sft-out
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
	model := flag.String("model", os.Getenv("LEM_MODEL"), "bf16 model snapshot directory (config.json + *.safetensors)")
	out := flag.String("out", "sft-out", "directory for the training set, checkpoints and adapter")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a bf16 model snapshot directory")
		os.Exit(2)
	}

	// Training rows are OpenAI-style {"messages": [...]}: the assistant turns
	// are the teaching targets. Real runs point -data at a curated artifact
	// (for example one refined from an SSD trace — see pkg/train/ssd).
	if err := os.MkdirAll(*out, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	data := filepath.Join(*out, "train.jsonl")
	rows := []string{
		`{"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}]}`,
		`{"messages": [{"role": "user", "content": "Name the keeper's first duty at dusk."}, {"role": "assistant", "content": "Light the lamp and log the time."}]}`,
		`{"messages": [{"role": "user", "content": "17 times 4?"}, {"role": "assistant", "content": "68"}]}`,
	}
	if err := os.WriteFile(data, []byte(strings.Join(rows, "\n")+"\n"), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// RunSFTCommand = load model + tokeniser, run the LoRA loop, save the
	// adapter. Rank 8 / alpha 16 / lr 1e-4 is the proven small-model recipe;
	// one epoch over three rows just demonstrates the plumbing.
	err := train.RunSFTCommand(context.Background(), train.SFTCommandConfig{
		ModelPath:     *model,
		DataPath:      data,
		CheckpointDir: *out,
		SavePath:      filepath.Join(*out, "adapter"),
		Rank:          8,
		Alpha:         16,
		LearningRate:  1e-4,
		Epochs:        1,
		BatchSize:     1,
		GradAccum:     1,
		MaxSeqLen:     256,
		Out:           os.Stdout,
		Log:           os.Stderr,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "sft:", err)
		os.Exit(1)
	}
}
