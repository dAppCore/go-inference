// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference/train"
)

// runSFTCommand parses the sft flags and hands them to train.RunSFTCommand.
// Thin: flag parsing + one library call + exit mapping. All SFT business logic
// (load, tokenise, the engine-trainer LoRA loop, checkpoint, eval, save) lives
// in dappco.re/go/inference/train.
func runSFTCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("sft"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelPath := fs.String("model", "", "model path to fine-tune (required)")
	dataPath := fs.String("data", "", "training JSONL — {\"messages\":[{role,content}…]} per line (required)")
	validPath := fs.String("valid", "", "validation JSONL; derives eval probes from its first user turns when --eval-prompts is absent")
	evalPromptsPath := fs.String("eval-prompts", "", "file of eval probes, one per line (overrides --valid derivation)")
	evalEvery := fs.Int("eval-every", 25, "run the eval probes every N optimiser steps (0 disables eval)")
	evalMaxTokens := fs.Int("eval-max-tokens", 200, "tokens per eval generation")
	evalProbes := fs.Int("eval-probes", 4, "probes derived from --valid when --eval-prompts is absent")
	evalTemp := fs.Float64("eval-temp", 0, "eval sampling temperature (0 = greedy)")
	scoreCascade := fs.Bool("score-cascade", false, "score every eval pass with the LEK scorer and pick the best checkpoint by windowed composite")
	scoreWindow := fs.Int("score-window", 3, "eval passes per windowed composite")
	rank := fs.Int("rank", 16, "LoRA rank")
	alpha := fs.Float64("alpha", 32, "LoRA alpha")
	lr := fs.Float64("lr", 1e-4, "AdamW learning rate")
	epochs := fs.Int("epochs", 1, "training epochs")
	batch := fs.Int("batch", 1, "batch size")
	gradAccum := fs.Int("grad-accum", 4, "gradient accumulation steps")
	maxSeqLen := fs.Int("max-seq", 1024, "max sequence length (longer samples truncate)")
	packing := fs.Bool("packing", false, "sequence packing (no effect on the head-LoRA trainer; noted honestly)")
	checkpointDir := fs.String("checkpoint-dir", "", "checkpoint directory")
	checkpointEvery := fs.Int("checkpoint-every", 50, "save a checkpoint every N optimiser steps (0 disables)")
	savePath := fs.String("save", "", "final adapter path (default <checkpoint-dir>/adapter when a dir is set)")
	resumePath := fs.String("resume", "", "resume from a saved adapter checkpoint")
	merge := fs.Bool("merge", false, "merge the adapter into the model weights after training (unsupported on head-LoRA; noted honestly)")
	contextLen := fs.Int("context", 0, "model context override; 0 uses the model default")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s sft --model <path> --data <train.jsonl> [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Native LoRA SFT through the engine-neutral trainer seam: the loaded engine\n")
		core.WriteString(stderr, "opens a head-LoRA trainer, the loop steps it over the training set, checkpoints\n")
		core.WriteString(stderr, "and evaluates on a fixed probe set, and saves a reloadable adapter package\n")
		core.WriteString(stderr, "(apply it at load with serve/generate --adapter).\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		fs.VisitAll(func(f *flag.Flag) {
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Examples:\n")
		core.WriteString(stderr, core.Sprintf("  %s sft --model ~/models/gemma-4-E2B-it-bf16 \\\n", name))
		core.WriteString(stderr, "      --data train.jsonl --valid valid.jsonl \\\n")
		core.WriteString(stderr, "      --rank 16 --epochs 2 --checkpoint-dir ~/Lethean/lem/sft/run1\n")
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if *modelPath == "" || *dataPath == "" {
		fs.Usage()
		return 2
	}

	err := train.RunSFTCommand(ctx, train.SFTCommandConfig{
		ModelPath:       *modelPath,
		DataPath:        *dataPath,
		ValidPath:       *validPath,
		EvalPromptsPath: *evalPromptsPath,
		CheckpointDir:   *checkpointDir,
		SavePath:        *savePath,
		ResumePath:      *resumePath,
		ContextLen:      *contextLen,
		Rank:            *rank,
		Alpha:           *alpha,
		LearningRate:    *lr,
		Epochs:          *epochs,
		BatchSize:       *batch,
		GradAccum:       *gradAccum,
		MaxSeqLen:       *maxSeqLen,
		Packing:         *packing,
		Merge:           *merge,
		EvalEvery:       *evalEvery,
		EvalMaxTokens:   *evalMaxTokens,
		EvalProbes:      *evalProbes,
		EvalTemp:        *evalTemp,
		CheckpointEvery: *checkpointEvery,
		ScoreCascade:    *scoreCascade,
		ScoreWindow:     *scoreWindow,
		Score:           lekScoreFunc(),
		Out:             stdout,
		Log:             stderr,
	})
	if err != nil {
		core.Print(stderr, "%s sft: %v", cliName(), err)
		return 1
	}
	return 0
}
