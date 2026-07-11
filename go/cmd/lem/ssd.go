// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference/train"
)

// runSSDCommand parses the ssd flags and hands them to train.RunSSDCommand.
// Thin: flag parsing + one library call + exit mapping. All self-distillation
// business logic (load, sample the frozen base, capture the trace, stop) lives
// in dappco.re/go/inference/train.
func runSSDCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("ssd"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelPath := fs.String("model", "", "frozen base model path to self-distil (required)")
	dataPath := fs.String("data", "", "prompt JSONL — {\"messages\":[…]} or {\"prompt\":…} per line; only prompts are read, responses are self-generated (required)")
	kernelPath := fs.String("kernel", "", "file holding the LEK-2 kernel prefix — rides every generation as KV state, never enters the captured rows (#97)")
	sampleMaxTokens := fs.Int("sample-max-tokens", 2048, "tokens per self-generated sample (gemma4 thinks first — small budgets truncate mid-thought into empty samples)")
	sampleTemp := fs.Float64("sample-temp", 0.7, "sampling temperature (must be non-unit ≠ 1.0 — diversity is the point)")
	sampleTopK := fs.Int("sample-top-k", 64, "sampling top-k")
	sampleTopP := fs.Float64("sample-top-p", 0.95, "sampling top-p")
	sampleMinP := fs.Float64("sample-min-p", 0, "sampling min-p")
	repPenalty := fs.Float64("rep-penalty", 1.0, "repetition penalty over self-samples")
	filterShortest := fs.Float64("filter-shortest", 10, "drop the shortest N%% of self-samples before the trace (0 keeps all)")
	scoreSamples := fs.Bool("score-samples", false, "score every self-sample at birth with the LEK scorer (writes birth-scores alongside the captured trace)")
	checkpointDir := fs.String("checkpoint-dir", "", "output dir for the scored trace — ssd-captures.jsonl")
	contextLen := fs.Int("context", 0, "model context override; 0 uses the model default")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s ssd --model <base> --data <prompts.jsonl> [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Self-distillation sampling (no-correct-answer): sample the FROZEN base over\n")
		core.WriteString(stderr, "the prompts, capture each self-output at birth, and STOP at the trace.\n")
		core.WriteString(stderr, "Nothing is taught — no reference answer, no verifier, no training. The lab\n")
		core.WriteString(stderr, "refines the trace into an SFT artifact; a separate `sft` run trains on it.\n")
		core.WriteString(stderr, "--kernel rides generation as KV state but never enters the captured rows (#97).\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		fs.VisitAll(func(f *flag.Flag) {
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
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

	err := train.RunSSDCommand(ctx, train.SSDCommandConfig{
		ModelPath:         *modelPath,
		DataPath:          *dataPath,
		KernelPath:        *kernelPath,
		CheckpointDir:     *checkpointDir,
		ContextLen:        *contextLen,
		SampleMaxTokens:   *sampleMaxTokens,
		SampleTemp:        *sampleTemp,
		SampleTopK:        *sampleTopK,
		SampleTopP:        *sampleTopP,
		SampleMinP:        *sampleMinP,
		RepetitionPenalty: *repPenalty,
		FilterShortest:    *filterShortest,
		ScoreSamples:      *scoreSamples,
		Score:             lekScoreFunc(),
		Out:               stdout,
		Log:               stderr,
	})
	if err != nil {
		core.Print(stderr, "%s ssd: %v", cliName(), err)
		return 1
	}
	return 0
}
