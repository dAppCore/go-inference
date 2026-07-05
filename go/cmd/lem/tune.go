// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/train/tune"
)

// runTuneCommand parses the tune flags and hands them to tune.RunTune. Thin:
// flag parsing + one library call + exit mapping. The tune business logic
// (drafter detection, block sweep, profile persistence) lives in
// dappco.re/go/inference/tune — the MTP block sweep itself is blocked on a
// speculative-pair engine seam go-inference does not yet expose (RunTune reports
// that honestly rather than faking a measurement).
func runTuneCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("tune"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelPath := fs.String("model", "", "Gemma 4 target model path (required)")
	draftPath := fs.String("draft", "auto", "MTP drafter: 'auto' detects one beside the target, a path forces it")
	depths := fs.String("depths", "4,5,6", "comma-separated draft blocks to sweep (verify forward = carried lead + block-1 proposals)")
	maxTokens := fs.Int("max-tokens", 256, "tokens per measurement run")
	prompt := fs.String("prompt", "Write a detailed Go function that reverses a singly linked list, with inline comments on every step, then explain the pointer dance.", "measurement prompt")
	workload := fs.String("workload", string(inference.TuningWorkloadChat), "workload the profile is scored + persisted under")
	profileDir := fs.String("profile-dir", "", "tuned-profile directory (default ~/Lethean/data/tuning)")
	jsonOut := fs.Bool("json", false, "emit JSONL tuning events instead of the text summary")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s tune --model <path> [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Measure plain AR decode against each MTP draft block on the real model,\n")
		core.WriteString(stderr, "then persist the winner as a tuning profile serve auto-applies. The block\n")
		core.WriteString(stderr, "sweep needs a speculative-pair loader no registered go-inference engine\n")
		core.WriteString(stderr, "exposes yet, so tune currently detects the drafter and reports the plan\n")
		core.WriteString(stderr, "without measuring (it lights up when the engine seam lands).\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		core.WriteString(stderr, core.Sprintf("%s tune: unexpected positional arguments\n", cliName()))
		fs.Usage()
		return 2
	}
	if core.Trim(*modelPath) == "" {
		core.WriteString(stderr, core.Sprintf("%s tune: --model is required\n", cliName()))
		fs.Usage()
		return 2
	}

	err := tune.RunTune(ctx, tune.Config{
		ModelPath:  *modelPath,
		DraftPath:  *draftPath,
		Depths:     *depths,
		MaxTokens:  *maxTokens,
		Prompt:     *prompt,
		Workload:   *workload,
		ProfileDir: *profileDir,
		JSON:       *jsonOut,
		Out:        stdout,
		Log:        stderr,
	})
	if err != nil {
		core.Print(stderr, "%s tune: %v", cliName(), err)
		return 1
	}
	return 0
}
