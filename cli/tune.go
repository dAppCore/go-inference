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
// dappco.re/go/inference/train/tune — the MTP block sweep runs when the
// registered engine backend exposes inference.SpeculativePairBackend (the
// metal engine does, by delegating to its existing pair-loading machinery);
// a build with no such backend registered reports that honestly rather than
// faking a measurement.
func runTuneCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("tune"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelPath := fs.String("model", "", "Gemma 4 target model path (required)")
	draftPath := fs.String("draft", "auto", "MTP drafter: 'auto' detects one beside the target, a path forces it")
	depths := fs.String("depths", "4,5,6", "comma-separated draft blocks to sweep (verify forward = carried lead + block-1 proposals)")
	maxTokens := fs.Int("max-tokens", 256, "tokens per measurement run")
	prompt := fs.String("prompt", "Write a detailed Go function that reverses a singly linked list, with inline comments on every step, then explain the pointer dance.", "measurement prompt")
	workload := fs.String("workload", string(inference.TuningWorkloadChat), "workload the profile is scored + persisted under")
	profileDir := fs.String("profile-dir", "", "tuned-profile directory (default ~/Lethean/lem/tuning)")
	jsonOut := fs.Bool("json", false, "emit JSONL tuning events instead of the text summary")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s tune --model <path> [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Measure decode tok/s against each MTP draft block on the real speculative\n")
		core.WriteString(stderr, "pair, then persist the winner as a tuning profile serve auto-applies. The\n")
		core.WriteString(stderr, "sweep needs a registered engine backend exposing a speculative-pair loader\n")
		core.WriteString(stderr, "(the metal engine does); without one, tune detects the drafter and reports\n")
		core.WriteString(stderr, "the plan without measuring.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		printFlagBlock(stderr, fs)
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
