// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/eval/bench"
)

// runBenchCommand parses the bench flags and drives bench.RunMatrix — the reproducible benchmark
// grid: models as DATA (a bench.json runs array, or positional refs), the timing discipline in
// eval/bench, this file only the composition root (it owns the engine loaders). The same verb + the
// same config reproduce the grid on every box the binary builds for, which is what makes numbers
// comparable across builds and machines (bench/gemma4.json is the locked house grid).
func runBenchCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("bench"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	configPath := fs.String("config", "", "bench.json matrix config (runs array); positional model refs are used instead when absent")
	tokens := fs.Int("tokens", 512, "tokens per timed generation (the N of tg-N)")
	warmup := fs.Int("warmup", 16, "untimed warmup generation's token budget")
	prompt := fs.String("prompt", "", "generation prompt (empty = the harness default)")
	draft := fs.String("draft", "", "MTP drafter ref for positional models (adds the mtp lane); explicit by design — no auto-detection")
	draftBlock := fs.Int("draft-block", 0, "MTP draft block; 0 = engine default")
	jsonOut := fs.String("json", "", "write the full MatrixReport JSON to this file")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s bench [flags] [model-ref ...]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Run the benchmark matrix: per model x lane (plain, mtp), one untimed warmup\n")
		core.WriteString(stderr, "then one timed greedy generation, reporting the engine's own decode tok/s.\n")
		core.WriteString(stderr, "Model refs are snapshot paths or HF cache repo names (org/name).\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		printFlagBlock(stderr, fs)
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Examples:\n")
		core.WriteString(stderr, core.Sprintf("  %s bench --config bench/gemma4.json --json bench-out.json\n", name))
		core.WriteString(stderr, "    # the locked gemma4 grid, table + machine-readable report\n")
		core.WriteString(stderr, core.Sprintf("  %s bench mlx-community/gemma-4-e2b-it-4bit --draft mlx-community/gemma-4-E2B-it-assistant-bf16\n", name))
		core.WriteString(stderr, "    # one model, plain + mtp lanes\n")
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	var cfg bench.MatrixConfig
	switch {
	case *configPath != "":
		read := core.ReadFile(*configPath)
		if !read.OK {
			core.Print(stderr, "%s bench: read --config %s: %s", cliName(), *configPath, read.Error())
			return 1
		}
		data := read.Bytes()
		if len(data) == 0 {
			core.Print(stderr, "%s bench: --config %s is empty", cliName(), *configPath)
			return 1
		}
		parsed, err := bench.LoadMatrixConfig(data)
		if err != nil {
			core.Print(stderr, "%s bench: %v", cliName(), err)
			return 1
		}
		cfg = parsed
	case fs.NArg() > 0:
		for _, ref := range fs.Args() {
			cfg.Runs = append(cfg.Runs, bench.MatrixRun{Model: ref, Draft: *draft, DraftBlock: *draftBlock})
		}
	default:
		core.WriteString(stderr, core.Sprintf("%s bench: pass --config <bench.json> or at least one model ref\n", cliName()))
		fs.Usage()
		return 2
	}
	// Flag overrides apply on top of the config's own values only when explicitly set.
	if *tokens != 512 || cfg.Tokens == 0 {
		cfg.Tokens = *tokens
	}
	if *warmup != 16 || cfg.Warmup == 0 {
		cfg.Warmup = *warmup
	}
	if *prompt != "" {
		cfg.Prompt = *prompt
	}

	report, err := bench.RunMatrix(ctx, cfg, benchMatrixLoad, stdout)
	if err != nil {
		core.Print(stderr, "%s bench: %v", cliName(), err)
		return 1
	}
	if *jsonOut != "" {
		data := core.JSONMarshalIndent(report, "", " ")
		if !data.OK {
			core.Print(stderr, "%s bench: encode report: %s", cliName(), data.Error())
			return 1
		}
		if w := core.WriteFile(*jsonOut, data.Bytes(), 0o644); !w.OK {
			core.Print(stderr, "%s bench: write --json %s: %s", cliName(), *jsonOut, w.Error())
			return 1
		}
		core.WriteString(stdout, core.Sprintf("report written to %s\n", *jsonOut))
	}
	for _, row := range report.Rows {
		if row.Err != "" {
			return 1 // the table already showed the holes; a partial grid exits non-zero
		}
	}
	return 0
}

// benchMatrixLoad is the composition root's loader: the plain lane loads through the registered
// backend (inference.LoadModel); the mtp lane loads target+drafter as one speculative TextModel via
// the engine's pair loader (nil off darwin/arm64 — the row then reports the missing lane honestly).
func benchMatrixLoad(_ context.Context, modelPath, draftPath string, draftBlock int) (bench.MatrixModel, error) {
	if draftPath != "" {
		if speculativeLoader == nil {
			return nil, core.NewError("this engine exposes no speculative path (mtp lane unavailable)")
		}
		m, err := speculativeLoader(modelPath, draftPath, draftBlock)
		if err != nil {
			return nil, err
		}
		return &benchTextModel{m: m}, nil
	}
	r := inference.LoadModel(modelPath)
	if !r.OK {
		return nil, core.NewError(r.Error())
	}
	m, ok := r.Value.(inference.TextModel)
	if !ok {
		return nil, core.NewError("loaded model is not a TextModel")
	}
	return &benchTextModel{m: m}, nil
}

// benchTextModel adapts inference.TextModel to bench.MatrixModel: greedy no-think drains, the
// engine's own GenerateMetrics mapped onto the bench shape, and the speculative counters when the
// pair engaged.
type benchTextModel struct{ m inference.TextModel }

func (b *benchTextModel) Drain(ctx context.Context, prompt string, maxTokens int) error {
	think := false
	// One CHAT-FRAMED turn (not a raw completion): drafters are trained on chat-framed text, so
	// a raw prompt is out-of-domain and the verify rejects most proposals (measured: 8%%
	// acceptance raw vs healthy framed — the adaptive controller then backs off and the mtp lane
	// reads as plain-minus-overhead). Chat is also the serving shape the grid exists to compare.
	// Fully greedy: temperature, top_p, top_k AND min_p all pinned to 0 — any unset field is
	// filled from the checkpoint's declared sampling defaults, and the pair's dispatch
	// (Temperature > 0 || MinP > 0) would route to the sampled verify lane.
	for range b.m.Chat(ctx, []inference.Message{{Role: "user", Content: prompt}},
		inference.WithMaxTokens(maxTokens),
		inference.WithTemperature(0),
		inference.WithTopP(0),
		inference.WithTopK(0),
		inference.WithMinP(0),
		inference.WithEnableThinking(&think)) {
	}
	if p, ok := b.m.(interface{ Err() core.Result }); ok {
		if r := p.Err(); !r.OK {
			return core.NewError(r.Error())
		}
	}
	return nil
}

func (b *benchTextModel) Metrics() bench.GenerationMetrics {
	mt := b.m.Metrics()
	return bench.GenerationMetrics{
		PromptTokens:        mt.PromptTokens,
		GeneratedTokens:     mt.GeneratedTokens,
		PrefillDuration:     mt.PrefillDuration,
		DecodeDuration:      mt.DecodeDuration,
		TotalDuration:       mt.TotalDuration,
		PrefillTokensPerSec: mt.PrefillTokensPerSec,
		DecodeTokensPerSec:  mt.DecodeTokensPerSec,
		PeakMemoryBytes:     mt.PeakMemoryBytes,
	}
}

func (b *benchTextModel) Close() error {
	if r := b.m.Close(); !r.OK {
		return core.NewError(r.Error())
	}
	return nil
}

// SpeculativeSummary exposes the pair's acceptance counters (bench.MatrixSpeculative) when the
// last generation ran the speculative lane.
func (b *benchTextModel) SpeculativeSummary() (bench.MatrixSpec, bool) {
	p, ok := b.m.(inference.SpeculativeMetricsProvider)
	if !ok {
		return bench.MatrixSpec{}, false
	}
	sm := p.SpeculativeMetrics()
	if sm.ProposedTokens == 0 {
		return bench.MatrixSpec{}, false
	}
	return bench.MatrixSpec{
		ProposedTokens: sm.ProposedTokens,
		AcceptedTokens: sm.AcceptedTokens,
		AcceptanceRate: sm.AcceptanceRate,
	}, true
}
