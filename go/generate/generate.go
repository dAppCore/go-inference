// SPDX-Licence-Identifier: EUPL-1.2

// Package generate is the one-shot generate + decode-tok/s bench + durable
// -state turn loop, rescued out of lthn-mlx's cmd/mlx generate command so the
// business logic lives in a go-inference library rather than dying with
// go-mlx's cmd/. cmd/lem generate is thin flag-parsing over RunGenerate.
//
// It loads a model, generates from a prompt with no HTTP serve in the path, and
// reports decode-only tok/s (prefill excluded) for like-for-like comparison
// against other engines on the same model + quant. It prints the generated text
// too, so it doubles as a quick one-shot run.
//
//	generate.RunGenerate(ctx, generate.Config{ModelPath: dir, Prompt: "hi", MaxTokens: 128, Out: os.Stdout, Log: os.Stderr})
package generate

import (
	"context"
	"io"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/serving"
)

// Config is the declarative generate request mirroring lthn-mlx's generate flag
// surface. RunGenerate turns it into a load + generate run (or, when StateName
// is set, one durable -state turn).
type Config struct {
	ModelPath  string
	Prompt     string
	MaxTokens  int
	Temp       float64
	Think      bool
	ContextLen int

	// Reactive MTP drafter (Gemma 4 targets) — same ladder as serve.
	DraftPath  string // "auto" runs the ladder, "" disables, a path forces the drafter
	DraftBlock int    // explicit MTP draft block; 0 = engine default

	// Engine knobs preserved for the drop-in flag surface. These have no
	// inference.LoadOption seam on the current engine/metal, so RunGenerate
	// prints an honest notice and loads the engine default (see the notice
	// wiring below); they light up when the engine exposes the seam.
	KVCacheMode string // paged, fp16, q8, kq8vq4, turboquant
	KVStorage   string // retained KV storage dtype
	Pipeline    bool   // one-ahead pipelined decode
	Native      bool   // no-cgo native token loop (the default go-inference metal engine already is)
	Trace       bool   // per-token decode phase budget

	// Durable -state turn loop.
	StateName  string // conversation state name (wake → generate → sleep); "" = stateless one-shot
	StateStore string // state store file (default ~/Lethean/data/state/agent.kv)
	Raw        bool   // with -state: skip chat-framing, run the raw completion-loop turn

	LoadOptions []inference.LoadOption
	Out         io.Writer // generated text + metrics
	Log         io.Writer // notices
}

// RunGenerate loads the model and runs one generate (or one -state turn). It is
// the generate business logic ported out of lthn-mlx's cmd/mlx.
func RunGenerate(ctx context.Context, cfg Config) error {
	loadOpts := append([]inference.LoadOption(nil), cfg.LoadOptions...)
	if cfg.ContextLen > 0 {
		loadOpts = append(loadOpts, inference.WithContextLen(cfg.ContextLen))
	}
	// KV-cache mode / storage dtype have no inference.LoadOption on the current
	// engine/metal; note honestly and load the engine default (the seam lights
	// these up when it arrives — matching the serve degradation).
	if core.Trim(cfg.KVCacheMode) != "" {
		printNote(cfg.Log, "generate: -kv-cache %s requested; the engine loads its default cache mode (per-engine cache-mode seam not yet exposed)", cfg.KVCacheMode)
	}
	if core.Trim(cfg.KVStorage) != "" {
		printNote(cfg.Log, "generate: -kv-storage %s requested; the engine loads its default KV storage dtype (seam not yet exposed)", cfg.KVStorage)
	}
	if cfg.Native {
		printNote(cfg.Log, "generate: native no-cgo token loop (the default go-inference metal engine already is native)")
	}

	if cfg.StateName != "" {
		return runStateTurn(ctx, cfg, loadOpts)
	}
	return runBasicGenerate(ctx, cfg, loadOpts)
}

// runBasicGenerate loads the model, warms the kernels, then times a prefill +
// decode run and reports decode-only tok/s (comparable to llama-bench's tg).
func runBasicGenerate(ctx context.Context, cfg Config, loadOpts []inference.LoadOption) error {
	// Reactive MTP pair resolution — same ladder as serve. A detected drafter
	// only engages when the engine exposes a speculative loader; the current
	// engine/metal doesn't, so a detected drafter degrades to plain decode with
	// an honest notice (faithful to lthn-mlx's own degradation).
	if det := serving.ResolveServeDraft(cfg.ModelPath, cfg.DraftPath, true); det.Active() {
		printNote(cfg.Log, "generate: drafter %s (%s) detected but this engine exposes no speculative path — generating plain autoregressive (block %d would apply)",
			det.DraftPath, det.Note, resolvedDraftBlock(cfg.DraftBlock))
	}

	tm, err := loadTextModel(cfg.ModelPath, loadOpts...)
	if err != nil {
		return core.E("generate.RunGenerate", "load", err)
	}
	defer tm.Close()

	off := !cfg.Think
	msgs := []inference.Message{{Role: "user", Content: cfg.Prompt}}
	genOpts := func(limit int) []inference.GenerateOption {
		opts := []inference.GenerateOption{
			inference.WithMaxTokens(limit),
			inference.WithEnableThinking(&off),
			inference.WithTemperature(float32(cfg.Temp)),
		}
		// -trace enables the engine's coarse per-token phase timing to its trace
		// log. The structured per-token phase-budget TABLE lthn-mlx printed reads
		// mlx.Metrics.TokenPhases, which go-inference's inference.GenerateMetrics
		// does not expose — that surface is a separate engine seam. Honest note.
		if cfg.Trace {
			opts = append(opts, inference.GenerateOption(func(c *inference.GenerateConfig) { c.TraceTokenPhases = true }))
		}
		return opts
	}
	if cfg.Trace {
		printNote(cfg.Log, "generate: -trace enables coarse per-token phase timing to the engine trace log; the structured phase-budget table needs a TokenPhases metrics surface (engine seam not yet exposed)")
	}

	// run generates up to limit tokens, timing prefill (start → first token)
	// separately from decode (first → last) so the reported rate is steady-state.
	run := func(limit int, collect *[]byte) (n int, prefill, decode time.Duration) {
		start := time.Now()
		var first time.Time
		for tok := range tm.Chat(ctx, msgs, genOpts(limit)...) {
			if n == 0 {
				first = time.Now()
				prefill = first.Sub(start)
			}
			if collect != nil {
				*collect = append(*collect, tok.Text...)
			}
			n++
		}
		decode = time.Since(first)
		return n, prefill, decode
	}

	run(8, nil) // warm the kernels — first call pays compilation + allocation
	if r := tm.Err(); !r.OK {
		return core.E("generate.RunGenerate", "warm", r.Value.(error))
	}
	var out []byte
	n, prefill, decode := run(cfg.MaxTokens, &out)
	if r := tm.Err(); !r.OK {
		return core.E("generate.RunGenerate", "generate", r.Value.(error))
	}
	if n < 2 {
		return core.E("generate.RunGenerate", core.Sprintf("produced only %d tokens", n), nil)
	}

	core.WriteString(cfg.Out, string(out))
	core.WriteString(cfg.Out, "\n\n")
	core.WriteString(cfg.Out, core.Sprintf(
		"decode %.1f tok/s  (%d tok / %.3fs, prefill %dms excluded)  ·  total %.1f tok/s\n",
		float64(n-1)/decode.Seconds(), n, decode.Seconds(), prefill.Milliseconds(),
		float64(n)/(prefill+decode).Seconds(),
	))
	printMTPMetrics(cfg.Out, tm)
	return nil
}

// resolvedDraftBlock reports the block the MTP lane would run for a flag value
// (0 = engine default).
func resolvedDraftBlock(flagBlock int) int {
	if flagBlock > 0 {
		return flagBlock
	}
	return serving.MTPDefaultDraftBlock
}

// printMTPMetrics appends the MTP acceptance line when the generation rode the
// speculative lane — the bench instrument's read on whether the drafter is
// earning its keep. It reads the engine-agnostic inference.SpeculativeMetrics
// (zero-valued, so silent, unless speculation engaged).
func printMTPMetrics(out io.Writer, tm inference.TextModel) {
	provider, ok := tm.(inference.SpeculativeMetricsProvider)
	if !ok {
		return
	}
	mtp := provider.SpeculativeMetrics()
	if mtp.ProposedTokens == 0 {
		return
	}
	core.WriteString(out, core.Sprintf(
		"mtp: %.0f%% acceptance (%d/%d drafted) over %d verify forwards\n",
		mtp.AcceptanceRate*100, mtp.AcceptedTokens, mtp.ProposedTokens, mtp.TargetVerifyCalls,
	))
}

// loadTextModel loads path through the registered "metal" backend (the no-cgo
// Apple engine) as an inference.TextModel.
func loadTextModel(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
	merged := append(append([]inference.LoadOption(nil), opts...), inference.WithBackend("metal"))
	result := inference.LoadModel(path, merged...)
	if !result.OK {
		if err, ok := result.Value.(error); ok {
			return nil, err
		}
		return nil, core.E("generate.loadTextModel", "metal backend failed to load model", nil)
	}
	tm, ok := result.Value.(inference.TextModel)
	if !ok || tm == nil {
		return nil, core.E("generate.loadTextModel", "metal backend returned non-TextModel value", nil)
	}
	return tm, nil
}

// printNote writes a generate notice to w (nil silences it).
func printNote(w io.Writer, format string, args ...any) {
	if w == nil {
		return
	}
	core.Print(w, format, args...)
}
