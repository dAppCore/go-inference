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

	// ImageSources are --image inputs threaded through the neutral multimodal
	// path, each a local file path or a base64 "data:" URL (the same shapes
	// serve accepts). They attach to the user turn as inference.Message.Images
	// and are gated on the model's inference.VisionModel capability, exactly as
	// serve's chat-completions handler carries image content parts. Only the
	// stateless one-shot path carries images; -state turns reject them (the
	// durable session prefills text prompts only).
	ImageSources []string
	// AudioSources are --audio inputs. There is no engine-neutral audio-input
	// seam yet (inference.Message carries Images, not audio), so a non-empty
	// value is rejected honestly rather than silently dropped — audio input is
	// a follow-up once the engine exposes the seam.
	AudioSources []string

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
	StateStore string // state store file (default ~/Lethean/lem/state/agent.kv)
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
	// KV-cache mode / storage dtype overrides are validated against the loaded
	// engine's reported capabilities (noteCacheKnobs, once the model is loaded)
	// rather than blanket-noted here — the note then names what the engine
	// actually honours instead of guessing.
	if cfg.Native {
		printNote(cfg.Log, "generate: native no-cgo token loop (the default go-inference metal engine already is native)")
	}

	// Audio input has no engine-neutral seam yet (inference.Message carries
	// Images, not audio): reject rather than silently drop it, so the caller
	// never gets a text-only answer that quietly ignored their audio.
	if len(cfg.AudioSources) > 0 {
		return core.E("generate.RunGenerate", "audio input is not yet exposed on the engine-neutral path — image input is wired, audio is a follow-up", nil)
	}

	if cfg.StateName != "" {
		// The durable -state turn loop prefills text prompts through the spine
		// session, which has no image seam; reject rather than drop the images.
		if len(cfg.ImageSources) > 0 {
			return core.E("generate.RunGenerate", "image input is not supported with -state yet — use stateless generate for vision (the durable session prefills text prompts only)", nil)
		}
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

	// Resolve --image sources to raw bytes BEFORE loading the model, so a bad
	// path or malformed data: URL fails fast without paying the load cost.
	images, err := resolveImageInputs(cfg.ImageSources)
	if err != nil {
		return core.E("generate.RunGenerate", "image input", err)
	}

	tm, err := loadTextModel(cfg.ModelPath, loadOpts...)
	if err != nil {
		return core.E("generate.RunGenerate", "load", err)
	}
	defer tm.Close()
	noteCacheKnobs(cfg, tm)

	// Gate images on the model's neutral vision capability, exactly as serve's
	// chat-completions handler does before prefill.
	if err := requireVision(tm, images); err != nil {
		return core.E("generate.RunGenerate", "vision", err)
	}

	off := !cfg.Think
	msgs := []inference.Message{{Role: "user", Content: cfg.Prompt, Images: images}}
	genOpts := func(limit int) []inference.GenerateOption {
		opts := []inference.GenerateOption{
			inference.WithMaxTokens(limit),
			inference.WithEnableThinking(&off),
			inference.WithTemperature(float32(cfg.Temp)),
		}
		// -trace turns on the engine's per-token phase timing. The engine folds the
		// aggregate GPU-busy / host-serial split into inference.GenerateMetrics
		// .DecodePhases (populated only if the active decode path is instrumented),
		// which printDecodePhaseBudget renders after the run.
		if cfg.Trace {
			opts = append(opts, inference.GenerateOption(func(c *inference.GenerateConfig) { c.TraceTokenPhases = true }))
		}
		return opts
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
	if cfg.Trace {
		if budget := tm.Metrics().DecodePhases; budget != nil && budget.Tokens > 0 {
			printDecodePhaseBudget(cfg.Out, budget)
		} else {
			printNote(cfg.Log, "generate: -trace: the active decode path reported no phase budget — this engine instruments its greedy GPU decode tail; a path without phase timing leaves the budget empty")
		}
	}
	return nil
}

// printDecodePhaseBudget renders the traced per-token decode budget: the GPU-busy
// vs host-serial split (the host-serial share is the GPU-idle wall a deeper
// pipeline could overlap — the perf headroom) and each engine-named phase's
// share. It is the go-inference equivalent of lthn-mlx's phase-budget table,
// reading the neutral inference.DecodePhaseBudget instead of test-only globals.
func printDecodePhaseBudget(out io.Writer, budget *inference.DecodePhaseBudget) {
	total := msPerToken(budget.TotalPerToken)
	gpu := msPerToken(budget.GPUPerToken)
	host := msPerToken(budget.HostPerToken())
	core.WriteString(out, core.Sprintf("\ndecode phase budget — %d tokens · %.3f ms/token · %.1f tok/s\n",
		budget.Tokens, total, tokPerSec(total)))
	core.WriteString(out, core.Sprintf("  GPU busy    %8.3f ms  %5.1f%%\n", gpu, 100*budget.GPUFraction()))
	ceiling := "n/a"
	if gpu > 0 {
		ceiling = core.Sprintf("%.1f", 1000.0/gpu)
	}
	core.WriteString(out, core.Sprintf("  host serial %8.3f ms  %5.1f%%   <- GPU idle; tok/s ceiling if zeroed: %s\n",
		host, 100*(1-budget.GPUFraction()), ceiling))
	for _, phase := range budget.Phases {
		ms := msPerToken(phase.PerToken)
		if ms < 0.001 {
			continue
		}
		lane := "host"
		if phase.GPU {
			lane = "GPU"
		}
		pct := 0.0
		if budget.TotalPerToken > 0 {
			pct = 100 * float64(phase.PerToken) / float64(budget.TotalPerToken)
		}
		core.WriteString(out, core.Sprintf("    %-24s %8.3f ms  %5.1f%%  (%s)\n", phase.Name, ms, pct, lane))
	}
}

// msPerToken renders a per-token duration in milliseconds.
func msPerToken(d time.Duration) float64 { return float64(d.Microseconds()) / 1000.0 }

// tokPerSec is tokens/sec from a per-token millisecond figure (0 when untimed).
func tokPerSec(ms float64) float64 {
	if ms <= 0 {
		return 0
	}
	return 1000.0 / ms
}

// noteCacheKnobs prints an honest, capability-driven note for the -kv-cache /
// -kv-storage overrides. The loaded engine reports which KV cache modes it
// honours (inference.CapabilityReport.CacheModes), so a requested mode outside
// that set is reported as ignored rather than silently dropped. The metal engine
// reports a single native cache and honours no go-mlx-era selector (fp16 / q8 /
// kq8vq4 / turboquant), so any override lands here naming what is supported; a
// future engine that honours a selector lists it and only an unknown mode notes.
func noteCacheKnobs(cfg Config, tm inference.TextModel) {
	if req := core.Trim(cfg.KVCacheMode); req != "" {
		modes := reportedCacheModes(tm)
		if !cacheModeHonoured(modes, req) {
			printNote(cfg.Log, "generate: -kv-cache %q is not honoured by this engine%s; it runs its built-in KV cache. Override ignored.",
				cfg.KVCacheMode, cacheModesSuffix(modes))
		}
	}
	if req := core.Trim(cfg.KVStorage); req != "" {
		printNote(cfg.Log, "generate: -kv-storage %q is not honoured by this engine; it runs its native KV storage dtype. Override ignored.", cfg.KVStorage)
	}
}

// reportedCacheModes returns the KV cache modes the loaded model's engine
// declares through the capability seam (nil when the model reports none).
func reportedCacheModes(tm inference.TextModel) []string {
	report, ok := inference.CapabilitiesOf(tm)
	if !ok {
		return nil
	}
	return report.CacheModes
}

// cacheModeHonoured reports whether want is one of the engine's declared cache
// modes (case-insensitive). An empty list means the engine declares no
// selectable mode, so nothing is honoured.
func cacheModeHonoured(modes []string, want string) bool {
	for _, mode := range modes {
		if core.Lower(mode) == core.Lower(want) {
			return true
		}
	}
	return false
}

// cacheModesSuffix renders the supported-modes hint for the note ("" when none).
func cacheModesSuffix(modes []string) string {
	if len(modes) == 0 {
		return ""
	}
	return core.Sprintf(" (this engine supports: %s)", core.Join(", ", modes...))
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
