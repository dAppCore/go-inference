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
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/serving"
	"dappco.re/go/inference/serving/provider/openai"
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
	// AudioSources are --audio inputs (WAV, 16-bit PCM mono 16 kHz), each a
	// local file path or a base64 "data:" URL. They attach to the user turn as
	// inference.Message.Audios and are gated on the model's audio capability.
	// Only the stateless one-shot path carries audio; -state turns reject it.
	AudioSources []string
	// VideoFrameSources are --video-frame inputs: the sampled frames of ONE
	// video in time order (PNG/JPEG path or data: URL each). Frames become
	// timestamped vision blocks (1 s apart) and gate on the vision capability.
	VideoFrameSources []string

	// Reactive MTP drafter (Gemma 4 targets) — same ladder as serve.
	DraftPath  string // "auto" runs the ladder, "" disables, a path forces the drafter
	DraftBlock int    // explicit MTP draft block; 0 = engine default
	// SpeculativeLoader loads a target+drafter pair as one speculative
	// inference.TextModel. Injected by the composition root (cmd/lem →
	// native.LoadSpeculativePair) so this package stays engine-neutral; nil
	// leaves generate on the plain path with the "no speculative path" note.
	SpeculativeLoader serving.SpeculativeLoader

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

	if cfg.StateName != "" {
		// The durable -state turn loop prefills text prompts through the spine
		// session, which has no multimodal seam; reject rather than drop.
		if len(cfg.ImageSources) > 0 {
			return core.E("generate.RunGenerate", "image input is not supported with -state yet — use stateless generate for vision (the durable session prefills text prompts only)", nil)
		}
		if len(cfg.AudioSources) > 0 {
			return core.E("generate.RunGenerate", "audio input is not supported with -state yet — use stateless generate for audio", nil)
		}
		if len(cfg.VideoFrameSources) > 0 {
			return core.E("generate.RunGenerate", "video input is not supported with -state yet — use stateless generate for video", nil)
		}
		return runStateTurn(ctx, cfg, loadOpts)
	}
	return runBasicGenerate(ctx, cfg, loadOpts)
}

// warmPrefixChars bounds the prompt text the kernel-warm pass prefills: ~8K chars
// ≈ 1.5K tokens ≈ three 512-row batched-prefill chunks — enough to compile every
// PSO and size every slab the timed run needs, at seconds instead of the full
// prompt's minutes at depth.
const warmPrefixChars = 8192

// warmPrefix returns the prompt bounded to warmPrefixChars, backed off to a rune
// boundary so the truncation never splits a UTF-8 sequence.
func warmPrefix(s string) string {
	if len(s) <= warmPrefixChars {
		return s
	}
	cut := warmPrefixChars
	for cut > 0 && s[cut]&0xC0 == 0x80 {
		cut--
	}
	return s[:cut]
}

// runBasicGenerate loads the model, warms the kernels, then times a prefill +
// decode run and reports decode-only tok/s (comparable to llama-bench's tg).
func runBasicGenerate(ctx context.Context, cfg Config, loadOpts []inference.LoadOption) error {
	// Resolve --image sources to raw bytes BEFORE loading the model, so a bad
	// path or malformed data: URL fails fast without paying the load cost. It is
	// also the gate on arming the speculative lane below.
	images, err := resolveImageInputs(cfg.ImageSources)
	if err != nil {
		return core.E("generate.RunGenerate", "image input", err)
	}
	audios, err := resolveAudioInputs(cfg.AudioSources)
	if err != nil {
		return core.E("generate.RunGenerate", "audio input", err)
	}
	videoFrames, err := resolveImageInputs(cfg.VideoFrameSources) // frames are images: same shapes + caps
	if err != nil {
		return core.E("generate.RunGenerate", "video frame input", err)
	}

	// Reactive MTP pair resolution — same ladder as serve. A detected drafter
	// arms the speculative lane when the engine exposes a loader AND the request
	// carries no images (an image turn routes through a multimodal prefill the
	// MTP loop does not carry). Greedy requests ride the greedy-exact verify
	// (byte-identical to plain decode at temp 0); sampled requests ride the
	// sampled verify lane, where the target's sampler decides every committed
	// token. Every miss degrades to plain autoregressive with an honest notice.
	var tm inference.TextModel
	if det := serving.ResolveServeDraft(cfg.ModelPath, cfg.DraftPath, true); det.Active() {
		block := resolvedDraftBlock(cfg.DraftBlock)
		switch {
		case cfg.SpeculativeLoader == nil:
			printNote(cfg.Log, "generate: drafter %s (%s) detected but this engine exposes no speculative path — generating plain autoregressive (block %d would apply)", det.DraftPath, det.Note, block)
		case len(images) > 0 || len(audios) > 0 || len(videoFrames) > 0:
			printNote(cfg.Log, "generate: drafter %s detected but multimodal input routes through a prefill the MTP loop does not carry — generating plain autoregressive", det.DraftPath)
		default:
			sm, serr := cfg.SpeculativeLoader(cfg.ModelPath, det.DraftPath, block, loadOpts...)
			if serr != nil {
				printNote(cfg.Log, "generate: drafter %s detected but the speculative pair failed to load (%v) — generating plain autoregressive", det.DraftPath, serr)
			} else {
				printNote(cfg.Log, "generate: MTP speculative lane armed — drafter %s, block %d", det.DraftPath, block)
				tm = sm
			}
		}
	}

	if tm == nil {
		plain, lerr := loadTextModel(cfg.ModelPath, loadOpts...)
		if lerr != nil {
			return core.E("generate.RunGenerate", "load", lerr)
		}
		tm = plain
	}
	defer tm.Close()
	noteCacheKnobs(cfg, tm)
	noteKVStorageInert(cfg) // -kv-storage bites only on the -state sleep path

	// Gate images/audio on the model's neutral capabilities, exactly as serve's
	// chat-completions handler does before prefill.
	if err := requireVision(tm, images); err != nil {
		return core.E("generate.RunGenerate", "vision", err)
	}
	if err := requireVision(tm, videoFrames); err != nil {
		return core.E("generate.RunGenerate", "video", err)
	}
	if err := requireAudio(tm, audios); err != nil {
		return core.E("generate.RunGenerate", "audio", err)
	}

	think := cfg.Think
	msgs := []inference.Message{{Role: "user", Content: cfg.Prompt, Images: images, Audios: audios, Videos: videoFrames}}
	genOpts := func(limit int) []inference.GenerateOption {
		opts := []inference.GenerateOption{
			inference.WithMaxTokens(limit),
			// enable semantics: -think renders the gemma4 <|think|> system prelude;
			// the default keeps it off so the decode rate stays clean.
			inference.WithEnableThinking(&think),
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
	run := func(m []inference.Message, limit int, collect *[]byte) (n int, prefill, decode time.Duration) {
		start := time.Now()
		var first time.Time
		for tok := range tm.Chat(ctx, m, genOpts(limit)...) {
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

	// Warm on a bounded PREFIX of the prompt, not the whole thing. The warm pass
	// exists to pay one-time costs — PSO compilation, scratch/slab allocation, the
	// batched-prefill lane's first-use setup — and a few 512-row chunks covers all
	// of them. Warming the full prompt runs the deep prefill twice: a 63K-token
	// prompt cost ~96s of unmeasured warm prefill before the ~96s measured one.
	// Chat calls on this path never reuse KV across runs (the measured prefill is
	// cold either way), so the prefix warm changes nothing in the timed window.
	warmMsgs := []inference.Message{{Role: "user", Content: warmPrefix(cfg.Prompt), Images: images, Audios: audios, Videos: videoFrames}}
	run(warmMsgs, 8, nil) // warm the kernels — first call pays compilation + allocation
	if r := tm.Err(); !r.OK {
		return core.E("generate.RunGenerate", "warm", r.Value.(error))
	}
	var out []byte
	n, prefill, decode := run(msgs, cfg.MaxTokens, &out)
	if r := tm.Err(); !r.OK {
		return core.E("generate.RunGenerate", "generate", r.Value.(error))
	}
	if n < 2 {
		return core.E("generate.RunGenerate", core.Sprintf("produced only %d tokens", n), nil)
	}

	// The raw stream carries gemma4's channel markers (the 26B emits an EMPTY thought
	// channel even with thinking off). Show what a serving client sees: the same shared
	// extractor every serving route runs, content only — the reasoning stream prints
	// under a `thought:` header when thinking is on.
	extractor := openai.NewThinkingExtractor()
	content, thought := extractor.Process(inference.Token{Text: string(out)})
	flushContent, flushThought := extractor.Flush()
	content += flushContent
	thought += flushThought
	if cfg.Think && core.Trim(thought) != "" {
		core.WriteString(cfg.Out, "thought: "+thought+"\n---\n")
	}
	core.WriteString(cfg.Out, content)
	core.WriteString(cfg.Out, "\n\n")
	// prefill rate alongside the decode rate: prompt tokens over the wall
	// prefill window (start → first token), the cold-ingest number llama-bench
	// reports as pp<N> — engines that don't report a prompt count keep the
	// bare-duration wording.
	prefillPart := core.Sprintf("prefill %dms excluded", prefill.Milliseconds())
	if pt := tm.Metrics().PromptTokens; pt > 0 && prefill > 0 {
		prefillPart = core.Sprintf("prefill %d tok @ %.0f tok/s (%.3fs)", pt, float64(pt)/prefill.Seconds(), prefill.Seconds())
	}
	core.WriteString(cfg.Out, core.Sprintf(
		"decode %.1f tok/s  (%d tok / %.3fs)  ·  %s  ·  total %.1f tok/s\n",
		float64(n-1)/decode.Seconds(), n, decode.Seconds(), prefillPart,
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

// noteCacheKnobs prints an honest, capability-driven note for the -kv-cache
// override — the LIVE decode cache. The loaded engine reports which KV cache
// modes it honours (inference.CapabilityReport.CacheModes); the metal engine
// runs a single native cache and honours no go-mlx-era live selector (fp16 / q8
// / kq8vq4 / turboquant), so any override lands here naming what is supported. A
// future engine that honours a live selector lists it and only an unknown mode
// notes. -kv-storage is the SNAPSHOT-storage knob, resolved separately (it is
// engine-neutral — the kv.Encoding set — and only bites on the -state path).
func noteCacheKnobs(cfg Config, tm inference.TextModel) {
	if req := core.Trim(cfg.KVCacheMode); req != "" {
		modes := reportedCacheModes(tm)
		if !cacheModeHonoured(modes, req) {
			printNote(cfg.Log, "generate: -kv-cache %q is not honoured by this engine%s; it runs its built-in KV cache. Override ignored.",
				cfg.KVCacheMode, cacheModesSuffix(modes))
		}
	}
}

// kvStorageEncoding resolves the -kv-storage flag to a portable KV snapshot
// kv.Encoding. recognised is false for a value outside the kv.Encoding set (the
// go-mlx-era fp16/bf16 storage-dtype vocabulary never mapped to a distinct
// portable encoding here); those fall back to native. All three real encodings
// are produce-able from a live metal -state sleep — native keeps its fast
// bf16-slab path, q8 quantises to int8+scale, float32 keeps exact tensors (the
// block capture emits per-head float32 for the non-native ones). The encodings
// live in the kv package — engine-neutral — so this validates directly rather
// than through the engine capability seam.
func kvStorageEncoding(raw string) (enc kv.Encoding, recognised bool) {
	switch kv.Encoding(core.Lower(core.Trim(raw))) {
	case "", kv.EncodingNative:
		return kv.EncodingNative, true
	case kv.EncodingQ8:
		return kv.EncodingQ8, true
	case kv.KVSnapshotEncodingFloat32:
		return kv.KVSnapshotEncodingFloat32, true
	default:
		return kv.EncodingNative, false
	}
}

// noteKVStorageInert reports that -kv-storage has no effect on a stateless run:
// only the -state sleep path persists KV, so a bench/one-shot generate stores
// nothing to encode. An unrecognised value is flagged here too.
func noteKVStorageInert(cfg Config) {
	raw := core.Trim(cfg.KVStorage)
	if raw == "" {
		return
	}
	if _, recognised := kvStorageEncoding(raw); !recognised {
		printNote(cfg.Log, "generate: -kv-storage %q is not a known KV snapshot encoding (native, q8, float32)", cfg.KVStorage)
	}
	printNote(cfg.Log, "generate: -kv-storage selects the -state snapshot encoding; this stateless run persists no KV, so it has no effect here")
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
