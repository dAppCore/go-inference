// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// speculative_model.go is the seam that was missing: it wires the tested
// AssistantPair MTP loop (draft → verify → accept, all in assistant_load.go) to
// the inference.TextModel contract that serve and generate drive. Before this,
// a detected drafter degraded to plain autoregressive with "this engine exposes
// no speculative path" — because nothing turned the pair into a TextModel a
// SpeculativeLoader could hand back. LoadSpeculativePair is that loader.
//
// Two verify lanes: at temperature 0 the greedy-exact verify accepts exactly
// the tokens the target would have argmax-produced (byte-identical to plain
// decode, only faster); at temperature > 0 the sampled verify lane lets the
// TARGET's sampler decide every committed token, with drafts only affecting
// acceptance — so a sampled request through the pair is a true sampled
// generation.
package native

import (
	"context"
	"iter"
	"slices"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/dflash"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/engine"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/mtp"
	coreio "dappco.re/go/io"
)

// speculativeModel adapts a target ArchSession + drafter AssistantPair to
// inference.TextModel. The target weights are resident in one ArchSession
// (loaded once by LoadDir); the pair carries the drafter plus the target arch
// config (no second copy of the target weights). Generate/Chat run the
// speculative loop and record inference.SpeculativeMetrics for the -draft
// bench read (printMTPMetrics).
type speculativeModel struct {
	target     *ArchSession
	pair       *AssistantPair
	dflash     *DFlashDrafter
	tok        *tokenizer.Tokenizer
	info       inference.ModelInfo
	modelType  string
	draftBlock int
	turns      engine.TurnTokens
	// declaredStops is the target checkpoint's generation_config eos set,
	// mirrored from the plain path so a speculative serve terminates turns
	// identically.
	declaredStops []int32
	// declaredSampling is the target checkpoint's generation_config sampling
	// intent, folded per request exactly as the plain engine seam folds it
	// (engine.SamplingDefaults.Apply) so a speculative serve honours the
	// model's declared temperature/top-p/top-k/min-p defaults identically.
	declaredSampling engine.SamplingDefaults

	mu          sync.Mutex
	lastErr     error
	lastMetrics inference.GenerateMetrics
	lastSpec    inference.SpeculativeMetrics
}

var (
	_ inference.TextModel                  = (*speculativeModel)(nil)
	_ inference.SpeculativeMetricsProvider = (*speculativeModel)(nil)
	_ inference.SerialModel                = (*speculativeModel)(nil)
)

// SerialGeneration marks the speculative pair as single-session
// (inference.SerialModel): the target ArchSession (its KV cache and decode
// scratch) and the pair's fused drafter (one draft command buffer, one resident
// drafter-KV set built once per pair via fusedDraft) are shared singletons, so
// two concurrent Generate/Chat calls race that GPU scratch and crash — the
// loadKV nil drafter-KV SIGSEGV seen under -scheduler interleave (#1842). A
// scheduler serialises this model's generation lane on the strength of this
// declaration; it always returns true because the pair is never per-request.
func (m *speculativeModel) SerialGeneration() bool { return true }

// LoadSpeculativePair loads a target checkpoint + drafter as one speculative
// inference.TextModel running draftBlock-wide MTP verify forwards — the
// serving.SpeculativeLoader shape. The target weights load once via LoadDir; the
// pair adds the drafter and validates the attachment (arch/backbone match).
// draftBlock ≤ 0 falls back to the shipped default of 5.
func LoadSpeculativePair(targetPath, draftPath string, draftBlock int, opts ...inference.LoadOption) (inference.TextModel, error) {
	cfg := inference.ApplyLoadOpts(opts)
	maxLen := cfg.ContextLen
	// maxLen <= 0 defers to the loader's checkpoint-window default.
	if draftBlock <= 0 {
		draftBlock = 5
	}
	// Family dispatch: a composed/hybrid TARGET (Qwen 3.5/3.6 — gated-delta recurrent state, no
	// shareable KV streams) cannot be an ArchSession, so its pairing binds through the composed arm.
	// The route stays ONE: same loader shape, same TextModel surfaces, different physics underneath.
	if spec, ok := model.LookupArch(probeModelType(targetPath)); ok && spec.Composed != nil {
		return loadComposedSpeculativePair(targetPath, draftPath, draftBlock, opts...)
	}
	target, err := LoadDir(targetPath, maxLen)
	if err != nil {
		return nil, core.E("native.LoadSpeculativePair", "load target checkpoint", err)
	}
	pair, err := LoadAssistantPairDirs(targetPath, draftPath)
	if err != nil {
		_ = target.Close()
		return nil, core.E("native.LoadSpeculativePair", "attach drafter", err)
	}
	var dflashDrafter *DFlashDrafter
	if pair.Method() == mtp.MTPDFlash {
		cfgJSON, readErr := coreio.Local.Read(core.PathJoin(draftPath, "config.json"))
		if readErr != nil {
			_ = target.Close()
			_ = pair.Close()
			return nil, core.E("native.LoadSpeculativePair", "read DFlash config", readErr)
		}
		cfg, ok := dflash.ParseConfig([]byte(cfgJSON))
		if !ok {
			_ = target.Close()
			_ = pair.Close()
			return nil, core.NewError("native.LoadSpeculativePair: DFlash method has no DFlash config")
		}
		dflashDrafter, err = newDFlashDrafter(pair.Assistant, cfg)
		if err != nil {
			_ = target.Close()
			_ = pair.Close()
			return nil, core.E("native.LoadSpeculativePair", "attach DFlash drafter", err)
		}
	}
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(targetPath, "tokenizer.json"))
	if err != nil {
		_ = target.Close()
		_ = pair.Close()
		return nil, core.E("native.LoadSpeculativePair", "load tokenizer", err)
	}
	// Stamp the target checkpoint's real architecture (config.json model_type),
	// not a hardcoded "gemma4" — the speculative pair must self-report as
	// truthfully as the plain path does.
	modelType := probeModelType(targetPath)
	return &speculativeModel{
		target:           target,
		pair:             pair,
		dflash:           dflashDrafter,
		tok:              tok,
		modelType:        modelType,
		draftBlock:       draftBlock,
		turns:            engine.DetectTurnTokens(tok),
		declaredStops:    loadGenerationConfigStops(targetPath),
		declaredSampling: loadGenerationConfigSamplingDefaults(targetPath),
		info: inference.ModelInfo{
			Architecture: modelType,
			VocabSize:    pair.TargetArch.Vocab,
			NumLayers:    len(pair.TargetArch.Layer),
			HiddenSize:   pair.TargetArch.Hidden,
		},
	}, nil
}

// Generate streams the speculative completion of a raw prompt.
func (m *speculativeModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.speculate(ctx, m.tok.Encode(prompt), inference.ApplyGenerateOpts(opts))
}

// Chat streams the speculative completion of a multi-turn conversation, framed
// through the SHARED engine render (engine.RenderChatTurns over the gemma
// ChatTemplate for the target's detected marker dialect) — byte-identical to
// the plain engine path's plain-turns framing, with no private template copy to
// drift.
func (m *speculativeModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	prompt := engine.RenderChatTurns(engine.GemmaChatTemplate(m.turns, false), messages)
	return m.speculate(ctx, m.tok.Encode(prompt), inference.ApplyGenerateOpts(opts))
}

// speculate runs the AssistantPair MTP loop over the target session, decoding
// and yielding each emitted token (accepted drafts + fallbacks) and blanking a
// terminator's text so it never leaks as a literal "<end_of_turn>". It records
// the run's speculative + generate metrics for the metric providers.
func (m *speculativeModel) speculate(ctx context.Context, ids []int32, cfg inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		start := time.Now()
		if ctx == nil {
			ctx = context.Background()
		}
		if len(ids) == 0 {
			m.setErr(core.NewError("native.speculativeModel.Generate: empty prompt after tokenisation"))
			return
		}
		// Fold the target checkpoint's declared sampling defaults (request-set >
		// model-declared > engine fallback) before the sample-vs-greedy decision
		// below reads cfg.Temperature/MinP — the same seam the plain engine path
		// folds at, so a speculative serve honours a declared temperature 0.7
		// instead of falling to greedy.
		cfg = m.declaredSampling.Apply(cfg)
		maxNew := cfg.MaxTokens
		if maxNew <= 0 {
			maxNew = 256
		}
		eos := int32(-1)
		if m.tok.HasEOSToken() {
			eos = m.tok.EOS()
		}
		stop := append([]int32(nil), cfg.StopTokens...)
		if eos >= 0 {
			stop = append(stop, eos)
		}
		// Mirror the plain engine path: the turn-close marker terminates an
		// assistant turn on chat-tuned checkpoints (gemma4 <turn|>), and the
		// checkpoint's declared generation_config stops join it.
		if id, ok := m.tok.TokenID(m.turns.Close); ok && !speculativeTokenInSet(id, stop) {
			stop = append(stop, id)
		}
		for _, id := range m.declaredStops {
			if id >= 0 && !speculativeTokenInSet(id, stop) {
				stop = append(stop, id)
			}
		}
		sink := func(id int32) bool {
			if ctx.Err() != nil {
				return false
			}
			text := m.tok.DecodeToken(id)
			inStop := speculativeTokenInSet(id, stop)
			if inStop {
				// A terminator's text is never content — blank it (the id still
				// surfaces for trackers), mirroring the plain decode path.
				text = ""
			}
			if !yield(inference.Token{ID: id, Text: text}) {
				return false
			}
			// A stop token ends the turn: the eosID plumbed into the pair loop
			// only covers <eos>, so the sink owns the turn-close (<turn|>) stop.
			return !inStop
		}
		var (
			res  AssistantGenerateResult
			gerr error
		)
		mtp := func() (AssistantGenerateResult, error) {
			if cfg.Temperature > 0 || cfg.MinP > 0 || cfg.RepeatPenalty > 1 {
				// The sampled verify lane: the target's sampler decides every committed
				// token (drafts only affect acceptance), so a temp>0 request through
				// the pair is a true sampled generation — not a silent greedy decode.
				return m.pair.GenerateSampledFromSessionEach(m.target, ids, maxNew, stop,
					model.NewSampler(cfg.Seed), speculativeSampleParams(cfg), m.draftBlock, sink)
			}
			return m.pair.GenerateFromSessionEach(m.target, ids, maxNew, int(eos), m.draftBlock, cfg.SuppressTokens, sink)
		}
		res, gerr = speculativeMethodRoute(m.pair, mtp, func() (AssistantGenerateResult, error) {
			return m.generateDFlash(ids, maxNew, stop, cfg.SuppressTokens, sink)
		})
		m.record(res, len(ids), time.Since(start), gerr)
	}
}

// speculativeMethodRoute is the serving decode switch. Keeping the switch
// independent of either engine loop lets scripted-engine tests prove that the
// pair's declared method, rather than checkpoint naming or caller state, is the
// sole routing input.
func speculativeMethodRoute(pair *AssistantPair, runMTP, dfl func() (AssistantGenerateResult, error)) (AssistantGenerateResult, error) {
	if pair != nil && pair.Method() == mtp.MTPDFlash {
		return dfl()
	}
	return runMTP()
}

// generateDFlash drives the block-diffusion proposer through the model-free,
// lossless DFlash verifier. Until the live aux tap lands, both the aux source and
// target oracle deliberately replay the full prefix. That is slower, but it does
// not depend on or corrupt an incremental serving cache and makes the routing
// complete behind the still-false serving probe.
func (m *speculativeModel) generateDFlash(ids []int32, maxNew int, stop, suppress []int32, sink AssistantTokenSink) (AssistantGenerateResult, error) {
	if m.dflash == nil {
		return AssistantGenerateResult{}, core.NewError("native.speculativeModel: DFlash pair has no DFlash drafter")
	}
	prompt := speculativeInts(ids)
	var runErr error
	proposer := NewDFlashProposer(m.dflash, func(context []int) ([][]byte, []byte, int, bool) {
		prefix := speculativeInt32s(context)
		aux, err := ExtractAuxHiddens(m.target, prefix, m.dflash.AuxLayers())
		if err != nil {
			runErr = core.E("native.speculativeModel.DFlash", "extract aux hiddens", err)
			return nil, nil, 0, false
		}
		anchor, err := m.target.embedID(prefix[len(prefix)-1])
		if err != nil {
			runErr = core.E("native.speculativeModel.DFlash", "embed anchor", err)
			return nil, nil, 0, false
		}
		return aux, append([]byte(nil), anchor...), len(prefix) - 1, true
	})
	next := func(prefix []int) int {
		if runErr != nil {
			return 0
		}
		if err := m.target.PrefillTokens(speculativeInt32s(prefix)); err != nil {
			runErr = core.E("native.speculativeModel.DFlash", "prefill verifier", err)
			return 0
		}
		logits, err := m.target.BoundaryLogits()
		if err != nil {
			runErr = core.E("native.speculativeModel.DFlash", "read verifier logits", err)
			return 0
		}
		id, err := greedyBF16Suppressed(logits, m.target.arch.Vocab, suppress)
		if err != nil {
			runErr = core.E("native.speculativeModel.DFlash", "select verifier token", err)
			return 0
		}
		return int(id)
	}
	tokens, stats := dflash.Generate(prompt, maxNew, proposer, next)
	res := AssistantGenerateResult{
		Tokens:            make([]int32, 0, len(tokens)),
		DraftTokens:       stats.ProposedTokens,
		AcceptedTokens:    stats.AcceptedTokens,
		RejectedTokens:    stats.ProposedTokens - stats.AcceptedTokens,
		TargetVerifyCalls: stats.Rounds,
		TargetCalls:       stats.TargetCalls,
		DraftCalls:        stats.Rounds,
	}
	for _, token := range tokens {
		id := int32(token)
		res.Tokens = append(res.Tokens, id)
		if runErr != nil || (sink != nil && !sink(id)) || speculativeTokenInSet(id, stop) {
			break
		}
	}
	return res, runErr
}

func speculativeInts(ids []int32) []int {
	out := make([]int, len(ids))
	for i, id := range ids {
		out[i] = int(id)
	}
	return out
}

func speculativeInt32s(ids []int) []int32 {
	out := make([]int32, len(ids))
	for i, id := range ids {
		out[i] = int32(id)
	}
	return out
}

// record folds one speculative run's counters into the metric surfaces the
// bench + trace read.
func (m *speculativeModel) record(res AssistantGenerateResult, promptLen int, dur time.Duration, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err != nil {
		m.lastErr = err
	}
	m.lastMetrics = inference.GenerateMetrics{
		PromptTokens:    promptLen,
		GeneratedTokens: len(res.Tokens),
		TotalDuration:   dur,
		DecodeDuration:  dur,
	}
	// The speculative run has no prefill/decode split (dur spans the whole
	// verify loop), so the decode rate is tokens over the whole run — filled
	// so Metrics() honours the interface's documented throughput field.
	if dur > 0 {
		m.lastMetrics.DecodeTokensPerSec = float64(len(res.Tokens)) / dur.Seconds()
	}
	spec := inference.SpeculativeMetrics{
		DraftTokenSchedule: res.DraftTokenSchedule,
		ProposedTokens:     res.DraftTokens,
		AcceptedTokens:     res.AcceptedTokens,
		RejectedTokens:     res.RejectedTokens,
		TargetVerifyCalls:  res.TargetVerifyCalls,
		TargetCalls:        res.TargetCalls,
		DraftCalls:         res.DraftCalls,
		WallDuration:       dur,
	}
	if spec.ProposedTokens > 0 {
		spec.AcceptanceRate = float64(spec.AcceptedTokens) / float64(spec.ProposedTokens)
	}
	m.lastSpec = spec
}

// SpeculativeMetrics reports the last run's draft/verify counters (inference
// .SpeculativeMetricsProvider) — what printMTPMetrics reads for the -draft bench
// line.
func (m *speculativeModel) SpeculativeMetrics() inference.SpeculativeMetrics {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastSpec
}

// Metrics reports the last generate run's token/timing counters.
func (m *speculativeModel) Metrics() inference.GenerateMetrics {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastMetrics
}

// Err reports the last run's error (OK when the last run succeeded).
func (m *speculativeModel) Err() core.Result {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.lastErr != nil {
		return core.Fail(m.lastErr)
	}
	return core.Ok(nil)
}

// ModelType is the target architecture identifier.
func (m *speculativeModel) ModelType() string { return m.modelType }

// Info is the target model's neutral metadata.
func (m *speculativeModel) Info() inference.ModelInfo { return m.info }

// Classify is not offered on the speculative path — speculation accelerates
// autoregressive generation, not batched prefill-only classification. Callers
// wanting Classify load the target plainly.
func (m *speculativeModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Fail(core.NewError("native.speculativeModel: Classify is not supported on the speculative path — load the target model plainly"))
}

// BatchGenerate is not offered on the speculative path — the MTP loop is a
// single-sequence draft/verify lane.
func (m *speculativeModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Fail(core.NewError("native.speculativeModel: BatchGenerate is not supported on the speculative path — load the target model plainly"))
}

// Close releases the drafter pair and the target session.
func (m *speculativeModel) Close() core.Result {
	var first error
	if m.pair != nil {
		if err := m.pair.Close(); err != nil {
			first = err
		}
	}
	if m.target != nil {
		if err := m.target.Close(); err != nil && first == nil {
			first = err
		}
	}
	if first != nil {
		return core.Fail(first)
	}
	return core.Ok(nil)
}

func (m *speculativeModel) setErr(err error) {
	m.mu.Lock()
	m.lastErr = err
	m.mu.Unlock()
}

// speculativeTokenInSet reports whether id is one of set — the terminator check
// the sink uses to blank a stop token's text.
func speculativeTokenInSet(id int32, set []int32) bool {
	return slices.Contains(set, id)
}

// speculativeSampleParams mirrors the plain engine path's SampleParams mapping
// so sampling behaviour through the pair is identical to plain decode.
func speculativeSampleParams(cfg inference.GenerateConfig) model.SampleParams {
	return model.SampleParams{
		Temperature:    cfg.Temperature,
		TopK:           cfg.TopK,
		TopP:           cfg.TopP,
		MinP:           cfg.MinP,
		RepeatPenalty:  cfg.RepeatPenalty,
		SuppressTokens: cfg.SuppressTokens,
	}
}
