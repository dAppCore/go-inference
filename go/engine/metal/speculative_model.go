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
	"dappco.re/go/inference/model/arch/Qwen/qwen35"
	zlabdflash "dappco.re/go/inference/model/arch/z-lab/dflash"
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
	target *ArchSession
	pair   *AssistantPair
	dflash *DFlashDrafter
	// zlab is set instead of pair+dflash for a z-lab-convention DFlash
	// pairing (assistant_dflash_zlab.go) — the real published-checkpoint
	// convention, which cannot load through AssistantPair at all (see that
	// file's header). pair stays nil in that case; every method that used to
	// read pair.TargetArch falls back to target.arch (identical data, the
	// same target checkpoint, just not re-derived through the AssistantPair
	// path).
	zlab       *zLabDFlashDrafter
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
	// TurboQuant live KV declines the MTP drafter pairing (v1): the verify
	// forward and the drafter's target-KV export both read the target cache as
	// bf16 rows. Refuse the combination loudly (never a silently-native pair).
	if tqMode, tqErr := parseTurboQuantCacheMode(cfg.CacheMode); tqErr != nil {
		return nil, tqErr
	} else if tqMode != nil {
		return nil, core.NewError("native.LoadSpeculativePair: -kv-cache turboquant declines the MTP drafter pairing (v1) — drop -draft or -kv-cache")
	}
	// maxLen <= 0 defers to the loader's checkpoint-window default.
	// draftBlock <= 0 defers to the arch-aware default resolved after the
	// pair attaches (blockDefaulted below); the z-lab path keeps the shipped 5.
	blockDefaulted := draftBlock <= 0
	if blockDefaulted {
		draftBlock = 5
	}
	// A gated-delta hybrid TARGET (Qwen 3.5/3.6) declines the MTP pairing. The drafter checkpoint
	// (qwen3_5_mtp) now parses as a real, weight-validated architecture — qwen35.ParseDrafterConfig /
	// Config.DrafterArch, #59 item 2 — but the pair LOAD itself is not wired, for two concrete
	// reasons named in full in docs/design-qwen-mtp-pair.md:
	//   - the drafter keeps its OWN attention state (never shares the target's K/V) — a different
	//     shape than this file's AssistantPair contract assumes, and mtp.MTPMethod has no honest
	//     value for it yet (MTPDraftModel is documented as sharing the target's K/V streams);
	//   - a genuinely accelerating pair needs the hybrid target's recurrent gated-delta state
	//     snapshotted/restored across a batched verify block; the primitive exists
	//     (session_state_blocks.go, #62) but has had no live caller since #50 retired
	//     model/composed's CloneState.
	// The retired composed engine used to serve this pairing (composed.LoadSpeculativePairDirs); it
	// is gone, and a factory pair route does not exist yet, so the load declines with the gap named
	// rather than mis-pairing.
	if mt := probeModelType(targetPath); qwen35.HybridModelType(mt) {
		return nil, core.NewError("native.LoadSpeculativePair: " + mt + " is a gated-delta hybrid — the qwen3_5_mtp drafter checkpoint now parses as a real architecture, but the factory pair load is not wired (the drafter's own-KV attention state has no engine forward, and target-side gated-delta rollback is unwired since #50 — see docs/design-qwen-mtp-pair.md); serve the base model without -draft")
	}
	target, err := LoadDir(targetPath, maxLen)
	if err != nil {
		return nil, core.E("native.LoadSpeculativePair", "load target checkpoint", err)
	}
	// The z-lab DFlash convention (every published checkpoint — unprefixed
	// layers.N.* tensors, no embedding/head/d2t of its own) cannot load
	// through LoadAssistantPairDirs below: it demands model.embed_tokens
	// .weight, which this convention architecturally lacks
	// (assistant_dflash_zlab.go's header, docs/design-dflash-forward.md §1/§3).
	// Recognise it FIRST, from config.json alone (no weights touched yet —
	// the same "read config only" posture DetectDFlashDraft uses), and take
	// the wholly separate z-lab load path. A draftPath that is not this
	// convention (including a non-DFlash drafter, or the speculators
	// convention) falls through to the existing flow untouched.
	if cfgStr, readErr := coreio.Local.Read(core.PathJoin(draftPath, "config.json")); readErr == nil {
		if _, ok := zlabdflash.ParseConfig([]byte(cfgStr)); ok {
			return loadZLabSpeculativePair(target, targetPath, draftPath, draftBlock)
		}
	}
	pair, err := LoadAssistantPairDirs(targetPath, draftPath)
	if err != nil {
		_ = target.Close()
		return nil, core.E("native.LoadSpeculativePair", "attach drafter", err)
	}
	// Arch-aware block default: a quant-MoE target's verify round carries a
	// higher fixed cost (router + expert gathers + the K-row head all amortise
	// per round), so a longer block pays there and only there — the 26B-A4B
	// pair sweep read 127.4/135.6/135.9/137.4/136.7/129.6 tok/s at blocks
	// 3..8 (knee at 6, plain 136.5), while the SAME sweep on E2B (dense-PLE)
	// regressed 96.6 -> 88.5 at 6. An explicit -draft-block always wins.
	if blockDefaulted && pair.TargetArch.HasMoE() {
		draftBlock = 6
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
// byte-identically to the plain engine path's Chat: the SAME detected template
// (engine.DetectChatTemplate over the target's marker dialect), the SAME
// large-variant thought-suppressor declaration (the geometry proxy
// NeedsThoughtChannelSuppressor keys on — gemma4 12B/26B/31B pre-close the
// empty thought channel on the thinking-off cue, E2B/E4B do not), and the SAME
// full-template render honouring the request's thinking flag.
//
// The pair previously rendered plain turns (engine.RenderChatTurns) with the
// suppressor unconditionally off, while the plain lane rendered the full
// declared template — on a 26B pair the two lanes conditioned on DIFFERENT
// prompt tokens (34 vs 38: the pair dropped the pre-closed thought channel),
// so pair-vs-plain greedy equality was broken at the prompt, before the verify
// machinery ever ran (#55).
func (m *speculativeModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	cfg := inference.ApplyGenerateOpts(opts)
	targetHeads := 0
	if m.pair != nil {
		targetHeads = m.pair.TargetArch.Heads
	} else if m.target != nil {
		// A z-lab DFlash pairing carries no AssistantPair (loadZLabSpeculativePair
		// leaves m.pair nil — see that function's header); the target session's
		// own arch is the same checkpoint pair.TargetArch would otherwise mirror.
		targetHeads = m.target.arch.Heads
	}
	prompt := speculativeChatPrompt(m.tok, m.turns, targetHeads, messages, cfg.EnableThinking)
	return m.speculate(ctx, m.tok.Encode(prompt), cfg)
}

// speculativeChatPrompt frames a speculative Chat request byte-identically to
// the plain engine path: the SAME detected template (DetectChatTemplate over
// the target's marker dialect), the SAME large-variant thought-suppressor
// geometry proxy (NeedsThoughtChannelSuppressor — heads >=
// largeVariantAttentionHeads), and the SAME full-template render honouring the
// request's thinking flag. Extracted so the #55 template-parity regression
// test pins the framing without loading a pair.
func speculativeChatPrompt(tok engine.TextTokenizer, turns engine.TurnTokens, targetHeads int, messages []inference.Message, enableThinking *bool) string {
	tmpl := engine.DetectChatTemplate(tok, turns, targetHeads >= largeVariantAttentionHeads)
	return engine.RenderChatPrompt(tmpl, messages, enableThinking)
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
		if m.zlab != nil {
			// A z-lab pairing has no AssistantPair to route through
			// speculativeMethodRoute (m.pair is nil — see
			// loadZLabSpeculativePair); it is unconditionally the
			// block-diffusion lane, the only method this drafter shape
			// supports.
			res, gerr = m.generateDFlashZLab(ids, maxNew, stop, cfg.SuppressTokens, sink)
		} else {
			res, gerr = speculativeMethodRoute(m.pair, mtp, func() (AssistantGenerateResult, error) {
				return m.generateDFlash(ids, maxNew, stop, cfg.SuppressTokens, sink)
			})
		}
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

// generateDFlashZLab drives the REAL z-lab block-diffusion proposer
// (model/arch/z-lab/dflash + dflash_zlab.go's DFlashZLabForward) through the
// same model-free, lossless DFlash verifier generateDFlash uses — the z-lab
// twin, for the drafter shape assistant_dflash_zlab.go loads. Both the aux
// source (ExtractAuxHiddensAllRaw) and the target verify oracle (next, below)
// deliberately replay the full prefix every call — the SAME posture
// generateDFlash's own next() already has (PrefillTokens always resets pos to
// 0), not a regression. Moving both to the incremental live-session taps
// (ExtractAuxHiddensLive) together is the named follow-up
// (docs/design-dflash-forward.md §7 item 5), not required for this lane's
// correctness gate.
func (m *speculativeModel) generateDFlashZLab(ids []int32, maxNew int, stop, suppress []int32, sink AssistantTokenSink) (AssistantGenerateResult, error) {
	if m.zlab == nil {
		return AssistantGenerateResult{}, core.NewError("native.speculativeModel: DFlash pair has no z-lab drafter")
	}
	maskEmbed, err := m.target.embedID(m.zlab.MaskTokenID())
	if err != nil {
		return AssistantGenerateResult{}, core.E("native.speculativeModel.DFlashZLab", "embed mask token", err)
	}
	maskEmbed = append([]byte(nil), maskEmbed...) // embedID hands back shared scratch — pin it
	prompt := speculativeInts(ids)
	var runErr error
	proposer := &zLabDFlashProposer{
		drafter:   m.zlab,
		maskEmbed: maskEmbed,
		source: func(context []int) ([]float32, int, []byte, bool) {
			prefix := speculativeInt32s(context)
			raw, err := ExtractAuxHiddensAllRaw(m.target, prefix, m.zlab.AuxLayers())
			if err != nil {
				runErr = core.E("native.speculativeModel.DFlashZLab", "extract aux hiddens", err)
				return nil, 0, nil, false
			}
			anchor, err := m.target.embedID(prefix[len(prefix)-1])
			if err != nil {
				runErr = core.E("native.speculativeModel.DFlashZLab", "embed anchor", err)
				return nil, 0, nil, false
			}
			return raw, len(prefix), append([]byte(nil), anchor...), true
		},
		head: func(hidden []byte) (int32, error) {
			logits, herr := m.target.headLogitsScratch(hidden, false)
			if herr != nil {
				return 0, herr
			}
			return greedyBF16Suppressed(logits, m.target.arch.Vocab, suppress)
		},
	}
	next := func(prefix []int) int {
		if runErr != nil {
			return 0
		}
		if err := m.target.PrefillTokens(speculativeInt32s(prefix)); err != nil {
			runErr = core.E("native.speculativeModel.DFlashZLab", "prefill verifier", err)
			return 0
		}
		logits, err := m.target.BoundaryLogits()
		if err != nil {
			runErr = core.E("native.speculativeModel.DFlashZLab", "read verifier logits", err)
			return 0
		}
		id, err := greedyBF16Suppressed(logits, m.target.arch.Vocab, suppress)
		if err != nil {
			runErr = core.E("native.speculativeModel.DFlashZLab", "select verifier token", err)
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
