// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"context"
	"iter"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/model"
)

// TokenModel is the loaded-decode-model surface a concrete engine must provide
// for [TextModel] to serve it: open a fresh retained [Session], and release the
// resident weights. The metal engine's *NativeTokenModel and the hip engine's
// token model satisfy it. The engine-specific model metadata (architecture,
// vocab, layer/hidden sizes, quant) is assembled by the engine and handed to
// [NewTextModel] as an inference.ModelInfo, so this surface stays minimal.
type TokenModel interface {
	// OpenEngineSession opens a fresh incremental decode session (empty KV cache)
	// as the engine [Session] the adapters drive.
	OpenEngineSession() (Session, error)
	// Close releases the model's resident weights.
	Close() error
}

// TextModel adapts a loaded engine [TokenModel] (+ its tokenizer) to
// inference.TextModel and inference.SessionFactory — the contract surface
// serving.NewMLXBackend and state/session.Session resolve against a registered
// backend. Each Generate/Chat opens a fresh incremental session (stateless per
// call); NewSession hands out a retained one for multi-turn conversation state.
type TextModel struct {
	tm        TokenModel
	tok       *tokenizer.Tokenizer
	modelType string
	info      inference.ModelInfo
	maxLen    int

	mu          sync.Mutex
	lastErr     core.Result
	lastMetrics inference.GenerateMetrics
}

var (
	_ inference.TextModel      = (*TextModel)(nil)
	_ inference.SessionFactory = (*TextModel)(nil)
	_ TrainerModel             = (*TextModel)(nil)
)

// NewTextModel wraps a loaded engine TokenModel as an inference.TextModel. tok
// is the model's tokenizer (text↔ids is the serve boundary the model carries
// once loaded); info + maxLen are the engine-built model metadata + context
// window; modelType is the architecture selector reported by ModelType.
func NewTextModel(tm TokenModel, tok *tokenizer.Tokenizer, modelType string, info inference.ModelInfo, maxLen int) *TextModel {
	return &TextModel{tm: tm, tok: tok, modelType: modelType, info: info, maxLen: maxLen, lastErr: core.Ok(nil)}
}

// openSession opens a fresh incremental decode session as the engine [Session]
// the adapters drive.
func (m *TextModel) openSession() (Session, error) {
	if m == nil || m.tm == nil {
		return nil, core.NewError("engine.TextModel: model is not initialised")
	}
	return m.tm.OpenEngineSession()
}

// OpenTrainer opens a retained LoRA SFT [Trainer] over the loaded model when the
// underlying engine [TokenModel] supports training ([TrainerModel]) — the forward
// that makes the head-LoRA train seam reachable through the neutral
// inference.LoadModel surface, so a training driver (dappco.re/go/inference/train)
// never needs the concrete engine type. Returns a clear error when the engine has
// no trainer, exactly as probing an unsupported capability should.
//
//	tr, err := loaded.(engine.TrainerModel).OpenTrainer(inference.TrainingConfig{...})
func (m *TextModel) OpenTrainer(cfg inference.TrainingConfig) (Trainer, error) {
	if m == nil || m.tm == nil {
		return nil, core.NewError("engine.TextModel: model is not initialised")
	}
	tm, ok := m.tm.(TrainerModel)
	if !ok {
		return nil, core.NewError("engine.TextModel: engine does not support training")
	}
	return tm.OpenTrainer(cfg)
}

// Generate streams tokens for a raw prompt (no chat template — Chat applies one).
func (m *TextModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.stream(ctx, m.encode(prompt), inference.ApplyGenerateOpts(opts))
}

// Chat renders the multi-turn conversation with the gemma turn template and
// streams the completion of a trailing model turn. A turn carrying images routes
// to the multimodal path when the loaded checkpoint has a vision tower; images
// against a text-only model are rejected rather than silently dropped.
func (m *TextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	cfg := inference.ApplyGenerateOpts(opts)
	if messagesHaveImages(messages) {
		if v, ok := m.tm.(VisionTokenModel); ok && v.AcceptsImageInput() {
			return m.chatMultimodal(ctx, messages, v, cfg)
		}
		return func(yield func(inference.Token) bool) {
			m.setErr(core.NewError("engine.TextModel.Chat: model does not accept image input"))
		}
	}
	return m.stream(ctx, m.encode(formatChatTurns(messages)), cfg)
}

func (m *TextModel) encode(prompt string) []int32 {
	if m == nil || m.tok == nil {
		return nil
	}
	return m.tok.Encode(prompt)
}

// decode is the STREAMING per-token decode: DecodeToken keeps the SentencePiece
// ▁ word boundary as a leading space (so concatenated stream chunks reassemble
// "hello world", not "helloworld") and preserves the gemma4 channel markers the
// ThinkingExtractor parses, while other specials stay invisible. DecodeOne is
// NOT this contract — it mirrors Decode-of-one and strips the boundary space,
// which is right only for a standalone label (see decodeLabel).
func (m *TextModel) decode(id int32) string {
	if m == nil || m.tok == nil {
		return ""
	}
	return m.tok.DecodeToken(id)
}

// decodeLabel decodes ONE token standalone — Decode([]int32{id}) semantics, the
// boundary space stripped ("▁positive" → "positive"). Classification results
// want the clean label; the stream loop must use decode instead.
func (m *TextModel) decodeLabel(id int32) string {
	if m == nil || m.tok == nil {
		return ""
	}
	return m.tok.DecodeOne(id)
}

// stream opens a fresh session, prefills ids, and yields decoded tokens up to
// the token budget. It bounds maxNew by the model's context window and honours
// stop tokens after yielding each token (so a stop token is still surfaced).
func (m *TextModel) stream(ctx context.Context, ids []int32, cfg inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		start := time.Now()
		if ctx == nil {
			ctx = context.Background()
		}
		if len(ids) == 0 {
			m.setErr(core.NewError("engine.TextModel.Generate: empty prompt after tokenisation"))
			return
		}
		sess, err := m.openSession()
		if err != nil {
			m.setErr(err)
			return
		}
		defer func() { _ = sess.Close() }()
		if err := sess.PrefillTokens(ids); err != nil {
			m.setErr(err)
			return
		}
		m.decodeFromPrefilled(ctx, sess, len(ids), cfg, start, yield)
	}
}

// decodeFromPrefilled runs the token budget over an ALREADY-prefilled session,
// yielding decoded tokens up to the budget and honouring stop tokens after each
// yield (so a stop token is still surfaced). It is the one decode loop shared by
// the text path (PrefillTokens) and the multimodal path (PrefillTokenEmbeddings)
// — the only difference upstream is how the prompt entered the KV cache.
// promptLen is the prompt token count (metrics); start is when the whole
// operation began.
func (m *TextModel) decodeFromPrefilled(ctx context.Context, sess Session, promptLen int, cfg inference.GenerateConfig, start time.Time, yield func(inference.Token) bool) {
	maxNew := cfg.MaxTokens
	if maxNew <= 0 {
		maxNew = 256
	}
	if sess.Pos()+maxNew > m.maxLen {
		maxNew = m.maxLen - sess.Pos()
	}
	if maxNew <= 0 {
		m.setErr(core.NewError("engine.TextModel.Generate: no room to generate in the context window"))
		return
	}
	// When the caller asked for a decode phase trace and the concrete engine
	// session can produce one, begin it before decoding and fold the aggregate
	// budget into metrics after — the neutral surface behind `generate -trace`.
	var stopTrace func() inference.DecodePhaseBudget
	if cfg.TraceTokenPhases {
		if tracer, ok := sess.(DecodePhaseTracer); ok {
			stopTrace = tracer.BeginDecodePhaseTrace()
		}
	}
	stop := m.stopTokens(cfg)
	count := 0
	emit := func(id int32) bool {
		if ctx.Err() != nil {
			return false
		}
		count++
		inStop := tokenInSet(id, stop)
		text := m.decode(id)
		if inStop {
			// A terminator's text is never content: gemma4 MLX snapshots ship
			// <end_of_turn> as a PLAIN vocab token (not in added_tokens), so
			// DecodeToken can't hide it — without this, every reply ended with
			// a literal "<end_of_turn>". The id still surfaces for trackers.
			text = ""
		}
		if !yield(inference.Token{ID: id, Text: text}) {
			return false
		}
		return !inStop
	}
	var gerr error
	if cfg.Temperature > 0 || cfg.MinP > 0 || cfg.RepeatPenalty > 1 {
		_, gerr = sess.GenerateSampledFromCacheEach(maxNew, stop, model.NewSampler(cfg.Seed), modelSampleParams(cfg), nil, emit)
	} else {
		// eosID -1: emit owns the stop decision (after yielding), so a stop token
		// is always surfaced and generation is bounded by maxNew.
		_, gerr = sess.GenerateFromCacheEach(maxNew, -1, emit)
	}
	m.setMetrics(promptLen, count, m.prefillSplit(start), start)
	if stopTrace != nil {
		budget := stopTrace()
		if budget.Tokens > 0 {
			m.setDecodePhases(&budget)
		}
	}
	if gerr != nil {
		m.setErr(gerr)
		return
	}
	if cerr := ctx.Err(); cerr != nil {
		m.setErr(cerr)
		return
	}
	m.setOK()
}

// prefillSplit is a coarse prefill/decode duration split — the conformance
// contract only reads GeneratedTokens, but real callers read the durations, so
// they are populated rather than left zero.
func (m *TextModel) prefillSplit(start time.Time) time.Duration {
	return time.Since(start)
}

// Classify runs prefill-only inference over each prompt and samples the single
// boundary token — the greedy next token is the classification. A fixture that
// cannot open a session returns a clean failure Result (callers skip).
func (m *TextModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	if ctx == nil {
		ctx = context.Background()
	}
	results := make([]inference.ClassifyResult, len(prompts))
	for i, prompt := range prompts {
		ids := m.encode(prompt)
		if len(ids) == 0 {
			return core.Fail(core.E("engine.TextModel.Classify", "empty prompt after tokenisation", nil))
		}
		sess, err := m.openSession()
		if err != nil {
			return core.Fail(core.E("engine.TextModel.Classify", "open session", err))
		}
		if err := sess.PrefillTokens(ids); err != nil {
			_ = sess.Close()
			return core.Fail(core.E("engine.TextModel.Classify", "prefill", err))
		}
		var got int32
		seen := false
		_, gerr := sess.GenerateFromCacheEach(1, -1, func(id int32) bool {
			got = id
			seen = true
			return false
		})
		_ = sess.Close()
		if gerr != nil || !seen {
			return core.Fail(core.E("engine.TextModel.Classify", "sample boundary token", gerr))
		}
		results[i] = inference.ClassifyResult{Token: inference.Token{ID: got, Text: m.decodeLabel(got)}}
	}
	return core.Ok(results)
}

// BatchGenerate runs autoregressive generation per prompt in sequence — the
// single-stream serve path means batching is a loop over the single path.
// Per-prompt errors ride in each BatchResult.
func (m *TextModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	if ctx == nil {
		ctx = context.Background()
	}
	cfg := inference.ApplyGenerateOpts(opts)
	results := make([]inference.BatchResult, len(prompts))
	for i, prompt := range prompts {
		var toks []inference.Token
		for tok := range m.stream(ctx, m.encode(prompt), cfg) {
			toks = append(toks, tok)
		}
		results[i] = inference.BatchResult{Tokens: toks}
		if r := m.Err(); !r.OK {
			if err, ok := r.Value.(error); ok {
				results[i].Err = err
			}
		}
	}
	return core.Ok(results)
}

// NewSession opens a fresh persistent conversation session over the loaded
// model, or nil when a session cannot be opened (SessionFactory).
func (m *TextModel) NewSession() inference.SessionHandle {
	sess, err := m.openSession()
	if err != nil {
		m.setErr(err)
		return nil
	}
	return NewSessionHandle(m, sess)
}

func (m *TextModel) ModelType() string { return m.modelType }

func (m *TextModel) Info() inference.ModelInfo { return m.info }

// CacheModeReporter is the optional seam a concrete engine [TokenModel]
// implements to declare which KV cache modes it honours as a load-time selector.
// An engine that runs a single automatic cache returns the descriptive name of
// that one mode (or nil); a future engine that honours fp16/q8/paged/… lists them
// so callers can validate a requested `-kv-cache` mode against real support.
type CacheModeReporter interface {
	SupportedCacheModes() []string
}

// Capabilities reports the loaded model's feature surface (inference
// .CapabilityReporter), starting from the interface-inferred base set and adding
// the concrete engine's supported KV cache modes when it declares them. This is
// the engine-agnostic seam `generate` consults to print an accurate `-kv-cache`
// note instead of a blanket "seam not yet exposed".
func (m *TextModel) Capabilities() inference.CapabilityReport {
	report := inference.TextModelCapabilities(inference.RuntimeIdentity{}, m)
	if m != nil && m.tm != nil {
		if reporter, ok := m.tm.(CacheModeReporter); ok {
			report.CacheModes = reporter.SupportedCacheModes()
		}
	}
	return report
}

func (m *TextModel) Metrics() inference.GenerateMetrics {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastMetrics
}

func (m *TextModel) Err() core.Result {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastErr
}

// Close releases the model's resident weights (delegated to the engine's
// TokenModel.Close — a no-op for in-memory weights; unmaps a directory-loaded
// checkpoint).
func (m *TextModel) Close() core.Result {
	if m == nil || m.tm == nil {
		return core.Ok(nil)
	}
	if err := m.tm.Close(); err != nil {
		return core.Fail(core.E("engine.TextModel.Close", "close token model", err))
	}
	return core.Ok(nil)
}

func (m *TextModel) stopTokens(cfg inference.GenerateConfig) []int32 {
	stop := append([]int32(nil), cfg.StopTokens...)
	if m.tok != nil {
		if eos := m.tok.EOS(); eos >= 0 {
			stop = append(stop, eos)
		}
	}
	return stop
}

func (m *TextModel) setErr(err error) {
	m.mu.Lock()
	m.lastErr = core.Fail(core.E("engine.TextModel.Generate", "generation failed", err))
	m.mu.Unlock()
}

func (m *TextModel) setOK() {
	m.mu.Lock()
	m.lastErr = core.Ok(nil)
	m.mu.Unlock()
}

// setDecodePhases attaches a traced decode phase budget to the last metrics.
// Called after setMetrics (which builds a fresh GenerateMetrics), so the budget
// rides the same metrics snapshot the caller reads via Metrics().
func (m *TextModel) setDecodePhases(budget *inference.DecodePhaseBudget) {
	m.mu.Lock()
	m.lastMetrics.DecodePhases = budget
	m.mu.Unlock()
}

func (m *TextModel) setMetrics(promptTokens, generated int, total time.Duration, start time.Time) {
	m.mu.Lock()
	m.lastMetrics = inference.GenerateMetrics{
		PromptTokens:    promptTokens,
		GeneratedTokens: generated,
		TotalDuration:   total,
		DecodeDuration:  time.Since(start),
	}
	m.mu.Unlock()
}

// formatChatTurns renders messages with the gemma turn template (user/model
// turns, a trailing open model turn to complete). Kept minimal: the serve path
// drives the same template pkg/model/gemma4/chat registers.
func formatChatTurns(messages []inference.Message) string {
	out := ""
	for _, msg := range messages {
		out += "<start_of_turn>" + chatTurnRole(msg.Role) + "\n" + msg.Content + "<end_of_turn>\n"
	}
	out += "<start_of_turn>model\n"
	return out
}

func chatTurnRole(role string) string {
	if role == "assistant" || role == "model" {
		return "model"
	}
	return "user"
}

// tokenInSet reports whether id is one of the stop tokens.
func tokenInSet(id int32, set []int32) bool {
	for _, s := range set {
		if id == s {
			return true
		}
	}
	return false
}
