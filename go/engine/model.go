// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"context"
	"iter"
	"slices"
	"strings"
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
	turns     TurnTokens

	mu          sync.Mutex
	lastErr     core.Result
	lastMetrics inference.GenerateMetrics
}

var (
	_ inference.TextModel      = (*TextModel)(nil)
	_ inference.SessionFactory = (*TextModel)(nil)
	_ TrainerModel             = (*TextModel)(nil)
)

// TurnTokens is the chat-turn marker dialect the loaded checkpoint was tuned
// on. Gemma 4 RENAMED the turn markers to <|turn>/<turn|> (reusing gemma3's
// token ids 105/106 under new spellings — <start_of_turn> is no longer in its
// vocab and tokenises as plain text); gemma3-era checkpoints keep
// <start_of_turn>/<end_of_turn>. Rendering the wrong dialect ships the turn
// structure as literal text: ~17 junk prompt tokens per turn pair, the reply
// polluted with a literal "<end_of_turn>" string, and measurable instruction
// damage (off-template E2B mangles "reverse 'sovereign'").
type TurnTokens struct {
	Open  string // opens a turn, followed by the role and \n
	Close string // closes a turn
}

// DetectTurnTokens picks the chat-turn dialect from the tokenizer's vocab: a
// checkpoint that carries <|turn> as a token is gemma4-templated; anything
// else keeps the legacy <start_of_turn> template.
func DetectTurnTokens(tok *tokenizer.Tokenizer) TurnTokens {
	if tok != nil {
		if _, ok := tok.TokenID("<|turn>"); ok {
			return TurnTokens{Open: "<|turn>", Close: "<turn|>"}
		}
	}
	return TurnTokens{Open: "<start_of_turn>", Close: "<end_of_turn>"}
}

// turnTokens is the model's detected turn dialect, defaulting a zero-value
// TextModel (no tokenizer seen) to the legacy template.
func (m *TextModel) turnTokens() TurnTokens {
	if m == nil || m.turns.Open == "" {
		return TurnTokens{Open: "<start_of_turn>", Close: "<end_of_turn>"}
	}
	return m.turns
}

// StopTokenDeclarer is the optional [TokenModel] capability for checkpoints
// that declare their stop set (generation_config.json eos_token_id — gemma4
// declares [<eos>, <turn|>, <|tool_response>]). [TextModel] folds the declared
// ids into every generation's stop set, so a tuned model's turn and tool
// boundaries terminate decoding exactly as the reference stack does.
type StopTokenDeclarer interface {
	DeclaredStopTokens() []int32
}

// NewTextModel wraps a loaded engine TokenModel as an inference.TextModel. tok
// is the model's tokenizer (text↔ids is the serve boundary the model carries
// once loaded); info + maxLen are the engine-built model metadata + context
// window; modelType is the architecture selector reported by ModelType. The
// chat-turn dialect is detected from the tokenizer's vocab.
func NewTextModel(tm TokenModel, tok *tokenizer.Tokenizer, modelType string, info inference.ModelInfo, maxLen int) *TextModel {
	return &TextModel{tm: tm, tok: tok, modelType: modelType, info: info, maxLen: maxLen, turns: DetectTurnTokens(tok), lastErr: core.Ok(nil)}
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
	return m.stream(ctx, m.encode(formatChatPrompt(m.turnTokens(), messages, cfg.EnableThinking)), cfg)
}

// FormatChatPrompt renders a fresh multi-turn prompt with the model's turn
// template — byte-identical to the serve path's framing (Chat above encodes the
// same formatChatTurns output). The durable -state loop calls this to open a
// fresh session so a stateful first turn is framed exactly like a stateless
// serve request: FormatChatPrompt([{user, "hi"}]) ->
// "<|turn>user\nhi<turn|>\n<|turn>model\n" on a gemma4 checkpoint
// (<start_of_turn> spelling on gemma3-era vocabs).
func (m *TextModel) FormatChatPrompt(messages []inference.Message) string {
	return formatChatPrompt(m.turnTokens(), messages, nil)
}

// FormatChatContinuation renders a woken-session turn with no replay of the
// retained history: it closes the model turn the restored KV prefix ends on
// (the prior answer, left open when generation stopped), appends the new user
// turn, and reopens the assistant header. The leading turn-close + \n is the
// close — FormatChatContinuation([{user, "and now?"}]) ->
// "<turn|>\n<|turn>user\nand now?<turn|>\n<|turn>model\n" on gemma4 — so the
// model resumes on a well-formed turn boundary rather than the raw prompt
// bleeding into its own open turn.
func (m *TextModel) FormatChatContinuation(messages []inference.Message) string {
	turns := m.turnTokens()
	return turns.Close + "\n" + formatChatTurns(turns, messages)
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
// model, or nil when a session cannot be opened (SessionFactory) — including
// on a nil receiver, mirroring the other capability probes.
func (m *TextModel) NewSession() inference.SessionHandle {
	if m == nil {
		return nil
	}
	sess, err := m.openSession()
	if err != nil {
		m.setErr(err)
		return nil
	}
	return NewSessionHandle(m, sess)
}

// ModelType reports the architecture selector; empty on a nil receiver, like
// Close/Capabilities tolerate one.
func (m *TextModel) ModelType() string {
	if m == nil {
		return ""
	}
	return m.modelType
}

// Info reports the loaded model's neutral metadata; zero on a nil receiver.
func (m *TextModel) Info() inference.ModelInfo {
	if m == nil {
		return inference.ModelInfo{}
	}
	return m.info
}

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

// Metrics reports the last generate run's token/timing counters; zero on a
// nil receiver.
func (m *TextModel) Metrics() inference.GenerateMetrics {
	if m == nil {
		return inference.GenerateMetrics{}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastMetrics
}

// Err reports the last run's error state; an uninitialised-model failure on a
// nil receiver rather than a panic.
func (m *TextModel) Err() core.Result {
	if m == nil {
		return core.Fail(core.NewError("engine.TextModel: model is not initialised"))
	}
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
		// The turn-close marker ends an assistant turn on chat-tuned checkpoints
		// (gemma4 declares <turn|> in its generation_config eos set; tuned models
		// emit it instead of <eos>). Without it a chat reply never terminates and
		// generation runs to the token budget.
		if id, ok := m.tok.TokenID(m.turnTokens().Close); ok && !tokenInSet(id, stop) {
			stop = append(stop, id)
		}
	}
	// A checkpoint-declared stop set (generation_config eos_token_id) outranks
	// the derived defaults — gemma4 adds <|tool_response>, stopping the model
	// before it hallucinates a tool's output.
	if d, ok := m.tm.(StopTokenDeclarer); ok {
		for _, id := range d.DeclaredStopTokens() {
			if id >= 0 && !tokenInSet(id, stop) {
				stop = append(stop, id)
			}
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

// formatChatTurns renders messages with the model's turn template (user/model
// turns in the detected marker dialect, a trailing open model turn to
// complete).
func formatChatTurns(turns TurnTokens, messages []inference.Message) string {
	var out strings.Builder
	for _, msg := range messages {
		out.WriteString(turns.Open + chatTurnRole(msg.Role) + "\n" + msg.Content + turns.Close + "\n")
	}
	out.WriteString(turns.Open + "model\n")
	return out.String()
}

// formatChatPrompt renders the full chat prompt: on the gemma4 dialect a
// leading system turn carries the thinking marker and/or a first system
// message, matching the checkpoint's chat_template.jinja byte-for-byte —
//
//	thinking only:  <|turn>system\n<|think|>\n<turn|>\n<|turn>user\n…
//	system+think:   <|turn>system\n<|think|>\nCONTENT<turn|>\n<|turn>user\n…
//	system only:    <|turn>system\nCONTENT<turn|>\n<|turn>user\n…
//
// The trained template is the thinking SWITCH: the <|think|> line in the first
// system turn is how a request turns reasoning on. Legacy-dialect models have
// no system-turn concept — system messages keep rendering as user turns and
// the prompt is exactly formatChatTurns.
func formatChatPrompt(turns TurnTokens, messages []inference.Message, enableThinking *bool) string {
	if turns.Open != "<|turn>" {
		return formatChatTurns(turns, messages)
	}
	thinking := enableThinking != nil && *enableThinking
	sysFirst := len(messages) > 0 && chatSystemRole(messages[0].Role)
	if !thinking && !sysFirst {
		return formatChatTurns(turns, messages)
	}
	var out strings.Builder
	out.WriteString(turns.Open + "system\n")
	if thinking {
		out.WriteString("<|think|>\n")
	}
	rest := messages
	if sysFirst {
		out.WriteString(strings.TrimSpace(messages[0].Content))
		rest = messages[1:]
	}
	out.WriteString(turns.Close + "\n")
	return out.String() + formatChatTurns(turns, rest)
}

// chatSystemRole reports whether role opens the leading system turn on the
// gemma4 template (its jinja accepts both spellings).
func chatSystemRole(role string) bool {
	return role == "system" || role == "developer"
}

func chatTurnRole(role string) string {
	if role == "assistant" || role == "model" {
		return "model"
	}
	return "user"
}

// tokenInSet reports whether id is one of the stop tokens.
func tokenInSet(id int32, set []int32) bool {
	return slices.Contains(set, id)
}
