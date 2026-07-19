// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// inference_register.go re-expresses go-mlx's register_native.go registration
// glue (which stays in go-mlx and dies with pkg/metal) against engine/metal's
// own loader. Importing this package self-registers the no-cgo Apple-GPU engine
// as inference backend "metal" — so serving.NewMLXBackend (WithBackend("metal"))
// and state/session.Session resolve a real model from go-inference alone, no
// go-mlx composition root. The registration is a plain init(): the concrete
// runtime package registers "metal", exactly as serving/backend_mlx.go documents.
package native

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/engine"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/arch/rwkv7"
	"dappco.re/go/inference/model/composed"
)

func init() { inference.Register(metalBackend{}) }

// metalBackend is the inference.Backend for the no-cgo Metal engine. Name is the
// stable "metal" selector; Available reports whether the Metal device + kernels
// initialise (ensureInit); LoadModel loads a checkpoint directory as an
// inference.TextModel through the reactive native loader + tokenizer.
// metalBackend additionally implements inference.SpeculativePairBackend
// (below), so the MTP pair-loading this engine already has (assistant-pair +
// composed-pair) is discoverable on the registered backend, not just injected
// by the composition root.
type metalBackend struct{}

var (
	_ inference.Backend                = metalBackend{}
	_ inference.SpeculativePairBackend = metalBackend{}
)

// Name is the registration/selection identifier.
func (metalBackend) Name() string { return "metal" }

// Available reports whether the Metal device and the compiled kernel library
// initialise on this host — the same gate the engine's own runtime tests use.
// Returns false (rather than panicking) on non-Apple hardware or a missing
// metallib, so inference.LoadModel fails cleanly with "not available".
func (metalBackend) Available() bool { return ensureInit() == nil }

// LoadModel reads the checkpoint directory at path and returns a ready
// inference.TextModel: the reactive native token model (dense / MoE / PLE, bf16
// or 4-bit) with the directory's tokenizer attached. WithContextLen sizes the
// KV cache (default 4096).
func (metalBackend) LoadModel(path string, opts ...inference.LoadOption) core.Result {
	cfg := inference.ApplyLoadOpts(opts)
	// maxLen <= 0 defers to the loader's checkpoint-window default
	// (resolveDefaultContext — the trained window capped at 32768).
	maxLen := cfg.ContextLen
	tm, err := LoadTokenModelDirWithConfig(path, maxLen, TokenModelLoadConfig{AdapterPath: cfg.AdapterPath, KVCacheMode: cfg.CacheMode})
	if err != nil {
		return core.Fail(core.E("native.metalBackend.LoadModel", "load token model", err))
	}
	hfTok, svcTok, terr := loadServeTokenizer(path, tm)
	if terr != nil {
		if closer, closeOK := tm.(interface{ Close() error }); closeOK {
			_ = closer.Close()
		}
		return core.Fail(core.E("native.metalBackend.LoadModel", "load tokenizer", terr))
	}
	// Stamp the checkpoint's real architecture (config.json model_type), not a
	// hardcoded "gemma4" — a second arch loaded through the metal backend must
	// self-report truthfully on ModelInfo.Architecture / ModelType.
	modelType := probeModelType(path)
	switch loaded := tm.(type) {
	case *NativeTokenModel:
		loaded.AttachTokenizer(hfTok)
		loaded.declaredStops = loadGenerationConfigStops(path)
		loaded.declaredSampling = loadGenerationConfigSamplingDefaults(path)
		loaded.defaultSystem = loadChatTemplateDefaultSystem(path)
		return core.Ok(newNativeTextModel(loaded, modelType))
	case model.SessionModel:
		// Neither the native decode struct: a config-composed hybrid (host-f32 gated-delta + full
		// attention, e.g. Qwen 3.6) OR a standalone-recurrent SSM (rwkv7, mamba2 — no attention, no KV
		// cache, a fixed per-layer state). Both wrap as an engine.TextModel through the shared
		// session-model serve source (sessionTextModel), which declares ChatML for the Qwen model_types
		// so a composed checkpoint frames its own dialect rather than gemma's, and the honest
		// plain-completion dialect for the standalone-recurrent archs (see DeclaredChatTemplate) rather
		// than a bracket template neither was tuned on. Size the context to the checkpoint window when
		// the caller left it unset, exactly as the native path's default does.
		serveLen := maxLen
		if serveLen <= 0 {
			serveLen = resolveDefaultContext(model.ProbeDirContextWindow(path))
		}
		info := inference.ModelInfo{Architecture: modelType, VocabSize: loaded.Vocab()}
		numLayers := 0
		if shaped, ok := loaded.(sessionModelShape); ok {
			info.NumLayers, info.HiddenSize = shaped.NumLayers(), shaped.HiddenSize()
			numLayers = shaped.NumLayers()
		}
		src := &sessionTextModel{sm: loaded, tok: svcTok, modelType: modelType, numLayers: numLayers}
		return core.Ok(engine.NewTextModel(src, svcTok, modelType, info, serveLen))
	default:
		if closer, closeOK := tm.(interface{ Close() error }); closeOK {
			_ = closer.Close()
		}
		return core.Fail(core.E("native.metalBackend.LoadModel", "loader returned an unservable token model (neither a NativeTokenModel nor a model.SessionModel)", nil))
	}
}

// loadServeTokenizer resolves the checkpoint directory's serving tokenizer: tokenizer.json when the
// checkpoint ships one (every transformer, composed/hybrid, and mamba2 checkpoint this engine loads
// carries one), else the loaded model's own arch-specific tokenizer when the arch declares one (rwkv7
// ships only a vocab .txt in a Python repr() grammar this port does not hand-parse —
// model/arch/rwkv7/tokenizer.go — so its World tokenizer's embedded canonical vocab is the fallback),
// else the original tokenizer.json error. hfTok is non-nil ONLY on the tokenizer.json path — the concrete
// type *NativeTokenModel.AttachTokenizer demands; svc is the engine.TextTokenizer every serve arm
// actually drives and is non-nil on any success (hfTok, upcast, on the tokenizer.json path; the arch
// tokenizer otherwise).
func loadServeTokenizer(path string, tm model.TokenModel) (hfTok *tokenizer.Tokenizer, svc engine.TextTokenizer, err error) {
	hfTok, terr := tokenizer.LoadTokenizer(core.PathJoin(path, "tokenizer.json"))
	if terr == nil {
		return hfTok, hfTok, nil
	}
	if _, ok := tm.(*rwkv7.RWKV7TokenModel); ok {
		wt, werr := rwkv7.NewWorldTokenizer()
		if werr != nil {
			return nil, nil, core.E("native.loadServeTokenizer", "load rwkv7 world tokenizer", werr)
		}
		return nil, wt, nil
	}
	return nil, nil, terr
}

// sessionModelShape is the optional capability a model.SessionModel MAY additionally implement to report
// its layer/hidden geometry for inference.ModelInfo: model/composed.ComposedTokenModel and the
// standalone-recurrent arch wrappers (rwkv7.RWKV7TokenModel, mamba2.MambaTokenModel) all do. A future
// model.SessionModel that doesn't leaves ModelInfo.NumLayers/HiddenSize honestly at 0 rather than
// guessing. Mirrors engine/hip's identical hipComposedModelShape probe.
type sessionModelShape interface {
	HiddenSize() int
	NumLayers() int
}

// compile-time proof rwkv7's World tokenizer satisfies the engine's serve-side tokenizer contract BY
// SHAPE (model/arch/rwkv7 never imports engine — AX-8; this assertion lives on the engine side of that
// boundary, where the import is legitimate).
var _ engine.TextTokenizer = (*rwkv7.WorldTokenizer)(nil)

// LoadSpeculativePair implements inference.SpeculativePairBackend: it loads
// targetPath + draftPath as one speculative-decode inference.TextModel by
// delegating straight to LoadSpeculativePair — this engine's own pair-loading
// entry point (assistant-pair MTP + the composed/hybrid arm), already proven
// by serve's MTP lane (serve wires the free function directly as its
// serving.SpeculativeLoader). This method is the seam that makes the SAME
// capability DISCOVERABLE on the registered "metal" backend, so an
// engine-neutral caller (train/tune's MTP block sweep) can find it via
// inference.Get/Default + a type assertion instead of importing this package.
func (metalBackend) LoadSpeculativePair(targetPath, draftPath string, draftBlock int, opts ...inference.LoadOption) (inference.TextModel, error) {
	return LoadSpeculativePair(targetPath, draftPath, draftBlock, opts...)
}

// sessionTextModel adapts a loaded model.SessionModel to engine.TokenModel, so metalBackend.LoadModel
// returns it as an inference.TextModel through the shared engine.TextModel wrapper exactly like the
// native decode model. Two families ride this ONE adapter: a config-composed hybrid (host-f32 — Qwen
// 3.6's gated-delta / full-attention stack) and a standalone-recurrent SSM (rwkv7, mamba2 — no attention,
// no KV cache, a fixed per-layer state) — neither is the native decode struct, and both already speak
// nothing but the neutral model.SessionModel contract, so one bridge serves both rather than two
// near-identical copies. It additionally DECLARES its chat dialect (engine.ChatTemplateDeclarer): ChatML
// for the Qwen model_types (composed.ChatMLDialect), the honest plain-completion dialect for the
// standalone-recurrent archs, the gemma fallback for anything else. The declared template WINS over
// engine.TextModel's tokenizer detection, so a Qwen checkpoint frames <|im_start|>/<|im_end|> turns even
// though its tokenizer carries no <|turn> marker. Named for what it bridges (any served
// model.SessionModel), not for one family — composedTextModel would now be a lie.
type sessionTextModel struct {
	sm        model.SessionModel
	tok       engine.TextTokenizer
	modelType string
	numLayers int
}

// sessionTextModel deliberately does NOT implement engine.LaneSetOpener — the
// absence IS the composed-in-laneSet verdict, not a gap. With no opener,
// engine.TextModel.BatchStepAvailable reports false and the scheduler keeps
// composed requests on their per-session paths instead of the continuous-
// batching laneSet. The original rationale (a host-f32 recurrence at every
// layer) is gone — the #18 campaign made the decode device-resident (device
// gated-delta state, device-KV attention, whole-token chaining) — but the
// verdict still holds for a narrower reason: laneSet advances K lanes through
// ONE shared RECORDED submission per round (lane_set.go), and the composed
// chain is re-ENCODED per token per session with its device state bound to
// that session's buffers. Until the composed lane records a replayable
// command stream (the CB-recording follow-up on the #18 board), there is no
// shared submission for laneSet to drive. Revisit when that lands. The
// standalone-recurrent archs share the same per-session re-encode shape, so
// the same verdict applies to them without a separate rationale.
var (
	_ engine.TokenModel           = (*sessionTextModel)(nil)
	_ engine.ChatTemplateDeclarer = (*sessionTextModel)(nil)
	_ engine.VisionTokenModel     = (*sessionTextModel)(nil)
)

// visionModel probes the underlying session model for the vision capability.
// model/composed implements the engine.VisionTokenModel method set BY SHAPE
// (it cannot import engine — AX-8), so the bridge forwards each method through
// this one assertion. A text-only model (or a build predating the vision
// tower, or any standalone-recurrent arch — none carries a vision tower)
// answers false and the serve path's image-refusal is unchanged.
func (m *sessionTextModel) visionModel() (engine.VisionTokenModel, bool) {
	if m == nil || m.sm == nil {
		return nil, false
	}
	v, ok := m.sm.(engine.VisionTokenModel)
	return v, ok
}

// AcceptsImageInput forwards the underlying model's vision declaration: true only
// when the loaded checkpoint shipped a vision tower (engine.VisionTokenModel).
func (m *sessionTextModel) AcceptsImageInput() bool {
	v, ok := m.visionModel()
	return ok && v.AcceptsImageInput()
}

// ImagePlaceholderTokenID forwards the underlying model's image token id; 0 (counts as
// no placeholder in the neutral splice) when the model carries no vision tower.
func (m *sessionTextModel) ImagePlaceholderTokenID() int32 {
	if v, ok := m.visionModel(); ok {
		return v.ImagePlaceholderTokenID()
	}
	return 0
}

// ImagePlaceholderBlock forwards the underlying model's placeholder block; empty when the
// model carries no vision tower (the splice never asks in that case — see
// AcceptsImageInput).
func (m *sessionTextModel) ImagePlaceholderBlock(softTokens int) string {
	if v, ok := m.visionModel(); ok {
		return v.ImagePlaceholderBlock(softTokens)
	}
	return ""
}

// ProjectImage forwards image projection through the underlying model's vision tower.
func (m *sessionTextModel) ProjectImage(image []byte) ([]byte, int, error) {
	v, ok := m.visionModel()
	if !ok {
		return nil, 0, core.NewError("native.sessionTextModel.ProjectImage: model carries no vision tower")
	}
	return v.ProjectImage(image)
}

// TokenEmbeddingsWithFeatures forwards the spliced-embedding build (text rows
// with projected features over the placeholder span) to the underlying model.
func (m *sessionTextModel) TokenEmbeddingsWithFeatures(ids []int32, imageFeatures, audioFeatures, videoFeatures []byte) ([][]byte, error) {
	v, ok := m.visionModel()
	if !ok {
		return nil, core.NewError("native.sessionTextModel.TokenEmbeddingsWithFeatures: model carries no vision tower")
	}
	return v.TokenEmbeddingsWithFeatures(ids, imageFeatures, audioFeatures, videoFeatures)
}

// OpenEngineSession opens a fresh decode session bridged to engine.Session. The bridge holds one open
// recurrent stepper and resumes it across prefill/append/generate (see composedEngineSession); the decode
// loop itself is the underlying model's own tested resume loop (model.GenerateSampledResumeEach), so no
// decode logic is re-rolled here — the same session type serves composed hybrids and standalone-recurrent
// archs alike, since both are driven purely through model.SessionModel.
func (m *sessionTextModel) OpenEngineSession() (engine.Session, error) {
	if m == nil || m.sm == nil {
		return nil, core.NewError("native.sessionTextModel: model is not initialised")
	}
	return &composedEngineSession{sm: m.sm, arch: m.modelType, numLayers: m.numLayers, pending: -1}, nil
}

// SessionsReusePrompts declares that these engine sessions implement
// engine.PromptReuseSession (PrefillTokensCached), so the resident-session prompt cache
// (#377) engages for chat — each turn forwards only its new tokens instead of
// re-prefilling the whole history.
func (m *sessionTextModel) SessionsReusePrompts() bool { return true }

// Close releases the underlying model's resident weights. Both the composed loader and the
// standalone-recurrent loaders widen the checkpoint to f32 and unmap the shard mmap during the build, so
// nothing is held past load — Close is a no-op.
func (m *sessionTextModel) Close() error { return nil }

// DeclaredChatTemplate declares the checkpoint's chat dialect (engine.ChatTemplateDeclarer): the ChatML
// family for the Qwen model_types (config-driven via composed.ChatMLDialect); a plain-completion dialect
// — no bracket markers, no role labels — for the standalone-recurrent archs (rwkv7, mamba2), because
// neither ChatML nor gemma is a template either checkpoint was tuned on (rwkv7-world ships its own
// "User: …\n\nAssistant: …\n\n" role-prefix jinja in tokenizer_config.json, which this engine's
// bracket-turn ChatTemplate primitive cannot express without extending renderChatTemplate's fixed
// Open+role+"\n"+content+Close shape — a v1 design call, tracked, not silently guessed; see
// plainCompletionChatTemplate); the gemma turn template built from the tokenizer for any other model. A
// declared template wins over engine.TextModel's detected gemma dialect, so the declaration is the
// precedence-setting seam.
func (m *sessionTextModel) DeclaredChatTemplate() (engine.ChatTemplate, bool) {
	if m != nil && composed.ChatMLDialect(m.modelType) {
		return chatMLChatTemplate(), true
	}
	if m != nil && standaloneRecurrentArch(m.modelType) {
		return plainCompletionChatTemplate(), true
	}
	return engine.GemmaChatTemplate(engine.DetectTurnTokens(m.tok), false), true
}

// standaloneRecurrentArch reports whether modelType is one of the standalone-recurrent SSM architectures
// (rwkv7, mamba2) that LoadModel routes through the generic model.SessionModel arm alongside the composed
// hybrids — neither attention nor a KV cache, a fixed per-layer state instead (see load.go).
func standaloneRecurrentArch(modelType string) bool {
	return modelType == "rwkv7" || modelType == "mamba2"
}

// plainCompletionChatTemplate is the v1 chat dialect for a standalone-recurrent checkpoint whose own
// chat convention this engine's ChatTemplate primitive cannot express (see DeclaredChatTemplate). Rather
// than pretend the checkpoint understands gemma's <start_of_turn> or ChatML's <|im_start|> — markers that
// do not exist in its vocabulary and would tokenise as literal junk text polluting the prompt — this
// declares NO bracket and NO role label: renderChatTemplate's fixed Open+role+"\n"+content+Close+"\n"
// shape collapses to "\n"+content+"\n" per turn (a blank-line-separated paragraph) and a bare "\n"
// generation cue — the plainest thing the existing primitive can express, honestly a v1 shortcut rather
// than the checkpoint's true dialect (teaching renderChatTemplate a role-prefix shape is the
// fuller-fidelity follow-up).
//
// NOTE decode/generate.RunGenerate's stateless path (the `lem generate` CLI, no -state) ALWAYS wraps its
// prompt as one user turn and calls TextModel.Chat — it is NOT the raw TextModel.Generate path despite
// the CLI's name, so THIS template renders there too (confirmed live: `lem generate -temp 0` on the
// RWKV7-Goose-World2.8-0.1B-HF checkpoint answers "A: Paris" — correct, just framed by this dialect
// rather than the checkpoint's own "User: …\n\nAssistant: …"). Only engine.TextModel.Generate itself
// (examples/pkg/generate, and the library-level model/arch/rwkv7 tests) bypasses chat templating
// entirely and reproduces the checkpoint's raw continuation byte-for-byte.
func plainCompletionChatTemplate() engine.ChatTemplate {
	return engine.ChatTemplate{
		Open:          "",
		Close:         "",
		UserRole:      "",
		AssistantRole: "",
		SystemRole:    "",
	}
}

// chatMLChatTemplate is the ChatML dialect as an engine.ChatTemplate: <|im_start|>role\n…<|im_end|> turns,
// an "assistant" generation cue, an in-place "system" turn (not folded), and the Qwen reasoning block
// after the cue — "<think>\n" opened for the model to fill when thinking is on, the pre-closed empty
// "<think>\n\n</think>\n\n" when it is off (Qwen3.5/3.6's default). Byte-identical to the checkpoint's
// chat_template.jinja for both thinking states (verified against transformers.apply_chat_template).
func chatMLChatTemplate() engine.ChatTemplate {
	return engine.ChatTemplate{
		Open:          "<|im_start|>",
		Close:         "<|im_end|>",
		UserRole:      "user",
		AssistantRole: "assistant",
		SystemRole:    "system",
		Thinking:      &engine.ChatThinking{OnSuffix: "<think>\n", OffSuffix: "<think>\n\n</think>\n\n"},
		Stops:         []string{"<|im_end|>"},
	}
}

// composedEngineSession bridges a model.SessionModel — a config-composed hybrid (composed.composedStepper)
// or a standalone-recurrent SSM (rwkv7, mamba2) — to engine.Session for the serve path
// (engine.TextModel.Generate / Chat) as a STATEFUL session (#25): it holds ONE open recurrent stepper
// across calls, driven purely through the neutral model.SessionModel/model.DecodeStepper contract, so
// the same session type serves either family unchanged. PrefillTokens batch-prefills the prompt through
// it at call time (so serve's prefill timing measures the real forward), AppendTokens forwards only the
// new tail, and the generate methods resume decoding from the last forwarded token's hidden — no
// per-request replay of the conversation prefix. PrefillTokensCached (engine.PromptReuseSession) extends
// a resident prefix in place, which is what removes the whole-history re-prefill from every chat turn.
//
// The recurrent state (gated-delta layers, or rwkv7/mamba2's per-layer WKV/SSM state) is not a
// transformer KV cache and has no kv.Snapshot Layers representation, so -state capture/restore stays a
// TOKEN-PREFIX snapshot (Tokens, no Layers): restore reinstates the prefix and the next generate
// recomputes byte-identical recurrent state from it (the decode is deterministic), so a resumed
// conversation continues exactly as an unbroken one. The prefix now includes generated tokens — a
// post-reply capture resumes WITH the reply, matching the arch session's semantics. Recurrent state
// cannot REWIND (it is cumulative, not per-token), so any prompt that does not extend the resident prefix
// re-prefills cold.
type composedEngineSession struct {
	sm        model.SessionModel
	arch      string // config.json model_type — stamped on the snapshot (informational; restore is token-based)
	numLayers int    // composed block count — stamped on the snapshot for parity with the native path

	st      model.DecodeStepper // the open recurrent stepper; nil until the first prefill (or after restore)
	hidden  []byte              // last forwarded token's output hidden — where the next decode resumes
	pending int32               // picked-but-unstepped final token of the last decode; -1 = none
	prompt  []int32             // the full token transcript in (or entering) the stepper, generated included

	// embRows holds a multimodal turn's spliced embedding rows (one per prompt
	// token, projected image features over the placeholder span) between
	// PrefillTokenEmbeddings and the generate that consumes them. Rows are NOT
	// token-replayable — while held, continuity appends and -state capture
	// refuse (the gemma4 metal lane's v1 bounds: an image turn is a stateless
	// turn, decoded through the one-shot embeddings loop, not the stepper).
	embRows [][]byte
}

var (
	_ engine.Session            = (*composedEngineSession)(nil)
	_ engine.VisionSession      = (*composedEngineSession)(nil)
	_ engine.PromptReuseSession = (*composedEngineSession)(nil)
)

// resetState drops the open stepper and its resume point (the retained prompt is the caller's
// business). The stepper's per-layer state is GC-managed (model.DecodeStepper contract), so
// dropping the reference is the release.
func (s *composedEngineSession) resetState() {
	s.st, s.hidden, s.pending = nil, nil, -1
	s.embRows = nil
}

// forwardBatch embeds ids and forwards them through the open stepper in one batch (the
// composed stepper's PrefillBatch; a per-token Step walk when a stepper lacks it), leaving
// s.hidden at the last token's output. A pending token from the previous decode enters the
// batch first — it is already part of s.prompt, just not yet in the stepper.
func (s *composedEngineSession) forwardBatch(ids []int32) error {
	batch := ids
	if s.pending >= 0 {
		batch = append([]int32{s.pending}, ids...)
		s.pending = -1
	}
	if len(batch) == 0 {
		return nil
	}
	embs := make([][]byte, len(batch))
	for i, id := range batch {
		emb, err := s.sm.Embed(id)
		if err != nil {
			return err
		}
		embs[i] = emb
	}
	if bp, ok := s.st.(model.BatchPrefillStepper); ok {
		hidden, err := bp.PrefillBatch(embs)
		if err != nil {
			return err
		}
		s.hidden = hidden
		return nil
	}
	for _, emb := range embs {
		hidden, err := s.st.Step(emb)
		if err != nil {
			return err
		}
		s.hidden = hidden
	}
	return nil
}

// prefillFresh replaces the session state with ids: fresh stepper, one batched prefill.
// An empty ids retains no state (matching the pre-stateful behaviour of erroring at the
// first generate rather than here).
func (s *composedEngineSession) prefillFresh(ids []int32) error {
	s.resetState()
	s.prompt = append(s.prompt[:0], ids...)
	if len(ids) == 0 {
		return nil
	}
	st, err := s.sm.OpenSession()
	if err != nil {
		return err
	}
	s.st = st
	if err := s.forwardBatch(ids); err != nil {
		s.resetState()
		return err
	}
	return nil
}

// ensureResident makes the retained prompt resident in an open stepper — the lazy
// re-prefill behind RestoreFromKV (and a cold AppendTokens): a token-prefix snapshot
// reinstates s.prompt only, and the first call that needs live state pays the one
// deterministic replay that recomputes it.
func (s *composedEngineSession) ensureResident() error {
	if s.st != nil {
		return nil
	}
	prompt := s.prompt
	s.prompt = nil
	return s.prefillFresh(prompt)
}

// PrefillTokens batch-prefills the prompt through a fresh recurrent stepper at call time —
// the timed prefill serve measures — and retains the stepper for the decode + any appends.
func (s *composedEngineSession) PrefillTokens(ids []int32) error {
	memWatermarkReset() // the operation's memory high-water starts here (#1843)
	return s.prefillFresh(ids)
}

// AppendTokens forwards ONLY the appended tail through the open stepper (the replay-free
// continuation). A session holding multimodal rows refuses: appended tokens have no spliced
// rows, so the row/token equality the embeddings replay depends on would silently break —
// an image turn is a stateless turn. With no live stepper (a restored session), the retained
// prefix becomes resident first, then the tail extends it.
func (s *composedEngineSession) AppendTokens(ids []int32) error {
	if len(s.embRows) > 0 {
		return core.NewError("native.composedEngineSession.AppendTokens: session holds multimodal prefill rows; an image turn does not continue")
	}
	memWatermarkReset() // a continuity turn's high-water starts here (#1843)
	if err := s.ensureResident(); err != nil {
		return err
	}
	if s.st == nil { // nothing retained and nothing appended stays stateless
		return s.prefillFresh(ids)
	}
	if err := s.forwardBatch(ids); err != nil {
		s.resetState() // the stepper advanced partially — unknown state must not serve
		return err
	}
	s.prompt = append(s.prompt, ids...)
	return nil
}

// PrefillTokensCached is the engine.PromptReuseSession entry the resident-session prompt
// cache drives (#377): ids that EXTEND the resident prefix keep the live recurrent state and
// forward only the divergent suffix, reporting the reused length; anything else re-prefills
// cold exactly as PrefillTokens (recurrent state cannot rewind — it is cumulative, so a
// shorter or divergent prompt has no partial reuse).
func (s *composedEngineSession) PrefillTokensCached(ids []int32) (int, error) {
	memWatermarkReset() // the operation's memory high-water starts here (#1843)
	reused := len(s.prompt)
	if s.st == nil || reused == 0 || len(s.embRows) > 0 || reused > len(ids) || !int32Prefix(ids, s.prompt) {
		return 0, s.prefillFresh(ids)
	}
	if s.pending >= 0 { // the pending token is in the prompt but not yet in the stepper
		reused--
	}
	tail := ids[len(s.prompt):]
	if err := s.forwardBatch(tail); err != nil {
		s.resetState()
		return 0, err
	}
	s.prompt = append(s.prompt, tail...)
	return reused, nil
}

// int32Prefix reports whether ids begins with prefix.
func int32Prefix(ids, prefix []int32) bool {
	if len(ids) < len(prefix) {
		return false
	}
	for i, v := range prefix {
		if ids[i] != v {
			return false
		}
	}
	return true
}

// PrefillTokenEmbeddings stores a multimodal turn's prompt: the token ids (position/budget accounting —
// Pos, stop handling) plus the spliced embedding rows the generate replays through the composed model's
// batch prefill (engine.VisionSession; rows come from TokenEmbeddingsWithFeatures). The rows ARE the
// retained state for this turn, consumed by the next generate through the one-shot embeddings loop —
// any live text-lane stepper is dropped (an image turn is a stateless turn).
func (s *composedEngineSession) PrefillTokenEmbeddings(ids []int32, embeddings [][]byte) error {
	if len(ids) != len(embeddings) {
		return core.NewError("native.composedEngineSession.PrefillTokenEmbeddings: token and embedding counts differ")
	}
	if len(ids) == 0 {
		return core.NewError("native.composedEngineSession.PrefillTokenEmbeddings: empty prompt")
	}
	memWatermarkReset() // the multimodal operation's memory high-water starts here (#1843)
	s.resetState()
	s.prompt = append(s.prompt[:0], ids...)
	s.embRows = embeddings
	return nil
}

// Pos is the retained transcript length (prompt + generated) — the budget engine.TextModel
// sizes generation against.
func (s *composedEngineSession) Pos() int { return len(s.prompt) }

// GenerateFromCacheEach greedily decodes up to maxNew tokens from the resident state, yielding each. eosID
// < 0 lets the caller own the stop decision (via yield returning false), matching engine.TextModel's emit.
// A zero-temperature sampler decodes greedily per token through the same resume loop as the sampled path.
func (s *composedEngineSession) GenerateFromCacheEach(maxNew, eosID int, yield func(int32) bool) ([]int32, error) {
	var stops []int32
	if eosID >= 0 {
		stops = []int32{int32(eosID)}
	}
	return s.GenerateSampledFromCacheEach(maxNew, stops, model.NewSampler(0), model.SampleParams{}, nil, yield)
}

// GenerateSampledFromCacheEach resumes decoding from the resident recurrent state — no prefix
// replay; the decode starts at the last forwarded token's hidden. Generated tokens join the
// retained transcript (Pos grows; a later capture or append continues after the reply). A
// multimodal turn (PrefillTokenEmbeddings) decodes through the one-shot embeddings loop
// instead — its rows cannot resume, so that turn stays stateless.
func (s *composedEngineSession) GenerateSampledFromCacheEach(maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	if len(s.embRows) > 0 {
		return model.GenerateSampledFromEmbeddingsEach(s.sm, sampler, params, s.prompt, s.embRows, maxNew, stopTokens, transform, yield)
	}
	if len(s.prompt) == 0 {
		return nil, core.NewError("model.Generate: empty prompt")
	}
	if err := s.ensureResident(); err != nil {
		return nil, err
	}
	gen, resume, err := model.GenerateSampledResumeEach(s.sm, s.st, model.SessionResume{Hidden: s.hidden, PendingID: s.pending}, sampler, params, maxNew, stopTokens, transform, yield)
	if err != nil {
		s.resetState() // the stepper may have advanced mid-decode — unknown state must not serve
		return nil, err
	}
	s.hidden, s.pending = resume.Hidden, resume.PendingID
	s.prompt = append(s.prompt, gen...)
	return gen, nil
}

// CaptureKVWithOptions captures the session's resumable state as a TOKEN-PREFIX kv.Snapshot: the retained
// token prefix (generated tokens included), no Layers — the recurrent conv/delta state has no kv.Snapshot
// representation, and the prefix alone reproduces it (see the type doc). RestoreFromKV reinstates the
// prefix and the next generate recomputes byte-identical recurrent state, so the snapshot resumes the
// conversation exactly. opts are ignored: there is no KV cache to window or de-float.
func (s *composedEngineSession) CaptureKVWithOptions(kv.CaptureOptions) (*kv.Snapshot, error) {
	if s == nil {
		return nil, core.NewError("native.composedEngineSession.CaptureKV: nil session")
	}
	if len(s.embRows) > 0 {
		// A token-prefix snapshot cannot reproduce spliced image rows on
		// restore (replay re-embeds ids, losing the projected features) — so a
		// multimodal turn refuses capture rather than resuming corrupted.
		return nil, core.NewError("native.composedEngineSession.CaptureKV: session holds multimodal prefill rows; token-prefix capture would drop them")
	}
	if len(s.prompt) == 0 {
		return nil, core.NewError("native.composedEngineSession.CaptureKV: empty session (nothing prefilled)")
	}
	return &kv.Snapshot{
		Version:      kv.SnapshotVersion,
		Architecture: s.arch,
		Tokens:       append([]int32(nil), s.prompt...),
		TokenOffset:  len(s.prompt),
		NumLayers:    s.numLayers,
	}, nil
}

// RangeKVBlocks streams the retained token prefix as contiguous TOKEN-ONLY blocks — the serve-continuity
// sleep lane (SaveKVBlocksToState → SaveStateBlocksFromStream). A composed hybrid holds no persistent KV
// cache to stream as K/V slabs, but its complete resumable state IS the token prefix (see the type doc), so
// each block carries a token-prefix kv.Snapshot (Tokens, no Layers). The block-load reassembly folds the
// tokens back into a Tokens-only snapshot and RestoreFromKV re-prefills it, resuming byte-identically.
//
// Tiling mirrors ArchSession's trusted-prefix contract (StateBlockSourceFrom / kvBlockFromStateBlock) so a
// multi-turn re-sleep with prefix reuse tiles against grafted parent blocks: blocks partition [0, position)
// on the uniform blockSize grid (boundaries 0, blockSize, 2·blockSize, …, position); block k covers
// [k·blockSize, min((k+1)·blockSize, position)). opts.BlockStartToken skips whole blocks ending at or before
// the trusted boundary, and yielded Index / TokenStart stay ABSOLUTE in that grid — so the first emitted
// block's Index equals the grafted parent's block count and its TokenStart continues where the parent ended,
// keeping the assembled bundle contiguous from index 0, token 0.
func (s *composedEngineSession) RangeKVBlocks(blockSize int, opts kv.CaptureOptions, yield func(kv.Block) (bool, error)) error {
	if s == nil {
		return core.NewError("native.composedEngineSession.RangeKVBlocks: nil session")
	}
	if yield == nil {
		return core.NewError("native.composedEngineSession.RangeKVBlocks: nil yield")
	}
	if blockSize <= 0 {
		return core.NewError("native.composedEngineSession.RangeKVBlocks: block size must be > 0")
	}
	if opts.BlockStartToken < 0 {
		return core.NewError("native.composedEngineSession.RangeKVBlocks: block start token must be >= 0")
	}
	if len(s.embRows) > 0 {
		// Same refusal as CaptureKVWithOptions: token blocks cannot carry the
		// spliced image rows, so a multimodal turn never sleeps.
		return core.NewError("native.composedEngineSession.RangeKVBlocks: session holds multimodal prefill rows; token-prefix blocks would drop them")
	}
	position := len(s.prompt)
	if position == 0 {
		return core.NewError("native.composedEngineSession.RangeKVBlocks: empty session (nothing prefilled)")
	}
	// totalBlocks tiles [0, position) on the uniform blockSize grid — the last block is a partial when
	// position is not a blockSize multiple.
	totalBlocks := (position + blockSize - 1) / blockSize
	// firstBlockIndex skips whole blocks whose end lands at or before the trusted boundary, exactly as
	// ArchSession.stateBlockPlan advances firstBlock while boundaries[firstBlock+1] <= startToken. The
	// serve path always passes a block-aligned boundary below position, but mirroring the loop keeps the
	// contract exact for a mid-block boundary (the block spanning it is re-emitted whole).
	firstBlockIndex := 0
	for firstBlockIndex < totalBlocks && min((firstBlockIndex+1)*blockSize, position) <= opts.BlockStartToken {
		firstBlockIndex++
	}
	for index := firstBlockIndex; index < totalBlocks; index++ {
		start := index * blockSize
		end := min(start+blockSize, position)
		block := kv.Block{
			Index:      index,
			TokenStart: start,
			TokenCount: end - start,
			Snapshot: &kv.Snapshot{
				Version:      kv.SnapshotVersion,
				Architecture: s.arch,
				Tokens:       append([]int32(nil), s.prompt[start:end]...),
				TokenOffset:  end,
				SeqLen:       end - start,
				NumLayers:    s.numLayers,
			},
		}
		ok, err := yield(block)
		if err != nil || !ok {
			return err
		}
	}
	return nil
}

// RestoreFromKV reinstates a token-prefix snapshot captured by CaptureKVWithOptions: it takes the
// snapshot's token prefix as the session prefix and drops any live stepper; the first call that
// needs live state (generate/append) re-prefills the prefix once and recomputes byte-identical
// recurrent state (composed decode is deterministic). Layers, if any, are ignored — a composed
// snapshot never carries KV slabs.
func (s *composedEngineSession) RestoreFromKV(_ context.Context, snapshot *kv.Snapshot) error {
	if s == nil {
		return core.NewError("native.composedEngineSession.RestoreFromKV: nil session")
	}
	if snapshot == nil {
		return core.NewError("native.composedEngineSession.RestoreFromKV: nil snapshot")
	}
	if len(snapshot.Tokens) == 0 {
		return core.NewError("native.composedEngineSession.RestoreFromKV: snapshot carries no token prefix")
	}
	s.resetState()
	s.prompt = append(s.prompt[:0], snapshot.Tokens...)
	return nil
}

// Close releases the session state: the stepper reference is dropped (its per-layer state is
// GC-managed) and the retained transcript cleared.
func (s *composedEngineSession) Close() error {
	s.resetState()
	s.prompt = nil
	return nil
}
