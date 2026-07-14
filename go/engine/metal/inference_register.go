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
	"dappco.re/go/inference/model/composed"
)

func init() { inference.Register(metalBackend{}) }

// metalBackend is the inference.Backend for the no-cgo Metal engine. Name is the
// stable "metal" selector; Available reports whether the Metal device + kernels
// initialise (ensureInit); LoadModel loads a checkpoint directory as an
// inference.TextModel through the reactive native loader + tokenizer.
type metalBackend struct{}

var _ inference.Backend = metalBackend{}

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
	tm, err := LoadTokenModelDirWithConfig(path, maxLen, TokenModelLoadConfig{AdapterPath: cfg.AdapterPath})
	if err != nil {
		return core.Fail(core.E("native.metalBackend.LoadModel", "load token model", err))
	}
	tok, terr := tokenizer.LoadTokenizer(core.PathJoin(path, "tokenizer.json"))
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
		loaded.AttachTokenizer(tok)
		loaded.declaredStops = loadGenerationConfigStops(path)
		loaded.declaredSampling = loadGenerationConfigSamplingDefaults(path)
		return core.Ok(newNativeTextModel(loaded, modelType))
	case *composed.ComposedTokenModel:
		// A composed/hybrid checkpoint (host-f32 gated-delta + full attention, e.g. Qwen 3.6) is not the
		// native decode struct, so wrap it as an engine.TextModel through the composed serve source. The
		// source declares ChatML for the Qwen model_types, so the served checkpoint frames its own dialect
		// rather than gemma's. Size the context to the checkpoint window when the caller left it unset,
		// exactly as the native path's default does.
		serveLen := maxLen
		if serveLen <= 0 {
			serveLen = resolveDefaultContext(model.ProbeDirContextWindow(path))
		}
		info := inference.ModelInfo{
			Architecture: modelType,
			VocabSize:    loaded.Vocab(),
			NumLayers:    loaded.NumLayers(),
			HiddenSize:   loaded.HiddenSize(),
		}
		src := &composedTextModel{sm: loaded, tok: tok, modelType: modelType, numLayers: loaded.NumLayers()}
		return core.Ok(engine.NewTextModel(src, tok, modelType, info, serveLen))
	default:
		if closer, closeOK := tm.(interface{ Close() error }); closeOK {
			_ = closer.Close()
		}
		return core.Fail(core.E("native.metalBackend.LoadModel", "loader returned an unservable token model (neither a NativeTokenModel nor a composed hybrid)", nil))
	}
}

// composedTextModel adapts a loaded composed hybrid (a host-f32 model.SessionModel — Qwen 3.6's
// gated-delta / full-attention stack) to engine.TokenModel, so metalBackend.LoadModel returns it as an
// inference.TextModel through the shared engine.TextModel wrapper exactly like the native decode model.
// It additionally DECLARES its chat dialect (engine.ChatTemplateDeclarer): ChatML for the Qwen model_types
// (composed.ChatMLDialect), the gemma fallback for any other composed arch. The declared template WINS
// over engine.TextModel's tokenizer detection, so a Qwen checkpoint frames <|im_start|>/<|im_end|> turns
// even though its tokenizer carries no <|turn> marker.
type composedTextModel struct {
	sm        model.SessionModel
	tok       *tokenizer.Tokenizer
	modelType string
	numLayers int
}

var (
	_ engine.TokenModel           = (*composedTextModel)(nil)
	_ engine.ChatTemplateDeclarer = (*composedTextModel)(nil)
)

// OpenEngineSession opens a fresh composed decode session bridged to engine.Session. The bridge threads
// the composed model's own tested token loop (model.Generate*), so no decode logic is re-rolled here.
func (m *composedTextModel) OpenEngineSession() (engine.Session, error) {
	if m == nil || m.sm == nil {
		return nil, core.NewError("native.composedTextModel: model is not initialised")
	}
	return &composedEngineSession{sm: m.sm, arch: m.modelType, numLayers: m.numLayers}, nil
}

// Close releases the composed model's resident weights. The composed loader widens the checkpoint to f32
// and unmaps the shard mmap during the build, so nothing is held past load — Close is a no-op.
func (m *composedTextModel) Close() error { return nil }

// DeclaredChatTemplate declares the composed checkpoint's chat dialect (engine.ChatTemplateDeclarer): the
// ChatML family for the Qwen model_types (config-driven via composed.ChatMLDialect), the gemma turn
// template built from the tokenizer for any other composed arch. A declared template wins over
// engine.TextModel's detected gemma dialect, so the declaration is the precedence-setting seam.
func (m *composedTextModel) DeclaredChatTemplate() (engine.ChatTemplate, bool) {
	if m != nil && composed.ChatMLDialect(m.modelType) {
		return chatMLChatTemplate(), true
	}
	var tok *tokenizer.Tokenizer
	if m != nil {
		tok = m.tok
	}
	return engine.GemmaChatTemplate(engine.DetectTurnTokens(tok), false), true
}

// chatMLChatTemplate is the ChatML dialect as an engine.ChatTemplate: <|im_start|>role\n…<|im_end|> turns,
// an "assistant" generation cue, an in-place "system" turn (not folded), and the Qwen no-think block
// "<think>\n\n</think>\n\n" appended after the cue when thinking is off. It matches the reference ChatML
// rendering pinned in engine/chat_template_test.go.
func chatMLChatTemplate() engine.ChatTemplate {
	return engine.ChatTemplate{
		Open:          "<|im_start|>",
		Close:         "<|im_end|>",
		UserRole:      "user",
		AssistantRole: "assistant",
		SystemRole:    "system",
		Thinking:      &engine.ChatThinking{OffSuffix: "<think>\n\n</think>\n\n"},
		Stops:         []string{"<|im_end|>"},
	}
}

// composedEngineSession bridges a composed model.SessionModel to engine.Session for the serve path
// (engine.TextModel.Generate / Chat): PrefillTokens stores the prompt, and the two generate methods
// delegate to the composed model's own tested token loop (model.GenerateSampledWithStopTokensTransformEach),
// which opens the recurrent session and threads every layer's state.
//
// A composed hybrid decodes STATELESS-REPLAY: each generate opens a fresh recurrent composed.ComposedSession
// and re-prefills the whole token prefix, so the session never holds persistent KV/recurrent tensors — its
// COMPLETE resumable state is the token prefix. The recurrent conv/delta state (gated-delta layers) is not a
// transformer KV cache and has no kv.Snapshot Layers representation, so -state capture/restore is a
// TOKEN-PREFIX snapshot (Tokens, no Layers): on restore the deterministic host-f32 forward recomputes
// byte-identical recurrent state from the identical prefix, so a resumed conversation continues exactly as an
// unbroken one — the -state acceptance semantics, expressed through the state the arch actually has.
type composedEngineSession struct {
	sm        model.SessionModel
	prompt    []int32
	arch      string // config.json model_type — stamped on the snapshot (informational; restore is token-based)
	numLayers int    // composed block count — stamped on the snapshot for parity with the native path
}

var _ engine.Session = (*composedEngineSession)(nil)

// PrefillTokens stores the prompt tokens; the composed loop re-runs prefill inside the generate delegate
// (one prefill per stateless request), so the stored prompt is the whole retained state.
func (s *composedEngineSession) PrefillTokens(ids []int32) error {
	memWatermarkReset() // the operation's memory high-water starts here (#1843)
	s.prompt = append(s.prompt[:0], ids...)
	return nil
}

// AppendTokens extends the stored prompt (the recurrent state carries no separate replay-free append).
func (s *composedEngineSession) AppendTokens(ids []int32) error {
	memWatermarkReset() // a continuity turn's high-water starts here (#1843)
	s.prompt = append(s.prompt, ids...)
	return nil
}

// Pos is the retained prompt length — the budget engine.TextModel sizes generation against.
func (s *composedEngineSession) Pos() int { return len(s.prompt) }

// GenerateFromCacheEach greedily decodes up to maxNew tokens from the stored prompt, yielding each. eosID
// < 0 lets the caller own the stop decision (via yield returning false), matching engine.TextModel's emit.
// A zero-temperature sampler decodes greedily per token, reusing the composed model's tested stepwise loop.
func (s *composedEngineSession) GenerateFromCacheEach(maxNew, eosID int, yield func(int32) bool) ([]int32, error) {
	var stops []int32
	if eosID >= 0 {
		stops = []int32{int32(eosID)}
	}
	return model.GenerateSampledWithStopTokensTransformEach(s.sm, model.NewSampler(0), model.SampleParams{}, s.prompt, maxNew, stops, nil, yield)
}

// GenerateSampledFromCacheEach decodes up to maxNew tokens with the sampler + params, honouring stopTokens
// and the optional per-token transform — delegated wholesale to the composed model's tested sampled loop.
func (s *composedEngineSession) GenerateSampledFromCacheEach(maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	return model.GenerateSampledWithStopTokensTransformEach(s.sm, sampler, params, s.prompt, maxNew, stopTokens, transform, yield)
}

// CaptureKVWithOptions captures the session's resumable state as a TOKEN-PREFIX kv.Snapshot: the retained
// token prefix, no Layers (the composed hybrid holds no persistent KV/recurrent tensors — see the type doc).
// RestoreFromKV reinstates the prefix and the next generate recomputes byte-identical recurrent state, so
// the snapshot resumes the conversation exactly. opts are ignored: there is no KV cache to window or de-float.
func (s *composedEngineSession) CaptureKVWithOptions(kv.CaptureOptions) (*kv.Snapshot, error) {
	if s == nil {
		return nil, core.NewError("native.composedEngineSession.CaptureKV: nil session")
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

// RestoreFromKV reinstates a token-prefix snapshot captured by CaptureKVWithOptions: it takes the snapshot's
// token prefix as the session prefix; the next generate re-prefills it and recomputes byte-identical
// recurrent state (composed decode is deterministic host f32). Layers, if any, are ignored — a composed
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
	s.prompt = append(s.prompt[:0], snapshot.Tokens...)
	return nil
}

// Close releases the session state (none held beyond the stored prompt).
func (s *composedEngineSession) Close() error { return nil }
