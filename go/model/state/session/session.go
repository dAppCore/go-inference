// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	"context"
	"iter"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/kv/blockcache"
	"dappco.re/go/inference/model/bundle"
	"dappco.re/go/inference/model/spine"
	state "dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/agent"
	"dappco.re/go/inference/parser"
)

// Constant validation errors hoisted to package vars — each previously
// allocated a fresh core.NewError on the (rare but hot under churn)
// failure path. errModelSessionNil fires from every session-bound
// method when session is nil — 12 sites in this file alone.
var (
	errModelSessionNil       = core.NewError("mlx: model session is nil")
	errStateBundleNil        = core.NewError("mlx: state bundle is nil")
	errStateKVBlockBundleNil = core.NewError("mlx: State KV block bundle is nil")
	errNativeNoTokenPrefill  = core.NewError("mlx: native model session does not support token prefill")
	errNativeNoTokenAppend   = core.NewError("mlx: native model session does not support token append")
	errNativeNoKVRestore     = core.NewError("mlx: native model session does not support KV restore")
	errNativeNilSessionFork  = core.NewError("mlx: native model returned nil session fork")
	errKVSnapshotNil         = core.NewError("mlx: KV snapshot is nil")
)

type nativeSessionRestorer interface {
	RestoreKV(context.Context, *kv.Snapshot) error
}

type nativeSessionKVBlockRestorer interface {
	RestoreKVBlocks(context.Context, kv.BlockSource) error
}

type nativeSessionKVSnapshotterWithOptions interface {
	CaptureKVWithOptions(context.Context, kv.CaptureOptions) (*kv.Snapshot, error)
}

type nativeSessionChunkPrefiller interface {
	PrefillChunks(context.Context, iter.Seq[string]) error
}

type nativeSessionChunkAppender interface {
	AppendPromptChunks(context.Context, iter.Seq[string]) error
}

type nativeSessionTokenPrefiller interface {
	PrefillTokens(context.Context, []int32) error
}

type nativeSessionTokenAppender interface {
	AppendTokens(context.Context, []int32) error
}

// Session is a persistent model-state handle with retained KV cache.
// The root mlx package aliases it as ModelSession, so the public API is
// unchanged; subpackages use it directly without importing root.
type Session struct {
	session     inference.SessionHandle
	info        spine.ModelInfo
	tok         *spine.Tokenizer
	agentMemory *agent.WakeReport
}

// New wraps an already-created native session handle. It is the
// construction seam the root mlx package builds on (Model.NewSession
// probes the native factory, then calls New); tests construct a Session
// from a fake handle the same way.
//
//	sess := session.New(handle, m.Info(), m.Tokenizer())
func New(handle inference.SessionHandle, info spine.ModelInfo, tok *spine.Tokenizer) *Session {
	return &Session{session: handle, info: info, tok: tok}
}

// Prefill loads prompt into the retained session KV state.
func (s *Session) Prefill(prompt string) error {
	if s == nil || s.session == nil {
		return errModelSessionNil
	}
	return s.session.Prefill(context.Background(), prompt)
}

// PrefillChunks loads bounded prompt chunks into the retained session KV state.
func (s *Session) PrefillChunks(ctx context.Context, chunks iter.Seq[string]) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return errModelSessionNil
	}
	if prefiller, ok := s.session.(nativeSessionChunkPrefiller); ok {
		return prefiller.PrefillChunks(ctx, chunks)
	}
	return s.Prefill(spine.PromptChunksToString(chunks))
}

// PrefillTokens loads model-native token IDs into the retained session KV state.
func (s *Session) PrefillTokens(ctx context.Context, tokens []int32) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return errModelSessionNil
	}
	if prefiller, ok := s.session.(nativeSessionTokenPrefiller); ok {
		return prefiller.PrefillTokens(ctx, tokens)
	}
	return errNativeNoTokenPrefill
}

// AppendPrompt appends prompt tokens to the retained session KV state without
// replaying the existing prefix.
func (s *Session) AppendPrompt(prompt string) error {
	if s == nil || s.session == nil {
		return errModelSessionNil
	}
	return s.session.AppendPrompt(context.Background(), prompt)
}

// AppendPromptChunks appends bounded prompt chunks to the retained session KV
// state without replaying the existing prefix.
func (s *Session) AppendPromptChunks(ctx context.Context, chunks iter.Seq[string]) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return errModelSessionNil
	}
	if appender, ok := s.session.(nativeSessionChunkAppender); ok {
		return appender.AppendPromptChunks(ctx, chunks)
	}
	return s.AppendPrompt(spine.PromptChunksToString(chunks))
}

// AppendTokens appends model-native token IDs to the retained session KV state
// without replaying the existing prefix.
func (s *Session) AppendTokens(ctx context.Context, tokens []int32) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return errModelSessionNil
	}
	if appender, ok := s.session.(nativeSessionTokenAppender); ok {
		return appender.AppendTokens(ctx, tokens)
	}
	return errNativeNoTokenAppend
}

// Generate produces a buffered string from the retained session state.
func (s *Session) Generate(opts ...inference.GenerateOption) (string, error) {
	if s == nil || s.session == nil {
		return "", errModelSessionNil
	}
	cfg := inference.ApplyGenerateOpts(opts)
	filter := parser.NewProcessor(cfg.Thinking, spine.ParserHint(s.info))
	builder := core.NewBuilder()
	// Pre-grow the Builder backing slice — generations typically produce
	// hundreds of tokens of text. Skips the early 64 -> 128 -> 256 -> 512
	// -> 1024 doubling sequence of internal slice reallocations during
	// token streaming. Mirror of GenerateAndSleepAgentMemory's hint —
	// the per-conversation cost is the same on both API entry points.
	builder.Grow(1024)
	for tok := range s.session.Generate(context.Background(), cfg) {
		builder.WriteString(filter.Process(sessionParserTokenText(s.tok, tok)))
	}
	builder.WriteString(filter.Flush())
	if err := s.session.Err(); err != nil {
		return "", err
	}
	return builder.String(), nil
}

// GenerateStream streams tokens from the retained session state.
func (s *Session) GenerateStream(ctx context.Context, opts ...inference.GenerateOption) <-chan spine.Token {
	out := make(chan spine.Token)
	go func() {
		defer close(out)
		if s == nil || s.session == nil {
			return
		}
		if ctx == nil {
			ctx = context.Background()
		}
		cfg := inference.ApplyGenerateOpts(opts)
		filter := parser.NewProcessor(cfg.Thinking, spine.ParserHint(s.info))
		for tok := range s.session.Generate(ctx, cfg) {
			if ctx.Err() != nil {
				return
			}
			text := filter.Process(sessionParserTokenText(s.tok, tok))
			if text == "" {
				continue
			}
			select {
			case out <- spine.Token{ID: tok.ID, Value: text, Text: text}:
			case <-ctx.Done():
				return
			}
		}
		if text := filter.Flush(); text != "" {
			select {
			case out <- spine.Token{Value: text, Text: text}:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out
}

func sessionParserTokenText(tok *spine.Tokenizer, token inference.Token) string {
	if tok != nil {
		if text := tok.IDToken(token.ID); sessionParserControlToken(text) {
			return text
		}
	}
	return token.Text
}

func sessionParserControlToken(text string) bool {
	if text == "" {
		return false
	}
	// Every control marker begins with '<'. A single byte-scan for the
	// opening angle prunes the entire 14-pattern probe set on the dominant
	// "ordinary token text" path. Tokens flow through this function once
	// per emitted token during GenerateStream — the cheaper miss matters.
	open := core.Index(text, "<")
	if open < 0 {
		return false
	}
	// Trim leading prefix that cannot contain a marker — the markers begin
	// at the first '<', so further pattern scans only need the tail.
	tail := text[open:]
	return core.Contains(tail, "<|channel>") ||
		core.Contains(tail, "<channel|>") ||
		core.Contains(tail, "<start_of_turn>") ||
		core.Contains(tail, "<end_of_turn>") ||
		core.Contains(tail, "<think>") ||
		core.Contains(tail, "</think>") ||
		core.Contains(tail, "<thinking>") ||
		core.Contains(tail, "</thinking>") ||
		core.Contains(tail, "<thought>") ||
		core.Contains(tail, "</thought>") ||
		core.Contains(tail, "<reasoning>") ||
		core.Contains(tail, "</reasoning>") ||
		core.Contains(tail, "<analysis>") ||
		core.Contains(tail, "</analysis>")
}

// CaptureKV copies the current retained KV cache tensors to CPU memory.
func (s *Session) CaptureKV() (*kv.Snapshot, error) {
	return s.CaptureKVWithOptions(kv.CaptureOptions{})
}

// CaptureKVWithOptions copies the current retained KV cache tensors to CPU
// memory with explicit capture options.
func (s *Session) CaptureKVWithOptions(opts kv.CaptureOptions) (*kv.Snapshot, error) {
	if s == nil || s.session == nil {
		return nil, errModelSessionNil
	}
	var (
		snapshot *kv.Snapshot
		err      error
	)
	if snapshotter, ok := s.session.(nativeSessionKVSnapshotterWithOptions); ok {
		snapshot, err = snapshotter.CaptureKVWithOptions(context.Background(), opts)
	} else {
		snapshot, err = s.session.CaptureKV(context.Background())
	}
	if err != nil {
		return nil, err
	}
	// Return an owned snapshot. The engine merge's predecessor captured through
	// the per-engine snapshot converter, which deep-copied as a side effect; the
	// neutral handle may hand back a reference into engine-held state (the fake
	// does), so clone to preserve the "caller owns the capture" contract before
	// any RawKVOnly mutation.
	snapshot = snapshot.Clone()
	if opts.RawKVOnly {
		kv.DropFloat32(snapshot)
	}
	return snapshot, nil
}

// kv.Analyze captures and analyses the current retained KV state.
func (s *Session) AnalyzeKV() (*kv.Analysis, error) {
	snapshot, err := s.CaptureKV()
	if err != nil {
		return nil, err
	}
	return kv.Analyze(snapshot), nil
}

// SaveKV captures and writes the current retained KV state to path.
func (s *Session) SaveKV(path string) error {
	snapshot, err := s.CaptureKV()
	if err != nil {
		return err
	}
	return snapshot.Save(path)
}

// RestoreKV replaces the retained session state with a restorable KV snapshot.
func (s *Session) RestoreKV(snapshot *kv.Snapshot) error {
	if s == nil || s.session == nil {
		return errModelSessionNil
	}
	if snapshot == nil {
		return errKVSnapshotNil
	}
	restorer, ok := s.session.(nativeSessionRestorer)
	if !ok {
		return errNativeNoKVRestore
	}
	if err := restorer.RestoreKV(context.Background(), snapshot); err != nil {
		return err
	}
	s.agentMemory = nil
	return nil
}

// LoadKV reads a KV snapshot from path and restores it into the session.
func (s *Session) LoadKV(path string) error {
	snapshot, err := kv.Load(path)
	if err != nil {
		return err
	}
	return s.RestoreKV(snapshot)
}

// SaveKVToState captures and writes the current retained KV state to a State
// store.
func (s *Session) SaveKVToState(ctx context.Context, store state.Writer, opts kv.StateOptions) (state.ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	captureOpts := kv.CaptureOptions{}
	if opts.KVEncoding == kv.EncodingNative {
		captureOpts.RawKVOnly = true
	}
	snapshot, err := s.CaptureKVWithOptions(captureOpts)
	if err != nil {
		return state.ChunkRef{}, err
	}
	return snapshot.SaveState(ctx, store, opts)
}

// SaveKVToMemvid captures and writes the current retained KV state to the old
// memvid-named State store.
//
// Deprecated: use SaveKVToState.
func (s *Session) SaveKVToMemvid(ctx context.Context, store state.Writer, opts kv.MemvidOptions) (state.ChunkRef, error) {
	return s.SaveKVToState(ctx, store, opts)
}

// LoadKVFromState restores retained session state from a State KV snapshot.
func (s *Session) LoadKVFromState(ctx context.Context, store state.Store, ref state.ChunkRef) error {
	if ctx == nil {
		ctx = context.Background()
	}
	snapshot, err := kv.LoadFromState(ctx, store, ref)
	if err != nil {
		return err
	}
	return s.RestoreKV(snapshot)
}

// LoadKVFromMemvid restores retained session state from an old memvid-named
// State KV snapshot.
//
// Deprecated: use LoadKVFromState.
func (s *Session) LoadKVFromMemvid(ctx context.Context, store state.Store, ref state.ChunkRef) error {
	return s.LoadKVFromState(ctx, store, ref)
}

// SaveKVBlocksToState captures retained KV state and writes per-block State
// chunks.
func (s *Session) SaveKVBlocksToState(ctx context.Context, store state.Writer, opts kv.StateBlockOptions) (*kv.StateBlockBundle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return nil, errModelSessionNil
	}
	captureOpts := kv.CaptureOptions{}
	if opts.KVEncoding == kv.EncodingNative {
		captureOpts.RawKVOnly = true
	}
	blockSize := opts.BlockSize
	if blockSize <= 0 {
		blockSize = blockcache.DefaultBlockSize
	}
	// Trusted-prefix sleep: skip GPU->CPU capture of the blocks the parent
	// bundle already holds — the assembler grafts them by reference.
	captureOpts.BlockStartToken = kv.TrustedReuseBoundary(opts, blockSize)
	return kv.SaveStateBlocksFromStream(ctx, store, opts, func(yield func(kv.Block) (bool, error)) error {
		return s.session.RangeKVBlocks(ctx, blockSize, captureOpts, yield)
	})
}

// SaveKVBlocksToMemvid captures retained KV state and writes per-block KV
// chunks.
//
// Deprecated: use SaveKVBlocksToState.
func (s *Session) SaveKVBlocksToMemvid(ctx context.Context, store state.Writer, opts kv.MemvidBlockOptions) (*kv.MemvidBlockBundle, error) {
	return s.SaveKVBlocksToState(ctx, store, opts)
}

// LoadKVBlocksFromState restores retained session state from per-block State
// chunks.
func (s *Session) LoadKVBlocksFromState(ctx context.Context, store state.Store, bundle *kv.StateBlockBundle) error {
	return s.LoadKVPrefixBlocksFromState(ctx, store, bundle, 0)
}

// LoadKVBlocksFromMemvid restores retained session state from per-block KV
// chunks.
//
// Deprecated: use LoadKVBlocksFromState.
func (s *Session) LoadKVBlocksFromMemvid(ctx context.Context, store state.Store, bundle *kv.MemvidBlockBundle) error {
	return s.LoadKVBlocksFromState(ctx, store, bundle)
}

// LoadKVPrefixBlocksFromState restores a retained session state from the
// State KV blocks needed to cover prefixTokens. Native sessions consume the
// blocks as a stream, avoiding a full CPU-side assembled snapshot.
func (s *Session) LoadKVPrefixBlocksFromState(ctx context.Context, store state.Store, bundle *kv.StateBlockBundle, prefixTokens int) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return errModelSessionNil
	}
	if bundle == nil {
		return errStateKVBlockBundleNil
	}
	if restorer, ok := s.session.(nativeSessionKVBlockRestorer); ok {
		source, err := kv.StateBlockSource(ctx, store, bundle, prefixTokens)
		if err != nil {
			return err
		}
		if err := restorer.RestoreKVBlocks(ctx, source); err != nil {
			return err
		}
		s.agentMemory = nil
		return nil
	}
	loadOpts := kv.LoadOptions{}
	if bundle.KVEncoding == kv.EncodingNative {
		loadOpts.RawKVOnly = true
	}
	snapshot, err := kv.LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, prefixTokens, loadOpts)
	if err != nil {
		return err
	}
	return s.RestoreKV(snapshot)
}

// LoadKVPrefixBlocksFromMemvid restores a retained session state from the old
// memvid-named KV blocks needed to cover prefixTokens. Native sessions consume the
// blocks as a stream, avoiding a full CPU-side assembled snapshot.
//
// Deprecated: use LoadKVPrefixBlocksFromState.
func (s *Session) LoadKVPrefixBlocksFromMemvid(ctx context.Context, store state.Store, bundle *kv.MemvidBlockBundle, prefixTokens int) error {
	return s.LoadKVPrefixBlocksFromState(ctx, store, bundle, prefixTokens)
}

// RestoreBundle restores the session from a state bundle.
func (s *Session) RestoreBundle(b *bundle.Bundle) error {
	if b == nil {
		return errStateBundleNil
	}
	if err := bundle.CheckCompatibility(spine.ModelInfoToBundle(s.info), b); err != nil {
		return err
	}
	snapshot, err := b.Snapshot()
	if err != nil {
		return err
	}
	return s.RestoreKV(snapshot)
}

// RestoreBundleFromState restores the session from a state bundle whose KV is
// held in a State store.
func (s *Session) RestoreBundleFromState(ctx context.Context, b *bundle.Bundle, store state.Store) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if b == nil {
		return errStateBundleNil
	}
	if err := bundle.CheckCompatibility(spine.ModelInfoToBundle(s.info), b); err != nil {
		return err
	}
	snapshot, err := b.SnapshotFromState(ctx, store)
	if err != nil {
		return err
	}
	return s.RestoreKV(snapshot)
}

// RestoreBundleFromMemvid restores the session from a state bundle whose KV is
// held in the old memvid-named State cold storage.
//
// Deprecated: use RestoreBundleFromState.
func (s *Session) RestoreBundleFromMemvid(ctx context.Context, b *bundle.Bundle, store state.Store) error {
	return s.RestoreBundleFromState(ctx, b, store)
}

// LoadBundle reads a state bundle from path and restores it into the session.
func (s *Session) LoadBundle(path string) error {
	b, err := bundle.Load(path)
	if err != nil {
		return err
	}
	return s.RestoreBundle(b)
}

// Fork creates an independent session that starts from the same retained state.
func (s *Session) Fork() (*Session, error) {
	if s == nil || s.session == nil {
		return nil, errModelSessionNil
	}
	forked, err := s.session.Fork(context.Background())
	if err != nil {
		return nil, err
	}
	if forked == nil {
		return nil, errNativeNilSessionFork
	}
	return &Session{session: forked, info: s.info, tok: s.tok, agentMemory: agent.CloneWakeReport(s.agentMemory)}, nil
}

// Reset releases retained state and leaves the session ready for another prefill.
func (s *Session) Reset() {
	if s == nil || s.session == nil {
		return
	}
	s.session.Reset()
	s.agentMemory = nil
}

// Close releases retained session state.
func (s *Session) Close() error {
	if s == nil || s.session == nil {
		return nil
	}
	err := s.session.Close()
	s.session = nil
	return err
}

// Native returns the underlying native session handle, or nil for a nil
// Session. It is the accessor callers outside the package build on instead
// of reaching the unexported field (e.g. the root live tests that drive
// the raw block-restore path directly).
//
//	handle := sess.Native()
func (s *Session) Native() inference.SessionHandle {
	if s == nil {
		return nil
	}
	return s.session
}

// Valid reports whether the session holds a live native handle. It is the
// exported form of the `s == nil || s.session == nil` guard for callers
// outside the package (root FoldAgentMemory's exhausted-session check).
func (s *Session) Valid() bool {
	return s != nil && s.session != nil
}

// Err returns the last session error.
func (s *Session) Err() error {
	if s == nil || s.session == nil {
		return nil
	}
	return s.session.Err()
}
