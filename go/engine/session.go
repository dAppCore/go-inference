// SPDX-Licence-Identifier: EUPL-1.2

// Package engine holds the engine-neutral serving adapters that turn a concrete
// decode engine (the Apple-GPU "metal" engine, the AMD "hip" engine, ...) into
// the inference contract surface. A concrete engine supplies a [Session] (its
// retained decode session) and a [TokenModel] (its loaded decode model); this
// package wraps them as [SessionHandle] (inference.SessionHandle +
// inference.KVRestorer) and [TextModel] (inference.TextModel +
// inference.SessionFactory) with the tokenise / generate / capture / restore /
// fork logic that is identical across engines.
//
// The wrapper logic is engine-agnostic: Prefill = tokenise + PrefillTokens;
// AppendPrompt = tokenise + AppendTokens; Generate streams via the engine's
// stepper; CaptureKV / RangeKVBlocks / RestoreFromKV delegate straight through
// (the engine already speaks [kv.Snapshot]); Fork = CaptureKV → open new →
// RestoreFromKV. Only the concrete [Session] / [TokenModel] are engine-specific.
//
// This package imports only the inference contracts (inference, inference/kv,
// inference/model, inference/tokenizer) and core — never a concrete engine — so
// each engine implements the same interfaces and shares this machinery.
package engine

import (
	"context"
	"iter"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model"
)

// Session is the retained-decode-session surface a concrete engine must provide
// for [SessionHandle] to drive it. Every method is expressed in inference/kv and
// inference/model terms only — the metal engine's *ArchSession and the hip
// engine's session both satisfy it. The method set is exactly the primitives
// [SessionHandle] calls; add nothing an engine is not asked for.
type Session interface {
	// PrefillTokens stores the prompt tokens' KV/logit state, replacing any prior state.
	PrefillTokens(ids []int32) error
	// AppendTokens extends the retained state without replaying the prefix.
	AppendTokens(ids []int32) error
	// Pos is the number of tokens currently in the retained cache.
	Pos() int
	// GenerateFromCacheEach greedily decodes up to maxNew tokens from the retained
	// cache, yielding each; eosID < 0 lets the caller own the stop decision.
	GenerateFromCacheEach(maxNew, eosID int, yield func(int32) bool) ([]int32, error)
	// GenerateSampledFromCacheEach decodes up to maxNew tokens with the sampler and
	// params, honouring stopTokens; transform is an optional per-token remap (nil = none).
	GenerateSampledFromCacheEach(maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error)
	// CaptureKVWithOptions copies the retained KV cache to a portable snapshot.
	CaptureKVWithOptions(opts kv.CaptureOptions) (*kv.Snapshot, error)
	// RangeKVBlocks streams the retained KV state as contiguous token blocks of blockSize.
	RangeKVBlocks(blockSize int, opts kv.CaptureOptions, yield func(kv.Block) (bool, error)) error
	// RestoreFromKV loads a portable snapshot into the retained cache.
	RestoreFromKV(ctx context.Context, snapshot *kv.Snapshot) error
	// Close releases the retained session state.
	Close() error
}

// SessionHandle adapts a retained engine [Session] (+ the model's tokenizer,
// reached through its parent [TextModel]) to inference.SessionHandle — the
// engine-neutral persistent conversation-state surface state/session.Session
// holds. It additionally satisfies inference.KVRestorer so the session package
// (and Fork) can restore a captured kv.Snapshot into it.
type SessionHandle struct {
	mu              sync.Mutex
	model           *TextModel
	sess            Session
	tokens          []int32
	generated       []int32
	prefillDuration time.Duration
	err             error
	closed          bool
}

var (
	_ inference.SessionHandle = (*SessionHandle)(nil)
	_ inference.KVRestorer    = (*SessionHandle)(nil)
)

// NewSessionHandle wraps a fresh engine Session (opened over model) as the
// 9-method inference.SessionHandle + inference.KVRestorer. model supplies the
// tokenizer, context window, and stop tokens, and is the factory Fork/Reset
// reopen a session through.
func NewSessionHandle(model *TextModel, sess Session) *SessionHandle {
	return &SessionHandle{model: model, sess: sess}
}

// Prefill tokenises prompt and stores its KV/logit state, replacing any prior
// retained state.
func (s *SessionHandle) Prefill(ctx context.Context, prompt string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("engine.SessionHandle.Prefill"); err != nil {
		s.err = err
		return err
	}
	if s.model == nil || s.model.tok == nil {
		err := core.NewError("engine.SessionHandle.Prefill: tokenizer is nil")
		s.err = err
		return err
	}
	return s.prefillTokensLocked(ctx, s.model.tok.Encode(prompt))
}

func (s *SessionHandle) prefillTokensLocked(ctx context.Context, tokens []int32) error {
	if len(tokens) == 0 {
		err := core.NewError("engine.SessionHandle.Prefill: empty prompt tokens")
		s.err = err
		return err
	}
	start := time.Now()
	ids := append([]int32(nil), tokens...)
	if err := s.sess.PrefillTokens(ids); err != nil {
		s.err = err
		return err
	}
	s.tokens = ids
	s.generated = nil
	s.prefillDuration = time.Since(start)
	s.err = nil
	return ctx.Err()
}

// AppendPrompt appends prompt to the retained state without replaying the prefix.
func (s *SessionHandle) AppendPrompt(ctx context.Context, prompt string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("engine.SessionHandle.AppendPrompt"); err != nil {
		s.err = err
		return err
	}
	if len(s.tokens) == 0 {
		err := core.NewError("engine.SessionHandle.AppendPrompt: no retained prefix")
		s.err = err
		return err
	}
	if s.model == nil || s.model.tok == nil {
		err := core.NewError("engine.SessionHandle.AppendPrompt: tokenizer is nil")
		s.err = err
		return err
	}
	ids := s.model.tok.Encode(prompt)
	if len(ids) == 0 {
		s.err = nil
		return nil
	}
	if err := s.sess.AppendTokens(ids); err != nil {
		s.err = err
		return err
	}
	s.tokens = append(s.tokens, ids...)
	s.err = nil
	return ctx.Err()
}

// Generate streams tokens from the retained session state, bounded by the token
// budget and the context window, honouring stop tokens after each yield.
func (s *SessionHandle) Generate(ctx context.Context, cfg inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		if ctx == nil {
			ctx = context.Background()
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		if err := s.readyForGenerateLocked("engine.SessionHandle.Generate"); err != nil {
			s.err = err
			return
		}
		maxNew := cfg.MaxTokens
		if maxNew <= 0 || s.sess.Pos()+maxNew > s.model.maxLen {
			maxNew = s.model.maxLen - s.sess.Pos()
		}
		if maxNew <= 0 {
			s.err = core.NewError("engine.SessionHandle.Generate: no room to generate in the context window")
			return
		}
		stop := s.model.stopTokens(cfg)
		emit := func(id int32) bool {
			if ctx.Err() != nil {
				return false
			}
			if yield != nil && !yield(inference.Token{ID: id, Text: s.model.decode(id)}) {
				return false
			}
			return !tokenInSet(id, stop)
		}
		var (
			out  []int32
			gerr error
		)
		if cfg.Temperature > 0 || cfg.MinP > 0 || cfg.RepeatPenalty > 1 {
			params := model.SampleParams{
				Temperature:    cfg.Temperature,
				TopK:           cfg.TopK,
				TopP:           cfg.TopP,
				MinP:           cfg.MinP,
				RepeatPenalty:  cfg.RepeatPenalty,
				SuppressTokens: cfg.SuppressTokens,
			}
			out, gerr = s.sess.GenerateSampledFromCacheEach(maxNew, stop, model.NewSampler(cfg.Seed), params, nil, emit)
		} else {
			out, gerr = s.sess.GenerateFromCacheEach(maxNew, -1, emit)
		}
		if gerr != nil {
			s.err = gerr
			return
		}
		s.tokens = append(s.tokens, out...)
		s.generated = append(s.generated, out...)
		s.err = ctx.Err()
	}
}

// CaptureKV copies the retained KV cache to a portable kv.Snapshot.
func (s *SessionHandle) CaptureKV(ctx context.Context) (*kv.Snapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyForGenerateLocked("engine.SessionHandle.CaptureKV"); err != nil {
		s.err = err
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		s.err = err
		return nil, err
	}
	snap, err := s.sess.CaptureKVWithOptions(kv.CaptureOptions{})
	if err != nil {
		s.err = err
		return nil, err
	}
	s.err = nil
	return snap, ctx.Err()
}

// RangeKVBlocks streams the retained KV state as contiguous token blocks.
func (s *SessionHandle) RangeKVBlocks(ctx context.Context, blockSize int, opts kv.CaptureOptions, yield func(kv.Block) (bool, error)) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if yield == nil {
		return core.NewError("engine.SessionHandle.RangeKVBlocks: nil yield")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyForGenerateLocked("engine.SessionHandle.RangeKVBlocks"); err != nil {
		s.err = err
		return err
	}
	if err := ctx.Err(); err != nil {
		s.err = err
		return err
	}
	if err := s.sess.RangeKVBlocks(blockSize, opts, yield); err != nil {
		s.err = err
		return err
	}
	s.err = nil
	return nil
}

// RestoreFromKV loads a portable kv.Snapshot into the retained cache so the next
// generation continues from it (inference.KVRestorer). The engine consumes the
// snapshot in kv.Snapshot terms directly (Session.RestoreFromKV).
func (s *SessionHandle) RestoreFromKV(ctx context.Context, snapshot *kv.Snapshot) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if snapshot == nil {
		return core.NewError("engine.SessionHandle.RestoreFromKV: nil snapshot")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("engine.SessionHandle.RestoreFromKV"); err != nil {
		s.err = err
		return err
	}
	if err := s.sess.RestoreFromKV(ctx, snapshot); err != nil {
		s.err = err
		return err
	}
	s.tokens = append([]int32(nil), snapshot.Tokens...)
	s.generated = nil
	s.prefillDuration = 0
	s.err = nil
	return ctx.Err()
}

// Fork creates an independent session from the same retained state by capturing
// this session's KV and restoring it into a fresh one.
func (s *SessionHandle) Fork(ctx context.Context) (inference.SessionHandle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	snapshot, err := s.CaptureKV(ctx)
	if err != nil {
		return nil, err
	}
	fork := s.model.NewSession()
	if fork == nil {
		return nil, core.NewError("engine.SessionHandle.Fork: model returned nil session")
	}
	restorer, ok := fork.(inference.KVRestorer)
	if !ok {
		_ = fork.Close()
		return nil, core.NewError("engine.SessionHandle.Fork: forked session cannot restore KV")
	}
	if err := restorer.RestoreFromKV(ctx, snapshot); err != nil {
		_ = fork.Close()
		return nil, err
	}
	return fork, nil
}

// Reset releases the retained state and reopens a fresh session ready for
// another prefill.
func (s *SessionHandle) Reset() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tokens = nil
	s.generated = nil
	s.prefillDuration = 0
	if s.model == nil || s.closed {
		return
	}
	next, err := s.model.openSession()
	if err != nil {
		s.err = err
		return
	}
	old := s.sess
	s.sess = next
	s.err = nil
	if old != nil {
		_ = old.Close()
	}
}

// Close releases the retained session state.
func (s *SessionHandle) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return s.err
	}
	s.closed = true
	s.tokens = nil
	s.generated = nil
	if s.sess == nil {
		return s.err
	}
	err := s.sess.Close()
	if err != nil {
		s.err = err
	}
	s.sess = nil
	return err
}

// Err returns the last session error.
func (s *SessionHandle) Err() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

func (s *SessionHandle) readyLocked(scope string) error {
	if s == nil || s.model == nil || s.sess == nil {
		return core.NewError(scope + ": nil session")
	}
	if s.closed {
		return core.NewError(scope + ": session is closed")
	}
	return nil
}

func (s *SessionHandle) readyForGenerateLocked(scope string) error {
	if err := s.readyLocked(scope); err != nil {
		return err
	}
	if s.sess.Pos() <= 0 {
		return core.NewError(scope + ": no retained prefill state")
	}
	return nil
}
