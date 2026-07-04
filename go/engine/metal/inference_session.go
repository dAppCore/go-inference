// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// inference_session.go re-expresses go-mlx's metal_session_adapter.go (the
// SessionHandle-adapter logic, which stays in go-mlx and dies with pkg/metal)
// against engine/metal's ArchSession and the inference contracts. It is a thin
// adapter: the native engine already speaks kv.Snapshot directly (see
// kv_contract.go / session_kv_snapshot.go — ArchSession.CaptureKVWithOptions,
// RangeKVBlocks, RestoreFromKV), so the 9 SessionHandle methods delegate to the
// engine's own prefill / generate / capture / restore primitives with no
// metal.* / kvconv conversion. Prefill/AppendPrompt tokenise through the model's
// attached tokenizer.
package native

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

// nativeSession adapts a retained ArchSession (+ the model's tokenizer) to
// inference.SessionHandle — the engine-neutral persistent conversation-state
// surface state/session.Session holds. It additionally satisfies
// inference.KVRestorer so the session package (and Fork) can restore a captured
// kv.Snapshot into it.
type nativeSession struct {
	mu              sync.Mutex
	model           *nativeTextModel
	sess            *ArchSession
	tokens          []int32
	generated       []int32
	prefillDuration time.Duration
	err             error
	closed          bool
}

var (
	_ inference.SessionHandle = (*nativeSession)(nil)
	_ inference.KVRestorer    = (*nativeSession)(nil)
)

// Prefill tokenises prompt and stores its KV/logit state, replacing any prior
// retained state.
func (s *nativeSession) Prefill(ctx context.Context, prompt string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("native.nativeSession.Prefill"); err != nil {
		s.err = err
		return err
	}
	if s.model == nil || s.model.tok == nil {
		err := core.NewError("native.nativeSession.Prefill: tokenizer is nil")
		s.err = err
		return err
	}
	return s.prefillTokensLocked(ctx, s.model.tok.Encode(prompt))
}

func (s *nativeSession) prefillTokensLocked(ctx context.Context, tokens []int32) error {
	if len(tokens) == 0 {
		err := core.NewError("native.nativeSession.Prefill: empty prompt tokens")
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
func (s *nativeSession) AppendPrompt(ctx context.Context, prompt string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("native.nativeSession.AppendPrompt"); err != nil {
		s.err = err
		return err
	}
	if len(s.tokens) == 0 {
		err := core.NewError("native.nativeSession.AppendPrompt: no retained prefix")
		s.err = err
		return err
	}
	if s.model == nil || s.model.tok == nil {
		err := core.NewError("native.nativeSession.AppendPrompt: tokenizer is nil")
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
func (s *nativeSession) Generate(ctx context.Context, cfg inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		if ctx == nil {
			ctx = context.Background()
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		if err := s.readyForGenerateLocked("native.nativeSession.Generate"); err != nil {
			s.err = err
			return
		}
		maxNew := cfg.MaxTokens
		if maxNew <= 0 || s.sess.Pos()+maxNew > s.model.maxLen {
			maxNew = s.model.maxLen - s.sess.Pos()
		}
		if maxNew <= 0 {
			s.err = core.NewError("native.nativeSession.Generate: no room to generate in the context window")
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
func (s *nativeSession) CaptureKV(ctx context.Context) (*kv.Snapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyForGenerateLocked("native.nativeSession.CaptureKV"); err != nil {
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
func (s *nativeSession) RangeKVBlocks(ctx context.Context, blockSize int, opts kv.CaptureOptions, yield func(kv.Block) (bool, error)) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if yield == nil {
		return core.NewError("native.nativeSession.RangeKVBlocks: nil yield")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyForGenerateLocked("native.nativeSession.RangeKVBlocks"); err != nil {
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
// generation continues from it (inference.KVRestorer). The native engine
// consumes the snapshot in kv.Snapshot terms directly (ArchSession.RestoreFromKV).
func (s *nativeSession) RestoreFromKV(ctx context.Context, snapshot *kv.Snapshot) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if snapshot == nil {
		return core.NewError("native.nativeSession.RestoreFromKV: nil snapshot")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("native.nativeSession.RestoreFromKV"); err != nil {
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
func (s *nativeSession) Fork(ctx context.Context) (inference.SessionHandle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	snapshot, err := s.CaptureKV(ctx)
	if err != nil {
		return nil, err
	}
	fork := s.model.NewSession()
	if fork == nil {
		return nil, core.NewError("native.nativeSession.Fork: model returned nil session")
	}
	restorer, ok := fork.(inference.KVRestorer)
	if !ok {
		_ = fork.Close()
		return nil, core.NewError("native.nativeSession.Fork: forked session cannot restore KV")
	}
	if err := restorer.RestoreFromKV(ctx, snapshot); err != nil {
		_ = fork.Close()
		return nil, err
	}
	return fork, nil
}

// Reset releases the retained state and reopens a fresh session ready for
// another prefill.
func (s *nativeSession) Reset() {
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
	next, err := s.model.openArchSession()
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
func (s *nativeSession) Close() error {
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
func (s *nativeSession) Err() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

func (s *nativeSession) readyLocked(scope string) error {
	if s == nil || s.model == nil || s.sess == nil {
		return core.NewError(scope + ": nil session")
	}
	if s.closed {
		return core.NewError(scope + ": session is closed")
	}
	return nil
}

func (s *nativeSession) readyForGenerateLocked(scope string) error {
	if err := s.readyLocked(scope); err != nil {
		return err
	}
	if s.sess.Pos() <= 0 {
		return core.NewError(scope + ": no retained prefill state")
	}
	return nil
}
