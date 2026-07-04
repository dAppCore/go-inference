// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"context"
	"iter"

	"dappco.re/go/inference/kv"
)

// Persistent conversation-state contracts — the engine-neutral session
// surface (docs/engine-merge.md names conversation state "the LEM edge", a
// first-class contract). A SessionHandle owns one retained KV/logit state for
// a loaded model: prefill a prompt, generate from the retained cache, capture
// or restore the KV as a portable [kv.Snapshot], fork the state, reset, close.
//
// Both engines satisfy this in inference types. The native engine speaks
// [kv.Snapshot] directly; the metal (cgo) engine wraps its pkg/metal handle in
// a go-mlx-side adapter that converts inward. Optional per-engine capabilities
// (token/chunk prefill, KV restore, capture-with-options) are probed off the
// handle by the session package exactly like [AttentionInspector] is probed
// off a model — they are not part of this core surface.
//
//	sess, err := factory.NewSession(), error(nil)
//	if err := sess.Prefill(ctx, "You are a helpful assistant."); err != nil {
//	    return err
//	}
//	for tok := range sess.Generate(ctx, inference.GenerateConfig{MaxTokens: 128}) {
//	    _ = tok
//	}
//	snap, err := sess.CaptureKV(ctx) // portable kv.Snapshot for sleep/wake
type SessionHandle interface {
	// Prefill tokenises prompt and stores its KV/logit state in the session,
	// replacing any prior retained state.
	Prefill(ctx context.Context, prompt string) error
	// AppendPrompt appends prompt to the retained state without replaying the
	// existing prefix.
	AppendPrompt(ctx context.Context, prompt string) error
	// Generate streams tokens from the retained session state. The iterator
	// stops on the config's stop tokens, the token budget, or ctx cancel; call
	// Err after ranging to observe a generation error.
	Generate(ctx context.Context, cfg GenerateConfig) iter.Seq[Token]
	// CaptureKV copies the current retained KV cache tensors to CPU memory as a
	// portable snapshot.
	CaptureKV(ctx context.Context) (*kv.Snapshot, error)
	// RangeKVBlocks streams the retained KV state as contiguous token blocks of
	// blockSize, without assembling a full CPU snapshot first. opts.BlockStartToken
	// skips blocks the caller already holds (the trusted-prefix sleep lane).
	RangeKVBlocks(ctx context.Context, blockSize int, opts kv.CaptureOptions, yield func(kv.Block) (bool, error)) error
	// Fork creates an independent session starting from the same retained state.
	Fork(ctx context.Context) (SessionHandle, error)
	// Reset releases retained state and leaves the session ready for another prefill.
	Reset()
	// Close releases retained session state.
	Close() error
	// Err returns the last session error.
	Err() error
}

// SessionFactory is the model-level probe that opens a persistent
// [SessionHandle] — the neutral form of an engine's NewSession, probed off a
// loaded model like [KVSnapshotter]. An engine whose loaded model supports
// retained sessions implements it; callers probe and fall back when absent.
//
//	if f, ok := model.(inference.SessionFactory); ok {
//	    sess := f.NewSession()
//	    defer sess.Close()
//	}
type SessionFactory interface {
	// NewSession opens a fresh persistent session over the loaded model, or nil
	// when the model cannot open one.
	NewSession() SessionHandle
}
