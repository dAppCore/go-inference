// SPDX-Licence-Identifier: EUPL-1.2

package enginetest

import (
	"context"
	"iter"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
)

// fakeSession is the kit's self-test implementer: the smallest conformant
// SessionHandle (plus the optional KVRestorer capability), demonstrating
// exactly what the suite demands of a real engine. It models state as a
// token counter — no model, no weights.
type fakeSession struct {
	tokens int
	err    error
	closed bool
}

func (f *fakeSession) Prefill(_ context.Context, prompt string) error {
	f.tokens = len(prompt)
	return nil
}

func (f *fakeSession) AppendPrompt(_ context.Context, prompt string) error {
	if f.tokens == 0 {
		f.tokens = 1
	}
	f.tokens += len(prompt)
	return nil
}

func (f *fakeSession) Generate(_ context.Context, cfg inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		n := cfg.MaxTokens
		if n <= 0 || n > 4 {
			n = 4
		}
		for i := 0; i < n; i++ {
			if !yield(inference.Token{ID: int32(i), Text: "x"}) {
				return
			}
		}
	}
}

func (f *fakeSession) CaptureKV(context.Context) (*kv.Snapshot, error) {
	return &kv.Snapshot{Architecture: "fake", SeqLen: f.tokens}, nil
}

func (f *fakeSession) RangeKVBlocks(_ context.Context, blockSize int, _ kv.CaptureOptions, yield func(kv.Block) (bool, error)) error {
	if blockSize <= 0 {
		blockSize = 16
	}
	_, err := yield(kv.Block{Index: 0, TokenStart: 0, TokenCount: min(f.tokens, blockSize)})
	return err
}

func (f *fakeSession) Fork(context.Context) (inference.SessionHandle, error) {
	return &fakeSession{tokens: f.tokens}, nil
}

func (f *fakeSession) Reset()       { f.tokens = 0 }
func (f *fakeSession) Close() error { f.closed = true; return nil }
func (f *fakeSession) Err() error   { return f.err }

// RestoreFromKV is the optional inference.KVRestorer capability.
func (f *fakeSession) RestoreFromKV(_ context.Context, snapshot *kv.Snapshot) error {
	f.tokens = snapshot.SeqLen
	return nil
}

var (
	_ inference.SessionHandle = (*fakeSession)(nil)
	_ inference.KVRestorer    = (*fakeSession)(nil)
)

// TestSessionHandle_SuiteSelfTest_Good proves the conformance suite runs
// end-to-end against a minimal conformant implementer — the kit's own gate,
// and the worked example an engine copies.
func TestSessionHandle_SuiteSelfTest_Good(t *testing.T) {
	SessionHandle(t, func(*testing.T) inference.SessionHandle {
		return &fakeSession{}
	})
}

// sessionWithoutRestore is conformant but deliberately does not advertise the
// optional KVRestorer capability. Engines that lack snapshot restore are still
// valid conformance-suite consumers.
type sessionWithoutRestore struct{ state fakeSession }

func (s *sessionWithoutRestore) Prefill(ctx context.Context, prompt string) error {
	return s.state.Prefill(ctx, prompt)
}

func (s *sessionWithoutRestore) AppendPrompt(ctx context.Context, prompt string) error {
	return s.state.AppendPrompt(ctx, prompt)
}

func (s *sessionWithoutRestore) Generate(ctx context.Context, cfg inference.GenerateConfig) iter.Seq[inference.Token] {
	return s.state.Generate(ctx, cfg)
}

func (s *sessionWithoutRestore) CaptureKV(ctx context.Context) (*kv.Snapshot, error) {
	return s.state.CaptureKV(ctx)
}

func (s *sessionWithoutRestore) RangeKVBlocks(ctx context.Context, blockSize int, opts kv.CaptureOptions, yield func(kv.Block) (bool, error)) error {
	return s.state.RangeKVBlocks(ctx, blockSize, opts, yield)
}

func (s *sessionWithoutRestore) Fork(ctx context.Context) (inference.SessionHandle, error) {
	return &sessionWithoutRestore{state: s.state}, nil
}

func (s *sessionWithoutRestore) Reset()       { s.state.Reset() }
func (s *sessionWithoutRestore) Close() error { return s.state.Close() }
func (s *sessionWithoutRestore) Err() error   { return s.state.Err() }

// TestSessionHandle_OptionalKVRestoreUnavailable_Ugly proves the conformance
// suite accepts an engine that omits its documented optional restore surface.
func TestSessionHandle_OptionalKVRestoreUnavailable_Ugly(t *testing.T) {
	SessionHandle(t, func(*testing.T) inference.SessionHandle {
		return &sessionWithoutRestore{}
	})
}
