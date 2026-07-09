// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
)

// fakeSessionHandle is a minimal in-memory SessionHandle recording what the
// contract methods receive, so probe + round-trip assertions can inspect it.
type fakeSessionHandle struct {
	prefilled string
	appended  string
	cfg       GenerateConfig
	tokens    []Token
	captured  bool
	ranged    kv.CaptureOptions
	rangeSize int
	reset     bool
	closed    bool
	err       error
}

func (h *fakeSessionHandle) Prefill(_ context.Context, prompt string) error {
	h.prefilled = prompt
	return nil
}

func (h *fakeSessionHandle) AppendPrompt(_ context.Context, prompt string) error {
	h.appended = prompt
	return nil
}

func (h *fakeSessionHandle) Generate(_ context.Context, cfg GenerateConfig) iter.Seq[Token] {
	h.cfg = cfg
	return func(yield func(Token) bool) {
		for _, tok := range h.tokens {
			if !yield(tok) {
				return
			}
		}
	}
}

func (h *fakeSessionHandle) CaptureKV(_ context.Context) (*kv.Snapshot, error) {
	h.captured = true
	return &kv.Snapshot{Architecture: "gemma4", SeqLen: 2}, nil
}

func (h *fakeSessionHandle) RangeKVBlocks(_ context.Context, blockSize int, opts kv.CaptureOptions, yield func(kv.Block) (bool, error)) error {
	h.rangeSize, h.ranged = blockSize, opts
	_, err := yield(kv.Block{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: &kv.Snapshot{SeqLen: 2}})
	return err
}

func (h *fakeSessionHandle) Fork(_ context.Context) (SessionHandle, error) {
	return &fakeSessionHandle{prefilled: h.prefilled}, nil
}

func (h *fakeSessionHandle) Reset()       { h.reset = true }
func (h *fakeSessionHandle) Close() error { h.closed = true; return nil }
func (h *fakeSessionHandle) Err() error   { return h.err }

// fakeSessionFactory opens fakeSessionHandles — the SessionFactory conformer.
type fakeSessionFactory struct {
	opened *fakeSessionHandle
}

func (f *fakeSessionFactory) NewSession() SessionHandle {
	f.opened = &fakeSessionHandle{}
	return f.opened
}

func TestSessionContract_SessionHandle_Good(t *testing.T) {
	var probe any = &fakeSessionHandle{tokens: []Token{{ID: 7, Text: "hi"}}}

	handle, ok := probe.(SessionHandle)
	checkTrue(t, ok)

	if err := handle.Prefill(context.Background(), "seed"); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	if err := handle.AppendPrompt(context.Background(), "more"); err != nil {
		t.Fatalf("AppendPrompt: %v", err)
	}
	var got []Token
	for tok := range handle.Generate(context.Background(), GenerateConfig{MaxTokens: 4}) {
		got = append(got, tok)
	}
	checkEqual(t, 1, len(got))
	checkEqual(t, int32(7), got[0].ID)

	snap, err := handle.CaptureKV(context.Background())
	if err != nil {
		t.Fatalf("CaptureKV: %v", err)
	}
	checkEqual(t, "gemma4", snap.Architecture)
}

func TestSessionContract_SessionHandle_GoodRangeForkResetClose(t *testing.T) {
	handle := &fakeSessionHandle{}

	var blocks int
	err := handle.RangeKVBlocks(context.Background(), 128, kv.CaptureOptions{BlockStartToken: 3}, func(kv.Block) (bool, error) {
		blocks++
		return true, nil
	})
	if err != nil {
		t.Fatalf("RangeKVBlocks: %v", err)
	}
	checkEqual(t, 1, blocks)
	checkEqual(t, 128, handle.rangeSize)
	checkEqual(t, 3, handle.ranged.BlockStartToken)

	fork, err := handle.Fork(context.Background())
	if err != nil {
		t.Fatalf("Fork: %v", err)
	}
	checkTrue(t, fork != nil)

	handle.Reset()
	checkTrue(t, handle.reset)
	checkTrue(t, handle.Close() == nil)
	checkTrue(t, handle.closed)
}

func TestSessionContract_SessionHandle_UglyErrSurfaced(t *testing.T) {
	sentinel := core.NewError("session boom")
	handle := &fakeSessionHandle{err: sentinel}

	checkEqual(t, sentinel, handle.Err())
}

func TestSessionContract_SessionFactory_Good(t *testing.T) {
	var probe any = &fakeSessionFactory{}

	factory, ok := probe.(SessionFactory)
	checkTrue(t, ok)

	handle := factory.NewSession()
	checkTrue(t, handle != nil)

	_, isHandle := handle.(SessionHandle)
	checkTrue(t, isHandle)
}

func ExampleSessionFactory() {
	var model any = &fakeSessionFactory{}
	if factory, ok := model.(SessionFactory); ok {
		sess := factory.NewSession()
		defer func() { _ = sess.Close() }()
		_ = sess.Prefill(context.Background(), "You are a helpful assistant.")
	}
	core.Println("session opened")
	// Output: session opened
}
