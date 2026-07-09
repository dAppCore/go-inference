// SPDX-Licence-Identifier: EUPL-1.2

package sessionfake

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/eval/probe"
	"dappco.re/go/inference/kv"
)

// House test-standard triplets — one clean
// TestSessionfake_Handle_<Method>_{Good,Bad,Ugly} per sessionfake.go public
// symbol, added alongside the richer scenario-named tests in
// sessionfake_coverage_test.go (which stay in place).

// TestSessionfake_Handle_Prefill_Good records the prompt and returns nil.
func TestSessionfake_Handle_Prefill_Good(t *testing.T) {
	h := &Handle{}

	if err := h.Prefill(context.Background(), "hello"); err != nil {
		t.Fatalf("Prefill() error = %v, want nil", err)
	}
	if h.PrefillPrompt != "hello" {
		t.Fatalf("PrefillPrompt = %q, want %q", h.PrefillPrompt, "hello")
	}
}

// TestSessionfake_Handle_Prefill_Bad — a seeded PrefillErr is returned even
// though the prompt is still recorded.
func TestSessionfake_Handle_Prefill_Bad(t *testing.T) {
	sentinel := core.NewError("prefill boom")
	h := &Handle{PrefillErr: sentinel}

	if err := h.Prefill(context.Background(), "hello"); !core.Is(err, sentinel) {
		t.Fatalf("Prefill() error = %v, want %v", err, sentinel)
	}
	if h.PrefillPrompt != "hello" {
		t.Fatalf("PrefillPrompt = %q, want recorded despite error", h.PrefillPrompt)
	}
}

// TestSessionfake_Handle_Prefill_Ugly — a second call overwrites the first
// recorded prompt rather than accumulating.
func TestSessionfake_Handle_Prefill_Ugly(t *testing.T) {
	h := &Handle{}

	_ = h.Prefill(context.Background(), "first")
	_ = h.Prefill(context.Background(), "second")

	if h.PrefillPrompt != "second" {
		t.Fatalf("PrefillPrompt = %q, want %q (no accumulation)", h.PrefillPrompt, "second")
	}
}

// TestSessionfake_Handle_PrefillChunks_Good collects the chunk sequence in
// order.
func TestSessionfake_Handle_PrefillChunks_Good(t *testing.T) {
	h := &Handle{}

	if err := h.PrefillChunks(context.Background(), seqOf("a", "b")); err != nil {
		t.Fatalf("PrefillChunks() error = %v, want nil", err)
	}
	if got := h.PrefillChunksSeen; len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("PrefillChunksSeen = %v, want [a b]", got)
	}
}

// TestSessionfake_Handle_PrefillChunks_Bad — a seeded PrefillErr is returned.
func TestSessionfake_Handle_PrefillChunks_Bad(t *testing.T) {
	sentinel := core.NewError("chunk boom")
	h := &Handle{PrefillErr: sentinel}

	if err := h.PrefillChunks(context.Background(), seqOf("a")); !core.Is(err, sentinel) {
		t.Fatalf("PrefillChunks() error = %v, want %v", err, sentinel)
	}
}

// TestSessionfake_Handle_PrefillChunks_Ugly — a nil sequence collects to an
// empty, non-nil slice.
func TestSessionfake_Handle_PrefillChunks_Ugly(t *testing.T) {
	h := &Handle{}

	if err := h.PrefillChunks(context.Background(), nil); err != nil {
		t.Fatalf("PrefillChunks(nil) error = %v, want nil", err)
	}
	if h.PrefillChunksSeen == nil || len(h.PrefillChunksSeen) != 0 {
		t.Fatalf("PrefillChunksSeen = %v, want empty non-nil", h.PrefillChunksSeen)
	}
}

// TestSessionfake_Handle_PrefillTokens_Good copies the token IDs.
func TestSessionfake_Handle_PrefillTokens_Good(t *testing.T) {
	h := &Handle{}

	if err := h.PrefillTokens(context.Background(), []int32{1, 2}); err != nil {
		t.Fatalf("PrefillTokens() error = %v, want nil", err)
	}
	if got := h.PrefillTokensSeen; len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("PrefillTokensSeen = %v, want [1 2]", got)
	}
}

// TestSessionfake_Handle_PrefillTokens_Bad — a seeded PrefillErr is
// returned.
func TestSessionfake_Handle_PrefillTokens_Bad(t *testing.T) {
	sentinel := core.NewError("tok boom")
	h := &Handle{PrefillErr: sentinel}

	if err := h.PrefillTokens(context.Background(), []int32{1}); !core.Is(err, sentinel) {
		t.Fatalf("PrefillTokens() error = %v, want %v", err, sentinel)
	}
}

// TestSessionfake_Handle_PrefillTokens_Ugly — the recorded tokens are a
// defensive copy; mutating the caller's slice afterwards must not alias.
func TestSessionfake_Handle_PrefillTokens_Ugly(t *testing.T) {
	h := &Handle{}
	in := []int32{7, 8}

	_ = h.PrefillTokens(context.Background(), in)
	in[0] = 99

	if h.PrefillTokensSeen[0] != 7 {
		t.Fatalf("PrefillTokensSeen aliased caller slice: %v", h.PrefillTokensSeen)
	}
}

// TestSessionfake_Handle_AppendPrompt_Good records the appended prompt.
func TestSessionfake_Handle_AppendPrompt_Good(t *testing.T) {
	h := &Handle{}

	if err := h.AppendPrompt(context.Background(), "more"); err != nil {
		t.Fatalf("AppendPrompt() error = %v, want nil", err)
	}
	if h.AppendPromptSeen != "more" {
		t.Fatalf("AppendPromptSeen = %q, want %q", h.AppendPromptSeen, "more")
	}
}

// TestSessionfake_Handle_AppendPrompt_Bad — a seeded AppendErr is returned.
func TestSessionfake_Handle_AppendPrompt_Bad(t *testing.T) {
	sentinel := core.NewError("append boom")
	h := &Handle{AppendErr: sentinel}

	if err := h.AppendPrompt(context.Background(), "more"); !core.Is(err, sentinel) {
		t.Fatalf("AppendPrompt() error = %v, want %v", err, sentinel)
	}
}

// TestSessionfake_Handle_AppendPrompt_Ugly — an empty-string prompt is still
// recorded, distinguishable from the unset zero value.
func TestSessionfake_Handle_AppendPrompt_Ugly(t *testing.T) {
	h := &Handle{AppendPromptSeen: "untouched"}

	if err := h.AppendPrompt(context.Background(), ""); err != nil {
		t.Fatalf("AppendPrompt(\"\") error = %v, want nil", err)
	}
	if h.AppendPromptSeen != "" {
		t.Fatalf("AppendPromptSeen = %q, want empty string recorded", h.AppendPromptSeen)
	}
}

// TestSessionfake_Handle_AppendPromptChunks_Good collects the appended
// chunk sequence.
func TestSessionfake_Handle_AppendPromptChunks_Good(t *testing.T) {
	h := &Handle{}

	if err := h.AppendPromptChunks(context.Background(), seqOf("x", "y")); err != nil {
		t.Fatalf("AppendPromptChunks() error = %v, want nil", err)
	}
	if got := h.AppendChunksSeen; len(got) != 2 || got[0] != "x" || got[1] != "y" {
		t.Fatalf("AppendChunksSeen = %v, want [x y]", got)
	}
}

// TestSessionfake_Handle_AppendPromptChunks_Bad — a seeded AppendErr is
// returned.
func TestSessionfake_Handle_AppendPromptChunks_Bad(t *testing.T) {
	sentinel := core.NewError("append chunk boom")
	h := &Handle{AppendErr: sentinel}

	if err := h.AppendPromptChunks(context.Background(), seqOf("x")); !core.Is(err, sentinel) {
		t.Fatalf("AppendPromptChunks() error = %v, want %v", err, sentinel)
	}
}

// TestSessionfake_Handle_AppendPromptChunks_Ugly — a nil sequence collects
// to an empty, non-nil slice.
func TestSessionfake_Handle_AppendPromptChunks_Ugly(t *testing.T) {
	h := &Handle{}

	if err := h.AppendPromptChunks(context.Background(), nil); err != nil {
		t.Fatalf("AppendPromptChunks(nil) error = %v, want nil", err)
	}
	if h.AppendChunksSeen == nil || len(h.AppendChunksSeen) != 0 {
		t.Fatalf("AppendChunksSeen = %v, want empty non-nil", h.AppendChunksSeen)
	}
}

// TestSessionfake_Handle_AppendTokens_Good copies the appended token IDs.
func TestSessionfake_Handle_AppendTokens_Good(t *testing.T) {
	h := &Handle{}

	if err := h.AppendTokens(context.Background(), []int32{3, 4}); err != nil {
		t.Fatalf("AppendTokens() error = %v, want nil", err)
	}
	if got := h.AppendTokensSeen; len(got) != 2 || got[0] != 3 || got[1] != 4 {
		t.Fatalf("AppendTokensSeen = %v, want [3 4]", got)
	}
}

// TestSessionfake_Handle_AppendTokens_Bad — a seeded AppendErr is returned.
func TestSessionfake_Handle_AppendTokens_Bad(t *testing.T) {
	sentinel := core.NewError("append tok boom")
	h := &Handle{AppendErr: sentinel}

	if err := h.AppendTokens(context.Background(), []int32{1}); !core.Is(err, sentinel) {
		t.Fatalf("AppendTokens() error = %v, want %v", err, sentinel)
	}
}

// TestSessionfake_Handle_AppendTokens_Ugly — the recorded tokens are a
// defensive copy.
func TestSessionfake_Handle_AppendTokens_Ugly(t *testing.T) {
	h := &Handle{}
	in := []int32{5, 6}

	_ = h.AppendTokens(context.Background(), in)
	in[0] = 99

	if h.AppendTokensSeen[0] != 5 {
		t.Fatalf("AppendTokensSeen aliased caller slice: %v", h.AppendTokensSeen)
	}
}

// TestSessionfake_Handle_Generate_Good drains the seeded tokens and records
// the generate config plus call count.
func TestSessionfake_Handle_Generate_Good(t *testing.T) {
	h := &Handle{Tokens: []inference.Token{{ID: 1, Text: "a"}, {ID: 2, Text: "b"}}}
	cfg := inference.GenerateConfig{MaxTokens: 5}

	var got []inference.Token
	for tok := range h.Generate(context.Background(), cfg) {
		got = append(got, tok)
	}

	if len(got) != 2 {
		t.Fatalf("Generate() yielded %v, want both seeded tokens", got)
	}
	if h.GenerateCalls != 1 || h.Cfg.MaxTokens != 5 {
		t.Fatalf("GenerateCalls/Cfg = %d/%+v, want 1/MaxTokens=5", h.GenerateCalls, h.Cfg)
	}
}

// TestSessionfake_Handle_Generate_Bad — the consumer stopping early (yield
// returns false) is honoured; only the tokens read before the stop are
// consumed.
func TestSessionfake_Handle_Generate_Bad(t *testing.T) {
	h := &Handle{Tokens: []inference.Token{{ID: 1}, {ID: 2}, {ID: 3}}}

	count := 0
	for range h.Generate(context.Background(), inference.GenerateConfig{}) {
		count++
		break
	}

	if count != 1 {
		t.Fatalf("consumed %d tokens, want 1 (early stop)", count)
	}
}

// TestSessionfake_Handle_Generate_Ugly — seeded ProbeEvents with a nil
// ProbeSink do not panic; the loop still yields the seeded tokens.
func TestSessionfake_Handle_Generate_Ugly(t *testing.T) {
	h := &Handle{
		Tokens:      []inference.Token{{ID: 1}},
		ProbeEvents: []probe.Event{{Step: 0}},
	}

	got := 0
	for range h.Generate(context.Background(), inference.GenerateConfig{}) {
		got++
	}

	if got != 1 {
		t.Fatalf("yielded %d tokens, want 1", got)
	}
}

// TestSessionfake_Handle_CaptureKV_Good returns the seeded snapshot and a
// nil error.
func TestSessionfake_Handle_CaptureKV_Good(t *testing.T) {
	snap := TestKVSnapshot()
	h := &Handle{KV: snap}

	got, err := h.CaptureKV(context.Background())

	if err != nil {
		t.Fatalf("CaptureKV() error = %v, want nil", err)
	}
	if got != snap {
		t.Fatalf("CaptureKV() = %p, want %p", got, snap)
	}
}

// TestSessionfake_Handle_CaptureKV_Bad — a seeded CaptureErr is returned.
func TestSessionfake_Handle_CaptureKV_Bad(t *testing.T) {
	sentinel := core.NewError("capture boom")
	h := &Handle{CaptureErr: sentinel}

	if _, err := h.CaptureKV(context.Background()); !core.Is(err, sentinel) {
		t.Fatalf("CaptureKV() error = %v, want %v", err, sentinel)
	}
}

// TestSessionfake_Handle_CaptureKV_Ugly — a zero-value handle returns a nil
// snapshot and a nil error, distinct from the seeded-error Bad case.
func TestSessionfake_Handle_CaptureKV_Ugly(t *testing.T) {
	h := &Handle{}

	snap, err := h.CaptureKV(context.Background())

	if snap != nil || err != nil {
		t.Fatalf("CaptureKV(zero) = %v/%v, want nil/nil", snap, err)
	}
}

// TestSessionfake_Handle_RangeKVBlocks_Good iterates the seeded blocks in
// order.
func TestSessionfake_Handle_RangeKVBlocks_Good(t *testing.T) {
	h := &Handle{KVBlocks: []kv.Block{{Index: 0}, {Index: 1}}}

	var seen []int
	err := h.RangeKVBlocks(context.Background(), 0, kv.CaptureOptions{}, func(b kv.Block) (bool, error) {
		seen = append(seen, b.Index)
		return true, nil
	})

	if err != nil {
		t.Fatalf("RangeKVBlocks() error = %v, want nil", err)
	}
	if len(seen) != 2 || seen[0] != 0 || seen[1] != 1 {
		t.Fatalf("iterated indices = %v, want [0 1]", seen)
	}
}

// TestSessionfake_Handle_RangeKVBlocks_Bad — no seeded blocks and no KV
// falls through to a nil return without ever calling yield.
func TestSessionfake_Handle_RangeKVBlocks_Bad(t *testing.T) {
	h := &Handle{}
	called := false

	err := h.RangeKVBlocks(context.Background(), 0, kv.CaptureOptions{}, func(kv.Block) (bool, error) {
		called = true
		return true, nil
	})

	if err != nil {
		t.Fatalf("RangeKVBlocks(empty) error = %v, want nil", err)
	}
	if called {
		t.Fatal("yield was called with no blocks and no KV")
	}
}

// TestSessionfake_Handle_RangeKVBlocks_Ugly — the yield callback's error is
// propagated verbatim.
func TestSessionfake_Handle_RangeKVBlocks_Ugly(t *testing.T) {
	sentinel := core.NewError("yield boom")
	h := &Handle{KVBlocks: []kv.Block{{Index: 0}}}

	err := h.RangeKVBlocks(context.Background(), 0, kv.CaptureOptions{}, func(kv.Block) (bool, error) {
		return true, sentinel
	})

	if !core.Is(err, sentinel) {
		t.Fatalf("RangeKVBlocks() error = %v, want %v", err, sentinel)
	}
}

// TestSessionfake_Handle_RestoreKV_Good records the restored snapshot and
// returns nil.
func TestSessionfake_Handle_RestoreKV_Good(t *testing.T) {
	snap := TestKVSnapshot()
	h := &Handle{}

	if err := h.RestoreKV(context.Background(), snap); err != nil {
		t.Fatalf("RestoreKV() error = %v, want nil", err)
	}
	if h.RestoredKV != snap {
		t.Fatalf("RestoredKV = %p, want %p", h.RestoredKV, snap)
	}
}

// TestSessionfake_Handle_RestoreKV_Bad — a seeded RestoreErr is returned.
func TestSessionfake_Handle_RestoreKV_Bad(t *testing.T) {
	sentinel := core.NewError("restore boom")
	h := &Handle{RestoreErr: sentinel}

	if err := h.RestoreKV(context.Background(), TestKVSnapshot()); !core.Is(err, sentinel) {
		t.Fatalf("RestoreKV() error = %v, want %v", err, sentinel)
	}
}

// TestSessionfake_Handle_RestoreKV_Ugly — a nil snapshot is recorded as-is;
// the fake performs no validation.
func TestSessionfake_Handle_RestoreKV_Ugly(t *testing.T) {
	h := &Handle{RestoredKV: TestKVSnapshot()}

	if err := h.RestoreKV(context.Background(), nil); err != nil {
		t.Fatalf("RestoreKV(nil) error = %v, want nil", err)
	}
	if h.RestoredKV != nil {
		t.Fatalf("RestoredKV = %+v, want nil recorded", h.RestoredKV)
	}
}

// TestSessionfake_Handle_RestoreKVBlocks_Good sets RestoredKV via the
// single-block shortcut.
func TestSessionfake_Handle_RestoreKVBlocks_Good(t *testing.T) {
	snap := TestKVSnapshot()
	h := &Handle{}
	src := kv.BlockSource{
		BlockCount:   1,
		PrefixTokens: 2,
		Load: func(_ context.Context, i int) (kv.Block, error) {
			return kv.Block{Index: i, TokenStart: 0, TokenCount: 2, Snapshot: snap}, nil
		},
	}

	if err := h.RestoreKVBlocks(context.Background(), src); err != nil {
		t.Fatalf("RestoreKVBlocks() error = %v, want nil", err)
	}
	if h.RestoredKV != snap {
		t.Fatalf("RestoredKV = %p, want %p (single-block shortcut)", h.RestoredKV, snap)
	}
}

// TestSessionfake_Handle_RestoreKVBlocks_Bad — a seeded RestoreBlocksErr
// returns immediately without ever calling Load.
func TestSessionfake_Handle_RestoreKVBlocks_Bad(t *testing.T) {
	sentinel := core.NewError("blocks boom")
	h := &Handle{RestoreBlocksErr: sentinel}
	src := kv.BlockSource{
		BlockCount: 1,
		Load: func(context.Context, int) (kv.Block, error) {
			t.Fatal("Load must not be called when RestoreBlocksErr is set")
			return kv.Block{}, nil
		},
	}

	if err := h.RestoreKVBlocks(context.Background(), src); !core.Is(err, sentinel) {
		t.Fatalf("RestoreKVBlocks() error = %v, want %v", err, sentinel)
	}
}

// TestSessionfake_Handle_RestoreKVBlocks_Ugly — multiple blocks loaded to
// cover the prefix boundary leave RestoredKV nil (no single-block shortcut).
func TestSessionfake_Handle_RestoreKVBlocks_Ugly(t *testing.T) {
	h := &Handle{}
	src := kv.BlockSource{
		BlockCount:   5,
		PrefixTokens: 4,
		Load: func(_ context.Context, i int) (kv.Block, error) {
			return kv.Block{Index: i, TokenStart: i * 2, TokenCount: 2}, nil
		},
	}

	if err := h.RestoreKVBlocks(context.Background(), src); err != nil {
		t.Fatalf("RestoreKVBlocks() error = %v, want nil", err)
	}
	if len(h.RestoredBlocks) != 2 {
		t.Fatalf("RestoredBlocks len = %d, want 2 (break at boundary)", len(h.RestoredBlocks))
	}
	if h.RestoredKV != nil {
		t.Fatalf("RestoredKV = %p, want nil (multi-block, no shortcut)", h.RestoredKV)
	}
}

// TestSessionfake_Handle_Fork_Good returns the seeded fork handle and a nil
// error.
func TestSessionfake_Handle_Fork_Good(t *testing.T) {
	child := &Handle{}
	h := &Handle{Forked: child}

	got, err := h.Fork(context.Background())

	if err != nil {
		t.Fatalf("Fork() error = %v, want nil", err)
	}
	if got != inference.SessionHandle(child) {
		t.Fatalf("Fork() = %v, want seeded child", got)
	}
}

// TestSessionfake_Handle_Fork_Bad — a seeded ForkErr is returned.
func TestSessionfake_Handle_Fork_Bad(t *testing.T) {
	sentinel := core.NewError("fork boom")
	h := &Handle{ForkErr: sentinel}

	if _, err := h.Fork(context.Background()); !core.Is(err, sentinel) {
		t.Fatalf("Fork() error = %v, want %v", err, sentinel)
	}
}

// TestSessionfake_Handle_Fork_Ugly — a zero-value handle returns a nil
// handle and a nil error, distinct from the seeded-error Bad case.
func TestSessionfake_Handle_Fork_Ugly(t *testing.T) {
	h := &Handle{}

	got, err := h.Fork(context.Background())

	if got != nil || err != nil {
		t.Fatalf("Fork(zero) = %v/%v, want nil/nil", got, err)
	}
}

// TestSessionfake_Handle_Reset_Good increments ResetCalls on each call.
func TestSessionfake_Handle_Reset_Good(t *testing.T) {
	h := &Handle{}

	h.Reset()

	if h.ResetCalls != 1 {
		t.Fatalf("ResetCalls = %d, want 1", h.ResetCalls)
	}
}

// TestSessionfake_Handle_Reset_Bad — a zero-value handle has never been
// reset.
func TestSessionfake_Handle_Reset_Bad(t *testing.T) {
	h := &Handle{}

	if h.ResetCalls != 0 {
		t.Fatalf("ResetCalls = %d, want 0 before any call", h.ResetCalls)
	}
}

// TestSessionfake_Handle_Reset_Ugly — repeated calls accumulate the count
// rather than saturating at 1.
func TestSessionfake_Handle_Reset_Ugly(t *testing.T) {
	h := &Handle{}

	h.Reset()
	h.Reset()

	if h.ResetCalls != 2 {
		t.Fatalf("ResetCalls = %d, want 2 (accumulates)", h.ResetCalls)
	}
}

// TestSessionfake_Handle_Close_Good increments CloseCalls and returns nil.
func TestSessionfake_Handle_Close_Good(t *testing.T) {
	h := &Handle{}

	if err := h.Close(); err != nil {
		t.Fatalf("Close() error = %v, want nil", err)
	}
	if h.CloseCalls != 1 {
		t.Fatalf("CloseCalls = %d, want 1", h.CloseCalls)
	}
}

// TestSessionfake_Handle_Close_Bad — a seeded CloseErr is returned.
func TestSessionfake_Handle_Close_Bad(t *testing.T) {
	sentinel := core.NewError("close boom")
	h := &Handle{CloseErr: sentinel}

	if err := h.Close(); !core.Is(err, sentinel) {
		t.Fatalf("Close() error = %v, want %v", err, sentinel)
	}
}

// TestSessionfake_Handle_Close_Ugly — repeated calls both count and both
// return the same seeded error (no idempotence guard).
func TestSessionfake_Handle_Close_Ugly(t *testing.T) {
	sentinel := core.NewError("close boom")
	h := &Handle{CloseErr: sentinel}

	err1 := h.Close()
	err2 := h.Close()

	if !core.Is(err1, sentinel) || !core.Is(err2, sentinel) {
		t.Fatalf("Close() calls = %v/%v, want %v both times", err1, err2, sentinel)
	}
	if h.CloseCalls != 2 {
		t.Fatalf("CloseCalls = %d, want 2", h.CloseCalls)
	}
}

// TestSessionfake_Handle_Err_Good returns nil for a zero-value handle.
func TestSessionfake_Handle_Err_Good(t *testing.T) {
	h := &Handle{}

	if err := h.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil", err)
	}
}

// TestSessionfake_Handle_Err_Bad — a seeded ErrValue is returned verbatim.
func TestSessionfake_Handle_Err_Bad(t *testing.T) {
	sentinel := core.NewError("err value")
	h := &Handle{ErrValue: sentinel}

	if err := h.Err(); !core.Is(err, sentinel) {
		t.Fatalf("Err() = %v, want %v", err, sentinel)
	}
}

// TestSessionfake_Handle_Err_Ugly — Err has no side effects; repeated calls
// return the same value.
func TestSessionfake_Handle_Err_Ugly(t *testing.T) {
	sentinel := core.NewError("err value")
	h := &Handle{ErrValue: sentinel}

	first := h.Err()
	second := h.Err()

	if !core.Is(first, sentinel) || !core.Is(second, sentinel) {
		t.Fatalf("Err() calls = %v/%v, want %v both times", first, second, sentinel)
	}
}

// TestSessionfake_TestKVSnapshot_Good returns the canonical two-token
// gemma4 snapshot.
func TestSessionfake_TestKVSnapshot_Good(t *testing.T) {
	snap := TestKVSnapshot()

	if snap == nil || snap.Architecture != "gemma4_text" || len(snap.Tokens) != 2 {
		t.Fatalf("TestKVSnapshot() = %+v, want the canonical two-token gemma4 fixture", snap)
	}
}

// TestSessionfake_TestKVSnapshot_Bad — each call returns an independent
// object; mutating one must not affect a second call's result.
func TestSessionfake_TestKVSnapshot_Bad(t *testing.T) {
	first := TestKVSnapshot()
	second := TestKVSnapshot()

	first.Tokens[0] = 99

	if second.Tokens[0] != 1 {
		t.Fatalf("TestKVSnapshot() shared backing storage across calls: %v", second.Tokens)
	}
}

// TestSessionfake_TestKVSnapshot_Ugly asserts the full layer/head shape the
// session and agent-memory tests rely on.
func TestSessionfake_TestKVSnapshot_Ugly(t *testing.T) {
	snap := TestKVSnapshot()

	if len(snap.Layers) != 1 || len(snap.Layers[0].Heads) != 1 {
		t.Fatalf("Layers shape = %+v, want 1 layer / 1 head", snap.Layers)
	}
	head := snap.Layers[0].Heads[0]
	if len(head.KeyBytes) != 16 || len(head.ValueBytes) != 16 {
		t.Fatalf("head byte lengths = key %d / value %d, want 16/16", len(head.KeyBytes), len(head.ValueBytes))
	}
}
