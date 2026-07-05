// SPDX-Licence-Identifier: EUPL-1.2

package sessionfake

import (
	"context"
	"errors"
	"iter"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/eval/probe"
)

// Coverage tests for the recording fake. Every method here is exercised
// against its branches: the trivial setter/counter methods (which record
// args + return the seeded *Err), and the four branch-heavy paths —
// Generate's deferred AfterGenerate / ProbeSink / early-stop yield,
// collectChunks's nil guard, RangeKVBlocks's synth-block / iterate /
// stop / error arms, and RestoreKVBlocks's early-err / load-err / break /
// single-block arms. White-box (package sessionfake): all fields are
// exported so behaviour is steered by seeding the Handle directly.

func TestPrefill_RecordsPromptAndReturnsErr(t *testing.T) {
	sentinel := errors.New("prefill boom")
	h := &Handle{PrefillErr: sentinel}
	if err := h.Prefill(context.Background(), "hello"); err != sentinel {
		t.Fatalf("Prefill err = %v, want %v", err, sentinel)
	}
	if h.PrefillPrompt != "hello" {
		t.Fatalf("PrefillPrompt = %q, want %q", h.PrefillPrompt, "hello")
	}

	ok := &Handle{}
	if err := ok.Prefill(context.Background(), "world"); err != nil {
		t.Fatalf("Prefill err = %v, want nil", err)
	}
	if ok.PrefillPrompt != "world" {
		t.Fatalf("PrefillPrompt = %q, want %q", ok.PrefillPrompt, "world")
	}
}

func TestPrefillChunks_CollectsAndReturnsErr(t *testing.T) {
	sentinel := errors.New("chunk boom")
	h := &Handle{PrefillErr: sentinel}
	if err := h.PrefillChunks(context.Background(), seqOf("a", "b", "c")); err != sentinel {
		t.Fatalf("PrefillChunks err = %v, want %v", err, sentinel)
	}
	if got := h.PrefillChunksSeen; len(got) != 3 || got[0] != "a" || got[2] != "c" {
		t.Fatalf("PrefillChunksSeen = %v, want [a b c]", got)
	}
}

func TestPrefillTokens_CopiesAndReturnsErr(t *testing.T) {
	sentinel := errors.New("tok boom")
	h := &Handle{PrefillErr: sentinel}
	in := []int32{7, 8, 9}
	if err := h.PrefillTokens(context.Background(), in); err != sentinel {
		t.Fatalf("PrefillTokens err = %v, want %v", err, sentinel)
	}
	if got := h.PrefillTokensSeen; len(got) != 3 || got[0] != 7 || got[2] != 9 {
		t.Fatalf("PrefillTokensSeen = %v, want [7 8 9]", got)
	}
	// The fake copies, so mutating the caller's slice must not change the record.
	in[0] = 100
	if h.PrefillTokensSeen[0] != 7 {
		t.Fatalf("PrefillTokensSeen aliased caller slice: %v", h.PrefillTokensSeen)
	}
}

func TestAppendPrompt_RecordsPromptAndReturnsErr(t *testing.T) {
	sentinel := errors.New("append boom")
	h := &Handle{AppendErr: sentinel}
	if err := h.AppendPrompt(context.Background(), "more"); err != sentinel {
		t.Fatalf("AppendPrompt err = %v, want %v", err, sentinel)
	}
	if h.AppendPromptSeen != "more" {
		t.Fatalf("AppendPromptSeen = %q, want %q", h.AppendPromptSeen, "more")
	}
}

func TestAppendPromptChunks_CollectsAndReturnsErr(t *testing.T) {
	h := &Handle{}
	if err := h.AppendPromptChunks(context.Background(), seqOf("x", "y")); err != nil {
		t.Fatalf("AppendPromptChunks err = %v, want nil", err)
	}
	if got := h.AppendChunksSeen; len(got) != 2 || got[0] != "x" || got[1] != "y" {
		t.Fatalf("AppendChunksSeen = %v, want [x y]", got)
	}
}

func TestAppendTokens_CopiesAndReturnsErr(t *testing.T) {
	sentinel := errors.New("append tok boom")
	h := &Handle{AppendErr: sentinel}
	if err := h.AppendTokens(context.Background(), []int32{1, 2}); err != sentinel {
		t.Fatalf("AppendTokens err = %v, want %v", err, sentinel)
	}
	if got := h.AppendTokensSeen; len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("AppendTokensSeen = %v, want [1 2]", got)
	}
}

func TestCollectChunks_NilReturnsEmpty(t *testing.T) {
	if got := collectChunks(nil); got == nil || len(got) != 0 {
		t.Fatalf("collectChunks(nil) = %v, want empty non-nil", got)
	}
	// Reach the nil guard through the public method too.
	h := &Handle{}
	if err := h.PrefillChunks(context.Background(), nil); err != nil {
		t.Fatalf("PrefillChunks(nil) err = %v, want nil", err)
	}
	if len(h.PrefillChunksSeen) != 0 {
		t.Fatalf("PrefillChunksSeen = %v, want empty", h.PrefillChunksSeen)
	}
}

func TestGenerate_DrainsTokensAndRecordsCfg(t *testing.T) {
	h := &Handle{
		Tokens: []inference.Token{{ID: 1, Text: "a"}, {ID: 2, Text: "b"}},
	}
	cfg := inference.GenerateConfig{MaxTokens: 5}
	var got []inference.Token
	for tok := range h.Generate(context.Background(), cfg) {
		got = append(got, tok)
	}
	if len(got) != 2 || got[0].ID != 1 || got[1].ID != 2 {
		t.Fatalf("Generate yielded %v, want both seeded tokens", got)
	}
	if h.GenerateCalls != 1 {
		t.Fatalf("GenerateCalls = %d, want 1", h.GenerateCalls)
	}
	if h.Cfg.MaxTokens != 5 {
		t.Fatalf("Cfg.MaxTokens = %d, want 5 (cfg not recorded)", h.Cfg.MaxTokens)
	}
}

func TestGenerate_EarlyStopHonoursYieldFalse(t *testing.T) {
	h := &Handle{
		Tokens: []inference.Token{{ID: 1}, {ID: 2}, {ID: 3}},
	}
	count := 0
	for range h.Generate(context.Background(), inference.GenerateConfig{}) {
		count++
		break // yield returns false after the first token -> hits the return.
	}
	if count != 1 {
		t.Fatalf("consumed %d tokens, want 1 (early stop)", count)
	}
}

func TestGenerate_EmitsProbeEventsAndRunsAfterGenerate(t *testing.T) {
	var emitted []probe.Event
	sink := probe.SinkFunc(func(e probe.Event) { emitted = append(emitted, e) })

	afterRan := false
	h := &Handle{
		ProbeEvents:   []probe.Event{{Step: 0}, {Step: 1}},
		Tokens:        []inference.Token{{ID: 9}},
		AfterGenerate: func(_ *Handle) { afterRan = true },
	}
	cfg := inference.GenerateConfig{ProbeSink: sink}

	var got []inference.Token
	for tok := range h.Generate(context.Background(), cfg) {
		got = append(got, tok)
	}

	if len(emitted) != 2 || emitted[0].Step != 0 || emitted[1].Step != 1 {
		t.Fatalf("emitted probes = %v, want both seeded events", emitted)
	}
	if len(got) != 1 || got[0].ID != 9 {
		t.Fatalf("Generate yielded %v, want one seeded token", got)
	}
	if !afterRan {
		t.Fatal("AfterGenerate hook did not run")
	}
}

func TestGenerate_NilProbeSinkSkipsEmit(t *testing.T) {
	// ProbeSink nil with seeded events: the EmitProbe branch is skipped but
	// the loop still runs over the events without panicking.
	h := &Handle{
		ProbeEvents: []probe.Event{{Step: 0}},
		Tokens:      []inference.Token{{ID: 1}},
	}
	got := 0
	for range h.Generate(context.Background(), inference.GenerateConfig{}) {
		got++
	}
	if got != 1 {
		t.Fatalf("yielded %d tokens, want 1", got)
	}
}

func TestCaptureKV_ReturnsSeededSnapshotAndErr(t *testing.T) {
	snap := TestKVSnapshot()
	sentinel := errors.New("capture boom")
	h := &Handle{KV: snap, CaptureErr: sentinel}
	gotSnap, gotErr := h.CaptureKV(context.Background())
	if gotSnap != snap {
		t.Fatalf("CaptureKV snapshot = %p, want %p", gotSnap, snap)
	}
	if gotErr != sentinel {
		t.Fatalf("CaptureKV err = %v, want %v", gotErr, sentinel)
	}
}

func TestRangeKVBlocks_SynthesisesWholeKVWhenNoBlocks(t *testing.T) {
	snap := TestKVSnapshot()
	h := &Handle{KV: snap}
	var seen []kv.Block
	err := h.RangeKVBlocks(context.Background(), 0, kv.CaptureOptions{},
		func(b kv.Block) (bool, error) {
			seen = append(seen, b)
			return true, nil
		})
	if err != nil {
		t.Fatalf("RangeKVBlocks err = %v, want nil", err)
	}
	if len(seen) != 1 {
		t.Fatalf("synthesised %d blocks, want 1", len(seen))
	}
	if seen[0].Snapshot != snap || seen[0].TokenCount != len(snap.Tokens) {
		t.Fatalf("synth block = %+v, want whole-KV block", seen[0])
	}
}

func TestRangeKVBlocks_SynthYieldPropagatesError(t *testing.T) {
	sentinel := errors.New("synth yield boom")
	h := &Handle{KV: TestKVSnapshot()}
	err := h.RangeKVBlocks(context.Background(), 0, kv.CaptureOptions{},
		func(kv.Block) (bool, error) { return false, sentinel })
	if err != sentinel {
		t.Fatalf("RangeKVBlocks err = %v, want %v", err, sentinel)
	}
}

func TestRangeKVBlocks_IteratesSeededBlocks(t *testing.T) {
	blocks := []kv.Block{
		{Index: 0, TokenStart: 0, TokenCount: 2},
		{Index: 1, TokenStart: 2, TokenCount: 2},
	}
	h := &Handle{KVBlocks: blocks}
	var seen []int
	err := h.RangeKVBlocks(context.Background(), 0, kv.CaptureOptions{},
		func(b kv.Block) (bool, error) {
			seen = append(seen, b.Index)
			return true, nil
		})
	if err != nil {
		t.Fatalf("RangeKVBlocks err = %v, want nil", err)
	}
	if len(seen) != 2 || seen[0] != 0 || seen[1] != 1 {
		t.Fatalf("iterated indices = %v, want [0 1]", seen)
	}
}

func TestRangeKVBlocks_StopsWhenYieldReturnsFalse(t *testing.T) {
	blocks := []kv.Block{{Index: 0}, {Index: 1}, {Index: 2}}
	h := &Handle{KVBlocks: blocks}
	seen := 0
	err := h.RangeKVBlocks(context.Background(), 0, kv.CaptureOptions{},
		func(kv.Block) (bool, error) {
			seen++
			return false, nil // stop after the first block.
		})
	if err != nil {
		t.Fatalf("RangeKVBlocks err = %v, want nil", err)
	}
	if seen != 1 {
		t.Fatalf("yielded %d blocks before stop, want 1", seen)
	}
}

func TestRangeKVBlocks_PropagatesIterError(t *testing.T) {
	sentinel := errors.New("iter boom")
	h := &Handle{KVBlocks: []kv.Block{{Index: 0}, {Index: 1}}}
	err := h.RangeKVBlocks(context.Background(), 0, kv.CaptureOptions{},
		func(kv.Block) (bool, error) { return true, sentinel })
	if err != sentinel {
		t.Fatalf("RangeKVBlocks err = %v, want %v", err, sentinel)
	}
}

func TestRangeKVBlocks_NoBlocksNoKVReturnsNil(t *testing.T) {
	h := &Handle{} // KVBlocks empty, KV nil -> falls through to return nil.
	called := false
	err := h.RangeKVBlocks(context.Background(), 0, kv.CaptureOptions{},
		func(kv.Block) (bool, error) {
			called = true
			return true, nil
		})
	if err != nil {
		t.Fatalf("RangeKVBlocks err = %v, want nil", err)
	}
	if called {
		t.Fatal("yield was called with no blocks and no KV")
	}
}

func TestRestoreKV_RecordsSnapshotAndReturnsErr(t *testing.T) {
	snap := TestKVSnapshot()
	sentinel := errors.New("restore boom")
	h := &Handle{RestoreErr: sentinel}
	if err := h.RestoreKV(context.Background(), snap); err != sentinel {
		t.Fatalf("RestoreKV err = %v, want %v", err, sentinel)
	}
	if h.RestoredKV != snap {
		t.Fatalf("RestoredKV = %p, want %p", h.RestoredKV, snap)
	}
}

func TestRestoreKVBlocks_EarlyErrorReturns(t *testing.T) {
	sentinel := errors.New("blocks boom")
	h := &Handle{RestoreBlocksErr: sentinel}
	src := kv.BlockSource{
		BlockCount: 1,
		Load: func(context.Context, int) (kv.Block, error) {
			t.Fatal("Load must not be called when RestoreBlocksErr is set")
			return kv.Block{}, nil
		},
	}
	if err := h.RestoreKVBlocks(context.Background(), src); err != sentinel {
		t.Fatalf("RestoreKVBlocks err = %v, want %v", err, sentinel)
	}
}

func TestRestoreKVBlocks_LoadErrorPropagates(t *testing.T) {
	sentinel := errors.New("load boom")
	h := &Handle{}
	src := kv.BlockSource{
		BlockCount:   2,
		PrefixTokens: 4,
		Load: func(_ context.Context, i int) (kv.Block, error) {
			return kv.Block{}, sentinel
		},
	}
	if err := h.RestoreKVBlocks(context.Background(), src); err != sentinel {
		t.Fatalf("RestoreKVBlocks err = %v, want %v", err, sentinel)
	}
	if len(h.RestoredBlocks) != 0 {
		t.Fatalf("RestoredBlocks = %v, want empty on load error", h.RestoredBlocks)
	}
}

func TestRestoreKVBlocks_SingleBlockSetsRestoredKV(t *testing.T) {
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
		t.Fatalf("RestoreKVBlocks err = %v, want nil", err)
	}
	if len(h.RestoredBlocks) != 1 {
		t.Fatalf("RestoredBlocks len = %d, want 1", len(h.RestoredBlocks))
	}
	if h.RestoredKV != snap {
		t.Fatalf("RestoredKV = %p, want %p (single-block shortcut)", h.RestoredKV, snap)
	}
}

func TestRestoreKVBlocks_BreaksAtPrefixBoundary(t *testing.T) {
	// First block does not reach PrefixTokens (no break); second one does
	// (break). Two blocks load => no single-block RestoredKV shortcut.
	h := &Handle{}
	src := kv.BlockSource{
		BlockCount:   5, // would loop 5x, but the boundary break stops it at 2.
		PrefixTokens: 4,
		Load: func(_ context.Context, i int) (kv.Block, error) {
			return kv.Block{Index: i, TokenStart: i * 2, TokenCount: 2}, nil
		},
	}
	if err := h.RestoreKVBlocks(context.Background(), src); err != nil {
		t.Fatalf("RestoreKVBlocks err = %v, want nil", err)
	}
	if len(h.RestoredBlocks) != 2 {
		t.Fatalf("RestoredBlocks len = %d, want 2 (break at boundary)", len(h.RestoredBlocks))
	}
	if h.RestoredKV != nil {
		t.Fatalf("RestoredKV = %p, want nil (multi-block, no shortcut)", h.RestoredKV)
	}
}

func TestFork_ReturnsSeededHandleAndErr(t *testing.T) {
	child := &Handle{}
	sentinel := errors.New("fork boom")
	h := &Handle{Forked: child, ForkErr: sentinel}
	got, err := h.Fork(context.Background())
	if got != inference.SessionHandle(child) {
		t.Fatalf("Fork handle = %v, want seeded child", got)
	}
	if err != sentinel {
		t.Fatalf("Fork err = %v, want %v", err, sentinel)
	}
}

func TestReset_CountsCalls(t *testing.T) {
	h := &Handle{}
	h.Reset()
	h.Reset()
	if h.ResetCalls != 2 {
		t.Fatalf("ResetCalls = %d, want 2", h.ResetCalls)
	}
}

func TestClose_CountsCallsAndReturnsErr(t *testing.T) {
	sentinel := errors.New("close boom")
	h := &Handle{CloseErr: sentinel}
	if err := h.Close(); err != sentinel {
		t.Fatalf("Close err = %v, want %v", err, sentinel)
	}
	if err := h.Close(); err != sentinel {
		t.Fatalf("Close err (2nd) = %v, want %v", err, sentinel)
	}
	if h.CloseCalls != 2 {
		t.Fatalf("CloseCalls = %d, want 2", h.CloseCalls)
	}
}

func TestErr_ReturnsSeededValue(t *testing.T) {
	sentinel := errors.New("err value")
	h := &Handle{ErrValue: sentinel}
	if err := h.Err(); err != sentinel {
		t.Fatalf("Err = %v, want %v", err, sentinel)
	}
	if err := (&Handle{}).Err(); err != nil {
		t.Fatalf("Err = %v, want nil for zero handle", err)
	}
}

func TestKVSnapshot_BuildsCanonicalTwoTokenSnapshot(t *testing.T) {
	snap := TestKVSnapshot()
	if snap == nil {
		t.Fatal("TestKVSnapshot returned nil")
	}
	if snap.Version != kv.SnapshotVersion {
		t.Fatalf("Version = %d, want %d", snap.Version, kv.SnapshotVersion)
	}
	if snap.Architecture != "gemma4_text" {
		t.Fatalf("Architecture = %q, want gemma4_text", snap.Architecture)
	}
	if len(snap.Tokens) != 2 || snap.Tokens[0] != 1 || snap.Tokens[1] != 2 {
		t.Fatalf("Tokens = %v, want [1 2]", snap.Tokens)
	}
	if len(snap.Layers) != 1 || len(snap.Layers[0].Heads) != 1 {
		t.Fatalf("Layers shape = %+v, want 1 layer / 1 head", snap.Layers)
	}
	head := snap.Layers[0].Heads[0]
	if len(head.KeyBytes) != 16 || len(head.ValueBytes) != 16 {
		t.Fatalf("head byte lengths = key %d / value %d, want 16/16", len(head.KeyBytes), len(head.ValueBytes))
	}
}

// seqOf returns an iter.Seq[string] over the given values.
func seqOf(values ...string) iter.Seq[string] {
	return func(yield func(string) bool) {
		for _, v := range values {
			if !yield(v) {
				return
			}
		}
	}
}
