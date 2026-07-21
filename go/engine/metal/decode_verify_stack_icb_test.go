// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

// newVerifyStackHybridQuantFixture builds the whole-stack lane's proving shape
// at synthetic scale: a quant (4-bit) gemma4 arch with the per-layer-input
// tower, a MIXED sliding/global layer pattern and a SMALL window (8) on a
// recorded-ICB session — so verify blocks cross the sliding ring wrap within a
// few passes and the staged per-row landing (the slot-rebind machinery) is
// exercised, not just the pre-wrap direct landing.
func newVerifyStackHybridQuantFixture(t testing.TB) *ArchSession {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim, gs, bits = 4, 64, 64, 4
	const maxLen, window = 64, 8
	layerTypes := []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"}
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization:  &model.QuantConfig{GroupSize: gs, Bits: bits},
		SlidingWindow: window,
		LayerTypes:    layerTypes,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	addPLETensors(t, ts, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	if !g.HasPLE() {
		t.Fatal("fixture model should have the per-layer-input tower")
	}
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	t.Cleanup(func() { sess.Close() })
	return sess
}

// verifyStackCacheBytes snapshots the written region of every owner layer's
// recorded-ICB K/V caches — the KV-write half of the byte-identity contract
// (hiddens alone would not catch a landing at a wrong slot that a later pass
// happens not to attend).
func verifyStackCacheBytes(t *testing.T, sess *ArchSession, endPos int) [][]byte {
	t.Helper()
	r := sess.state.icb
	if r == nil {
		t.Fatal("fixture must run the recorded-ICB session")
	}
	var out [][]byte
	for li := range sess.state.specs {
		if !sess.state.specs[li].OwnsCache() {
			continue
		}
		rows := min(endPos, r.cacheRows[li])
		n := rows * r.rowBytes[li]
		for _, buf := range []interface{ Contents() unsafe.Pointer }{r.kCaches[li], r.vCaches[li]} {
			cp := make([]byte, n)
			copy(cp, unsafe.Slice((*byte)(buf.Contents()), n))
			out = append(out, cp)
		}
	}
	return out
}

// TestVerifyStackICBRecordsReplaysAndMatchesLive pins the whole-stack verify
// ICB contract on a hybrid sliding+global quant PLE fixture:
//
//   - blocks 1-2 straddle the sliding ring wrap, so the shape phase flips from
//     the direct landing to the staged per-row landing and the width
//     re-records once under the staged key;
//   - blocks 3-4 REPLAY the K=5 staged recording at two consecutive positions
//     whose ring slots differ (block 4's slots cross the wrap inside the pass)
//     — the slot-rebind proof;
//   - K=2 and K=6 record and replay their own widths (K=6 regrows the fold
//     slabs, proving the key tracks slab identity);
//   - every block's hiddens AND the final KV cache bytes are byte-identical to
//     a session with the lane force-disabled, and the engagement counter
//     proves the replays actually executed.
func TestVerifyStackICBRecordsReplaysAndMatchesLive(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	prevDisabled := verifyStackICBDisabled
	verifyStackICBDisabled = false // the lane is opt-in until its replay race is fixed; parity still gates the mechanism
	t.Cleanup(func() { verifyStackICBDisabled = prevDisabled })
	blocks := [][]int32{
		{1, 5, 3, 2, 7},    // K=5 pos 0  — pre-wrap (direct) phase: records
		{4, 9, 6, 8, 2},    // K=5 pos 5  — staged phase: re-records (phase flip)
		{3, 1, 5, 9, 4},    // K=5 pos 10 — REPLAY (slots 2..6)
		{2, 6, 4, 1, 8},    // K=5 pos 15 — REPLAY (slots 7,0,1,2,3 — wrap inside the pass)
		{7, 3},             // K=2 pos 20 — records
		{5, 1},             // K=2 pos 22 — REPLAY
		{2, 4, 6, 8, 1, 3}, // K=6 pos 24 — records (fold slabs regrow at K=6)
		{9, 7, 5, 3, 1, 2}, // K=6 pos 30 — REPLAY
	}

	// control: the live fold for every block.
	verifyStackICBDisabledForTest = true
	control := newVerifyStackHybridQuantFixture(t)
	want := runVerifyBlocks(t, control, blocks)
	endPos := 0
	for _, b := range blocks {
		endPos += len(b)
	}
	wantKV := verifyStackCacheBytes(t, control, endPos)
	verifyStackICBDisabledForTest = false

	// candidate: the lane records and replays (default-on).
	candidate := newVerifyStackHybridQuantFixture(t)
	base := verifyStackReplays.Load()
	got := runVerifyBlocks(t, candidate, blocks)
	gotKV := verifyStackCacheBytes(t, candidate, endPos)

	for _, k := range []int{5, 2, 6} {
		vs := candidate.state.verifyStack[k]
		if vs == nil {
			t.Fatalf("no whole-stack ICB recorded for K=%d — the lane never armed", k)
		}
		if vs.key.k != k || !vs.key.staged {
			t.Fatalf("K=%d recording carries key %+v — want its own staged-phase key", k, vs.key)
		}
		if len(vs.rebinds) == 0 || len(vs.dyn) == 0 {
			t.Fatalf("K=%d recording has %d rebinds / %d dynamic lengths — the per-pass machinery is missing", k, len(vs.rebinds), len(vs.dyn))
		}
	}
	// blocks 3, 4, 6 and 8 replay; 1, 2, 5 and 7 are recording passes.
	if replays := verifyStackReplays.Load() - base; replays != 4 {
		t.Fatalf("whole-stack replay executes = %d, want 4 (blocks 3, 4, 6 and 8)", replays)
	}

	for b := range blocks {
		if len(got[b]) != len(want[b]) {
			t.Fatalf("block %d: hidden count %d != control %d", b, len(got[b]), len(want[b]))
		}
		for i := range got[b] {
			if !bytes.Equal(got[b][i], want[b][i]) {
				t.Fatalf("block %d row %d: whole-stack ICB hiddens diverge from the live fold", b, i)
			}
		}
	}
	if len(gotKV) != len(wantKV) {
		t.Fatalf("KV snapshot count %d != control %d", len(gotKV), len(wantKV))
	}
	for i := range gotKV {
		if !bytes.Equal(gotKV[i], wantKV[i]) {
			t.Fatalf("KV cache snapshot %d: whole-stack ICB cache bytes diverge from the live fold", i)
		}
	}
}

// TestVerifyStackICBKillSwitchKeepsLiveFold pins the kill-switch contract:
// with the lane force-disabled the fold neither records nor replays, and the
// hiddens are the live fold's own.
func TestVerifyStackICBKillSwitchKeepsLiveFold(t *testing.T) {
	prevDisabled := verifyStackICBDisabled
	verifyStackICBDisabled = false
	t.Cleanup(func() { verifyStackICBDisabled = prevDisabled })
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	verifyStackICBDisabledForTest = true
	t.Cleanup(func() { verifyStackICBDisabledForTest = false })
	sess := newVerifyStackHybridQuantFixture(t)
	base := verifyStackReplays.Load()
	runVerifyBlocks(t, sess, [][]int32{{1, 5, 3, 2, 7}, {4, 9, 6, 8, 2}})
	if len(sess.state.verifyStack) != 0 {
		t.Fatal("kill-switch set but a whole-stack recording was armed")
	}
	if verifyStackReplays.Load() != base {
		t.Fatal("kill-switch set but a whole-stack replay executed")
	}
}
