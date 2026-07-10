// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

// newVerifyTailQuantFixture builds the e2b VERIFY shape at synthetic scale: a
// quant (4-bit) gemma4 arch with the per-layer-input tower on a recorded-ICB
// session — the exact lane the MTP verify folds through (verifyFoldSmallK on
// an icb session). numLayers=4 so the recordable interior [1, nLayers-2] is
// non-empty.
func newVerifyTailQuantFixture(t testing.TB) *ArchSession {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim, gs, bits = 4, 64, 64, 4
	const maxLen = 16
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
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

// runVerifyBlocks drives the MTP verify's exact entry (verifyFoldSmallK around
// verifyBatchedHiddens) over the blocks, each at the position the previous
// block left (all-accepted semantics — verify itself never advances pos).
// Returns deep copies of every block's hiddens (the scratch rows are reused).
func runVerifyBlocks(t *testing.T, sess *ArchSession, blocks [][]int32) [][][]byte {
	t.Helper()
	sess.state.verifyFoldSmallK = true
	defer func() { sess.state.verifyFoldSmallK = false }()
	out := make([][][]byte, len(blocks))
	pos := 0
	for b, ids := range blocks {
		sess.pos = pos
		hiddens, ok, err := sess.verifyBatchedHiddens(ids)
		if err != nil {
			t.Fatalf("block %d: verifyBatchedHiddens: %v", b, err)
		}
		if !ok {
			t.Fatalf("block %d: batched verify declined — this parity exercises nothing", b)
		}
		cp := make([][]byte, len(hiddens))
		for i, h := range hiddens {
			cp[i] = append([]byte(nil), h...)
		}
		out[b] = cp
		pos += len(ids)
	}
	return out
}

// TestVerifyTailICBRecordsReplaysAndMatchesLive pins the #372 verify-tail ICB
// contract on the e2b-shaped quant PLE fixture: block 1 RECORDS the
// pos-independent per-layer tail alongside its own live encodes, block 2
// REPLAYS the recorded ranges (the engagement counter proves it), block 3's
// truncated K misses the validity key and runs live — and every block's
// hiddens are byte-identical to a session with the lane force-disabled.
func TestVerifyTailICBRecordsReplaysAndMatchesLive(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	blocks := [][]int32{{1, 5, 3, 2, 7}, {4, 9, 6, 8, 2}, {3, 1, 5}}

	// control: the live tail encodes for every block.
	verifyTailICBDisabledForTest = true
	t.Cleanup(func() { verifyTailICBDisabledForTest = false })
	control := newVerifyTailQuantFixture(t)
	want := runVerifyBlocks(t, control, blocks)
	verifyTailICBDisabledForTest = false

	// candidate: record on block 1, replay on block 2, key-miss live on block 3.
	candidate := newVerifyTailQuantFixture(t)
	base := verifyTailReplays.Load()
	got := runVerifyBlocks(t, candidate, blocks)

	vt := candidate.state.verifyTail
	if vt == nil {
		t.Fatal("block 1 did not record the verify-tail ICB — the replay lane never armed")
	}
	if vt.key.k != len(blocks[0]) {
		t.Fatalf("recorded key K = %d, want %d", vt.key.k, len(blocks[0]))
	}
	nLayers := len(candidate.state.specs)
	for li := 1; li <= nLayers-2; li++ {
		if vt.ranges[li].Length == 0 {
			t.Fatalf("interior layer %d: no recorded range", li)
		}
	}
	if vt.ranges[0].Length != 0 || vt.ranges[nLayers-1].Length != 0 {
		t.Fatal("edge layers must stay live (unexpected recorded range)")
	}
	// exactly block 2 replays: (nLayers-2) interior executes. Block 1 is the
	// recording pass (0 executes); block 3's K=3 misses the k=5 key (0).
	if replays := verifyTailReplays.Load() - base; replays != int64(nLayers-2) {
		t.Fatalf("replay executes = %d, want %d (block 2's interior layers only)", replays, nLayers-2)
	}

	for b := range blocks {
		if len(got[b]) != len(want[b]) {
			t.Fatalf("block %d: hidden count %d != control %d", b, len(got[b]), len(want[b]))
		}
		for i := range got[b] {
			if !bytes.Equal(got[b][i], want[b][i]) {
				t.Fatalf("block %d row %d: verify-tail ICB hiddens diverge from the live tail encodes", b, i)
			}
		}
	}
}
