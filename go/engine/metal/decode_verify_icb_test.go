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
// pos-independent per-layer tail for K=5 alongside its own live encodes,
// block 2 REPLAYS the recorded ranges (the engagement counter proves it),
// block 3's truncated K=3 records ITS OWN width (the adaptive draft cap
// wobbles K in production — recordings are per-K), block 4 replays the K=3
// recording — and every block's hiddens are byte-identical to a session with
// the lane force-disabled.
func TestVerifyTailICBRecordsReplaysAndMatchesLive(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	blocks := [][]int32{{1, 5, 3, 2, 7}, {4, 9, 6, 8, 2}, {3, 1, 5}, {2, 6, 4}}

	// the lane is opt-in (break-even receipt) — force it on for the candidate.
	wasDisabled := verifyTailICBDisabled
	verifyTailICBDisabled = false
	t.Cleanup(func() { verifyTailICBDisabled = wasDisabled })

	// control: the live tail encodes for every block.
	verifyTailICBDisabledForTest = true
	t.Cleanup(func() { verifyTailICBDisabledForTest = false })
	control := newVerifyTailQuantFixture(t)
	want := runVerifyBlocks(t, control, blocks)
	verifyTailICBDisabledForTest = false

	// candidate: record K=5 on block 1, replay it on block 2, record K=3 on
	// block 3, replay it on block 4.
	candidate := newVerifyTailQuantFixture(t)
	base := verifyTailReplays.Load()
	got := runVerifyBlocks(t, candidate, blocks)

	nLayers := len(candidate.state.specs)
	for _, k := range []int{5, 3} {
		vt := candidate.state.verifyTail[k]
		if vt == nil {
			t.Fatalf("no verify-tail ICB recorded for K=%d — the replay lane never armed", k)
		}
		if vt.key.k != k {
			t.Fatalf("K=%d recording carries key K=%d", k, vt.key.k)
		}
		for li := 1; li <= nLayers-2; li++ {
			if vt.ranges[li].Length == 0 {
				t.Fatalf("K=%d interior layer %d: no recorded range", k, li)
			}
		}
		if vt.ranges[0].Length != 0 || vt.ranges[nLayers-1].Length != 0 {
			t.Fatalf("K=%d: edge layers must stay live (unexpected recorded range)", k)
		}
	}
	// blocks 2 and 4 replay their widths' interiors; blocks 1 and 3 are
	// recording passes (0 executes).
	if replays := verifyTailReplays.Load() - base; replays != int64(2*(nLayers-2)) {
		t.Fatalf("replay executes = %d, want %d (blocks 2+4's interior layers)", replays, 2*(nLayers-2))
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
