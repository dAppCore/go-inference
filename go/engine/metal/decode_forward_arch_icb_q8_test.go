// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

// newKVQ8ICBFixture builds a GLOBAL-attention quant session at a q8-qualifying
// geometry (hd=256, kvd 64-aligned) with maxLen past sdpa2PassMinKV, so the
// recorded ICB exercises BOTH the q8 store rebinds and the 2-pass q8 SDPA.
func newKVQ8ICBFixture(t testing.TB) *ArchSession {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 256, 256, 64
	const numLayers, gs, bits = 3, 64, 4
	const maxLen = 2048
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	lm, err := model.Assemble(quantGemma4Tensors(t, arch, gs, bits), arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	t.Cleanup(func() { sess.Close() })
	return sess
}

// TestKVQ8ICBDecodeTracksBF16 pins the #367 slice-B contract: with
// LTHN_KV_Q8_ICB armed, the recorded ICB stores GLOBAL K/V as int8 + group
// scales and reads them through the q8 SDPA — the per-step hiddens must stay
// within quantisation distance of the bf16-ICB session (a structural bug —
// wrong offset, missing rebind, stale staging — produces garbage far outside
// it). Also pins the v1 coherence gates: the batched dense lane and KV
// snapshots decline on q8 sessions.
func TestKVQ8ICBDecodeTracksBF16(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	// bf16 reference session first (the flag is read at session build).
	ref := newKVQ8ICBFixture(t)
	if ref.state.icb == nil {
		t.Fatal("fixture must record an ICB")
	}
	if ref.state.icb.hasKVQ8() {
		t.Fatal("reference session must be bf16 (flag leaked)")
	}

	kvQ8ICBForTest = true
	t.Cleanup(func() { kvQ8ICBForTest = false })
	q8 := newKVQ8ICBFixture(t)
	if q8.state.icb == nil {
		t.Fatal("q8 fixture must record an ICB")
	}
	if !q8.state.icb.hasKVQ8() {
		t.Fatal("q8 session did not arm q8 KV — no global layer qualified; the lane was never exercised")
	}
	armed := 0
	for li := range q8.state.specs {
		if q8.state.icb.kvQ8.on(li) {
			if q8.state.specs[li].Attention != model.GlobalAttention {
				t.Fatalf("layer %d: q8 armed on a non-global layer", li)
			}
			armed++
		}
	}
	t.Logf("q8 armed on %d global owner layers", armed)

	// step the same token sequence through both; hiddens must track within
	// quantisation distance from the FIRST step (structural failures are
	// orders of magnitude outside this).
	ids := []int32{1, 5, 3, 9, 7, 2, 4, 8, 6, 1, 5, 3}
	worstAll := 0.0
	for i, id := range ids {
		hq, err := q8.stepID(id)
		if err != nil {
			t.Fatalf("q8 stepID(%d): %v", i, err)
		}
		hqCopy := append([]byte(nil), hq...)
		hr, err := ref.stepID(id)
		if err != nil {
			t.Fatalf("ref stepID(%d): %v", i, err)
		}
		worst, scale := 0.0, 0.0
		for j := 0; j+1 < len(hr); j += 2 {
			a := float64(bf16ToF32(hqCopy[j], hqCopy[j+1]))
			b := float64(bf16ToF32(hr[j], hr[j+1]))
			if d := math.Abs(a - b); d > worst {
				worst = d
			}
			if m := math.Abs(b); m > scale {
				scale = m
			}
		}
		if worst > worstAll {
			worstAll = worst
		}
		if worst > 0.05*math.Max(scale, 1) {
			t.Fatalf("step %d: q8 hidden diverges structurally: worst |Δ|=%g (hidden scale %g)", i, worst, scale)
		}
	}
	t.Logf("q8 tracked bf16 over %d steps: worst |Δ| = %.5g", len(ids), worstAll)

	// slice C: the batched verify ENGAGES on q8 (multiQ q8 read + staged
	// rows-store landing) while shallow — the fold is the supported shape.
	q8.state.verifyFoldSmallK = true
	_, batched, err := q8.verifyBatchedHiddens([]int32{2, 4, 6})
	q8.state.verifyFoldSmallK = false
	if err != nil {
		t.Fatalf("verifyBatchedHiddens: %v", err)
	}
	if !batched {
		t.Fatal("batched verify must ENGAGE on q8 ICB caches at shallow depth (slice C)")
	}
	// …and KV snapshots error rather than misread int8 as bf16.
	q8li := -1
	for li := range q8.state.specs {
		if q8.state.icb.kvQ8.on(li) {
			q8li = li
			break
		}
	}
	if _, _, _, _, serr := q8.snapshotCacheViews(q8li); serr == nil {
		t.Fatal("snapshotCacheViews must error on a q8 ICB layer")
	}
}

// TestKVQ8ICBBatchedPrefillMatchesSequential pins slice C: a prompt-scale
// chunk ENGAGES the batched dense fold on a q8 session (staged rows-store
// landing + the multi-query causal q8 read) and its per-row hiddens track a
// second q8 session stepping the same ids through the per-token replay — the
// landed q8 bytes are the same store math either way; only the attention
// summation order differs (the fold's token-identity tier).
func TestKVQ8ICBBatchedPrefillMatchesSequential(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	kvQ8ICBForTest = true
	t.Cleanup(func() { kvQ8ICBForTest = false })
	batchedSess := newKVQ8ICBFixture(t)
	seqSess := newKVQ8ICBFixture(t)
	if !batchedSess.state.icb.hasKVQ8() {
		t.Fatal("fixture did not arm q8")
	}

	ids := make([]int32, 40) // > batchedDenseICBMaxRows: the fold lane
	for i := range ids {
		ids[i] = int32((i*7 + 3) % 60)
	}
	embs := make([][]byte, len(ids))
	for i, id := range ids {
		emb, err := batchedSess.embedID(id)
		if err != nil {
			t.Fatalf("embedID: %v", err)
		}
		embs[i] = append([]byte(nil), emb...)
	}
	var out [][]byte
	var ok bool
	var err error
	withAutoreleasePool(func() {
		out, ok, err = batchedSess.state.stepTokensBatchedDense(embs, 0)
	})
	if err != nil {
		t.Fatalf("stepTokensBatchedDense: %v", err)
	}
	if !ok {
		t.Fatal("prompt-scale batch must ENGAGE the fold on a q8 session (slice C)")
	}
	batchedRows := make([][]byte, len(out))
	for i := range out {
		batchedRows[i] = append([]byte(nil), out[i]...)
	}

	worstAll := 0.0
	for i, id := range ids {
		hs, serr := seqSess.stepID(id)
		if serr != nil {
			t.Fatalf("sequential stepID(%d): %v", i, serr)
		}
		worst, hscale := 0.0, 0.0
		for j := 0; j+1 < len(hs); j += 2 {
			a := float64(bf16ToF32(batchedRows[i][j], batchedRows[i][j+1]))
			b := float64(bf16ToF32(hs[j], hs[j+1]))
			if d := math.Abs(a - b); d > worst {
				worst = d
			}
			if m := math.Abs(b); m > hscale {
				hscale = m
			}
		}
		if worst > worstAll {
			worstAll = worst
		}
		if worst > 0.05*math.Max(hscale, 1) {
			t.Fatalf("row %d: batched q8 prefill diverges structurally: worst |Δ|=%g (scale %g)", i, worst, hscale)
		}
	}
	t.Logf("batched q8 prefill tracked sequential over %d rows: worst |Δ| = %.5g", len(ids), worstAll)

	// the verify block over the just-prefilled q8 caches engages too and tracks
	// the sequential continuation.
	batchedSess.pos = len(ids)
	draft := []int32{2, 4, 6, 8, 1}
	batchedSess.state.verifyFoldSmallK = true
	vh, vBatched, verr := batchedSess.verifyBatchedHiddens(draft)
	batchedSess.state.verifyFoldSmallK = false
	if verr != nil {
		t.Fatalf("verifyBatchedHiddens: %v", verr)
	}
	if !vBatched {
		t.Fatal("verify must ENGAGE on the q8 session at shallow depth")
	}
	for i, id := range draft {
		hs, serr := seqSess.stepID(id)
		if serr != nil {
			t.Fatalf("verify sequential stepID(%d): %v", i, serr)
		}
		worst, hscale := 0.0, 0.0
		for j := 0; j+1 < len(hs); j += 2 {
			a := float64(bf16ToF32(vh[i][j], vh[i][j+1]))
			b := float64(bf16ToF32(hs[j], hs[j+1]))
			if d := math.Abs(a - b); d > worst {
				worst = d
			}
			if m := math.Abs(b); m > hscale {
				hscale = m
			}
		}
		if worst > 0.05*math.Max(hscale, 1) {
			t.Fatalf("verify row %d: diverges structurally: worst |Δ|=%g (scale %g)", i, worst, hscale)
		}
	}
	t.Logf("q8 batched verify tracked sequential over %d rows", len(draft))
}
