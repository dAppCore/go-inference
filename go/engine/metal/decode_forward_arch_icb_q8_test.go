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

	// v1 coherence gates: the batched dense lane declines (falls back to the
	// per-token replay — verifyBatchedHiddens reports batched=false)…
	q8.state.verifyFoldSmallK = true
	_, batched, err := q8.verifyBatchedHiddens([]int32{2, 4, 6})
	q8.state.verifyFoldSmallK = false
	if err != nil {
		t.Fatalf("verifyBatchedHiddens: %v", err)
	}
	if batched {
		t.Fatal("batched dense lane must DECLINE on q8 ICB caches (bf16 rows would corrupt int8)")
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
