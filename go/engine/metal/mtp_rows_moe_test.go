// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
)

// mtp_rows_moe_test.go — the #53 byte-identity pin: mtpRowsMoEBatched (mtp_rows_moe.go) grouped
// by expert must equal K independent MoEBlockQuantInto calls, bit for bit. Small synthetic quant
// MoE weights (test_helpers_test.go's quantizeProj/syntheticFloat32/toBF16Bytes — the same helpers
// moe_session_test.go's gemma4-shaped fixtures use), no model/session/tokenizer needed: the block
// primitive takes raw hidden bytes in, raw hidden bytes out.

// mtpRowsMoETestDims is sized so the multi-row tiled route actually engages: dModel/dFF/expertDFF
// are all 512 (the qmv_rows fast-twin envelope needs outDim%8==0 && inDim%512==0 for every
// projection — local AND expert, gate/up AND down), and numExperts==topK==2 makes EVERY row route
// to BOTH experts, so each expert's pair-group is deterministically size K — no reliance on
// router tie-breaks to exercise the grouped lane.
const (
	mtpRowsMoETestDModel     = 512
	mtpRowsMoETestDFF        = 512
	mtpRowsMoETestExpertDFF  = 512
	mtpRowsMoETestNumExperts = 2
	mtpRowsMoETestTopK       = 2
	mtpRowsMoETestGroupSize  = 64
	mtpRowsMoETestBits       = 4
	mtpRowsMoETestEps        = 1e-5
)

func mtpRowsMoETestWeights(t testing.TB) MoEQuantLayerWeights {
	t.Helper()
	const dModel, dFF, expertDFF, numExperts = mtpRowsMoETestDModel, mtpRowsMoETestDFF, mtpRowsMoETestExpertDFF, mtpRowsMoETestNumExperts
	const gs, bits = mtpRowsMoETestGroupSize, mtpRowsMoETestBits
	mkNorm := func(salt int) []byte { return toBF16Bytes(syntheticFloat32(dModel, salt)) }
	mkQuant := func(outDim, inDim, salt int) QuantWeight {
		p, s, b := quantizeProj(t, outDim, inDim, gs, bits, salt)
		return QuantWeight{Packed: p, Scales: s, Biases: b, GroupSize: gs, Bits: bits}
	}
	return MoEQuantLayerWeights{
		NumExperts: numExperts, TopK: mtpRowsMoETestTopK, ExpertDFF: expertDFF,
		ExpertGroupSize: gs, ExpertBits: bits,
		LocalGroupSize: gs, LocalBits: bits,
		RouterGroupSize: gs, RouterBits: bits,
		PreFFNormW: mkNorm(1), PreFFNorm2W: mkNorm(2),
		PostFFNorm1W: mkNorm(3), PostFFNorm2W: mkNorm(4), PostFFNormW: mkNorm(5),
		LocalGate: mkQuant(dFF, dModel, 6), LocalUp: mkQuant(dFF, dModel, 7), LocalDown: mkQuant(dModel, dFF, 8),
		RouterNormWScaled: mkNorm(9), Router: mkQuant(numExperts, dModel, 10),
		ExpGate: mkQuant(numExperts*expertDFF, dModel, 11), ExpUp: mkQuant(numExperts*expertDFF, dModel, 12),
		ExpDown: mkQuant(numExperts*dModel, expertDFF, 13),
	}
}

// TestMTPRowsMoEBatchedMatchesPerRow_Good pins the #53 contract: the grouped-by-expert batched
// block equals K sequential per-row block calls, byte for byte, AND the compare actually engaged
// the multi-row tiled kernel (maxGroup > 1) — a vacuous all-groups-of-one compare would prove
// nothing about the grouping this lane exists for.
func TestMTPRowsMoEBatchedMatchesPerRow_Good(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, dFF, K = mtpRowsMoETestDModel, mtpRowsMoETestDFF, 4
	w := mtpRowsMoETestWeights(t)

	if !mtpRowsMoEEligible(w, dModel, dFF) {
		t.Fatal("fixture geometry declined mtpRowsMoEEligible — the fixture must exercise the batched lane")
	}

	rowBytes := dModel * bf16Size
	hSlab := toBF16Bytes(syntheticFloat32(K*dModel, 99))
	if len(hSlab) != K*rowBytes {
		t.Fatalf("fixture hSlab bytes = %d, want %d", len(hSlab), K*rowBytes)
	}

	got, ok, err := mtpRowsMoEBatched(hSlab, w, dModel, dFF, K, mtpRowsMoETestEps)
	if err != nil {
		t.Fatalf("mtpRowsMoEBatched: %v", err)
	}
	if !ok {
		t.Fatal("mtpRowsMoEBatched declined on an eligible fixture")
	}
	if len(got) != K*rowBytes {
		t.Fatalf("mtpRowsMoEBatched out bytes = %d, want %d", len(got), K*rowBytes)
	}

	want := make([]byte, 0, K*rowBytes)
	for r := range K {
		row := hSlab[r*rowBytes : (r+1)*rowBytes]
		out, rerr := MoEBlockQuantInto(nil, row, w, dModel, dFF, mtpRowsMoETestEps)
		if rerr != nil {
			t.Fatalf("MoEBlockQuantInto row %d: %v", r, rerr)
		}
		if len(out) != rowBytes {
			t.Fatalf("MoEBlockQuantInto row %d bytes = %d, want %d", r, len(out), rowBytes)
		}
		want = append(want, out...)
	}

	maxGroup := mtpRowsMoEMaxGroupSize.Load()
	if !bytes.Equal(got, want) {
		firstDiff := -1
		for i := range got {
			if got[i] != want[i] {
				firstDiff = i
				break
			}
		}
		t.Fatalf("mtpRowsMoEBatched diverged from the per-row reference at byte %d (K=%d, maxGroup=%d)", firstDiff, K, maxGroup)
	}
	if maxGroup < 2 {
		t.Fatalf("fixture never grouped >1 pair onto one expert (maxGroup=%d) — the compare never engaged the multi-row tiled lane", maxGroup)
	}
	t.Logf("mtpRowsMoEBatched == %d x MoEBlockQuantInto, byte for byte; max expert group size %d (K=%d numExperts=%d topK=%d)",
		K, maxGroup, K, mtpRowsMoETestNumExperts, mtpRowsMoETestTopK)
}

// TestMTPRowsMoEBatchedMatchesPerRow_Bad exercises K=1 — the degenerate single-row block, where
// every expert group is size 1 and the whole grouped path collapses to the same per-row qmv the
// sequential lane already uses. Still must equal the per-row reference exactly.
func TestMTPRowsMoEBatchedMatchesPerRow_Bad(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, dFF, K = mtpRowsMoETestDModel, mtpRowsMoETestDFF, 1
	w := mtpRowsMoETestWeights(t)
	hSlab := toBF16Bytes(syntheticFloat32(K*dModel, 77))

	got, ok, err := mtpRowsMoEBatched(hSlab, w, dModel, dFF, K, mtpRowsMoETestEps)
	if err != nil {
		t.Fatalf("mtpRowsMoEBatched: %v", err)
	}
	if !ok {
		t.Fatal("mtpRowsMoEBatched declined on K=1")
	}
	want, err := MoEBlockQuantInto(nil, hSlab, w, dModel, dFF, mtpRowsMoETestEps)
	if err != nil {
		t.Fatalf("MoEBlockQuantInto: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("mtpRowsMoEBatched K=1 diverged from MoEBlockQuantInto")
	}
}

// mtpRowsMoETestWeightsFused is mtpRowsMoETestWeights with the expert gate/up fused into ONE
// ExpGateUp tensor via fuseExpertGateUpQuant (load_shared.go) — the SAME synthesis
// loadedToQuant's moeToQuant runs for every gemma4 checkpoint (Arch.FuseExpertGateUp), split or
// already-fused. ExpGate/ExpUp are cleared, mirroring moeToQuant's own post-fuse cleanup, so a
// test on this fixture can only pass via the fused code path — there is no split fallback left to
// accidentally exercise instead.
func mtpRowsMoETestWeightsFused(t testing.TB) MoEQuantLayerWeights {
	t.Helper()
	w := mtpRowsMoETestWeights(t)
	const dModel, expertDFF, numExperts = mtpRowsMoETestDModel, mtpRowsMoETestExpertDFF, mtpRowsMoETestNumExperts
	const gs, bits = mtpRowsMoETestGroupSize, mtpRowsMoETestBits
	w.ExpGateUp = fuseExpertGateUpQuant(w.ExpGate, w.ExpUp, numExperts, expertDFF, dModel, gs, bits)
	w.ExpGate, w.ExpUp = QuantWeight{}, QuantWeight{}
	return w
}

// TestMTPRowsMoEBatchedMatchesPerRow_FusedGateUp_Good is
// TestMTPRowsMoEBatchedMatchesPerRow_Good's fused-ExpGateUp twin — the LIVE gemma4 26B-A4B shape
// (Arch.FuseExpertGateUp synthesises this at every load, TestLoadGemma4QuantMoEFusedGateUpMatchesSplitExperts).
// Byte-identical to K sequential MoEBlockQuantInto calls on the SAME fused weights, with the SAME
// engagement guard as the split-case test (maxGroup > 1) — a lane that only proved split geometry
// would never have a receipt for what actually ships.
func TestMTPRowsMoEBatchedMatchesPerRow_FusedGateUp_Good(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, dFF, K = mtpRowsMoETestDModel, mtpRowsMoETestDFF, 4
	w := mtpRowsMoETestWeightsFused(t)

	if !mtpRowsMoEEligible(w, dModel, dFF) {
		t.Fatal("fused fixture geometry declined mtpRowsMoEEligible — the fixture must exercise the fused-expert branch")
	}

	rowBytes := dModel * bf16Size
	hSlab := toBF16Bytes(syntheticFloat32(K*dModel, 55))
	if len(hSlab) != K*rowBytes {
		t.Fatalf("fixture hSlab bytes = %d, want %d", len(hSlab), K*rowBytes)
	}

	got, ok, err := mtpRowsMoEBatched(hSlab, w, dModel, dFF, K, mtpRowsMoETestEps)
	if err != nil {
		t.Fatalf("mtpRowsMoEBatched: %v", err)
	}
	if !ok {
		t.Fatal("mtpRowsMoEBatched declined on an eligible fused fixture")
	}
	if len(got) != K*rowBytes {
		t.Fatalf("mtpRowsMoEBatched out bytes = %d, want %d", len(got), K*rowBytes)
	}

	want := make([]byte, 0, K*rowBytes)
	for r := range K {
		row := hSlab[r*rowBytes : (r+1)*rowBytes]
		out, rerr := MoEBlockQuantInto(nil, row, w, dModel, dFF, mtpRowsMoETestEps)
		if rerr != nil {
			t.Fatalf("MoEBlockQuantInto row %d: %v", r, rerr)
		}
		want = append(want, out...)
	}

	maxGroup := mtpRowsMoEMaxGroupSize.Load()
	if !bytes.Equal(got, want) {
		firstDiff := -1
		for i := range got {
			if got[i] != want[i] {
				firstDiff = i
				break
			}
		}
		t.Fatalf("mtpRowsMoEBatched (fused) diverged from the per-row reference at byte %d (K=%d, maxGroup=%d)", firstDiff, K, maxGroup)
	}
	if maxGroup < 2 {
		t.Fatalf("fused fixture never grouped >1 pair onto one expert (maxGroup=%d) — the compare never engaged the multi-row tiled lane", maxGroup)
	}
	t.Logf("mtpRowsMoEBatched (fused ExpGateUp) == %d x MoEBlockQuantInto, byte for byte; max expert group size %d", K, maxGroup)
}

// TestMTPRowsMoEEligible_Ugly pins the decline discriminators: gpt_oss (ClampedSwiGLU), qwen (a
// bound SharedGate), and a malformed (wrong-shaped) fused ExpGateUp tensor all decline —
// mtpRowsMoEBatched must never be reached on a geometry it does not implement or cannot validate.
// A WELL-FORMED fused checkpoint is the live gemma4 26B-A4B shape and IS eligible — see
// TestMTPRowsMoEBatchedMatchesPerRow_FusedGateUp_Good.
func TestMTPRowsMoEEligible_Ugly(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, dFF = mtpRowsMoETestDModel, mtpRowsMoETestDFF
	base := mtpRowsMoETestWeights(t)

	gptOSS := base
	gptOSS.ClampedSwiGLU = true
	if mtpRowsMoEEligible(gptOSS, dModel, dFF) {
		t.Fatal("gpt_oss (ClampedSwiGLU) must decline — it decodes on encGptOssMoEHalf, not this lane")
	}

	qwen := base
	qwen.SharedGate = QuantWeight{Packed: []byte{1}, Scales: []byte{1}, Biases: []byte{1}, GroupSize: 1, Bits: 4}
	if mtpRowsMoEEligible(qwen, dModel, dFF) {
		t.Fatal("qwen (bound SharedGate) must decline — it decodes on encQwenMoEHalf, not this lane")
	}

	malformedFused := base
	malformedFused.ExpGateUp = QuantWeight{Packed: []byte{1}, Scales: []byte{1}, Biases: []byte{1}, GroupSize: 1, Bits: 4}
	if mtpRowsMoEEligible(malformedFused, dModel, dFF) {
		t.Fatal("a wrong-shaped fused ExpGateUp tensor must decline — quantWeightViewsForShape validates the [numExperts*2*expertDFF, dModel] geometry")
	}
}

// TestMTPRowsMoEForced_Good pins the LTHN_MTP_ROWS_MOE env parsing — mirrors mtpVerifyFoldForced's
// own test idiom (mtp_exact_lane_test.go).
func TestMTPRowsMoEForced_Good(t *testing.T) {
	if os.Getenv("LTHN_MTP_ROWS_MOE") == "1" && !mtpRowsMoEForced {
		t.Fatal("LTHN_MTP_ROWS_MOE=1 in the environment but mtpRowsMoEForced is false")
	}
	if os.Getenv("LTHN_MTP_ROWS_MOE") != "1" && mtpRowsMoEForced {
		t.Fatal("mtpRowsMoEForced is true without LTHN_MTP_ROWS_MOE=1 in the environment")
	}
}
