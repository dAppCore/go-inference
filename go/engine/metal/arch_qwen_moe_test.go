// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// sliceExpertQuant carves expert e's [outDim × inDim] quant weight out of a batched SwitchGLU tensor
// (numExperts stacked expert-major, matching MoEExpertsQuantSiLU's own gatePacked/downPacked byte-size
// convention in moe.go) — the same slicing TestMoEExpertsQuant/TestMoEExpertsQuantSiLU do inline for a
// composed reference, factored out here so
// TestEncQwenMoEHalf_SharedExpertDistinctWidth_Good can build a routed-expert reference across BOTH
// experts in its fixture without duplicating the offset arithmetic twice.
func sliceExpertQuant(w QuantWeight, e, outDim, inDim int) QuantWeight {
	packedPer := outDim * inDim * w.Bits / 8
	scalePer := outDim * (inDim / w.GroupSize) * bf16Size
	return QuantWeight{
		Packed:    w.Packed[e*packedPer : (e+1)*packedPer],
		Scales:    w.Scales[e*scalePer : (e+1)*scalePer],
		Biases:    w.Biases[e*scalePer : (e+1)*scalePer],
		GroupSize: w.GroupSize,
		Bits:      w.Bits,
	}
}

// TestEncQwenMoEHalf_SharedExpertDistinctWidth_Good is the #61 decode-time regression: encQwenMoEHalf
// must size its shared-expert MoEExpertsQuantSiLU dispatch from the shared expert's OWN width
// (moe.SharedDFF), not the routed experts' moe.ExpertDFF. The fixture gives routed ff=32 and shared
// ff=64 — genuinely distinct (the same 2x direction as real llama4 Scout's intermediate_size 8192
// routed vs intermediate_size_mlp 16384 shared, a live checkpoint that reaches this exact code path
// since it binds a shared expert and is otherwise arch-name-agnostic). Before #61 this call sized the
// shared dispatch off ff=32 while SharedGate/Up/Down's packed bytes are genuinely shaped for 64 —
// MoEExpertsQuantSiLU's own "batched expert weight size mismatch" length check caught it, so
// encQwenMoEHalf returned a hard error on EVERY decode step for a checkpoint shaped like this (not
// silently wrong output).
//
// topK == numExperts (both experts always selected) so the routed reference below can sum over every
// expert regardless of encQwenMoEHalf's own top-k pick order, without re-deriving that selection
// algorithm — the sum is order-independent once every expert is included. Dims are multiples of 32 (the
// smallest group size the metallib actually instantiates — affine_qmv only ships gs∈{32,64,128}).
func TestEncQwenMoEHalf_SharedExpertDistinctWidth_Good(t *testing.T) {
	requireNativeRuntime(t)
	const D, nE, topK, ff, sharedFF, gs, bits = 32, 2, 2, 32, 64, 32, 4
	const eps = float32(1e-5)

	preFFNorm := toBF16Bytes(syntheticFloat32(D, 2))
	router := quantWeightFixture(t, nE, D, gs, bits, 11)
	expGate := quantWeightFixture(t, nE*ff, D, gs, bits, 3)
	expUp := quantWeightFixture(t, nE*ff, D, gs, bits, 51)
	expDown := quantWeightFixture(t, nE*D, ff, gs, bits, 91)
	sGate := quantWeightFixture(t, sharedFF, D, gs, bits, 17)
	sUp := quantWeightFixture(t, sharedFF, D, gs, bits, 29)
	sDown := quantWeightFixture(t, D, sharedFF, gs, bits, 41)
	sSigmoid := quantWeightFixture(t, 1, D, gs, bits, 61)

	moe := &MoEQuantLayerWeights{
		NumExperts: nE, TopK: topK, ExpertDFF: ff, SharedDFF: sharedFF,
		ExpertGroupSize: gs, ExpertBits: bits,
		PreFFNormW: preFFNorm,
		Router:     router,
		ExpGate:    expGate, ExpUp: expUp, ExpDown: expDown,
		SharedGate: sGate, SharedUp: sUp, SharedDown: sDown, SharedSigmoid: sSigmoid,
	}

	hF32 := syntheticFloat32(D, 7)
	hBytes := f32ToBf16Slice(hF32)
	outBytes := make([]byte, D*bf16Size)

	var gotErr error
	var gotOut []byte
	err := withPinnedNoCopyBytes(hBytes, func(hBuf metal.MTLBuffer) error {
		return withPinnedNoCopyBytes(outBytes, func(outBuf metal.MTLBuffer) error {
			s := &archDecodeState{dModel: D, eps: eps, hBuf: hBuf}
			gotErr = s.encQwenMoEHalf(0, moe, outBuf)
			if gotErr == nil {
				gotOut = append([]byte(nil), unsafe.Slice((*byte)(outBuf.Contents()), D*bf16Size)...)
			}
			return nil
		})
	})
	if err != nil {
		t.Fatalf("withPinnedNoCopyBytes: %v", err)
	}
	if gotErr != nil {
		t.Fatalf("encQwenMoEHalf with genuinely distinct shared/routed widths: %v (want success — the shared dispatch must size off SharedDFF=%d, not ExpertDFF=%d)", gotErr, sharedFF, ff)
	}

	// Host reference: the SAME router primitive (projQuantAttn) feeding a softmax over both experts —
	// topK==numExperts makes the combine sum order-independent — then each routed expert and the shared
	// expert via swigluSiLUHost (the pre-fusion oracle TestEncQwenMoESharedExpertMatchesDevice also
	// uses), the shared one at its OWN width (sharedFF), not ff.
	normed := rmsNormHostF32(hF32, bf16VecToF32(preFFNorm), eps)
	logits, err := projQuantAttn(router, normed, D, nE, nil)
	if err != nil {
		t.Fatalf("reference router: %v", err)
	}
	maxL := math.Inf(-1)
	for _, l := range logits {
		if float64(l) > maxL {
			maxL = float64(l)
		}
	}
	probs := make([]float64, nE)
	denom := 0.0
	for e, l := range logits {
		probs[e] = math.Exp(float64(l) - maxL)
		denom += probs[e]
	}
	routed := make([]float32, D)
	for e := range nE {
		ge := sliceExpertQuant(expGate, e, ff, D)
		ue := sliceExpertQuant(expUp, e, ff, D)
		de := sliceExpertQuant(expDown, e, D, ff)
		se, err := swigluSiLUHost(normed, ge, ue, de, D, ff)
		if err != nil {
			t.Fatalf("reference routed expert %d: %v", e, err)
		}
		w := float32(probs[e] / denom)
		for d := range routed {
			routed[d] += se[d] * w
		}
	}
	sigma, err := sharedGateSigmoid(sSigmoid, normed, D)
	if err != nil {
		t.Fatalf("reference shared gate: %v", err)
	}
	shared, err := swigluSiLUHost(normed, sGate, sUp, sDown, D, sharedFF)
	if err != nil {
		t.Fatalf("reference shared expert: %v", err)
	}
	want := make([]float32, D)
	for d := range want {
		want[d] = hF32[d] + routed[d] + float32(sigma)*shared[d]
	}
	wantBF := f32ToBf16Slice(want)

	worst, at := bf16UlpDist(gotOut, wantBF)
	t.Logf("encQwenMoEHalf (distinct shared/routed widths) vs host reference: worst %d bf16 ULP at elem %d", worst, at)
	if worst > 8 {
		t.Fatalf("encQwenMoEHalf diverges from the host reference by %d bf16 ULP at elem %d — beyond the expected bf16-rounding-schedule tolerance (the device kernel's per-stage rounding vs the host's continuous float64 SiLU), likely a wiring bug (wrong weight/expert/width), not rounding", worst, at)
	}
}

// TestEncQwenMoEHalf_SharedDFFZeroFallsBackToExpertDFF_Ugly is the #61 zero-change guard at the decode
// call site: a MoEQuantLayerWeights built without SharedDFF populated (SharedDFF's zero value — either
// a pre-#61 hand-built fixture, or an arch that never declares a distinct shared width so moeToQuant's
// own fallback already made it equal to ExpertDFF) must still decode correctly, sizing the shared
// dispatch off ff — encQwenMoEHalf's own defensive fallback (mirroring moeToQuant's) — because here the
// shared expert's REAL width genuinely IS ff (routed and shared equal), so ff is not just a fallback of
// convenience, it is the correct width.
func TestEncQwenMoEHalf_SharedDFFZeroFallsBackToExpertDFF_Ugly(t *testing.T) {
	requireNativeRuntime(t)
	const D, nE, topK, ff, gs, bits = 32, 2, 2, 32, 32, 4
	const eps = float32(1e-5)

	preFFNorm := toBF16Bytes(syntheticFloat32(D, 2))
	router := quantWeightFixture(t, nE, D, gs, bits, 11)
	expGate := quantWeightFixture(t, nE*ff, D, gs, bits, 3)
	expUp := quantWeightFixture(t, nE*ff, D, gs, bits, 51)
	expDown := quantWeightFixture(t, nE*D, ff, gs, bits, 91)
	sGate := quantWeightFixture(t, ff, D, gs, bits, 17)
	sUp := quantWeightFixture(t, ff, D, gs, bits, 29)
	sDown := quantWeightFixture(t, D, ff, gs, bits, 41)
	sSigmoid := quantWeightFixture(t, 1, D, gs, bits, 61)

	moe := &MoEQuantLayerWeights{
		NumExperts: nE, TopK: topK, ExpertDFF: ff, // SharedDFF deliberately left zero
		ExpertGroupSize: gs, ExpertBits: bits,
		PreFFNormW: preFFNorm,
		Router:     router,
		ExpGate:    expGate, ExpUp: expUp, ExpDown: expDown,
		SharedGate: sGate, SharedUp: sUp, SharedDown: sDown, SharedSigmoid: sSigmoid,
	}

	hBytes := f32ToBf16Slice(syntheticFloat32(D, 7))
	outBytes := make([]byte, D*bf16Size)
	var gotErr error
	err := withPinnedNoCopyBytes(hBytes, func(hBuf metal.MTLBuffer) error {
		return withPinnedNoCopyBytes(outBytes, func(outBuf metal.MTLBuffer) error {
			s := &archDecodeState{dModel: D, eps: eps, hBuf: hBuf}
			gotErr = s.encQwenMoEHalf(0, moe, outBuf)
			return nil
		})
	})
	if err != nil {
		t.Fatalf("withPinnedNoCopyBytes: %v", err)
	}
	if gotErr != nil {
		t.Fatalf("encQwenMoEHalf with SharedDFF==0 (equal-width arch): %v (want success — must fall back to ExpertDFF)", gotErr)
	}
}
