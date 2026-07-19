// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"strings"
	"testing"
)

// train_real_layer_test.go gates the REAL-ARCH layer reference (#40 stage 2) by central finite
// differences — pure host, NO runtime gate (the whole point of the host-pure re-derivation: these
// gates cannot silently skip). Each gemma4 feature rung lands with its own FD sub-gate here before
// the next feature is written; the trainer wiring (stage 3) is allowed to accept only shapes whose
// every feature gate below is green.

// tinyRealLayer builds the tiny synthetic REAL layer the FD checks run on — base feature set
// (rung 1): pre-norm GQA attention, full standard rotary, global window, pre-norm gated-GELU MLP.
// Small weights keep the deep f32 chain numerically clean (the tinyTrainLayer precedent).
func tinyRealLayer() *RealTrainLayerF32 {
	const T, dModel, dFF, H, Hkv, d = 3, 8, 12, 2, 1, 4
	s := func(n, salt int) []float32 { return scaleSlice(syntheticFloat32(n, salt), 0.3) }
	return &RealTrainLayerF32{
		AttnNormW: syntheticFloat32(dModel, 11), WQ: s(H*d*dModel, 12), WK: s(Hkv*d*dModel, 13),
		WV: s(Hkv*d*dModel, 14), WO: s(dModel*H*d, 15),
		MLPNormW: syntheticFloat32(dModel, 16), WGate: s(dFF*dModel, 17), WUp: s(dFF*dModel, 18), WDown: s(dModel*dFF, 19),
		T: T, DModel: dModel, DFF: dFF, Heads: H, KVHeads: Hkv, HeadDim: d,
		RopeInvFreq: realRopeInvFreqs(d, 10000), RopePairHalf: d / 2, RopeScale: 1,
		AttnScale: 0.5, Window: 0, Eps: 1e-5,
	}
}

// realLayerFDCheck runs the shared FD gate for one layer/target: loss = Σ RealLayerForwardF32(h)·cot
// with the LoRA's effective weight substituted at target, and the analytic dA/dB (strided full
// coverage) and dH (sample) from RealLayerProjLoRABackwardF32 must match central differences —
// the same eps/tolerance bar as the #31 simplified-layer gate (train_lora_layer_test.go).
func realLayerFDCheck(t *testing.T, L *RealTrainLayerF32, target string) {
	t.Helper()
	const rank = 2
	scaling := float32(16.0 / rank)
	out, in, err := L.projDims(target)
	if err != nil {
		t.Fatalf("projDims: %v", err)
	}
	h := scaleSlice(syntheticFloat32(L.T*L.DModel, 21), 0.5)
	cot := syntheticFloat32(L.T*L.DModel, 22)
	a := scaleSlice(syntheticFloat32(rank*in, 23), 0.2)
	b := scaleSlice(syntheticFloat32(out*rank, 24), 0.2)

	frozen := L.projWeight(target)
	loss := func() float64 {
		eff, err := LoRAEffectiveWeightF32(frozen, a, b, out, in, rank, scaling)
		if err != nil {
			t.Fatalf("eff: %v", err)
		}
		saved := L.setProjWeight(target, eff)
		y, err := RealLayerForwardF32(h, L)
		L.setProjWeight(target, saved)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		var s float64
		for i := range y {
			s += float64(y[i]) * float64(cot[i])
		}
		return s
	}

	dA, dB, dH, err := RealLayerProjLoRABackwardF32(cot, h, L, target, a, b, rank, scaling)
	if err != nil {
		t.Fatalf("RealLayerProjLoRABackwardF32: %v", err)
	}

	const eps = 1.0 / 512
	check := func(name string, params, grad []float32) {
		step := 1
		if len(params) > 12 {
			step = len(params) / 12
		}
		for i := 0; i < len(params); i += step {
			orig := params[i]
			params[i] = orig + eps
			lp := loss()
			params[i] = orig - eps
			lm := loss()
			params[i] = orig
			fd := (lp - lm) / (2 * eps)
			if math.Abs(fd-float64(grad[i])) > 2e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dA", a, dA)
	check("dB", b, dB)
	check("dH", h, dH)
	if !t.Failed() {
		t.Logf("%s: dA[%d] dB[%d] dH[%d] match finite differences through the real layer", target, len(dA), len(dB), len(dH))
	}
}

// realLayerTargets are the projection targets a dense layer carries — every FD gate runs all of them.
var realLayerTargets = []string{ProjQ, ProjK, ProjV, ProjO, ProjGate, ProjUp, ProjDown}

// TestRealLayerProjLoRABackwardF32 finite-difference-checks the BASE real-layer backward (rung 1:
// pre-norm GQA + standard rope + causal + gated-GELU MLP) for every canonical projection target —
// the host-pure foundation every later feature rung extends.
func TestRealLayerProjLoRABackwardF32(t *testing.T) {
	for _, target := range realLayerTargets {
		t.Run(target, func(t *testing.T) {
			realLayerFDCheck(t, tinyRealLayer(), target)
		})
	}
}

// TestRealLayerProjLoRABackwardF32_Bad: an unknown target, a nil layer, a wrong-shaped upstream
// gradient, and a broken geometry are refused before any work.
func TestRealLayerProjLoRABackwardF32_Bad(t *testing.T) {
	L := tinyRealLayer()
	const rank = 1
	out, in, _ := L.projDims(ProjDown)
	a := make([]float32, rank*in)
	b := make([]float32, out*rank)
	dout := make([]float32, L.T*L.DModel)
	h := make([]float32, L.T*L.DModel)

	_, _, _, err := RealLayerProjLoRABackwardF32(dout, h, L, "lm_head", a, b, rank, 1)
	if err == nil {
		t.Fatal("a non-layer target must be refused")
	}
	if !strings.Contains(err.Error(), "lm_head") || !strings.Contains(err.Error(), ProjDown) {
		t.Fatalf("unknown-target error must name the request and the supported set; got: %s", err.Error())
	}
	if _, _, _, err := RealLayerProjLoRABackwardF32(dout, h, nil, ProjDown, a, b, rank, 1); err == nil {
		t.Fatal("a nil layer must be refused")
	}
	if _, _, _, err := RealLayerProjLoRABackwardF32(dout[:1], h, L, ProjDown, a, b, rank, 1); err == nil {
		t.Fatal("a wrong-shaped dout must be refused")
	}
	bad := tinyRealLayer()
	bad.KVHeads = 0
	if _, _, _, err := RealLayerProjLoRABackwardF32(dout, h, bad, ProjDown, a, b, rank, 1); err == nil {
		t.Fatal("a zero KVHeads geometry must be refused")
	}
	bad2 := tinyRealLayer()
	bad2.RopePairHalf = bad2.HeadDim // > HeadDim/2
	if _, _, _, err := RealLayerProjLoRABackwardF32(dout, h, bad2, ProjDown, a, b, rank, 1); err == nil {
		t.Fatal("a rope pairing wider than the head must be refused")
	}
}

// TestRealLayerForwardF32_Good pins the base forward against the PROVEN simplified-layer block
// forwards on the exact configuration where the two must coincide (full rotary, no window, no
// gemma4 extras): the host-pure re-derivation may differ from the steel-GEMM chain only by
// accumulation noise. Runtime-gated — the comparison target is the GPU-backed reference.
func TestRealLayerForwardF32_Good(t *testing.T) {
	requireNativeRuntime(t)
	L := tinyRealLayer()
	h := scaleSlice(syntheticFloat32(L.T*L.DModel, 21), 0.5)
	got, err := RealLayerForwardF32(h, L)
	if err != nil {
		t.Fatalf("RealLayerForwardF32: %v", err)
	}
	attnOut, err := MultiHeadAttnBlockForwardF32(h, L.AttnNormW, L.WQ, L.WK, L.WV, L.WO, L.T, L.DModel, L.Heads, L.KVHeads, L.HeadDim, L.HeadDim, 10000, L.AttnScale, L.Eps, true)
	if err != nil {
		t.Fatalf("attn block fwd: %v", err)
	}
	want, err := MLPBlockForwardF32(attnOut, L.MLPNormW, L.WGate, L.WUp, L.WDown, L.T, L.DModel, L.DFF, L.Eps)
	if err != nil {
		t.Fatalf("mlp block fwd: %v", err)
	}
	assertFloat32Near(t, "real-vs-simplified forward", got, want, 1e-4)
}

// TestRealLayerForwardF32_Bad: shape refusals at the forward entry.
func TestRealLayerForwardF32_Bad(t *testing.T) {
	L := tinyRealLayer()
	if _, err := RealLayerForwardF32(make([]float32, 1), L); err == nil {
		t.Fatal("a wrong-shaped h must be refused")
	}
	if _, err := RealLayerForwardF32(nil, nil); err == nil {
		t.Fatal("a nil layer must be refused")
	}
}
