// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"strings"
	"testing"
)

// train_lora_layer_test.go gates the host-side per-layer projection LoRA reference (#31): the two pure
// host primitives are finite-difference-checked entry by entry (no GPU), and the composed layer backward
// is FD-checked per projection target through the real block forwards (runtime-gated, like every other
// composed-block gradient check in train_backward_test.go).

// TestLoRAEffectiveWeightF32_Good: B = 0 folds to the base weight exactly (the untrained adapter is a
// no-op), and a hand-computed rank-1 case lands on the exact values.
func TestLoRAEffectiveWeightF32_Good(t *testing.T) {
	const out, in, rank = 3, 4, 2
	w := syntheticFloat32(out*in, 1)
	a := syntheticFloat32(rank*in, 2)
	b := make([]float32, out*rank)
	eff, err := LoRAEffectiveWeightF32(w, a, b, out, in, rank, 2)
	if err != nil {
		t.Fatalf("LoRAEffectiveWeightF32: %v", err)
	}
	for i := range w {
		if eff[i] != w[i] {
			t.Fatalf("B=0 must fold to the base weight exactly: eff[%d]=%v w[%d]=%v", i, eff[i], i, w[i])
		}
	}
	// Hand case: w=[1 2], A=[1 1], B=[1], scaling=2 → eff = w + 2·(B·A) = [3 4].
	eff, err = LoRAEffectiveWeightF32([]float32{1, 2}, []float32{1, 1}, []float32{1}, 1, 2, 1, 2)
	if err != nil {
		t.Fatalf("hand case: %v", err)
	}
	if eff[0] != 3 || eff[1] != 4 {
		t.Fatalf("hand case: want [3 4], got %v", eff)
	}
}

// TestLoRAEffectiveWeightF32_Bad: every size mismatch is refused.
func TestLoRAEffectiveWeightF32_Bad(t *testing.T) {
	ok := []float32{0, 0}
	if _, err := LoRAEffectiveWeightF32([]float32{0}, ok, []float32{0}, 1, 2, 1, 1); err == nil {
		t.Fatal("short w must be refused")
	}
	if _, err := LoRAEffectiveWeightF32(ok, []float32{0}, []float32{0}, 1, 2, 1, 1); err == nil {
		t.Fatal("short A must be refused")
	}
	if _, err := LoRAEffectiveWeightF32(ok, ok, []float32{}, 1, 2, 1, 1); err == nil {
		t.Fatal("short B must be refused")
	}
}

// TestLoRAFactorGradsF32 verifies the factor-gradient fold against central finite differences — pure
// host, every entry checked. With L(A,B) = Σ eff(A,B)·cot and eff = W + scaling·(B·A), the weight
// gradient is exactly the cotangent (∂L/∂eff = cot), so LoRAFactorGradsF32(cot, …) must match the
// finite differences of L under perturbation of each A and B entry.
func TestLoRAFactorGradsF32(t *testing.T) {
	const out, in, rank = 4, 5, 2
	scaling := float32(16.0 / rank)
	w := syntheticFloat32(out*in, 1)
	a := scaleSlice(syntheticFloat32(rank*in, 2), 0.5)
	b := scaleSlice(syntheticFloat32(out*rank, 3), 0.5)
	cot := syntheticFloat32(out*in, 4)

	loss := func() float64 {
		eff, err := LoRAEffectiveWeightF32(w, a, b, out, in, rank, scaling)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		var s float64
		for i := range eff {
			s += float64(eff[i]) * float64(cot[i])
		}
		return s
	}

	dA, dB, err := LoRAFactorGradsF32(cot, a, b, out, in, rank, scaling)
	if err != nil {
		t.Fatalf("LoRAFactorGradsF32: %v", err)
	}

	const eps = 1.0 / 512
	check := func(name string, params, grad []float32) {
		for i := range params {
			orig := params[i]
			params[i] = orig + eps
			lp := loss()
			params[i] = orig - eps
			lm := loss()
			params[i] = orig
			fd := (lp - lm) / (2 * eps)
			if math.Abs(fd-float64(grad[i])) > 1e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dA", a, dA)
	check("dB", b, dB)
	t.Logf("factor-gradient fold matches finite differences: dA[%d] dB[%d] all within tol", len(dA), len(dB))
}

// TestLoRAFactorGradsF32_Bad: every size mismatch is refused.
func TestLoRAFactorGradsF32_Bad(t *testing.T) {
	ok := []float32{0, 0}
	if _, _, err := LoRAFactorGradsF32([]float32{0}, ok, []float32{0}, 1, 2, 1, 1); err == nil {
		t.Fatal("short dW must be refused")
	}
	if _, _, err := LoRAFactorGradsF32(ok, []float32{0}, []float32{0}, 1, 2, 1, 1); err == nil {
		t.Fatal("short A must be refused")
	}
	if _, _, err := LoRAFactorGradsF32(ok, ok, []float32{}, 1, 2, 1, 1); err == nil {
		t.Fatal("short B must be refused")
	}
}

// tinyTrainLayer builds the tiny synthetic simplified layer the composed FD checks run on — small
// weights keep the deep f32 chain numerically clean (the stableLayerWeights precedent).
func tinyTrainLayer() *TrainLayerF32 {
	const T, dModel, dFF, H, Hkv, d = 3, 8, 12, 2, 1, 4
	s := func(n, salt int) []float32 { return scaleSlice(syntheticFloat32(n, salt), 0.3) }
	return &TrainLayerF32{
		AttnNormW: syntheticFloat32(dModel, 11), WQ: s(H*d*dModel, 12), WK: s(Hkv*d*dModel, 13),
		WV: s(Hkv*d*dModel, 14), WO: s(dModel*H*d, 15),
		MLPNormW: syntheticFloat32(dModel, 16), WGate: s(dFF*dModel, 17), WUp: s(dFF*dModel, 18), WDown: s(dModel*dFF, 19),
		T: T, DModel: dModel, DFF: dFF, Heads: H, KVHeads: Hkv, HeadDim: d, RotaryDim: d,
		RopeBase: 10000, AttnScale: 0.5, Eps: 1e-5, Causal: true,
	}
}

// TestLayerProjLoRABackwardF32 finite-difference-checks the composed layer backward for EVERY canonical
// projection target on a tiny synthetic layer: the loss is Σ layerForward(h; A,B)·cot with the LoRA's
// effective weight substituted at the target, and the returned dA/dB (strided full coverage) and dH
// (sample) must match central differences — the correctness gate the future real-arch trainer wiring
// builds against.
func TestLayerProjLoRABackwardF32(t *testing.T) {
	requireNativeRuntime(t)
	const rank = 2
	scaling := float32(16.0 / rank)
	for _, target := range []string{ProjQ, ProjK, ProjV, ProjO, ProjGate, ProjUp, ProjDown} {
		t.Run(target, func(t *testing.T) {
			L := tinyTrainLayer()
			out, in, err := L.projDims(target)
			if err != nil {
				t.Fatalf("projDims: %v", err)
			}
			h := scaleSlice(syntheticFloat32(L.T*L.DModel, 21), 0.5)
			cot := syntheticFloat32(L.T*L.DModel, 22)
			a := scaleSlice(syntheticFloat32(rank*in, 23), 0.2)
			b := scaleSlice(syntheticFloat32(out*rank, 24), 0.2)

			forward := func() []float32 {
				eff, err := LoRAEffectiveWeightF32(L.projWeight(target), a, b, out, in, rank, scaling)
				if err != nil {
					t.Fatalf("eff: %v", err)
				}
				wQ, wK, wV, wO, wGate, wUp, wDown := L.WQ, L.WK, L.WV, L.WO, L.WGate, L.WUp, L.WDown
				switch target {
				case ProjQ:
					wQ = eff
				case ProjK:
					wK = eff
				case ProjV:
					wV = eff
				case ProjO:
					wO = eff
				case ProjGate:
					wGate = eff
				case ProjUp:
					wUp = eff
				case ProjDown:
					wDown = eff
				}
				attnOut, err := MultiHeadAttnBlockForwardF32(h, L.AttnNormW, wQ, wK, wV, wO, L.T, L.DModel, L.Heads, L.KVHeads, L.HeadDim, L.RotaryDim, L.RopeBase, L.AttnScale, L.Eps, L.Causal)
				if err != nil {
					t.Fatalf("attn fwd: %v", err)
				}
				outH, err := MLPBlockForwardF32(attnOut, L.MLPNormW, wGate, wUp, wDown, L.T, L.DModel, L.DFF, L.Eps)
				if err != nil {
					t.Fatalf("mlp fwd: %v", err)
				}
				return outH
			}
			loss := func() float64 {
				y := forward()
				var s float64
				for i := range y {
					s += float64(y[i]) * float64(cot[i])
				}
				return s
			}

			dA, dB, dH, err := LayerProjLoRABackwardF32(cot, h, L, target, a, b, rank, scaling)
			if err != nil {
				t.Fatalf("LayerProjLoRABackwardF32: %v", err)
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
			t.Logf("%s: dA[%d] dB[%d] dH[%d] match finite differences through the composed layer", target, len(dA), len(dB), len(dH))
		})
	}
}

// TestLayerProjLoRABackwardF32_Bad: an unknown target, a nil layer, and a wrong-shaped upstream
// gradient are refused before any block work — all host-side (validation precedes the GPU path).
func TestLayerProjLoRABackwardF32_Bad(t *testing.T) {
	L := tinyTrainLayer()
	const rank = 1
	out, in, _ := L.projDims(ProjDown)
	a := make([]float32, rank*in)
	b := make([]float32, out*rank)
	dout := make([]float32, L.T*L.DModel)
	h := make([]float32, L.T*L.DModel)

	_, _, _, err := LayerProjLoRABackwardF32(dout, h, L, "lm_head", a, b, rank, 1)
	if err == nil {
		t.Fatal("a non-layer target must be refused")
	}
	if !strings.Contains(err.Error(), "lm_head") || !strings.Contains(err.Error(), ProjDown) {
		t.Fatalf("unknown-target error must name the request and the supported set; got: %s", err.Error())
	}
	if _, _, _, err := LayerProjLoRABackwardF32(dout, h, nil, ProjDown, a, b, rank, 1); err == nil {
		t.Fatal("a nil layer must be refused")
	}
	if _, _, _, err := LayerProjLoRABackwardF32(dout[:1], h, L, ProjDown, a, b, rank, 1); err == nil {
		t.Fatal("a wrong-shaped dout must be refused")
	}
	if _, _, _, err := LayerProjLoRABackwardF32(dout, h, L, ProjDown, a[:1], b, rank, 1); err == nil {
		t.Fatal("a wrong-shaped A must be refused")
	}
}
