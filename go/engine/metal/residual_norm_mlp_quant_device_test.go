// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/quant/mlxaffine"
)

// residual_norm_mlp_quant_device_test.go gates the packed-weight FFN-tail fold (#8-B slice 1) against
// the path it actually replaces — the UNFUSED quant tail: MatMulQuantF32NTInto per projection (the
// proven seam, bf16 across each qmv) with the f32 host glue between. The fold encodes the same
// kernels behind the same casts, so it must match that reference TIGHTLY (1e-3·(1+|want|) — f32 glue
// noise only; a stage-level bisect measured the fold's projection outputs byte-tracking the seam's).
// A second, LOOSE gate against the f64 dequantised reference (5e-2·(1+|want|)) documents the
// intrinsic bf16-activation qmv drift BOTH paths share — a fold bug would blow the tight gate long
// before the loose one. EVERY run must set the metallib path inline — TestMain exits 0 without it,
// so a bare "ok" proves nothing.

// rnmqQuantWeight packs a deterministic (outDim × inDim) synthetic weight and wraps it as the
// model.QuantWeight the composed loader would carry.
func rnmqQuantWeight(t *testing.T, outDim, inDim, bits, gs int) (*model.QuantWeight, []float32) {
	t.Helper()
	w := cbqSynthW(outDim, inDim)
	packed, scales, biases, err := mlxaffine.QuantizeTensor(w, outDim, inDim, bits, gs)
	if err != nil {
		t.Fatalf("quantise %dx%d b%d gs%d: %v", outDim, inDim, bits, gs, err)
	}
	deq, err := mlxaffine.DequantizeTensor(packed, scales, biases, outDim, inDim, bits, gs)
	if err != nil {
		t.Fatalf("dequantise: %v", err)
	}
	return &model.QuantWeight{
		Packed: packed, Scales: scales, Biases: biases,
		Bits: bits, GroupSize: gs, OutDim: outDim, InDim: inDim,
	}, deq
}

// rnmqHostTail is the f64 host reference of the fused tail over dequantised weights:
// hplus = h + mix; normed = plain RMSNorm(hplus, normW); y = hplus + silu(normed·gateᵀ)⊙(normed·upᵀ)·downᵀ.
func rnmqHostTail(h, mix, normW, gateW, upW, downW []float32, L, D, FF int, eps float32) []float32 {
	hplus := make([]float32, L*D)
	for i := range hplus {
		hplus[i] = h[i] + mix[i]
	}
	normed := make([]float32, L*D)
	for r := 0; r < L; r++ {
		var ss float64
		for i := 0; i < D; i++ {
			v := float64(hplus[r*D+i])
			ss += v * v
		}
		inv := 1 / math.Sqrt(ss/float64(D)+float64(eps))
		for i := 0; i < D; i++ {
			normed[r*D+i] = float32(float64(hplus[r*D+i]) * inv * float64(normW[i]))
		}
	}
	g := cbqHostMatNT(normed, gateW, L, D, FF)
	u := cbqHostMatNT(normed, upW, L, D, FF)
	s := make([]float32, L*FF)
	for i := range s {
		x := float64(g[i])
		s[i] = float32(x / (1 + math.Exp(-x)) * float64(u[i]))
	}
	mlp := cbqHostMatNT(s, downW, L, FF, D)
	y := make([]float32, L*D)
	for i := range y {
		y[i] = hplus[i] + mlp[i]
	}
	return y
}

// rnmqSeamTail is the unfused reference: the tail computed exactly as the quant bypass computes it —
// each projection through MatMulQuantF32NTInto (device bf16 qmv/qmm_t), the residuals, norm and silu
// as f32 host glue.
func rnmqSeamTail(t *testing.T, h, mix, normW []float32, gate, up, down *model.QuantWeight, L, D, FF int, eps float32) []float32 {
	t.Helper()
	hplus := make([]float32, L*D)
	for i := range hplus {
		hplus[i] = h[i] + mix[i]
	}
	normed := make([]float32, L*D)
	for r := 0; r < L; r++ {
		var ss float64
		for i := 0; i < D; i++ {
			v := float64(hplus[r*D+i])
			ss += v * v
		}
		inv := 1 / math.Sqrt(ss/float64(D)+float64(eps))
		for i := 0; i < D; i++ {
			normed[r*D+i] = float32(float64(hplus[r*D+i]) * inv * float64(normW[i]))
		}
	}
	g, err := MatMulQuantF32NTInto(nil, normed, gate.Packed, gate.Scales, gate.Biases, L, D, FF, gate.GroupSize, gate.Bits)
	if err != nil {
		t.Fatalf("seam gate: %v", err)
	}
	u, err := MatMulQuantF32NTInto(nil, normed, up.Packed, up.Scales, up.Biases, L, D, FF, up.GroupSize, up.Bits)
	if err != nil {
		t.Fatalf("seam up: %v", err)
	}
	sAct := make([]float32, L*FF)
	for i := range sAct {
		x := float64(g[i])
		sAct[i] = float32(x / (1 + math.Exp(-x)) * float64(u[i]))
	}
	mlp, err := MatMulQuantF32NTInto(nil, sAct, down.Packed, down.Scales, down.Biases, L, FF, D, down.GroupSize, down.Bits)
	if err != nil {
		t.Fatalf("seam down: %v", err)
	}
	y := make([]float32, L*D)
	for i := range y {
		y[i] = hplus[i] + mlp[i]
	}
	return y
}

func TestResidualNormMLPQuantDevice_ParityDecodeAndPrefill(t *testing.T) {
	for _, tc := range []struct {
		name               string
		L, D, FF, bits, gs int
	}{
		{"decode-gs64-b4", 1, 128, 192, 4, 64},  // the 4-bit checkpoint shape class, L=1 qmv lane
		{"prefill-gs64-b4", 4, 128, 192, 4, 64}, // L>1 qmm_t lane
		{"decode-gs128-b2", 1, 128, 256, 2, 128},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gate, gateW := rnmqQuantWeight(t, tc.FF, tc.D, tc.bits, tc.gs)
			up, upW := rnmqQuantWeight(t, tc.FF, tc.D, tc.bits, tc.gs)
			down, downW := rnmqQuantWeight(t, tc.D, tc.FF, tc.bits, tc.gs)
			h := make([]float32, tc.L*tc.D)
			mix := make([]float32, tc.L*tc.D)
			normW := make([]float32, tc.D)
			for i := range h {
				h[i] = float32((i%13)-6) * 0.041
				mix[i] = float32((i%7)-3) * 0.029
			}
			for i := range normW {
				normW[i] = 1 + float32(i%5)*0.03
			}
			const eps = 1e-6
			seamWant := rnmqSeamTail(t, h, mix, normW, gate, up, down, tc.L, tc.D, tc.FF, eps)
			f64Want := rnmqHostTail(h, mix, normW, gateW, upW, downW, tc.L, tc.D, tc.FF, eps)
			got, err := ResidualNormMLPQuantDevice(h, mix, normW, gate, up, down, tc.L, tc.D, tc.FF, eps)
			if err != nil {
				t.Fatalf("ResidualNormMLPQuantDevice: %v", err)
			}
			if len(got) != tc.L*tc.D {
				t.Fatalf("length %d, want %d", len(got), tc.L*tc.D)
			}
			scale := 0.0
			for i := range f64Want {
				if a := math.Abs(float64(f64Want[i])); a > scale {
					scale = a
				}
			}
			worstSeam, worstF64 := 0.0, 0.0
			for i := range seamWant {
				if d := math.Abs(float64(got[i] - seamWant[i])); d > worstSeam {
					worstSeam = d
				}
				if d := math.Abs(float64(got[i] - f64Want[i])); d > worstF64 {
					worstF64 = d
				}
				if d := math.Abs(float64(got[i] - seamWant[i])); d > 1e-3*(1+math.Abs(float64(seamWant[i]))) {
					t.Fatalf("i=%d: fold %g vs unfused seam tail %g (Δ=%g) — the fold changed the lane's numerics", i, got[i], seamWant[i], d)
				}
			}
			// The f64 comparison documents the drift BOTH quant paths share (bf16 activations across
			// the qmv, f32 kernel accumulation); it is scale-relative because bf16 rounding scales
			// with row magnitude, not with each output's own value. A fold bug trips the tight seam
			// gate above long before this bound.
			if worstF64 > 5e-2*(1+scale) {
				t.Fatalf("fold vs f64 reference: worst |Δ|=%g exceeds 5e-2·(1+scale) at scale %g", worstF64, scale)
			}
			t.Logf("%s: fold tracks the unfused seam tail (worst |Δ|=%.4g) and the f64 reference (worst |Δ|=%.4g at scale %.3g)", tc.name, worstSeam, worstF64, scale)
		})
	}
}

// TestResidualNormMLPQuantDevice_Errors pins the decline paths: a size mismatch and an unsupported
// quant geometry must error (the composed wiring then falls back to the host tail) rather than
// half-encode a command buffer.
func TestResidualNormMLPQuantDevice_Errors(t *testing.T) {
	gate, _ := rnmqQuantWeight(t, 192, 128, 4, 64)
	up, _ := rnmqQuantWeight(t, 192, 128, 4, 64)
	down, _ := rnmqQuantWeight(t, 128, 192, 4, 64)
	h := make([]float32, 128)
	mix := make([]float32, 128)
	normW := make([]float32, 128)
	if _, err := ResidualNormMLPQuantDevice(h[:64], mix, normW, gate, up, down, 1, 128, 192, 1e-6); err == nil {
		t.Fatal("expected a size-mismatch error")
	}
	bad := &model.QuantWeight{Packed: gate.Packed, Scales: gate.Scales, Biases: gate.Biases, Bits: 7, GroupSize: 64, OutDim: 192, InDim: 128}
	if _, err := ResidualNormMLPQuantDevice(h, mix, normW, bad, up, down, 1, 128, 192, 1e-6); err == nil {
		t.Fatal("expected an unsupported-geometry error for bits=7")
	}
}
