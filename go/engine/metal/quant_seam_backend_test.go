// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/quant/mlxaffine"
)

// composed_quant_backend_test.go gates the composed lane's quant matvec seam against the host reference
// (mlxaffine.DequantizeTensor + a plain host matNT) at small synthetic dims — the same device-vs-host
// tolerance discipline (1e-2·(1+|want|)) the composed device-GEMM tests use. It covers the M=1 decode qmv
// AND the M>1 prefill qmm_t (both N-aligned and ragged), across the acceptance (gs, bits): gs=64 b=4 (the
// 4-bit checkpoint) and gs=128 b=2 (Bonsai's 1-bit repacked to 2-bit). EVERY run must set the metallib
// path inline — TestMain exits 0 without it, so a bare "ok" proves nothing.

// cbqSynthW builds a deterministic (outDim × inDim) weight spanning both signs so the affine derivation
// exercises its min- and max-edge branches.
func cbqSynthW(outDim, inDim int) []float32 {
	w := make([]float32, outDim*inDim)
	for i := range w {
		w[i] = float32((i%17)-8) * 0.037
	}
	return w
}

// cbqHostMatNT is the f64 host reference: out[M,N] = x[M,K] @ w[N,K]ᵀ over the DEQUANTISED weight.
func cbqHostMatNT(x, w []float32, M, K, N int) []float32 {
	out := make([]float32, M*N)
	for m := range M {
		for n := range N {
			var acc float64
			for k := range K {
				acc += float64(x[m*K+k]) * float64(w[n*K+k])
			}
			out[m*N+n] = float32(acc)
		}
	}
	return out
}

func TestMatMulQuantF32NTInto_ParityDecodeAndPrefill(t *testing.T) {
	for _, tc := range []struct {
		name              string
		M, K, N, bits, gs int
	}{
		{"decode-qmv-gs64-b4", 1, 512, 40, 4, 64},   // 4-bit checkpoint decode geometry
		{"decode-qmv-gs128-b2", 1, 256, 24, 2, 128}, // Bonsai repacked, decode
		{"prefill-qmm_t-alN-true", 3, 512, 64, 4, 64},
		{"prefill-qmm_t-alN-false", 5, 256, 40, 2, 128}, // N%32 != 0 → ragged tile
		{"prefill-qmm_t-b8-smallN", 2, 128, 8, 8, 64},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := cbqSynthW(tc.N, tc.K)
			packed, scales, biases, err := mlxaffine.QuantizeTensor(w, tc.N, tc.K, tc.bits, tc.gs)
			if err != nil {
				t.Fatalf("quantise: %v", err)
			}
			deqW, err := mlxaffine.DequantizeTensor(packed, scales, biases, tc.N, tc.K, tc.bits, tc.gs)
			if err != nil {
				t.Fatalf("dequantise: %v", err)
			}
			x := make([]float32, tc.M*tc.K)
			for i := range x {
				x[i] = float32((i%9)-4) * 0.05
			}
			want := cbqHostMatNT(x, deqW, tc.M, tc.K, tc.N)
			got, err := MatMulQuantF32NTInto(nil, x, packed, scales, biases, tc.M, tc.K, tc.N, tc.gs, tc.bits)
			if err != nil {
				t.Fatalf("MatMulQuantF32NTInto: %v", err)
			}
			if len(got) != tc.M*tc.N {
				t.Fatalf("length %d, want %d", len(got), tc.M*tc.N)
			}
			for i := range want {
				if d := math.Abs(float64(got[i] - want[i])); d > 1e-2*(1+math.Abs(float64(want[i]))) {
					t.Fatalf("i=%d: device %g vs host %g (Δ=%g exceeds f32 tol)", i, got[i], want[i], d)
				}
			}
			t.Logf("%s: device quant matvec matches host dequant+matNT within f32 tol", tc.name)
		})
	}
}

// TestMatMulQuantF32NTInto_Errors pins the rejection paths: a mismatched activation length and an invalid
// quant geometry (groupSize not dividing K) must error rather than dispatch garbage.
func TestMatMulQuantF32NTInto_Errors(t *testing.T) {
	packed, scales, biases, err := mlxaffine.QuantizeTensor(cbqSynthW(16, 128), 16, 128, 4, 64)
	if err != nil {
		t.Fatalf("setup quantise: %v", err)
	}
	if _, err := MatMulQuantF32NTInto(nil, make([]float32, 100), packed, scales, biases, 1, 128, 16, 64, 4); err == nil {
		t.Error("mismatched len(x) must error")
	}
	if _, err := MatMulQuantF32NTInto(nil, make([]float32, 128), packed, scales, biases, 1, 128, 16, 60, 4); err == nil {
		t.Error("groupSize not dividing K must error")
	}
}
