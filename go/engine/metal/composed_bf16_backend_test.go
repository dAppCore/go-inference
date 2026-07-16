// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"dappco.re/go/inference/model"
)

// composed_bf16_backend_test.go gates the dense bf16 matvec seam (#26) against the widened host
// reference: same weights widened to f32 through matNT-shaped f64 accumulation.

func bf16SeamRequire(t *testing.T) {
	t.Helper()
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — bf16 seam")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("Metal runtime unavailable — bf16 seam: %v", err)
	}
}

// TestMatMulBF16WF32NTInto_Good gates decode (M=1) and a prefill-shaped M=4 against the widened
// f64-accumulated host reference at a real projection shape.
func TestMatMulBF16WF32NTInto_Good(t *testing.T) {
	bf16SeamRequire(t)
	const K, N = 256, 512
	rng := rand.New(rand.NewSource(3))
	wf := make([]float32, N*K)
	for i := range wf {
		wf[i] = -0.5 + rng.Float32()
	}
	wb := f32sToBF16Bytes(wf) // the checkpoint form; also the values the reference must use
	for i := range wf {      // reference reads the SAME bf16-rounded values the kernel reads
		wf[i] = bf16ToF32(wb[2*i], wb[2*i+1])
	}
	for _, M := range []int{1, 4} {
		x := make([]float32, M*K)
		for i := range x {
			x[i] = -1 + 2*rng.Float32()
		}
		want := make([]float32, M*N)
		// bf16 activations at the seam: the reference must round x identically.
		xr := make([]float32, len(x))
		xb := f32sToBF16Bytes(x)
		for i := range xr {
			xr[i] = bf16ToF32(xb[2*i], xb[2*i+1])
		}
		for m := 0; m < M; m++ {
			for n := 0; n < N; n++ {
				var acc float64
				for k := 0; k < K; k++ {
					acc += float64(xr[m*K+k]) * float64(wf[n*K+k])
				}
				want[m*N+n] = float32(acc)
			}
		}
		got, err := MatMulBF16WF32NTInto(nil, x, wb, M, K, N)
		if err != nil {
			t.Fatalf("MatMulBF16WF32NTInto(M=%d): %v", M, err)
		}
		var worst, scale float64
		for i := range got {
			if a := math.Abs(float64(want[i])); a > scale {
				scale = a
			}
			if d := math.Abs(float64(got[i]) - float64(want[i])); d > worst {
				worst = d
			}
		}
		rel := worst / (scale + 1e-12)
		t.Logf("M=%d scaled max diff %.3e", M, rel)
		// bf16 kernel output storage (8-bit mantissa) vs the f64 host accumulation.
		if rel > 2e-2 {
			t.Fatalf("M=%d drift %.3e beyond bf16 tolerance", M, rel)
		}
	}
}

// TestMatMulBF16WF32NTInto_Bad pins the rejections: bad geometry and size mismatches error before
// any dispatch, and the BF16Weight wrapper rejects a geometry mismatch.
func TestMatMulBF16WF32NTInto_Bad(t *testing.T) {
	bf16SeamRequire(t)
	if _, err := MatMulBF16WF32NTInto(nil, make([]float32, 8), make([]byte, 8*4*2), 0, 4, 8); err == nil {
		t.Fatal("M=0 must error")
	}
	if _, err := MatMulBF16WF32NTInto(nil, make([]float32, 3), make([]byte, 8*4*2), 1, 4, 8); err == nil {
		t.Fatal("x size mismatch must error")
	}
	if _, err := MatMulBF16WF32NTInto(nil, make([]float32, 4), make([]byte, 7), 1, 4, 8); err == nil {
		t.Fatal("w size mismatch must error")
	}
	if _, err := MatMulBF16WeightF32NTInto(nil, make([]float32, 4), &model.BF16Weight{Data: make([]byte, 8*4*2), OutDim: 9, InDim: 4}, 1, 4, 8); err == nil {
		t.Fatal("BF16Weight geometry mismatch must error")
	}
}
