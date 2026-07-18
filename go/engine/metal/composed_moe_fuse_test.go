// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
)

// composed_moe_fuse_test.go is the DEVICE receipt for the composed MoE gate+up fusion (#17). The composed
// lane dispatches each expert projection through the bound quant seam (composed.ProjQuantMatMulInto =
// MatMulQuantF32NTInto), so fuseExpertGateUp turns a routed expert's TWO gate/up matvecs into ONE over the
// [gate‖up] concat — one GPU submit per expert instead of two. Here that fusion is exercised against the
// real Metal affine_qmv kernel: the host byte-identity (model/composed.TestSwigluExpertQuantInto_-
// FusedMatchesUnfused) is re-proven on the DEVICE, and the timing win the host cannot show (no per-submit
// overhead there) is measured in composed_moe_fuse_bench_test.go.

// mkQMVInput fills a deterministic [D] activation for the quant matvec.
func mkQMVInput(d int) []float32 {
	x := make([]float32, d)
	for i := range x {
		x[i] = float32((i%7)-3) * 0.1
	}
	return x
}

// TestMatMulQuantF32NTInto_FusedGateUpMatchesSplit pins DEVICE byte-identity: one quant matvec over the
// row-concatenated [gate‖up] weight ([2·FF, D]) produces, in its two halves, EXACTLY what two separate
// gate and up matvecs ([FF, D] each) produce — because affine_qmv computes each output row independently,
// so concatenating gate's rows ahead of up's is a pure GPU-submit-count change. Real 35B-A3B expert dims.
func TestMatMulQuantF32NTInto_FusedGateUpMatchesSplit(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const D, FF, gs, bits = 2048, 512, 32, 4 // Qwen3.5-35B-A3B: hidden_size, moe_intermediate_size
	gp, gsc, gb := quantizeProj(t, FF, D, gs, bits, 3)
	up, usc, ub := quantizeProj(t, FF, D, gs, bits, 51)
	fp := append(append([]byte{}, gp...), up...) // [gate‖up] row-concat — what model.ConcatQuantRows builds
	fsc := append(append([]byte{}, gsc...), usc...)
	fb := append(append([]byte{}, gb...), ub...)

	x := mkQMVInput(D)
	gOut, err := MatMulQuantF32NTInto(make([]float32, FF), x, gp, gsc, gb, 1, D, FF, gs, bits)
	if err != nil {
		t.Fatalf("gate matvec: %v", err)
	}
	uOut, err := MatMulQuantF32NTInto(make([]float32, FF), x, up, usc, ub, 1, D, FF, gs, bits)
	if err != nil {
		t.Fatalf("up matvec: %v", err)
	}
	fOut, err := MatMulQuantF32NTInto(make([]float32, 2*FF), x, fp, fsc, fb, 1, D, 2*FF, gs, bits)
	if err != nil {
		t.Fatalf("fused matvec: %v", err)
	}

	for f := range FF {
		if fOut[f] != gOut[f] {
			t.Fatalf("fused gate half [%d] = %v, want %v (split gate)", f, fOut[f], gOut[f])
		}
		if fOut[FF+f] != uOut[f] {
			t.Fatalf("fused up half [%d] = %v, want %v (split up)", f, fOut[FF+f], uOut[f])
		}
	}
}
