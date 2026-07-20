// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
)

// composed_moe_fuse_bench_test.go measures the composed MoE gate+up fusion (#17) ON THE GPU at real
// 35B-A3B expert dims (D 2048, FF 512, 4-bit): the fused path is ONE quant matvec over [2·FF, D] where the
// split path is two over [FF, D] (gate then up). Unlike the retired composed engine's host micro-bench (which came
// out neutral — the host per-row dequant has no per-launch overhead), the device pays a submit/wait per
// matvec (~0.77ms/token gap, #23), so removing one submit per routed expert is the composed lane's slice of
// the metal lane's measured ~34% MoE win — visible only here. Byte-identity of the two is pinned by
// TestMatMulQuantF32NTInto_FusedGateUpMatchesSplit.

func benchGateUpSeam(b *testing.B, fused bool) {
	if os.Getenv(MetallibPathEnv) == "" {
		b.Skip("metallib not set")
	}
	const D, FF, gs, bits = 2048, 512, 32, 4
	gp, gsc, gb := quantizeProj(b, FF, D, gs, bits, 3)
	up, usc, ub := quantizeProj(b, FF, D, gs, bits, 51)
	fp := append(append([]byte{}, gp...), up...)
	fsc := append(append([]byte{}, gsc...), usc...)
	fb := append(append([]byte{}, gb...), ub...)
	x := mkQMVInput(D)
	gOut, uOut, fOut := make([]float32, FF), make([]float32, FF), make([]float32, 2*FF)

	do := func() {
		if fused {
			if _, err := MatMulQuantF32NTInto(fOut, x, fp, fsc, fb, 1, D, 2*FF, gs, bits); err != nil {
				b.Fatal(err)
			}
			return
		}
		if _, err := MatMulQuantF32NTInto(gOut, x, gp, gsc, gb, 1, D, FF, gs, bits); err != nil {
			b.Fatal(err)
		}
		if _, err := MatMulQuantF32NTInto(uOut, x, up, usc, ub, 1, D, FF, gs, bits); err != nil {
			b.Fatal(err)
		}
	}

	do() // warm up: exclude lazy device init + first-use kernel compile from the timer
	b.ResetTimer()
	for range b.N {
		do()
	}
}

// BenchmarkComposedMoEGateUp_Split is a routed expert's gate+up as the unfused pair: two GPU quant matvecs.
func BenchmarkComposedMoEGateUp_Split(b *testing.B) { benchGateUpSeam(b, false) }

// BenchmarkComposedMoEGateUp_Fused is the same gate+up as ONE matvec over [gate‖up]: one GPU submit.
func BenchmarkComposedMoEGateUp_Fused(b *testing.B) { benchGateUpSeam(b, true) }
