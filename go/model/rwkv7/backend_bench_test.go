// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import "testing"

// The projMatMul bench baselines the block's projection seam on its DEFAULT (nil-hook) path
// (AX-11): with no backend wired, projMatMul falls through to the host-f32 matNT — the
// compute hot spot of the block per block.go's doc. This isolates one projection GEMM (the
// per-token cost multiplied across R/W/K/A/B/V/out) from the recurrence. Dims: a
// hidden→(H·K) projection, D=1024, H·K=1024, decode row L=1. Pure Go.

// BenchmarkProjMatMul_HostDefault — one projection y = x[1,D] @ w[N,D]ᵀ through the nil-hook
// default (host matNT): the single result allocation, the f64-accumulated GEMM the block
// pays per projection when no device GEMM is injected.
func BenchmarkProjMatMul_HostDefault(b *testing.B) {
	const D, N = 1024, 1024
	x := benchRwkvF32(1 * D)
	w := benchRwkvF32(N * D)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := projMatMul(x, w, 1, D, N); err != nil {
			b.Fatal(err)
		}
	}
}
