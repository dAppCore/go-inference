// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import "testing"

// The projMatMul bench baselines the block's projection seam on its DEFAULT (nil-hook) path
// (AX-11): with no backend wired, projMatMul falls through to the host-f32 matNT — the
// block's compute hot spot per backend.go's doc (the in/out projections dominate; the scan +
// conv are cheap). This isolates one projection GEMM (the per-token cost the block pays for
// in_proj / out_proj) from the SSD scan. Dims: a d_model→d_model projection at decode row
// L=1, D=2048. Pure Go.

// BenchmarkProjMatMul_HostDefault — one projection y = x[1,D] @ w[N,D]ᵀ through the nil-hook
// default (host matNT): the single result allocation + the f64-accumulated GEMM the block
// pays per projection when no device GEMM is injected.
func BenchmarkProjMatMul_HostDefault(b *testing.B) {
	const D, N = 2048, 2048
	x := benchMambaF32(1 * D)
	w := benchMambaF32(N * D)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := projMatMul(x, w, 1, D, N); err != nil {
			b.Fatal(err)
		}
	}
}
