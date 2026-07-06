// SPDX-Licence-Identifier: EUPL-1.2

package model

// MatNT computes the naive reference matmul out = in · wᵀ — for row-major in
// [M×K] and w [N×K] (weight stored transposed, hence the NT), accumulating each
// dot product in float64 for numerical stability and returning out [M×N]. The
// shared pure-Go reference the arch packages use for CPU-side linear
// projections, replacing the identical per-package matNT copies.
//
//	out := model.MatNT(hidden, weight, m, k, n)
func MatNT(in, w []float32, M, K, N int) []float32 {
	out := make([]float32, M*N)
	for m := range M {
		for n := range N {
			var acc float64
			for k := range K {
				acc += float64(in[m*K+k]) * float64(w[n*K+k])
			}
			out[m*N+n] = float32(acc)
		}
	}
	return out
}
