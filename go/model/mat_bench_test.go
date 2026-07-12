// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

// The MatNT benches baseline the pure-Go reference matmul (AX-11): out = in·wᵀ with a
// float64 accumulator, the CPU-side linear projection the arch packages fall back to when
// no backend kernel runs it. It sits on the per-token decode path (a token's hidden state
// projected through q/k/v/o and the FFN), so its allocation shape — the single out[M·N]
// result slice — is the per-projection cost. Realistic decode dims: a single-token step
// (M=1) and a small prefill batch (M=8) through a hidden-sized projection.

const (
	benchMatHidden = 2048 // a typical small-model hidden size
	benchMatOut    = 2048 // a square projection (q_proj / o_proj shape)
)

func benchMatInputs(m, k, n int) (in, w []float32) {
	in = make([]float32, m*k)
	w = make([]float32, n*k)
	for i := range in {
		in[i] = float32((i*131)%4096-2048) * 0.001
	}
	for i := range w {
		w[i] = float32((i*97)%4096-2048) * 0.001
	}
	return in, w
}

// BenchmarkMatNT_SingleToken — one token's hidden state through a square projection: the
// per-token decode cost. The single result allocation (out[1·N]) is the whole alloc story.
func BenchmarkMatNT_SingleToken(b *testing.B) {
	const m, k, n = 1, benchMatHidden, benchMatOut
	in, w := benchMatInputs(m, k, n)
	b.SetBytes(int64(m * n * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatNT(in, w, m, k, n)
	}
}

// BenchmarkMatNT_Prefill — an 8-token prefill batch through the same projection: the out
// slice grows to M·N but the allocation COUNT stays one, so this pins that batching does
// not multiply the per-call allocation.
func BenchmarkMatNT_Prefill(b *testing.B) {
	const m, k, n = 8, benchMatHidden, benchMatOut
	in, w := benchMatInputs(m, k, n)
	b.SetBytes(int64(m * n * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatNT(in, w, m, k, n)
	}
}
