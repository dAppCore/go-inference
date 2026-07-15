// SPDX-Licence-Identifier: EUPL-1.2

package deltanet

import "testing"

// The GatedDeltaRuleF32 benches baseline the Qwen 3.5/3.6 gated delta-rule recurrence
// (AX-11) — the linear-attention mixer that runs per decode token. The scratch (kn/read/be)
// is hoisted out of the timestep loop already, so the allocation is the o [L,H,D] output +
// the advanced state [H,D,D]; the cost is the O(L·H·D²) f64-accumulated recurrence. Two
// shapes: a single-token decode carrying prior state (the hot path) and a short prefill from
// a fresh state. Dims: 16 heads of head-dim 128 (a small hybrid layer). Pure Go.

const (
	benchDeltaHeads   = 16
	benchDeltaHeadDim = 128
)

func benchDeltaF32(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32((i*131)%4096-2048) * 0.001
	}
	return s
}

// BenchmarkGatedDeltaRuleF32_Decode — one token over a carried [H,D,D] state (the decode hot
// path): one timestep of decay + rank-1 read/write + read-out per head. Allocates o + the
// new state.
func BenchmarkGatedDeltaRuleF32_Decode(b *testing.B) {
	const L, H, D = 1, benchDeltaHeads, benchDeltaHeadDim
	q, k, v := benchDeltaF32(L*H*D), benchDeltaF32(L*H*D), benchDeltaF32(L*H*D)
	beta, alpha := benchDeltaF32(L*H), benchDeltaF32(L*H)
	prior := benchDeltaF32(H * D * D)
	scale := float32(1.0 / 11.3137) // 1/sqrt(128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := GatedDeltaRuleF32(q, k, v, beta, alpha, prior, L, H, D, scale, 1e-6); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGatedDeltaRuleF32_Prefill — a 32-token prefill from a fresh (nil) state: the same
// per-timestep recurrence run over the sequence, pinning that the scratch stays one
// allocation regardless of L (only o + state grow).
func BenchmarkGatedDeltaRuleF32_Prefill(b *testing.B) {
	const L, H, D = 32, benchDeltaHeads, benchDeltaHeadDim
	q, k, v := benchDeltaF32(L*H*D), benchDeltaF32(L*H*D), benchDeltaF32(L*H*D)
	beta, alpha := benchDeltaF32(L*H), benchDeltaF32(L*H)
	scale := float32(1.0 / 11.3137)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := GatedDeltaRuleF32(q, k, v, beta, alpha, nil, L, H, D, scale, 1e-6); err != nil {
			b.Fatal(err)
		}
	}
}
