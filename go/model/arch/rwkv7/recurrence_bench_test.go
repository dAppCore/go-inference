// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import "testing"

// The WKV7F32 benches baseline the RWKV-7 ("Goose") generalised delta-rule recurrence
// (AX-11) — the per-decode-token mixer at the heart of an RWKV-7 time-mix block. The sa
// scratch is hoisted out of the timestep loop, so the allocation is the o [L,H,V] output +
// the advanced state [H,K,V]; the cost is the O(L·H·K·V) recurrence (decay + rank-1
// transition + rank-1 write + read-out per timestep). Two shapes: single-token decode over a
// carried state, and a short prefill from fresh. Dims: 16 heads, K=V=64. Pure Go.

const (
	benchRwkvHeads = 16
	benchRwkvK     = 64
	benchRwkvV     = 64
)

func benchRwkvF32(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32((i*131)%4096-2048) * 0.001
	}
	return s
}

// BenchmarkWKV7F32_Decode — one token over a carried [H,K,V] state (the decode hot path):
// one recurrence step per head. Allocates o + the new state.
func BenchmarkWKV7F32_Decode(b *testing.B) {
	const L, H, K, V = 1, benchRwkvHeads, benchRwkvK, benchRwkvV
	r, w, k := benchRwkvF32(L*H*K), benchRwkvF32(L*H*K), benchRwkvF32(L*H*K)
	a, bb := benchRwkvF32(L*H*K), benchRwkvF32(L*H*K)
	v := benchRwkvF32(L * H * V)
	prior := benchRwkvF32(H * K * V)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := WKV7F32(r, w, k, v, a, bb, prior, L, H, K, V); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkWKV7F32_Prefill — a 32-token prefill from a fresh (nil) state: the recurrence over
// the sequence, pinning the sa scratch stays one allocation regardless of L.
func BenchmarkWKV7F32_Prefill(b *testing.B) {
	const L, H, K, V = 32, benchRwkvHeads, benchRwkvK, benchRwkvV
	r, w, k := benchRwkvF32(L*H*K), benchRwkvF32(L*H*K), benchRwkvF32(L*H*K)
	a, bb := benchRwkvF32(L*H*K), benchRwkvF32(L*H*K)
	v := benchRwkvF32(L * H * V)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := WKV7F32(r, w, k, v, a, bb, nil, L, H, K, V); err != nil {
			b.Fatal(err)
		}
	}
}
