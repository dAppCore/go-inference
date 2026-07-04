// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import (
	"math"
	"testing"
)

// syn is a deterministic synthetic vector (seeded), values in [-1, 1).
func syn(n, seed int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*seed+7)%101-50) * 0.02
	}
	return out
}

// TestWKV7L1ClosedForm checks the recurrence against the closed form for a single step from a zero state:
// S=k⊗v (Sa=0, decay·0=0), so o[v]=Σ_k r[k]·k[k]·v[v]=(r·k)·v[v] per head.
func TestWKV7L1ClosedForm(t *testing.T) {
	const H, K, V = 3, 5, 4
	r := syn(H*K, 1)
	w := syn(H*K, 2)
	k := syn(H*K, 3)
	v := syn(H*V, 4)
	a := syn(H*K, 5)
	b := syn(H*K, 6)
	o, _, err := WKV7F32(r, w, k, v, a, b, nil, 1, H, K, V)
	if err != nil {
		t.Fatalf("WKV7F32: %v", err)
	}
	for h := 0; h < H; h++ {
		var rk float64
		for i := 0; i < K; i++ {
			rk += float64(r[h*K+i]) * float64(k[h*K+i])
		}
		for j := 0; j < V; j++ {
			want := rk * float64(v[h*V+j])
			if got := float64(o[h*V+j]); math.Abs(got-want) > 1e-4*(1+math.Abs(want)) {
				t.Errorf("o[%d,%d] = %v, closed form (r·k)·v = %v", h, j, got, want)
			}
		}
	}
	t.Log("WKV7 L=1 matches the closed form (r·k)·v")
}

// TestWKV7ChunkCarry proves the decode-boundary invariant: running a sequence in one pass equals running
// it as two chunks where the first chunk's final [H,K,V] state is carried into the second. BIT-EXACT (the
// per-step recurrence is identical regardless of the boundary) — what makes RWKV-7 decode correct.
func TestWKV7ChunkCarry(t *testing.T) {
	const L, split, H, K, V = 7, 4, 2, 3, 5
	r := syn(L*H*K, 1)
	w := syn(L*H*K, 2)
	k := syn(L*H*K, 3)
	v := syn(L*H*V, 4)
	a := syn(L*H*K, 5)
	b := syn(L*H*K, 6)

	oFull, sFull, err := WKV7F32(r, w, k, v, a, b, nil, L, H, K, V)
	if err != nil {
		t.Fatalf("full: %v", err)
	}
	o1, s1, err := WKV7F32(r[:split*H*K], w[:split*H*K], k[:split*H*K], v[:split*H*V], a[:split*H*K], b[:split*H*K], nil, split, H, K, V)
	if err != nil {
		t.Fatalf("chunk1: %v", err)
	}
	rem := L - split
	o2, s2, err := WKV7F32(r[split*H*K:], w[split*H*K:], k[split*H*K:], v[split*H*V:], a[split*H*K:], b[split*H*K:], s1, rem, H, K, V)
	if err != nil {
		t.Fatalf("chunk2: %v", err)
	}

	for i := range o1 {
		if o1[i] != oFull[i] {
			t.Fatalf("chunk1 o[%d] = %v != full %v", i, o1[i], oFull[i])
		}
	}
	for i := range o2 {
		if o2[i] != oFull[split*H*V+i] {
			t.Fatalf("chunk2 o[%d] = %v != full %v", i, o2[i], oFull[split*H*V+i])
		}
	}
	for i := range s2 {
		if s2[i] != sFull[i] {
			t.Fatalf("carried state[%d] = %v != full %v", i, s2[i], sFull[i])
		}
	}
	t.Logf("WKV7 chunk-carry bit-exact: split %d|%d, o and state identical to the one-pass recurrence", split, rem)
}
