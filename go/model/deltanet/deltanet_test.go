// SPDX-Licence-Identifier: EUPL-1.2

package deltanet

import (
	"math"
	"testing"
)

func syn(n, seed int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*seed+7)%101-50) * 0.02
	}
	return out
}

const testEps = 1e-6

// TestGatedDeltaL1ClosedForm checks the recurrence against the closed form for a single step from a zero
// state: decay·0=0, read=0, so S=k̂⊗(β·v) and o[v]=Σ_k (scale·q[k])·k̂[k]·β·v[v] = β·scale·(q·k̂)·v[v]
// (α is irrelevant from zero). k̂ = k/√(Σk²+eps).
func TestGatedDeltaL1ClosedForm(t *testing.T) {
	const H, D = 3, 6
	const scale = float32(0.40824829) // 1/√6
	q := syn(H*D, 1)
	k := syn(H*D, 2)
	v := syn(H*D, 3)
	beta := syn(H, 4)
	alpha := syn(H, 5)
	o, _, err := GatedDeltaRuleF32(q, k, v, beta, alpha, nil, 1, H, D, scale, testEps)
	if err != nil {
		t.Fatalf("GatedDeltaRuleF32: %v", err)
	}
	for h := 0; h < H; h++ {
		var ss float64
		for i := 0; i < D; i++ {
			ss += float64(k[h*D+i]) * float64(k[h*D+i])
		}
		inv := 1.0 / math.Sqrt(ss+testEps)
		var qk float64 // q · k̂
		for i := 0; i < D; i++ {
			qk += float64(q[h*D+i]) * float64(k[h*D+i]) * inv
		}
		for j := 0; j < D; j++ {
			want := float64(beta[h]) * float64(scale) * qk * float64(v[h*D+j])
			if got := float64(o[h*D+j]); math.Abs(got-want) > 1e-4*(1+math.Abs(want)) {
				t.Errorf("o[%d,%d] = %v, closed form β·scale·(q·k̂)·v = %v", h, j, got, want)
			}
		}
	}
	t.Log("gated delta L=1 matches the closed form β·scale·(q·k̂)·v")
}

// TestGatedDeltaChunkCarry proves the decode-boundary invariant: one pass over a sequence equals two
// chunks carrying the [H,D,D] delta state across the boundary — BIT-EXACT, the Qwen 3.6 decode correctness.
func TestGatedDeltaChunkCarry(t *testing.T) {
	const L, split, H, D = 7, 4, 2, 5
	const scale = float32(0.4472136) // 1/√5
	q := syn(L*H*D, 1)
	k := syn(L*H*D, 2)
	v := syn(L*H*D, 3)
	beta := syn(L*H, 4)
	alpha := make([]float32, L*H) // α ∈ (0,1): map syn into (0,1) so decay is realistic
	for i, s := range syn(L*H, 5) {
		alpha[i] = float32(0.5 + 0.4*float64(s))
	}

	oFull, sFull, err := GatedDeltaRuleF32(q, k, v, beta, alpha, nil, L, H, D, scale, testEps)
	if err != nil {
		t.Fatalf("full: %v", err)
	}
	o1, s1, err := GatedDeltaRuleF32(q[:split*H*D], k[:split*H*D], v[:split*H*D], beta[:split*H], alpha[:split*H], nil, split, H, D, scale, testEps)
	if err != nil {
		t.Fatalf("chunk1: %v", err)
	}
	rem := L - split
	o2, s2, err := GatedDeltaRuleF32(q[split*H*D:], k[split*H*D:], v[split*H*D:], beta[split*H:], alpha[split*H:], s1, rem, H, D, scale, testEps)
	if err != nil {
		t.Fatalf("chunk2: %v", err)
	}

	for i := range o1 {
		if o1[i] != oFull[i] {
			t.Fatalf("chunk1 o[%d] = %v != full %v", i, o1[i], oFull[i])
		}
	}
	for i := range o2 {
		if o2[i] != oFull[split*H*D+i] {
			t.Fatalf("chunk2 o[%d] = %v != full %v", i, o2[i], oFull[split*H*D+i])
		}
	}
	for i := range s2 {
		if s2[i] != sFull[i] {
			t.Fatalf("carried state[%d] = %v != full %v", i, s2[i], sFull[i])
		}
	}
	t.Logf("gated delta chunk-carry bit-exact: split %d|%d, o and delta state identical to the one-pass run", split, rem)
}
