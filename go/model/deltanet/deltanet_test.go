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
	for h := range H {
		var ss float64
		for i := range D {
			ss += float64(k[h*D+i]) * float64(k[h*D+i])
		}
		inv := 1.0 / math.Sqrt(ss+testEps)
		var qk float64 // q · k̂
		for i := range D {
			qk += float64(q[h*D+i]) * float64(k[h*D+i]) * inv
		}
		for j := range D {
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

// TestGatedDeltaRuleF32_Golden pins the exact f32 bit-pattern of the recurrence outputs (o and the advanced
// state) for a fixed input, gating alloc-reduction refactors on bit-identical behaviour.
func TestGatedDeltaRuleF32_Golden(t *testing.T) {
	const L, H, D = 3, 2, 4
	alpha := syn(L*H, 23)
	for i := range alpha {
		alpha[i] = float32(1.0 / (1.0 + math.Exp(-float64(alpha[i])))) // decay ∈ (0,1]
	}
	o, state, err := GatedDeltaRuleF32(syn(L*H*D, 11), syn(L*H*D, 13), syn(L*H*D, 17), syn(L*H, 19), alpha, nil, L, H, D, 0.5, testEps)
	if err != nil {
		t.Fatalf("GatedDeltaRuleF32: %v", err)
	}
	wantO := []uint32{0x3edbaebe, 0x3e84d4d2, 0x3db7eb99, 0xbda37c17, 0xbdd02dc4, 0xbe2edec3, 0x3e2edec3, 0x3dd02dc4, 0x3ba1af64, 0x3b628f60, 0x3b01c01a, 0x3a03c354, 0xb9b480a3, 0xbd002dc1, 0x3d57144e, 0x3cb09f1f, 0xbe2200ee, 0xbe71adfe, 0x3e520d77, 0x3e026068, 0x3c1fa5df, 0x3d753dbf, 0x3dccb71b, 0x3e19b09f}
	wantState := []uint32{0xbebc9975, 0xbf03ba26, 0x3ed8cdd5, 0x3e8df2ff, 0xbe802a66, 0xbeae32e6, 0x3e8bd764, 0x3e3b9dc9, 0xbe0776af, 0xbe29e301, 0x3dfb83cd, 0x3db6ab29, 0xbc698907, 0x3c09fcaf, 0xbce157db, 0xbb9e5403, 0xbc6c2fd2, 0xbd9ffb88, 0xbde086ac, 0xbe317e1d, 0xbd0d3564, 0xbe1ed564, 0xbe4a7529, 0xbea2fe9a, 0xbd5f5ed4, 0xbe6dad03, 0xbe92537e, 0xbeed3e25, 0xbb84000d, 0x3e1551a0, 0x3ee84d2a, 0x3f1a82fd}
	checkGoldenBits(t, "o", o, wantO)
	checkGoldenBits(t, "state", state, wantState)
}

func checkGoldenBits(t *testing.T, name string, got []float32, want []uint32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if b := math.Float32bits(got[i]); b != want[i] {
			t.Fatalf("%s[%d] bits 0x%08x, want 0x%08x", name, i, b, want[i])
		}
	}
}
