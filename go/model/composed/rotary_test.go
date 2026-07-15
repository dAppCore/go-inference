// SPDX-Licence-Identifier: EUPL-1.2

package composed

import "testing"

// TestMRoPEInterleavedGolden pins the interleaved mRoPE angle construction against hand-computed values.
// section [1,1,1] over inv_freq [1, 0.5, 0.25] at position (T=2, H=3, W=5): index 0 → T (2·1.0), index 1 →
// H (3·0.5), index 2 → W (5·0.25). section [2,1,1] over four freqs adds index 3 → T (2·0.125), exercising
// the "past a section's span falls back to T" branch (index 3 is a multiple of 3 ⇒ T).
func TestMRoPEInterleavedGolden(t *testing.T) {
	got := mRoPEInterleavedFreqs(2, 3, 5, []float64{1, 0.5, 0.25}, [3]int{1, 1, 1})
	want := []float64{2.0, 1.5, 1.25}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("[1,1,1] freqs[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	got = mRoPEInterleavedFreqs(2, 3, 5, []float64{1, 0.5, 0.25, 0.125}, [3]int{2, 1, 1})
	want = []float64{2.0, 1.5, 1.25, 0.25} // T,H,W,T
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("[2,1,1] freqs[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	t.Log("interleaved mRoPE angles match the hand-computed [T,H,W,…] assignment")
}

// TestMRoPESectionAssignment pins the T/H/W assignment for the real Qwen 3.6 section [11,11,10] over the 32
// rotary pairs (rotary_dim 64 = 0.25·256): 11 temporal, 11 height, 10 width, in the stride-3 interleave.
func TestMRoPESectionAssignment(t *testing.T) {
	section := [3]int{11, 11, 10}
	var counts [3]int
	for i := range 32 {
		counts[mRoPESectionOf(i, section)]++
	}
	if counts != [3]int{11, 11, 10} {
		t.Fatalf("section counts over 32 pairs = %v, want [11 11 10]", counts)
	}
	// spot checks: index 0 → T (multiple of 3), 1 → H (offset 1), 2 → W (offset 2), 29 → W (last, 29<30),
	// 31 → H (31%3==1).
	for _, c := range []struct {
		i, want int
	}{{0, 0}, {1, 1}, {2, 2}, {3, 0}, {29, 2}, {31, 1}} {
		if got := mRoPESectionOf(c.i, section); got != c.want {
			t.Fatalf("mRoPESectionOf(%d) = %d, want %d", c.i, got, c.want)
		}
	}
	t.Log("mRoPE section [11,11,10]: 11 T / 11 H / 10 W over the 32 rotary pairs, stride-3 interleave")
}

// TestMRoPEReducesToPartialRotary is the load-bearing property for the text decode: for a pure-text
// position (all three position dims equal) the interleaved mRoPE angles collapse to the standard 1D
// partial-rotary angles p·inv_freq — for ANY section split — and driving the rotation from those angles is
// BIT-IDENTICAL to applyRotaryHalf (the reduced form the composed attention forward actually uses). So the
// text path is provably the mRoPE reduction, not an approximation.
func TestMRoPEReducesToPartialRotary(t *testing.T) {
	const rotaryDim, headDim = 8, 10 // 2 unrotated tail dims
	const theta = 1e6
	invFreq := rotaryInvFreq(rotaryDim, theta)
	sections := [][3]int{{4, 0, 0}, {2, 1, 1}, {1, 2, 1}, {0, 2, 2}}
	for _, p := range []int{0, 1, 5, 17} {
		for _, section := range sections {
			angles := mRoPEInterleavedFreqs(float64(p), float64(p), float64(p), invFreq, section)
			for i := range invFreq {
				if want := float64(p) * invFreq[i]; angles[i] != want {
					t.Fatalf("p=%d section=%v: angle[%d] = %v, want p·inv_freq %v (text must collapse the interleave)", p, section, i, angles[i], want)
				}
			}
			xa := syn(headDim, 3)
			xb := syn(headDim, 3)
			applyRotaryAngles(xa, angles)
			applyRotaryHalf(xb, p, rotaryDim, theta)
			for i := range headDim {
				if xa[i] != xb[i] {
					t.Fatalf("p=%d section=%v: mRoPE-angle rotation x[%d] = %v != applyRotaryHalf %v", p, section, i, xa[i], xb[i])
				}
			}
		}
	}
	t.Log("text mRoPE (equal 3D positions) is bit-identical to standard partial rotary for every section split — the reduction the forward relies on")
}
