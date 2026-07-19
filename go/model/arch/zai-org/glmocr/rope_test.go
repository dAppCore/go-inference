// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	"math"
	"testing"
)

func TestVisionPosIDs_Good(t *testing.T) {
	// grid 4x4, merge 2 -> 2 blocks per side, each block contributes (mh,mw) in {0,1}x{0,1}
	hpos, wpos := visionPosIDs(1, 4, 4, 2)
	golden := readBlockGoldens(t)
	if len(hpos) != len(golden.Vision.PosIDs) {
		t.Fatalf("visionPosIDs produced %d patches, golden has %d", len(hpos), len(golden.Vision.PosIDs))
	}
	for i, pair := range golden.Vision.PosIDs {
		if hpos[i] != pair[0] || wpos[i] != pair[1] {
			t.Fatalf("visionPosIDs[%d] = (%d,%d), want (%d,%d)", i, hpos[i], wpos[i], pair[0], pair[1])
		}
	}
}

func TestVisionPosIDs_Bad(t *testing.T) {
	// a 2x2 grid with merge 2 collapses to exactly ONE block -> every patch shares that block's
	// (mh,mw) origin (0,0)..(1,1), never exceeding the grid.
	hpos, wpos := visionPosIDs(1, 2, 2, 2)
	if len(hpos) != 4 {
		t.Fatalf("visionPosIDs(1,2,2,2) produced %d patches, want 4", len(hpos))
	}
	for i := range hpos {
		if hpos[i] > 1 || wpos[i] > 1 {
			t.Fatalf("visionPosIDs[%d] = (%d,%d) exceeds the 2x2 grid", i, hpos[i], wpos[i])
		}
	}
}

func TestVisionPosIDs_Ugly(t *testing.T) {
	// gridT=2 repeats the SAME spatial pattern twice (temporal frames don't shift h/w)
	hpos1, wpos1 := visionPosIDs(2, 2, 2, 2)
	if len(hpos1) != 8 {
		t.Fatalf("visionPosIDs(2,2,2,2) produced %d patches, want 8", len(hpos1))
	}
	for i := range 4 {
		if hpos1[i] != hpos1[i+4] || wpos1[i] != wpos1[i+4] {
			t.Fatalf("visionPosIDs temporal frame 2 diverged from frame 1 at %d: (%d,%d) vs (%d,%d)", i, hpos1[i], wpos1[i], hpos1[i+4], wpos1[i+4])
		}
	}
}

func TestVisionRotaryFreqs_Good(t *testing.T) {
	// theta irrelevant at position 0: every frequency row is all-zero
	table := visionRotaryFreqs(3, 8, 10000)
	for _, v := range table[0] {
		if v != 0 {
			t.Fatalf("visionRotaryFreqs[0] = %v, want all-zero (pos 0 * anything = 0)", table[0])
		}
	}
}

func TestVisionRotaryFreqs_Bad(t *testing.T) {
	// headDim/2/2 = 1 frequency for headDim=4 (the toy geometry this package's golden uses)
	table := visionRotaryFreqs(2, 4, 10000)
	if len(table[0]) != 1 {
		t.Fatalf("visionRotaryFreqs headDim=4 row width = %d, want 1", len(table[0]))
	}
}

func TestVisionRotaryFreqs_Ugly(t *testing.T) {
	// position 1's frequency IS inv_freq[0] = 1/theta^0 = 1 exactly, regardless of theta
	table := visionRotaryFreqs(2, 4, 12345)
	if d := absDiff32(table[1][0], 1); d > 1e-6 {
		t.Fatalf("visionRotaryFreqs[1][0] = %v, want 1 (inv_freq[0] is always theta^0=1)", table[1][0])
	}
}

func TestVisionCosSin_Good(t *testing.T) {
	// position (0,0): every angle is 0 -> cos=1, sin=0 everywhere
	cos, sin := visionCosSin([]int{0}, []int{0}, 8, 10000)
	for i := range cos[0] {
		if d := absDiff32(cos[0][i], 1); d > 1e-6 {
			t.Fatalf("visionCosSin cos[0][%d] = %v, want 1", i, cos[0][i])
		}
		if d := absDiff32(sin[0][i], 0); d > 1e-6 {
			t.Fatalf("visionCosSin sin[0][%d] = %v, want 0", i, sin[0][i])
		}
	}
}

func TestVisionCosSin_Bad(t *testing.T) {
	// cos²+sin²=1 pointwise, for a non-trivial position
	cos, sin := visionCosSin([]int{3}, []int{5}, 8, 10000)
	for i := range cos[0] {
		sum := float64(cos[0][i])*float64(cos[0][i]) + float64(sin[0][i])*float64(sin[0][i])
		if d := sum - 1; d > 1e-4 || d < -1e-4 {
			t.Fatalf("visionCosSin[%d]: cos²+sin² = %v, want 1", i, sum)
		}
	}
}

func TestVisionCosSin_Ugly(t *testing.T) {
	// the first quarter of the angle vector depends ONLY on hpos, the second quarter ONLY on
	// wpos (before doubling) — swapping hpos/wpos between two patches with h≠w must swap which
	// quarter changes
	cosA, _ := visionCosSin([]int{2}, []int{9}, 8, 10000)
	cosB, _ := visionCosSin([]int{9}, []int{2}, 8, 10000)
	quarter := 8 / 4
	sameFirst := true
	for i := range quarter {
		if cosA[0][i] != cosB[0][i] {
			sameFirst = false
		}
	}
	if sameFirst {
		t.Fatalf("visionCosSin: swapping hpos/wpos did not change the h-frequency quarter")
	}
}

func TestRotateHalf_Good(t *testing.T) {
	got := rotateHalf([]float32{1, 2, 3, 4})
	want := []float32{-3, -4, 1, 2}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("rotateHalf = %v, want %v", got, want)
		}
	}
}

func TestRotateHalf_Bad(t *testing.T) {
	got := rotateHalf([]float32{0, 0})
	if got[0] != 0 || got[1] != 0 {
		t.Fatalf("rotateHalf of zeros = %v, want zeros", got)
	}
}

func TestRotateHalf_Ugly(t *testing.T) {
	// applying rotateHalf FOUR times returns to the original (a full rotation cycle: two
	// applications negate, so it takes four to return to +x)
	x := []float32{1, 2, 3, 4}
	y := rotateHalf(rotateHalf(rotateHalf(rotateHalf(x))))
	for i := range x {
		if d := absDiff32(x[i], y[i]); d > 1e-6 {
			t.Fatalf("rotateHalf^4 = %v, want back to %v", y, x)
		}
	}
}

func TestApplyRopeVision_Good(t *testing.T) {
	// cos=1,sin=0 (position 0) is the identity rotation
	x := []float32{1, 2, 3, 4}
	cos := []float32{1, 1, 1, 1}
	sin := []float32{0, 0, 0, 0}
	got := applyRopeVision(x, cos, sin)
	for i := range x {
		if got[i] != x[i] {
			t.Fatalf("applyRopeVision identity = %v, want %v", got, x)
		}
	}
}

func TestApplyRopeVision_Bad(t *testing.T) {
	// rotation preserves the vector's norm
	x := []float32{3, 4, 5, 6}
	cos := []float32{0.6, 0.8, 0.6, 0.8}
	sin := []float32{0.8, 0.6, 0.8, 0.6}
	got := applyRopeVision(x, cos, sin)
	var beforeSq, afterSq float64
	for i := range x {
		beforeSq += float64(x[i]) * float64(x[i])
		afterSq += float64(got[i]) * float64(got[i])
	}
	if d := beforeSq - afterSq; d > 1e-2 || d < -1e-2 {
		t.Fatalf("applyRopeVision changed the vector norm: before=%v after=%v", beforeSq, afterSq)
	}
}

func TestApplyRopeVision_Ugly(t *testing.T) {
	// a 90-degree rotation (cos=0,sin=1) sends [a,b] (half) to exactly rotateHalf's output
	x := []float32{1, 2, 3, 4}
	cos := []float32{0, 0, 0, 0}
	sin := []float32{1, 1, 1, 1}
	got := applyRopeVision(x, cos, sin)
	want := rotateHalf(x)
	for i := range want {
		if d := absDiff32(got[i], want[i]); d > 1e-6 {
			t.Fatalf("applyRopeVision at 90deg = %v, want rotateHalf(x) = %v", got, want)
		}
	}
}

func TestTextRotaryFreqPos_Good(t *testing.T) {
	// mrope_section [1,1,2] (headDim=8,half=4): band0(k=0)->temporal, band1(k=1)->height,
	// band2,3(k=2,3)->width — distinct position values per axis expose exactly which one each
	// frequency index actually used.
	freqs := textRotaryFreqPos(100, 200, 300, 8, 10000, []int{1, 1, 2})
	invFreq := func(k int) float64 { return 1.0 / math.Pow(10000, float64(2*k)/8) }
	want := []float32{
		float32(100 * invFreq(0)),
		float32(200 * invFreq(1)),
		float32(300 * invFreq(2)),
		float32(300 * invFreq(3)),
	}
	for i := range want {
		if d := absDiff32(freqs[i], want[i]); d > 1e-3 {
			t.Fatalf("textRotaryFreqPos[%d] = %v, want %v (band selection wrong)", i, freqs[i], want[i])
		}
	}
}

func TestTextRotaryFreqPos_Bad(t *testing.T) {
	// a plain text position (t==h==w) collapses to ordinary 1D rope regardless of section split
	freqs := textRotaryFreqPos(7, 7, 7, 8, 10000, []int{1, 1, 2})
	for k, f := range freqs {
		want := float32(7.0 / math.Pow(10000, float64(2*k)/8))
		if d := absDiff32(f, want); d > 1e-3 {
			t.Fatalf("textRotaryFreqPos[%d] (t=h=w) = %v, want %v", k, f, want)
		}
	}
}

func TestTextRotaryFreqPos_Ugly(t *testing.T) {
	// position 0 on every axis is the zero vector regardless of mrope_section
	freqs := textRotaryFreqPos(0, 0, 0, 8, 10000, []int{1, 1, 2})
	for i, f := range freqs {
		if f != 0 {
			t.Fatalf("textRotaryFreqPos[%d] at position 0 = %v, want 0", i, f)
		}
	}
}

func TestApplyRopeTextPair_Good(t *testing.T) {
	// freq=0 (position 0) is the identity
	x := []float32{1, 2, 3, 4}
	got := applyRopeTextPair(x, []float32{0, 0})
	for i := range x {
		if d := absDiff32(got[i], x[i]); d > 1e-6 {
			t.Fatalf("applyRopeTextPair identity = %v, want %v", got, x)
		}
	}
}

func TestApplyRopeTextPair_Bad(t *testing.T) {
	// rotation preserves each PAIR's norm independently
	x := []float32{3, 4, 5, 12}
	got := applyRopeTextPair(x, []float32{0.5, 1.2})
	pairNorm := func(s []float32, i int) float64 { return float64(s[i])*float64(s[i]) + float64(s[i+1])*float64(s[i+1]) }
	if d := pairNorm(x, 0) - pairNorm(got, 0); d > 1e-2 || d < -1e-2 {
		t.Fatalf("applyRopeTextPair changed pair-0 norm: %v vs %v", pairNorm(x, 0), pairNorm(got, 0))
	}
	if d := pairNorm(x, 2) - pairNorm(got, 2); d > 1e-2 || d < -1e-2 {
		t.Fatalf("applyRopeTextPair changed pair-1 norm: %v vs %v", pairNorm(x, 2), pairNorm(got, 2))
	}
}

func TestApplyRopeTextPair_Ugly(t *testing.T) {
	// pairs rotate INDEPENDENTLY: a zero freq on pair 0 and a 90deg rotation on pair 1
	// must leave pair 0 untouched while pair 1 rotates as (a,b) -> (-b,a)
	x := []float32{9, -9, 1, 2}
	got := applyRopeTextPair(x, []float32{0, float32(math.Pi / 2)})
	if d := absDiff32(got[0], 9) + absDiff32(got[1], -9); d > 1e-3 {
		t.Fatalf("applyRopeTextPair pair 0 (freq=0) = %v, want unchanged [9 -9]", got[:2])
	}
	if d := absDiff32(got[2], -2) + absDiff32(got[3], 1); d > 1e-3 {
		t.Fatalf("applyRopeTextPair pair 1 (90deg) = %v, want [-2 1]", got[2:])
	}
}
