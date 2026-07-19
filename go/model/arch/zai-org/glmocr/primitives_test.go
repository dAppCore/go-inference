// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	"math"
	"testing"
)

func TestAddVec_Good(t *testing.T) {
	got := addVec([]float32{1, 2, 3}, []float32{10, 20, 30})
	want := []float32{11, 22, 33}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("addVec = %v, want %v", got, want)
		}
	}
}

func TestAddVec_Bad(t *testing.T) {
	got := addVec([]float32{}, []float32{})
	if len(got) != 0 {
		t.Fatalf("addVec of empty slices = %v, want empty", got)
	}
}

func TestAddVec_Ugly(t *testing.T) {
	got := addVec([]float32{1.5, -2.5}, []float32{-1.5, 2.5})
	if got[0] != 0 || got[1] != 0 {
		t.Fatalf("addVec cancellation = %v, want [0 0]", got)
	}
}

func TestLinearForward_Good(t *testing.T) {
	// y = x·Wᵀ + b, x=[1,2], W=[[1,0],[0,1],[1,1]] (Out=3,In=2), b=[10,20,30]
	w := LinearWeights{Weight: []float32{1, 0, 0, 1, 1, 1}, Bias: []float32{10, 20, 30}, In: 2, Out: 3}
	got := linearForward([]float32{1, 2}, w, 1)
	want := []float32{11, 22, 33}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("linearForward = %v, want %v", got, want)
		}
	}
}

func TestLinearForward_Bad(t *testing.T) {
	// no bias ⇒ pure matmul
	w := LinearWeights{Weight: []float32{2, 0, 0, 2}, In: 2, Out: 2}
	got := linearForward([]float32{3, 4}, w, 1)
	if got[0] != 6 || got[1] != 8 {
		t.Fatalf("linearForward (no bias) = %v, want [6 8]", got)
	}
}

func TestLinearForward_Ugly(t *testing.T) {
	// T=2 rows processed independently
	w := LinearWeights{Weight: []float32{1, 0, 0, 1}, In: 2, Out: 2}
	got := linearForward([]float32{1, 2, 3, 4}, w, 2)
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("linearForward T=2 = %v, want %v", got, want)
		}
	}
}

func TestRmsNormForward_Good(t *testing.T) {
	// x=[3,4], mean(x^2)=(9+16)/2=12.5, rsqrt(12.5+0)=1/sqrt(12.5)
	w := RMSNormWeights{Weight: []float32{1, 1}}
	got := rmsNormForward([]float32{3, 4}, w, 1, 2, 0)
	inv := float32(1.0 / math.Sqrt(12.5))
	want := []float32{3 * inv, 4 * inv}
	if diff := absDiff32(got[0], want[0]) + absDiff32(got[1], want[1]); diff > 1e-6 {
		t.Fatalf("rmsNormForward = %v, want %v", got, want)
	}
}

func TestRmsNormForward_Bad(t *testing.T) {
	// all-zero input never divides by zero thanks to eps
	w := RMSNormWeights{Weight: []float32{5}}
	got := rmsNormForward([]float32{0}, w, 1, 1, 1e-6)
	if got[0] != 0 {
		t.Fatalf("rmsNormForward(0) = %v, want 0", got[0])
	}
}

func TestRmsNormForward_Ugly(t *testing.T) {
	// weight scales AFTER normalisation, not before
	w := RMSNormWeights{Weight: []float32{2, 0}}
	got := rmsNormForward([]float32{1, 1}, w, 1, 2, 0)
	if got[1] != 0 {
		t.Fatalf("rmsNormForward with zero weight channel = %v, want [x 0]", got)
	}
	if got[0] <= 1 { // weight 2 amplifies the normalised value (which has unit-ish magnitude)
		t.Fatalf("rmsNormForward weight scaling = %v, want channel 0 > 1", got)
	}
}

func TestLayerNormForward_Good(t *testing.T) {
	// x=[1,3], mean=2, var=1, inv=1/sqrt(1+1e-5)≈1
	w := LayerNormWeights{Weight: []float32{1, 1}, Bias: []float32{0, 0}}
	got := layerNormForward([]float32{1, 3}, w, 1, 2)
	if got[0] >= 0 || got[1] <= 0 {
		t.Fatalf("layerNormForward = %v, want [negative positive]", got)
	}
	if d := absDiff32(got[0], -got[1]); d > 1e-3 {
		t.Fatalf("layerNormForward not symmetric about the mean: %v", got)
	}
}

func TestLayerNormForward_Bad(t *testing.T) {
	// bias shifts the normalised output
	w := LayerNormWeights{Weight: []float32{0, 0}, Bias: []float32{7, 7}}
	got := layerNormForward([]float32{1, 3}, w, 1, 2)
	if got[0] != 7 || got[1] != 7 {
		t.Fatalf("layerNormForward with zero weight = %v, want [7 7] (pure bias)", got)
	}
}

func TestLayerNormForward_Ugly(t *testing.T) {
	// a constant row has zero variance; eps keeps it finite, output collapses to bias
	w := LayerNormWeights{Weight: []float32{3, 3}, Bias: []float32{1, 1}}
	got := layerNormForward([]float32{5, 5}, w, 1, 2)
	if math.IsNaN(float64(got[0])) || math.IsInf(float64(got[0]), 0) {
		t.Fatalf("layerNormForward on constant row = %v, want finite", got)
	}
	if d := absDiff32(got[0], 1); d > 1e-2 {
		t.Fatalf("layerNormForward on constant row = %v, want ≈bias [1 1]", got)
	}
}

func TestSilu_Good(t *testing.T) {
	if got := silu(0); got != 0 {
		t.Fatalf("silu(0) = %v, want 0", got)
	}
}

func TestSilu_Bad(t *testing.T) {
	// silu is monotonically increasing for x>0 and bounded below by 0 for large negative x
	if got := silu(-20); got >= -0.01 == false {
		t.Fatalf("silu(-20) = %v, want ≈0 from below", got)
	}
}

func TestSilu_Ugly(t *testing.T) {
	// silu(x) ≈ x for large positive x (sigmoid saturates to 1)
	got := silu(20)
	if d := absDiff32(got, 20); d > 1e-3 {
		t.Fatalf("silu(20) = %v, want ≈20", got)
	}
}

func TestGeluExact_Good(t *testing.T) {
	if got := geluExact(0); got != 0 {
		t.Fatalf("geluExact(0) = %v, want 0", got)
	}
}

func TestGeluExact_Bad(t *testing.T) {
	got := geluExact(-10)
	if d := absDiff32(got, 0); d > 1e-3 {
		t.Fatalf("geluExact(-10) = %v, want ≈0", got)
	}
}

func TestGeluExact_Ugly(t *testing.T) {
	// geluExact(x) ≈ x for large positive x
	got := geluExact(10)
	if d := absDiff32(got, 10); d > 1e-3 {
		t.Fatalf("geluExact(10) = %v, want ≈10", got)
	}
}

func TestSwiGLUForward_Good(t *testing.T) {
	gate := LinearWeights{Weight: []float32{1}, In: 1, Out: 1}
	up := LinearWeights{Weight: []float32{1}, In: 1, Out: 1}
	down := LinearWeights{Weight: []float32{1}, In: 1, Out: 1}
	got := swiGLUForward([]float32{0}, gate, up, down, 1)
	if got[0] != 0 {
		t.Fatalf("swiGLUForward(0) = %v, want 0 (silu(0)=0)", got)
	}
}

func TestSwiGLUForward_Bad(t *testing.T) {
	// zero gate weight -> silu(0)=0 -> output always zero regardless of up/down
	gate := LinearWeights{Weight: []float32{0}, In: 1, Out: 1}
	up := LinearWeights{Weight: []float32{99}, In: 1, Out: 1}
	down := LinearWeights{Weight: []float32{99}, In: 1, Out: 1}
	got := swiGLUForward([]float32{5}, gate, up, down, 1)
	if got[0] != 0 {
		t.Fatalf("swiGLUForward with zero gate = %v, want 0", got)
	}
}

func TestSwiGLUForward_Ugly(t *testing.T) {
	// T=2 rows processed independently, distinct results
	gate := LinearWeights{Weight: []float32{1}, In: 1, Out: 1}
	up := LinearWeights{Weight: []float32{1}, In: 1, Out: 1}
	down := LinearWeights{Weight: []float32{1}, In: 1, Out: 1}
	got := swiGLUForward([]float32{1, 2}, gate, up, down, 2)
	if got[0] == got[1] {
		t.Fatalf("swiGLUForward T=2 rows collapsed: %v", got)
	}
}

func TestMhaCore_Good(t *testing.T) {
	// T=1: attention over a single position is trivially the identity (softmax of one score is 1)
	headDim := 2
	q := []float32{1, 0}
	k := []float32{1, 0}
	v := []float32{5, 7}
	got := mhaCore(q, k, v, 1, 1, 1, headDim, false)
	if d := absDiff32(got[0], 5) + absDiff32(got[1], 7); d > 1e-5 {
		t.Fatalf("mhaCore T=1 = %v, want [5 7]", got)
	}
}

func TestMhaCore_Bad(t *testing.T) {
	// causal T=2: row 0 can only see key 0, so its output is v[0] regardless of what v[1] is
	headDim := 1
	q := []float32{1, 1}
	k := []float32{1, 1}
	v := []float32{3, 999}
	got := mhaCore(q, k, v, 2, 1, 1, headDim, true)
	if d := absDiff32(got[0], 3); d > 1e-5 {
		t.Fatalf("mhaCore causal row0 = %v, want ≈3 (must not see key 1)", got[0])
	}
}

func TestMhaCore_Ugly(t *testing.T) {
	// GQA: 2 query heads sharing 1 KV head must both attend the SAME single key/value
	headDim := 1
	q := []float32{1, 2} // head0, head1 (T=1)
	k := []float32{1}    // 1 kv head
	v := []float32{42}
	got := mhaCore(q, k, v, 1, 2, 1, headDim, false)
	if d := absDiff32(got[0], 42) + absDiff32(got[1], 42); d > 1e-5 {
		t.Fatalf("mhaCore GQA both heads = %v, want [42 42] (single key -> softmax is 1)", got)
	}
}

func absDiff32(a, b float32) float64 {
	d := float64(a) - float64(b)
	if d < 0 {
		d = -d
	}
	return d
}
