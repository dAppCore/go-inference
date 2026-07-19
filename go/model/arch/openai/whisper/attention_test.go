// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import (
	"math"
	"testing"
)

// TestLinearForward_Good hand-verifies y = x·Wᵀ + b on a 2×2 case: x=[1,2], W=[[1,0],[0,1]] (identity),
// b=[10,20] ⇒ y=[11,22].
func TestLinearForward_Good(t *testing.T) {
	w := LinearWeights{Weight: []float32{1, 0, 0, 1}, Bias: []float32{10, 20}, In: 2, Out: 2}
	got := linearForward([]float32{1, 2}, w, 1)
	want := []float32{11, 22}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("linearForward = %v, want %v", got, want)
		}
	}
}

// TestLinearForward_Bad proves a nil Bias (Whisper's k_proj) adds nothing, not zero-filled garbage or a
// panic.
func TestLinearForward_Bad(t *testing.T) {
	w := LinearWeights{Weight: []float32{2, 0, 0, 2}, Bias: nil, In: 2, Out: 2}
	got := linearForward([]float32{3, 4}, w, 1)
	want := []float32{6, 8}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("linearForward with nil bias = %v, want %v", got, want)
		}
	}
}

// TestLinearForward_Ugly proves T>1 rows are independent (row 1's projection cannot leak into row 0).
func TestLinearForward_Ugly(t *testing.T) {
	w := LinearWeights{Weight: []float32{1, 0, 0, 1}, In: 2, Out: 2}
	got := linearForward([]float32{1, 2, 100, 200}, w, 2)
	want := []float32{1, 2, 100, 200}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("linearForward(T=2) = %v, want %v (rows must not mix)", got, want)
		}
	}
}

// TestLayerNormForward_Good proves weight=1/bias=0 normalises to zero mean, unit variance.
func TestLayerNormForward_Good(t *testing.T) {
	w := LayerNormWeights{Weight: []float32{1, 1, 1, 1}, Bias: []float32{0, 0, 0, 0}}
	got := layerNormForward([]float32{1, 2, 3, 4}, w, 1, 4)
	var mean, variance float64
	for _, v := range got {
		mean += float64(v)
	}
	mean /= 4
	for _, v := range got {
		d := float64(v) - mean
		variance += d * d
	}
	variance /= 4
	if math.Abs(mean) > 1e-5 {
		t.Fatalf("normalised mean = %g, want ~0", mean)
	}
	if math.Abs(variance-1) > 1e-3 {
		t.Fatalf("normalised variance = %g, want ~1", variance)
	}
}

// TestLayerNormForward_Bad proves the affine weight/bias are applied AFTER normalising (weight=2,bias=5
// on a constant row ⇒ every output element is exactly bias, since a constant row normalises to zero).
func TestLayerNormForward_Bad(t *testing.T) {
	w := LayerNormWeights{Weight: []float32{2, 2, 2}, Bias: []float32{5, 5, 5}}
	got := layerNormForward([]float32{7, 7, 7}, w, 1, 3)
	for i, v := range got {
		if math.Abs(float64(v-5)) > 1e-3 {
			t.Fatalf("layerNormForward(constant row)[%d] = %v, want 5 (weight·0+bias)", i, v)
		}
	}
}

func TestGelu_Good(t *testing.T) {
	// Exact reference values: 0.5·x·(1+erf(x/√2)) computed independently in Python's math.erf.
	cases := map[float32]float32{
		-3.0: -0.00404969409489031,
		-1.0: -0.15865525393145707,
		-0.5: -0.15426876936299347,
		0.0:  0.0,
		0.5:  0.3457312306370065,
		1.0:  0.8413447460685429,
		2.5:  2.4844758366855597,
	}
	for x, want := range cases {
		got := gelu(x)
		if math.Abs(float64(got-want)) > 1e-6 {
			t.Fatalf("gelu(%v) = %v, want %v", x, got, want)
		}
	}
}

// TestMHACore_Good proves non-causal attention lets every query see every key: with a uniform (all-zero)
// Q/K score, softmax is uniform, so the output at every position is the mean of V.
func TestMHACore_Good(t *testing.T) {
	// H=1, headDim=2, Tq=Tk=3, all-zero q and k (uniform scores), distinct v rows.
	q := make([]float32, 3*2)
	k := make([]float32, 3*2)
	v := []float32{1, 1, 3, 3, 5, 5}
	out := mhaCore(q, k, v, 3, 3, 1, 2, false)
	wantMean := float32(3) // mean of {1,3,5}
	for i := 0; i < 3; i++ {
		if math.Abs(float64(out[i*2]-wantMean)) > 1e-4 {
			t.Fatalf("mhaCore non-causal row %d = %v, want mean-of-V %v (uniform attention)", i, out[i*2], wantMean)
		}
	}
}

// TestMHACore_Bad proves causal masking blocks the future: at query position 0, only key 0 is visible,
// so a uniform-score causal attention at position 0 must equal v[0] exactly — NOT the mean over all
// three keys the non-causal case (_Good) produces.
func TestMHACore_Bad(t *testing.T) {
	q := make([]float32, 3*2)
	k := make([]float32, 3*2)
	v := []float32{1, 1, 3, 3, 5, 5}
	out := mhaCore(q, k, v, 3, 3, 1, 2, true)
	if out[0] != 1 || out[1] != 1 {
		t.Fatalf("mhaCore causal row 0 = %v, want exactly v[0]=[1,1] (only key 0 is visible)", out[0:2])
	}
	// row 2 (last) sees all three keys, same as the non-causal case.
	if math.Abs(float64(out[2*2]-3)) > 1e-4 {
		t.Fatalf("mhaCore causal row 2 = %v, want mean-of-V 3 (every key ≤2 is visible)", out[2*2])
	}
}

// TestMHACore_Ugly proves multiple heads are independent: head 1's V values never leak into head 0's
// output column range.
func TestMHACore_Ugly(t *testing.T) {
	// H=2, headDim=1, Tq=Tk=1. head0's v=100, head1's v=200 — distinguishable per head.
	q := []float32{0, 0}
	k := []float32{0, 0}
	v := []float32{100, 200}
	out := mhaCore(q, k, v, 1, 1, 2, 1, false)
	if out[0] != 100 || out[1] != 200 {
		t.Fatalf("mhaCore heads = %v, want [100,200] (heads must not mix)", out)
	}
}

// TestProjectQScaled_Good proves the 1/√headDim scale is applied (headDim=4 ⇒ scale 0.5).
func TestProjectQScaled_Good(t *testing.T) {
	w := LinearWeights{Weight: []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, In: 4, Out: 4}
	got := projectQScaled([]float32{2, 4, 6, 8}, w, 1, 4)
	want := []float32{1, 2, 3, 4} // ×0.5
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("projectQScaled = %v, want %v", got, want)
		}
	}
}

func TestSelfAttentionForward_Good(t *testing.T) {
	aw := AttnWeights{
		Q:   LinearWeights{Weight: []float32{1, 0, 0, 1}, Bias: []float32{0, 0}, In: 2, Out: 2},
		K:   LinearWeights{Weight: []float32{1, 0, 0, 1}, In: 2, Out: 2},
		V:   LinearWeights{Weight: []float32{1, 0, 0, 1}, Bias: []float32{0, 0}, In: 2, Out: 2},
		Out: LinearWeights{Weight: []float32{1, 0, 0, 1}, Bias: []float32{0, 0}, In: 2, Out: 2},
	}
	out, err := selfAttentionForward([]float32{1, 2, 3, 4}, 2, 2, 1, false, aw)
	if err != nil {
		t.Fatalf("selfAttentionForward: %v", err)
	}
	if len(out) != 4 {
		t.Fatalf("len = %d, want 4", len(out))
	}
}

func TestSelfAttentionForward_Bad(t *testing.T) {
	aw := AttnWeights{Q: LinearWeights{In: 3, Out: 3}, K: LinearWeights{In: 3, Out: 3}, V: LinearWeights{In: 3, Out: 3}, Out: LinearWeights{In: 3, Out: 3}}
	if _, err := selfAttentionForward(make([]float32, 3), 1, 3, 2, false, aw); err == nil {
		t.Fatal("selfAttentionForward accepted D=3 not divisible by H=2")
	}
}

// TestSelfAttentionForward_Ugly proves H=D (headDim=1, the degenerate single-channel-per-head case)
// still runs without error.
func TestSelfAttentionForward_Ugly(t *testing.T) {
	aw := AttnWeights{
		Q: LinearWeights{Weight: []float32{1, 0, 0, 1}, In: 2, Out: 2}, K: LinearWeights{Weight: []float32{1, 0, 0, 1}, In: 2, Out: 2},
		V: LinearWeights{Weight: []float32{1, 0, 0, 1}, In: 2, Out: 2}, Out: LinearWeights{Weight: []float32{1, 0, 0, 1}, In: 2, Out: 2},
	}
	if _, err := selfAttentionForward([]float32{1, 2}, 1, 2, 2, true, aw); err != nil {
		t.Fatalf("selfAttentionForward(H=D headDim=1): %v", err)
	}
}

func TestPrecomputeCrossKV_Good(t *testing.T) {
	aw := AttnWeights{K: LinearWeights{Weight: []float32{1, 0, 0, 1}, In: 2, Out: 2}, V: LinearWeights{Weight: []float32{2, 0, 0, 2}, In: 2, Out: 2}}
	k, v := precomputeCrossKV([]float32{1, 2, 3, 4}, 2, aw)
	if k[0] != 1 || k[1] != 2 || v[0] != 2 || v[1] != 4 {
		t.Fatalf("precomputeCrossKV k=%v v=%v, want k==input (identity K) v==2×input", k, v)
	}
}

func TestCrossAttentionForward_Good(t *testing.T) {
	aw := AttnWeights{
		Q:   LinearWeights{Weight: []float32{1, 0, 0, 1}, In: 2, Out: 2},
		Out: LinearWeights{Weight: []float32{1, 0, 0, 1}, Bias: []float32{0, 0}, In: 2, Out: 2},
	}
	encK := []float32{0, 0, 0, 0}
	encV := []float32{9, 9, 21, 21}
	out, err := crossAttentionForward([]float32{1, 1}, 1, 2, 1, aw, encK, encV, 2)
	if err != nil {
		t.Fatalf("crossAttentionForward: %v", err)
	}
	want := float32(15) // uniform attention over encV's two rows: mean(9,21)=15
	if math.Abs(float64(out[0]-want)) > 1e-3 {
		t.Fatalf("crossAttentionForward = %v, want ~%v (uniform score ⇒ mean of encV)", out, want)
	}
}

func TestCrossAttentionForward_Bad(t *testing.T) {
	aw := AttnWeights{Q: LinearWeights{In: 3, Out: 3}, Out: LinearWeights{In: 3, Out: 3}}
	if _, err := crossAttentionForward(make([]float32, 3), 1, 3, 2, aw, make([]float32, 3), make([]float32, 3), 1); err == nil {
		t.Fatal("crossAttentionForward accepted D=3 not divisible by H=2")
	}
}

// TestCrossAttentionForward_Ugly proves Tq (decoder length) and Tenc (encoder length) can differ freely
// — cross-attention's query/key counts are independent, unlike self-attention's Tq==Tk.
func TestCrossAttentionForward_Ugly(t *testing.T) {
	aw := AttnWeights{
		Q:   LinearWeights{Weight: []float32{1, 0, 0, 1}, In: 2, Out: 2},
		Out: LinearWeights{Weight: []float32{1, 0, 0, 1}, In: 2, Out: 2},
	}
	encK := make([]float32, 5*2) // Tenc=5
	encV := make([]float32, 5*2)
	out, err := crossAttentionForward(make([]float32, 3*2), 3, 2, 1, aw, encK, encV, 5) // Tq=3
	if err != nil {
		t.Fatalf("crossAttentionForward(Tq≠Tenc): %v", err)
	}
	if len(out) != 3*2 {
		t.Fatalf("len(out) = %d, want %d (Tq×D, independent of Tenc)", len(out), 3*2)
	}
}
