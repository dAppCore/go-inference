// SPDX-Licence-Identifier: EUPL-1.2

package needle

import (
	"math"
	"testing"
)

// TestLinalg_linearNoBias_RowDot confirms y = x . Wᵀ: output o is x dotted with
// weight row o (row-major [outDim, inDim]).
func TestLinalg_linearNoBias_RowDot(t *testing.T) {
	x := []float32{1, 2, 3}
	// W = [[1,0,0],[0,1,0],[1,1,1]] flattened row-major.
	w := []float32{1, 0, 0, 0, 1, 0, 1, 1, 1}
	got := linearNoBias(x, w, 3, 3)
	want := []float32{1, 2, 6}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("linearNoBias[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestLinalg_softmaxInPlace_Distribution confirms softmax sums to 1 and is
// monotonic in the inputs.
func TestLinalg_softmaxInPlace_Distribution(t *testing.T) {
	s := []float32{1, 2, 3}
	softmaxInPlace(s)
	var sum float32
	for _, v := range s {
		sum += v
	}
	if math.Abs(float64(sum-1)) > 1e-6 {
		t.Errorf("softmax sum = %v, want 1", sum)
	}
	if !(s[0] < s[1] && s[1] < s[2]) {
		t.Errorf("softmax not monotonic: %v", s)
	}
}

// TestLinalg_sigmoid_Midpoint confirms sigmoid(0) = 0.5, the init value of every
// residual gate (weight init 0 -> half-open gate).
func TestLinalg_sigmoid_Midpoint(t *testing.T) {
	if got := sigmoid(0); math.Abs(float64(got-0.5)) > 1e-6 {
		t.Errorf("sigmoid(0) = %v, want 0.5", got)
	}
}

// TestLinalg_clipAdd_Clamps confirms the bf16-range clamp mirrors _add_clipped.
func TestLinalg_clipAdd_Clamps(t *testing.T) {
	if got := clipAdd(65000, 1000); got != 65500 {
		t.Errorf("clipAdd over-range = %v, want 65500", got)
	}
	if got := clipAdd(-65000, -1000); got != -65500 {
		t.Errorf("clipAdd under-range = %v, want -65500", got)
	}
	if got := clipAdd(1.5, 2.5); got != 4 {
		t.Errorf("clipAdd in-range = %v, want 4", got)
	}
}
