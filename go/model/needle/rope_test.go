// SPDX-Licence-Identifier: EUPL-1.2

package needle

import (
	"math"
	"testing"
)

// TestRope_apply_Position0Identity confirms position 0 is the identity rotation
// (cos 0 = 1, sin 0 = 0), so RoPE at the first token leaves the vector unchanged.
func TestRope_apply_Position0Identity(t *testing.T) {
	rt := newRopeTable(4, 10000, 4)
	vec := []float32{1, 2, 3, 4}
	rt.apply(vec, 0)
	for i, v := range []float32{1, 2, 3, 4} {
		if math.Abs(float64(vec[i]-v)) > 1e-6 {
			t.Errorf("rope@0 [%d] = %.6f, want %.6f (identity)", i, vec[i], v)
		}
	}
}

// TestRope_apply_KnownRotation checks the NeoX rotate_half formula at position 1
// for headDim 4, theta 10000, vec = [1,0,0,0]:
//
//	inv_freq = [1, 0.01]; out[0] = cos(1), out[2] = sin(1), rest 0.
func TestRope_apply_KnownRotation(t *testing.T) {
	rt := newRopeTable(4, 10000, 4)
	vec := []float32{1, 0, 0, 0}
	rt.apply(vec, 1)
	want := []float32{float32(math.Cos(1)), 0, float32(math.Sin(1)), 0}
	for i := range want {
		if math.Abs(float64(vec[i]-want[i])) > 1e-6 {
			t.Errorf("rope@1 [%d] = %.6f, want %.6f", i, vec[i], want[i])
		}
	}
}

// TestRope_apply_NormPreserving verifies a rotation preserves the paired-dimension
// norm (a rotation is orthogonal), a cheap invariant that catches sign/index bugs.
func TestRope_apply_NormPreserving(t *testing.T) {
	rt := newRopeTable(8, 10000, 16)
	vec := []float32{0.3, -1.2, 0.7, 0.1, -0.5, 0.9, 0.2, -0.8}
	var before float32
	for _, v := range vec {
		before += v * v
	}
	rt.apply(vec, 5)
	var after float32
	for _, v := range vec {
		after += v * v
	}
	if math.Abs(float64(before-after)) > 1e-4 {
		t.Errorf("rope changed vector norm: %.6f -> %.6f", before, after)
	}
}
