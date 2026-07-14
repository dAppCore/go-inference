// SPDX-Licence-Identifier: EUPL-1.2

package needle

import (
	"math"
	"testing"
)

// TestNorm_zcRMSNorm_Formula pins the exact ZCRMSNorm maths: variance is the mean
// of squares (NOT centred), and the weight enters as (1 + weight). x = [1,2,3,4],
// weight = 0 -> x / rms(x), rms = sqrt(30/4).
func TestNorm_zcRMSNorm_Formula(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	weight := []float32{0, 0, 0, 0}
	inv := float32(1.0 / math.Sqrt(7.5+1e-6))
	want := []float32{1 * inv, 2 * inv, 3 * inv, 4 * inv}
	got := zcRMSNorm(x, weight, 1e-6)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Errorf("zcRMSNorm[%d] = %.6f, want %.6f", i, got[i], want[i])
		}
	}
}

// TestNorm_zcRMSNorm_WeightShift confirms the (1 + weight) scale: a weight of 1
// doubles the normalised output, a weight of 0 leaves it, matching the
// zero-centred convention (init-0 weight => identity scale).
func TestNorm_zcRMSNorm_WeightShift(t *testing.T) {
	x := []float32{2, 0, 0, 0} // rms = sqrt(4/4) = 1, so normalised == x
	base := zcRMSNorm(x, []float32{0, 0, 0, 0}, 1e-6)
	if math.Abs(float64(base[0]-2)) > 1e-4 {
		t.Fatalf("weight 0: got %.6f, want 2", base[0])
	}
	shifted := zcRMSNorm(x, []float32{1, 0, 0, 0}, 1e-6)
	if math.Abs(float64(shifted[0]-4)) > 1e-4 {
		t.Fatalf("weight 1: got %.6f, want 4 (2 * (1+1))", shifted[0])
	}
}

// TestNorm_zcRMSNorm_NotMeanCentred guards the classic mistake: if the mean were
// subtracted, a constant vector would normalise to zero. ZCRMSNorm must NOT do
// that — a constant vector normalises to +-1 * (1+weight).
func TestNorm_zcRMSNorm_NotMeanCentred(t *testing.T) {
	got := zcRMSNorm([]float32{3, 3, 3, 3}, []float32{0, 0, 0, 0}, 1e-6)
	for i, v := range got {
		if math.Abs(float64(v-1)) > 1e-4 {
			t.Errorf("constant-vector norm[%d] = %.6f, want 1 (mean must NOT be subtracted)", i, v)
		}
	}
}
