// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import (
	"math"
	"testing"
)

// TestLloydMax_Good checks the solved centroids are sorted ascending and
// symmetric about zero — both guaranteed by the sphere-marginal density's
// symmetry, and both properties nearestCentroid's linear scan and the
// paper's distortion targets depend on.
func TestLloydMax_Good(t *testing.T) {
	tbl := densityTableFor(128)
	for _, levels := range []int{2, 4, 8, 16} {
		c := lloydMax(tbl, levels)
		if len(c) != levels {
			t.Fatalf("levels=%d: lloydMax returned %d centroids", levels, len(c))
		}
		for i := 1; i < len(c); i++ {
			if c[i-1] >= c[i] {
				t.Errorf("levels=%d: centroids not strictly sorted at %d: %v >= %v", levels, i, c[i-1], c[i])
			}
		}
		for i := 0; i < len(c); i++ {
			mirror := c[len(c)-1-i]
			if !approxEqual(c[i], -mirror, 1e-4) {
				t.Errorf("levels=%d: centroid[%d]=%v is not the mirror of centroid[%d]=%v", levels, i, c[i], len(c)-1-i, mirror)
			}
		}
	}
}

// TestLloydMax_Ugly checks the degenerate 1-level case: the single centroid
// is the density's overall mean, which is 0 by symmetry — Q_prod's stage 1
// at total bit-width 1 depends on this being exactly the no-information
// reconstruction.
func TestLloydMax_Ugly(t *testing.T) {
	tbl := densityTableFor(128)
	c := lloydMax(tbl, 1)
	if len(c) != 1 {
		t.Fatalf("lloydMax(_, 1) returned %d centroids, want 1", len(c))
	}
	if !approxEqual(c[0], 0, 1e-9) {
		t.Errorf("lloydMax(_, 1)[0] = %v, want ≈0", c[0])
	}
}

// TestLloydMax_Bad checks levels <= 0 returns nil rather than panicking.
func TestLloydMax_Bad(t *testing.T) {
	tbl := densityTableFor(128)
	if got := lloydMax(tbl, 0); got != nil {
		t.Errorf("lloydMax(_, 0) = %v, want nil", got)
	}
}

// TestNearestCentroid_Good checks a straightforward nearest-neighbour pick.
func TestNearestCentroid_Good(t *testing.T) {
	centroids := []float64{-0.5, 0.5}
	if got := nearestCentroid(0.9, centroids); got != 1 {
		t.Errorf("nearestCentroid(0.9, {-0.5,0.5}) = %d, want 1", got)
	}
	if got := nearestCentroid(-0.9, centroids); got != 0 {
		t.Errorf("nearestCentroid(-0.9, {-0.5,0.5}) = %d, want 0", got)
	}
}

// TestNearestCentroid_Ugly checks a tie (exactly equidistant) resolves to
// the first (lowest-index) centroid — a deterministic, documented tie-break.
func TestNearestCentroid_Ugly(t *testing.T) {
	centroids := []float64{-1, 1}
	if got := nearestCentroid(0, centroids); got != 0 {
		t.Errorf("nearestCentroid(0, {-1,1}) = %d, want 0 (first on a tie)", got)
	}
}

// TestCentroidsFor_Good checks the level count matches 2^bits and caching
// returns the identical slice for repeated calls.
func TestCentroidsFor_Good(t *testing.T) {
	for bits := 1; bits <= 4; bits++ {
		c := centroidsFor(128, bits)
		if len(c) != 1<<uint(bits) {
			t.Errorf("bits=%d: centroidsFor returned %d centroids, want %d", bits, len(c), 1<<uint(bits))
		}
	}
	a := centroidsFor(128, 2)
	b := centroidsFor(128, 2)
	if &a[0] != &b[0] {
		t.Error("centroidsFor(128,2) called twice returned different backing arrays, want the cached slice")
	}
}

// TestCentroidsFor_Ugly checks bits == 0 returns the single {0} centroid
// (Q_prod's degenerate b=1 stage 1).
func TestCentroidsFor_Ugly(t *testing.T) {
	c := centroidsFor(128, 0)
	if len(c) != 1 || !approxEqual(c[0], 0, 1e-9) {
		t.Errorf("centroidsFor(128,0) = %v, want {≈0}", c)
	}
}

// TestLloydMax_DistortionMonotoneInBits_Good checks that adding bits
// (doubling the codebook) never increases the density's expected squared
// quantisation error — the basic sanity a broken solver (e.g. one that
// doesn't actually converge) would violate.
func TestLloydMax_DistortionMonotoneInBits_Good(t *testing.T) {
	tbl := densityTableFor(128)
	prevMSE := math.Inf(1)
	for bits := 1; bits <= 4; bits++ {
		c := lloydMax(tbl, 1<<uint(bits))
		mse := expectedQuantisationMSE(tbl, c)
		if mse > prevMSE+1e-9 {
			t.Errorf("bits=%d: MSE %v exceeds the previous (fewer-bit) MSE %v, want non-increasing", bits, mse, prevMSE)
		}
		prevMSE = mse
	}
}

// expectedQuantisationMSE numerically estimates E[(X - centroid(X))²] over
// table's density by summing each cell's contribution: ∫(x-c)²f(x)dx =
// ∫x²f(x)dx - 2c·∫x·f(x)dx + c²·∫f(x)dx. The table's cached integrate gives
// the last two terms exactly; ∫x²f(x)dx has no cached table, so this
// test-only helper estimates it with a midpoint Riemann sum against the
// exact closed-form density — accurate enough for the loose monotonicity
// check above.
func expectedQuantisationMSE(table *densityTable, centroids []float64) float64 {
	boundaries := make([]float64, len(centroids)+1)
	boundaries[0], boundaries[len(centroids)] = -1, 1
	for i := 1; i < len(centroids); i++ {
		boundaries[i] = (centroids[i-1] + centroids[i]) / 2
	}
	logConst := sphereMarginalLogConst(128)
	const steps = 20000
	step := 2.0 / steps

	var mse float64
	for cellIdx, c := range centroids {
		lo, hi := boundaries[cellIdx], boundaries[cellIdx+1]
		mass, moment := table.integrate(lo, hi)

		n := int((hi-lo)/step) + 1
		h := (hi - lo) / float64(n)
		var x2f float64
		for i := 0; i < n; i++ {
			x := lo + (float64(i)+0.5)*h
			x2f += x * x * sphereMarginalDensity(x, 128, logConst) * h
		}
		mse += x2f - 2*c*moment + c*c*mass
	}
	return mse
}
