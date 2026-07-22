// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import (
	"math"
	"testing"
)

// TestSphereMarginalDensity_Good checks the density is symmetric (f(x) =
// f(-x)) and peaks at x=0 — both direct consequences of the closed form for
// any d.
func TestSphereMarginalDensity_Good(t *testing.T) {
	for _, d := range []int{64, 128, 256} {
		c := sphereMarginalLogConst(d)
		peak := sphereMarginalDensity(0, d, c)
		for _, x := range []float64{0.1, 0.2, 0.3} {
			pos := sphereMarginalDensity(x, d, c)
			neg := sphereMarginalDensity(-x, d, c)
			if !approxEqual(pos, neg, 1e-12) {
				t.Errorf("d=%d: f(%v)=%v, f(%v)=%v, want equal (symmetry)", d, x, pos, -x, neg)
			}
			if pos > peak {
				t.Errorf("d=%d: f(%v)=%v exceeds f(0)=%v, want the peak at 0", d, x, pos, peak)
			}
		}
	}
}

// TestSphereMarginalDensity_Ugly checks the density is exactly 0 outside
// [-1,1] — the sphere-marginal support boundary.
func TestSphereMarginalDensity_Ugly(t *testing.T) {
	c := sphereMarginalLogConst(128)
	for _, x := range []float64{1, -1, 1.5, -2} {
		if got := sphereMarginalDensity(x, 128, c); got != 0 {
			t.Errorf("f(%v) = %v, want 0 (outside [-1,1] support)", x, got)
		}
	}
}

// TestBuildDensityTable_Good checks the cumulative mass over the whole
// domain is ≈1 (a valid density) and the first moment is ≈0 (symmetric
// about zero) — the two invariants Lloyd-Max's cell integrals depend on.
func TestBuildDensityTable_Good(t *testing.T) {
	for _, d := range []int{64, 128, 256} {
		tbl := densityTableFor(d)
		mass, moment := tbl.integrate(-1, 1)
		if !approxEqual(mass, 1, 1e-6) {
			t.Errorf("d=%d: total mass = %v, want ≈1", d, mass)
		}
		if !approxEqual(moment, 0, 1e-6) {
			t.Errorf("d=%d: total first moment = %v, want ≈0 (symmetric)", d, moment)
		}
	}
}

// TestDensityTableIntegrate_Good checks integrating the two halves of the
// domain separately sums to the whole-domain integral (additivity).
func TestDensityTableIntegrate_Good(t *testing.T) {
	tbl := densityTableFor(128)
	leftMass, _ := tbl.integrate(-1, 0)
	rightMass, _ := tbl.integrate(0, 1)
	wholeMass, _ := tbl.integrate(-1, 1)
	if !approxEqual(leftMass+rightMass, wholeMass, 1e-9) {
		t.Errorf("integrate(-1,0)+integrate(0,1) = %v, want integrate(-1,1) = %v", leftMass+rightMass, wholeMass)
	}
	// Symmetric density: each half should carry ≈half the mass.
	if !approxEqual(leftMass, 0.5, 1e-6) {
		t.Errorf("integrate(-1,0) = %v, want ≈0.5", leftMass)
	}
}

// TestDensityTableAt_Ugly checks the interpolated cdf/m1 clamp correctly at
// and beyond the domain boundary.
func TestDensityTableAt_Ugly(t *testing.T) {
	tbl := densityTableFor(128)
	cdf, m1 := tbl.at(-5)
	if cdf != 0 || m1 != 0 {
		t.Errorf("at(-5) = (%v,%v), want (0,0) below the domain", cdf, m1)
	}
	cdfHi, m1Hi := tbl.at(5)
	cdfEnd, m1End := tbl.at(1)
	if cdfHi != cdfEnd || m1Hi != m1End {
		t.Errorf("at(5) = (%v,%v), want the domain's upper endpoint (%v,%v)", cdfHi, m1Hi, cdfEnd, m1End)
	}
}

// TestDensityTableQuantile_Good checks the quantile function's boundary and
// median values against the known symmetric-density facts.
func TestDensityTableQuantile_Good(t *testing.T) {
	tbl := densityTableFor(128)
	if got := tbl.quantile(0); got != -1 {
		t.Errorf("quantile(0) = %v, want -1", got)
	}
	if got := tbl.quantile(1); got != 1 {
		t.Errorf("quantile(1) = %v, want 1", got)
	}
	if got := tbl.quantile(0.5); !approxEqual(got, 0, 1e-3) {
		t.Errorf("quantile(0.5) = %v, want ≈0 (symmetric median)", got)
	}
}

// TestDensityTableQuantile_Ugly checks quantile is monotone non-decreasing
// across a spread of probabilities — a broken binary search would show up
// as a non-monotone result.
func TestDensityTableQuantile_Ugly(t *testing.T) {
	tbl := densityTableFor(128)
	prev := math.Inf(-1)
	for _, p := range []float64{0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99} {
		x := tbl.quantile(p)
		if x < prev {
			t.Errorf("quantile(%v) = %v is less than quantile at a lower p (%v), want monotone", p, x, prev)
		}
		prev = x
	}
}

// TestDensityTableFor_Good checks caching returns the identical pointer for
// repeated calls at the same d.
func TestDensityTableFor_Good(t *testing.T) {
	a := densityTableFor(200)
	b := densityTableFor(200)
	if a != b {
		t.Error("densityTableFor(200) called twice returned different pointers, want the cached table")
	}
}
