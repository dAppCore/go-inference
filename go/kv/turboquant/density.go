// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import (
	"math"
	"sync"
)

// densityGridPoints is the fixed quadrature grid size used to build a
// densityTable. The sphere-marginal density concentrates within roughly
// ±12/√d of zero (concentration of measure: a coordinate of a random unit
// vector in dimension d has standard deviation ≈1/√d), so a uniform grid
// across the full [-1,1] domain at this resolution still lands thousands of
// nodes inside the density's active region even at d=256 — comfortably
// enough for the composite-trapezoidal cumulative integrals below to be
// accurate to well beyond the ±15% oracle tolerance this package is held to.
const densityGridPoints = 131073 // 2^17 + 1

// sphereMarginalLogConst returns the natural log of the sphere-marginal
// density's normalising constant Γ(d/2) / (√π·Γ((d-1)/2)), computed via
// Lgamma so it stays finite for the head dims this package targets (d up to
// a few hundred) where Γ(d/2) itself is astronomically large.
//
//	c := math.Exp(sphereMarginalLogConst(128))
func sphereMarginalLogConst(d int) float64 {
	dl := float64(d)
	lg1, _ := math.Lgamma(dl / 2)
	lg2, _ := math.Lgamma((dl - 1) / 2)
	return lg1 - 0.5*math.Log(math.Pi) - lg2
}

// sphereMarginalDensity evaluates f(x) = Γ(d/2)/(√π·Γ((d-1)/2))·(1-x²)^((d-3)/2)
// — the marginal density of one coordinate of a uniform random vector on the
// (d-1)-sphere — the density TurboQuant's Lloyd-Max quantiser is built
// against. logConst is sphereMarginalLogConst(d), passed in so a caller
// evaluating the density many times (building a quadrature grid) pays the
// Lgamma cost once.
//
//	c := sphereMarginalLogConst(128)
//	sphereMarginalDensity(0, 128, c) // the density's peak, > 0
func sphereMarginalDensity(x float64, d int, logConst float64) float64 {
	if x <= -1 || x >= 1 {
		return 0
	}
	exponent := (float64(d) - 3) / 2
	return math.Exp(logConst + exponent*math.Log(1-x*x))
}

// densityTable is a fixed quadrature grid over [-1,1] for the dimension-d
// sphere-marginal density, plus cumulative trapezoidal integrals of f and
// x·f — so a Lloyd-Max cell mean becomes two table lookups (integrate)
// instead of a fresh numerical integral per iteration.
type densityTable struct {
	x   []float64 // grid nodes, -1..1, ascending
	cdf []float64 // cdf[i] = ∫_{-1}^{x[i]} f
	m1  []float64 // m1[i]  = ∫_{-1}^{x[i]} t·f(t) dt
}

// buildDensityTable constructs the quadrature grid and cumulative tables for
// dimension d. Expensive (O(densityGridPoints)) but pure and
// dimension-only — callers should go through densityTableFor to memoise it.
func buildDensityTable(d int) *densityTable {
	n := densityGridPoints
	logConst := sphereMarginalLogConst(d)
	t := &densityTable{
		x:   make([]float64, n),
		cdf: make([]float64, n),
		m1:  make([]float64, n),
	}
	step := 2.0 / float64(n-1)
	fPrev := sphereMarginalDensity(-1, d, logConst)
	gPrev := -1 * fPrev
	t.x[0] = -1
	for i := 1; i < n; i++ {
		xi := -1 + float64(i)*step
		if i == n-1 {
			xi = 1 // avoid float drift missing the exact endpoint
		}
		fi := sphereMarginalDensity(xi, d, logConst)
		gi := xi * fi
		t.x[i] = xi
		t.cdf[i] = t.cdf[i-1] + 0.5*(fPrev+fi)*step
		t.m1[i] = t.m1[i-1] + 0.5*(gPrev+gi)*step
		fPrev, gPrev = fi, gi
	}
	// Normalise away the (tiny) composite-trapezoidal quadrature error so
	// cdf[last] is exactly 1 — cell masses computed from this table then
	// partition the density exactly rather than off by the grid's residual.
	total := t.cdf[n-1]
	if total > 0 {
		for i := range t.cdf {
			t.cdf[i] /= total
			t.m1[i] /= total
		}
	}
	return t
}

// integrate returns (∫_lo^hi f, ∫_lo^hi t·f(t) dt) via linear interpolation
// into the cumulative tables — the mass and first moment of one Lloyd-Max
// cell.
//
//	tbl := densityTableFor(128)
//	mass, moment := tbl.integrate(-1, 1) // mass ≈ 1, moment ≈ 0 (symmetric)
func (t *densityTable) integrate(lo, hi float64) (mass, moment float64) {
	cLo, mLo := t.at(lo)
	cHi, mHi := t.at(hi)
	return cHi - cLo, mHi - mLo
}

// at linearly interpolates the cumulative tables at x, via binary search
// over the grid nodes.
func (t *densityTable) at(x float64) (cdf, m1 float64) {
	if x <= -1 {
		return 0, 0
	}
	if x >= 1 {
		return t.cdf[len(t.cdf)-1], t.m1[len(t.m1)-1]
	}
	lo, hi := 0, len(t.x)-1
	for lo < hi {
		mid := (lo + hi) / 2
		if t.x[mid] < x {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo == 0 {
		return t.cdf[0], t.m1[0]
	}
	x0, x1 := t.x[lo-1], t.x[lo]
	frac := (x - x0) / (x1 - x0)
	cdf = t.cdf[lo-1] + frac*(t.cdf[lo]-t.cdf[lo-1])
	m1 = t.m1[lo-1] + frac*(t.m1[lo]-t.m1[lo-1])
	return cdf, m1
}

// quantile returns the value x such that ∫_{-1}^{x} f ≈ p, via binary search
// over the cdf table — used to seed Lloyd-Max's initial centroids at the
// density's quantiles (fast, empty-cell-free convergence).
//
//	tbl := densityTableFor(128)
//	tbl.quantile(0.5) // ≈ 0, the median of a symmetric density
func (t *densityTable) quantile(p float64) float64 {
	if p <= 0 {
		return -1
	}
	if p >= 1 {
		return 1
	}
	lo, hi := 0, len(t.cdf)-1
	for lo < hi {
		mid := (lo + hi) / 2
		if t.cdf[mid] < p {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo == 0 {
		return t.x[0]
	}
	c0, c1 := t.cdf[lo-1], t.cdf[lo]
	if c1 == c0 {
		return t.x[lo]
	}
	frac := (p - c0) / (c1 - c0)
	return t.x[lo-1] + frac*(t.x[lo]-t.x[lo-1])
}

var densityTableCache sync.Map // key int (d) -> *densityTable

// densityTableFor returns the cached densityTable for dimension d, building
// it on first use.
func densityTableFor(d int) *densityTable {
	if cached, ok := densityTableCache.Load(d); ok {
		return cached.(*densityTable)
	}
	t := buildDensityTable(d)
	actual, _ := densityTableCache.LoadOrStore(d, t)
	return actual.(*densityTable)
}
