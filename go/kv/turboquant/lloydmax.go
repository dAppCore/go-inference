// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import (
	"math"
	"sync"
)

const (
	lloydMaxMaxIters = 500
	lloydMaxTol      = 1e-13
)

// lloydMax solves for the levels centroids that minimise expected squared
// error against the dimension-d sphere-marginal density — the fixed-point
// iteration: recompute each centroid as the conditional mean of its cell,
// recompute cell boundaries as the midpoints between neighbouring centroids,
// repeat to convergence. Returns centroids sorted ascending.
//
// Initialised at the density's quantiles (table.quantile), which starts
// Lloyd-Max already sorted and cell-balanced, so it converges in a handful
// of iterations rather than risking an empty cell from a naive linspace
// start.
//
//	tbl := densityTableFor(128)
//	c := lloydMax(tbl, 4) // 4 centroids for a 2-bit quantiser
func lloydMax(table *densityTable, levels int) []float64 {
	if levels <= 0 {
		return nil
	}
	centroids := make([]float64, levels)
	for i := range centroids {
		p := (float64(i) + 0.5) / float64(levels)
		centroids[i] = table.quantile(p)
	}
	if levels == 1 {
		// A single level's boundary is the whole domain; its centroid is
		// the density's overall mean, which is 0 by symmetry. Skip the
		// iteration below (boundaries has no interior midpoint to compute).
		_, moment := table.integrate(-1, 1)
		centroids[0] = moment
		return centroids
	}

	boundaries := make([]float64, levels+1)
	boundaries[0], boundaries[levels] = -1, 1
	for iter := 0; iter < lloydMaxMaxIters; iter++ {
		for i := 1; i < levels; i++ {
			boundaries[i] = (centroids[i-1] + centroids[i]) / 2
		}
		maxDelta := 0.0
		for i := 0; i < levels; i++ {
			mass, moment := table.integrate(boundaries[i], boundaries[i+1])
			next := centroids[i]
			if mass > 0 {
				next = moment / mass
			}
			if delta := math.Abs(next - centroids[i]); delta > maxDelta {
				maxDelta = delta
			}
			centroids[i] = next
		}
		if maxDelta < lloydMaxTol {
			break
		}
	}
	return centroids
}

// nearestCentroid returns the index of the centroid closest to y — a linear
// scan, which is fine at the level counts this package uses (at most 16, for
// a 4-bit quantiser).
//
//	nearestCentroid(0.9, []float64{-0.5, 0.5}) // 1
func nearestCentroid(y float64, centroids []float64) int {
	best := 0
	bestDist := math.Abs(y - centroids[0])
	for i := 1; i < len(centroids); i++ {
		if d := math.Abs(y - centroids[i]); d < bestDist {
			bestDist = d
			best = i
		}
	}
	return best
}

var lloydMaxCache sync.Map // key lloydMaxCacheKey -> []float64

type lloydMaxCacheKey struct {
	d, bits int
}

// centroidsFor returns the cached Lloyd-Max centroids for (d, bits) —
// 2^bits centroids solved against dimension d's sphere-marginal density,
// computed once and reused by every row TurboQuant quantises at that (d,
// bits) pair. bits == 0 yields the single centroid {0} (Q_prod's degenerate
// stage-1 case at total bit-width 1).
//
//	c := centroidsFor(128, 2) // 4 centroids
func centroidsFor(d, bits int) []float64 {
	key := lloydMaxCacheKey{d: d, bits: bits}
	if cached, ok := lloydMaxCache.Load(key); ok {
		return cached.([]float64)
	}
	levels := 1 << uint(bits)
	c := lloydMax(densityTableFor(d), levels)
	actual, _ := lloydMaxCache.LoadOrStore(key, c)
	return actual.([]float64)
}
