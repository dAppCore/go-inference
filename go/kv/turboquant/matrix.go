// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import (
	"math"
	"math/rand/v2"
	"sync"
)

// matrix is a dense d×d float64 matrix in row-major storage, shared by the
// TurboQuant rotation (Π, orthogonal) and the QJL projection (S, plain
// i.i.d. Gaussian — never orthogonalised). float64 throughout, per the
// package's float64-accumulation rule.
type matrix struct {
	d    int
	data []float64 // data[i*d+j] = M[i][j]
}

// newMatrix allocates a zeroed d×d matrix.
func newMatrix(d int) *matrix {
	return &matrix{d: d, data: make([]float64, d*d)}
}

func (m *matrix) at(i, j int) float64     { return m.data[i*m.d+j] }
func (m *matrix) set(i, j int, v float64) { m.data[i*m.d+j] = v }

// mulVec returns M·v.
//
//	m := identityMatrix(2)
//	m.mulVec([]float64{3, 4}) // []float64{3, 4}
func (m *matrix) mulVec(v []float64) []float64 {
	out := make([]float64, m.d)
	for i := 0; i < m.d; i++ {
		var sum float64
		row := m.data[i*m.d : i*m.d+m.d]
		for j, vj := range v {
			sum += row[j] * vj
		}
		out[i] = sum
	}
	return out
}

// mulVecT returns Mᵀ·v — the transpose applied without materialising Mᵀ.
// Used to un-rotate (Πᵀ) and to reconstruct the QJL residual (Sᵀ·q).
//
//	m := identityMatrix(2)
//	m.mulVecT([]float64{3, 4}) // []float64{3, 4}
func (m *matrix) mulVecT(v []float64) []float64 {
	out := make([]float64, m.d)
	for j := 0; j < m.d; j++ {
		var sum float64
		for i, vi := range v {
			sum += m.data[i*m.d+j] * vi
		}
		out[j] = sum
	}
	return out
}

// identityMatrix returns the d×d identity — used as the QR accumulator seed
// and directly as the degenerate d==1 rotation.
func identityMatrix(d int) *matrix {
	m := newMatrix(d)
	for i := 0; i < d; i++ {
		m.set(i, i, 1)
	}
	return m
}

// splitmix64 is a fast, well-distributed integer hash used to derive
// decorrelated sub-seeds from one caller-supplied seed (Π's rotation stream
// vs S's QJL projection stream must not share state). Public domain
// algorithm (Vigna).
func splitmix64(x uint64) uint64 {
	x += 0x9E3779B97F4A7C15
	z := x
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// deriveSeed mixes seed with a purpose salt so one caller-facing seed can
// drive several statistically-independent random streams (rotation matrix,
// QJL matrix, calibration sampling) without the caller juggling several
// seeds.
//
//	deriveSeed(42, seedPurposeRotation) != deriveSeed(42, seedPurposeQJL)
func deriveSeed(seed uint64, salt uint64) uint64 {
	return splitmix64(seed ^ splitmix64(salt))
}

const (
	seedPurposeRotation uint64 = 1 // Π, the TurboQuant orthogonal rotation
	seedPurposeQJL      uint64 = 2 // S, the QJL sign-projection matrix
)

// gaussianMatrix fills a d×d matrix with i.i.d. N(0,1) entries from a
// splitmix64-seeded PCG stream — deterministic given seed.
func gaussianMatrix(seed uint64, d int) *matrix {
	src := rand.NewPCG(seed, splitmix64(seed))
	r := rand.New(src)
	m := newMatrix(d)
	for i := range m.data {
		m.data[i] = r.NormFloat64()
	}
	return m
}

// householderQR factorises a into Q·R via Householder reflections (float64),
// returning Q. R is discarded — callers only ever want the orthogonal
// factor. a is consumed as scratch (copied internally; the caller's matrix
// is untouched).
//
// Q alone, from an unsigned Householder QR, is orthogonal but NOT
// Haar-distributed over O(d) — the sign of each R diagonal entry biases the
// distribution (Mezzadri 2007, "How to generate random matrices from the
// classical compact groups"). We correct it: flip the sign of Q's column i
// wherever R's diagonal entry i is negative, which is equivalent to
// requiring R to have a non-negative diagonal. That correction is what makes
// the rotation below a genuine Haar-uniform sample, which matters when Π is
// applied to a REAL (non-isotropic) KV row — an uncorrected Q can leave
// systematic structure in y that an isotropic-only synthetic test would
// never surface.
func householderQR(a *matrix) *matrix {
	d := a.d
	r := newMatrix(d)
	copy(r.data, a.data)
	q := identityMatrix(d)

	col := make([]float64, d)
	v := make([]float64, d)
	for k := 0; k < d-1; k++ {
		// Householder vector for column k, rows k..d-1.
		var normSq float64
		for i := k; i < d; i++ {
			col[i] = r.at(i, k)
			normSq += col[i] * col[i]
		}
		norm := math.Sqrt(normSq)
		if norm == 0 {
			continue
		}
		alpha := -norm
		if col[k] < 0 {
			alpha = norm
		}
		for i := k; i < d; i++ {
			v[i] = col[i]
		}
		v[k] -= alpha
		var vNormSq float64
		for i := k; i < d; i++ {
			vNormSq += v[i] * v[i]
		}
		if vNormSq == 0 {
			continue
		}

		// R := H_k · R, restricted to rows/cols >= k (H_k is identity
		// elsewhere).
		for j := k; j < d; j++ {
			var dot float64
			for i := k; i < d; i++ {
				dot += v[i] * r.at(i, j)
			}
			f := 2 * dot / vNormSq
			for i := k; i < d; i++ {
				r.set(i, j, r.at(i, j)-f*v[i])
			}
		}
		// Q := Q · H_k (H_k symmetric, so this accumulates the product that
		// makes A = Q·R hold).
		for i := 0; i < d; i++ {
			var dot float64
			for j := k; j < d; j++ {
				dot += q.at(i, j) * v[j]
			}
			f := 2 * dot / vNormSq
			for j := k; j < d; j++ {
				q.set(i, j, q.at(i, j)-f*v[j])
			}
		}
	}

	// Mezzadri sign correction: flip Q's column i if R's diagonal entry i is
	// negative, so R would have a non-negative diagonal.
	for i := 0; i < d; i++ {
		if r.at(i, i) < 0 {
			for row := 0; row < d; row++ {
				q.set(row, i, -q.at(row, i))
			}
		}
	}
	return q
}

var (
	rotationCache sync.Map // key rotationCacheKey -> *matrix
	gaussianCache sync.Map // key rotationCacheKey -> *matrix
)

type rotationCacheKey struct {
	seed uint64
	d    int
}

// rotationFor returns the deterministic random orthogonal d×d matrix Π for
// seed, generated once (Householder QR of a seeded Gaussian matrix, Haar
// sign-corrected) and cached thereafter — every row TurboQuant encodes at
// this (seed, d) reuses the same Π, exactly as the algorithm requires.
//
//	pi := rotationFor(42, 128)
//	y := pi.mulVec(u)
func rotationFor(seed uint64, d int) *matrix {
	key := rotationCacheKey{seed: deriveSeed(seed, seedPurposeRotation), d: d}
	if cached, ok := rotationCache.Load(key); ok {
		return cached.(*matrix)
	}
	var m *matrix
	if d <= 1 {
		m = identityMatrix(max(d, 0))
	} else {
		m = householderQR(gaussianMatrix(key.seed, d))
	}
	actual, _ := rotationCache.LoadOrStore(key, m)
	return actual.(*matrix)
}

// qjlMatrixFor returns the deterministic random i.i.d. N(0,1) d×d matrix S
// for seed (QJL stage of Q_prod) — generated once and cached, distinct from
// and uncorrelated with rotationFor's Π at the same seed.
//
//	s := qjlMatrixFor(42, 128)
//	sr := s.mulVec(residual)
func qjlMatrixFor(seed uint64, d int) *matrix {
	key := rotationCacheKey{seed: deriveSeed(seed, seedPurposeQJL), d: d}
	if cached, ok := gaussianCache.Load(key); ok {
		return cached.(*matrix)
	}
	m := gaussianMatrix(key.seed, max(d, 0))
	actual, _ := gaussianCache.LoadOrStore(key, m)
	return actual.(*matrix)
}
