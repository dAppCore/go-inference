// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import (
	"math"
	"testing"
)

// TestMatrixAtSet_Good checks the row-major indexing contract directly.
func TestMatrixAtSet_Good(t *testing.T) {
	m := newMatrix(3)
	m.set(1, 2, 7)
	if got := m.at(1, 2); got != 7 {
		t.Errorf("at(1,2) = %v, want 7", got)
	}
	if got := m.at(2, 1); got != 0 {
		t.Errorf("at(2,1) = %v, want 0 (unset)", got)
	}
}

// TestIdentityMatrix_Good checks the identity acts as a no-op under mulVec
// and mulVecT.
func TestIdentityMatrix_Good(t *testing.T) {
	m := identityMatrix(3)
	v := []float64{1, -2, 3.5}
	if got := m.mulVec(v); got[0] != v[0] || got[1] != v[1] || got[2] != v[2] {
		t.Errorf("identity.mulVec(%v) = %v, want %v", v, got, v)
	}
	if got := m.mulVecT(v); got[0] != v[0] || got[1] != v[1] || got[2] != v[2] {
		t.Errorf("identity.mulVecT(%v) = %v, want %v", v, got, v)
	}
}

// TestMulVec_Good checks a hand-computed 2×2 case.
func TestMulVec_Good(t *testing.T) {
	m := newMatrix(2)
	m.set(0, 0, 1)
	m.set(0, 1, 2)
	m.set(1, 0, 3)
	m.set(1, 1, 4)
	got := m.mulVec([]float64{5, 6})
	want := []float64{1*5 + 2*6, 3*5 + 4*6}
	if got[0] != want[0] || got[1] != want[1] {
		t.Errorf("mulVec = %v, want %v", got, want)
	}
}

// TestMulVecT_Good checks mulVecT computes the transpose product, distinct
// from mulVec for a non-symmetric matrix.
func TestMulVecT_Good(t *testing.T) {
	m := newMatrix(2)
	m.set(0, 0, 1)
	m.set(0, 1, 2)
	m.set(1, 0, 3)
	m.set(1, 1, 4)
	got := m.mulVecT([]float64{5, 6})
	want := []float64{1*5 + 3*6, 2*5 + 4*6}
	if got[0] != want[0] || got[1] != want[1] {
		t.Errorf("mulVecT = %v, want %v", got, want)
	}
}

// TestSplitmix64_Good checks determinism (same input, same output) and that
// distinct inputs are extremely unlikely to collide across a small sample.
func TestSplitmix64_Good(t *testing.T) {
	if splitmix64(42) != splitmix64(42) {
		t.Fatal("splitmix64(42) is not deterministic")
	}
	seen := map[uint64]bool{}
	for i := uint64(0); i < 1000; i++ {
		h := splitmix64(i)
		if seen[h] {
			t.Fatalf("splitmix64 collided within the first 1000 inputs at %d", i)
		}
		seen[h] = true
	}
}

// TestDeriveSeed_Good checks distinct salts decorrelate the same base seed.
func TestDeriveSeed_Good(t *testing.T) {
	a := deriveSeed(42, seedPurposeRotation)
	b := deriveSeed(42, seedPurposeQJL)
	if a == b {
		t.Fatal("deriveSeed(42, rotation) == deriveSeed(42, qjl), want distinct streams")
	}
	if deriveSeed(42, seedPurposeRotation) != a {
		t.Fatal("deriveSeed is not deterministic")
	}
}

// TestGaussianMatrix_Good checks determinism and that entries are not
// degenerate (not all zero/identical — a broken RNG wiring symptom).
func TestGaussianMatrix_Good(t *testing.T) {
	a := gaussianMatrix(42, 8)
	b := gaussianMatrix(42, 8)
	for i := range a.data {
		if a.data[i] != b.data[i] {
			t.Fatalf("gaussianMatrix(42,8) is not deterministic at index %d: %v vs %v", i, a.data[i], b.data[i])
		}
	}
	distinct := map[float64]bool{}
	for _, v := range a.data {
		distinct[v] = true
	}
	if len(distinct) < len(a.data)/2 {
		t.Errorf("gaussianMatrix looks degenerate: only %d distinct values among %d entries", len(distinct), len(a.data))
	}
}

// TestHouseholderQR_Good checks Q is orthogonal (QᵀQ = I, to float64
// tolerance) across a spread of dimensions — the property both TurboQuant
// codecs depend on for norm preservation.
func TestHouseholderQR_Good(t *testing.T) {
	for _, d := range []int{2, 3, 4, 8, 16, 64} {
		a := gaussianMatrix(uint64(d)*97+1, d)
		q := householderQR(a)
		for i := 0; i < d; i++ {
			for j := 0; j < d; j++ {
				var dot float64
				for k := 0; k < d; k++ {
					dot += q.at(k, i) * q.at(k, j)
				}
				want := 0.0
				if i == j {
					want = 1.0
				}
				if !approxEqual(dot, want, 1e-8) {
					t.Errorf("d=%d: <Q col %d, Q col %d> = %v, want %v", d, i, j, dot, want)
				}
			}
		}
	}
}

// TestHouseholderQR_Ugly checks the degenerate d==1 rotation (via
// rotationFor, which special-cases d<=1 rather than calling householderQR)
// behaves as the identity — there is nothing to rotate in one dimension.
func TestHouseholderQR_Ugly(t *testing.T) {
	pi := rotationFor(1, 1)
	got := pi.mulVec([]float64{3.5})
	if len(got) != 1 || got[0] != 3.5 {
		t.Errorf("rotationFor(_, 1).mulVec({3.5}) = %v, want {3.5}", got)
	}
}

// TestRotation_PreservesNorm_Good checks ||Π·v|| == ||v|| for a rotation
// built via the real rotationFor path — the property Q_mse's normalise-
// then-rotate step depends on.
func TestRotation_PreservesNorm_Good(t *testing.T) {
	const d = 32
	pi := rotationFor(123, d)
	v := make([]float64, d)
	for i := range v {
		v[i] = float64(i%7) - 3
	}
	y := pi.mulVec(v)
	if !approxEqual(l2Norm(y), l2Norm(v), 1e-8) {
		t.Errorf("||Π·v|| = %v, ||v|| = %v, want equal", l2Norm(y), l2Norm(v))
	}
}

// TestRotation_MulVecT_Inverts_Good checks Πᵀ·(Π·v) recovers v — the
// un-rotate step DecodeQMSE depends on.
func TestRotation_MulVecT_Inverts_Good(t *testing.T) {
	const d = 16
	pi := rotationFor(456, d)
	v := make([]float64, d)
	for i := range v {
		v[i] = math.Sin(float64(i))
	}
	back := pi.mulVecT(pi.mulVec(v))
	for i := range v {
		if !approxEqual(back[i], v[i], 1e-8) {
			t.Errorf("Πᵀ·Π·v[%d] = %v, want %v", i, back[i], v[i])
		}
	}
}

// TestRotationFor_Good checks caching returns bit-identical matrices for
// repeated calls at the same (seed, d), and distinct matrices for distinct
// seeds.
func TestRotationFor_Good(t *testing.T) {
	a := rotationFor(1, 8)
	b := rotationFor(1, 8)
	if a != b {
		t.Error("rotationFor(1,8) called twice returned different pointers, want the cached matrix")
	}
	c := rotationFor(2, 8)
	same := true
	for i := range a.data {
		if a.data[i] != c.data[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("rotationFor(1,8) and rotationFor(2,8) produced identical matrices, want distinct seeds to decorrelate")
	}
}

// TestQJLMatrixFor_Good checks qjlMatrixFor is cached, deterministic, and
// distinct from rotationFor at the same seed (the two streams must not
// collide — Q_prod's stage 2 would otherwise correlate with stage 1's
// rotation).
func TestQJLMatrixFor_Good(t *testing.T) {
	a := qjlMatrixFor(42, 8)
	b := qjlMatrixFor(42, 8)
	if a != b {
		t.Error("qjlMatrixFor(42,8) called twice returned different pointers, want the cached matrix")
	}
	rot := rotationFor(42, 8)
	same := true
	for i := range a.data {
		if a.data[i] != rot.data[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("qjlMatrixFor and rotationFor produced identical matrices at the same seed, want decorrelated streams")
	}
}
