// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import (
	"math"
	"testing"
)

// TestNorm_tokenShift_Good proves the shift-by-one + prior-seed semantics: delta[0] uses `prior` (not
// zero), and delta[t>0] uses x[t-1] within the same chunk — fla's token_shift_ref.
func TestNorm_tokenShift_Good(t *testing.T) {
	const L, D = 3, 2
	x := []float32{1, 2, 10, 20, 100, 200}
	prior := []float32{-1, -2}

	delta, newPrior := tokenShift(x, prior, L, D)
	want := []float32{
		-1 - 1, -2 - 2, // delta[0] = prior - x[0]
		1 - 10, 2 - 20, // delta[1] = x[0] - x[1]
		10 - 100, 20 - 200, // delta[2] = x[1] - x[2]
	}
	for i := range want {
		if delta[i] != want[i] {
			t.Fatalf("delta[%d] = %v, want %v", i, delta[i], want[i])
		}
	}
	if newPrior[0] != 100 || newPrior[1] != 200 {
		t.Fatalf("newPrior = %v, want x's last row [100 200]", newPrior)
	}
}

// TestNorm_tokenShift_Bad proves a nil prior seeds delta[0] with the zero vector (a fresh sequence),
// distinct from the real-prior _Good case.
func TestNorm_tokenShift_Bad(t *testing.T) {
	const L, D = 2, 2
	x := []float32{5, 6, 7, 8}
	delta, _ := tokenShift(x, nil, L, D)
	if delta[0] != -5 || delta[1] != -6 {
		t.Fatalf("delta[0] = %v, want -x[0] (nil prior treated as zero)", delta[:2])
	}
}

// TestNorm_tokenShift_Ugly proves the decode-boundary invariant: one pass over a sequence and two chunks
// carrying newPrior across the split produce bit-identical delta — the carry the whole RWKV-7 model
// relies on to make chunked decode reproduce a one-pass prefill.
func TestNorm_tokenShift_Ugly(t *testing.T) {
	const L, split, D = 5, 2, 3
	x := syn(L*D, 7)

	fullDelta, _ := tokenShift(x, nil, L, D)
	d1, p1 := tokenShift(x[:split*D], nil, split, D)
	d2, _ := tokenShift(x[split*D:], p1, L-split, D)

	for i := range d1 {
		if d1[i] != fullDelta[i] {
			t.Fatalf("chunk1 delta[%d] = %v != full %v", i, d1[i], fullDelta[i])
		}
	}
	for i := range d2 {
		if d2[i] != fullDelta[split*D+i] {
			t.Fatalf("chunk2 delta[%d] = %v != full %v", i, d2[i], fullDelta[split*D+i])
		}
	}
}

func TestNorm_addcmulRows_Good(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	delta := []float32{10, 10, 10, 10}
	mix := []float32{0.5, 0.25}
	got := addcmulRows(x, delta, mix, 2, 2)
	want := []float32{1 + 5, 2 + 2.5, 3 + 5, 4 + 2.5}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("out[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestNorm_addcmulRows_Ugly proves a zero delta is the identity (out == x) — the boundary a fresh
// sequence's very first token hits when prior is also zero.
func TestNorm_addcmulRows_Ugly(t *testing.T) {
	x := syn(6, 3)
	zero := make([]float32, 6)
	mix := syn(2, 9)
	got := addcmulRows(x, zero, mix, 3, 2)
	for i := range x {
		if got[i] != x[i] {
			t.Fatalf("out[%d] = %v, want x[%d] = %v (zero delta must be identity)", i, got[i], i, x[i])
		}
	}
}

// TestNorm_layerNormRows_Good proves the output is truly normalised (zero mean, unit variance) before the
// affine transform, by using weight=1/bias=0 and reading the pre-affine statistics back out.
func TestNorm_layerNormRows_Good(t *testing.T) {
	const d = 4
	x := []float32{1, 2, 3, 4}
	w := []float32{1, 1, 1, 1}
	got := layerNormRows(x, w, nil, 1, d, 1e-5)
	var sum, ss float64
	for _, v := range got {
		sum += float64(v)
	}
	mean := sum / d
	for _, v := range got {
		ss += (float64(v) - mean) * (float64(v) - mean)
	}
	variance := ss / d
	if math.Abs(mean) > 1e-3 {
		t.Fatalf("mean = %v, want ~0", mean)
	}
	if math.Abs(variance-1) > 1e-2 {
		t.Fatalf("variance = %v, want ~1", variance)
	}
}

// TestNorm_layerNormRows_Bad proves a nil bias is treated as norm_bias=false (no shift applied) rather
// than panicking or defaulting to a non-zero shift.
func TestNorm_layerNormRows_Bad(t *testing.T) {
	const d = 3
	x := syn(d, 4)
	w := []float32{1, 1, 1}
	got := layerNormRows(x, w, nil, 1, d, 1e-5)
	withZeroBias := layerNormRows(x, w, []float32{0, 0, 0}, 1, d, 1e-5)
	for i := range got {
		if got[i] != withZeroBias[i] {
			t.Fatalf("nil-bias[%d] = %v, want == zero-bias %v", i, got[i], withZeroBias[i])
		}
	}
}

// TestNorm_layerNormRows_Ugly proves each row is normalised INDEPENDENTLY: scaling one row of a
// multi-row input must not change another row's output.
func TestNorm_layerNormRows_Ugly(t *testing.T) {
	const rows, d = 2, 4
	w := syn(d, 1)
	x := append(syn(d, 2), syn(d, 3)...)
	base := layerNormRows(x, w, nil, rows, d, 1e-5)

	x2 := append([]float32(nil), x...)
	for i := range d {
		x2[i] *= 100 // perturb row 0 only
	}
	got := layerNormRows(x2, w, nil, rows, d, 1e-5)
	for i := range d {
		if got[d+i] != base[d+i] {
			t.Fatalf("row 1 changed when only row 0 was perturbed: [%d] = %v, want %v", i, got[d+i], base[d+i])
		}
	}
}

// TestNorm_groupNormHeads_Good proves each head is normalised independently over exactly its own V
// channels — perturbing head 1's input must not change head 0's output within the same row.
func TestNorm_groupNormHeads_Good(t *testing.T) {
	const rows, H, V = 1, 2, 3
	w := syn(H*V, 1)
	b := syn(H*V, 2)
	x := syn(H*V, 3)
	base := groupNormHeads(x, w, b, rows, H, V, 1e-5)

	x2 := append([]float32(nil), x...)
	for i := V; i < 2*V; i++ {
		x2[i] *= 50 // perturb head 1 only
	}
	got := groupNormHeads(x2, w, b, rows, H, V, 1e-5)
	for i := range V {
		if got[i] != base[i] {
			t.Fatalf("head 0 changed when only head 1 was perturbed: [%d] = %v, want %v", i, got[i], base[i])
		}
	}
}

// TestNorm_groupNormHeads_Bad proves a nil bias omits the shift (elementwise_affine's bias half), the
// same contract as layerNormRows.
func TestNorm_groupNormHeads_Bad(t *testing.T) {
	const rows, H, V = 1, 2, 3
	w := syn(H*V, 1)
	x := syn(H*V, 3)
	got := groupNormHeads(x, w, nil, rows, H, V, 1e-5)
	withZeroBias := groupNormHeads(x, w, make([]float32, H*V), rows, H, V, 1e-5)
	for i := range got {
		if got[i] != withZeroBias[i] {
			t.Fatalf("nil-bias[%d] = %v, want == zero-bias %v", i, got[i], withZeroBias[i])
		}
	}
}

// TestNorm_groupNormHeads_Ugly proves a constant row within a head normalises to exactly zero (the
// eps-guarded zero-variance edge — GroupNorm must not divide by zero or blow up).
func TestNorm_groupNormHeads_Ugly(t *testing.T) {
	const rows, H, V = 1, 1, 4
	x := []float32{5, 5, 5, 5}
	w := []float32{1, 1, 1, 1}
	got := groupNormHeads(x, w, nil, rows, H, V, 1e-5)
	for i, v := range got {
		if math.Abs(float64(v)) > 1e-2 {
			t.Fatalf("constant-input out[%d] = %v, want ~0", i, v)
		}
	}
}

func TestNorm_sigmoidF32_Good(t *testing.T) {
	if v := sigmoidF32(0); math.Abs(float64(v)-0.5) > 1e-6 {
		t.Fatalf("sigmoid(0) = %v, want 0.5", v)
	}
}

// TestNorm_sigmoidF32_Ugly proves saturation at large-magnitude inputs stays within (0,1) rather than
// overflowing/NaN-ing.
func TestNorm_sigmoidF32_Ugly(t *testing.T) {
	hi, lo := sigmoidF32(80), sigmoidF32(-80)
	if hi <= 0.999999 || hi > 1 {
		t.Fatalf("sigmoid(80) = %v, want ~1", hi)
	}
	if lo < 0 || lo >= 0.000001 {
		t.Fatalf("sigmoid(-80) = %v, want ~0", lo)
	}
}

func TestNorm_tanhF32_Good(t *testing.T) {
	if v := tanhF32(0); v != 0 {
		t.Fatalf("tanh(0) = %v, want 0", v)
	}
}

// TestNorm_tanhF32_Ugly proves saturation stays within (-1,1) at large magnitude, symmetric about 0.
func TestNorm_tanhF32_Ugly(t *testing.T) {
	hi, lo := tanhF32(50), tanhF32(-50)
	if hi <= 0.999999 || hi > 1 {
		t.Fatalf("tanh(50) = %v, want ~1", hi)
	}
	if lo != -hi {
		t.Fatalf("tanh(-50) = %v, want -tanh(50) = %v (odd function)", lo, -hi)
	}
}
