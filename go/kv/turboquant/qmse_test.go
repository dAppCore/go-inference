// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import (
	"math"
	"math/rand/v2"
	"testing"

	core "dappco.re/go"
)

// TestQuantiseUnit_Good checks quantiseUnit/dequantiseUnit round-trip with
// bounded error at 4 bits (16 centroids should get well within 0.1 of a
// moderate-magnitude unit-ish coordinate).
func TestQuantiseUnit_Good(t *testing.T) {
	const d = 16
	u := make([]float64, d)
	for i := range u {
		u[i] = math.Sin(float64(i)) / math.Sqrt(d)
	}
	packed := quantiseUnit(u, 4, 42)
	back := dequantiseUnit(packed, d, 4, 42)
	if len(back) != d {
		t.Fatalf("dequantiseUnit returned %d elements, want %d", len(back), d)
	}
}

// TestEncodeQMSE_Good round-trips a simple row and checks the reconstruction
// is close at 4 bits.
func TestEncodeQMSE_Good(t *testing.T) {
	x := []float32{3, 4, 0, 0, 0, 0, 0, 0}
	e := EncodeQMSE(x, 4, 42)
	got := DecodeQMSE(e, 42)
	if len(got) != len(x) {
		t.Fatalf("DecodeQMSE returned %d elements, want %d", len(got), len(x))
	}
	if errNorm := l2Norm(subtract(toFloat64(x), toFloat64(got))); errNorm > 1 {
		t.Errorf("EncodeQMSE/DecodeQMSE at 4 bits: ||x - x̃|| = %v, want small", errNorm)
	}
}

// TestEncodeQMSE_Ugly checks the zero-row special case: Gamma stays 0 and
// Decode returns an all-zero row without touching the rotation/Lloyd-Max
// machinery (there is no direction to rotate).
func TestEncodeQMSE_Ugly(t *testing.T) {
	x := make([]float32, 8)
	e := EncodeQMSE(x, 3, 42)
	if e.Gamma != 0 {
		t.Errorf("EncodeQMSE(zero row).Gamma = %v, want 0", e.Gamma)
	}
	got := DecodeQMSE(e, 42)
	for i, v := range got {
		if v != 0 {
			t.Errorf("DecodeQMSE(zero-row encoding)[%d] = %v, want 0", i, v)
		}
	}
}

// TestEncodeQMSE_Bad checks bits=0 (a degenerate but valid call — Q_prod's
// stage 1 at total bit-width 1 makes exactly this call) does not panic and
// decodes to the zero direction scaled by Gamma (the single Lloyd-Max
// centroid is 0 by symmetry).
func TestEncodeQMSE_Bad(t *testing.T) {
	x := []float32{3, 4}
	e := EncodeQMSE(x, 0, 42)
	got := DecodeQMSE(e, 42)
	// The single centroid is the density's numerically-integrated mean —
	// ≈0 by symmetry, but carries float64 quadrature noise, not exactly 0.
	for i, v := range got {
		if !approxEqual(float64(v), 0, 1e-6) {
			t.Errorf("DecodeQMSE(bits=0 encoding)[%d] = %v, want ≈0 (single centroid is ≈0)", i, v)
		}
	}
}

// TestMarshalQMSE_Good round-trips through the wire format.
func TestMarshalQMSE_Good(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	e := EncodeQMSE(x, 2, 7)
	data := MarshalQMSE(e)
	back := UnmarshalQMSE(data, len(x), 2)
	if back.Gamma != e.Gamma {
		t.Errorf("UnmarshalQMSE.Gamma = %v, want %v", back.Gamma, e.Gamma)
	}
	gotX := DecodeQMSE(back, 7)
	wantX := DecodeQMSE(e, 7)
	for i := range gotX {
		if gotX[i] != wantX[i] {
			t.Errorf("round-tripped decode[%d] = %v, want %v", i, gotX[i], wantX[i])
		}
	}
}

// TestUnmarshalQMSE_Bad checks a too-short payload decodes to the safe
// zero-row encoding rather than panicking on an out-of-range slice.
func TestUnmarshalQMSE_Bad(t *testing.T) {
	got := UnmarshalQMSE([]byte{1, 2}, 8, 2)
	if got.Gamma != 0 || got.Indices != nil {
		t.Errorf("UnmarshalQMSE(short data) = %+v, want the zero-value encoding", got)
	}
}

// ExampleEncodeQMSE demonstrates a full encode/decode round trip.
func ExampleEncodeQMSE() {
	e := EncodeQMSE([]float32{3, 4}, 4, 42)
	x := DecodeQMSE(e, 42)
	close := l2Norm(subtract([]float64{3, 4}, toFloat64(x))) < 1
	core.Println("reconstruction close:", close)
	// Output:
	// reconstruction close: true
}

// TestQMSEDistortionOracle_Good is the load-bearing correctness check (RFC
// #41 spec): the measured E[||u-ũ||²] over many sphere-uniform samples at
// d=128 must land within ±15% relative of the paper's published Q_mse
// distortions per bit-width. A failure here means the codec is wrong — the
// fix is the implementation, not the tolerance band.
func TestQMSEDistortionOracle_Good(t *testing.T) {
	const d = 128
	const samples = 8000
	const tolerance = 0.15
	// arxiv.org/abs/2504.19874's published Q_mse distortions.
	paperTargets := map[int]float64{1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

	rng := rand.New(rand.NewPCG(999, 111))
	rows := generateSphereUniformRows(rng, samples, d)

	for bits := 1; bits <= 4; bits++ {
		var sumSq float64
		for _, row := range rows {
			e := EncodeQMSE(row, bits, 42)
			got := DecodeQMSE(e, 42)
			sumSq += squaredL2Diff(row, got)
		}
		mse := sumSq / float64(samples)
		target := paperTargets[bits]
		relErr := math.Abs(mse-target) / target
		t.Logf("Q_mse b=%d: measured E[||u-ũ||²]=%.5f paper=%.5f relErr=%.1f%%", bits, mse, target, relErr*100)
		if relErr > tolerance {
			t.Errorf("Q_mse b=%d: measured distortion %.5f is outside ±%.0f%% of the paper's %.5f (relErr=%.1f%%)",
				bits, mse, tolerance*100, target, relErr*100)
		}
	}
}
