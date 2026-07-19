// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import (
	"math"
	"math/rand/v2"
	"testing"

	core "dappco.re/go"
)

// TestEncodeQProd_Good round-trips a simple row and checks the
// reconstruction is close at 4 total bits.
func TestEncodeQProd_Good(t *testing.T) {
	x := []float32{3, 4, 0, 0, 0, 0, 0, 0}
	e := EncodeQProd(x, 4, 42)
	got := DecodeQProd(e, 42)
	if len(got) != len(x) {
		t.Fatalf("DecodeQProd returned %d elements, want %d", len(got), len(x))
	}
	if errNorm := l2Norm(subtract(toFloat64(x), toFloat64(got))); errNorm > 1 {
		t.Errorf("EncodeQProd/DecodeQProd at 4 bits: ||x - x̃|| = %v, want small", errNorm)
	}
}

// TestEncodeQProd_Ugly checks the zero-row special case short-circuits both
// stages.
func TestEncodeQProd_Ugly(t *testing.T) {
	x := make([]float32, 8)
	e := EncodeQProd(x, 3, 42)
	if e.Gamma != 0 || e.Rho != 0 {
		t.Errorf("EncodeQProd(zero row) = %+v, want Gamma=0 and Rho=0", e)
	}
	got := DecodeQProd(e, 42)
	for i, v := range got {
		if v != 0 {
			t.Errorf("DecodeQProd(zero-row encoding)[%d] = %v, want 0", i, v)
		}
	}
}

// TestEncodeQProd_Bad checks totalBits=1 — stage 1 degenerates to 0 bits (a
// single always-0 centroid), so the entire reconstruction comes from the
// QJL sign sketch alone. Must not panic, and Stage1Indices must be empty.
func TestEncodeQProd_Bad(t *testing.T) {
	x := []float32{3, 4, 1, -2}
	e := EncodeQProd(x, 1, 42)
	if len(e.Stage1Indices) != 0 {
		t.Errorf("EncodeQProd(totalBits=1).Stage1Indices = %v, want empty (0-bit stage 1)", e.Stage1Indices)
	}
	got := DecodeQProd(e, 42)
	if len(got) != len(x) {
		t.Fatalf("DecodeQProd returned %d elements, want %d", len(got), len(x))
	}
}

// TestMarshalQProd_Good round-trips through the wire format.
func TestMarshalQProd_Good(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	e := EncodeQProd(x, 3, 7)
	data := MarshalQProd(e)
	back := UnmarshalQProd(data, len(x), 3)
	if back.Gamma != e.Gamma || back.Rho != e.Rho {
		t.Errorf("UnmarshalQProd Gamma/Rho = %v/%v, want %v/%v", back.Gamma, back.Rho, e.Gamma, e.Rho)
	}
	gotX := DecodeQProd(back, 7)
	wantX := DecodeQProd(e, 7)
	for i := range gotX {
		if gotX[i] != wantX[i] {
			t.Errorf("round-tripped decode[%d] = %v, want %v", i, gotX[i], wantX[i])
		}
	}
}

// TestUnmarshalQProd_Bad checks a too-short payload decodes to the safe
// zero-value encoding rather than panicking.
func TestUnmarshalQProd_Bad(t *testing.T) {
	got := UnmarshalQProd([]byte{1, 2}, 8, 3)
	if got.Gamma != 0 || got.Stage1Indices != nil {
		t.Errorf("UnmarshalQProd(short data) = %+v, want the zero-value encoding", got)
	}
}

// ExampleEncodeQProd demonstrates a full encode/decode round trip.
func ExampleEncodeQProd() {
	e := EncodeQProd([]float32{3, 4}, 4, 42)
	x := DecodeQProd(e, 42)
	close := l2Norm(subtract([]float64{3, 4}, toFloat64(x))) < 1
	core.Println("reconstruction close:", close)
	// Output:
	// reconstruction close: true
}

// TestQProdOracle_Good is the load-bearing correctness check (RFC #41
// spec): over many random (x, y) unit-vector pairs at d=128,
//  1. unbiasedness — mean(<y,x̃> - <y,x>) must be small relative to the
//     spread of that delta (a genuine bias would show up as a mean many
//     standard deviations from 0; a "small relative to RMS" check is
//     equivalent and avoids a flaky exact-CI computation while still being
//     a real assertion, not a vacuous one), and
//  2. the inner-product MSE must land within ±15% relative of the paper's
//     published Q_prod targets (stage-1 at b-1 bits + 1 QJL bit).
//
// A failure here means the codec is wrong — the fix is the implementation,
// not the tolerance band.
func TestQProdOracle_Good(t *testing.T) {
	const d = 128
	const samples = 8000
	const tolerance = 0.15
	// arxiv.org/abs/2504.19874's published Q_prod inner-product MSE targets,
	// as a MSE·d constant (the paper reports MSE ≈ constant/d).
	paperTargets := map[int]float64{1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}

	rng := rand.New(rand.NewPCG(4242, 24))
	xs := generateSphereUniformRows(rng, samples, d)
	ys := generateSphereUniformRows(rng, samples, d)

	for bits := 1; bits <= 4; bits++ {
		var sumDelta, sumDeltaSq float64
		for i := 0; i < samples; i++ {
			e := EncodeQProd(xs[i], bits, 42)
			xHat := DecodeQProd(e, 42)
			exact := dot(toFloat64(ys[i]), toFloat64(xs[i]))
			cand := dot(toFloat64(ys[i]), toFloat64(xHat))
			delta := cand - exact
			sumDelta += delta
			sumDeltaSq += delta * delta
		}
		meanDelta := sumDelta / float64(samples)
		mse := sumDeltaSq / float64(samples)
		rmse := math.Sqrt(mse)
		target := paperTargets[bits] / float64(d)
		relErr := math.Abs(mse-target) / target
		biasRatio := math.Abs(meanDelta) / rmse

		t.Logf("Q_prod b=%d: mean(delta)=%.6f rmse=%.6f |bias|/rmse=%.3f mse=%.6f paper/d=%.6f relErr=%.1f%%",
			bits, meanDelta, rmse, biasRatio, mse, target, relErr*100)

		if relErr > tolerance {
			t.Errorf("Q_prod b=%d: measured inner-product MSE %.6f is outside ±%.0f%% of the paper's %.6f (relErr=%.1f%%)",
				bits, mse, tolerance*100, target, relErr*100)
		}
		if biasRatio > 0.15 {
			t.Errorf("Q_prod b=%d: mean(delta)=%.6f is %.3fx rmse, want << 1 (statistically ~0, i.e. unbiased)",
				bits, meanDelta, biasRatio)
		}
	}
}
