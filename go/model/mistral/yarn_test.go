// SPDX-Licence-Identifier: EUPL-1.2

package mistral_test

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/mistral"
)

func relDiff(a, b float32) float64 {
	return math.Abs(float64(a)-float64(b)) / (math.Abs(float64(b)) + 1e-12)
}

// plainRope is the standard RoPE inverse frequency base^(-2i/dim).
func plainRope(base float64, i, dim int) float64 {
	return math.Pow(base, -float64(2*i)/float64(dim))
}

// TestYaRNInvFreqs_FactorOne_PlainRope_Good proves YaRN with no context extension
// (factor 1) is exactly standard RoPE — the no-op identity.
func TestYaRNInvFreqs_FactorOne_PlainRope_Good(t *testing.T) {
	const base, dim = 1e6, 128
	got := mistral.YaRNInvFreqs(base, 1, 32, 1, 16384, dim)
	if len(got) != dim/2 {
		t.Fatalf("len %d, want %d", len(got), dim/2)
	}
	for i := range got {
		want := float32(plainRope(base, i, dim))
		if relDiff(got[i], want) > 1e-5 {
			t.Fatalf("factor=1 freq[%d]=%g, want plain rope %g", i, got[i], want)
		}
	}
}

// TestYaRNInvFreqs_Ministral_Good pins the NTK-by-parts split against the real
// Ministral-3 params: high-frequency dims extrapolate (== plain rope), low-
// frequency dims interpolate (== plain rope / factor), every dim stays within
// [interpolated, extrapolated], and the sequence is monotonically non-increasing.
func TestYaRNInvFreqs_Ministral_Good(t *testing.T) {
	const base, factor, dim = 1e6, 16.0, 128
	const betaFast, betaSlow, origMax = 32.0, 1.0, 16384
	got := mistral.YaRNInvFreqs(base, factor, betaFast, betaSlow, origMax, dim)

	// high-frequency dims (well below the ramp) extrapolate → plain rope.
	for _, i := range []int{0, 5, 15} {
		want := float32(plainRope(base, i, dim))
		if relDiff(got[i], want) > 1e-5 {
			t.Errorf("extrapolate dim %d: %g, want plain rope %g", i, got[i], want)
		}
	}
	// low-frequency dims (well above the ramp) interpolate → plain rope / factor.
	for _, i := range []int{45, 55, 63} {
		want := float32(plainRope(base, i, dim) / factor)
		if relDiff(got[i], want) > 1e-5 {
			t.Errorf("interpolate dim %d: %g, want plain/factor %g", i, got[i], want)
		}
	}
	// every dim sits within [plain/factor, plain] and the sequence never rises.
	for i := range got {
		extra := plainRope(base, i, dim)
		inter := extra / factor
		if float64(got[i]) > extra*(1+1e-5) || float64(got[i]) < inter*(1-1e-5) {
			t.Errorf("dim %d freq %g outside [inter %g, extra %g]", i, got[i], inter, extra)
		}
		if i > 0 && got[i] > got[i-1]*(1+1e-5) {
			t.Errorf("freqs not monotonically non-increasing at dim %d: %g > %g", i, got[i], got[i-1])
		}
	}
}

// TestYaRNInvFreqs_RampBetween_Good proves the transition dims are a genuine blend
// — strictly below plain rope (interpolated down) yet strictly above the fully-
// interpolated value — so the ramp actually ramps rather than stepping.
func TestYaRNInvFreqs_RampBetween_Good(t *testing.T) {
	const base, factor, dim = 1e6, 16.0, 128
	got := mistral.YaRNInvFreqs(base, factor, 32, 1, 16384, dim)
	rampSeen := false
	for i := range got {
		extra := plainRope(base, i, dim)
		inter := extra / factor
		if float64(got[i]) < extra*(1-1e-4) && float64(got[i]) > inter*(1+1e-4) {
			rampSeen = true // a dim genuinely between the two — the ramp
		}
	}
	if !rampSeen {
		t.Fatal("no ramp dims found — YaRN degenerated to a hard extra/inter step")
	}
}
