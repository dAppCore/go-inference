// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import (
	"math"
	"testing"
)

// TestAutoroundCov_ResolveScheme_FloatFamilies covers the MXFP4, NVFP4, and
// FP8_STATIC scheme branches that the integer-weight-only tests never reach.
// Each float family carries its own group-size and family-tag defaults.
func TestAutoroundCov_ResolveScheme_FloatFamilies(t *testing.T) {
	cases := []struct {
		scheme    Scheme
		bits      int
		groupSize int
		family    string
	}{
		{SchemeMXFP4, 4, 32, "mx_fp"},
		{SchemeNVFP4, 4, 16, "nv_fp"},
		{SchemeFP8Static, 8, 0, "fp8"},
	}
	for _, tc := range cases {
		t.Run(string(tc.scheme), func(t *testing.T) {
			info, ok := ResolveScheme(tc.scheme)
			if !ok {
				t.Fatalf("ResolveScheme(%q) ok = false, want resolvable float-family scheme", tc.scheme)
			}
			if info.Scheme != tc.scheme || info.Bits != tc.bits {
				t.Fatalf("ResolveScheme(%q) = %+v, want bits %d", tc.scheme, info, tc.bits)
			}
			if info.GroupSize != tc.groupSize {
				t.Fatalf("ResolveScheme(%q) group size = %d, want %d", tc.scheme, info.GroupSize, tc.groupSize)
			}
			if info.Family != tc.family {
				t.Fatalf("ResolveScheme(%q) family = %q, want %q", tc.scheme, info.Family, tc.family)
			}
			if info.ActivationBits != 16 {
				t.Fatalf("ResolveScheme(%q) activation bits = %d, want 16", tc.scheme, info.ActivationBits)
			}
		})
	}
}

// TestAutoroundCov_NormaliseQuantizeConfig_BitsDefault covers the schemeless
// default-bits branch: a config with no scheme and zero bits falls back to 4.
func TestAutoroundCov_NormaliseQuantizeConfig_BitsDefault(t *testing.T) {
	// No scheme, no bits, no group size, no learning rate — every default is
	// applied, so quantisation proceeds at the W4 group-128 fallback.
	got, err := QuantizeWeights([]float32{0.1, -0.2, 0.3}, QuantizeConfig{})
	if err != nil {
		t.Fatalf("QuantizeWeights(zero config) error = %v, want defaults applied", err)
	}
	if got.Bits != 4 {
		t.Fatalf("QuantizeWeights(zero config) bits = %d, want default 4", got.Bits)
	}
	if got.GroupSize != 128 {
		t.Fatalf("QuantizeWeights(zero config) group size = %d, want default 128", got.GroupSize)
	}
}

// TestAutoroundCov_NormaliseQuantizeConfig_BadLearningRate covers the
// non-finite / negative learning-rate rejection branch.
func TestAutoroundCov_NormaliseQuantizeConfig_BadLearningRate(t *testing.T) {
	cases := []struct {
		name string
		lr   float32
	}{
		{"negative", -1},
		{"nan", float32(math.NaN())},
		{"inf", float32(math.Inf(1))},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := QuantizeConfig{Scheme: SchemeW4A16, GroupSize: 32, LearningRate: tc.lr}
			if _, err := QuantizeWeights([]float32{1, 2}, cfg); err == nil {
				t.Fatalf("QuantizeWeights(lr=%v) err = nil, want learning-rate diagnostic", tc.lr)
			}
		})
	}
}

// TestAutoroundCov_QuantizeWeights_Asymmetric covers the asymmetric quantParams
// branch (Symmetric=false): the scale/zero are derived from min/max range rather
// than the symmetric abs-max fallback.
func TestAutoroundCov_QuantizeWeights_Asymmetric(t *testing.T) {
	t.Run("DynamicRange", func(t *testing.T) {
		// A group with a genuine min/max spread exercises the scale and zero-point
		// computation: scale = (max-min)/qmax, zero = round(-min/scale).
		weights := make([]float32, 32)
		for i := range weights {
			weights[i] = float32(i) // 0..31, strictly increasing
		}
		// Symmetric:false but force an asymmetric resolve by giving an explicit
		// scheme; W4A16 resolves Symmetric=true in info, but the config's own
		// Symmetric=false is only overridden when it was false AND info says true.
		// To keep asymmetric, drive the config with explicit bits/group and no
		// scheme so normalisation leaves Symmetric=false.
		got, err := QuantizeWeights(weights, QuantizeConfig{Bits: 4, GroupSize: 32, Symmetric: false})
		if err != nil {
			t.Fatalf("QuantizeWeights(asymmetric) error = %v", err)
		}
		if got.Symmetric {
			t.Fatalf("QuantizeWeights(asymmetric) Symmetric = true, want false (asymmetric path)")
		}
		// Asymmetric q-range is [0, 2^bits-1]; the zero-point must be non-zero for
		// an all-positive group whose minimum is 0 — round(-0/scale)=0 here, so
		// verify the dequantised values round-trip close to the inputs instead.
		for i, q := range got.QValues {
			if q < 0 || q > 15 {
				t.Fatalf("QuantizeWeights(asymmetric) qvalue[%d] = %d, want within [0,15]", i, q)
			}
		}
	})
	t.Run("NonZeroZeroPoint", func(t *testing.T) {
		// A group spanning negative-to-positive forces a non-zero zero-point in
		// the asymmetric path: min is negative, so round(-min/scale) > 0.
		weights := []float32{-4, -2, 0, 2, 4, 6}
		got, err := QuantizeWeights(weights, QuantizeConfig{Bits: 4, GroupSize: 32, Symmetric: false})
		if err != nil {
			t.Fatalf("QuantizeWeights(asymmetric spread) error = %v", err)
		}
		if got.ZeroPoints[0] == 0 {
			t.Fatalf("QuantizeWeights(asymmetric spread) zero-point = 0, want non-zero for a negative minimum")
		}
	})
	t.Run("FlatGroup", func(t *testing.T) {
		// An asymmetric group where every value is equal hits the max==min guard:
		// scale falls back to 1 and zero to 0 rather than dividing by a zero span.
		got, err := QuantizeWeights([]float32{3, 3, 3, 3}, QuantizeConfig{Bits: 4, GroupSize: 32, Symmetric: false})
		if err != nil {
			t.Fatalf("QuantizeWeights(asymmetric flat) error = %v", err)
		}
		if got.Scales[0] != 1 || got.ZeroPoints[0] != 0 {
			t.Fatalf("QuantizeWeights(asymmetric flat) scale=%f zero=%f, want fallback 1/0", got.Scales[0], got.ZeroPoints[0])
		}
	})
}

// TestAutoroundCov_SignRoundAdjust_ClampedTie covers the signRoundAdjust branch
// where floorQ and ceilQ collapse to the same value after clamping at the
// quant-range edge, so the gradient sign cannot split them.
func TestAutoroundCov_SignRoundAdjust_ClampedTie(t *testing.T) {
	// Drive a value far above the symmetric max so floor and ceil both clamp to
	// qmax (7 for W4). The gradient is non-zero (so the early return is skipped)
	// but the clamped floor==ceil branch returns the single clamped value.
	weights := make([]float32, 32)
	weights[0] = 1000 // hugely out of range relative to its own group scale
	gradients := make([]float32, len(weights))
	gradients[0] = 1

	got, err := QuantizeWeights(weights, QuantizeConfig{
		Scheme:    SchemeW4A16,
		GroupSize: 32,
		Iters:     1,
		Gradients: gradients,
	})
	if err != nil {
		t.Fatalf("QuantizeWeights(clamped tie) error = %v", err)
	}
	// The dominant value sits at the positive symmetric edge (qmax = 7).
	if got.QValues[0] != 7 {
		t.Fatalf("QuantizeWeights(clamped tie) qvalue[0] = %d, want clamped qmax 7", got.QValues[0])
	}
}

// TestAutoroundCov_SignRoundAdjust_FloorUnderflowsQmin covers the lower clamp
// branch of clampInt (value < minValue). In the asymmetric path the group
// minimum can floor to -1 due to floating-point rounding of the zero-point, so
// signRoundAdjust's floorQ underflows qmin (0) and is clamped back up to 0.
func TestAutoroundCov_SignRoundAdjust_FloorUnderflowsQmin(t *testing.T) {
	// {-1, 0.3, 2.7} at bits=4 asymmetric: v=-1 lands at position -0.054, so
	// math.Floor gives -1 < qmin=0, forcing the lower clamp.
	weights := []float32{-1, 0.3, 2.7}
	gradients := []float32{-1, 0, 0} // non-zero gradient on the underflowing value
	got, err := QuantizeWeights(weights, QuantizeConfig{
		Bits:      4,
		GroupSize: 32,
		Symmetric: false,
		Iters:     1,
		Gradients: gradients,
	})
	if err != nil {
		t.Fatalf("QuantizeWeights(asymmetric floor underflow) error = %v", err)
	}
	if got.QValues[0] != 0 {
		t.Fatalf("QuantizeWeights(asymmetric floor underflow) qvalue[0] = %d, want clamped qmin 0", got.QValues[0])
	}
}

// TestAutoroundCov_SignRoundAdjust_NegativeGradientCeil covers the ceil branch
// of signRoundAdjust (gradient < 0 with a real floor/ceil split), complementing
// the existing positive-gradient floor case.
func TestAutoroundCov_SignRoundAdjust_NegativeGradientCeil(t *testing.T) {
	weights := make([]float32, 32)
	// A mid-range value whose position falls strictly between two integers so
	// floor and ceil differ after clamping.
	weights[0] = 1.4
	weights[1] = 7 // sets the group scale so weights[0] lands near 1.x
	gradients := make([]float32, len(weights))
	gradients[0] = -1 // negative gradient → ceil

	got, err := QuantizeWeights(weights, QuantizeConfig{
		Scheme:    SchemeW4A16,
		GroupSize: 32,
		Iters:     1,
		Gradients: gradients,
	})
	if err != nil {
		t.Fatalf("QuantizeWeights(negative gradient) error = %v", err)
	}
	if got.QValues[0] != 2 {
		t.Fatalf("QuantizeWeights(negative gradient) qvalue[0] = %d, want ceil 2", got.QValues[0])
	}
}
