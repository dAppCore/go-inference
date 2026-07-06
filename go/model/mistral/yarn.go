// SPDX-Licence-Identifier: EUPL-1.2

package mistral

import "math"

// yarn.go computes the YaRN (Yet another RoPE extensioN) per-dimension inverse
// frequencies — the long-context RoPE scheme Ministral-3 declares (rope_type
// "yarn") and Qwen's 1M-context variants use. It is the NTK-by-parts remap:
// rather than scaling every frequency uniformly (linear interpolation) or none
// (extrapolation), YaRN splits by wavelength —
//
//   - HIGH-frequency dims (short wavelength, small i) EXTRAPOLATE — keep the base
//     frequency, preserving local positional resolution.
//   - LOW-frequency dims (long wavelength, large i) INTERPOLATE — divide the base
//     frequency by the context-extension factor, so they cover the longer span.
//   - a smooth linear ramp blends the two across the dims between, the ramp edges
//     fixed by the beta_fast / beta_slow rotation counts over the original context.
//
// This is the pure float computation; the resolved frequencies feed a freqs-aware
// RoPE in the decode path. mscale (the attention magnitude scaling) is a separate
// concern and 1.0 for Ministral-3, so it is not applied here.

// YaRNInvFreqs returns the dim/2 inverse frequencies for YaRN rotary embedding.
// base is rope_theta; factor the context-extension factor; betaFast/betaSlow the
// ramp's rotation-count edges; origMaxPos the pre-extension context length; dim
// the rotary dimension (the head dim for full rotary). A factor ≤ 1 yields the
// plain RoPE frequencies (base^(-2i/dim)) — YaRN with no extension is a no-op.
func YaRNInvFreqs(base, factor, betaFast, betaSlow float64, origMaxPos, dim int) []float32 {
	half := dim / 2
	out := make([]float32, half)
	if factor < 1 {
		factor = 1
	}
	low, high := yarnCorrectionRange(betaFast, betaSlow, base, origMaxPos, dim)
	for i := range half {
		extra := math.Pow(base, -float64(2*i)/float64(dim)) // standard RoPE inv-freq
		inter := extra / factor                             // interpolated (context-stretched)
		ramp := yarnRamp(low, high, i)                      // 0 at/below low → 1 at/above high
		out[i] = float32(extra*(1-ramp) + inter*ramp)       // extrapolate→interpolate blend
	}
	return out
}

// yarnCorrectionDim is the rotary dimension at which a given number of rotations
// completes over the original context: dim·ln(L / (rot·2π)) / (2·ln(base)).
func yarnCorrectionDim(numRotations, base float64, origMaxPos, dim int) float64 {
	return float64(dim) * math.Log(float64(origMaxPos)/(numRotations*2*math.Pi)) / (2 * math.Log(base))
}

// yarnCorrectionRange resolves the ramp's [low, high] dimension bounds, clamped
// to [0, dim/2-1]. low comes from the faster rotation count (beta_fast), high
// from the slower (beta_slow); beta_fast > beta_slow ⇒ low < high.
func yarnCorrectionRange(betaFast, betaSlow, base float64, origMaxPos, dim int) (float64, float64) {
	low := math.Floor(yarnCorrectionDim(betaFast, base, origMaxPos, dim))
	high := math.Ceil(yarnCorrectionDim(betaSlow, base, origMaxPos, dim))
	if low < 0 {
		low = 0
	}
	if max := float64(dim/2 - 1); high > max {
		high = max
	}
	return low, high
}

// yarnRamp is the clamped linear interpolation weight (i-low)/(high-low) ∈ [0,1]:
// 0 for the extrapolated high-frequency dims, 1 for the interpolated low-frequency
// dims, linear between.
func yarnRamp(low, high float64, i int) float64 {
	if high == low {
		high += 0.001 // avoid a divide-by-zero degenerate ramp
	}
	r := (float64(i) - low) / (high - low)
	if r < 0 {
		return 0
	}
	if r > 1 {
		return 1
	}
	return r
}
