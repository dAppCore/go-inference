// SPDX-Licence-Identifier: EUPL-1.2

package gptoss

import (
	"math"

	core "dappco.re/go"
)

// yarn.go computes GPT-OSS's YaRN ("Yet another RoPE extensioN") long-context rope correction: the
// per-dimension inverse-frequency table Config.buildArch feeds into model.Arch.RopeFreqs (consumed
// end-to-end by the ALREADY-GENERIC engine/metal RoPEFreqsBF16/encRopeDecode path — no engine change
// needed for the frequency table itself), plus the attention_factor/mscale (yarnAttentionFactor) —
// APPLIED by buildArch as an mscale² fold into the SDPA scale (exact for full-rotary heads; the
// engine's rope hook, arch.RopeScale, scales the rope ANGLE pre-cos/sin and stays 1 — a magnitude
// scale folded there would be confidently wrong, which is why the fold lives on AttnScale instead).
//
// Formula verified against the canonical implementation, fetched directly (not from training-data
// recall):
//
//	https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/modeling_rope_utils.py
//	  function _compute_yarn_parameters (lines 327-459 at fetch time)
//
// which is the SAME function GPT-OSS's own GptOssRotaryEmbedding dispatches to via
// ROPE_INIT_FUNCTIONS["yarn"] (transformers/models/gpt_oss/modeling_gpt_oss.py, class
// GptOssRotaryEmbedding.__init__ + .forward: "cos = emb.cos() * self.attention_scaling; sin =
// emb.sin() * self.attention_scaling" — confirming attention_scaling multiplies cos/sin AFTER they are
// computed from position*inv_freq, not the angle itself). mlx-lm's mlx_lm/models/gpt_oss.py (the direct
// lineage of the InferenceIllusionist MLX-4bit checkpoint this package targets) delegates the same
// scaling_config to its own initialize_rope, consistent with the same published YaRN definition.

// yarnCorrectionDim is find_correction_dim: "Inverse dimension formula to find the dimension based on the
// number of rotations" — the dimension INDEX (on the full-dim coordinate, NOT dim/2) at which a rotary
// sub-frequency completes numRotations full turns within origMaxPos positions.
func yarnCorrectionDim(numRotations float64, dim int, base float64, origMaxPos int) float64 {
	return (float64(dim) * math.Log(float64(origMaxPos)/(numRotations*2*math.Pi))) / (2 * math.Log(base))
}

// yarnCorrectionRange is find_correction_range: the [low, high] dimension-index bounds (full-dim
// coordinate) of the ramp between "keep extrapolating" (short-wavelength dims, unchanged) and "fully
// interpolate" (long-wavelength dims, divided by factor). truncate floors/ceils the raw bounds to
// integers when the config asks for it (GPT-OSS's checkpoint sets truncate:false, so the raw floats
// pass through unmodified — see yarnRopeFreqs). Both bounds are clamped into [0, dim-1].
func yarnCorrectionRange(betaFast, betaSlow float64, dim int, base float64, origMaxPos int, truncate bool) (low, high float64) {
	low = yarnCorrectionDim(betaFast, dim, base, origMaxPos)
	high = yarnCorrectionDim(betaSlow, dim, base, origMaxPos)
	if truncate {
		low = math.Floor(low)
		high = math.Ceil(high)
	}
	if low < 0 {
		low = 0
	}
	if maxIdx := float64(dim - 1); high > maxIdx {
		high = maxIdx
	}
	return low, high
}

// yarnLinearRamp is linear_ramp_factor: a length-n ramp from 0 (index <= min) to 1 (index >= max),
// linear in between. min==max is nudged apart (the reference's "prevent singularity" guard) rather than
// dividing by zero.
func yarnLinearRamp(min, max float64, n int) []float64 {
	if min == max {
		max += 0.001
	}
	ramp := make([]float64, n)
	for i := range ramp {
		v := (float64(i) - min) / (max - min)
		switch {
		case v < 0:
			v = 0
		case v > 1:
			v = 1
		}
		ramp[i] = v
	}
	return ramp
}

// yarnRopeFreqs computes the YaRN-corrected inverse-frequency table for model.Arch.RopeFreqs — length
// rotaryDim/2, directly consumable by the existing engine/metal freqs-rope path (rope_freqs.go inverts
// them into periods; no engine change). Returns (nil, nil) — the "derive uniformly from RopeBase"
// signal RopeFreqs already documents — when the config doesn't declare rope_type "yarn" (a gpt_oss
// config always does, but this keeps a non-YaRN gpt_oss variant, were one ever published, on the plain
// base-derived spectrum instead of silently mis-scaling it).
func (c *Config) yarnRopeFreqs(rotaryDim int, ropeBase float32) ([]float32, error) {
	rs := c.RopeScaling
	if rs.RopeType != "yarn" {
		return nil, nil
	}
	if rotaryDim <= 0 || rotaryDim%2 != 0 {
		return nil, core.NewError("gptoss.Config.yarnRopeFreqs: rotaryDim must be even and > 0")
	}
	factor := float64(rs.Factor)
	if factor <= 0 {
		return nil, core.NewError("gptoss.Config.yarnRopeFreqs: rope_scaling.factor must be > 0 for rope_type \"yarn\"")
	}
	if rs.OriginalMaxPositionEmbeddings <= 0 {
		return nil, core.NewError("gptoss.Config.yarnRopeFreqs: rope_scaling.original_max_position_embeddings must be > 0 for rope_type \"yarn\"")
	}
	betaFast := float64(rs.BetaFast)
	if betaFast <= 0 {
		betaFast = 32 // the paper's default, matching transformers' "rope_parameters_dict.get(\"beta_fast\") or 32"
	}
	betaSlow := float64(rs.BetaSlow)
	if betaSlow <= 0 {
		betaSlow = 1
	}
	base := float64(ropeBase)
	half := rotaryDim / 2

	low, high := yarnCorrectionRange(betaFast, betaSlow, rotaryDim, base, rs.OriginalMaxPositionEmbeddings, rs.Truncate)
	ramp := yarnLinearRamp(low, high, half) // extrapolation_factor = 1 - ramp

	freqs := make([]float32, half)
	for i := range freqs {
		posFreq := math.Pow(base, float64(2*i)/float64(rotaryDim))
		extrapolation := 1.0 / posFreq
		interpolation := 1.0 / (factor * posFreq)
		// inv_freq = interpolation*(1-extrap_factor) + extrapolation*extrap_factor, extrap_factor = 1-ramp[i]
		// ⇒ inv_freq = interpolation*ramp[i] + extrapolation*(1-ramp[i])
		freqs[i] = float32(interpolation*ramp[i] + extrapolation*(1-ramp[i]))
	}
	return freqs, nil
}

// yarnAttentionFactor is get_mscale(factor) — the reference's attention_factor default when the config
// (as every real gpt_oss checkpoint does) sets neither "attention_factor" nor "mscale"/"mscale_all_dim"
// under rope_scaling. Returns 1.0 (no scaling) for factor<=1, matching get_mscale's own guard. APPLIED
// by buildArch as AttnScale = mscale²/√headDim — the reference scales cos AND sin (equivalently, the
// pre-rope q/k input) by mscale, so both q and k carry it and the attention logits carry its square;
// full derivation + both fetched sources in buildArch's comment.
func yarnAttentionFactor(factor float32) float32 {
	f := float64(factor)
	if f <= 1 {
		return 1
	}
	return float32(0.1*math.Log(f) + 1.0)
}
