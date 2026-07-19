// SPDX-Licence-Identifier: EUPL-1.2

package gptoss

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/arch/Qwen/qwen35"
)

const yarnTol = 1e-3

func approxEqual(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// TestYarnCorrectionDim_Good pins find_correction_dim against the real gpt-oss-20b YaRN parameters
// (dim=64, base=150000, original_max_position_embeddings=4096), values independently computed from the
// verified reference formula (see yarn.go doc) — beta_fast=32 -> low bound, beta_slow=1 -> high bound.
func TestYarnCorrectionDim_Good(t *testing.T) {
	low := yarnCorrectionDim(32, 64, 150000, 4096)
	high := yarnCorrectionDim(1, 64, 150000, 4096)
	if !approxEqual(low, 8.0928, yarnTol) {
		t.Fatalf("yarnCorrectionDim(32,...) = %v, want ~8.0928", low)
	}
	if !approxEqual(high, 17.3980, yarnTol) {
		t.Fatalf("yarnCorrectionDim(1,...) = %v, want ~17.3980", high)
	}
}

// TestYarnCorrectionRange_Good proves truncate=false (GPT-OSS's own config setting) passes the raw
// float bounds through UNMODIFIED — not floored/ceiled — matching gpt-oss-20b's real low/high.
func TestYarnCorrectionRange_Good(t *testing.T) {
	low, high := yarnCorrectionRange(32, 1, 64, 150000, 4096, false)
	if !approxEqual(low, 8.0928, yarnTol) || !approxEqual(high, 17.3980, yarnTol) {
		t.Fatalf("yarnCorrectionRange(truncate=false) = (%v, %v), want (~8.0928, ~17.3980)", low, high)
	}
}

// TestYarnCorrectionRange_Bad proves truncate=true floors the low bound and ceils the high bound to
// integers — the behaviour gpt-oss-20b's own config explicitly opts OUT of (truncate:false).
func TestYarnCorrectionRange_Bad(t *testing.T) {
	low, high := yarnCorrectionRange(32, 1, 64, 150000, 4096, true)
	if low != 8 || high != 18 {
		t.Fatalf("yarnCorrectionRange(truncate=true) = (%v, %v), want (8, 18)", low, high)
	}
}

// TestYarnCorrectionRange_Ugly proves both ends of the [0, dim-1] clamp fire: an extreme beta_fast
// drives the raw low bound negative (clamped to 0), an extreme beta_slow drives the raw high bound past
// dim-1 (clamped to 63) — surprising-but-valid inputs the formula itself does not bound.
func TestYarnCorrectionRange_Ugly(t *testing.T) {
	low, high := yarnCorrectionRange(100000, 1e-8, 64, 150000, 4096, false)
	if low != 0 {
		t.Fatalf("yarnCorrectionRange low = %v, want 0 (clamped, raw was negative)", low)
	}
	if high != 63 {
		t.Fatalf("yarnCorrectionRange high = %v, want 63 = dim-1 (clamped, raw exceeded it)", high)
	}
}

func TestYarnLinearRamp_Good(t *testing.T) {
	ramp := yarnLinearRamp(2, 6, 8)
	want := []float64{0, 0, 0, 0.25, 0.5, 0.75, 1, 1}
	for i, w := range want {
		if !approxEqual(ramp[i], w, 1e-9) {
			t.Fatalf("yarnLinearRamp(2,6,8)[%d] = %v, want %v", i, ramp[i], w)
		}
	}
}

// TestYarnLinearRamp_Bad proves values below min clamp to 0 and above max clamp to 1 (not merely
// "close to" — the ramp is not linear outside [min,max]), using bounds offset from zero.
func TestYarnLinearRamp_Bad(t *testing.T) {
	ramp := yarnLinearRamp(10, 12, 4) // indices 0..3 are all < min=10
	for i, v := range ramp {
		if v != 0 {
			t.Fatalf("yarnLinearRamp[%d] = %v, want 0 (every index here is below min)", i, v)
		}
	}
}

// TestYarnLinearRamp_Ugly proves min==max is nudged apart rather than dividing by zero (NaN/Inf) — the
// reference's "prevent singularity" guard.
func TestYarnLinearRamp_Ugly(t *testing.T) {
	ramp := yarnLinearRamp(5, 5, 8)
	for i, v := range ramp {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("yarnLinearRamp(5,5,...)[%d] = %v, want a finite clamped value (singularity guard)", i, v)
		}
	}
	if ramp[5] != 0 { // i == min: (5-5)/0.001 == 0
		t.Fatalf("yarnLinearRamp(5,5,...)[5] = %v, want 0", ramp[5])
	}
	if ramp[6] != 1 { // i == min+1: (6-5)/0.001 == 1000, clamped to 1
		t.Fatalf("yarnLinearRamp(5,5,...)[6] = %v, want 1 (clamped)", ramp[6])
	}
}

// TestConfig_yarnRopeFreqs_Good pins the full table against the real gpt-oss-20b YaRN parameters at
// the two closed-form boundary points (i=0 is always exactly the unscaled base frequency since ramp(0)
// is clamped to 0; i=rotaryDim/2-1 is always exactly extrapolation/factor since ramp(last) clamps to 1)
// plus one independently-computed interior spot-check.
func TestConfig_yarnRopeFreqs_Good(t *testing.T) {
	cfg := realConfig()
	freqs, err := cfg.yarnRopeFreqs(64, 150000)
	if err != nil {
		t.Fatalf("yarnRopeFreqs: %v", err)
	}
	if len(freqs) != 32 {
		t.Fatalf("yarnRopeFreqs length = %d, want 32", len(freqs))
	}
	if !approxEqual(float64(freqs[0]), 1.0, 1e-6) {
		t.Fatalf("yarnRopeFreqs[0] = %v, want 1.0 (i=0 is always pure extrapolation, base^0=1)", freqs[0])
	}
	wantLast := 3.0235114281192144e-07
	if !approxEqual(float64(freqs[31]), wantLast, wantLast*1e-3) {
		t.Fatalf("yarnRopeFreqs[31] = %v, want ~%v (pure interpolation: extrapolation[31]/factor)", freqs[31], wantLast)
	}
	wantMid := 0.050813274815461475 // i=8, still just inside the pure-extrapolation zone (low≈8.09)
	if !approxEqual(float64(freqs[8]), wantMid, wantMid*1e-3) {
		t.Fatalf("yarnRopeFreqs[8] = %v, want ~%v", freqs[8], wantMid)
	}
}

// TestConfig_yarnRopeFreqs_Bad proves a non-"yarn" rope_type returns (nil, nil) — the "derive uniformly
// from RopeBase" signal model.Arch.RopeFreqs documents — rather than fabricating a table gpt_oss never
// asked for. This IS the cross-arch regression contract: see TestNonYaRNArch_RopeFreqs_Unchanged for the
// proof that a real non-YaRN arch package is unaffected end-to-end.
func TestConfig_yarnRopeFreqs_Bad(t *testing.T) {
	cfg := realConfig()
	cfg.RopeScaling.RopeType = ""
	freqs, err := cfg.yarnRopeFreqs(64, 150000)
	if err != nil || freqs != nil {
		t.Fatalf("yarnRopeFreqs with no rope_type = (%v, %v), want (nil, nil)", freqs, err)
	}
}

// TestConfig_yarnRopeFreqs_Ugly proves an odd rotaryDim is rejected — YaRN's dim/2-length frequency
// table has no defined shape for an odd rotary width (every real gpt_oss head_dim is even).
func TestConfig_yarnRopeFreqs_Ugly(t *testing.T) {
	cfg := realConfig()
	if _, err := cfg.yarnRopeFreqs(63, 150000); err == nil {
		t.Fatal("yarnRopeFreqs accepted an odd rotaryDim")
	}
}

func TestYarnAttentionFactor_Good(t *testing.T) {
	got := yarnAttentionFactor(32)
	if !approxEqual(float64(got), 1.3465735902799727, 1e-6) {
		t.Fatalf("yarnAttentionFactor(32) = %v, want ~1.34657359", got)
	}
}

// TestYarnAttentionFactor_Bad proves factor<=1 (no scaling requested) returns exactly 1.0 — the
// reference's get_mscale guard — rather than evaluating log(<=1) (zero or negative domain).
func TestYarnAttentionFactor_Bad(t *testing.T) {
	if got := yarnAttentionFactor(1); got != 1 {
		t.Fatalf("yarnAttentionFactor(1) = %v, want 1", got)
	}
	if got := yarnAttentionFactor(0.5); got != 1 {
		t.Fatalf("yarnAttentionFactor(0.5) = %v, want 1", got)
	}
}

// TestYarnAttentionFactor_Ugly proves a negative factor (nonsensical, but syntactically a float32) is
// treated the same as factor<=1 — the guard is "<=1", not "<0", so it does not panic into log of a
// negative number either way.
func TestYarnAttentionFactor_Ugly(t *testing.T) {
	if got := yarnAttentionFactor(-5); got != 1 {
		t.Fatalf("yarnAttentionFactor(-5) = %v, want 1 (guarded, not log(negative))", got)
	}
}

// TestNonYaRNArch_RopeFreqs_Unchanged is the cross-arch regression proof the task requires: a real,
// already-shipped arch package (qwen35, which also sets RopeBase/RopeScale on every layer) must resolve
// a nil RopeFreqs and its configured RopeBase UNCHANGED by this gpt_oss/YaRN work — gpt_oss's YaRN table
// construction lives entirely inside the gptoss package and is consumed via the PRE-EXISTING generic
// model.Arch.RopeFreqs/engine.RoPEFreqsBF16 mechanism, so no shared code path was touched; this test is
// the anchor that would catch a future refactor accidentally coupling the two.
func TestNonYaRNArch_RopeFreqs_Unchanged(t *testing.T) {
	cfg := qwen35.Config{
		HiddenSize: 8, NumHiddenLayers: 4, NumAttentionHeads: 4, NumKeyValueHeads: 4, HeadDim: 2,
		RopeTheta: 1_000_000, FullAttentionInterval: 4,
	}
	a, err := cfg.Arch()
	if err != nil {
		t.Fatalf("qwen35.Config.Arch: %v", err)
	}
	if a.RopeFreqs != nil {
		t.Fatalf("qwen35 Arch.RopeFreqs = %v, want nil (qwen35 derives uniformly from RopeBase, no YaRN)", a.RopeFreqs)
	}
	if a.RopeBase != 1_000_000 || a.RopeScale != 1 {
		t.Fatalf("qwen35 Arch.RopeBase/RopeScale = %v/%v, want 1000000/1 (unchanged by gptoss's YaRN work)", a.RopeBase, a.RopeScale)
	}
}
