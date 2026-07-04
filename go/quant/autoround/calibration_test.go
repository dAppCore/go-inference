// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import "testing"

func TestCalibration_CalibrationConfigFromProfile_Good(t *testing.T) {
	profile, ok := LookupProfile(ProfileAutoRoundBest)
	if !ok {
		t.Fatal("LookupProfile(auto-round-best) ok = false")
	}
	cfg := CalibrationConfigFromProfile(profile)
	if cfg.Scheme != SchemeW2A16 || cfg.Bits != 2 || cfg.GroupSize != 32 || !cfg.Symmetric {
		t.Fatalf("CalibrationConfigFromProfile = %+v, want W2A16 group 32 symmetric", cfg)
	}
	if cfg.NSamples != 512 || cfg.SeqLen != 2048 || cfg.Iters != 1000 {
		t.Fatalf("CalibrationConfigFromProfile = %+v, want best calibration knobs", cfg)
	}
}

func TestCalibration_CalibrationConfigFromProfile_Bad(t *testing.T) {
	// CalibrationConfigFromProfile derives quant fields via ConfigFromProfile,
	// so an unresolvable scheme yields zero bits. The resulting config is not
	// usable until BuildCalibrationPlan's normalisation rejects it.
	cfg := CalibrationConfigFromProfile(Profile{Scheme: "bogus", NSamples: 4, SeqLen: 8})
	if cfg.Bits != 0 {
		t.Fatalf("CalibrationConfigFromProfile(unknown scheme) bits = %d, want 0", cfg.Bits)
	}
	if _, err := BuildCalibrationPlan(nil, cfg); err == nil {
		t.Fatal("BuildCalibrationPlan(unknown-scheme config) err = nil, want unsupported scheme error")
	}
}

func TestCalibration_CalibrationConfigFromProfile_Ugly(t *testing.T) {
	// Zero-value profile carries no sample/sequence overrides; the produced
	// config leaves NSamples and SeqLen at zero so BuildCalibrationPlan can fill
	// its own defaults rather than inheriting a phantom width.
	cfg := CalibrationConfigFromProfile(Profile{})
	if cfg.NSamples != 0 || cfg.SeqLen != 0 {
		t.Fatalf("CalibrationConfigFromProfile(zero) = %+v, want zero sample/seq knobs", cfg)
	}
}

func TestCalibration_BuildCalibrationPlan_Good(t *testing.T) {
	profile, ok := LookupProfile(ProfileAutoRoundLight)
	if !ok {
		t.Fatal("missing auto-round-light profile")
	}
	cfg := CalibrationConfigFromProfile(profile)
	cfg.NSamples = 2
	cfg.SeqLen = 3
	samples := []CalibrationSample{
		{ID: "a", Text: "one two three four"},
		{ID: "b", TokenN: 2},
		{ID: "c", TokenN: 1},
	}

	plan, err := BuildCalibrationPlan(samples, cfg)
	if err != nil {
		t.Fatalf("BuildCalibrationPlan() error = %v", err)
	}
	if plan.InputSamples != 3 || plan.SelectedSamples != 2 || !plan.Truncated || plan.TokenCount != 5 {
		t.Fatalf("plan = %+v, want two selected samples and bounded token count", plan)
	}
	if plan.Config.Iters != 50 || plan.Config.NSamples != 2 || plan.Config.SeqLen != 3 {
		t.Fatalf("plan config = %+v, want profile calibration defaults with overrides", plan.Config)
	}
}

func TestCalibration_BuildCalibrationPlan_Bad(t *testing.T) {
	// An invalid quant field (group size 16 is not in the allowed set) must be
	// rejected by the shared normalisation before any sample is planned.
	if _, err := BuildCalibrationPlan(nil, CalibrationConfig{Bits: 4, GroupSize: 16}); err == nil {
		t.Fatal("BuildCalibrationPlan(bad group size) err = nil, want group size error")
	}
	// Negative sample/sequence counts are rejected too.
	if _, err := BuildCalibrationPlan(nil, CalibrationConfig{Scheme: SchemeW4A16, NSamples: -1}); err == nil {
		t.Fatal("BuildCalibrationPlan(negative nsamples) err = nil, want nsamples error")
	}
	if _, err := BuildCalibrationPlan(nil, CalibrationConfig{Scheme: SchemeW4A16, SeqLen: -1}); err == nil {
		t.Fatal("BuildCalibrationPlan(negative seqlen) err = nil, want seqlen error")
	}
}

func TestCalibration_BuildCalibrationPlan_Ugly(t *testing.T) {
	// No samples is the degenerate plan: it succeeds, fills default knobs, and
	// selects nothing rather than truncating an empty list.
	plan, err := BuildCalibrationPlan(nil, CalibrationConfig{Scheme: SchemeW4A16})
	if err != nil {
		t.Fatalf("BuildCalibrationPlan(no samples) error = %v", err)
	}
	if plan.SelectedSamples != 0 || plan.Truncated || plan.InputSamples != 0 {
		t.Fatalf("BuildCalibrationPlan(no samples) plan = %+v, want empty unselected plan", plan)
	}
	if plan.Config.NSamples != 128 || plan.Config.SeqLen != 2048 {
		t.Fatalf("BuildCalibrationPlan(no samples) config = %+v, want default calibration knobs", plan.Config)
	}
	// An empty-text sample with no explicit token count is marked skipped and
	// contributes zero tokens rather than panicking on the empty word scan.
	plan, err = BuildCalibrationPlan([]CalibrationSample{{ID: "blank"}}, CalibrationConfig{Scheme: SchemeW4A16})
	if err != nil {
		t.Fatalf("BuildCalibrationPlan(blank sample) error = %v", err)
	}
	if plan.SelectedSamples != 1 || !plan.Samples[0].Skipped || plan.TokenCount != 0 {
		t.Fatalf("BuildCalibrationPlan(blank sample) plan = %+v, want one skipped zero-token sample", plan)
	}
}

func TestCalibration_QuantizeWithCalibration_Good(t *testing.T) {
	weights := make([]float32, 32)
	weights[0] = 1.4
	weights[1] = 1.4
	weights[2] = 7
	gradients := make([]float32, len(weights))
	gradients[0] = 1
	gradients[1] = -1

	run, err := QuantizeWithCalibration(weights, gradients, []CalibrationSample{{ID: "a", TokenN: 4}}, CalibrationConfig{
		Scheme:    SchemeW4A16,
		GroupSize: 32,
		Iters:     1,
		NSamples:  1,
		SeqLen:    8,
	})
	if err != nil {
		t.Fatalf("QuantizeWithCalibration() error = %v", err)
	}
	if run.Plan.SelectedSamples != 1 || run.Weights.QValues[0] != 1 || run.Weights.QValues[1] != 2 {
		t.Fatalf("run = %+v, want calibration plan plus SignRound split", run)
	}
}

func TestCalibration_QuantizeWithCalibration_Bad(t *testing.T) {
	// A bad calibration config fails in the plan stage, before any weight is
	// touched.
	if _, err := QuantizeWithCalibration([]float32{1, 2}, nil, nil, CalibrationConfig{Bits: 5}); err == nil {
		t.Fatal("QuantizeWithCalibration(bad bits) err = nil, want validation error")
	}
	// A valid plan but empty weights fails in the quant stage.
	if _, err := QuantizeWithCalibration(nil, nil, []CalibrationSample{{ID: "a", TokenN: 1}}, CalibrationConfig{Scheme: SchemeW4A16, GroupSize: 32}); err == nil {
		t.Fatal("QuantizeWithCalibration(empty weights) err = nil, want weights-required error")
	}
}

func TestCalibration_QuantizeWithCalibration_Ugly(t *testing.T) {
	// Mismatched gradient length is the degenerate SignRound input: when iters
	// are requested but the gradient count differs from the weight count, the
	// quant stage rejects it rather than indexing out of range.
	if _, err := QuantizeWithCalibration([]float32{1, 2, 3}, []float32{1}, nil, CalibrationConfig{
		Scheme:    SchemeW4A16,
		GroupSize: 32,
		Iters:     1,
	}); err == nil {
		t.Fatal("QuantizeWithCalibration(gradient mismatch) err = nil, want gradient-count error")
	}
	// Zero iters with no gradients is plain RTN: it succeeds and still attaches
	// the calibration plan.
	run, err := QuantizeWithCalibration([]float32{0.5, -0.5}, nil, nil, CalibrationConfig{
		Scheme:    SchemeW4A16,
		GroupSize: 32,
	})
	if err != nil {
		t.Fatalf("QuantizeWithCalibration(rtn) error = %v", err)
	}
	if len(run.Weights.QValues) != 2 {
		t.Fatalf("QuantizeWithCalibration(rtn) qvalues = %d, want 2", len(run.Weights.QValues))
	}
}
