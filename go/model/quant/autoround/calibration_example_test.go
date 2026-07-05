// SPDX-Licence-Identifier: EUPL-1.2

package autoround_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/autoround"
)

// ExampleCalibrationConfigFromProfile expands a profile into the calibration
// config that BuildCalibrationPlan consumes, carrying the sample and sequence
// budgets across.
func ExampleCalibrationConfigFromProfile() {
	profile, _ := autoround.LookupProfile(autoround.ProfileAutoRoundBest)
	cfg := autoround.CalibrationConfigFromProfile(profile)
	core.Println(cfg.Scheme, cfg.Bits, cfg.NSamples, cfg.SeqLen)
	// Output: W2A16 2 512 2048
}

// ExampleBuildCalibrationPlan selects a bounded sample set, truncating to the
// configured NSamples and clamping each sample's token count to SeqLen.
func ExampleBuildCalibrationPlan() {
	cfg := autoround.CalibrationConfig{Scheme: autoround.SchemeW4A16, GroupSize: 32, NSamples: 2, SeqLen: 3}
	samples := []autoround.CalibrationSample{
		{ID: "a", Text: "one two three four"},
		{ID: "b", TokenN: 2},
		{ID: "c", TokenN: 1},
	}
	plan, err := autoround.BuildCalibrationPlan(samples, cfg)
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(plan.InputSamples, plan.SelectedSamples, plan.Truncated, plan.TokenCount)
	// Output: 3 2 true 5
}

// ExampleQuantizeWithCalibration plans calibration and quantises a tensor in
// one call, applying SignRound where per-weight gradients are supplied.
func ExampleQuantizeWithCalibration() {
	weights := make([]float32, 32)
	weights[0] = 1.4
	weights[1] = 1.4
	weights[2] = 7
	gradients := make([]float32, len(weights))
	gradients[0] = 1
	gradients[1] = -1
	run, err := autoround.QuantizeWithCalibration(weights, gradients, []autoround.CalibrationSample{{ID: "a", TokenN: 4}}, autoround.CalibrationConfig{
		Scheme:    autoround.SchemeW4A16,
		GroupSize: 32,
		Iters:     1,
		NSamples:  1,
		SeqLen:    8,
	})
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(run.Plan.SelectedSamples, run.Weights.QValues[0], run.Weights.QValues[1])
	// Output: 1 1 2
}
