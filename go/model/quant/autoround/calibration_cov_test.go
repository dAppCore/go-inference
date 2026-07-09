// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import "testing"

// TestCalibrationCov_BuildCalibrationPlan_NegativeTokenN covers the
// boundedCalibrationTokenCount floor branch: a sample carrying a negative
// explicit token count (with no text to recompute from) is clamped to zero and
// marked skipped rather than contributing a negative token total.
func TestCalibrationCov_BuildCalibrationPlan_NegativeTokenN(t *testing.T) {
	samples := []CalibrationSample{
		{ID: "neg", TokenN: -5},           // negative explicit count, no text
		{ID: "ok", Text: "one two three"}, // a normal companion sample
	}
	plan, err := BuildCalibrationPlan(samples, CalibrationConfig{Scheme: SchemeW4A16, NSamples: 8, SeqLen: 16})
	if err != nil {
		t.Fatalf("BuildCalibrationPlan(negative token count) error = %v", err)
	}
	if len(plan.Samples) != 2 {
		t.Fatalf("BuildCalibrationPlan(negative token count) selected %d samples, want 2", len(plan.Samples))
	}
	// The negative sample is floored to zero tokens and flagged skipped.
	if plan.Samples[0].TokenN != 0 || !plan.Samples[0].Skipped {
		t.Fatalf("BuildCalibrationPlan(negative token count) sample[0] = %+v, want zero-token skipped", plan.Samples[0])
	}
	// Only the valid companion sample contributes tokens; the negative one adds none.
	if plan.TokenCount != 3 {
		t.Fatalf("BuildCalibrationPlan(negative token count) total tokens = %d, want 3 from the valid sample only", plan.TokenCount)
	}
}
