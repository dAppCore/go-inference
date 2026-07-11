// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import (
	core "dappco.re/go"
	"testing"
)

// The calibration benches baseline the AutoRound calibration-planning path (AX-11) — the
// native metadata work that precedes the SignRound quantise (QuantizeWeights, benched
// separately in autoround_bench_test.go): BuildCalibrationPlan normalises the config, selects
// up to NSamples, and counts tokens per sample; boundedCalibrationTokenCount /
// countCalibrationTextFields are the per-sample token estimate. Run once per calibration
// (not per token). Synthetic samples — no file, no model.

func benchCalibrationSamples(n int) []CalibrationSample {
	out := make([]CalibrationSample, n)
	for i := range out {
		out[i] = CalibrationSample{
			ID:   "sample-" + core.Sprintf("%d", i),
			Text: "The quick brown fox jumps over the lazy dog and then reconsiders its life choices in the cold dawn light.",
		}
	}
	return out
}

// BenchmarkBuildCalibrationPlan — planning a 64-of-128-sample calibration: the config
// normalise + the sample selection + the per-sample token count. The plan's Samples slice +
// the Notes are the allocation.
func BenchmarkBuildCalibrationPlan(b *testing.B) {
	samples := benchCalibrationSamples(128)
	cfg := CalibrationConfig{Scheme: SchemeW4A16, Bits: 4, GroupSize: 128, NSamples: 64, SeqLen: 2048, Iters: 200, LearningRate: 0.005}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := BuildCalibrationPlan(samples, cfg); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkBoundedCalibrationTokenCount — the per-sample token estimate (field count clamped
// to seqLen): a whitespace scan, no allocation. Run per selected sample.
func BenchmarkBoundedCalibrationTokenCount(b *testing.B) {
	sample := CalibrationSample{Text: "The quick brown fox jumps over the lazy dog and then reconsiders its life choices in the cold dawn light."}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = boundedCalibrationTokenCount(sample, 2048)
	}
}
