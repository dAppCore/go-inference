// SPDX-Licence-Identifier: EUPL-1.2

package eval

import (
	"context"

	core "dappco.re/go"
)

// --- RunDataset -------------------------------------------------------------

func TestEval_RunDataset_Good(t *core.T) {
	samples := buildEvalSamples(5)
	report, err := RunDataset(context.Background(), evalRunner(samples), &evalSampleIter{samples: samples}, Config{})
	core.RequireNoError(t, err)
	core.AssertNotNil(t, report)
	core.AssertEqual(t, ReportVersion, report.Version)
	core.AssertEqual(t, 5, report.Metrics.Samples)
	core.AssertTrue(t, report.Metrics.Batches > 0)
	core.AssertEqual(t, "qwen3", report.ModelInfo.Architecture)
	core.AssertTrue(t, report.Duration > 0)
}

func TestEval_RunDataset_Bad(t *core.T) {
	samples := buildEvalSamples(2)

	// A runner without EvaluateBatch is rejected.
	_, err := RunDataset(context.Background(),
		Runner{BuildBatches: evalRunner(samples).BuildBatches},
		&evalSampleIter{samples: samples}, Config{})
	core.AssertError(t, err)

	// A nil dataset is rejected.
	_, err = RunDataset(context.Background(), evalRunner(samples), nil, Config{})
	core.AssertError(t, err)
}

func TestEval_RunDataset_Ugly(t *core.T) {
	// MaxSamples caps an over-long stream.
	samples := buildEvalSamples(50)
	report, err := RunDataset(context.Background(), evalRunner(samples), &evalSampleIter{samples: samples}, Config{MaxSamples: 10})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 10, report.Metrics.Samples)

	// An empty dataset produces no samples, which is an error.
	_, err = RunDataset(context.Background(), evalRunner(nil), &evalSampleIter{samples: nil}, Config{})
	core.AssertError(t, err)
}

// --- AdapterInfo.IsEmpty ----------------------------------------------------

func TestEval_IsEmpty_Good(t *core.T) {
	core.AssertTrue(t, AdapterInfo{}.IsEmpty())
}

func TestEval_IsEmpty_Bad(t *core.T) {
	core.AssertFalse(t, AdapterInfo{Name: "lora-1"}.IsEmpty())
	core.AssertFalse(t, AdapterInfo{Rank: 8}.IsEmpty())
	core.AssertFalse(t, AdapterInfo{Scale: 2.0}.IsEmpty())
}

func TestEval_IsEmpty_Ugly(t *core.T) {
	// Only the slice field set — still not empty.
	core.AssertFalse(t, AdapterInfo{TargetKeys: []string{"q_proj"}}.IsEmpty())
}

// --- ResponseCoverageProbe --------------------------------------------------

func sampleText(s Sample) (string, string) {
	es := s.(evalSampleShape)
	return es.Text, es.Response
}

func TestEval_ResponseCoverageProbe_Good(t *core.T) {
	probe := ResponseCoverageProbe()
	core.AssertEqual(t, "response_coverage", probe.Name)

	check := probe.Check(QualityContext{
		Samples: []Sample{
			evalSampleShape{Text: "q1", Response: "a1"},
			evalSampleShape{Text: "q2", Response: "a2"},
		},
		SampleText: sampleText,
	})
	core.AssertTrue(t, check.Pass)
	core.AssertInDelta(t, 1.0, check.Score, 0.001)
	core.AssertEqual(t, "2/2", check.Detail)
}

func TestEval_ResponseCoverageProbe_Bad(t *core.T) {
	// No SampleText accessor — the probe cannot inspect samples.
	check := ResponseCoverageProbe().Check(QualityContext{
		Samples: []Sample{evalSampleShape{Text: "q", Response: "a"}},
	})
	core.AssertFalse(t, check.Pass)
	core.AssertContains(t, check.Detail, "no SampleText accessor")
}

func TestEval_ResponseCoverageProbe_Ugly(t *core.T) {
	// Half the samples carry content — a partial-coverage fraction.
	check := ResponseCoverageProbe().Check(QualityContext{
		Samples: []Sample{
			evalSampleShape{Text: "q", Response: "a"},
			evalSampleShape{},
		},
		SampleText: sampleText,
	})
	core.AssertFalse(t, check.Pass)
	core.AssertInDelta(t, 0.5, check.Score, 0.001)
	core.AssertEqual(t, "1/2", check.Detail)
}
