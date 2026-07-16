// SPDX-Licence-Identifier: EUPL-1.2

package eval

import (
	"context"

	core "dappco.re/go"
)

// ExampleRunDataset demonstrates evaluating a dataset stream end to end:
// RunDataset drives the runner's BuildBatches/EvaluateBatch callbacks over
// every sample and returns a Report carrying the aggregated metrics.
func ExampleRunDataset() {
	samples := buildEvalSamples(5)
	report, err := RunDataset(context.Background(), evalRunner(samples), &evalSampleIter{samples: samples}, Config{})
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(report.Metrics.Samples)
	core.Println(report.ModelInfo.Architecture)
	// Output:
	// 5
	// qwen3
}

// ExampleResponseCoverageProbe demonstrates the built-in quality probe that
// checks every sample carries readable text or response content.
func ExampleResponseCoverageProbe() {
	probe := ResponseCoverageProbe()
	check := probe.Check(QualityContext{
		Samples: []Sample{
			evalSampleShape{Text: "q1", Response: "a1"},
			evalSampleShape{Text: "q2", Response: "a2"},
		},
		SampleText: sampleText,
	})
	core.Println(check.Pass)
	core.Println(check.Detail)
	// Output:
	// true
	// 2/2
}
