package classify

import (
	"context"
	"testing"

	"dappco.re/go/inference"
)

// twoLabelModel classifies every prompt as label — lets a calibration run
// exercise both the agreement and the confusion-pair (disagreement) paths.
func twoLabelModel(label string) *mockModel {
	return &mockModel{
		classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i := range prompts {
				results[i] = inference.ClassifyResult{Token: inference.Token{Text: label}}
			}
			return results, nil
		},
	}
}

func BenchmarkCalibrateDomains(b *testing.B) {
	// Model A and B disagree on every sample → the confusion-pair path runs
	// once per sample; ground truth populates the accuracy path too.
	modelA := twoLabelModel("technical")
	modelB := twoLabelModel("creative")
	ctx := context.Background()
	samples := make([]CalibrationSample, 16)
	for i := range samples {
		samples[i] = CalibrationSample{Text: "She wrote a vivid poem about the sea", TrueDomain: "creative"}
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchResultSink = CalibrateDomains(ctx, modelA, modelB, samples)
	}
}
