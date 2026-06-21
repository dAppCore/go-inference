package classify

import (
	"context"
	"testing"

	"dappco.re/go/inference"
)

func TestCalibrateDomains_FullAgreement(t *testing.T) {
	// Both models return the same domain for all samples.
	model := &mockModel{
		classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i := range prompts {
				results[i] = inference.ClassifyResult{Token: inference.Token{Text: "technical"}}
			}
			return results, nil
		},
	}

	samples := []CalibrationSample{
		{Text: "Delete the file", TrueDomain: "technical"},
		{Text: "Build the project", TrueDomain: "technical"},
		{Text: "Run the tests", TrueDomain: "technical"},
	}

	stats, err := valueFromResult[*CalibrationStats](CalibrateDomains(context.Background(), model, model, samples))
	if err != nil {
		t.Fatalf("CalibrateDomains: %v", err)
	}

	if stats.Total != 3 {
		t.Errorf("Total = %d, want 3", stats.Total)
	}
	if stats.Agreed != 3 {
		t.Errorf("Agreed = %d, want 3", stats.Agreed)
	}
	if stats.AgreementRate != 1.0 {
		t.Errorf("AgreementRate = %f, want 1.0", stats.AgreementRate)
	}
	if stats.AccuracyA != 1.0 {
		t.Errorf("AccuracyA = %f, want 1.0", stats.AccuracyA)
	}
	if stats.AccuracyB != 1.0 {
		t.Errorf("AccuracyB = %f, want 1.0", stats.AccuracyB)
	}
	if len(stats.ConfusionPairs) != 0 {
		t.Errorf("ConfusionPairs = %v, want empty", stats.ConfusionPairs)
	}
}

func TestCalibrateDomains_Disagreement(t *testing.T) {
	// Model A always says "technical", model B always says "creative".
	modelA := &mockModel{
		classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i := range prompts {
				results[i] = inference.ClassifyResult{Token: inference.Token{Text: "technical"}}
			}
			return results, nil
		},
	}
	modelB := &mockModel{
		classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i := range prompts {
				results[i] = inference.ClassifyResult{Token: inference.Token{Text: "creative"}}
			}
			return results, nil
		},
	}

	samples := []CalibrationSample{
		{Text: "She wrote a poem", TrueDomain: "creative"},
		{Text: "He painted the sky", TrueDomain: "creative"},
	}

	stats, err := valueFromResult[*CalibrationStats](CalibrateDomains(context.Background(), modelA, modelB, samples))
	if err != nil {
		t.Fatalf("CalibrateDomains: %v", err)
	}

	if stats.Agreed != 0 {
		t.Errorf("Agreed = %d, want 0", stats.Agreed)
	}
	if stats.AgreementRate != 0 {
		t.Errorf("AgreementRate = %f, want 0", stats.AgreementRate)
	}
	if stats.CorrectA != 0 {
		t.Errorf("CorrectA = %d, want 0 (A said technical, truth is creative)", stats.CorrectA)
	}
	if stats.CorrectB != 2 {
		t.Errorf("CorrectB = %d, want 2", stats.CorrectB)
	}
	if stats.ConfusionPairs["technical->creative"] != 2 {
		t.Errorf("ConfusionPairs[technical->creative] = %d, want 2", stats.ConfusionPairs["technical->creative"])
	}
}

func TestCalibrateDomains_MixedAgreement(t *testing.T) {
	// Model A and B agree on first sample, disagree on second.
	callCount := 0
	modelA := &mockModel{
		classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i := range prompts {
				results[i] = inference.ClassifyResult{Token: inference.Token{Text: "ethical"}}
			}
			return results, nil
		},
	}
	modelB := &mockModel{
		classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			callCount++
			results := make([]inference.ClassifyResult, len(prompts))
			for i, p := range prompts {
				if i == 0 && callCount == 1 {
					// First batch: agree on first item
					results[i] = inference.ClassifyResult{Token: inference.Token{Text: "ethical"}}
				} else {
					_ = p
					results[i] = inference.ClassifyResult{Token: inference.Token{Text: "technical"}}
				}
			}
			return results, nil
		},
	}

	samples := []CalibrationSample{
		{Text: "We should act fairly"},
		{Text: "Delete the config"},
	}

	stats, err := valueFromResult[*CalibrationStats](CalibrateDomains(context.Background(), modelA, modelB, samples, WithBatchSize(16)))
	if err != nil {
		t.Fatalf("CalibrateDomains: %v", err)
	}

	if stats.Total != 2 {
		t.Errorf("Total = %d, want 2", stats.Total)
	}
	if stats.Agreed != 1 {
		t.Errorf("Agreed = %d, want 1", stats.Agreed)
	}
	if got := stats.AgreementRate; got != 0.5 {
		t.Errorf("AgreementRate = %f, want 0.5", got)
	}
}

func TestCalibrateDomains_NoGroundTruth(t *testing.T) {
	// Samples without TrueDomain: accuracy should be 0, agreement still measured.
	model := &mockModel{
		classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i := range prompts {
				results[i] = inference.ClassifyResult{Token: inference.Token{Text: "casual"}}
			}
			return results, nil
		},
	}

	samples := []CalibrationSample{
		{Text: "Went to the store"},
		{Text: "Had coffee this morning"},
	}

	stats, err := valueFromResult[*CalibrationStats](CalibrateDomains(context.Background(), model, model, samples))
	if err != nil {
		t.Fatalf("CalibrateDomains: %v", err)
	}

	if stats.WithTruth != 0 {
		t.Errorf("WithTruth = %d, want 0", stats.WithTruth)
	}
	if stats.AccuracyA != 0 {
		t.Errorf("AccuracyA = %f, want 0 (no ground truth)", stats.AccuracyA)
	}
	if stats.Agreed != 2 {
		t.Errorf("Agreed = %d, want 2", stats.Agreed)
	}
}

func TestCalibrateDomains_EmptySamples(t *testing.T) {
	model := &mockModel{
		classifyFunc: func(_ context.Context, _ []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			return nil, nil
		},
	}

	_, err := valueFromResult[*CalibrationStats](CalibrateDomains(context.Background(), model, model, nil))
	if err == nil {
		t.Error("expected error for empty samples, got nil")
	}
}

func TestCalibrateDomains_BatchBoundary(t *testing.T) {
	// 7 samples with batch size 3: tests partial last batch.
	model := &mockModel{
		classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i := range prompts {
				results[i] = inference.ClassifyResult{Token: inference.Token{Text: "technical"}}
			}
			return results, nil
		},
	}

	samples := make([]CalibrationSample, 7)
	for i := range samples {
		samples[i] = CalibrationSample{Text: "Build the project"}
	}

	stats, err := valueFromResult[*CalibrationStats](CalibrateDomains(context.Background(), model, model, samples, WithBatchSize(3)))
	if err != nil {
		t.Fatalf("CalibrateDomains: %v", err)
	}

	if stats.Total != 7 {
		t.Errorf("Total = %d, want 7", stats.Total)
	}
	if stats.Agreed != 7 {
		t.Errorf("Agreed = %d, want 7", stats.Agreed)
	}
}

func TestCalibrateDomains_ResultsSlice(t *testing.T) {
	// Verify individual results are populated correctly.
	modelA := &mockModel{
		classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i := range prompts {
				results[i] = inference.ClassifyResult{Token: inference.Token{Text: "ethical"}}
			}
			return results, nil
		},
	}
	modelB := &mockModel{
		classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i := range prompts {
				results[i] = inference.ClassifyResult{Token: inference.Token{Text: "casual"}}
			}
			return results, nil
		},
	}

	samples := []CalibrationSample{
		{Text: "Be fair to everyone", TrueDomain: "ethical"},
	}

	stats, err := valueFromResult[*CalibrationStats](CalibrateDomains(context.Background(), modelA, modelB, samples))
	if err != nil {
		t.Fatalf("CalibrateDomains: %v", err)
	}

	if len(stats.Results) != 1 {
		t.Fatalf("Results len = %d, want 1", len(stats.Results))
	}

	r := stats.Results[0]
	if r.Text != "Be fair to everyone" {
		t.Errorf("Text = %q", r.Text)
	}
	if r.TrueDomain != "ethical" {
		t.Errorf("TrueDomain = %q", r.TrueDomain)
	}
	if r.DomainA != "ethical" {
		t.Errorf("DomainA = %q, want ethical", r.DomainA)
	}
	if r.DomainB != "casual" {
		t.Errorf("DomainB = %q, want casual", r.DomainB)
	}
	if r.Agree {
		t.Error("Agree = true, want false")
	}
}

// --- AX-7 canonical triplets ---

func TestCalibrate_CalibrateDomains_Good(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := &mockModel{classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i := range prompts {
				results[i] = inference.ClassifyResult{Token: inference.Token{Text: "technical"}}
			}
			return results, nil
		}}
		samples := []CalibrationSample{{Text: "Delete the file", TrueDomain: "technical"}}
		stats, err := valueFromResult[*CalibrationStats](CalibrateDomains(context.Background(), model, model, samples))
		if err != nil || stats.Total != 1 {
			t.Fatalf("stats=%+v err=%v", stats, err)
		}
	})
	if !called {
		t.Fatal("CalibrateDomains was not exercised")
	}
}

func TestCalibrate_CalibrateDomains_Bad(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := &mockModel{classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i := range prompts {
				results[i] = inference.ClassifyResult{Token: inference.Token{Text: "technical"}}
			}
			return results, nil
		}}
		_, err := valueFromResult[*CalibrationStats](CalibrateDomains(context.Background(), model, model, nil))
		if err == nil {
			t.Fatal("expected error")
		}
	})
	if !called {
		t.Fatal("CalibrateDomains was not exercised")
	}
}

func TestCalibrate_CalibrateDomains_Ugly(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := &mockModel{classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i := range prompts {
				results[i] = inference.ClassifyResult{Token: inference.Token{Text: "technical"}}
			}
			return results, nil
		}}
		samples := []CalibrationSample{{Text: "No truth label"}}
		stats, err := valueFromResult[*CalibrationStats](CalibrateDomains(context.Background(), model, model, samples))
		if err != nil || stats.WithTruth != 0 {
			t.Fatalf("stats=%+v err=%v", stats, err)
		}
	})
	if !called {
		t.Fatal("CalibrateDomains was not exercised")
	}
}
