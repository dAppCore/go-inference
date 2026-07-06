// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the Model-bound entry RunSSDModel, driven with the fake
// inference.TextModel declared in sft_model_test.go. This proves the neutral
// wiring end to end: SSD rides model.Generate and the capability-probed
// PromptCacheWarmer — no engine type in the caller.

package train

import (
	"context"
	"testing"
)

// Good: a model implementing inference.PromptCacheWarmer gets the kernel
// warmed exactly once, and every sample echoes its (kernel-prefixed)
// generation prompt.
func TestSsdModel_RunSSDModel_Good(t *testing.T) {
	warms := 0
	prompts := []string{}
	model := &fakeTextModel{warmCalls: &warms, lastPrompts: &prompts}
	cfg := SSDConfig{SampleTemperature: 0.7, KernelPrefix: "K::", DisableCapture: true, FilterShortestPercent: 0}
	result, err := RunSSDModel(context.Background(), model, ssdDataset("q1", "q2"), cfg, nil)
	if err != nil {
		t.Fatalf("RunSSDModel: %v", err)
	}
	if warms != 1 {
		t.Fatalf("warm calls = %d, want 1", warms)
	}
	if len(result.Samples) != 2 || result.Samples[0].Response != "echo:K::q1" {
		t.Fatalf("samples = %+v", result.Samples)
	}
}

// Bad: a nil model is rejected up front rather than reaching a nil-pointer
// dereference on the first Generate call.
func TestSsdModel_RunSSDModel_Bad(t *testing.T) {
	_, err := RunSSDModel(context.Background(), nil, ssdDataset("q1"), SSDConfig{SampleTemperature: 0.7}, nil)
	if err == nil {
		t.Fatalf("expected an error for a nil model")
	}
}

// Ugly: a model that does NOT implement inference.PromptCacheWarmer (the
// capability probe misses) still runs correctly — the kernel prefix rides
// the generation prompt itself rather than the (absent) warm call, per the
// "still correct, just not cached" fallback ssd.go documents.
func TestSsdModel_RunSSDModel_Ugly(t *testing.T) {
	prompts := []string{}
	inner := &fakeTextModel{lastPrompts: &prompts}
	model := plainModel{inner: inner} // no WarmPromptCache method
	cfg := SSDConfig{SampleTemperature: 0.7, KernelPrefix: "K::", DisableCapture: true}
	result, err := RunSSDModel(context.Background(), model, ssdDataset("q1"), cfg, nil)
	if err != nil {
		t.Fatalf("RunSSDModel with a non-warming model: %v", err)
	}
	if len(result.Samples) != 1 || result.Samples[0].Response != "echo:K::q1" {
		t.Fatalf("samples = %+v, want the kernel prefix still applied via the prompt", result.Samples)
	}
}
