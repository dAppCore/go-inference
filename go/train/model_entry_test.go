// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the Model-bound entries RunSSDModel / RunSFTModel, driven with a
// fake inference.TextModel that also implements engine.TrainerModel and
// inference.PromptCacheWarmer. This proves the neutral wiring end to end: SSD
// rides model.Generate + the probed PromptCacheWarmer; SFT probes the
// engine.TrainerModel seam, opens a trainer, and runs the loop — no engine type
// in either caller.

package train

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
	"dappco.re/go/inference/train/dataset"
)

// fakeTextModel is a minimal inference.TextModel: Generate yields the prompt
// echoed as a single token. It optionally forwards a trainer (engine.TrainerModel)
// and a prompt-cache warmer (inference.PromptCacheWarmer).
type fakeTextModel struct {
	trainer     engine.Trainer
	warmCalls   *int
	lastPrompts *[]string
}

func (m *fakeTextModel) Generate(_ context.Context, prompt string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	if m.lastPrompts != nil {
		*m.lastPrompts = append(*m.lastPrompts, prompt)
	}
	return func(yield func(inference.Token) bool) {
		yield(inference.Token{ID: 1, Text: "echo:" + prompt})
	}
}

func (m *fakeTextModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}
func (m *fakeTextModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}
func (m *fakeTextModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}
func (m *fakeTextModel) ModelType() string                  { return "fake" }
func (m *fakeTextModel) Info() inference.ModelInfo          { return inference.ModelInfo{Architecture: "fake"} }
func (m *fakeTextModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *fakeTextModel) Err() core.Result                   { return core.Ok(nil) }
func (m *fakeTextModel) Close() core.Result                 { return core.Ok(nil) }

// OpenTrainer forwards the fake trainer — the engine.TrainerModel seam
// RunSFTModel probes for.
func (m *fakeTextModel) OpenTrainer(inference.TrainingConfig) (engine.Trainer, error) {
	if m.trainer == nil {
		return nil, core.NewError("fake: no trainer")
	}
	return m.trainer, nil
}

// WarmPromptCache records the warm call — the inference.PromptCacheWarmer seam
// RunSSDModel probes for the kernel lane.
func (m *fakeTextModel) WarmPromptCache(_ context.Context, _ string) error {
	if m.warmCalls != nil {
		*m.warmCalls = *m.warmCalls + 1
	}
	return nil
}

// TestRunSSDModel_RidesGenerateAndWarmer asserts RunSSDModel builds the runner
// from model.Generate and the probed PromptCacheWarmer: the kernel warms once
// and generation echoes each prompt.
func TestRunSSDModel_RidesGenerateAndWarmer(t *testing.T) {
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

// TestRunSFTModel_ProbesTrainerAndRuns asserts RunSFTModel probes the
// engine.TrainerModel seam, opens the trainer, runs the loop, and closes the
// trainer on return.
func TestRunSFTModel_ProbesTrainerAndRuns(t *testing.T) {
	trainer := &fakeTrainer{}
	model := &fakeTextModel{trainer: trainer}
	cfg := SFTConfig{Config: Config{BatchSize: 1, GradientAccumulationSteps: 1, Epochs: 1}, LoRA: inference.LoRAConfig{Rank: 8, Alpha: 16}}
	result, err := RunSFTModel(context.Background(), model, runeEncode, sftDataset(3), cfg)
	if err != nil {
		t.Fatalf("RunSFTModel: %v", err)
	}
	if result.Steps != 3 {
		t.Fatalf("steps = %d, want 3", result.Steps)
	}
	if !trainer.closed {
		t.Fatalf("trainer was not closed on return")
	}
}

// TestRunSFTModel_NoTrainerSupport asserts a model without the engine.TrainerModel
// seam gets a clear error rather than a panic — the honest "engine does not
// support training" boundary.
func TestRunSFTModel_NoTrainerSupport(t *testing.T) {
	model := &fakeTextModel{} // OpenTrainer present but the caller sees the seam; use a non-trainer model
	// Wrap so the type assertion to engine.TrainerModel fails.
	var plain inference.TextModel = plainModel{model}
	_, err := RunSFTModel(context.Background(), plain, runeEncode, sftDataset(1), SFTConfig{Config: Config{Epochs: 1}})
	if err == nil {
		t.Fatalf("expected an error for a model without the trainer seam")
	}
}

// plainModel wraps a fakeTextModel to hide OpenTrainer — the not-trainable path.
type plainModel struct{ inner *fakeTextModel }

func (p plainModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return p.inner.Generate(ctx, prompt, opts...)
}
func (p plainModel) Chat(ctx context.Context, msgs []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return p.inner.Chat(ctx, msgs, opts...)
}
func (p plainModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	return p.inner.Classify(ctx, prompts, opts...)
}
func (p plainModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	return p.inner.BatchGenerate(ctx, prompts, opts...)
}
func (p plainModel) ModelType() string                  { return p.inner.ModelType() }
func (p plainModel) Info() inference.ModelInfo          { return p.inner.Info() }
func (p plainModel) Metrics() inference.GenerateMetrics { return p.inner.Metrics() }
func (p plainModel) Err() core.Result                   { return p.inner.Err() }
func (p plainModel) Close() core.Result                 { return p.inner.Close() }

var _ inference.TextModel = plainModel{}
var _ dataset.Resetter = (*dataset.SliceDataset)(nil)
