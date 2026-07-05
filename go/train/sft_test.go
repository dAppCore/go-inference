// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the engine-neutral SFT loop, driven with a fake engine.Trainer
// (records steps + save paths, returns a falling loss) and injected encode/gen
// hooks. No model, no Metal — the loop drives the Trainer seam, so a fake
// Trainer exercises the whole step/checkpoint/eval/save orchestration.

package train

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/train/dataset"
	coreio "dappco.re/go/io"
)

// fakeTrainer is a stand-in engine.Trainer: it counts optimiser updates, returns
// a monotonically falling loss, records every Save path (creating the dir like a
// real Save does), and flags Close.
type fakeTrainer struct {
	steps  int
	saved  []string
	closed bool
}

func (f *fakeTrainer) Step(inference.Batch) (float64, error) {
	f.steps++
	return 1.0 / float64(f.steps), nil
}

func (f *fakeTrainer) StepAccumulated([]inference.Batch) (float64, error) {
	f.steps++
	return 1.0 / float64(f.steps), nil
}

func (f *fakeTrainer) Loss(inference.Batch) (float64, error) { return 1.0, nil }

func (f *fakeTrainer) Save(path string) error {
	f.saved = append(f.saved, path)
	if res := core.MkdirAll(path, core.FileMode(0o755)); !res.OK {
		if err, ok := res.Value.(error); ok {
			return err
		}
	}
	return nil
}

func (f *fakeTrainer) Close() error { f.closed = true; return nil }

// runeEncode tokenises text as one id per rune — deterministic, dependency-free,
// and long enough for the ≥2-token causal-target floor on any non-trivial text.
func runeEncode(text string) []int32 {
	runes := []rune(text)
	ids := make([]int32, len(runes))
	for i, r := range runes {
		ids[i] = int32(r)
	}
	return ids
}

func sftDataset(n int) inference.DatasetStream {
	samples := make([]dataset.Sample, n)
	for i := range samples {
		samples[i] = dataset.Sample{Prompt: "prompt", Response: "a helpful response body"}
	}
	return dataset.NewSliceDataset(samples)
}

// TestRunSFT_StepsCheckpointsAndSaves drives four samples at batch=1/accum=1 and
// asserts four optimiser steps, two checkpoints (every 2 steps), eval captures,
// and a saved final adapter.
func TestRunSFT_StepsCheckpointsAndSaves(t *testing.T) {
	dir := t.TempDir()
	trainer := &fakeTrainer{}
	cfg := SFTConfig{
		Config: Config{
			BatchSize:                 1,
			GradientAccumulationSteps: 1,
			Epochs:                    1,
			CheckpointDir:             dir,
			CheckpointEvery:           2,
			EvalEvery:                 2,
			EvalPrompts:               []string{"probe"},
			EvalMaxTokens:             8,
		},
		SavePath: core.PathJoin(dir, "adapter"),
	}
	gen := func(_ context.Context, prompt string, _ int, _ float32) (string, error) { return "gen:" + prompt, nil }

	result, err := RunSFT(context.Background(), trainer, runeEncode, gen, sftDataset(4), cfg)
	if err != nil {
		t.Fatalf("RunSFT: %v", err)
	}
	if result.Steps != 4 || result.Samples != 4 {
		t.Fatalf("steps=%d samples=%d, want 4/4", result.Steps, result.Samples)
	}
	if len(result.Checkpoints) != 2 {
		t.Fatalf("checkpoints = %d, want 2", len(result.Checkpoints))
	}
	if result.AdapterPath != cfg.SavePath {
		t.Fatalf("adapter path = %q, want %q", result.AdapterPath, cfg.SavePath)
	}
	if result.LastLoss <= 0 || result.LastLoss >= result.Losses[0] {
		t.Fatalf("loss did not fall: first=%f last=%f", result.Losses[0], result.LastLoss)
	}
	// Eval capture landed at the checkpoint cadence.
	read, err := coreio.Local.Read(core.PathJoin(dir, "captures.jsonl"))
	if err != nil {
		t.Fatalf("captures read: %v", err)
	}
	if core.Index(read, "gen:probe") < 0 {
		t.Fatalf("eval capture missing: %q", read)
	}
}

// TestRunSFT_GradAccumOneStepPerAccumWindow asserts gradient accumulation groups
// micro-batches: batch=1, accum=2 over 4 samples yields 2 optimiser steps.
func TestRunSFT_GradAccumOneStepPerAccumWindow(t *testing.T) {
	trainer := &fakeTrainer{}
	cfg := SFTConfig{Config: Config{BatchSize: 1, GradientAccumulationSteps: 2, Epochs: 1}}
	result, err := RunSFT(context.Background(), trainer, runeEncode, nil, sftDataset(4), cfg)
	if err != nil {
		t.Fatalf("RunSFT: %v", err)
	}
	if result.OptimizerSteps != 2 {
		t.Fatalf("optimizer steps = %d, want 2 (batch=1, accum=2, 4 samples)", result.OptimizerSteps)
	}
	if trainer.steps != 2 {
		t.Fatalf("trainer updates = %d, want 2", trainer.steps)
	}
}

// TestRunSFT_MultiEpochResetsDataset asserts a two-epoch run replays the dataset
// (SliceDataset implements Reset) — 2 samples × 2 epochs = 4 processed.
func TestRunSFT_MultiEpochResetsDataset(t *testing.T) {
	trainer := &fakeTrainer{}
	cfg := SFTConfig{Config: Config{BatchSize: 1, GradientAccumulationSteps: 1, Epochs: 2}}
	result, err := RunSFT(context.Background(), trainer, runeEncode, nil, sftDataset(2), cfg)
	if err != nil {
		t.Fatalf("RunSFT: %v", err)
	}
	if result.Samples != 4 {
		t.Fatalf("samples = %d, want 4 (2 × 2 epochs)", result.Samples)
	}
	if result.Epochs != 2 {
		t.Fatalf("epochs = %d, want 2", result.Epochs)
	}
}

// TestRunSFT_EmptyDatasetErrors asserts a dataset that produces no trainable
// sequences is a loud error, not a silent no-op.
func TestRunSFT_EmptyDatasetErrors(t *testing.T) {
	trainer := &fakeTrainer{}
	cfg := SFTConfig{Config: Config{BatchSize: 1, GradientAccumulationSteps: 1, Epochs: 1}}
	_, err := RunSFT(context.Background(), trainer, runeEncode, nil, dataset.NewSliceDataset(nil), cfg)
	if err == nil {
		t.Fatalf("expected an error for an empty dataset")
	}
}

// TestRunSFT_NilTrainerErrors asserts a nil trainer is rejected up front.
func TestRunSFT_NilTrainerErrors(t *testing.T) {
	_, err := RunSFT(context.Background(), nil, runeEncode, nil, sftDataset(1), SFTConfig{})
	if err == nil {
		t.Fatalf("expected an error for a nil trainer")
	}
}
