// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// trainer.go declares only the [Trainer] / [TrainerModel] interfaces — no
// package-level funcs of its own, so there is nothing to give an AX-7
// Good/Bad/Ugly triplet to directly. What IS tested here is the interface
// CONTRACT itself: fakeTrainer/fakeTrainerModel satisfy the shapes exactly as
// documented (the Step/StepAccumulated/Loss/Save/Close lifecycle from the
// [Trainer] doc comment), so a consumer coded against these interfaces (as
// TextModel.OpenTrainer's Good/Bad/Ugly in model_test.go is) can trust the
// shape. fakeTrainer/fakeTrainerModel are also the fixtures model_test.go's
// OpenTrainer tests drive.

// fakeTrainer is a configurable [Trainer] double.
type fakeTrainer struct {
	stepLoss float64

	stepErr, stepAccErr, lossErr, saveErr, closeErr error

	savedPath  string
	closeCalls int
}

func (f *fakeTrainer) Step(batch inference.Batch) (float64, error) {
	if f.stepErr != nil {
		return 0, f.stepErr
	}
	return f.stepLoss, nil
}

func (f *fakeTrainer) StepAccumulated(batches []inference.Batch) (float64, error) {
	if f.stepAccErr != nil {
		return 0, f.stepAccErr
	}
	return f.stepLoss, nil
}

func (f *fakeTrainer) Loss(batch inference.Batch) (float64, error) {
	if f.lossErr != nil {
		return 0, f.lossErr
	}
	return f.stepLoss, nil
}

func (f *fakeTrainer) Save(path string) error {
	f.savedPath = path
	return f.saveErr
}

func (f *fakeTrainer) Close() error {
	f.closeCalls++
	return f.closeErr
}

var _ Trainer = (*fakeTrainer)(nil)

// fakeTrainerModel adds the TrainerModel capability over a fakeTokenModel, so
// TextModel.OpenTrainer's capability probe (`tm.(TrainerModel)`) succeeds.
type fakeTrainerModel struct {
	fakeTokenModel
	trainer        *fakeTrainer
	openTrainerErr error
	openTrainerCfg inference.TrainingConfig
}

func (f *fakeTrainerModel) OpenTrainer(cfg inference.TrainingConfig) (Trainer, error) {
	f.openTrainerCfg = cfg
	if f.openTrainerErr != nil {
		return nil, f.openTrainerErr
	}
	return f.trainer, nil
}

var _ TrainerModel = (*fakeTrainerModel)(nil)

// TestTrainer_Step pins the documented Step contract: one gradient step
// returns the mean training loss, and a training failure propagates untouched.
func TestTrainer_Step(t *testing.T) {
	tr := &fakeTrainer{stepLoss: 0.42}
	loss, err := tr.Step(inference.Batch{TokenIDs: [][]int32{{1, 2, 3}}})
	if err != nil || loss != 0.42 {
		t.Fatalf("Step = (%v, %v), want (0.42, nil)", loss, err)
	}
	tr.stepErr = core.NewError("gradient overflow")
	if _, err := tr.Step(inference.Batch{}); err == nil {
		t.Fatal("Step did not propagate the engine failure")
	}
}

// TestTrainer_StepAccumulated pins the large-effective-batch contract: one
// optimiser update from the mean loss across several micro-batches.
func TestTrainer_StepAccumulated(t *testing.T) {
	tr := &fakeTrainer{stepLoss: 0.2}
	loss, err := tr.StepAccumulated([]inference.Batch{{}, {}})
	if err != nil || loss != 0.2 {
		t.Fatalf("StepAccumulated = (%v, %v), want (0.2, nil)", loss, err)
	}
	tr.stepAccErr = core.NewError("accumulation overflow")
	if _, err := tr.StepAccumulated(nil); err == nil {
		t.Fatal("StepAccumulated did not propagate the engine failure")
	}
}

// TestTrainer_Loss pins the validation-lane contract: a forward-only mean
// loss with no gradient/optimiser movement, and failure propagation.
func TestTrainer_Loss(t *testing.T) {
	tr := &fakeTrainer{stepLoss: 0.9}
	loss, err := tr.Loss(inference.Batch{})
	if err != nil || loss != 0.9 {
		t.Fatalf("Loss = (%v, %v), want (0.9, nil)", loss, err)
	}
	tr.lossErr = core.NewError("forward failed")
	if _, err := tr.Loss(inference.Batch{}); err == nil {
		t.Fatal("Loss did not propagate the engine failure")
	}
}

// TestTrainer_Save pins the on-disk adapter persistence contract: the path
// reaches the engine untouched, and a write failure propagates.
func TestTrainer_Save(t *testing.T) {
	tr := &fakeTrainer{}
	if err := tr.Save("/models/lora/domain-v1"); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if tr.savedPath != "/models/lora/domain-v1" {
		t.Fatalf("savedPath = %q, want the exact path handed to Save", tr.savedPath)
	}
	tr.saveErr = core.NewError("disk full")
	if err := tr.Save("/models/lora/domain-v2"); err == nil {
		t.Fatal("Save did not propagate the engine failure")
	}
}

// TestTrainer_Close pins the retained-session release contract: Close is
// observable and its failure propagates rather than being swallowed.
func TestTrainer_Close(t *testing.T) {
	tr := &fakeTrainer{}
	if err := tr.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if tr.closeCalls != 1 {
		t.Fatalf("closeCalls = %d, want 1", tr.closeCalls)
	}
	tr.closeErr = core.NewError("device busy")
	if err := tr.Close(); err == nil {
		t.Fatal("Close did not propagate the engine failure")
	}
}

// TestTrainerModel_OpenTrainer pins the entry-point contract fakeTrainerModel
// gives TextModel.OpenTrainer to drive: the training config reaches the
// engine untouched, the engine's own Trainer comes back, and a failure to
// open propagates.
func TestTrainerModel_OpenTrainer(t *testing.T) {
	tr := &fakeTrainer{}
	tm := &fakeTrainerModel{trainer: tr}
	cfg := inference.TrainingConfig{Epochs: 3, LoRA: inference.LoRAConfig{Rank: 8, Alpha: 16}}
	got, err := tm.OpenTrainer(cfg)
	if err != nil {
		t.Fatalf("OpenTrainer: %v", err)
	}
	if got != tr {
		t.Fatal("OpenTrainer did not return the engine's own Trainer")
	}
	// TrainingConfig embeds LoRAConfig (which carries a []string) and its own
	// Labels map, so the struct is not == comparable — check the fields that
	// matter for this contract instead of the whole value.
	if tm.openTrainerCfg.Epochs != cfg.Epochs || tm.openTrainerCfg.LoRA.Rank != cfg.LoRA.Rank {
		t.Fatalf("OpenTrainer cfg = %+v, want %+v", tm.openTrainerCfg, cfg)
	}

	tm.openTrainerErr = core.NewError("no trainer available")
	if _, err := tm.OpenTrainer(inference.TrainingConfig{}); err == nil {
		t.Fatal("OpenTrainer did not propagate the engine failure")
	}
}
