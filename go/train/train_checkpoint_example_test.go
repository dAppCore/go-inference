// SPDX-Licence-Identifier: EUPL-1.2

package train_test

import (
	"fmt"

	core "dappco.re/go"
	"dappco.re/go/inference/eval"
	"dappco.re/go/inference/train"
)

// ExampleNewCheckpointMetadata builds the portable checkpoint sidecar
// shape from a driver's own config and running totals.
func ExampleNewCheckpointMetadata() {
	cfg := train.Config{BatchSize: 4, GradientAccumulationSteps: 2, LearningRate: 1e-4}
	snapshot := train.CheckpointSnapshot{
		Step:    100,
		Samples: 400,
		Loss:    0.42,
		Model:   eval.Info{Architecture: "gemma4"},
	}

	meta := train.NewCheckpointMetadata("/ckpt/step-100", cfg, snapshot, 1)
	fmt.Println("step:", meta.Step)
	fmt.Println("epoch:", meta.Epoch)
	fmt.Println("loss:", meta.Loss)
	fmt.Println("model:", meta.Model.Architecture)
	// Output:
	// step: 100
	// epoch: 1
	// loss: 0.42
	// model: gemma4
}

// ExampleSaveCheckpointMetadata and the paired load show the round trip a
// driver's own training loop uses at its checkpoint cadence: save beside
// the checkpoint artifacts, then load it back (e.g. on --resume) to
// recover reproducible run state.
func ExampleSaveCheckpointMetadata() {
	baseResult := core.MkdirTemp("", "train-example-*")
	if !baseResult.OK {
		panic("tempdir failed")
	}
	dir := baseResult.Value.(string)
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "step-1")

	if err := train.SaveCheckpointMetadata(path, train.CheckpointMetadata{Step: 1, Loss: 0.9}); err != nil {
		panic(err)
	}
	loaded, err := train.LoadCheckpointMetadata(path)
	if err != nil {
		panic(err)
	}
	fmt.Println("step:", loaded.Step)
	fmt.Println("loss:", loaded.Loss)
	// Output:
	// step: 1
	// loss: 0.9
}

// ExampleLoadCheckpointMetadata reads back a sidecar written by
// SaveCheckpointMetadata — the shape a driver's --resume flag loads to
// recover reproducible run state.
func ExampleLoadCheckpointMetadata() {
	baseResult := core.MkdirTemp("", "train-example-*")
	if !baseResult.OK {
		panic("tempdir failed")
	}
	dir := baseResult.Value.(string)
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "step-7")

	if err := train.SaveCheckpointMetadata(path, train.CheckpointMetadata{Step: 7, Epoch: 2}); err != nil {
		panic(err)
	}
	meta, err := train.LoadCheckpointMetadata(path)
	if err != nil {
		panic(err)
	}
	fmt.Println("step:", meta.Step)
	fmt.Println("epoch:", meta.Epoch)
	// Output:
	// step: 7
	// epoch: 2
}

// ExampleLoadResumeMetadata shows the soft-missing-file semantics a
// driver's own loop relies on for --resume: a path that was never saved
// yields (nil, nil), not an error, so a first run treats it as "start
// fresh".
func ExampleLoadResumeMetadata() {
	baseResult := core.MkdirTemp("", "train-example-*")
	if !baseResult.OK {
		panic("tempdir failed")
	}
	dir := baseResult.Value.(string)
	defer core.RemoveAll(dir)

	meta, err := train.LoadResumeMetadata(core.PathJoin(dir, "never-saved"))
	if err != nil {
		panic(err)
	}
	fmt.Println("meta is nil:", meta == nil)
	// Output:
	// meta is nil: true
}

// ExampleFormatStepDir shows the conventional zero-padded checkpoint
// directory name a driver's own loop joins onto its CheckpointDir.
func ExampleFormatStepDir() {
	fmt.Println(train.FormatStepDir(42))
	fmt.Println(train.FormatStepDir(100000))
	// Output:
	// step-000042
	// step-100000
}
