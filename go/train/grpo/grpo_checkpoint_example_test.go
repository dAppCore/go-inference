// SPDX-Licence-Identifier: EUPL-1.2

package grpo_test

import (
	"fmt"

	core "dappco.re/go"
	"dappco.re/go/inference/eval"
	"dappco.re/go/inference/train/grpo"
)

// ExampleNewCheckpointMetadata builds the portable checkpoint sidecar
// shape from a driver's own config and running totals.
func ExampleNewCheckpointMetadata() {
	cfg := grpo.Config{GroupSize: 4, KLCoefficient: 0.1}
	snapshot := grpo.CheckpointSnapshot{
		Samples:  400,
		Rollouts: 1600,
		Policy:   eval.Info{Architecture: "policy-arch"},
	}
	update := grpo.Update{Step: 100, Epoch: 1, Loss: 0.42}

	meta := grpo.NewCheckpointMetadata("/ckpt/step-100", cfg, snapshot, update)
	fmt.Println("step:", meta.Step)
	fmt.Println("epoch:", meta.Epoch)
	fmt.Println("loss:", meta.Loss)
	fmt.Println("policy:", meta.Policy.Architecture)
	// Output:
	// step: 100
	// epoch: 1
	// loss: 0.42
	// policy: policy-arch
}

// ExampleSaveCheckpointMetadata and the paired load show the round trip a
// driver's own training loop uses at its checkpoint cadence: save beside
// the checkpoint artifacts, then load it back (e.g. on --resume) to
// recover reproducible run state.
func ExampleSaveCheckpointMetadata() {
	baseResult := core.MkdirTemp("", "grpo-example-*")
	if !baseResult.OK {
		panic("tempdir failed")
	}
	dir := baseResult.Value.(string)
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "step-1")

	if err := grpo.SaveCheckpointMetadata(path, grpo.CheckpointMetadata{Step: 1, Loss: 0.9}); err != nil {
		panic(err)
	}
	loaded, err := grpo.LoadCheckpointMetadata(path)
	if err != nil {
		panic(err)
	}
	fmt.Println("step:", loaded.Step)
	fmt.Println("loss:", loaded.Loss)
	// Output:
	// step: 1
	// loss: 0.9
}

// ExampleLoadResumeMetadata shows the soft-missing-file semantics a
// driver's own loop relies on for --resume: a path that was never saved
// yields (nil, nil), not an error, so a first run treats it as "start
// fresh".
func ExampleLoadResumeMetadata() {
	baseResult := core.MkdirTemp("", "grpo-example-*")
	if !baseResult.OK {
		panic("tempdir failed")
	}
	dir := baseResult.Value.(string)
	defer core.RemoveAll(dir)

	meta, err := grpo.LoadResumeMetadata(core.PathJoin(dir, "never-saved"))
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
	fmt.Println(grpo.FormatStepDir(42))
	fmt.Println(grpo.FormatStepDir(100000))
	// Output:
	// step-000042
	// step-100000
}
