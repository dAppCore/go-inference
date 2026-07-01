// SPDX-Licence-Identifier: EUPL-1.2

package distill_test

import (
	"fmt"

	core "dappco.re/go"
	"dappco.re/go/inference/distill"
	"dappco.re/go/inference/eval"
)

// ExampleNewCheckpointMetadata builds the portable checkpoint sidecar
// shape from a driver's own config and running totals.
func ExampleNewCheckpointMetadata() {
	cfg := distill.Config{Temperature: 1, Loss: distill.LossKL}
	snapshot := distill.CheckpointSnapshot{
		Step:    100,
		Samples: 400,
		Tokens:  12000,
		Teacher: eval.Info{Architecture: "teacher-arch"},
		Student: eval.Info{Architecture: "student-arch"},
	}
	loss := distill.Loss{Value: 0.42, KL: 0.42}

	meta := distill.NewCheckpointMetadata("/ckpt/step-100", cfg, snapshot, loss, 1)
	fmt.Println("step:", meta.Step)
	fmt.Println("epoch:", meta.Epoch)
	fmt.Println("loss:", meta.Loss)
	fmt.Println("teacher:", meta.Teacher.Architecture)
	// Output:
	// step: 100
	// epoch: 1
	// loss: 0.42
	// teacher: teacher-arch
}

// ExampleSaveCheckpointMetadata and ExampleLoadCheckpointMetadata together
// show the round trip a driver's own training loop uses at its
// checkpoint cadence: save beside the checkpoint artifacts, then load it
// back (e.g. on --resume) to recover reproducible run state.
func ExampleSaveCheckpointMetadata() {
	baseResult := core.MkdirTemp("", "distill-example-*")
	if !baseResult.OK {
		panic("tempdir failed")
	}
	dir := baseResult.Value.(string)
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "step-1")

	if err := distill.SaveCheckpointMetadata(path, distill.CheckpointMetadata{Step: 1, Loss: 0.9}); err != nil {
		panic(err)
	}
	loaded, err := distill.LoadCheckpointMetadata(path)
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
	baseResult := core.MkdirTemp("", "distill-example-*")
	if !baseResult.OK {
		panic("tempdir failed")
	}
	dir := baseResult.Value.(string)
	defer core.RemoveAll(dir)

	meta, err := distill.LoadResumeMetadata(core.PathJoin(dir, "never-saved"))
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
	fmt.Println(distill.FormatStepDir(42))
	fmt.Println(distill.FormatStepDir(100000))
	// Output:
	// step-000042
	// step-100000
}
