// SPDX-Licence-Identifier: EUPL-1.2

package checkpoint

import (
	core "dappco.re/go"
)

// ExampleSave demonstrates writing a checkpoint sidecar, creating any
// missing parent directory along the way.
func ExampleSave() {
	baseResult := core.MkdirTemp("", "checkpoint-example-*")
	if !baseResult.OK {
		panic("tempdir failed")
	}
	dir := baseResult.Value.(string)
	defer core.RemoveAll(dir)

	sidecarPath := core.PathJoin(dir, "meta.json")
	if err := Save(sidecarPath, sample{Version: 1, Step: 5, Name: "x"}); err != nil {
		core.Println(err)
		return
	}
	core.Println(core.ReadFile(sidecarPath).OK)
	// Output:
	// true
}

// ExampleLoad demonstrates reading back what Save wrote.
func ExampleLoad() {
	baseResult := core.MkdirTemp("", "checkpoint-example-*")
	if !baseResult.OK {
		panic("tempdir failed")
	}
	dir := baseResult.Value.(string)
	defer core.RemoveAll(dir)

	sidecarPath := core.PathJoin(dir, "meta.json")
	if err := Save(sidecarPath, sample{Version: 1, Step: 7, Name: "round-trip"}); err != nil {
		panic(err)
	}

	got, err := Load[sample](sidecarPath)
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(got.Step)
	core.Println(got.Name)
	// Output:
	// 7
	// round-trip
}

// ExampleLoadResume demonstrates the soft-missing-file semantics a --resume
// flow relies on: a sidecar that was never saved resumes as (nil, nil)
// rather than an error.
func ExampleLoadResume() {
	baseResult := core.MkdirTemp("", "checkpoint-example-*")
	if !baseResult.OK {
		panic("tempdir failed")
	}
	dir := baseResult.Value.(string)
	defer core.RemoveAll(dir)

	got, err := LoadResume[sample](core.PathJoin(dir, "never-saved.json"))
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(got == nil)
	// Output:
	// true
}

// ExampleFormatStepDir demonstrates the "step-NNNNNN" checkpoint dirname
// convention shared by every domain package's own checkpoint sidecars.
func ExampleFormatStepDir() {
	core.Println(FormatStepDir(42))
	// Output:
	// step-000042
}
