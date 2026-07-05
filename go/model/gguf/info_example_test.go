// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import core "dappco.re/go"

// ExampleReadInfo reads a GGUF file's metadata without loading any tensor
// data into a concrete engine's array type.
func ExampleReadInfo() {
	dirResult := core.MkdirTemp("", "gguf-example-*")
	if !dirResult.OK {
		core.Println("tempdir failed")
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)

	path := core.Path(dir, "model.gguf")
	if err := writeMinimalExampleGGUF(path, "qwen3"); err != nil {
		core.Println("write failed")
		return
	}

	info, err := ReadInfo(path)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println(info.Architecture, info.TensorCount)
	// Output: qwen3 0
}
