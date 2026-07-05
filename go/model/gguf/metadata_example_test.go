// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import core "dappco.re/go"

// ExampleMetadata reads a .gguf file's raw key/value metadata map.
func ExampleMetadata() {
	dirResult := core.MkdirTemp("", "gguf-metadata-example-*")
	if !dirResult.OK {
		core.Println("tempdir failed")
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)

	path := core.Path(dir, "model.gguf")
	if err := writeMinimalExampleGGUF(path, "gemma3"); err != nil {
		core.Println("write failed")
		return
	}

	meta, err := Metadata(path)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println(meta["general.architecture"])
	// Output: gemma3
}
