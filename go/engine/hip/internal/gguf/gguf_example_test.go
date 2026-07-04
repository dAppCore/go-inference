package gguf

import core "dappco.re/go"

func ExampleFileTypeName() { core.Println(FileTypeName(15)) /* Output: Q4_K_M */ }
func ExampleReadMetadata() {
	_, err := ReadMetadata(core.PathJoin(core.TempDir(), "missing.gguf"))
	core.Println(err != nil) /* Output: true */
}
