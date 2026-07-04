// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	core "dappco.re/go"
)

// ExampleAssembleBlocks splits a native-dtype snapshot into fixed-size blocks
// and reassembles it — the in-memory prefill-block round-trip. AssembleBlocks
// stitches the per-block native slabs back into the full-length layer tensors,
// recovering the original token count and raw byte payload exactly.
func ExampleAssembleBlocks() {
	source := exampleNativeLayerSnapshot()

	blocks, err := source.SplitBlocks(2)
	if err != nil {
		core.Println("split error:", err)
		return
	}

	assembled, err := AssembleBlocks(blocks)
	if err != nil {
		core.Println("assemble error:", err)
		return
	}
	core.Println("blocks:", len(blocks))
	core.Println("tokens:", len(assembled.Tokens))
	core.Println("key bytes recovered:", equalBytes(assembled.Layers[0].KeyBytes, source.Layers[0].KeyBytes))
	// Output:
	// blocks: 2
	// tokens: 4
	// key bytes recovered: true
}
