// SPDX-Licence-Identifier: EUPL-1.2

// Runnable examples for lora.go — kept separate from lora_test.go so the
// godoc-attached usage snippets stay readable.

package lora

import core "dappco.re/go"

// ExampleAdapterRef_ID shows the deterministic id: the same Name+Path always
// derives the same id, so every node minting refs from the same
// --adapter-paths agrees on the id without coordination.
func ExampleAdapterRef_ID() {
	a := AdapterRef{Name: "support-tone", Path: "/adapters/support"}
	b := AdapterRef{Name: "support-tone", Path: "/adapters/support"}
	core.Println(a.ID() == b.ID())

	// Output:
	// true
}
