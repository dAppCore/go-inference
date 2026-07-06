// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// ExampleBackend shows the backend-agnostic decode contract: T input token embeddings
// (bf16 bytes) in, T output hidden states out — the seam the reactive engine drives
// without knowing whether the compute is native, metal, or a future rocm backend.
func ExampleBackend() {
	var b Backend = echoBackend{} // a real backend runs the transformer stack instead
	out, err := b.DecodeForward([][]byte{{1, 2}, {3, 4}})
	if err != nil {
		return
	}
	core.Println(len(out))
	// Output: 2
}
