// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "fmt"

// Example shows a gemma-4 per-layer attention weight mapping to its canonical
// GGUF name and its safetensors shape reversing to GGUF ne[] order.
func Example_gemma4CanonicalTensorName() {
	name, _ := gemma4CanonicalTensorName("language_model.model.layers.4.self_attn.q_proj.weight")
	shape := gemma4GGUFShape([]uint64{2048, 1536})
	fmt.Println(name, shape)
	// Output: blk.4.attn_q.weight [1536 2048]
}
