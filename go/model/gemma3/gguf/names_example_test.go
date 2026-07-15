// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import core "dappco.re/go"

// Example_gemma3CanonicalTensorName shows an HF gemma-3 tensor name mapped to
// the canonical GGUF name llama.cpp looks the text stack up by (unexported, so
// the example documents the mapping without widening the package's API).
func Example_gemma3CanonicalTensorName() {
	name, _ := gemma3CanonicalTensorName("model.layers.7.self_attn.q_proj.weight")
	core.Println(name)
	// Output:
	// blk.7.attn_q.weight
}
