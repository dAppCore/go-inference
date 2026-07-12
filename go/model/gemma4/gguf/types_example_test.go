// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"fmt"

	basegguf "dappco.re/go/inference/model/gguf"
)

// Example shows the q4_k_m policy bumping ffn_down to Q6_K on a use_more_bits
// layer while a plain projection weight stays Q4_K, and q8_0's pure policy
// leaving that same tensor at the Q8_0 bulk on every layer.
func Example_gemma4TensorType() {
	fmt.Println(
		gemma4TensorType(basegguf.QuantizeQ4_K_M, "blk.6.ffn_down.weight", 6, 35) == basegguf.TensorTypeQ6K,
		gemma4TensorType(basegguf.QuantizeQ4_K_M, "blk.5.ffn_down.weight", 5, 35) == basegguf.TensorTypeQ4K,
		gemma4TensorType(basegguf.QuantizeQ8_0, "blk.6.ffn_down.weight", 6, 35) == basegguf.TensorTypeQ8_0,
	)
	// Output: true true true
}
