// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	core "dappco.re/go"
	basegguf "dappco.re/go/inference/model/gguf"
)

// gemma3KQuantType returns the K-quant (256-element-superblock) GGML type a
// requested format maps its bulk projection weights to, and whether the format
// is a K-quant at all. The pure 32-block formats (q8_0) report ok=false — every
// tensor is that one type, gated only on the 32-element block divisibility.
func gemma3KQuantType(format basegguf.QuantizeFormat) (uint32, bool) {
	switch format {
	case basegguf.QuantizeQ4_K_M, basegguf.QuantizeQ4_K:
		return basegguf.TensorTypeQ4K, true
	case basegguf.QuantizeQ5_K_M, basegguf.QuantizeQ5_K:
		return basegguf.TensorTypeQ5K, true
	case basegguf.QuantizeQ6_K:
		return basegguf.TensorTypeQ6K, true
	case basegguf.QuantizeQ3_K_M, basegguf.QuantizeQ3_K:
		return basegguf.TensorTypeQ3K, true
	case basegguf.QuantizeQ2_K_M: // QuantizeQ2_K is the same "q2_k" string
		return basegguf.TensorTypeQ2K, true
	default:
		return 0, false
	}
}

// gemma3TensorType returns the GGML tensor type gemma-3's export policy assigns
// to a canonical tensor of inner (ne0) row length rowLen under format.
//
// Two constraints shape the policy, both about producing a file llama.cpp
// actually loads rather than matching any oracle's byte-exact recipe:
//
//   - RMS-norm weights (every *norm.weight) stay F32. They are 1-D, carry the
//     folded gemma "(1 + weight)" bias, and are never block-quantised by
//     llama.cpp.
//
//   - A K-quant tiles rows in 256-element superblocks, so it is only valid when
//     ne0 is a multiple of 256. gemma-3-1B's hidden size is 1152 (256·4.5), so
//     the many tensors whose inner dimension is the hidden size cannot be a
//     K-quant; they fall back to Q8_0 (32-element blocks, ne0 divisible by 32).
//     A row divisible by neither block size stays F32. This mirrors what
//     llama.cpp's own quantiser does for such models (it drops those rows to a
//     32-block quant) — the file loads and generates; it is simply larger than
//     a hypothetical all-256-divisible model's same-name recipe.
//
//     gemma3TensorType(gguf.QuantizeQ4_K_M, "blk.0.ffn_down.weight", 6912) // Q4_K (256 | 6912)
//     gemma3TensorType(gguf.QuantizeQ4_K_M, "blk.0.attn_q.weight", 1152)   // Q8_0 (256 ∤ 1152, 32 | 1152)
//     gemma3TensorType(gguf.QuantizeQ4_K_M, "blk.0.attn_norm.weight", 1152)// F32  (norm)
func gemma3TensorType(format basegguf.QuantizeFormat, canonical string, rowLen uint64) uint32 {
	if core.HasSuffix(canonical, "norm.weight") {
		return basegguf.TensorTypeF32
	}
	if kType, isK := gemma3KQuantType(format); isK && rowLen%256 == 0 {
		return kType
	}
	if rowLen%32 == 0 && rowLen > 0 {
		return basegguf.TensorTypeQ8_0
	}
	return basegguf.TensorTypeF32
}
