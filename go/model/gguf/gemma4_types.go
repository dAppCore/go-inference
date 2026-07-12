// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import core "dappco.re/go"

// gemma4UseMoreBits reproduces llama.cpp's use_more_bits selector (the rule
// llama_tensor_get_type applies to attn_v and ffn_down under the *_K_M "medium"
// quant policy): the first eighth of layers, the last eighth, and every third
// layer in between are bumped to the higher-precision type. Verbatim from
// llama.cpp so the per-layer Q6_K selection matches the oracle exactly (for
// n=35 this yields layers {0,1,2,3,6,9,12,15,18,21,24,27,30,31,32,33,34}).
//
//	gemma4UseMoreBits(6, 35)  // true
//	gemma4UseMoreBits(5, 35)  // false
func gemma4UseMoreBits(layerIndex, layerCount int) bool {
	return layerIndex < layerCount/8 ||
		layerIndex >= 7*layerCount/8 ||
		(layerIndex-layerCount/8)%3 == 2
}

// gemma4BulkType returns the GGML type format quantises the bulk of the
// projection weights to, before gemma4TensorType layers any per-tensor
// override on top. Q8_0 and Q6_K carry no per-tensor override at all in
// llama.cpp's own quantize table (llama_tensor_get_type_impl has no
// LLAMA_FTYPE_MOSTLY_Q8_0 / _Q6_K case) — every quantisable tensor is the
// bulk type for those two, oracle-confirmed for q8_0 against the on-disk
// unsloth gemma-4-E2B-it-Q8_0.gguf (uniform Q8_0, 247/247 quantised
// tensors). Q5_K_M and Q3_K_M are llama.cpp's "_M" mixed-precision recipes;
// gemma4AttnVType / gemma4FfnDownType / gemma4AttnOutputType layer their
// overrides on top of this bulk type. Q4_K_M is the existing unsloth-oracle-
// matched policy.
//
// Reference: llama.cpp src/llama-quant.cpp, llama_ftype_get_default_type
// (ggml-org/llama.cpp@master, fetched 2026-07-12).
func gemma4BulkType(format QuantizeFormat) uint32 {
	switch format {
	case QuantizeQ8_0:
		return TensorTypeQ8_0
	case QuantizeQ6_K:
		return ggufTensorTypeQ6K
	case QuantizeQ5_K_M:
		return ggufTensorTypeQ5K
	case QuantizeQ3_K_M:
		return ggufTensorTypeQ3K
	default: // QuantizeQ4_K_M
		return ggufTensorTypeQ4K
	}
}

// gemma4AttnVType returns attn_v.weight's type for format at layerIndex of
// layerCount. llama.cpp's category_is_attn_v branch bumps this
// attention-sensitive tensor above the bulk type on specific layers for
// every "_M" mixed recipe it defines: Q4_K_M and Q5_K_M share the identical
// use_more_bits(i, n) selector (gemma4UseMoreBits); Q3_K_M instead has its
// own fixed "first two layers" rule, independent of layerCount. Q8_0/Q6_K
// have no override — always bulk.
func gemma4AttnVType(format QuantizeFormat, bulk uint32, layerIndex, layerCount int) uint32 {
	switch format {
	case QuantizeQ4_K_M, QuantizeQ5_K_M:
		if gemma4UseMoreBits(layerIndex, layerCount) {
			return ggufTensorTypeQ6K
		}
		return bulk
	case QuantizeQ3_K_M:
		if layerIndex < 2 {
			return ggufTensorTypeQ5K
		}
		return ggufTensorTypeQ4K
	default:
		return bulk
	}
}

// gemma4FfnDownType returns ffn_down.weight's type for format at layerIndex
// of layerCount. Same use_more_bits(i, n) bump as attn_v for Q4_K_M/Q5_K_M;
// Q3_K_M instead bumps the first layerCount/16 layers to Q5_K and every
// other layer to Q4_K — llama.cpp's Q3_K_M FFN_DOWN branch on a non-Falcon
// arch (gemma is never Falcon, so its use_more_bits fallback never applies
// here: the tensor is always bumped away from the Q3_K bulk).
func gemma4FfnDownType(format QuantizeFormat, bulk uint32, layerIndex, layerCount int) uint32 {
	switch format {
	case QuantizeQ4_K_M, QuantizeQ5_K_M:
		if gemma4UseMoreBits(layerIndex, layerCount) {
			return ggufTensorTypeQ6K
		}
		return bulk
	case QuantizeQ3_K_M:
		if layerIndex < layerCount/16 {
			return ggufTensorTypeQ5K
		}
		return ggufTensorTypeQ4K
	default:
		return bulk
	}
}

// gemma4AttnOutputType returns attn_output.weight's type for format.
// llama.cpp's Q3_K_M ATTENTION_OUTPUT branch bumps this tensor to Q4_K on
// every layer (non-Falcon arch, non-8-expert model — gemma-4-E2B is both);
// every other format leaves it at bulk.
func gemma4AttnOutputType(format QuantizeFormat, bulk uint32) uint32 {
	if format == QuantizeQ3_K_M {
		return ggufTensorTypeQ4K
	}
	return bulk
}

// gemma4TensorType returns the GGML tensor type format's policy assigns to a
// canonical gemma-4 tensor.
//
// The exceptions below hold for every format, independent of the requested
// quant: llama.cpp's tensor_allows_quantization excludes RMS-norm weights,
// the per-layer input gate/projection/output-scale (its AltUp/Laurel
// mechanism), and per_layer_model_proj from quantisation entirely, and
// rope_freqs is a computed tensor (gemma4RopeFreqsTensor) with no source
// counterpart to quantise:
//
//   - RMS-norm weights, the per-layer input gate/projection, the per-layer
//     output scale, and rope_freqs stay F32;
//   - per_layer_model_proj stays BF16 (passed through from the source dtype).
//
// Everything else follows format's bulk type (gemma4BulkType) with these
// per-tensor overrides layered on top:
//
//   - per_layer_token_embd is bumped to Q5_K under q4_k_m specifically — the
//     unsloth gemma-4-E2B-it-Q4_K_M oracle's own choice, with no analogue in
//     llama.cpp's own quantize table for any format (not extended to the
//     other formats: q8_0's own on-disk oracle keeps this tensor at bulk,
//     the one case that can be checked);
//   - attn_v, ffn_down, and attn_output get llama.cpp's per-format "_M"
//     mixed-precision bump (see gemma4AttnVType / gemma4FfnDownType /
//     gemma4AttnOutputType) — a no-op (stays at bulk) for the pure q8_0/q6_k
//     formats;
//   - token_embd and the remaining projection weights are the bulk type.
//
// layerIndex is the block index for a per-layer (blk.N.*) tensor and is ignored
// for whole-model tensors; layerCount is the model's block_count.
func gemma4TensorType(format QuantizeFormat, canonical string, layerIndex, layerCount int) uint32 {
	switch canonical {
	case "per_layer_model_proj.weight":
		return ggufTensorTypeBF16
	case "output_norm.weight", "per_layer_proj_norm.weight", "rope_freqs.weight":
		return ggufTensorTypeF32
	}
	switch {
	case core.HasSuffix(canonical, "_norm.weight"):
		// attn_norm, ffn_norm, post_attention_norm, post_ffw_norm, post_norm,
		// attn_q_norm, attn_k_norm.
		return ggufTensorTypeF32
	case core.HasSuffix(canonical, ".layer_output_scale.weight"),
		core.HasSuffix(canonical, ".inp_gate.weight"),
		core.HasSuffix(canonical, ".proj.weight"):
		return ggufTensorTypeF32
	}

	bulk := gemma4BulkType(format)

	switch canonical {
	case "per_layer_token_embd.weight":
		if format == QuantizeQ4_K_M {
			return ggufTensorTypeQ5K
		}
		return bulk
	case "token_embd.weight":
		return bulk
	}

	switch {
	case core.HasSuffix(canonical, ".attn_v.weight"):
		return gemma4AttnVType(format, bulk, layerIndex, layerCount)
	case core.HasSuffix(canonical, ".ffn_down.weight"):
		return gemma4FfnDownType(format, bulk, layerIndex, layerCount)
	case core.HasSuffix(canonical, ".attn_output.weight"):
		return gemma4AttnOutputType(format, bulk)
	default:
		// attn_q, attn_k, ffn_gate, ffn_up.
		return bulk
	}
}
