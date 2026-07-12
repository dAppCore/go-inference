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

// gemma4TensorType returns the GGML tensor type the q4_k_m policy assigns to a
// canonical gemma-4 tensor, mirroring the unsloth gemma-4-E2B-it-Q4_K_M oracle
// per-tensor type map:
//
//   - RMS-norm weights, the per-layer input gate/projection, the per-layer
//     output scale, and rope_freqs stay F32 (llama.cpp keeps these in full
//     precision regardless of the requested bulk quant);
//   - per_layer_model_proj stays BF16 (passed through from the source dtype);
//   - per_layer_token_embd is Q5_K;
//   - attn_v and ffn_down are Q6_K on the use_more_bits layers, Q4_K elsewhere;
//   - token_embd and the remaining projection weights are Q4_K.
//
// layerIndex is the block index for a per-layer (blk.N.*) tensor and is ignored
// for whole-model tensors; layerCount is the model's block_count.
func gemma4TensorType(canonical string, layerIndex, layerCount int) uint32 {
	switch canonical {
	case "per_layer_model_proj.weight":
		return ggufTensorTypeBF16
	case "per_layer_token_embd.weight":
		return ggufTensorTypeQ5K
	case "token_embd.weight":
		return ggufTensorTypeQ4K
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
	case core.HasSuffix(canonical, ".attn_v.weight"),
		core.HasSuffix(canonical, ".ffn_down.weight"):
		if gemma4UseMoreBits(layerIndex, layerCount) {
			return ggufTensorTypeQ6K
		}
		return ggufTensorTypeQ4K
	default:
		// attn_q, attn_k, attn_output, ffn_gate, ffn_up.
		return ggufTensorTypeQ4K
	}
}
