// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"strconv"

	core "dappco.re/go"
)

// gemma4TopLevelTensorNames maps gemma-4 source checkpoint tensor names (the
// HF/mlx safetensors convention, e.g. language_model.model.embed_tokens.weight)
// to the canonical GGUF names llama.cpp looks the text stack up by. These are
// the whole-model tensors that carry no per-layer index.
//
// Derived by diffing the mlx-community gemma-4-E2B-it-bf16 checkpoint's
// language_model.model.* tensors against the unsloth gemma-4-E2B-it GGUF's 601
// tensor directory — the mapping is mechanical once both lists are in hand.
var gemma4TopLevelTensorNames = map[string]string{
	"language_model.model.embed_tokens.weight":               "token_embd.weight",
	"language_model.model.embed_tokens_per_layer.weight":     "per_layer_token_embd.weight",
	"language_model.model.norm.weight":                       "output_norm.weight",
	"language_model.model.per_layer_model_projection.weight": "per_layer_model_proj.weight",
	"language_model.model.per_layer_projection_norm.weight":  "per_layer_proj_norm.weight",
}

// gemma4LayerTensorNames maps a gemma-4 per-layer tensor's suffix (everything
// after language_model.model.layers.<N>.) to the canonical GGUF suffix that
// follows blk.<N>. in the file. layer_scalar is the one source name with no
// .weight suffix; llama.cpp carries it as blk.<N>.layer_output_scale.weight.
var gemma4LayerTensorNames = map[string]string{
	"input_layernorm.weight":            "attn_norm.weight",
	"self_attn.q_proj.weight":           "attn_q.weight",
	"self_attn.k_proj.weight":           "attn_k.weight",
	"self_attn.v_proj.weight":           "attn_v.weight",
	"self_attn.o_proj.weight":           "attn_output.weight",
	"self_attn.q_norm.weight":           "attn_q_norm.weight",
	"self_attn.k_norm.weight":           "attn_k_norm.weight",
	"mlp.gate_proj.weight":              "ffn_gate.weight",
	"mlp.up_proj.weight":                "ffn_up.weight",
	"mlp.down_proj.weight":              "ffn_down.weight",
	"pre_feedforward_layernorm.weight":  "ffn_norm.weight",
	"post_feedforward_layernorm.weight": "post_ffw_norm.weight",
	"post_attention_layernorm.weight":   "post_attention_norm.weight",
	"post_per_layer_input_norm.weight":  "post_norm.weight",
	"per_layer_input_gate.weight":       "inp_gate.weight",
	"per_layer_projection.weight":       "proj.weight",
	"layer_scalar":                      "layer_output_scale.weight",
}

// gemma4CanonicalTensorName maps a gemma-4 source checkpoint tensor name to the
// canonical GGUF tensor name llama.cpp maps the text stack by (token_embd.weight,
// blk.N.attn_q.weight, …). It covers the text stack only; multimodal-tower
// tensors are excluded before this call (IsMultimodalTowerTensor) and never
// reach it. An unrecognised text-stack name is an error rather than a silent
// drop — a checkpoint carrying a text tensor this mapping has not seen must fail
// loudly so the mapping can be extended, not produce a GGUF missing a weight.
//
//	name, err := gemma4CanonicalTensorName("language_model.model.layers.7.self_attn.q_proj.weight")
//	// name == "blk.7.attn_q.weight"
func gemma4CanonicalTensorName(src string) (string, error) {
	if canonical, ok := gemma4TopLevelTensorNames[src]; ok {
		return canonical, nil
	}
	if rest, ok := core.CutPrefix(src, "language_model.model.layers."); ok {
		if index, suffix, found := core.Cut(rest, "."); found {
			if _, err := strconv.Atoi(index); err == nil {
				if canonical, ok := gemma4LayerTensorNames[suffix]; ok {
					return core.Concat("blk.", index, ".", canonical), nil
				}
			}
		}
	}
	return "", core.NewError("gguf: gemma4 has no canonical GGUF name for tensor " + src)
}

// gemma4GGUFShape converts a safetensors row-major shape ([out_features,
// in_features, …]) into the GGUF ne[] convention ([in_features, out_features,
// …]) — the reverse of the source order. Only the reported dimension order
// flips: the tensor's row-major bytes are unchanged, because GGUF's ne0 is the
// contiguous inner dimension whereas safetensors lists the outermost dimension
// first. A GGUF written with the source order unreversed places the wrong
// dimension in ne0 and llama.cpp rejects the tensor.
//
//	gemma4GGUFShape([]uint64{2048, 1536}) // -> []uint64{1536, 2048}
func gemma4GGUFShape(safetensorsShape []uint64) []uint64 {
	reversed := make([]uint64, len(safetensorsShape))
	for i, dim := range safetensorsShape {
		reversed[len(safetensorsShape)-1-i] = dim
	}
	return reversed
}
