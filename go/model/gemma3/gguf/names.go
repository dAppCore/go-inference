// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"strconv"

	core "dappco.re/go"
)

// gemma3TopLevelTensorNames maps a gemma-3 source checkpoint's whole-model
// tensor names (the HF safetensors convention, e.g. model.embed_tokens.weight)
// to the canonical GGUF names llama.cpp looks the gemma3 text stack up by.
// Derived by diffing the source safetensors directory against a real
// llama.cpp-loadable gemma3 GGUF's tensor directory (token_embd / output /
// output_norm). lm_head.weight is present only in an untied checkpoint; a tied
// model omits it and llama.cpp reuses token_embd for the output projection.
var gemma3TopLevelTensorNames = map[string]string{
	"model.embed_tokens.weight": "token_embd.weight",
	"lm_head.weight":            "output.weight",
	"model.norm.weight":         "output_norm.weight",
}

// gemma3LayerTensorNames maps a gemma-3 per-layer tensor's suffix (everything
// after model.layers.<N>.) to the canonical GGUF suffix that follows blk.<N>.
// in the file.
var gemma3LayerTensorNames = map[string]string{
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
}

// gemma3CanonicalTensorName maps a gemma-3 source checkpoint tensor name to the
// canonical GGUF tensor name llama.cpp maps the text stack by (token_embd.weight,
// blk.N.attn_q.weight, …). An unrecognised name is an error rather than a
// silent drop — a checkpoint carrying a tensor this mapping has not seen must
// fail loudly so the mapping can be extended, not produce a GGUF missing a weight.
//
//	name, err := gemma3CanonicalTensorName("model.layers.7.self_attn.q_proj.weight")
//	// name == "blk.7.attn_q.weight"
func gemma3CanonicalTensorName(src string) (string, error) {
	if canonical, ok := gemma3TopLevelTensorNames[src]; ok {
		return canonical, nil
	}
	if rest, ok := core.CutPrefix(src, "model.layers."); ok {
		if index, suffix, found := core.Cut(rest, "."); found {
			if _, err := strconv.Atoi(index); err == nil {
				if canonical, ok := gemma3LayerTensorNames[suffix]; ok {
					return core.Concat("blk.", index, ".", canonical), nil
				}
			}
		}
	}
	return "", core.NewError("gguf: gemma3 has no canonical GGUF name for tensor " + src)
}

// gemma3GGUFShape converts a safetensors row-major shape ([out_features,
// in_features]) into the GGUF ne[] convention ([in_features, out_features]) —
// the reverse of the source order. Only the reported dimension order flips: the
// row-major bytes are unchanged, because GGUF's ne0 is the contiguous inner
// dimension whereas safetensors lists the outermost dimension first. A GGUF
// written with the source order unreversed places the wrong dimension in ne0
// and llama.cpp rejects the tensor.
//
//	gemma3GGUFShape([]uint64{1024, 1152}) // -> []uint64{1152, 1024}
func gemma3GGUFShape(safetensorsShape []uint64) []uint64 {
	reversed := make([]uint64, len(safetensorsShape))
	for i, dim := range safetensorsShape {
		reversed[len(safetensorsShape)-1-i] = dim
	}
	return reversed
}

// gemma3TensorRowLength returns ne0 — the GGUF inner (contiguous) dimension a
// block quantiser tiles along, which must divide evenly into the quant's block
// size. For a source shape [out, in] this is the last (inner) source dimension;
// a 1-D tensor's row length is its single dimension. An empty shape yields 0
// (the caller treats it as unquantisable).
func gemma3TensorRowLength(safetensorsShape []uint64) uint64 {
	if len(safetensorsShape) == 0 {
		return 0
	}
	return safetensorsShape[len(safetensorsShape)-1]
}
