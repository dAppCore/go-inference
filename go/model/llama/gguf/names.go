// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"strconv"

	core "dappco.re/go"
)

var llamaTopLevelTensorNames = map[string]string{
	"model.embed_tokens.weight": "token_embd.weight",
	"model.norm.weight":         "output_norm.weight",
	"lm_head.weight":            "output.weight",
}

var llamaLayerTensorNames = map[string]string{
	"input_layernorm.weight":          "attn_norm.weight",
	"post_attention_layernorm.weight": "ffn_norm.weight",
	"self_attn.q_proj.weight":         "attn_q.weight",
	"self_attn.k_proj.weight":         "attn_k.weight",
	"self_attn.v_proj.weight":         "attn_v.weight",
	"self_attn.o_proj.weight":         "attn_output.weight",
	"mlp.gate_proj.weight":            "ffn_gate.weight",
	"mlp.up_proj.weight":              "ffn_up.weight",
	"mlp.down_proj.weight":            "ffn_down.weight",
}

func llamaCanonicalTensorName(source string) (string, error) {
	if canonical, ok := llamaTopLevelTensorNames[source]; ok {
		return canonical, nil
	}
	rest, ok := core.CutPrefix(source, "model.layers.")
	if ok {
		index, suffix, found := core.Cut(rest, ".")
		if found {
			if _, err := strconv.Atoi(index); err == nil {
				if canonical, exists := llamaLayerTensorNames[suffix]; exists {
					return core.Concat("blk.", index, ".", canonical), nil
				}
			}
		}
	}
	return "", core.NewError("gguf: llama has no canonical GGUF name for tensor " + source)
}

func llamaGGUFShape(shape []uint64) []uint64 {
	reversed := make([]uint64, len(shape))
	for i, dimension := range shape {
		reversed[len(shape)-1-i] = dimension
	}
	return reversed
}

func llamaCanonicalLayerIndex(name string) int {
	rest, ok := core.CutPrefix(name, "blk.")
	if !ok {
		return -1
	}
	index, _, found := core.Cut(rest, ".")
	if !found {
		return -1
	}
	value, err := strconv.Atoi(index)
	if err != nil {
		return -1
	}
	return value
}
