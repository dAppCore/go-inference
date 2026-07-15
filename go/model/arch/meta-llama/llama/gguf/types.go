// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	core "dappco.re/go"
	basegguf "dappco.re/go/inference/model/gguf"
)

func llamaUseMoreBits(layerIndex, layerCount int) bool {
	return layerIndex < layerCount/8 || layerIndex >= 7*layerCount/8 || (layerIndex-layerCount/8)%3 == 2
}

func llamaTensorType(format basegguf.QuantizeFormat, canonical string, layerIndex, layerCount int, tiedEmbeddings bool) uint32 {
	if core.HasSuffix(canonical, "_norm.weight") {
		return basegguf.TensorTypeF32
	}
	if format == basegguf.QuantizeQ8_0 {
		return basegguf.TensorTypeQ8_0
	}
	if canonical == "output.weight" || tiedEmbeddings && canonical == "token_embd.weight" {
		return basegguf.TensorTypeQ6K
	}
	if (core.HasSuffix(canonical, ".attn_v.weight") || core.HasSuffix(canonical, ".ffn_down.weight")) && llamaUseMoreBits(layerIndex, layerCount) {
		return basegguf.TensorTypeQ6K
	}
	return basegguf.TensorTypeQ4K
}
