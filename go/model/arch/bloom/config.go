// SPDX-Licence-Identifier: EUPL-1.2

// Package bloom declares the BLOOM ALiBi transformer family.
package bloom

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// Config is the architecture-relevant subset of a Hugging Face BLOOM config.
type Config struct {
	HiddenSize        int     `json:"n_embed"`
	IntermediateSize  *int    `json:"n_inner"`
	NumHiddenLayers   int     `json:"n_layer"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	VocabSize         int     `json:"vocab_size"`
	LayerNormEpsilon  float32 `json:"layer_norm_epsilon"`
}

func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch resolves BLOOM's embedding width, ALiBi attention, and default 4x FFN.
func (c Config) Arch() (model.Arch, error) {
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.VocabSize <= 0 || c.HiddenSize%c.NumAttentionHeads != 0 {
		return model.Arch{}, core.E("bloom.Config.Arch", "invalid transformer geometry", nil)
	}
	ff := 4 * c.HiddenSize
	if c.IntermediateSize != nil && *c.IntermediateSize > 0 {
		ff = *c.IntermediateSize
	}
	eps := c.LayerNormEpsilon
	if eps == 0 {
		eps = 1e-5
	}
	headDim := c.HiddenSize / c.NumAttentionHeads
	layers := model.DeriveLayers(make([]string, c.NumHiddenLayers), 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = headDim, c.NumAttentionHeads
	}
	return model.Arch{Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: c.NumAttentionHeads, HeadDim: headDim, GlobalHeadDim: headDim, GlobalKVHeads: c.NumAttentionHeads, FF: ff, Vocab: c.VocabSize, Eps: eps, AttnScale: float32(1 / math.Sqrt(float64(headDim))), EmbedScale: 1, ALiBi: true, Layer: layers}, nil
}
