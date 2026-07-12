// SPDX-Licence-Identifier: EUPL-1.2

// Package falcon declares the ALiBi Falcon transformer family. Falcon-H1 and
// Mamba variants are intentionally outside this package.
package falcon

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// Config is the architecture-relevant subset of a Hugging Face Falcon config.
type Config struct {
	HiddenSize             int     `json:"hidden_size"`
	NumHiddenLayers        int     `json:"num_hidden_layers"`
	NumAttentionHeads      int     `json:"num_attention_heads"`
	NumKVHeads             int     `json:"num_kv_heads"`
	VocabSize              int     `json:"vocab_size"`
	LayerNormEpsilon       float32 `json:"layer_norm_epsilon"`
	MultiQuery             bool    `json:"multi_query"`
	NewDecoderArchitecture bool    `json:"new_decoder_architecture"`
	ALiBi                  bool    `json:"alibi"`
}

func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch resolves old/new Falcon decoder and multi-query geometry.
func (c Config) Arch() (model.Arch, error) {
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.VocabSize <= 0 || c.HiddenSize%c.NumAttentionHeads != 0 {
		return model.Arch{}, core.E("falcon.Config.Arch", "invalid transformer geometry", nil)
	}
	kv := c.NumAttentionHeads
	if c.MultiQuery {
		kv = 1
	}
	if c.NewDecoderArchitecture && c.NumKVHeads > 0 {
		kv = c.NumKVHeads
	}
	if kv <= 0 || c.NumAttentionHeads%kv != 0 {
		return model.Arch{}, core.E("falcon.Config.Arch", "attention heads must be divisible by KV heads", nil)
	}
	eps := c.LayerNormEpsilon
	if eps == 0 {
		eps = 1e-5
	}
	headDim := c.HiddenSize / c.NumAttentionHeads
	layers := model.DeriveLayers(make([]string, c.NumHiddenLayers), 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = headDim, kv
	}
	return model.Arch{Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: kv, HeadDim: headDim, GlobalHeadDim: headDim, GlobalKVHeads: kv, FF: 4 * c.HiddenSize, Vocab: c.VocabSize, Eps: eps, AttnScale: float32(1 / math.Sqrt(float64(headDim))), EmbedScale: 1, ALiBi: c.ALiBi, Layer: layers}, nil
}
