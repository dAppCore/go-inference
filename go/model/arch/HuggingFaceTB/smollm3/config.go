// SPDX-Licence-Identifier: EUPL-1.2

// Package smollm3 declares the SmolLM3 dense GQA decoder.
package smollm3

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	"math"
)

// Config is the architecture-relevant subset of a SmolLM3 config.json.
type Config struct {
	ModelType           string  `json:"model_type"`
	HiddenSize          int     `json:"hidden_size"`
	IntermediateSize    int     `json:"intermediate_size"`
	NumHiddenLayers     int     `json:"num_hidden_layers"`
	NumAttentionHeads   int     `json:"num_attention_heads"`
	NumKeyValueHeads    int     `json:"num_key_value_heads"`
	VocabSize           int     `json:"vocab_size"`
	RMSNormEps          float32 `json:"rms_norm_eps"`
	RopeTheta           float32 `json:"rope_theta"`
	TieWordEmbeddings   *bool   `json:"tie_word_embeddings"`
	NoRopeLayerInterval int     `json:"no_rope_layer_interval"`
	NoRopeLayers        []int   `json:"no_rope_layers"`
}

func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch resolves SmolLM3's Llama-shaped geometry and explicit NoPE schedule.
func (c *Config) Arch() (model.Arch, error) {
	if c.ModelType != "smollm3" || c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.HiddenSize%c.NumAttentionHeads != 0 || c.VocabSize <= 0 {
		return model.Arch{}, core.E("smollm3.Config.Arch", "invalid transformer geometry", nil)
	}
	kv := c.NumKeyValueHeads
	if kv <= 0 || c.NumAttentionHeads%kv != 0 {
		return model.Arch{}, core.E("smollm3.Config.Arch", "invalid key/value heads", nil)
	}
	if len(c.NoRopeLayers) != 0 && len(c.NoRopeLayers) != c.NumHiddenLayers {
		return model.Arch{}, core.E("smollm3.Config.Arch", "no_rope_layers length mismatch", nil)
	}
	hd := c.HiddenSize / c.NumAttentionHeads
	eps := c.RMSNormEps
	if eps == 0 {
		eps = 1e-6
	}
	theta := c.RopeTheta
	if theta == 0 {
		theta = 10000
	}
	layers := model.DeriveLayers(make([]string, c.NumHiddenLayers), 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = hd, kv
		if len(c.NoRopeLayers) > 0 {
			layers[i].DisableRotary = c.NoRopeLayers[i] != 0
		} else if c.NoRopeLayerInterval > 0 {
			layers[i].DisableRotary = (i+1)%c.NoRopeLayerInterval != 0
		}
	}
	return model.Arch{Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: kv, HeadDim: hd, GlobalHeadDim: hd, GlobalKVHeads: kv, FF: c.IntermediateSize, Vocab: c.VocabSize, Eps: eps, AttnScale: float32(1 / math.Sqrt(float64(hd))), EmbedScale: 1, RopeBase: theta, RopeLocalBase: theta, RopeScale: 1, RotaryDim: hd, RotaryDimLocal: hd, TieWordEmbeddings: c.TieWordEmbeddings, Activation: "silu", Layer: layers}, nil
}
