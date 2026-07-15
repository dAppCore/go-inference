// SPDX-Licence-Identifier: EUPL-1.2

// Package stablelm declares the dense StableLM decoder family.
package stablelm

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/attn"
	"dappco.re/go/inference/model/safetensors"
)

// Config is the architecture-relevant subset of a StableLM config.json.
type Config struct {
	ModelType           string  `json:"model_type"`
	HiddenSize          int     `json:"hidden_size"`
	IntermediateSize    int     `json:"intermediate_size"`
	NumHiddenLayers     int     `json:"num_hidden_layers"`
	NumAttentionHeads   int     `json:"num_attention_heads"`
	NumKeyValueHeads    int     `json:"num_key_value_heads"`
	VocabSize           int     `json:"vocab_size"`
	LayerNormEps        float32 `json:"layer_norm_eps"`
	PartialRotaryFactor float32 `json:"partial_rotary_factor"`
	RopeTheta           float32 `json:"rope_theta"`
	HiddenAct           string  `json:"hidden_act"`
	TieWordEmbeddings   *bool   `json:"tie_word_embeddings"`
	QKLayerNorm         bool    `json:"qk_layernorm"`
	UseParallelResidual bool    `json:"use_parallel_residual"`
}

func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch resolves StableLM's GQA and partial rotary geometry.
func (c *Config) Arch() (model.Arch, error) {
	if c.ModelType != "stablelm" || c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.HiddenSize%c.NumAttentionHeads != 0 || c.VocabSize <= 0 {
		return model.Arch{}, core.E("stablelm.Config.Arch", "invalid transformer geometry", nil)
	}
	hd := c.HiddenSize / c.NumAttentionHeads
	kv := c.NumKeyValueHeads
	if kv == 0 {
		kv = c.NumAttentionHeads
	}
	if kv <= 0 || c.NumAttentionHeads%kv != 0 {
		return model.Arch{}, core.E("stablelm.Config.Arch", "invalid key/value heads", nil)
	}
	rd, err := (attn.RopeParams{HeadDim: hd, PartialRotaryFactor: c.PartialRotaryFactor}).RotaryDim()
	if err != nil {
		return model.Arch{}, core.E("stablelm.Config.Arch", "resolve partial rotary", err)
	}
	eps := c.LayerNormEps
	if eps == 0 {
		eps = 1e-5
	}
	theta := c.RopeTheta
	if theta == 0 {
		theta = 10000
	}
	layers := model.DeriveLayers(make([]string, c.NumHiddenLayers), 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = hd, kv
	}
	qk := model.QKNone
	if c.QKLayerNorm {
		qk = model.QKLayerNorm
	}
	return model.Arch{Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: kv, HeadDim: hd, GlobalHeadDim: hd, GlobalKVHeads: kv, FF: c.IntermediateSize, Vocab: c.VocabSize, Eps: eps, AttnScale: float32(1 / math.Sqrt(float64(hd))), EmbedScale: 1, RopeBase: theta, RopeLocalBase: theta, RopeScale: 1, RotaryDim: rd, RotaryDimLocal: rd, TieWordEmbeddings: c.TieWordEmbeddings, Activation: c.HiddenAct, QKNormalization: qk, ParallelResidual: c.UseParallelResidual, Layer: layers}, nil
}
