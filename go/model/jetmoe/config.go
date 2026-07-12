// SPDX-Licence-Identifier: EUPL-1.2

// Package jetmoe declares the JetMoE sparse transformer architecture.
package jetmoe

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

const defaultRopeTheta float32 = 10_000

// Config is the architecture-relevant subset of a JetMoE config.json.
type Config struct {
	ModelType         string  `json:"model_type"`
	HiddenSize        int     `json:"hidden_size"`
	FFNHiddenSize     int     `json:"ffn_hidden_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	KVChannels        int     `json:"kv_channels"`
	MoENumExperts     int     `json:"moe_num_experts"`
	MoETopK           int     `json:"moe_top_k"`
	VocabSize         int     `json:"vocab_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	LayerNormEpsilon  float32 `json:"layer_norm_epsilon"`
	RopeTheta         float32 `json:"rope_theta"`
	RotaryPercent     float32 `json:"rotary_percent"`
	TieWordEmbeddings *bool   `json:"tie_word_embeddings"`
}

// InferFromWeights satisfies model.ArchConfig. Published JetMoE configs declare their geometry.
func (c *Config) InferFromWeights(weights map[string]safetensors.Tensor) { weights = nil }

// Arch resolves JetMoE's grouped-query attention and routed FFN declaration.
func (c Config) Arch() (model.Arch, error) {
	if c.HiddenSize <= 0 || c.FFNHiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.MoENumExperts <= 0 || c.MoETopK <= 0 || c.VocabSize <= 0 {
		return model.Arch{}, core.NewError("jetmoe.Config.Arch: all architecture dimensions must be > 0")
	}
	if c.MoETopK > c.MoENumExperts {
		return model.Arch{}, core.NewError("jetmoe.Config.Arch: moe_top_k exceeds moe_num_experts")
	}
	headDim := c.KVChannels
	if headDim == 0 {
		if c.HiddenSize%c.NumAttentionHeads != 0 {
			return model.Arch{}, core.NewError("jetmoe.Config.Arch: hidden_size must divide by num_attention_heads when kv_channels is absent")
		}
		headDim = c.HiddenSize / c.NumAttentionHeads
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("jetmoe.Config.Arch: num_attention_heads must be a multiple of num_key_value_heads")
	}
	eps := c.RMSNormEps
	if eps == 0 {
		eps = c.LayerNormEpsilon
	}
	if eps == 0 {
		eps = 1e-5
	}
	rope := c.RopeTheta
	if rope == 0 {
		rope = defaultRopeTheta
	}
	rotaryPercent := c.RotaryPercent
	if rotaryPercent == 0 {
		rotaryPercent = 1
	}
	rotaryDim := int(float32(headDim) * rotaryPercent)
	layerTypes := make([]string, c.NumHiddenLayers)
	for i := range layerTypes {
		layerTypes[i] = "full_attention"
	}
	layers := model.DeriveLayers(layerTypes, 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads, layers[i].MoE = headDim, kvHeads, true
	}
	return model.Arch{
		Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: kvHeads,
		HeadDim: headDim, GlobalHeadDim: headDim, GlobalKVHeads: kvHeads,
		FF: c.FFNHiddenSize, Vocab: c.VocabSize, Experts: c.MoENumExperts,
		TopK: c.MoETopK, ExpertFF: c.FFNHiddenSize,
		MoEGating: model.MoEGatingSoftmax, NormaliseMoETopK: true, SharedExperts: 0,
		Eps: eps, AttnScale: float32(1 / core.Pow(float64(headDim), 0.5)), EmbedScale: 1,
		RopeBase: rope, RopeLocalBase: rope, RopeScale: 1,
		RotaryDim: rotaryDim, RotaryDimLocal: rotaryDim,
		TieWordEmbeddings: c.TieWordEmbeddings, Layer: layers,
	}, nil
}
