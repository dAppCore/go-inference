// SPDX-Licence-Identifier: EUPL-1.2

// Package olmo declares the OLMo and OLMo 2 causal transformer families.
package olmo

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

const (
	defaultLayerNormEps float32 = 1e-5
	defaultRopeTheta    float32 = 10000
)

// Config is the architecture-relevant subset shared by Hugging Face OLMo and OLMo 2 configs.
type Config struct {
	ModelType         string  `json:"model_type"`
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	VocabSize         int     `json:"vocab_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	RopeTheta         float32 `json:"rope_theta"`
	HiddenAct         string  `json:"hidden_act"`
	TieWordEmbeddings *bool   `json:"tie_word_embeddings"`
}

// ParseConfig parses an OLMo-family config.json without loading model weights.
func ParseConfig(data []byte) (*Config, error) {
	var cfg Config
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.NewError("olmo.ParseConfig: config.json parse failed")
	}
	return &cfg, nil
}

// InferFromWeights satisfies model.ArchConfig. OLMo configs declare their geometry.
func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch resolves OLMo generation-specific norm and attention strategy.
func (c *Config) Arch() (model.Arch, error) {
	if c.ModelType != "olmo" && c.ModelType != "olmo2" {
		return model.Arch{}, core.NewError("olmo.Config.Arch: unsupported model_type")
	}
	if c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.VocabSize <= 0 || c.HiddenSize%c.NumAttentionHeads != 0 {
		return model.Arch{}, core.NewError("olmo.Config.Arch: invalid architecture dimensions")
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if kvHeads <= 0 || c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("olmo.Config.Arch: num_attention_heads must be a multiple of num_key_value_heads")
	}
	headDim := c.HiddenSize / c.NumAttentionHeads
	eps := c.RMSNormEps
	if eps < 0 {
		return model.Arch{}, core.NewError("olmo.Config.Arch: rms_norm_eps must be > 0")
	}
	if eps == 0 {
		eps = defaultLayerNormEps
	}
	rope := c.RopeTheta
	if rope < 0 {
		return model.Arch{}, core.NewError("olmo.Config.Arch: rope_theta must be > 0")
	}
	if rope == 0 {
		rope = defaultRopeTheta
	}
	activation := c.HiddenAct
	if activation == "" {
		activation = "silu"
	}
	placement := model.NormPlacementPre
	nonParametric := true
	if c.ModelType == "olmo2" {
		placement = model.NormPlacementPost
		nonParametric = false
	}
	types := make([]string, c.NumHiddenLayers)
	for i := range types {
		types[i] = "full_attention"
	}
	layers := model.DeriveLayers(types, 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = headDim, kvHeads
	}
	return model.Arch{
		Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: kvHeads,
		HeadDim: headDim, GlobalHeadDim: headDim, GlobalKVHeads: kvHeads,
		FF: c.IntermediateSize, Vocab: c.VocabSize, Eps: eps,
		AttnScale: float32(1 / core.Pow(float64(headDim), .5)), EmbedScale: 1,
		RopeBase: rope, RopeLocalBase: rope, RopeScale: 1,
		RotaryDim: headDim, RotaryDimLocal: headDim,
		TieWordEmbeddings: c.TieWordEmbeddings, Activation: activation,
		NormPlacement: placement, NonParametricLayerNorm: nonParametric,
		Layer: layers,
	}, nil
}
