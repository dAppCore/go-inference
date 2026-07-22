// SPDX-Licence-Identifier: EUPL-1.2

// Package granitemoe declares IBM Granite sparse transformer checkpoints.
package granitemoe

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// Config is the architecture-relevant GraniteMoE config.json surface.
type Config struct {
	ModelType           string  `json:"model_type"`
	HiddenSize          int     `json:"hidden_size"`
	IntermediateSize    int     `json:"intermediate_size"`
	NumHiddenLayers     int     `json:"num_hidden_layers"`
	NumAttentionHeads   int     `json:"num_attention_heads"`
	NumKeyValueHeads    int     `json:"num_key_value_heads"`
	NumLocalExperts     int     `json:"num_local_experts"`
	NumExpertsPerTok    int     `json:"num_experts_per_tok"`
	VocabSize           int     `json:"vocab_size"`
	RMSNormEps          float32 `json:"rms_norm_eps"`
	RopeTheta           float32 `json:"rope_theta"`
	TieWordEmbeddings   *bool   `json:"tie_word_embeddings"`
	HiddenActivation    string  `json:"hidden_act"`
	LogitsScaling       float32 `json:"logits_scaling"`
	ResidualMultiplier  float32 `json:"residual_multiplier"`
	EmbeddingMultiplier float32 `json:"embedding_multiplier"`
	AttentionMultiplier float32 `json:"attention_multiplier"`
}

// ParseConfig parses an IBM GraniteMoE config.json.
func ParseConfig(data []byte) core.Result {
	var cfg Config
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return core.Fail(core.E("granitemoe.ParseConfig", "config.json parse failed", nil))
	}
	return core.Ok(&cfg)
}

// InferFromWeights satisfies model.ArchConfig; GraniteMoE declares its geometry.
func (c *Config) InferFromWeights(weights map[string]safetensors.Tensor) { weights = nil }

// Arch resolves Granite's scalar operations and declared sparse-expert policy.
func (c *Config) Arch() (model.Arch, error) {
	if c.ModelType != "" && c.ModelType != "granitemoe" {
		return model.Arch{}, core.E("granitemoe.Config.Arch", "model_type must be granitemoe", nil)
	}
	if c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.NumKeyValueHeads <= 0 || c.NumLocalExperts <= 0 || c.NumExpertsPerTok <= 0 || c.VocabSize <= 0 {
		return model.Arch{}, core.E("granitemoe.Config.Arch", "all architecture dimensions must be > 0", nil)
	}
	if c.HiddenSize%c.NumAttentionHeads != 0 || c.NumAttentionHeads%c.NumKeyValueHeads != 0 {
		return model.Arch{}, core.E("granitemoe.Config.Arch", "attention geometry is not divisible", nil)
	}
	if c.NumExpertsPerTok > c.NumLocalExperts {
		return model.Arch{}, core.E("granitemoe.Config.Arch", "num_experts_per_tok exceeds num_local_experts", nil)
	}
	if c.RMSNormEps <= 0 || c.RopeTheta <= 0 || c.LogitsScaling <= 0 || c.ResidualMultiplier <= 0 || c.EmbeddingMultiplier <= 0 || c.AttentionMultiplier <= 0 {
		return model.Arch{}, core.E("granitemoe.Config.Arch", "norm, rope, and scalar declarations must be > 0", nil)
	}
	headDim := c.HiddenSize / c.NumAttentionHeads
	layerTypes := make([]string, c.NumHiddenLayers)
	for i := range layerTypes {
		layerTypes[i] = "full_attention"
	}
	layers := model.DeriveLayers(layerTypes, 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads, layers[i].MoE = headDim, c.NumKeyValueHeads, true
	}
	return model.Arch{
		Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: c.NumKeyValueHeads, HeadDim: headDim,
		GlobalHeadDim: headDim, GlobalKVHeads: c.NumKeyValueHeads, FF: c.IntermediateSize, Vocab: c.VocabSize,
		Experts: c.NumLocalExperts, TopK: c.NumExpertsPerTok, ExpertFF: c.IntermediateSize,
		MoEGating: model.MoEGatingSoftmax, NormaliseMoETopK: true, SharedExperts: 0,
		Eps: c.RMSNormEps, AttnScale: c.AttentionMultiplier, EmbedScale: c.EmbeddingMultiplier,
		LogitsScaling: c.LogitsScaling, ResidualMultiplier: c.ResidualMultiplier,
		RopeBase: c.RopeTheta, RopeLocalBase: c.RopeTheta, RopeScale: 1, RotaryDim: headDim, RotaryDimLocal: headDim,
		TieWordEmbeddings: c.TieWordEmbeddings, Activation: c.HiddenActivation, Layer: layers,
	}, nil
}
