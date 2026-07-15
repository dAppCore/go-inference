// SPDX-Licence-Identifier: EUPL-1.2

// Package granite declares IBM Granite dense text checkpoints to the reactive
// model loader. Granite MoE and Granite MoE Hybrid checkpoints are excluded.
package granite

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// Config is the architecture-relevant subset of a Hugging Face Granite config.
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
	HiddenActivation    string  `json:"hidden_act"`
	LogitsScaling       float32 `json:"logits_scaling"`
	ResidualMultiplier  float32 `json:"residual_multiplier"`
	EmbeddingMultiplier float32 `json:"embedding_multiplier"`
	AttentionMultiplier float32 `json:"attention_multiplier"`
}

// ParseConfig parses a dense IBM Granite config.json.
func ParseConfig(data []byte) core.Result {
	var cfg Config
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return core.Fail(core.NewError("granite.ParseConfig: config.json parse failed"))
	}
	return core.Ok(&cfg)
}

// InferFromWeights satisfies model.ArchConfig; Granite declares its geometry.
func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch resolves Granite's Llama-shaped dense geometry and four scalar operations.
func (c *Config) Arch() (model.Arch, error) {
	if c.ModelType != "" && c.ModelType != "granite" {
		return model.Arch{}, core.NewError("granite.Config.Arch: model_type must be granite")
	}
	if c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.VocabSize <= 0 {
		return model.Arch{}, core.NewError("granite.Config.Arch: dimensions must be > 0")
	}
	if c.HiddenSize%c.NumAttentionHeads != 0 {
		return model.Arch{}, core.NewError("granite.Config.Arch: hidden_size must be divisible by num_attention_heads")
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads <= 0 || c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("granite.Config.Arch: num_attention_heads must be a multiple of num_key_value_heads")
	}
	if c.RMSNormEps <= 0 || c.RopeTheta <= 0 || c.LogitsScaling <= 0 || c.ResidualMultiplier <= 0 || c.EmbeddingMultiplier <= 0 || c.AttentionMultiplier <= 0 {
		return model.Arch{}, core.NewError("granite.Config.Arch: norm, rope, and scalar declarations must be > 0")
	}
	headDim := c.HiddenSize / c.NumAttentionHeads
	layerTypes := make([]string, c.NumHiddenLayers)
	for i := range layerTypes {
		layerTypes[i] = "full_attention"
	}
	layers := model.DeriveLayers(layerTypes, 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = headDim, kvHeads
	}
	return model.Arch{
		Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: kvHeads,
		HeadDim: headDim, GlobalHeadDim: headDim, GlobalKVHeads: kvHeads,
		FF: c.IntermediateSize, Vocab: c.VocabSize, Eps: c.RMSNormEps,
		AttnScale: c.AttentionMultiplier, EmbedScale: c.EmbeddingMultiplier,
		LogitsScaling: c.LogitsScaling, ResidualMultiplier: c.ResidualMultiplier,
		RopeBase: c.RopeTheta, RopeLocalBase: c.RopeTheta, RopeScale: 1,
		RotaryDim: headDim, RotaryDimLocal: headDim, TieWordEmbeddings: c.TieWordEmbeddings,
		Activation: c.HiddenActivation, Layer: layers,
	}, nil
}
