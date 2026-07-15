// SPDX-Licence-Identifier: EUPL-1.2

// Package olmoe declares AllenAI's OLMoE sparse transformer architecture.
package olmoe

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

const defaultRMSNormEps float32 = 1e-5

// Config is the architecture-relevant subset of an AllenAI OLMoE config.json.
type Config struct {
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	NumExperts        int     `json:"num_experts"`
	NumExpertsPerTok  int     `json:"num_experts_per_tok"`
	VocabSize         int     `json:"vocab_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	RopeTheta         float32 `json:"rope_theta"`
	NormTopKProb      bool    `json:"norm_topk_prob"`
	TieWordEmbeddings *bool   `json:"tie_word_embeddings"`
}

// InferFromWeights satisfies model.ArchConfig. OLMoE declares its geometry.
func (c *Config) InferFromWeights(weights map[string]safetensors.Tensor) { weights = nil }

// Arch resolves OLMoE's QK-normalised attention and routed-expert geometry.
func (c Config) Arch() (model.Arch, error) {
	if c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.VocabSize <= 0 || c.NumExperts <= 0 || c.NumExpertsPerTok <= 0 {
		return model.Arch{}, core.NewError("olmoe.Config.Arch: all architecture dimensions must be > 0")
	}
	if c.HiddenSize%c.NumAttentionHeads != 0 {
		return model.Arch{}, core.NewError("olmoe.Config.Arch: hidden_size must be divisible by num_attention_heads")
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("olmoe.Config.Arch: num_attention_heads must be a multiple of num_key_value_heads")
	}
	if c.NumExpertsPerTok > c.NumExperts {
		return model.Arch{}, core.NewError("olmoe.Config.Arch: num_experts_per_tok exceeds num_experts")
	}
	headDim := c.HiddenSize / c.NumAttentionHeads
	eps := c.RMSNormEps
	if eps == 0 {
		eps = defaultRMSNormEps
	}
	rope := c.RopeTheta
	if rope == 0 {
		rope = 10_000
	}
	layerTypes := make([]string, c.NumHiddenLayers)
	for i := range layerTypes {
		layerTypes[i] = "full_attention"
	}
	layers := model.DeriveLayers(layerTypes, 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads, layers[i].MoE = headDim, kvHeads, true
	}
	return model.Arch{
		Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: kvHeads, HeadDim: headDim,
		GlobalHeadDim: headDim, GlobalKVHeads: kvHeads, FF: c.IntermediateSize, Vocab: c.VocabSize,
		Experts: c.NumExperts, TopK: c.NumExpertsPerTok, ExpertFF: c.IntermediateSize,
		MoEGating: model.MoEGatingSoftmax, NormaliseMoETopK: c.NormTopKProb, SharedExperts: 0,
		Eps: eps, AttnScale: float32(1 / core.Pow(float64(headDim), 0.5)), EmbedScale: 1,
		RopeBase: rope, RopeLocalBase: rope, RopeScale: 1, RotaryDim: headDim, RotaryDimLocal: headDim,
		TieWordEmbeddings: c.TieWordEmbeddings, Layer: layers,
	}, nil
}
