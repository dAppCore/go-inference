// SPDX-Licence-Identifier: EUPL-1.2

// Package qwenmoe declares Qwen2-MoE and Qwen3-MoE sparse transformers.
package qwenmoe

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

const (
	defaultRopeTheta  float32 = 1_000_000
	defaultRMSNormEps float32 = 1e-6
)

// Config is the architecture-relevant common subset of Qwen2-MoE and Qwen3-MoE configs.
type Config struct {
	ModelType                    string  `json:"model_type"`
	HiddenSize                   int     `json:"hidden_size"`
	IntermediateSize             int     `json:"intermediate_size"`
	MoEIntermediateSize          int     `json:"moe_intermediate_size"`
	SharedExpertIntermediateSize int     `json:"shared_expert_intermediate_size"`
	NumHiddenLayers              int     `json:"num_hidden_layers"`
	NumAttentionHeads            int     `json:"num_attention_heads"`
	NumKeyValueHeads             int     `json:"num_key_value_heads"`
	HeadDim                      int     `json:"head_dim"`
	NumExperts                   int     `json:"num_experts"`
	NumExpertsPerTok             int     `json:"num_experts_per_tok"`
	VocabSize                    int     `json:"vocab_size"`
	RMSNormEps                   float32 `json:"rms_norm_eps"`
	RopeTheta                    float32 `json:"rope_theta"`
	NormTopKProb                 bool    `json:"norm_topk_prob"`
	TieWordEmbeddings            *bool   `json:"tie_word_embeddings"`
}

// InferFromWeights resolves dimensions omitted by older exported configs.
func (c *Config) InferFromWeights(weights map[string]safetensors.Tensor) {
	if c.HeadDim == 0 {
		for layer := 0; layer < c.NumHiddenLayers; layer++ {
			if dim := model.InferHeadDim(weights, core.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer), c.NumAttentionHeads); dim > 0 {
				c.HeadDim = dim
				break
			}
		}
	}
	if c.VocabSize == 0 {
		if weight, ok := model.WeightAny(weights, "model.embed_tokens.weight", "model.embed_tokens"); ok && len(weight.Shape) > 0 {
			c.VocabSize = weight.Shape[0]
		}
	}
}

// Arch resolves attention, router, top-k normalisation, and optional shared-expert policy.
func (c *Config) Arch() (model.Arch, error) {
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.VocabSize <= 0 || c.NumExperts <= 0 || c.NumExpertsPerTok <= 0 || c.MoEIntermediateSize <= 0 {
		return model.Arch{}, core.NewError("qwenmoe.Config.Arch: all architecture dimensions must be > 0")
	}
	if c.NumExpertsPerTok > c.NumExperts {
		return model.Arch{}, core.NewError("qwenmoe.Config.Arch: num_experts_per_tok exceeds num_experts")
	}
	headDim := c.HeadDim
	if headDim == 0 {
		if c.HiddenSize%c.NumAttentionHeads != 0 {
			return model.Arch{}, core.NewError("qwenmoe.Config.Arch: hidden_size must divide by num_attention_heads when head_dim is absent")
		}
		headDim = c.HiddenSize / c.NumAttentionHeads
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if kvHeads <= 0 || c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("qwenmoe.Config.Arch: num_attention_heads must be a multiple of num_key_value_heads")
	}
	eps := c.RMSNormEps
	if eps == 0 {
		eps = defaultRMSNormEps
	}
	rope := c.RopeTheta
	if rope == 0 {
		rope = defaultRopeTheta
	}
	types := make([]string, c.NumHiddenLayers)
	for layer := range types {
		types[layer] = "full_attention"
	}
	layers := model.DeriveLayers(types, 0)
	for layer := range layers {
		layers[layer].HeadDim, layers[layer].KVHeads, layers[layer].MoE = headDim, kvHeads, true
	}
	sharedExperts := 0
	if c.SharedExpertIntermediateSize > 0 {
		sharedExperts = 1
	}
	return model.Arch{
		Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: kvHeads,
		HeadDim: headDim, GlobalHeadDim: headDim, GlobalKVHeads: kvHeads,
		FF: c.IntermediateSize, Vocab: c.VocabSize, Experts: c.NumExperts,
		TopK: c.NumExpertsPerTok, ExpertFF: c.MoEIntermediateSize,
		MoEGating: model.MoEGatingSoftmax, NormaliseMoETopK: c.NormTopKProb,
		SharedExperts: sharedExperts, Eps: eps,
		AttnScale: float32(1 / core.Pow(float64(headDim), 0.5)), EmbedScale: 1,
		RopeBase: rope, RopeLocalBase: rope, RopeScale: 1,
		RotaryDim: headDim, RotaryDimLocal: headDim,
		TieWordEmbeddings: c.TieWordEmbeddings, Layer: layers,
	}, nil
}
