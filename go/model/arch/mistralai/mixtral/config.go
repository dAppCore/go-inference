// SPDX-Licence-Identifier: EUPL-1.2

// Package mixtral declares the sparse Mixtral text architecture.
package mixtral

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

const (
	defaultRMSNormEps float32 = 1e-5
	defaultRopeTheta  float32 = 1_000_000
)

// Config is the architecture-relevant subset of a Hugging Face Mixtral config.json.
type Config struct {
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	NumLocalExperts   int     `json:"num_local_experts"`
	NumExpertsPerTok  int     `json:"num_experts_per_tok"`
	VocabSize         int     `json:"vocab_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	RopeTheta         float32 `json:"rope_theta"`
	TieWordEmbeddings *bool   `json:"tie_word_embeddings"`
	// HiddenActivation is the routed-expert SwiGLU gate ("silu" on every real Mixtral checkpoint).
	// Forwarded verbatim into Arch.Activation (mirroring granitemoe's HiddenActivation) so the MoE
	// expert combine (engine/metal) can select SiLU instead of gemma4's GELU (#63).
	HiddenActivation string `json:"hidden_act"`
}

// InferFromWeights satisfies model.ArchConfig. Mixtral declares its geometry.
func (c *Config) InferFromWeights(weights map[string]safetensors.Tensor) { weights = nil }

// Arch resolves Mixtral GQA and sparse-expert geometry.
func (c Config) Arch() (model.Arch, error) {
	if c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.VocabSize <= 0 || c.NumLocalExperts <= 0 || c.NumExpertsPerTok <= 0 {
		return model.Arch{}, core.NewError("mixtral.Config.Arch: all architecture dimensions must be > 0")
	}
	if c.HiddenSize%c.NumAttentionHeads != 0 {
		return model.Arch{}, core.NewError("mixtral.Config.Arch: hidden_size must be divisible by num_attention_heads")
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("mixtral.Config.Arch: num_attention_heads must be a multiple of num_key_value_heads")
	}
	if c.NumExpertsPerTok > c.NumLocalExperts {
		return model.Arch{}, core.NewError("mixtral.Config.Arch: num_experts_per_tok exceeds num_local_experts")
	}
	headDim := c.HiddenSize / c.NumAttentionHeads
	eps := c.RMSNormEps
	if eps == 0 {
		eps = defaultRMSNormEps
	}
	rope := c.RopeTheta
	if rope == 0 {
		rope = defaultRopeTheta
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
		Experts: c.NumLocalExperts, TopK: c.NumExpertsPerTok, ExpertFF: c.IntermediateSize,
		MoEGating: model.MoEGatingSoftmax, NormaliseMoETopK: true, Eps: eps, AttnScale: float32(1 / core.Pow(float64(headDim), 0.5)), EmbedScale: 1,
		RopeBase: rope, RopeLocalBase: rope, RopeScale: 1, RotaryDim: headDim, RotaryDimLocal: headDim,
		TieWordEmbeddings: c.TieWordEmbeddings, Activation: c.HiddenActivation, Layer: layers,
	}, nil
}
