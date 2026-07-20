// SPDX-Licence-Identifier: EUPL-1.2

// Package dbrx declares Databricks' DBRX sparse transformer architecture.
package dbrx

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

const defaultLayerNormEps float32 = 1e-5

// AttentionConfig carries DBRX's nested attn_config block.
type AttentionConfig struct {
	KVHeads   int     `json:"kv_n_heads"`
	RopeTheta float32 `json:"rope_theta"`
	ClipQKV   float32 `json:"clip_qkv"`
}

// FFNConfig carries DBRX's nested ffn_config block.
type FFNConfig struct {
	HiddenSize int `json:"ffn_hidden_size"`
	Experts    int `json:"moe_num_experts"`
	TopK       int `json:"moe_top_k"`
	// ActFn is DBRX's routed-expert activation, nested as ffn_config.ffn_act_fn.name ("silu" on every
	// real DBRX checkpoint) rather than a flat hidden_act — unlike olmoe/mixtral's top-level field, DBRX
	// buries it inside its own already-nested ffn_config block.
	ActFn FFNActivation `json:"ffn_act_fn"`
}

// FFNActivation carries DBRX's ffn_config.ffn_act_fn.name activation declaration.
type FFNActivation struct {
	Name string `json:"name"`
}

// Config is the architecture-relevant subset of a DBRX config.json.
type Config struct {
	DModel            int             `json:"d_model"`
	Heads             int             `json:"n_heads"`
	Layers            int             `json:"n_layers"`
	VocabSize         int             `json:"vocab_size"`
	LayerNormEps      float32         `json:"layer_norm_epsilon"`
	TieWordEmbeddings *bool           `json:"tie_word_embeddings"`
	Attention         AttentionConfig `json:"attn_config"`
	FFN               FFNConfig       `json:"ffn_config"`
}

// InferFromWeights satisfies model.ArchConfig. DBRX declares its geometry.
func (c *Config) InferFromWeights(weights map[string]safetensors.Tensor) { weights = nil }

// Arch resolves DBRX's nested attention and routed-expert declarations.
func (c Config) Arch() (model.Arch, error) {
	if c.DModel <= 0 || c.Heads <= 0 || c.Layers <= 0 || c.VocabSize <= 0 || c.FFN.HiddenSize <= 0 || c.FFN.Experts <= 0 || c.FFN.TopK <= 0 || c.Attention.KVHeads <= 0 {
		return model.Arch{}, core.NewError("dbrx.Config.Arch: all architecture dimensions must be > 0")
	}
	if c.DModel%c.Heads != 0 || c.Heads%c.Attention.KVHeads != 0 {
		return model.Arch{}, core.NewError("dbrx.Config.Arch: invalid attention head geometry")
	}
	if c.FFN.TopK > c.FFN.Experts {
		return model.Arch{}, core.NewError("dbrx.Config.Arch: moe_top_k exceeds moe_num_experts")
	}
	headDim := c.DModel / c.Heads
	eps := c.LayerNormEps
	if eps == 0 {
		eps = defaultLayerNormEps
	}
	rope := c.Attention.RopeTheta
	if rope == 0 {
		rope = 10_000
	}
	kinds := make([]string, c.Layers)
	for i := range kinds {
		kinds[i] = "full_attention"
	}
	layers := model.DeriveLayers(kinds, 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads, layers[i].MoE = headDim, c.Attention.KVHeads, true
	}
	return model.Arch{
		Hidden: c.DModel, Heads: c.Heads, KVHeads: c.Attention.KVHeads, HeadDim: headDim,
		GlobalHeadDim: headDim, GlobalKVHeads: c.Attention.KVHeads, FF: c.FFN.HiddenSize, Vocab: c.VocabSize,
		Experts: c.FFN.Experts, TopK: c.FFN.TopK, ExpertFF: c.FFN.HiddenSize,
		MoEGating: model.MoEGatingSoftmax, NormaliseMoETopK: false, SharedExperts: 0,
		Eps: eps, AttnScale: float32(1 / core.Pow(float64(headDim), 0.5)), EmbedScale: 1,
		RopeBase: rope, RopeLocalBase: rope, RopeScale: 1, RotaryDim: headDim, RotaryDimLocal: headDim,
		TieWordEmbeddings: c.TieWordEmbeddings, LayerNormBefore: true, LayerNorm: true, QKVClip: c.Attention.ClipQKV, NormPlacement: model.NormPlacementPre, Activation: c.FFN.ActFn.Name, Layer: layers,
	}, nil
}
