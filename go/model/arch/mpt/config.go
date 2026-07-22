// SPDX-Licence-Identifier: EUPL-1.2

// Package mpt declares MosaicML's MPT dense decoder family.
package mpt

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

type AttentionConfig struct {
	ALiBi bool `json:"alibi"`
}

// Config is the architecture-relevant subset of an MPT config.json.
type Config struct {
	ModelType      string          `json:"model_type"`
	DModel         int             `json:"d_model"`
	NHeads         int             `json:"n_heads"`
	NLayers        int             `json:"n_layers"`
	ExpansionRatio int             `json:"expansion_ratio"`
	VocabSize      int             `json:"vocab_size"`
	LearnedPosEmb  bool            `json:"learned_pos_emb"`
	AttnConfig     AttentionConfig `json:"attn_config"`
}

func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch resolves MPT's independently declared ALiBi and learned-position features.
func (c *Config) Arch() (model.Arch, error) {
	if c.ModelType != "mpt" || c.DModel <= 0 || c.NHeads <= 0 || c.DModel%c.NHeads != 0 || c.NLayers <= 0 || c.ExpansionRatio <= 0 || c.VocabSize <= 0 {
		return model.Arch{}, core.E("mpt.Config.Arch", "invalid transformer geometry", nil)
	}
	headDim := c.DModel / c.NHeads
	layers := model.DeriveLayers(make([]string, c.NLayers), 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = headDim, c.NHeads
	}
	return model.Arch{Hidden: c.DModel, Heads: c.NHeads, KVHeads: c.NHeads, HeadDim: headDim, GlobalHeadDim: headDim, GlobalKVHeads: c.NHeads, FF: c.DModel * c.ExpansionRatio, Vocab: c.VocabSize, Eps: 1e-5, AttnScale: float32(1 / math.Sqrt(float64(headDim))), EmbedScale: 1, ALiBi: c.AttnConfig.ALiBi, LearnedAbsolutePositions: c.LearnedPosEmb, Layer: layers}, nil
}
