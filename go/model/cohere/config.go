// SPDX-Licence-Identifier: EUPL-1.2

// Package cohere declares the dense Cohere and Cohere2 transformer families.
// Cohere2-MoE is intentionally outside this package.
package cohere

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

const (
	defaultLayerNormEps  float32 = 1e-5
	defaultLogitScale    float32 = 0.0625
	defaultRopeTheta     float32 = 10000
	defaultWindowPattern         = 4
)

// Config is the architecture-relevant subset shared by Cohere and Cohere2.
type Config struct {
	ModelType            string   `json:"model_type"`
	HiddenSize           int      `json:"hidden_size"`
	IntermediateSize     int      `json:"intermediate_size"`
	NumHiddenLayers      int      `json:"num_hidden_layers"`
	NumAttentionHeads    int      `json:"num_attention_heads"`
	NumKeyValueHeads     int      `json:"num_key_value_heads"`
	VocabSize            int      `json:"vocab_size"`
	LayerNormEps         float32  `json:"layer_norm_eps"`
	LogitScale           float32  `json:"logit_scale"`
	RopeTheta            float32  `json:"rope_theta"`
	UseQKNorm            *bool    `json:"use_qk_norm"`
	SlidingWindow        int      `json:"sliding_window"`
	SlidingWindowPattern int      `json:"sliding_window_pattern"`
	LayerTypes           []string `json:"layer_types"`
	TieWordEmbeddings    *bool    `json:"tie_word_embeddings"`
}

func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch resolves the checkpoint declaration into the neutral decode contract.
func (c *Config) Arch() (model.Arch, error) {
	if c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.VocabSize <= 0 || c.HiddenSize%c.NumAttentionHeads != 0 {
		return model.Arch{}, core.NewError("cohere.Config.Arch: invalid transformer geometry")
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if kvHeads <= 0 || c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("cohere.Config.Arch: attention heads must be divisible by KV heads")
	}
	headDim := c.HiddenSize / c.NumAttentionHeads
	eps := c.LayerNormEps
	if eps == 0 {
		eps = defaultLayerNormEps
	}
	logitScale := c.LogitScale
	if logitScale == 0 {
		logitScale = defaultLogitScale
	}
	theta := c.RopeTheta
	if theta == 0 {
		theta = defaultRopeTheta
	}

	types := make([]string, c.NumHiddenLayers)
	if len(c.LayerTypes) == c.NumHiddenLayers {
		copy(types, c.LayerTypes)
	} else if c.ModelType == "cohere2" {
		pattern := c.SlidingWindowPattern
		if pattern == 0 {
			pattern = defaultWindowPattern
		}
		if pattern < 1 || c.SlidingWindow <= 0 {
			return model.Arch{}, core.NewError("cohere.Config.Arch: Cohere2 requires a positive sliding window and pattern")
		}
		for i := range types {
			types[i] = "sliding_attention"
			if (i+1)%pattern == 0 {
				types[i] = "full_attention"
			}
		}
	} else {
		for i := range types {
			types[i] = "full_attention"
		}
	}
	layers := model.DeriveLayers(types, 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = headDim, kvHeads
	}
	qk := model.QKNone
	if c.ModelType != "cohere2" && c.UseQKNorm != nil && *c.UseQKNorm {
		qk = model.QKLayerNorm
	}
	return model.Arch{
		Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: kvHeads,
		HeadDim: headDim, GlobalHeadDim: headDim, GlobalKVHeads: kvHeads,
		FF: c.IntermediateSize, Vocab: c.VocabSize, Eps: eps,
		AttnScale: float32(1 / math.Sqrt(float64(headDim))), EmbedScale: 1,
		RopeBase: theta, RopeLocalBase: theta, RopeScale: 1,
		RotaryDim: headDim, RotaryDimLocal: headDim,
		SlidingWindow: c.SlidingWindow, ParallelResidual: true,
		Activation: "silu", QKNormalization: qk, LogitScale: logitScale,
		TieWordEmbeddings: c.TieWordEmbeddings, Layer: layers,
	}, nil
}
