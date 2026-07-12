// SPDX-Licence-Identifier: EUPL-1.2

// Package qwen2 declares dense Qwen2 and Qwen2.5 text checkpoints to the
// backend-neutral reactive model loader. Qwen2-MoE is intentionally excluded.
package qwen2

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

const (
	defaultRopeTheta  float32 = 10_000
	defaultRMSNormEps float32 = 1e-6
)

// Config is the architecture-relevant subset of a Qwen2ForCausalLM config.json.
type Config struct {
	ModelType             string  `json:"model_type"`
	HiddenSize            int     `json:"hidden_size"`
	IntermediateSize      int     `json:"intermediate_size"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	NumKeyValueHeads      int     `json:"num_key_value_heads"`
	HeadDim               int     `json:"head_dim"`
	VocabSize             int     `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	TieWordEmbeddings     *bool   `json:"tie_word_embeddings"`
	UseSlidingWindow      bool    `json:"use_sliding_window"`
	SlidingWindow         int     `json:"sliding_window"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
}

// ParseConfig parses a dense Qwen2 or Qwen2.5 Hugging Face config.
func ParseConfig(data []byte) (*Config, error) {
	var cfg Config
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.NewError("qwen2.ParseConfig: config.json parse failed")
	}
	return &cfg, nil
}

// InferFromWeights resolves dimensions omitted by older exported configs.
func (c *Config) InferFromWeights(weights map[string]safetensors.Tensor) {
	if c.HeadDim == 0 {
		for i := 0; i < c.NumHiddenLayers; i++ {
			if dim := model.InferHeadDim(weights, core.Sprintf("model.layers.%d.self_attn.q_proj.weight", i), c.NumAttentionHeads); dim > 0 {
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

// Arch resolves dense Qwen2 GQA geometry into the neutral transformer declaration.
func (c *Config) Arch() (model.Arch, error) {
	if c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.VocabSize <= 0 {
		return model.Arch{}, core.NewError("qwen2.Config.Arch: hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, vocab_size must be > 0")
	}
	headDim := c.HeadDim
	if headDim == 0 {
		if c.HiddenSize%c.NumAttentionHeads != 0 {
			return model.Arch{}, core.NewError("qwen2.Config.Arch: hidden_size must divide by num_attention_heads")
		}
		headDim = c.HiddenSize / c.NumAttentionHeads
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if kvHeads <= 0 || c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("qwen2.Config.Arch: num_attention_heads must be a multiple of num_key_value_heads")
	}
	eps := c.RMSNormEps
	if eps == 0 {
		eps = defaultRMSNormEps
	}
	ropeBase := c.RopeTheta
	if ropeBase == 0 {
		ropeBase = defaultRopeTheta
	}
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
		FF: c.IntermediateSize, Vocab: c.VocabSize, Eps: eps,
		AttnScale: float32(1 / math.Sqrt(float64(headDim))), EmbedScale: 1,
		RopeBase: ropeBase, RopeLocalBase: ropeBase, RopeScale: 1,
		RotaryDim: headDim, RotaryDimLocal: headDim,
		TieWordEmbeddings: c.TieWordEmbeddings,
		Layer:             layers,
	}, nil
}
