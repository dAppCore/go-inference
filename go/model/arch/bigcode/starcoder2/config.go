// SPDX-Licence-Identifier: EUPL-1.2

// Package starcoder2 declares StarCoder2 and compatible CodeGen checkpoints.
package starcoder2

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// Config is the architecture subset used by StarCoder2 and CodeGen configs.
type Config struct {
	ModelType             string  `json:"model_type"`
	HiddenSize            int     `json:"hidden_size"`
	IntermediateSize      int     `json:"intermediate_size"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	NumKeyValueHeads      int     `json:"num_key_value_heads"`
	VocabSize             int     `json:"vocab_size"`
	NormEpsilon           float32 `json:"norm_epsilon"`
	HiddenActivation      string  `json:"hidden_act"`
	RopeTheta             float32 `json:"rope_theta"`
	SlidingWindow         int     `json:"sliding_window"`
	TieWordEmbeddings     *bool   `json:"tie_word_embeddings"`

	ActivationFunction string  `json:"activation_function"`
	LayerNormEpsilon   float32 `json:"layer_norm_epsilon"`
	Hidden             int     `json:"n_embd"`
	Heads              int     `json:"n_head"`
	Inner              int     `json:"n_inner"`
	Layers             int     `json:"n_layer"`
	Positions          int     `json:"n_positions"`
	RotaryDimension    int     `json:"rotary_dim"`
}

// ParseConfig parses a Hugging Face StarCoder2 or CodeGen config.json.
func ParseConfig(data []byte) (*Config, error) {
	var cfg Config
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.E("starcoder2.ParseConfig", "config.json parse failed", nil)
	}
	return &cfg, nil
}

// InferFromWeights satisfies model.ArchConfig; config.json owns this geometry.
func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch resolves StarCoder2 or CodeGen into the neutral transformer declaration.
func (c *Config) Arch() (model.Arch, error) {
	if c.ModelType == "codegen" {
		return c.codegenArch()
	}
	return c.starcoder2Arch()
}

func (c *Config) starcoder2Arch() (model.Arch, error) {
	if c.ModelType != "starcoder2" || c.HiddenSize <= 0 || c.NumAttentionHeads <= 0 || c.HiddenSize%c.NumAttentionHeads != 0 || c.NumHiddenLayers <= 0 || c.IntermediateSize <= 0 || c.MaxPositionEmbeddings <= 0 || c.VocabSize <= 0 {
		return model.Arch{}, core.E("starcoder2.Config.Arch", "invalid geometry", nil)
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if kvHeads <= 0 || c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.E("starcoder2.Config.Arch", "invalid KV geometry", nil)
	}
	headDim := c.HiddenSize / c.NumAttentionHeads
	theta := c.RopeTheta
	if theta == 0 {
		theta = 10_000
	}
	eps := c.NormEpsilon
	if eps == 0 {
		eps = 1e-5
	}
	layerTypes := make([]string, c.NumHiddenLayers)
	if c.SlidingWindow > 0 {
		for i := range layerTypes {
			layerTypes[i] = "sliding_attention"
		}
	}
	layers := model.DeriveLayers(layerTypes, 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = headDim, kvHeads
	}
	tied := c.TieWordEmbeddings
	if tied == nil {
		v := true
		tied = &v
	}
	return model.Arch{Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: kvHeads, GlobalKVHeads: kvHeads,
		HeadDim: headDim, GlobalHeadDim: headDim, FF: c.IntermediateSize, Vocab: c.VocabSize, Eps: eps,
		AttnScale: float32(1 / math.Sqrt(float64(headDim))), EmbedScale: 1, RopeBase: theta, RopeLocalBase: theta,
		RopeScale: 1, RotaryDim: headDim, RotaryDimLocal: headDim, SlidingWindow: c.SlidingWindow,
		TieWordEmbeddings: tied, Activation: c.HiddenActivation, Layer: layers}, nil
}

func (c *Config) codegenArch() (model.Arch, error) {
	if c.Hidden <= 0 || c.Heads <= 0 || c.Hidden%c.Heads != 0 || c.Layers <= 0 || c.Positions <= 0 || c.VocabSize <= 0 || c.RotaryDimension <= 0 {
		return model.Arch{}, core.E("starcoder2.Config.Arch", "invalid CodeGen geometry", nil)
	}
	ff := c.Inner
	if ff == 0 {
		ff = 4 * c.Hidden
	}
	eps := c.LayerNormEpsilon
	if eps == 0 {
		eps = 1e-5
	}
	headDim := c.Hidden / c.Heads
	if c.RotaryDimension > headDim || c.RotaryDimension%2 != 0 {
		return model.Arch{}, core.E("starcoder2.Config.Arch", "invalid CodeGen rotary geometry", nil)
	}
	layers := model.DeriveLayers(make([]string, c.Layers), 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = headDim, c.Heads
	}
	tied := c.TieWordEmbeddings
	if tied == nil {
		v := false
		tied = &v
	}
	return model.Arch{Hidden: c.Hidden, Heads: c.Heads, KVHeads: c.Heads, GlobalKVHeads: c.Heads,
		HeadDim: headDim, GlobalHeadDim: headDim, FF: ff, Vocab: c.VocabSize, Eps: eps,
		AttnScale: float32(1 / math.Sqrt(float64(headDim))), EmbedScale: 1, RopeBase: 10_000, RopeLocalBase: 10_000,
		RopeScale: 1, RotaryDim: c.RotaryDimension, RotaryDimLocal: c.RotaryDimension,
		ParallelResidual: true, TieWordEmbeddings: tied, Activation: c.ActivationFunction, Layer: layers}, nil
}
