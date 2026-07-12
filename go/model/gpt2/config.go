// SPDX-Licence-Identifier: EUPL-1.2

// Package gpt2 declares GPT-2, GPT-SW3 and GPT-BigCode/StarCoder checkpoints.
package gpt2

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	"math"
)

// Config is the architecture subset shared by Hugging Face GPT-2 and GPT-BigCode.
type Config struct {
	Hidden     int     `json:"n_embd"`
	Heads      int     `json:"n_head"`
	Layers     int     `json:"n_layer"`
	Inner      int     `json:"n_inner"`
	Positions  int     `json:"n_positions"`
	Vocab      int     `json:"vocab_size"`
	Eps        float32 `json:"layer_norm_epsilon"`
	Activation string  `json:"activation_function"`
	MultiQuery bool    `json:"multi_query"`
}

func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch resolves the pre-LayerNorm, learned-position transformer geometry.
func (c *Config) Arch() (model.Arch, error) {
	if c.Hidden <= 0 || c.Heads <= 0 || c.Hidden%c.Heads != 0 || c.Layers <= 0 || c.Positions <= 0 || c.Vocab <= 0 {
		return model.Arch{}, core.NewError("gpt2.Config.Arch: invalid geometry")
	}
	ff := c.Inner
	if ff == 0 {
		ff = 4 * c.Hidden
	}
	eps := c.Eps
	if eps == 0 {
		eps = 1e-5
	}
	kv := c.Heads
	if c.MultiQuery {
		kv = 1
	}
	layerTypes := make([]string, c.Layers)
	layers := model.DeriveLayers(layerTypes, 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = c.Hidden/c.Heads, kv
	}
	return model.Arch{Hidden: c.Hidden, Heads: c.Heads, KVHeads: kv, GlobalKVHeads: kv,
		HeadDim: c.Hidden / c.Heads, GlobalHeadDim: c.Hidden / c.Heads, FF: ff, Vocab: c.Vocab, Eps: eps,
		AttnScale: float32(1 / math.Sqrt(float64(c.Hidden/c.Heads))), EmbedScale: 1,
		LearnedAbsolutePositions: true, MultiQueryAttention: c.MultiQuery, Activation: c.Activation,
		TieWordEmbeddings: boolPtr(true), Layer: layers}, nil
}

func boolPtr(v bool) *bool { return &v }
