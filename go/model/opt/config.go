// SPDX-Licence-Identifier: EUPL-1.2

// Package opt declares Meta's Open Pre-trained Transformer checkpoints.
package opt

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	"math"
)

// Config is the architecture subset shared by Hugging Face OPT checkpoints.
type Config struct {
	Hidden          int     `json:"hidden_size"`
	EmbedDim        int     `json:"word_embed_proj_dim"`
	Heads           int     `json:"num_attention_heads"`
	Layers          int     `json:"num_hidden_layers"`
	FF              int     `json:"ffn_dim"`
	Positions       int     `json:"max_position_embeddings"`
	Vocab           int     `json:"vocab_size"`
	Eps             float32 `json:"layer_norm_eps"`
	Activation      string  `json:"activation_function"`
	LayerNormBefore bool    `json:"do_layer_norm_before"`
	TieEmbeddings   *bool   `json:"tie_word_embeddings"`
}

func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch resolves OPT's learned-position, pre/post-norm transformer geometry.
func (c *Config) Arch() (model.Arch, error) {
	if c.Hidden <= 0 || c.Heads <= 0 || c.Hidden%c.Heads != 0 || c.Layers <= 0 || c.FF <= 0 || c.Positions <= 0 || c.Vocab <= 0 {
		return model.Arch{}, core.NewError("opt.Config.Arch: invalid geometry")
	}
	embedDim := c.EmbedDim
	if embedDim == 0 {
		embedDim = c.Hidden
	}
	if embedDim <= 0 {
		return model.Arch{}, core.NewError("opt.Config.Arch: invalid embedding geometry")
	}
	eps := c.Eps
	if eps == 0 {
		eps = 1e-5
	}
	headDim := c.Hidden / c.Heads
	layers := model.DeriveLayers(make([]string, c.Layers), 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = headDim, c.Heads
	}
	tied := c.TieEmbeddings
	if tied == nil {
		value := true
		tied = &value
	}
	return model.Arch{
		Hidden: c.Hidden, EmbeddingDim: embedDim, Heads: c.Heads, KVHeads: c.Heads,
		GlobalKVHeads: c.Heads, HeadDim: headDim, GlobalHeadDim: headDim, FF: c.FF,
		Vocab: c.Vocab, Eps: eps, AttnScale: float32(1 / math.Sqrt(float64(headDim))), EmbedScale: 1,
		LearnedAbsolutePositions: true, PositionOffset: 2, LayerNormBefore: c.LayerNormBefore,
		NoFinalNorm: !c.LayerNormBefore, Activation: c.Activation, TieWordEmbeddings: tied, Layer: layers,
	}, nil
}
