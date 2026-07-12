// SPDX-Licence-Identifier: EUPL-1.2

// Package gptneox declares GPT-NeoX, GPT-J and GPT-Neo checkpoint geometry.
package gptneox

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// Config is the architecture subset shared by the three Hugging Face families.
type Config struct {
	ModelType           string   `json:"model_type"`
	HiddenSize          int      `json:"hidden_size"`
	IntermediateSize    int      `json:"intermediate_size"`
	NumHiddenLayers     int      `json:"num_hidden_layers"`
	NumAttentionHeads   int      `json:"num_attention_heads"`
	VocabSize           int      `json:"vocab_size"`
	LayerNormEps        float32  `json:"layer_norm_eps"`
	RotaryPct           float32  `json:"rotary_pct"`
	RotaryEmbBase       float32  `json:"rotary_emb_base"`
	UseParallelResidual bool     `json:"use_parallel_residual"`
	NEmbd               int      `json:"n_embd"`
	NInner              int      `json:"n_inner"`
	NLayer              int      `json:"n_layer"`
	NHead               int      `json:"n_head"`
	RotaryDim           int      `json:"rotary_dim"`
	LayerNormEpsilon    float32  `json:"layer_norm_epsilon"`
	AttentionLayers     []string `json:"attention_layers"`
	WindowSize          int      `json:"window_size"`
}

func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch resolves aliases and configurable rotary dimensions into the neutral declaration.
func (c *Config) Arch() (model.Arch, error) {
	hidden, layers, heads := c.HiddenSize, c.NumHiddenLayers, c.NumAttentionHeads
	ff, eps := c.IntermediateSize, c.LayerNormEps
	if c.ModelType == "gptj" || c.ModelType == "gpt_neo" {
		hidden, layers, heads = c.NEmbd, c.NLayer, c.NHead
		ff, eps = c.NInner, c.LayerNormEpsilon
	}
	if hidden <= 0 || layers <= 0 || heads <= 0 || hidden%heads != 0 || c.VocabSize <= 0 {
		return model.Arch{}, core.NewError("gptneox.Config.Arch: invalid architecture dimensions")
	}
	if ff == 0 {
		ff = 4 * hidden
	}
	if eps == 0 {
		eps = 1e-5
	}
	headDim := hidden / heads
	rotary := c.RotaryDim
	if rotary == 0 {
		pct := c.RotaryPct
		if pct == 0 {
			pct = 1
		}
		rotary = int(float32(headDim) * pct)
	}
	rotary -= rotary % 2
	if rotary <= 0 || rotary > headDim {
		return model.Arch{}, core.NewError("gptneox.Config.Arch: rotary dimension outside head dimension")
	}
	types := make([]string, layers)
	for i := range types {
		types[i] = "full_attention"
	}
	window := 0
	if c.ModelType == "gpt_neo" && len(c.AttentionLayers) == layers {
		for i, kind := range c.AttentionLayers {
			if kind == "local" {
				types[i] = "sliding_attention"
				window = c.WindowSize
			}
		}
	}
	derived := model.DeriveLayers(types, 0)
	for i := range derived {
		derived[i].HeadDim, derived[i].KVHeads = headDim, heads
	}
	base := c.RotaryEmbBase
	if base == 0 {
		base = 10000
	}
	return model.Arch{Hidden: hidden, Heads: heads, KVHeads: heads, HeadDim: headDim, GlobalHeadDim: headDim, GlobalKVHeads: heads,
		FF: ff, Vocab: c.VocabSize, Eps: eps, AttnScale: float32(1 / core.Pow(float64(headDim), .5)), EmbedScale: 1,
		RopeBase: base, RopeLocalBase: base, RopeScale: 1, RotaryDim: rotary, RotaryDimLocal: rotary,
		ParallelResidual: c.ModelType == "gptj" || c.UseParallelResidual, SlidingWindow: window, Layer: derived}, nil
}
