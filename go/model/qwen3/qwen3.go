// SPDX-Licence-Identifier: EUPL-1.2

// Package qwen3 declares the DENSE qwen3 architecture to the engine's reactive loader. qwen3 dense is a
// standard pre-norm transformer (input_layernorm + post_attention_layernorm, like mistral/llama) with
// per-head QK-norm (q_norm/k_norm) and GQA — it reuses the shared decode entirely. It differs from
// mistral only in carrying QK-norm weights (which bind automatically from the standard names when
// present) and reading rope_theta directly. RMSNorm is PLAIN (qwen is not gemma — no +1). The sparse
// qwen3_moe and the qwen3.6 gated-delta (linear-attention) variants are separate: MoE routing + the
// GatedDeltaNet mixer are not in native yet, so only the dense family registers here.
package qwen3

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/safetensors"
)

const (
	defaultRopeTheta  float32 = 1_000_000
	defaultRMSNormEps float32 = 1e-6
)

// Config is the arch-relevant subset of a dense qwen3 config.json.
type Config struct {
	HiddenSize        int     `json:"hidden_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"`
	IntermediateSize  int     `json:"intermediate_size"`
	VocabSize         int     `json:"vocab_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	RopeTheta         float32 `json:"rope_theta"`

	TextConfig   *Config            `json:"text_config"`
	Quantization *model.QuantConfig `json:"quantization"`
}

// ResolvedQuant returns the checkpoint's quantization block (top-level or nested), nil = bf16.
func (c *Config) ResolvedQuant() *model.QuantConfig {
	if c.Quantization != nil {
		return c.Quantization
	}
	if c.TextConfig != nil {
		return c.TextConfig.Quantization
	}
	return nil
}

// InferFromWeights resolves head_dim from a q_proj's rows and vocab from the embedding rows when the
// config omits them (the don't-guess rule). Satisfies model.ArchConfig.
func (c *Config) InferFromWeights(weights map[string]safetensors.Tensor) {
	if c.TextConfig != nil {
		c.TextConfig.InferFromWeights(weights)
		return
	}
	if c.HeadDim == 0 {
		for i := 0; i < c.NumHiddenLayers; i++ {
			if hd := model.InferHeadDim(weights, core.Sprintf("model.layers.%d.self_attn.q_proj.weight", i), c.NumAttentionHeads); hd > 0 {
				c.HeadDim = hd
				break
			}
		}
		if c.HeadDim == 0 && c.HiddenSize > 0 && c.NumAttentionHeads > 0 {
			c.HeadDim = c.HiddenSize / c.NumAttentionHeads
		}
	}
	if c.VocabSize == 0 {
		if w, ok := model.WeightAny(weights, "model.embed_tokens.weight", "model.embed_tokens"); ok && len(w.Shape) > 0 && w.Shape[0] > 0 {
			c.VocabSize = int(w.Shape[0])
		}
	}
}

// Arch builds the neutral model.Arch: a standard GQA transformer, scale 1/sqrt(head_dim), full rotary,
// all-global attention (no sliding), no value-norm / softcap. QK-norm is carried as weights (bound by
// the standard q_norm/k_norm names when present), not an Arch flag. Satisfies model.ArchConfig.
func (c *Config) Arch() (model.Arch, error) {
	if c.TextConfig != nil {
		return c.TextConfig.Arch()
	}
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 {
		return model.Arch{}, core.NewError("qwen3.Config.Arch: hidden_size, num_hidden_layers, num_attention_heads must be > 0")
	}
	headDim := c.HeadDim
	if headDim == 0 {
		if c.HiddenSize%c.NumAttentionHeads != 0 {
			return model.Arch{}, core.NewError("qwen3.Config.Arch: head_dim absent and hidden_size not divisible by num_attention_heads")
		}
		headDim = c.HiddenSize / c.NumAttentionHeads
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("qwen3.Config.Arch: num_attention_heads must be a multiple of num_key_value_heads")
	}
	eps := c.RMSNormEps
	if eps == 0 {
		eps = defaultRMSNormEps
	}
	ropeBase := c.RopeTheta
	if ropeBase == 0 {
		ropeBase = defaultRopeTheta
	}
	types := make([]string, c.NumHiddenLayers)
	for i := range types {
		types[i] = "full_attention" // qwen3 dense is all-global attention
	}
	layers := model.DeriveLayers(types, 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = headDim, kvHeads
	}
	return model.Arch{
		Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: kvHeads,
		HeadDim: headDim, GlobalHeadDim: headDim, GlobalKVHeads: kvHeads,
		FF: c.IntermediateSize, Vocab: c.VocabSize, Eps: eps,
		AttnScale:      float32(1.0 / math.Sqrt(float64(headDim))),
		RopeBase:       ropeBase,
		RopeLocalBase:  ropeBase,
		RotaryDim:      headDim,
		RotaryDimLocal: headDim,
		RopeScale:      1,
		Layer:          layers,
	}, nil
}
