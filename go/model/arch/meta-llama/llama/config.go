// SPDX-Licence-Identifier: EUPL-1.2

// Package llama declares the dense Llama 3 text architecture to the reactive
// model loader. It intentionally excludes multimodal and sparse-expert variants.
package llama

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

const (
	defaultRopeTheta  float32 = 10000
	defaultRMSNormEps float32 = 1e-6
)

// Config is the architecture-relevant subset of a Hugging Face Llama config.json.
type Config struct {
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"`
	VocabSize         int     `json:"vocab_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	RopeTheta         float32 `json:"rope_theta"`
	TieWordEmbeddings *bool   `json:"tie_word_embeddings"`

	RopeScaling *RopeScaling `json:"rope_scaling"`
}

// RopeScaling describes the Hugging Face Llama RoPE variants. Type is retained
// for older configs; current transformers writes RopeType.
type RopeScaling struct {
	Type                          string  `json:"type"`
	RopeType                      string  `json:"rope_type"`
	Factor                        float32 `json:"factor"`
	LowFreqFactor                 float32 `json:"low_freq_factor"`
	HighFreqFactor                float32 `json:"high_freq_factor"`
	OriginalMaxPositionEmbeddings int     `json:"original_max_position_embeddings"`
}

func (r RopeScaling) kind() string {
	if r.RopeType != "" {
		return r.RopeType
	}
	return r.Type
}

// InferFromWeights satisfies model.ArchConfig. Llama configs declare their geometry.
func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch resolves dense Llama GQA geometry and RoPE into the neutral transformer declaration.
func (c *Config) Arch() (model.Arch, error) {
	if c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.VocabSize <= 0 {
		return model.Arch{}, core.NewError("llama.Config.Arch: hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, vocab_size must be > 0")
	}
	headDim := c.HeadDim
	if headDim == 0 {
		if c.HiddenSize%c.NumAttentionHeads != 0 {
			return model.Arch{}, core.NewError("llama.Config.Arch: head_dim absent and hidden_size not divisible by num_attention_heads")
		}
		headDim = c.HiddenSize / c.NumAttentionHeads
	}
	if headDim < 0 {
		return model.Arch{}, core.NewError("llama.Config.Arch: head_dim must be > 0")
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if kvHeads <= 0 || c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("llama.Config.Arch: num_attention_heads must be a multiple of num_key_value_heads")
	}
	eps := c.RMSNormEps
	if eps < 0 {
		return model.Arch{}, core.NewError("llama.Config.Arch: rms_norm_eps must be > 0")
	}
	if eps == 0 {
		eps = defaultRMSNormEps
	}
	ropeBase, ropeScale := c.RopeTheta, float32(1)
	if ropeBase < 0 {
		return model.Arch{}, core.NewError("llama.Config.Arch: rope_theta must be > 0")
	}
	if ropeBase == 0 {
		ropeBase = defaultRopeTheta
	}
	var ropeFreqs []float32
	if rp := c.RopeScaling; rp != nil {
		switch rp.kind() {
		case "", "default":
		case "linear":
			if rp.Factor <= 0 {
				return model.Arch{}, core.NewError("llama.Config.Arch: linear rope factor must be > 0")
			}
			ropeScale = 1 / rp.Factor
		case "llama3":
			ropeFreqs = Llama3InvFreqs(ropeBase, *rp, headDim)
			if ropeFreqs == nil {
				return model.Arch{}, core.NewError("llama.Config.Arch: invalid llama3 rope_scaling")
			}
		default:
			return model.Arch{}, core.NewError("llama.Config.Arch: unsupported rope_scaling variant " + rp.kind())
		}
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
		RopeBase: ropeBase, RopeLocalBase: ropeBase, RopeScale: ropeScale,
		RotaryDim: headDim, RotaryDimLocal: headDim, RopeFreqs: ropeFreqs,
		TieWordEmbeddings: c.TieWordEmbeddings,
		Layer:             layers,
	}, nil
}

// Llama3InvFreqs returns the Llama 3 piecewise-scaled inverse frequencies.
func Llama3InvFreqs(theta float32, r RopeScaling, rotaryDim int) []float32 {
	if theta <= 0 || rotaryDim <= 0 || rotaryDim%2 != 0 || r.Factor <= 1 ||
		r.LowFreqFactor <= 0 || r.HighFreqFactor <= r.LowFreqFactor ||
		r.OriginalMaxPositionEmbeddings <= 0 {
		return nil
	}
	out := make([]float32, rotaryDim/2)
	oldContext := float64(r.OriginalMaxPositionEmbeddings)
	low, high := float64(r.LowFreqFactor), float64(r.HighFreqFactor)
	for i := range out {
		plain := math.Pow(float64(theta), -float64(2*i)/float64(rotaryDim))
		waveLength := 2 * math.Pi / plain
		scaled := plain
		switch {
		case waveLength > oldContext/low:
			scaled = plain / float64(r.Factor)
		case waveLength >= oldContext/high:
			smooth := (oldContext/waveLength - low) / (high - low)
			scaled = (1-smooth)*plain/float64(r.Factor) + smooth*plain
		}
		out[i] = float32(scaled)
	}
	return out
}
