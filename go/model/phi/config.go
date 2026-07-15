// SPDX-Licence-Identifier: EUPL-1.2

// Package phi declares the distinct Phi-2 and Phi-3 dense text generations.
package phi

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/attn"
	"dappco.re/go/inference/model/safetensors"
)

// Config is the union of architecture fields declared by Phi-2 and Phi-3.
// ModelType selects the generation-specific defaults and weight declaration.
type Config struct {
	ModelType                     string       `json:"model_type"`
	HiddenSize                    int          `json:"hidden_size"`
	IntermediateSize              int          `json:"intermediate_size"`
	NumHiddenLayers               int          `json:"num_hidden_layers"`
	NumAttentionHeads             int          `json:"num_attention_heads"`
	NumKeyValueHeads              int          `json:"num_key_value_heads"`
	VocabSize                     int          `json:"vocab_size"`
	LayerNormEps                  float32      `json:"layer_norm_eps"`
	RMSNormEps                    float32      `json:"rms_norm_eps"`
	RopeTheta                     float32      `json:"rope_theta"`
	PartialRotaryFactor           float32      `json:"partial_rotary_factor"`
	SlidingWindow                 int          `json:"sliding_window"`
	OriginalMaxPositionEmbeddings int          `json:"original_max_position_embeddings"`
	TieWordEmbeddings             *bool        `json:"tie_word_embeddings"`
	RopeScaling                   *RopeScaling `json:"rope_scaling"`
}

// RopeScaling declares Phi-3's position-dependent LongRoPE factors.
type RopeScaling struct {
	Type        string    `json:"type"`
	RopeType    string    `json:"rope_type"`
	LongFactor  []float32 `json:"long_factor"`
	ShortFactor []float32 `json:"short_factor"`
}

func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch translates either Phi generation into the neutral transformer declaration.
func (c *Config) Arch() (model.Arch, error) {
	if c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.VocabSize <= 0 || c.HiddenSize%c.NumAttentionHeads != 0 {
		return model.Arch{}, core.NewError("phi.Config.Arch: invalid transformer geometry")
	}
	headDim := c.HiddenSize / c.NumAttentionHeads
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if kvHeads <= 0 || c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("phi.Config.Arch: invalid key/value head count")
	}
	if c.ModelType == "phi3" && kvHeads != c.NumAttentionHeads {
		return model.Arch{}, core.NewError("phi.Config.Arch: Phi-3 fused qkv requires equal query and key/value head counts")
	}
	rotaryDim, err := (attn.RopeParams{HeadDim: headDim, PartialRotaryFactor: c.PartialRotaryFactor}).RotaryDim()
	if err != nil {
		return model.Arch{}, core.E("phi.Config.Arch", "resolve partial rotary", err)
	}
	eps := c.RMSNormEps
	if c.ModelType == "phi" {
		eps = c.LayerNormEps
	}
	if eps <= 0 {
		return model.Arch{}, core.NewError("phi.Config.Arch: norm epsilon must be > 0")
	}
	theta := c.RopeTheta
	if theta == 0 {
		theta = 10000
	}
	if theta < 0 {
		return model.Arch{}, core.NewError("phi.Config.Arch: rope theta must be > 0")
	}
	layerTypes := make([]string, c.NumHiddenLayers)
	for i := range layerTypes {
		layerTypes[i] = "full_attention"
	}
	layers := model.DeriveLayers(layerTypes, 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = headDim, kvHeads
	}
	a := model.Arch{Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: kvHeads, HeadDim: headDim, GlobalHeadDim: headDim, GlobalKVHeads: kvHeads, FF: c.IntermediateSize, Vocab: c.VocabSize, Eps: eps, AttnScale: float32(1 / math.Sqrt(float64(headDim))), EmbedScale: 1, RopeBase: theta, RopeLocalBase: theta, RopeScale: 1, RotaryDim: rotaryDim, RotaryDimLocal: rotaryDim, SlidingWindow: c.SlidingWindow, TieWordEmbeddings: c.TieWordEmbeddings, Layer: layers}
	if c.RopeScaling != nil {
		kind := c.RopeScaling.RopeType
		if kind == "" {
			kind = c.RopeScaling.Type
		}
		if kind != "longrope" || c.OriginalMaxPositionEmbeddings <= 0 {
			return model.Arch{}, core.NewError("phi.Config.Arch: unsupported rope scaling")
		}
		a.RopeFreqs, err = longRoPEFreqs(theta, rotaryDim, c.RopeScaling.LongFactor)
		if err != nil {
			return model.Arch{}, err
		}
		a.RopeShortFreqs, err = longRoPEFreqs(theta, rotaryDim, c.RopeScaling.ShortFactor)
		if err != nil {
			return model.Arch{}, err
		}
		a.RopeOriginalContext = c.OriginalMaxPositionEmbeddings
	}
	return a, nil
}

func longRoPEFreqs(theta float32, rotaryDim int, factors []float32) ([]float32, error) {
	if len(factors) != rotaryDim/2 {
		return nil, core.NewError("phi.longRoPEFreqs: factor count must equal rotary pairs")
	}
	out := make([]float32, len(factors))
	for i, factor := range factors {
		if factor <= 0 {
			return nil, core.NewError("phi.longRoPEFreqs: factors must be > 0")
		}
		out[i] = float32(math.Pow(float64(theta), -float64(2*i)/float64(rotaryDim)) / float64(factor))
	}
	return out, nil
}
