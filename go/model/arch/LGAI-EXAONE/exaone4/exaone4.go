// SPDX-Licence-Identifier: EUPL-1.2

// Package exaone4 declares LG AI Research EXAONE 4 dense checkpoints.
package exaone4

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/arch/meta-llama/llama"
	"dappco.re/go/inference/model/safetensors"
)

type Config struct {
	ModelType            string             `json:"model_type"`
	HiddenSize           int                `json:"hidden_size"`
	IntermediateSize     int                `json:"intermediate_size"`
	NumHiddenLayers      int                `json:"num_hidden_layers"`
	NumAttentionHeads    int                `json:"num_attention_heads"`
	NumKeyValueHeads     int                `json:"num_key_value_heads"`
	HeadDim              int                `json:"head_dim"`
	VocabSize            int                `json:"vocab_size"`
	SlidingWindow        int                `json:"sliding_window"`
	SlidingWindowPattern string             `json:"sliding_window_pattern"`
	RMSNormEps           float32            `json:"rms_norm_eps"`
	RopeTheta            float32            `json:"rope_theta"`
	RopeScaling          *llama.RopeScaling `json:"rope_scaling"`
	TieWordEmbeddings    *bool              `json:"tie_word_embeddings"`
}

func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}
func (c *Config) Arch() (model.Arch, error) {
	if c.ModelType != "exaone4" || c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.NumKeyValueHeads <= 0 || c.HeadDim <= 0 || c.VocabSize <= 0 || c.RMSNormEps <= 0 || c.RopeTheta <= 0 || c.NumAttentionHeads%c.NumKeyValueHeads != 0 {
		return model.Arch{}, core.NewError("exaone4.Config.Arch: invalid EXAONE 4 declaration")
	}
	types := make([]string, c.NumHiddenLayers)
	for i := range types {
		if c.SlidingWindowPattern != "" && c.SlidingWindowPattern[i%len(c.SlidingWindowPattern)] != 'G' {
			types[i] = "sliding_attention"
		}
	}
	l := model.DeriveLayers(types, 0)
	for i := range l {
		l[i].HeadDim, l[i].KVHeads = c.HeadDim, c.NumKeyValueHeads
	}
	var freqs []float32
	if c.RopeScaling != nil {
		freqs = llama.Llama3InvFreqs(c.RopeTheta, *c.RopeScaling, c.HeadDim)
		if freqs == nil {
			return model.Arch{}, core.NewError("exaone4.Config.Arch: invalid llama3 rope scaling")
		}
	}
	return model.Arch{Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: c.NumKeyValueHeads, HeadDim: c.HeadDim, GlobalHeadDim: c.HeadDim, GlobalKVHeads: c.NumKeyValueHeads, FF: c.IntermediateSize, Vocab: c.VocabSize, Eps: c.RMSNormEps, AttnScale: float32(1 / math.Sqrt(float64(c.HeadDim))), EmbedScale: 1, RopeBase: c.RopeTheta, RopeLocalBase: c.RopeTheta, RopeScale: 1, RopeFreqs: freqs, RotaryDim: c.HeadDim, RotaryDimLocal: c.HeadDim, SlidingWindow: c.SlidingWindow, TieWordEmbeddings: c.TieWordEmbeddings, NormPlacement: model.NormPlacementPost, Layer: l}, nil
}
func init() {
	w := model.StandardWeightNames()
	w.AttnNorm = ""
	w.MLPNorm = ""
	w.PostAttnNorm = ".post_attention_layernorm.weight"
	w.PostFFNorm = ".post_feedforward_layernorm.weight"
	model.RegisterArch(model.ArchSpec{ModelTypes: []string{"exaone4"}, Parse: parse, Weights: w})
}
func parse(data []byte) (model.ArchConfig, error) {
	var c Config
	if r := core.JSONUnmarshal(data, &c); !r.OK {
		return nil, core.NewError("exaone4.Parse: config.json parse failed")
	}
	return &c, nil
}
