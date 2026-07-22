// SPDX-Licence-Identifier: EUPL-1.2

// Package ernie45 declares Baidu ERNIE 4.5 dense checkpoints.
package ernie45

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	"math"
)

type Config struct {
	ModelType         string  `json:"model_type"`
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
}

func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}
func (c *Config) Arch() (model.Arch, error) {
	if c.ModelType != "ernie4_5" || c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.NumKeyValueHeads <= 0 || c.HeadDim <= 0 || c.VocabSize <= 0 || c.RMSNormEps <= 0 || c.RopeTheta <= 0 || c.NumAttentionHeads%c.NumKeyValueHeads != 0 {
		return model.Arch{}, core.NewError("ernie45.Config.Arch: invalid dense ERNIE 4.5 declaration")
	}
	l := model.DeriveLayers(make([]string, c.NumHiddenLayers), 0)
	for i := range l {
		l[i].HeadDim, l[i].KVHeads = c.HeadDim, c.NumKeyValueHeads
	}
	return model.Arch{Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: c.NumKeyValueHeads, HeadDim: c.HeadDim, GlobalHeadDim: c.HeadDim, GlobalKVHeads: c.NumKeyValueHeads, FF: c.IntermediateSize, Vocab: c.VocabSize, Eps: c.RMSNormEps, AttnScale: float32(1 / math.Sqrt(float64(c.HeadDim))), EmbedScale: 1, RopeBase: c.RopeTheta, RopeLocalBase: c.RopeTheta, RopeScale: 1, RotaryDim: c.HeadDim, RotaryDimLocal: c.HeadDim, TieWordEmbeddings: c.TieWordEmbeddings, Layer: l}, nil
}
func init() {
	w := model.StandardWeightNames()
	w.MLPNorm = ".post_attention_layernorm.weight"
	w.PostAttnNorm = ""
	w.PostFFNorm = ""
	w.QNorm = ""
	w.KNorm = ""
	model.RegisterArch(model.ArchSpec{ModelTypes: []string{"ernie4_5"}, Parse: parse, Weights: w})
}
func parse(data []byte) (model.ArchConfig, error) {
	var c Config
	if r := core.JSONUnmarshal(data, &c); !r.OK {
		return nil, core.NewError("ernie45.Parse: config.json parse failed")
	}
	return &c, nil
}
