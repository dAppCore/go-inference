// SPDX-Licence-Identifier: EUPL-1.2

// Package glm4 declares the dense Hugging Face GLM-4 architecture.
package glm4

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/attn"
	"dappco.re/go/inference/model/safetensors"
)

// Config is sourced from zai-org/GLM-4-9B-0414 config.json.
type Config struct {
	ModelType           string  `json:"model_type"`
	HiddenSize          int     `json:"hidden_size"`
	IntermediateSize    int     `json:"intermediate_size"`
	NumHiddenLayers     int     `json:"num_hidden_layers"`
	NumAttentionHeads   int     `json:"num_attention_heads"`
	NumKeyValueHeads    int     `json:"num_key_value_heads"`
	HeadDim             int     `json:"head_dim"`
	VocabSize           int     `json:"vocab_size"`
	RMSNormEps          float32 `json:"rms_norm_eps"`
	RopeTheta           float32 `json:"rope_theta"`
	PartialRotaryFactor float32 `json:"partial_rotary_factor"`
	TieWordEmbeddings   *bool   `json:"tie_word_embeddings"`
}

func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}
func (c *Config) Arch() (model.Arch, error) {
	if c.ModelType != "glm4" || c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.NumKeyValueHeads <= 0 || c.HeadDim <= 0 || c.VocabSize <= 0 || c.RMSNormEps <= 0 || c.RopeTheta <= 0 || c.NumAttentionHeads%c.NumKeyValueHeads != 0 {
		return model.Arch{}, core.NewError("glm4.Config.Arch: invalid dense GLM-4 declaration")
	}
	rd, err := (attn.RopeParams{HeadDim: c.HeadDim, PartialRotaryFactor: c.PartialRotaryFactor}).RotaryDim()
	if err != nil {
		return model.Arch{}, core.E("glm4.Config.Arch", "partial rotary", err)
	}
	l := model.DeriveLayers(make([]string, c.NumHiddenLayers), 0)
	for i := range l {
		l[i].HeadDim, l[i].KVHeads = c.HeadDim, c.NumKeyValueHeads
	}
	return model.Arch{Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: c.NumKeyValueHeads, HeadDim: c.HeadDim, GlobalHeadDim: c.HeadDim, GlobalKVHeads: c.NumKeyValueHeads, FF: c.IntermediateSize, Vocab: c.VocabSize, Eps: c.RMSNormEps, AttnScale: float32(1 / math.Sqrt(float64(c.HeadDim))), EmbedScale: 1, RopeBase: c.RopeTheta, RopeLocalBase: c.RopeTheta, RopeScale: 1, RotaryDim: rd, RotaryDimLocal: rd, TieWordEmbeddings: c.TieWordEmbeddings, Layer: l}, nil
}
func init() {
	w := model.StandardWeightNames()
	w.MLPNorm = ".post_attention_layernorm.weight"
	w.PostAttnNorm = ".post_self_attn_layernorm.weight"
	w.PostFFNorm = ".post_mlp_layernorm.weight"
	w.QNorm = ""
	w.KNorm = ""
	model.RegisterArch(model.ArchSpec{ModelTypes: []string{"glm4"}, Parse: parse, Weights: w, Normalize: func(in map[string]safetensors.Tensor) map[string]safetensors.Tensor {
		out := in
		for i := 0; ; i++ {
			p := core.Sprintf("model.layers.%d.mlp.", i)
			if _, ok := out[p+"gate_up_proj.weight"]; !ok {
				break
			}
			out = attn.SplitContiguousGateUp(out, p+"gate_up_proj", p+"gate_proj", p+"up_proj")
		}
		return out
	}})
}
func parse(data []byte) (model.ArchConfig, error) {
	var c Config
	if r := core.JSONUnmarshal(data, &c); !r.OK {
		return nil, core.NewError("glm4.Parse: config.json parse failed")
	}
	return &c, nil
}
