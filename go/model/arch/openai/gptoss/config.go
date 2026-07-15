// SPDX-Licence-Identifier: EUPL-1.2

// Package gptoss declares openai/gpt-oss-* (GptOssForCausalLM) to the backend-neutral reactive model
// loader. Unlike its openai/ siblings (privacyfilter is a token-classifier, whisper is an ASR
// encoder-decoder), GPT-OSS IS a generative MoE causal-LM — a decoder-only transformer alternating sliding-
// window and full-attention layers, YaRN long-context rope, and a routed MoE MLP — the same decode shape
// lem's other MoE arches (e.g. Qwen MoE) already fit. This package registers the model_type and parses its
// real config.json, so the checkpoint is recognised and its geometry captured; it deliberately stops short
// of a Weights mapping and Assemble wiring, which have not been confirmed against a real GPT-OSS safetensors
// index (the MoE expert tensor layout + mxfp4 quantisation are their own lane). Config.Arch therefore still
// refuses, but with a "recognised + configured, not yet validated" message rather than an architectural
// mismatch — the honest state per lem's no-test-no-claim rule: nothing here has driven a forward pass, so
// nothing here claims generation works. See model/quant/jang/jang.go for the separate GGUF quant-scheme
// name mapping already registered for "gpt_oss" (a quant concern, not this arch registration).
//
// Source: https://huggingface.co/openai/gpt-oss-20b/resolve/main/config.json
package gptoss

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// RopeScaling is GPT-OSS's YaRN long-context extension. Unlike privacyfilter's "rope_parameters", GPT-OSS
// carries rope_theta at Config's top level and nests only the scaling factors here under "rope_scaling".
type RopeScaling struct {
	RopeType                      string  `json:"rope_type"`
	Factor                        float32 `json:"factor"`
	BetaFast                      float32 `json:"beta_fast"`
	BetaSlow                      float32 `json:"beta_slow"`
	OriginalMaxPositionEmbeddings int     `json:"original_max_position_embeddings"`
	Truncate                      bool    `json:"truncate"`
}

// Config is the architecture-relevant subset of a GptOssForCausalLM config.json.
type Config struct {
	ModelType             string      `json:"model_type"`
	HiddenAct             string      `json:"hidden_act"`
	HiddenSize            int         `json:"hidden_size"`
	IntermediateSize      int         `json:"intermediate_size"`
	NumHiddenLayers       int         `json:"num_hidden_layers"`
	NumAttentionHeads     int         `json:"num_attention_heads"`
	NumKeyValueHeads      int         `json:"num_key_value_heads"`
	HeadDim               int         `json:"head_dim"`
	VocabSize             int         `json:"vocab_size"`
	RMSNormEps            float32     `json:"rms_norm_eps"`
	RopeTheta             float32     `json:"rope_theta"`
	RopeScaling           RopeScaling `json:"rope_scaling"`
	MaxPositionEmbeddings int         `json:"max_position_embeddings"`
	SlidingWindow         int         `json:"sliding_window"`
	LayerTypes            []string    `json:"layer_types"`
	NumLocalExperts       int         `json:"num_local_experts"`
	NumExpertsPerTok      int         `json:"num_experts_per_tok"`
	ExpertsPerToken       int         `json:"experts_per_token"`
	SwigluLimit           float32     `json:"swiglu_limit"`
	TieWordEmbeddings     *bool       `json:"tie_word_embeddings"`
}

// ParseConfig parses a GPT-OSS (GptOssForCausalLM) Hugging Face config.
func ParseConfig(data []byte) (*Config, error) {
	var cfg Config
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.NewError("gptoss.ParseConfig: config.json parse failed")
	}
	return &cfg, nil
}

// InferFromWeights is a no-op: GPT-OSS's config.json declares every geometry field this package captures,
// and Config.Arch refuses before any dimension would be consulted for an Assemble — there is nothing to
// resolve from the checkpoint's weight shapes yet (the don't-guess rule; see qwen2.Config.InferFromWeights
// for the pattern the follow-up serving lane would extend).
func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch refuses: GPT-OSS is a generative MoE causal-LM that fits lem's decode path shape (unlike its
// openai/ siblings), and this package parses its real config — but the MoE Weights mapping and the
// end-to-end forward have not been validated against a real checkpoint in this engine, so Arch does not
// yet hand back a decode Arch for model.Assemble. The model_type is registered (model.LookupArch succeeds)
// and the config is genuinely parsed; serving is a follow-up, not a claim made here.
func (c *Config) Arch() (model.Arch, error) {
	return model.Arch{}, core.NewError(core.Sprintf(
		"gptoss.Config.Arch: gpt_oss is a generative MoE causal-LM (%d layers, %d local experts / "+
			"%d per token, vocab %d) that fits lem's decode path shape — config recognised and parsed, but "+
			"the full forward is not yet validated end-to-end in this engine; recognised + configured, "+
			"serving is a follow-up",
		c.NumHiddenLayers, c.NumLocalExperts, c.NumExpertsPerTok, c.VocabSize))
}
