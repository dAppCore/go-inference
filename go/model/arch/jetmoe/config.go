// SPDX-Licence-Identifier: EUPL-1.2

// Package jetmoe declares the JetMoE sparse transformer architecture.
package jetmoe

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// Config is the architecture-relevant subset of a JetMoE config.json.
type Config struct {
	ModelType         string  `json:"model_type"`
	HiddenSize        int     `json:"hidden_size"`
	FFNHiddenSize     int     `json:"ffn_hidden_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	KVChannels        int     `json:"kv_channels"`
	MoENumExperts     int     `json:"moe_num_experts"`
	MoETopK           int     `json:"moe_top_k"`
	VocabSize         int     `json:"vocab_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	LayerNormEpsilon  float32 `json:"layer_norm_epsilon"`
	RopeTheta         float32 `json:"rope_theta"`
	RotaryPercent     float32 `json:"rotary_percent"`
	TieWordEmbeddings *bool   `json:"tie_word_embeddings"`
}

// InferFromWeights satisfies model.ArchConfig. Published JetMoE configs declare their geometry.
func (c *Config) InferFromWeights(weights map[string]safetensors.Tensor) { weights = nil }

// Arch deliberately refuses: every published JetMoE checkpoint routes attention through
// per-expert query/output projections sharing one KV projection (Mixture-of-Attention, MoA —
// see MOA_GAP.md), a primitive the engine does not implement. Misreading MoA as standard dense
// attention would silently drop the routed half, so Arch refuses rather than resolve a wrong
// geometry — the same early-refusal shape as deepseek's MLA gap
// (model/arch/deepseek-ai/deepseek/config.go Config.Arch).
func (c Config) Arch() (model.Arch, error) {
	if c.HiddenSize <= 0 || c.FFNHiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.MoENumExperts <= 0 || c.MoETopK <= 0 || c.VocabSize <= 0 {
		return model.Arch{}, core.NewError("jetmoe.Config.Arch: all architecture dimensions must be > 0")
	}
	if c.MoETopK > c.MoENumExperts {
		return model.Arch{}, core.NewError("jetmoe.Config.Arch: moe_top_k exceeds moe_num_experts")
	}
	return model.Arch{}, core.NewError("jetmoe.Config.Arch: Mixture-of-Attention (MoA) requires routed query/output attention projections with shared KV; the engine implements no such attention primitive")
}
