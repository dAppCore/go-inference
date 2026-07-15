// SPDX-Licence-Identifier: EUPL-1.2

// Package deepseek declares DeepSeek-V2/V3 sparse MoE and MLA geometry.
package deepseek

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// Config is the architecture-relevant subset shared by DeepSeek-V2 and V3.
type Config struct {
	HiddenSize          int     `json:"hidden_size"`
	IntermediateSize    int     `json:"intermediate_size"`
	MoEIntermediateSize int     `json:"moe_intermediate_size"`
	NumHiddenLayers     int     `json:"num_hidden_layers"`
	NumAttentionHeads   int     `json:"num_attention_heads"`
	VocabSize           int     `json:"vocab_size"`
	KVLoRARank          int     `json:"kv_lora_rank"`
	QLoRARank           *int    `json:"q_lora_rank"`
	QKNoPEHeadDim       int     `json:"qk_nope_head_dim"`
	QKRoPEHeadDim       int     `json:"qk_rope_head_dim"`
	ValueHeadDim        int     `json:"v_head_dim"`
	NumRoutedExperts    int     `json:"n_routed_experts"`
	NumSharedExperts    int     `json:"n_shared_experts"`
	NumExpertsPerTok    int     `json:"num_experts_per_tok"`
	FirstKDenseReplace  int     `json:"first_k_dense_replace"`
	MoELayerFreq        int     `json:"moe_layer_freq"`
	RMSNormEps          float32 `json:"rms_norm_eps"`
	RopeTheta           float32 `json:"rope_theta"`
	RoutedScalingFactor float32 `json:"routed_scaling_factor"`
	ScoringFunc         string  `json:"scoring_func"`
	NormTopKProb        bool    `json:"norm_topk_prob"`
	TieWordEmbeddings   *bool   `json:"tie_word_embeddings"`
}

// QHeadDim is the decompressed query/key width per head.
func (c Config) QHeadDim() int { return c.QKNoPEHeadDim + c.QKRoPEHeadDim }

// KVHeadDim is the decompressed key width per head.
func (c Config) KVHeadDim() int { return c.QKNoPEHeadDim + c.QKRoPEHeadDim }

// Validate checks the MLA and sparse-expert dimensions without lowering MLA to standard attention.
func (c Config) Validate() error {
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.VocabSize <= 0 || c.KVLoRARank <= 0 || c.QKNoPEHeadDim <= 0 || c.QKRoPEHeadDim <= 0 || c.ValueHeadDim <= 0 {
		return core.NewError("deepseek.Config.Validate: MLA dimensions must be > 0")
	}
	if c.NumRoutedExperts <= 0 || c.NumExpertsPerTok <= 0 || c.MoEIntermediateSize <= 0 || c.NumExpertsPerTok > c.NumRoutedExperts {
		return core.NewError("deepseek.Config.Validate: invalid sparse-expert geometry")
	}
	return nil
}

// InferFromWeights satisfies model.ArchConfig. DeepSeek declares its MLA geometry.
func (c *Config) InferFromWeights(weights map[string]safetensors.Tensor) { weights = nil }

// Arch deliberately refuses to misrepresent MLA as standard Q/K/V attention.
func (c Config) Arch() (model.Arch, error) {
	if err := c.Validate(); err != nil {
		return model.Arch{}, err
	}
	return model.Arch{}, core.NewError("deepseek.Config.Arch: MLA requires a separate attention implementation")
}
