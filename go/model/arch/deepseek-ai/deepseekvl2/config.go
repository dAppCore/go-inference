// SPDX-Licence-Identifier: EUPL-1.2

// Package deepseekvl2 declares DeepSeek-VL2's vision-language architecture (model_type
// "deepseek_vl_v2") to the reactive loader — the arch DeepSeek-OCR / OCR-2 checkpoints reuse
// unchanged (https://huggingface.co/deepseek-ai/DeepSeek-OCR). It is registered so
// model.LookupArch gives a user pointing lem at a DeepSeek-OCR checkpoint direction, not
// "unknown model architecture"; the dual-tower (CLIP-L + SAM ViT-B) vision encoder and MoE
// language-decoder forward are NOT implemented — Arch and Composed both refuse cleanly (see
// register.go).
package deepseekvl2

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// VisionConfig is the DeepSeek-VL2 dual-tower (CLIP-L-14-224 + SAM ViT-B) vision encoder's
// top-level shape — captured for recognition only; the per-tower "width" breakdown is left
// unparsed (unknown JSON fields are simply ignored), since no vision forward reads it.
type VisionConfig struct {
	ModelType string  `json:"model_type"`
	ModelName string  `json:"model_name"` // e.g. "deeplip_b_l"
	ImageSize int     `json:"image_size"`
	MLPRatio  float32 `json:"mlp_ratio"`
}

// LanguageConfig is the DeepSeek-VL2 MoE text-decoder subset, nested under config.json's
// "language_config" (a DeepSeek-V2-shaped backbone, MLA disabled for the OCR checkpoint —
// distinct from ../deepseek's MLA-on config, hence its own package rather than a shared one).
type LanguageConfig struct {
	HiddenSize        int `json:"hidden_size"`
	IntermediateSize  int `json:"intermediate_size"`
	NumHiddenLayers   int `json:"num_hidden_layers"`
	NumAttentionHeads int `json:"num_attention_heads"`
	NumKeyValueHeads  int `json:"num_key_value_heads"`
	VocabSize         int `json:"vocab_size"`
	NumRoutedExperts  int `json:"n_routed_experts"`
	NumSharedExperts  int `json:"n_shared_experts"`
	NumExpertsPerTok  int `json:"num_experts_per_tok"`
}

// Config is the architecture-relevant subset of a DeepSeek-OCR / DeepSeek-VL2 config.json:
// https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/config.json — model_type
// "deepseek_vl_v2", a nested language_config MoE decoder plus a nested vision_config dual tower.
type Config struct {
	ModelType      string          `json:"model_type"`
	LanguageConfig *LanguageConfig `json:"language_config"`
	VisionConfig   *VisionConfig   `json:"vision_config"`
}

// ParseConfig parses a DeepSeek-OCR / DeepSeek-VL2 config.json far enough to report what it is;
// it never fails on a well-formed document — the forward refusal lives in Arch, not here.
func ParseConfig(data []byte) (*Config, error) {
	var cfg Config
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.NewError("deepseekvl2.ParseConfig: config.json parse failed")
	}
	return &cfg, nil
}

// InferFromWeights is a no-op: Arch refuses unconditionally, so no dimension this config omits
// is ever consumed downstream of a weight shape.
func (c *Config) InferFromWeights(weights map[string]safetensors.Tensor) { _ = weights }

// Arch deliberately refuses: DeepSeek-VL2's dual-tower vision encoder and MoE OCR-decoder
// forward are not implemented in this engine. ParseConfig (above) still gives LookupArch/Parse
// enough to report hidden_size/layers/vocab and confirm a vision_config is present.
func (c *Config) Arch() (model.Arch, error) {
	mt := c.ModelType
	if mt == "" {
		mt = "deepseek_vl_v2"
	}
	return model.Arch{}, core.NewError(mt + " (DeepSeek-OCR) is a recognised OCR vision-language arch; its vision encoder + OCR decoder forward is not yet implemented in this engine")
}
