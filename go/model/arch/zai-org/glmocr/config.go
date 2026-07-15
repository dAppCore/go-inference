// SPDX-Licence-Identifier: EUPL-1.2

// Package glmocr declares GLM-OCR's vision-language architecture — model_type "glm_ocr", a
// multimodal wrapper whose nested text_config carries the alias "glm_ocr_text" — to the reactive
// loader (https://huggingface.co/zai-org/GLM-OCR). Registered so a checkpoint resolves to a
// named, recognised architecture — distinct from the existing ../glm4 dense-text arch — rather
// than "unknown model architecture"; the vision tower and OCR-decoder forward are NOT
// implemented — Arch and Composed both refuse cleanly (see register.go).
package glmocr

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// VisionConfig is the GLM-OCR ViT tower shape (nested vision_config, model_type
// "glm_ocr_vision") — captured for recognition only; no vision forward is implemented against it.
type VisionConfig struct {
	ModelType     string `json:"model_type"`
	HiddenSize    int    `json:"hidden_size"`
	Depth         int    `json:"depth"`
	NumHeads      int    `json:"num_heads"`
	PatchSize     int    `json:"patch_size"`
	ImageSize     int    `json:"image_size"`
	OutHiddenSize int    `json:"out_hidden_size"`
}

// TextConfig is the GLM-OCR text-decoder subset (nested text_config, model_type
// "glm_ocr_text") — a GQA transformer with mrope-style rotary parameters.
type TextConfig struct {
	ModelType         string `json:"model_type"`
	HiddenSize        int    `json:"hidden_size"`
	IntermediateSize  int    `json:"intermediate_size"`
	NumHiddenLayers   int    `json:"num_hidden_layers"`
	NumAttentionHeads int    `json:"num_attention_heads"`
	NumKeyValueHeads  int    `json:"num_key_value_heads"`
	HeadDim           int    `json:"head_dim"`
	VocabSize         int    `json:"vocab_size"`
}

// Config is the architecture-relevant subset of a GLM-OCR config.json:
// https://huggingface.co/zai-org/GLM-OCR/resolve/main/config.json — a multimodal wrapper whose
// hidden/layers/vocab live under the nested text_config, mirroring the gemma4/composed
// wrapper-plus-text_config-alias pattern used elsewhere in this tree.
type Config struct {
	ModelType    string        `json:"model_type"`
	TextConfig   *TextConfig   `json:"text_config"`
	VisionConfig *VisionConfig `json:"vision_config"`
}

// ParseConfig parses a GLM-OCR config.json far enough to report what it is; it never fails on a
// well-formed document — the forward refusal lives in Arch, not here.
func ParseConfig(data []byte) (*Config, error) {
	var cfg Config
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.NewError("glmocr.ParseConfig: config.json parse failed")
	}
	return &cfg, nil
}

// InferFromWeights is a no-op: Arch refuses unconditionally, so no dimension this config omits
// is ever consumed downstream of a weight shape.
func (c *Config) InferFromWeights(weights map[string]safetensors.Tensor) { _ = weights }

// Arch deliberately refuses: GLM-OCR's vision tower and OCR-decoder forward are not implemented
// in this engine. ParseConfig (above) still gives LookupArch/Parse enough to report
// hidden_size/layers/vocab and confirm a vision_config is present.
func (c *Config) Arch() (model.Arch, error) {
	mt := c.ModelType
	if mt == "" {
		mt = "glm_ocr"
	}
	return model.Arch{}, core.NewError(mt + " (GLM-OCR) is a recognised OCR vision-language arch; its vision encoder + OCR decoder forward is not yet implemented in this engine")
}
