// SPDX-Licence-Identifier: EUPL-1.2

// Package dotsocr declares DOTS-OCR's vision-language architecture — model_type "dots_ocr", and
// its "dots_ocr_1_5" successor — to the reactive loader (https://huggingface.co/rednote-hilab/dots.ocr).
// The text decoder is a flat, Qwen2Config-shaped GQA transformer (upstream's DotsOCRConfig
// subclasses transformers' Qwen2Config directly) plus a nested vision_config ViT tower. It is
// registered so a checkpoint resolves to a named, recognised architecture rather than "unknown
// model architecture"; the ViT vision tower and OCR-decoder forward are NOT implemented — Arch
// and Composed both refuse cleanly (see register.go).
package dotsocr

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// VisionConfig is the DOTS-OCR ViT tower shape (nested vision_config) — captured for
// recognition only; no vision forward is implemented against it.
type VisionConfig struct {
	HiddenSize        int `json:"hidden_size"`
	IntermediateSize  int `json:"intermediate_size"`
	NumHiddenLayers   int `json:"num_hidden_layers"`
	NumAttentionHeads int `json:"num_attention_heads"`
	PatchSize         int `json:"patch_size"`
	SpatialMergeSize  int `json:"spatial_merge_size"`
}

// Config is the architecture-relevant subset of a DOTS-OCR config.json:
// https://huggingface.co/rednote-hilab/dots.ocr/resolve/main/config.json. The text decoder's
// fields sit flat at the top level (the Qwen2Config shape); a nested vision_config carries the
// ViT tower.
type Config struct {
	ModelType         string        `json:"model_type"`
	HiddenSize        int           `json:"hidden_size"`
	IntermediateSize  int           `json:"intermediate_size"`
	NumHiddenLayers   int           `json:"num_hidden_layers"`
	NumAttentionHeads int           `json:"num_attention_heads"`
	NumKeyValueHeads  int           `json:"num_key_value_heads"`
	VocabSize         int           `json:"vocab_size"`
	RMSNormEps        float32       `json:"rms_norm_eps"`
	RopeTheta         float32       `json:"rope_theta"`
	VisionConfig      *VisionConfig `json:"vision_config"`
}

// ParseConfig parses a DOTS-OCR / DOTS-OCR-1.5 config.json far enough to report what it is; it
// never fails on a well-formed document — the forward refusal lives in Arch, not here.
func ParseConfig(data []byte) (*Config, error) {
	var cfg Config
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.NewError("dotsocr.ParseConfig: config.json parse failed")
	}
	return &cfg, nil
}

// InferFromWeights is a no-op: Arch refuses unconditionally, so no dimension this config omits
// is ever consumed downstream of a weight shape.
func (c *Config) InferFromWeights(weights map[string]safetensors.Tensor) { _ = weights }

// Arch deliberately refuses: DOTS-OCR's ViT vision tower and OCR-decoder forward are not
// implemented in this engine. ParseConfig (above) still gives LookupArch/Parse enough to report
// hidden_size/layers/vocab and confirm a vision_config is present.
func (c *Config) Arch() (model.Arch, error) {
	mt := c.ModelType
	if mt == "" {
		mt = "dots_ocr"
	}
	return model.Arch{}, core.NewError(mt + " (DOTS-OCR) is a recognised OCR vision-language arch; its vision encoder + OCR decoder forward is not yet implemented in this engine")
}
