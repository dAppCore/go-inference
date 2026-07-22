// SPDX-Licence-Identifier: EUPL-1.2

// Package glmocr declares GLM-OCR's vision-language architecture — model_type "glm_ocr", a
// multimodal wrapper whose nested text_config carries the alias "glm_ocr_text" — to the reactive
// loader (https://huggingface.co/zai-org/GLM-OCR). Registered so a checkpoint resolves to a
// named, recognised architecture — distinct from the existing ../glm4 dense-text arch — rather
// than "unknown model architecture". model.LookupArch's shared entry points (Arch — the
// generate/serve causal-LM path — and Composed, engine/metal's factory path) both still refuse
// cleanly (see register.go): a vision-language OCR forward does not fit the neutral decoder-
// only-causal-LM contract those paths assemble. This package's OWN forward — image in, OCR text
// out — IS implemented, host-f32, as a standalone library path that never enters
// model.Assemble/model.LookupArch: see ocr.go's Load/Model.OCR, mirroring
// ../../openai/whisper's "own loader, own session" shape for the same reason (an ASR
// encoder-decoder forward doesn't fit that contract either).
package glmocr

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// VisionConfig is the GLM-OCR ViT tower shape (nested vision_config, model_type
// "glm_ocr_vision") — the CogViT-class tower: Conv3d patch embed (temporal duplication for a
// static image), 2D (height/width) rotary position embeddings, depth pre-LN blocks with
// per-head RMSNorm on Q/K, a spatial_merge_size×spatial_merge_size Conv2d downsample, and a
// GLU patch merger projecting into the text decoder's hidden size — read verbatim off
// https://huggingface.co/zai-org/GLM-OCR/resolve/main/config.json and transformers'
// modeling_glm_ocr.py (GlmOcrVisionModel), never guessed.
type VisionConfig struct {
	ModelType         string  `json:"model_type"`
	HiddenSize        int     `json:"hidden_size"`
	Depth             int     `json:"depth"`
	NumHeads          int     `json:"num_heads"`
	PatchSize         int     `json:"patch_size"`
	ImageSize         int     `json:"image_size"`
	OutHiddenSize     int     `json:"out_hidden_size"`
	InChannels        int     `json:"in_channels"`
	IntermediateSize  int     `json:"intermediate_size"`
	HiddenAct         string  `json:"hidden_act"`
	AttentionBias     bool    `json:"attention_bias"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	SpatialMergeSize  int     `json:"spatial_merge_size"`
	TemporalPatchSize int     `json:"temporal_patch_size"`
}

// RopeParameters is GLM-OCR's text-decoder rotary config (nested text_config.rope_parameters)
// — mrope_section splits each head's half-width rotary frequencies into
// [temporal,height,width] bands (summing to head_dim/2·partial_rotary_factor) for the 3D
// multimodal RoPE Glm4v-family text decoders use — see rope.go's textCosSin.
type RopeParameters struct {
	RopeType            string  `json:"rope_type"`
	RopeTheta           float32 `json:"rope_theta"`
	MropeSection        []int   `json:"mrope_section"`
	PartialRotaryFactor float32 `json:"partial_rotary_factor"`
}

// TextConfig is the GLM-OCR text-decoder subset (nested text_config, model_type
// "glm_ocr_text") — a GQA transformer (GLM-4-style sandwich norm: post_self_attn_layernorm/
// post_mlp_layernorm on the residual branches, interleaved-pair rotary) with 3D mrope-style
// rotary parameters. num_nextn_predict_layers names the MTP layer this package's Load NEVER
// reads (the checkpoint's model.language_model.layers.16.* — see weights.go's doc comment).
type TextConfig struct {
	ModelType         string          `json:"model_type"`
	HiddenSize        int             `json:"hidden_size"`
	IntermediateSize  int             `json:"intermediate_size"`
	NumHiddenLayers   int             `json:"num_hidden_layers"`
	NumAttentionHeads int             `json:"num_attention_heads"`
	NumKeyValueHeads  int             `json:"num_key_value_heads"`
	HeadDim           int             `json:"head_dim"`
	VocabSize         int             `json:"vocab_size"`
	RMSNormEps        float32         `json:"rms_norm_eps"`
	RopeParameters    *RopeParameters `json:"rope_parameters"`
	TieWordEmbeddings *bool           `json:"tie_word_embeddings"`
	PadTokenID        int             `json:"pad_token_id"`
	NumNextNPredict   int             `json:"num_nextn_predict_layers"`
}

// Config is the architecture-relevant subset of a GLM-OCR config.json:
// https://huggingface.co/zai-org/GLM-OCR/resolve/main/config.json — a multimodal wrapper whose
// hidden/layers/vocab live under the nested text_config, mirroring the gemma4/composed
// wrapper-plus-text_config-alias pattern used elsewhere in this tree. ImageTokenID/
// ImageStartTokenID/ImageEndTokenID place a decoded image's merged vision tokens inside the
// text token stream (prompt.go); VideoStartTokenID/VideoEndTokenID/VideoTokenID are recognised
// but never produced — video is a named boundary this package's OCR forward does not serve.
type Config struct {
	ModelType         string        `json:"model_type"`
	TextConfig        *TextConfig   `json:"text_config"`
	VisionConfig      *VisionConfig `json:"vision_config"`
	ImageTokenID      int32         `json:"image_token_id"`
	ImageStartTokenID int32         `json:"image_start_token_id"`
	ImageEndTokenID   int32         `json:"image_end_token_id"`
	VideoTokenID      int32         `json:"video_token_id"`
	VideoStartTokenID int32         `json:"video_start_token_id"`
	VideoEndTokenID   int32         `json:"video_end_token_id"`
}

// ParseConfig parses a GLM-OCR config.json far enough to report what it is; it never fails on a
// well-formed document — the forward refusal lives in Arch, not here. VisionConfig.InChannels
// defaults to 3 (RGB) when the document omits it — every published GLM-OCR config.json does
// (transformers' GlmOcrVisionConfig dataclass default fills it the same way), so an explicit
// zero-value here would silently under-declare the checkpoint's real geometry.
func ParseConfig(data []byte) (*Config, error) {
	var cfg Config
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.NewError("glmocr.ParseConfig: config.json parse failed")
	}
	if cfg.VisionConfig != nil && cfg.VisionConfig.InChannels == 0 {
		cfg.VisionConfig.InChannels = 3
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
