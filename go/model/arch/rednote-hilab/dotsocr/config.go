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

// VisionConfig is the DOTS-OCR ViT tower shape (nested vision_config), field-complete against
// https://huggingface.co/rednote-hilab/dots.ocr/resolve/main/config.json's vision_config block —
// every field EncodeImage (vision.go) reads to run the real NaViT-style tower forward. EmbedDim
// is the per-patch tower width (patch_embed/blocks/RoPE all operate at this width); HiddenSize is
// the width AFTER PatchMerger's projection (matches the text decoder's HiddenSize) — the two
// happen to be equal for the shipped checkpoint (1536 either way) but are read as distinct fields
// since the reference config declares them separately (DotsVisionConfig.__init__).
type VisionConfig struct {
	EmbedDim          int     `json:"embed_dim"`
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumChannels       int     `json:"num_channels"`
	PatchSize         int     `json:"patch_size"`
	SpatialMergeSize  int     `json:"spatial_merge_size"`
	TemporalPatchSize int     `json:"temporal_patch_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	// UseBias gates the vision blocks' qkv/proj bias (DotsVisionBlock reads config.use_bias
	// dynamically, unlike the text decoder's q/k/v — see Config.AttentionBias's doc comment).
	UseBias bool `json:"use_bias"`
	// PostNorm gates the tower's post_trunk_norm (config.post_norm); true for every published
	// DOTS-OCR checkpoint, but EncodeImage still branches on it rather than assuming.
	PostNorm bool `json:"post_norm"`
}

// Config is the architecture-relevant subset of a DOTS-OCR config.json:
// https://huggingface.co/rednote-hilab/dots.ocr/resolve/main/config.json. The text decoder's
// fields sit flat at the top level (the Qwen2Config shape — DotsOCRConfig subclasses Qwen2Config
// directly, and its decoder IS an unmodified Qwen2ForCausalLM, per modeling_dots_ocr.py); a
// nested vision_config carries the ViT tower. Field-complete against the real checkpoint's
// config.json (verified against rednote-hilab/dots.ocr's shipped file) — every field
// the host forward (decoder.go/vision.go/ocr.go) reads.
type Config struct {
	ModelType             string  `json:"model_type"`
	HiddenSize            int     `json:"hidden_size"`
	IntermediateSize      int     `json:"intermediate_size"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	NumKeyValueHeads      int     `json:"num_key_value_heads"`
	VocabSize             int     `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	// AttentionBias is parsed for introspection/completeness only — Qwen2Attention (the real
	// decoder's actual module, per modeling_dots_ocr.py's Qwen2ForCausalLM base) hard-codes
	// q_proj/k_proj/v_proj bias=True and o_proj bias=False UNCONDITIONALLY in the shipped
	// transformers 5.5 source, never consulting this config field — decoder.go's loader mirrors
	// that hard-coded reality (confirmed against the real safetensors tensor names, "never
	// guessed": q/k/v carry .bias, o_proj does not), not this flag.
	AttentionBias bool `json:"attention_bias"`
	// TieWordEmbeddings is false for every published DOTS-OCR checkpoint (a standalone
	// lm_head.weight tensor ships separately from model.embed_tokens.weight) — weights.go reads
	// this to decide whether lm_head.weight is required on disk or falls back to the tied
	// embedding, rather than assuming either shape.
	TieWordEmbeddings bool `json:"tie_word_embeddings"`
	// ImageTokenID is the placeholder id (<|imgpad|>, 151665) prompt.go expands and ocr.go's
	// vision-embedding scatter matches against in input_ids.
	ImageTokenID int           `json:"image_token_id"`
	VideoTokenID int           `json:"video_token_id"`
	VisionConfig *VisionConfig `json:"vision_config"`
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
