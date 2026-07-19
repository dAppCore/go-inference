// SPDX-Licence-Identifier: EUPL-1.2

// Package deepseekvl2 declares DeepSeek-VL2's vision-language architecture (model_type
// "deepseek_vl_v2") to the reactive loader AND implements DeepSeek-OCR's forward: a dual-tower
// vision encoder (SAM ViT-B feeding CLIP-L, DeepSeek-OCR's "DeepEncoder") plus a DeepSeek-V2-lite
// MoE decoder — the arch DeepSeek-OCR / OCR-2 checkpoints reuse unchanged
// (https://huggingface.co/deepseek-ai/DeepSeek-OCR). It is registered so model.LookupArch gives a
// user pointing lem at a DeepSeek-OCR checkpoint direction, not "unknown model architecture"; the
// generate/serve decoder-only causal-LM path still refuses (Arch, see below) — DeepSeek-OCR's
// forward runs through `lem ocr` instead (own loader, own session: see ocr.go's doc comment for
// why, mirroring whisper's ASR shape).
package deepseekvl2

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// VisionConfig is the DeepSeek-VL2 dual-tower (CLIP-L-14-224 + SAM ViT-B) vision encoder's
// top-level shape — captured for recognition/reporting only. Unlike LanguageConfig's fields
// below, the tower's actual dimensions are NOT read from here at load time: deepencoder.py's
// build_sam_vit_b()/build_clip_l() hardcode every dimension (768/12/12 for SAM, 1024/24/16 for
// CLIP — see samGeometry/clipGeometry in vision_sam.go/vision_clip.go) and take no config
// argument at all, so this checkpoint's vision_config.width sub-dict is documentation of that
// hardcoding, not a load-bearing input. VisionForward's own checks (input channel count, patch
// divisibility) are what actually validate a real checkpoint's tower shape.
type VisionConfig struct {
	ModelType string  `json:"model_type"`
	ModelName string  `json:"model_name"` // e.g. "deeplip_b_l"
	ImageSize int     `json:"image_size"`
	MLPRatio  float32 `json:"mlp_ratio"`
}

// LanguageConfig is the DeepSeek-VL2 MoE text-decoder subset, nested under config.json's
// "language_config" — kept for recognition/reporting only (see Config's doc comment: the REAL
// decoder geometry this package's weight loader and forward actually use comes from the
// TOP-LEVEL fields below, confirmed by instantiating DeepseekOCRConfig — the checkpoint's own
// custom_code class — directly and reading back every field DeepseekV2Model/DeepseekV2Attention/
// DeepseekV2MoE consult: they all read config.hidden_size/config.num_hidden_layers/etc., never
// config.language_config.hidden_size).
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
// "deepseek_vl_v2". The decoder-geometry fields below are read from the JSON document's TOP
// LEVEL (config.json duplicates them under the nested "language_config" key too, but that nested
// copy is never consulted by the checkpoint's own custom_code — see LanguageConfig's doc
// comment). Several of these fields (NRoutedExperts/NSharedExperts/NumExpertsPerTok/UseMLA/
// FirstKDenseReplace) select DIFFERENT ARITHMETIC per layer — they are read here, not guessed —
// but downstream fields DeepseekV2Config.__init__ defaults when config.json omits them (RoPE
// theta 10000.0, RMSNorm eps 1e-6, SiLU activation, softmax scoring, no top-k-weight
// renormalisation, routed-scaling factor 1.0) are NOT present in this checkpoint's config.json at
// all (confirmed absent by reading the raw document) — this package's decoder forward hardcodes
// those as constants (decoderRopeTheta etc. in decoder.go), matching DeepseekV2Config's own
// Python-side defaults exactly, not an assumption.
type Config struct {
	ModelType      string          `json:"model_type"`
	LanguageConfig *LanguageConfig `json:"language_config"`
	VisionConfig   *VisionConfig   `json:"vision_config"`

	HiddenSize            int  `json:"hidden_size"`
	IntermediateSize      int  `json:"intermediate_size"`
	MoEIntermediateSize   int  `json:"moe_intermediate_size"`
	NumHiddenLayers       int  `json:"num_hidden_layers"`
	NumAttentionHeads     int  `json:"num_attention_heads"`
	NumKeyValueHeads      int  `json:"num_key_value_heads"`
	VocabSize             int  `json:"vocab_size"`
	NRoutedExperts        int  `json:"n_routed_experts"`
	NSharedExperts        int  `json:"n_shared_experts"`
	NumExpertsPerTok      int  `json:"num_experts_per_tok"`
	FirstKDenseReplace    int  `json:"first_k_dense_replace"`
	UseMLA                bool `json:"use_mla"`
	MaxPositionEmbeddings int  `json:"max_position_embeddings"`
	BOSTokenID            int  `json:"bos_token_id"`
	EOSTokenID            int  `json:"eos_token_id"`
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

// InferFromWeights is a no-op: every dimension the decoder/tower forward needs is read from
// config.json's own top-level fields (never a guessed or weight-shape-derived default) — see
// Config's doc comment.
func (c *Config) InferFromWeights(weights map[string]safetensors.Tensor) { _ = weights }

// Arch deliberately refuses: DeepSeek-OCR's dual-tower vision encoder + MoE decoder forward does
// not fit model.Assemble's decoder-only causal-LM contract (an image-conditioned prefill merge
// ahead of the token decode loop, not a plain embed→layers→lm_head chain) — there is no neutral
// decode Arch to derive, mirroring whisper.Config.Arch's encoder-decoder refusal. OCR IS fully
// implemented in this package (weights.go/vision_*.go/decoder.go/ocr.go), driven by
// `lem ocr --model <dir> <image>` (ocr.go's Load/Model.OCR), which never calls Arch.
func (c *Config) Arch() (model.Arch, error) {
	mt := c.ModelType
	if mt == "" {
		mt = "deepseek_vl_v2"
	}
	return model.Arch{}, core.NewError(mt + " (DeepSeek-OCR) is a dual-tower vision encoder + MoE decoder OCR architecture, not a decoder-only causal-LM — there is no neutral decode Arch to derive; use `lem ocr --model <dir> <image>` instead of generate/serve")
}
