// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"strconv"
	"strings"
)

const (
	BOIToken   = "<|image>"
	ImageToken = "<|image|>"
	EOIToken   = "<image|>"
	VideoToken = "<|video|>"
	BOAToken   = "<|audio>"
	AudioToken = "<|audio|>"
	EOAToken   = "<audio|>"
)

// VisionConfig is the backend-neutral Gemma-4 vision metadata surface read
// from config.json / processor metadata.
type VisionConfig struct {
	ImageTokenID          int
	ImageTokenIndex       int
	VideoTokenID          int
	VideoTokenIndex       int
	BOITokenID            int
	BOITokenIndex         int
	EOITokenID            int
	EOITokenIndex         int
	SoftTokensPerImage    int
	MMTokensPerImage      int
	ModelType             string
	DType                 string
	ImageSize             int
	PatchSize             int
	NumChannels           int
	HiddenSize            int
	IntermediateSize      int
	NumHiddenLayers       int
	NumAttentionHeads     int
	NumKeyValueHeads      int
	HeadDim               int
	GlobalHeadDim         int
	PoolingKernelSize     int
	PositionEmbeddingSize int
	DefaultOutputLength   int
	HiddenActivation      string
	RMSNormEps            float64
	RoPEParameters        RoPEParameters
	Standardize           bool
	UseClippedLinears     bool
}

func (cfg VisionConfig) Present() bool {
	return cfg.ModelType != "" ||
		cfg.DType != "" ||
		cfg.ImageSize > 0 ||
		cfg.PatchSize > 0 ||
		cfg.NumChannels > 0 ||
		cfg.HiddenSize > 0 ||
		cfg.IntermediateSize > 0 ||
		cfg.NumHiddenLayers > 0 ||
		cfg.NumAttentionHeads > 0 ||
		cfg.NumKeyValueHeads > 0 ||
		cfg.HeadDim > 0 ||
		cfg.GlobalHeadDim > 0 ||
		cfg.PoolingKernelSize > 0 ||
		cfg.PositionEmbeddingSize > 0 ||
		cfg.DefaultOutputLength > 0 ||
		cfg.ImageToken() > 0 ||
		cfg.VideoToken() > 0 ||
		cfg.SoftTokens() > 0
}

func (cfg VisionConfig) ImageToken() int {
	return firstPositiveIntValue(cfg.ImageTokenID, cfg.ImageTokenIndex)
}

func (cfg VisionConfig) VideoToken() int {
	return firstPositiveIntValue(cfg.VideoTokenID, cfg.VideoTokenIndex)
}

func (cfg VisionConfig) SoftTokens() int {
	return firstPositiveIntValue(cfg.SoftTokensPerImage, cfg.MMTokensPerImage, cfg.DefaultOutputLength)
}

// AudioConfig is the backend-neutral Gemma-4 audio metadata surface read from
// config.json / processor metadata.
type AudioConfig struct {
	AudioTokenID                int
	AudioTokenIndex             int
	BOATokenID                  int
	BOATokenIndex               int
	EOATokenID                  int
	EOATokenIndex               int
	ModelType                   string
	HiddenSize                  int
	AudioEmbedDim               int
	AudioSamplesPerToken        int
	NumHiddenLayers             int
	NumAttentionHeads           int
	AttentionChunkSize          int
	AttentionContextLeft        int
	AttentionContextRight       int
	AttentionLogitCap           float64
	AttentionInvalidLogitsValue float64
	ConvKernelSize              int
	OutputProjDims              int
	RMSNormEps                  float64
	GradientClipping            float64
	ResidualWeight              float64
	HiddenAct                   string
	UseClippedLinears           bool
}

func (cfg AudioConfig) Present() bool {
	return cfg.ModelType != "" ||
		cfg.HiddenSize > 0 ||
		cfg.AudioEmbedDim > 0 ||
		cfg.AudioSamplesPerToken > 0 ||
		cfg.NumHiddenLayers > 0 ||
		cfg.NumAttentionHeads > 0 ||
		cfg.AttentionChunkSize > 0 ||
		cfg.AttentionContextLeft > 0 ||
		cfg.AttentionContextRight > 0 ||
		cfg.ConvKernelSize > 0 ||
		cfg.OutputProjDims > 0 ||
		cfg.AudioToken() > 0
}

func (cfg AudioConfig) AudioToken() int {
	return firstPositiveIntValue(cfg.AudioTokenID, cfg.AudioTokenIndex)
}

func ApplyVisionConfigLabels(labels map[string]string, cfg VisionConfig) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if !cfg.Present() {
		return labels
	}
	setPositiveIntLabel(labels, "image_token_id", cfg.ImageToken())
	setPositiveIntLabel(labels, "video_token_id", cfg.VideoToken())
	setPositiveIntLabel(labels, "boi_token_id", cfg.BOITokenID)
	setPositiveIntLabel(labels, "boi_token_index", cfg.BOITokenIndex)
	setPositiveIntLabel(labels, "eoi_token_id", cfg.EOITokenID)
	setPositiveIntLabel(labels, "eoi_token_index", cfg.EOITokenIndex)
	setPositiveIntLabel(labels, "vision_soft_tokens_per_image", cfg.SoftTokens())
	if cfg.ModelType != "" {
		labels["vision_model_type"] = normalizeConfigLabelToken(cfg.ModelType)
	}
	if cfg.DType != "" {
		labels["vision_dtype"] = normalizeDTypeLabel(cfg.DType)
	}
	setPositiveIntLabel(labels, "vision_image_size", cfg.ImageSize)
	setPositiveIntLabel(labels, "vision_patch_size", cfg.PatchSize)
	setPositiveIntLabel(labels, "vision_num_channels", cfg.NumChannels)
	setPositiveIntLabel(labels, "vision_hidden_size", cfg.HiddenSize)
	setPositiveIntLabel(labels, "vision_intermediate_size", cfg.IntermediateSize)
	setPositiveIntLabel(labels, "vision_num_hidden_layers", cfg.NumHiddenLayers)
	setPositiveIntLabel(labels, "vision_attention_heads", cfg.NumAttentionHeads)
	setPositiveIntLabel(labels, "vision_kv_heads", cfg.NumKeyValueHeads)
	setPositiveIntLabel(labels, "vision_head_dim", cfg.HeadDim)
	setPositiveIntLabel(labels, "vision_global_head_dim", cfg.GlobalHeadDim)
	setPositiveIntLabel(labels, "vision_pooling_kernel_size", cfg.PoolingKernelSize)
	setPositiveIntLabel(labels, "vision_position_embedding_size", cfg.PositionEmbeddingSize)
	if cfg.HiddenActivation != "" {
		labels["vision_hidden_activation"] = cfg.HiddenActivation
	}
	setPositiveFloatLabel(labels, "vision_rms_norm_eps", cfg.RMSNormEps)
	setPositiveFloatLabel(labels, "vision_rope_theta", cfg.RoPEParameters.RopeTheta)
	if cfg.RoPEParameters.RopeType != "" {
		labels["vision_rope_type"] = cfg.RoPEParameters.RopeType
	}
	labels["vision_standardize"] = strconv.FormatBool(cfg.Standardize)
	labels["vision_use_clipped_linears"] = strconv.FormatBool(cfg.UseClippedLinears)
	return labels
}

func ApplyAudioConfigLabels(labels map[string]string, cfg AudioConfig) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if !cfg.Present() {
		return labels
	}
	setPositiveIntLabel(labels, "audio_token_id", cfg.AudioToken())
	setPositiveIntLabel(labels, "boa_token_id", cfg.BOATokenID)
	setPositiveIntLabel(labels, "boa_token_index", cfg.BOATokenIndex)
	setPositiveIntLabel(labels, "eoa_token_id", cfg.EOATokenID)
	setPositiveIntLabel(labels, "eoa_token_index", cfg.EOATokenIndex)
	if cfg.ModelType != "" {
		labels["audio_model_type"] = normalizeConfigLabelToken(cfg.ModelType)
	}
	setPositiveIntLabel(labels, "audio_hidden_size", cfg.HiddenSize)
	setPositiveIntLabel(labels, "audio_embed_dim", cfg.AudioEmbedDim)
	setPositiveIntLabel(labels, "audio_samples_per_token", cfg.AudioSamplesPerToken)
	setPositiveIntLabel(labels, "audio_num_hidden_layers", cfg.NumHiddenLayers)
	setPositiveIntLabel(labels, "audio_attention_heads", cfg.NumAttentionHeads)
	setPositiveIntLabel(labels, "audio_attention_chunk_size", cfg.AttentionChunkSize)
	setPositiveIntLabel(labels, "audio_attention_context_left", cfg.AttentionContextLeft)
	setPositiveIntLabel(labels, "audio_attention_context_right", cfg.AttentionContextRight)
	setPositiveFloatLabel(labels, "audio_attention_logit_cap", cfg.AttentionLogitCap)
	if cfg.AttentionInvalidLogitsValue != 0 {
		labels["audio_attention_invalid_logits_value"] = formatRoPEFloat(cfg.AttentionInvalidLogitsValue)
	}
	setPositiveIntLabel(labels, "audio_conv_kernel_size", cfg.ConvKernelSize)
	setPositiveIntLabel(labels, "audio_output_proj_dims", cfg.OutputProjDims)
	setPositiveFloatLabel(labels, "audio_rms_norm_eps", cfg.RMSNormEps)
	setPositiveFloatLabel(labels, "audio_gradient_clipping", cfg.GradientClipping)
	setPositiveFloatLabel(labels, "audio_residual_weight", cfg.ResidualWeight)
	if cfg.HiddenAct != "" {
		labels["audio_hidden_act"] = cfg.HiddenAct
	}
	labels["audio_use_clipped_linears"] = strconv.FormatBool(cfg.UseClippedLinears)
	return labels
}

func firstPositiveIntValue(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func setPositiveIntLabel(labels map[string]string, key string, value int) {
	if value > 0 {
		labels[key] = strconv.Itoa(value)
	}
}

func setPositiveFloatLabel(labels map[string]string, key string, value float64) {
	if value > 0 {
		labels[key] = formatRoPEFloat(value)
	}
}

func normalizeConfigLabelToken(value string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(value)), "-", "_")
}

func normalizeDTypeLabel(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "bfloat16", "bf16":
		return "bf16"
	case "float16", "fp16", "f16":
		return "f16"
	case "float32", "fp32", "f32":
		return "f32"
	default:
		return strings.ToLower(strings.TrimSpace(value))
	}
}
