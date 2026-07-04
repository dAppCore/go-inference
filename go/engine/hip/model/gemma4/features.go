// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"strconv"
	"strings"

	"dappco.re/go/inference"
)

// Features is the Gemma-4 model-family settings surface. It describes what a
// loaded config or metadata label set declares, so runtime packages can react
// to the model rather than branching on model names.
type Features struct {
	Mixture      bool              `json:"mixture,omitempty"`
	NumExperts   int               `json:"num_experts,omitempty"`
	TopKExperts  int               `json:"top_k_experts,omitempty"`
	Vision       bool              `json:"vision,omitempty"`
	Audio        bool              `json:"audio,omitempty"`
	Attention    AttentionClass    `json:"attention,omitempty"`
	Quantization QuantizationClass `json:"quantization,omitempty"`
	Structure    StructurePlan     `json:"structure,omitempty"`
}

// AttentionClass is the attention topology Gemma-4 declares from config.
type AttentionClass struct {
	SlidingWindow  int `json:"sliding_window,omitempty"`
	SlidingPattern int `json:"sliding_pattern,omitempty"`
	SharedKVLayers int `json:"shared_kv_layers,omitempty"`
}

func (attention AttentionClass) Hybrid() bool {
	return attention.SlidingWindow > 0
}

// QuantizationClass is the quantization family the loaded Gemma-4 build
// declares. Kernel-specific engine features can react to it without inspecting
// repository paths or loader names.
type QuantizationClass struct {
	Bits int    `json:"bits,omitempty"`
	Mode string `json:"mode,omitempty"`
}

func (quant QuantizationClass) Q6Bitstream() bool {
	if quant.Bits == 6 {
		return true
	}
	switch strings.ToLower(strings.TrimSpace(quant.Mode)) {
	case "q6", "q6-status", "6bit", "6-bit", "6_bit":
		return true
	default:
		return false
	}
}

// TextConfig carries only the Gemma-4 settings the ROCm engine needs for
// feature selection. Runtime-specific config structs adapt into this shape.
type TextConfig struct {
	NumLayers                 int
	LayerTypes                []string
	EnableMoEBlock            bool
	NumExperts                int
	TopKExperts               int
	Vision                    bool
	VisionConfig              VisionConfig
	Audio                     bool
	AudioConfig               AudioConfig
	SlidingWindow             int
	SlidingWindowPattern      int
	KVSharedLayers            int
	KVSharedLayersSet         bool
	GlobalPartialRotaryFactor float64
	RoPEParameters            map[string]RoPEParameters
	AttentionKEqV             bool
	AttentionKEqVSet          bool
	HiddenSizePerLayer        int
	VocabSizePerLayer         int
	UseDoubleWideMLP          bool
	MoEIntermediateSize       int
	QuantBits                 int
	QuantMode                 string
}

// EngineFeatures is the Gemma-4 family contribution to runtime feature
// selection. Backend packages can map it into their native feature structs
// while keeping config-derived cache decisions owned by this model package.
type EngineFeatures struct {
	DirectGreedyToken           bool `json:"direct_greedy_token,omitempty"`
	NativeMLPMatVec             bool `json:"native_mlp_matvec,omitempty"`
	NativeLinearMatVec          bool `json:"native_linear_matvec,omitempty"`
	NativeQ6BitstreamMatVec     bool `json:"native_q6_bitstream_matvec,omitempty"`
	NativeAttentionOMatVec      bool `json:"native_attention_o_matvec,omitempty"`
	NativeFixedSlidingAttention bool `json:"native_fixed_sliding_attention,omitempty"`
	GenerationStream            bool `json:"generation_stream,omitempty"`
	AsyncDecodePrefetch         bool `json:"async_decode_prefetch,omitempty"`
	ModelContextWindow          bool `json:"model_context_window,omitempty"`
	FixedSlidingCache           bool `json:"fixed_sliding_cache,omitempty"`
	FixedSlidingCacheBound      bool `json:"fixed_sliding_cache_bound,omitempty"`
	CompiledLayerDecode         bool `json:"compiled_layer_decode,omitempty"`
	PipelinedDecode             bool `json:"pipelined_decode,omitempty"`
}

const largeVariantAttentionHeads = 16

func FeaturesOf(cfg TextConfig) Features {
	features := Features{
		Mixture:     cfg.EnableMoEBlock,
		Vision:      cfg.Vision || cfg.VisionConfig.Present(),
		Audio:       cfg.Audio || cfg.AudioConfig.Present(),
		NumExperts:  positiveInt(cfg.NumExperts),
		TopKExperts: positiveInt(cfg.TopKExperts),
		Attention: AttentionClass{
			SlidingWindow:  positiveInt(cfg.SlidingWindow),
			SlidingPattern: positiveInt(cfg.SlidingWindowPattern),
			SharedKVLayers: positiveInt(cfg.KVSharedLayers),
		},
		Quantization: QuantizationClass{
			Bits: positiveInt(cfg.QuantBits),
			Mode: strings.ToLower(strings.TrimSpace(cfg.QuantMode)),
		},
		Structure: StructurePlanOf(cfg),
	}
	if !features.Mixture {
		features.NumExperts = 0
		features.TopKExperts = 0
	}
	return features
}

func FeaturesOfIdentity(identity inference.ModelIdentity) Features {
	features := FeaturesOfLabels(identity.Labels)
	if features.Quantization.Bits <= 0 {
		features.Quantization.Bits = positiveInt(identity.QuantBits)
	}
	if features.Quantization.Mode == "" {
		features.Quantization.Mode = strings.ToLower(strings.TrimSpace(firstNonEmptyString(ModelPackQuantModeForPath(identity, identity.Path), identity.QuantType)))
	}
	return features
}

func EngineFeaturesOf(features Features) EngineFeatures {
	hybrid := features.Attention.Hybrid()
	return EngineFeatures{
		NativeQ6BitstreamMatVec: features.Quantization.Q6Bitstream(),
		ModelContextWindow:      true,
		FixedSlidingCache:       hybrid,
		FixedSlidingCacheBound:  hybrid,
	}
}

func LinkedGenerationEngineFeatures(features EngineFeatures) EngineFeatures {
	q6Bitstream := features.NativeQ6BitstreamMatVec
	features.DirectGreedyToken = true
	features.NativeMLPMatVec = true
	features.NativeLinearMatVec = true
	features.NativeQ6BitstreamMatVec = q6Bitstream
	features.NativeAttentionOMatVec = true
	features.NativeFixedSlidingAttention = features.FixedSlidingCache
	features.GenerationStream = true
	features.AsyncDecodePrefetch = true
	return features
}

func EngineFeaturesOfIdentity(identity inference.ModelIdentity) EngineFeatures {
	return EngineFeaturesOf(FeaturesOfIdentity(identity))
}

func NeedsThoughtChannelSuppressorForAttentionHeads(attentionHeads int) (bool, bool) {
	if attentionHeads <= 0 {
		return false, false
	}
	return attentionHeads >= largeVariantAttentionHeads, true
}

func NeedsThoughtChannelSuppressorForIdentity(identity inference.ModelIdentity) (bool, bool) {
	return NeedsThoughtChannelSuppressorForAttentionHeads(firstPositiveIntLabel(identity.Labels, "attention_heads", "num_attention_heads", "gemma4_attention_heads"))
}

func SizeNeedsThoughtChannelSuppressor(size string) bool {
	switch strings.ToUpper(strings.TrimSpace(size)) {
	case "12B", "26B-A4B", "31B":
		return true
	default:
		return false
	}
}

func FeaturesOfLabels(labels map[string]string) Features {
	return Features{
		Mixture:     labelValue(labels, "gemma4_enable_moe_block") == "true",
		NumExperts:  positiveIntLabel(labels, "gemma4_num_experts"),
		TopKExperts: positiveIntLabel(labels, "gemma4_top_k_experts"),
		Vision:      declaredVision(labels),
		Audio:       declaredAudio(labels),
		Attention: AttentionClass{
			SlidingWindow:  firstPositiveIntLabel(labels, "gemma4_sliding_window", "sliding_window", "attention_sliding_window"),
			SlidingPattern: firstPositiveIntLabel(labels, "gemma4_sliding_window_pattern", "sliding_window_pattern", "attention_sliding_pattern"),
			SharedKVLayers: firstPositiveIntLabel(labels, "gemma4_attention_kv_shared_layers", "attention_kv_shared_layers"),
		},
		Quantization: QuantizationClass{
			Bits: firstPositiveIntLabel(labels, "gemma4_quant_bits", "production_quant_bits", "engine_quant_loader_bits", "quant_bits", "quantization_bits"),
			Mode: firstNonEmptyLabel(labels, "gemma4_quant_mode", "production_quant_mode", "engine_quant_loader_mode", "quant_mode", "quant_type"),
		},
		Structure: StructurePlanOfLabels(labels),
	}
}

func ApplyConfigFeatureLabels(labels map[string]string, features Features) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if features.Attention.SlidingWindow > 0 {
		value := strconv.Itoa(features.Attention.SlidingWindow)
		labels["sliding_window"] = value
		labels["gemma4_sliding_window"] = value
	}
	if features.Attention.SlidingPattern > 0 {
		value := strconv.Itoa(features.Attention.SlidingPattern)
		labels["sliding_window_pattern"] = value
		labels["gemma4_sliding_window_pattern"] = value
	}
	if features.Attention.SharedKVLayers > 0 {
		value := strconv.Itoa(features.Attention.SharedKVLayers)
		labels["attention_kv_shared_layers"] = value
		labels["gemma4_attention_kv_shared_layers"] = value
	}
	if features.Mixture {
		labels["gemma4_enable_moe_block"] = "true"
	}
	if features.NumExperts > 0 {
		labels["gemma4_num_experts"] = strconv.Itoa(features.NumExperts)
	}
	if features.TopKExperts > 0 {
		labels["gemma4_top_k_experts"] = strconv.Itoa(features.TopKExperts)
	}
	if features.Vision || features.Audio {
		labels["gemma4_multimodal"] = "true"
	}
	if features.Vision {
		labels["gemma4_vision"] = "true"
	}
	if features.Audio {
		labels["gemma4_audio"] = "true"
	}
	if features.Quantization.Bits > 0 {
		labels["gemma4_quant_bits"] = strconv.Itoa(features.Quantization.Bits)
	}
	if features.Quantization.Mode != "" {
		labels["gemma4_quant_mode"] = features.Quantization.Mode
	}
	return labels
}

// ApplyConfigLabels writes labels for the full Gemma-4 config feature surface.
func ApplyConfigLabels(labels map[string]string, cfg TextConfig) map[string]string {
	labels = ApplyConfigFeatureLabels(labels, FeaturesOf(cfg))
	labels = ApplyStructurePlanLabels(labels, StructurePlanOf(cfg))
	labels = ApplyCacheTopologyLabels(labels, CacheTopologyOf(cfg))
	labels = ApplyAttentionWindowPolicyLabels(labels, AttentionWindowPolicyOf(cfg))
	labels = ApplyRoPEPolicyLabels(labels, RoPEPolicyOf(cfg))
	if cfg.KVSharedLayersSet {
		value := strconv.Itoa(cfg.KVSharedLayers)
		labels["attention_kv_shared_layers"] = value
		labels["gemma4_attention_kv_shared_layers"] = value
	}
	if cfg.HiddenSizePerLayer > 0 {
		labels["gemma4_hidden_size_per_layer_input"] = strconv.Itoa(cfg.HiddenSizePerLayer)
	}
	if cfg.VocabSizePerLayer > 0 {
		labels["gemma4_vocab_size_per_layer_input"] = strconv.Itoa(cfg.VocabSizePerLayer)
	}
	if cfg.UseDoubleWideMLP {
		labels["gemma4_use_double_wide_mlp"] = "true"
	}
	if cfg.MoEIntermediateSize > 0 {
		labels["gemma4_moe_intermediate_size"] = strconv.Itoa(cfg.MoEIntermediateSize)
	}
	return labels
}

func ApplyDeclaredFeatureLabels(labels map[string]string, features Features) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if features.Attention.SlidingWindow > 0 {
		labels["gemma4_attention_sliding_window"] = strconv.Itoa(features.Attention.SlidingWindow)
	}
	if features.Attention.SlidingPattern > 0 {
		labels["gemma4_attention_sliding_pattern"] = strconv.Itoa(features.Attention.SlidingPattern)
	}
	if features.Attention.SharedKVLayers > 0 {
		labels["gemma4_attention_kv_shared_layers"] = strconv.Itoa(features.Attention.SharedKVLayers)
	}
	if features.Mixture {
		labels["gemma4_mixture"] = "true"
	}
	if features.NumExperts > 0 {
		labels["gemma4_num_experts"] = strconv.Itoa(features.NumExperts)
	}
	if features.TopKExperts > 0 {
		labels["gemma4_top_k_experts"] = strconv.Itoa(features.TopKExperts)
	}
	if features.Vision || features.Audio {
		labels["gemma4_multimodal"] = "true"
	}
	if features.Vision {
		labels["gemma4_vision"] = "true"
	}
	if features.Audio {
		labels["gemma4_audio"] = "true"
	}
	if features.Quantization.Bits > 0 {
		labels["gemma4_quant_bits"] = strconv.Itoa(features.Quantization.Bits)
	}
	if features.Quantization.Mode != "" {
		labels["gemma4_quant_mode"] = features.Quantization.Mode
	}
	return labels
}

func declaredVision(labels map[string]string) bool {
	return anyTruthyLabel(labels, "gemma4_vision", "engine_multimodal_processor_vision") ||
		anySetLabel(labels,
			"vision_reference", "vision_runtime", "vision_projector_runtime", "vision_model_type",
			"image_token_id", "image_token_index", "video_token_id", "video_token_index",
			"image_processor", "video_processor", "image_processor_max_soft_tokens", "video_processor_max_soft_tokens",
			"vision_soft_tokens_per_image", "mm_tokens_per_image", "vision_hidden_size", "vision_num_hidden_layers",
			"engine_multimodal_processor_vision_reference", "engine_multimodal_processor_vision_runtime",
			"engine_multimodal_processor_vision_projector_runtime", "engine_multimodal_processor_vision_model_type",
			"engine_multimodal_processor_image_token_id", "engine_multimodal_processor_image_token_index",
			"engine_multimodal_processor_video_token_id", "engine_multimodal_processor_video_token_index",
			"engine_multimodal_processor_soft_tokens_per_image", "engine_multimodal_processor_mm_tokens_per_image",
			"engine_multimodal_processor_vision_hidden_size", "engine_multimodal_processor_vision_layers")
}

func declaredAudio(labels map[string]string) bool {
	return anyTruthyLabel(labels, "gemma4_audio", "engine_multimodal_processor_audio") ||
		anySetLabel(labels,
			"audio_reference", "audio_runtime", "audio_projector_runtime", "audio_frontend_runtime", "audio_front_end_runtime", "audio_model_type",
			"audio_token_id", "audio_token_index", "audio_samples_per_token", "audio_hidden_size", "audio_num_hidden_layers", "audio_embed_dim",
			"audio_feature_extractor", "processor_audio_ms_per_token", "processor_audio_seq_length",
			"engine_multimodal_processor_audio_reference", "engine_multimodal_processor_audio_runtime",
			"engine_multimodal_processor_audio_projector_runtime", "engine_multimodal_processor_audio_front_end_runtime",
			"engine_multimodal_processor_audio_model_type", "engine_multimodal_processor_audio_token_id",
			"engine_multimodal_processor_audio_token_index", "engine_multimodal_processor_audio_samples_per_token",
			"engine_multimodal_processor_audio_hidden_size", "engine_multimodal_processor_audio_layers",
			"engine_multimodal_processor_audio_embed_dim")
}

func anyTruthyLabel(labels map[string]string, keys ...string) bool {
	for _, key := range keys {
		switch labelValue(labels, key) {
		case "true", "1", "yes":
			return true
		}
	}
	return false
}

func anySetLabel(labels map[string]string, keys ...string) bool {
	for _, key := range keys {
		value := labelValue(labels, key)
		if value != "" && value != "false" && value != "0" && value != "none" {
			return true
		}
	}
	return false
}

func firstPositiveIntLabel(labels map[string]string, keys ...string) int {
	for _, key := range keys {
		if value := positiveIntLabel(labels, key); value > 0 {
			return value
		}
	}
	return 0
}

func firstNonEmptyLabel(labels map[string]string, keys ...string) string {
	for _, key := range keys {
		if value := labelValue(labels, key); value != "" {
			return value
		}
	}
	return ""
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func positiveIntLabel(labels map[string]string, key string) int {
	raw := strings.TrimSpace(labels[key])
	if raw == "" {
		return 0
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value <= 0 {
		return 0
	}
	return value
}

func labelValue(labels map[string]string, key string) string {
	return strings.ToLower(strings.TrimSpace(labels[key]))
}

func positiveInt(value int) int {
	if value > 0 {
		return value
	}
	return 0
}
