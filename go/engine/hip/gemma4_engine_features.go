// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"strconv"

	"dappco.re/go/inference"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
)

type Gemma4DeclaredFeatures struct {
	Mixture     bool                 `json:"mixture,omitempty"`
	NumExperts  int                  `json:"num_experts,omitempty"`
	TopKExperts int                  `json:"top_k_experts,omitempty"`
	Vision      bool                 `json:"vision,omitempty"`
	Audio       bool                 `json:"audio,omitempty"`
	Attention   Gemma4AttentionClass `json:"attention,omitempty"`
}

type Gemma4AttentionClass struct {
	SlidingWindow  int `json:"sliding_window,omitempty"`
	SlidingPattern int `json:"sliding_pattern,omitempty"`
	SharedKVLayers int `json:"shared_kv_layers,omitempty"`
}

func (attention Gemma4AttentionClass) Hybrid() bool {
	return attention.SlidingWindow > 0
}

type Gemma4EngineFeatures struct {
	MLXAffineDecode             bool `json:"mlx_affine_decode,omitempty"`
	DenseBF16Decode             bool `json:"dense_bf16_decode,omitempty"`
	TextGenerate                bool `json:"text_generate,omitempty"`
	DirectGreedyToken           bool `json:"direct_greedy_token,omitempty"`
	NativeMLPMatVec             bool `json:"native_mlp_matvec,omitempty"`
	NativeLinearMatVec          bool `json:"native_linear_matvec,omitempty"`
	NativeQ6BitstreamMatVec     bool `json:"native_q6_bitstream_matvec,omitempty"`
	NativeAttentionOMatVec      bool `json:"native_attention_o_matvec,omitempty"`
	NativeFixedSlidingAttention bool `json:"native_fixed_sliding_attention,omitempty"`
	GenerationStream            bool `json:"generation_stream,omitempty"`
	AsyncDecodePrefetch         bool `json:"async_decode_prefetch,omitempty"`
	ModelContextWindow          bool `json:"model_context_window,omitempty"`
	DeviceKVState               bool `json:"device_kv_state,omitempty"`
	FixedSlidingCache           bool `json:"fixed_sliding_cache,omitempty"`
	FixedSlidingCacheBound      bool `json:"fixed_sliding_cache_bound,omitempty"`
	CompiledLayerDecode         bool `json:"compiled_layer_decode,omitempty"`
	PipelinedDecode             bool `json:"pipelined_decode,omitempty"`
}

func Gemma4EngineFeaturesForModel(info inference.ModelInfo) Gemma4EngineFeatures {
	return Gemma4EngineFeaturesForIdentity(inference.ModelIdentity{
		Architecture: info.Architecture,
		NumLayers:    info.NumLayers,
		HiddenSize:   info.HiddenSize,
		VocabSize:    info.VocabSize,
		QuantBits:    info.QuantBits,
		QuantGroup:   info.QuantGroup,
	})
}

func Gemma4EngineFeaturesForIdentity(identity inference.ModelIdentity) Gemma4EngineFeatures {
	if !isROCmGemma4Architecture(identity.Architecture) {
		return Gemma4EngineFeatures{}
	}
	features := rocmGemma4EngineFeaturesForModel(identity)
	if gemma4EngineGenerateLinked(identity) {
		denseBF16 := gemma4EngineDenseBF16Linked(identity)
		if denseBF16 {
			features.DenseBF16Decode = true
		} else {
			features.MLXAffineDecode = true
		}
		features.TextGenerate = true
		features.DeviceKVState = true
		features = rocmGemma4LinkedGenerationEngineFeatures(features)
		if denseBF16 {
			features.DirectGreedyToken = false
			features.NativeQ6BitstreamMatVec = false
			features.AsyncDecodePrefetch = false
		}
	} else {
		features.NativeQ6BitstreamMatVec = false
	}
	return features
}

func (features Gemma4EngineFeatures) GenerateLinked() bool {
	return (features.MLXAffineDecode || features.DenseBF16Decode) && features.TextGenerate
}

func gemma4EngineGenerateLinked(identity inference.ModelIdentity) bool {
	return rocmGemma4SupportMatrixGenerateLinked(identity)
}

func gemma4EngineDenseBF16Linked(identity inference.ModelIdentity) bool {
	size := rocmGemma4ModelPackSize(identity, identity.Path)
	mode := rocmGemma4ModelPackQuantModeForPath(identity, identity.Path)
	return rocmGemma4NormalizeSizeQuantMode(size, mode) == "bf16"
}

func Gemma4DeclaredFeaturesOfNativeConfig(cfg nativeGemma4TextConfig) Gemma4DeclaredFeatures {
	return rocmGemma4DeclaredFeaturesFromModel(modelgemma4.FeaturesOf(rocmGemma4TextConfigFromNativeConfig(cfg)))
}

func rocmGemma4TextConfigFromNativeConfig(cfg nativeGemma4TextConfig) modelgemma4.TextConfig {
	return modelgemma4.TextConfig{
		NumLayers:            firstPositiveInt(cfg.NumLayers, len(cfg.LayerTypes)),
		LayerTypes:           cfg.LayerTypes,
		EnableMoEBlock:       cfg.EnableMoEBlock,
		NumExperts:           cfg.NumExperts,
		TopKExperts:          cfg.TopKExperts,
		Vision:               cfg.Vision,
		Audio:                cfg.Audio,
		SlidingWindow:        cfg.SlidingWindow,
		SlidingWindowPattern: cfg.SlidingWindowPattern,
		KVSharedLayers:       cfg.KVSharedLayers,
		KVSharedLayersSet:    cfg.KVSharedLayersSet,
		RoPEParameters:       rocmGemma4RoPEParametersFromNativeConfig(cfg.RoPEParameters),
		HiddenSizePerLayer:   cfg.HiddenSizePerLayerInput,
		VocabSizePerLayer:    cfg.VocabSizePerLayerInput,
		UseDoubleWideMLP:     cfg.UseDoubleWideMLP,
		MoEIntermediateSize:  cfg.MoEIntermediateSize,
	}
}

func rocmGemma4RoPEParametersFromNativeConfig(src map[string]nativeGemma4RoPEParameters) map[string]modelgemma4.RoPEParameters {
	if len(src) == 0 {
		return nil
	}
	params := make(map[string]modelgemma4.RoPEParameters, len(src))
	for attentionType, value := range src {
		if attentionType == "" {
			continue
		}
		params[attentionType] = modelgemma4.RoPEParameters{
			PartialRotaryFactor: value.PartialRotaryFactor,
			RopeTheta:           value.RopeTheta,
			RopeType:            value.RopeType,
			Factor:              value.Factor,
		}
	}
	if len(params) == 0 {
		return nil
	}
	return params
}

func Gemma4DeclaredFeaturesForIdentity(identity inference.ModelIdentity) Gemma4DeclaredFeatures {
	return rocmGemma4DeclaredFeaturesForModel(identity)
}

func rocmApplyGemma4NativeConfigFeatureLabels(labels map[string]string, cfg nativeGemma4TextConfig) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	return rocmApplyGemma4ConfigLabels(labels, rocmGemma4TextConfigFromNativeConfig(cfg))
}

func rocmApplyGemma4EngineFeatureLabels(labels map[string]string, features Gemma4EngineFeatures, declared Gemma4DeclaredFeatures) {
	if labels == nil {
		return
	}
	labels["engine_model_context_window"] = strconv.FormatBool(features.ModelContextWindow)
	labels["engine_text_generate"] = strconv.FormatBool(features.TextGenerate)
	labels["engine_mlx_affine_decode"] = strconv.FormatBool(features.MLXAffineDecode)
	labels["engine_dense_bf16_decode"] = strconv.FormatBool(features.DenseBF16Decode)
	labels["engine_device_kv_state"] = strconv.FormatBool(features.DeviceKVState)
	labels["engine_direct_greedy_token"] = strconv.FormatBool(features.DirectGreedyToken)
	labels["engine_native_mlp_matvec"] = strconv.FormatBool(features.NativeMLPMatVec)
	labels["engine_native_linear_matvec"] = strconv.FormatBool(features.NativeLinearMatVec)
	labels["engine_native_q6_bitstream_matvec"] = strconv.FormatBool(features.NativeQ6BitstreamMatVec)
	labels["engine_native_attention_o_matvec"] = strconv.FormatBool(features.NativeAttentionOMatVec)
	labels["engine_native_fixed_sliding_attention"] = strconv.FormatBool(features.NativeFixedSlidingAttention)
	labels["engine_generation_stream"] = strconv.FormatBool(features.GenerationStream)
	labels["engine_async_decode_prefetch"] = strconv.FormatBool(features.AsyncDecodePrefetch)
	labels["engine_fixed_sliding_cache"] = strconv.FormatBool(features.FixedSlidingCache)
	labels["engine_fixed_sliding_cache_bound"] = strconv.FormatBool(features.FixedSlidingCacheBound)
	labels["engine_compiled_layer_decode"] = strconv.FormatBool(features.CompiledLayerDecode)
	labels["engine_pipelined_decode"] = strconv.FormatBool(features.PipelinedDecode)
	rocmApplyGemma4DeclaredFeatureLabels(labels, declared)
}

func rocmGemma4PlanModelFitPackLoadOK(identity inference.ModelIdentity) bool {
	if !isROCmGemma4Architecture(identity.Architecture) {
		return true
	}
	identity = rocmGemma4ModelWithInferredPathQuant(identity)
	if rocmGemma4LabelValue(identity.Labels, "gemma4_pack_supported") == "false" ||
		rocmGemma4LabelValue(identity.Labels, "gemma4_runnable_on_card") == "false" ||
		rocmGemma4LabelValue(identity.Labels, "gemma4_generate_status") == Gemma4GeneratePlannedOnly {
		return false
	}
	size := rocmGemma4ModelPackSize(identity, identity.Path)
	mode := rocmGemma4ModelPackQuantModeForPath(identity, identity.Path)
	mode = rocmGemma4NormalizeSizeQuantMode(size, mode)
	if size == "" || mode == "" {
		return true
	}
	sizeSupport, ok := Gemma4SizeQuantSupportBySize(size)
	if !ok || !sizeSupport.RunnableOnCard {
		return false
	}
	support, ok := Gemma4QuantModeSupportBySize(size, mode)
	return ok && support.GenerateStatus != Gemma4GeneratePlannedOnly
}
