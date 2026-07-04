// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"strings"
	"testing"

	"dappco.re/go/inference"
)

func linkedGemma4TestLabels(size, mode string) map[string]string {
	return map[string]string{
		"gemma4_size":       size,
		"gemma4_quant_mode": mode,
	}
}

func TestGemma4EngineFeaturesForModel(t *testing.T) {
	for _, bits := range []int{4, 6, 8} {
		features := Gemma4EngineFeaturesForModel(inference.ModelInfo{Architecture: "gemma4_text", QuantBits: bits, NumLayers: productionLaneGemma4E2BLayers, HiddenSize: productionLaneGemma4E2BHiddenSize})
		if features.GenerateLinked() || features.DeviceKVState || !features.ModelContextWindow {
			t.Fatalf("Gemma4 E2B q%d model-info features = %+v, want context support without shape-only linked generation", bits, features)
		}
	}
	bitOnly := Gemma4EngineFeaturesForModel(inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6})
	if bitOnly.GenerateLinked() || !bitOnly.ModelContextWindow {
		t.Fatalf("Gemma4 bit-only features = %+v, want context support without linked generation", bitOnly)
	}
	unified := Gemma4EngineFeaturesForModel(inference.ModelInfo{Architecture: "gemma4_unified", QuantBits: 6, NumLayers: 48, HiddenSize: 3840})
	if unified.GenerateLinked() || !unified.ModelContextWindow {
		t.Fatalf("Gemma4 unified q6 model-info features = %+v, want context support without shape-only generation", unified)
	}
	e4BQ6 := Gemma4EngineFeaturesForModel(inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6, NumLayers: 26, HiddenSize: 2304})
	if e4BQ6.GenerateLinked() || !e4BQ6.ModelContextWindow {
		t.Fatalf("Gemma4 E4B q6 model-info features = %+v, want context support without shape-only generation", e4BQ6)
	}
	twelveBQ6 := Gemma4EngineFeaturesForModel(inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6, NumLayers: 48, HiddenSize: 3840})
	if twelveBQ6.GenerateLinked() || !twelveBQ6.ModelContextWindow {
		t.Fatalf("Gemma4 12B q6 model-info features = %+v, want context support without shape-only generation", twelveBQ6)
	}
	twelveBQ4 := Gemma4EngineFeaturesForModel(inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 4, NumLayers: 48, HiddenSize: 3840})
	if twelveBQ4.GenerateLinked() || !twelveBQ4.ModelContextWindow {
		t.Fatalf("Gemma4 12B q4 model-info features = %+v, want context support without shape-only generation", twelveBQ4)
	}
	features := Gemma4EngineFeaturesForModel(inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 16})
	if features.GenerateLinked() || !features.ModelContextWindow {
		t.Fatalf("Gemma4 BF16 features = %+v, want context support without MLX-affine generate", features)
	}
	if Gemma4EngineFeaturesForModel(inference.ModelInfo{Architecture: "qwen3", QuantBits: 4}) != (Gemma4EngineFeatures{}) {
		t.Fatalf("non-Gemma4 features should be empty")
	}
}

func TestGemma4EngineFeaturesForIdentityUsesPathMetadata(t *testing.T) {
	for _, tc := range []struct {
		name string
		path string
		want bool
	}{
		{name: "e2b_q4", path: "/models/lmstudio-community-gemma-4-e2b-it-4bit", want: true},
		{name: "e4b_q8", path: "/models/lmstudio-community-gemma-4-e4b-it-8bit", want: true},
		{name: "12b_q6", path: "/models/lmstudio-community-gemma-4-12b-it-6bit", want: true},
		{name: "12b_q4", path: "/models/lmstudio-community-gemma-4-12b-it-4bit", want: true},
		{name: "31b_q6", path: "/models/lmstudio-community-gemma-4-31b-it-6bit"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			features := Gemma4EngineFeaturesForIdentity(inference.ModelIdentity{
				Path:         tc.path,
				Architecture: "gemma4_text",
			})
			if features.GenerateLinked() != tc.want || !features.ModelContextWindow {
				t.Fatalf("features = %+v, want linked=%t from declared path metadata", features, tc.want)
			}
		})
	}
}

func TestGemma4EngineFeaturesForIdentityUsesLabels(t *testing.T) {
	linked := Gemma4EngineFeaturesForIdentity(inference.ModelIdentity{
		Architecture: "gemma4_text",
		QuantBits:    6,
		NumLayers:    26,
		HiddenSize:   2304,
		Labels: map[string]string{
			"gemma4_size":       " E4B ",
			"gemma4_quant_mode": " Q6 ",
		},
	})
	if !linked.GenerateLinked() || !linked.DeviceKVState || !linked.ModelContextWindow {
		t.Fatalf("Gemma4 E4B q6 identity features = %+v, want linked generation", linked)
	}

	for name, identity := range map[string]inference.ModelIdentity{
		"gguf": {
			Architecture: "gemma4_text",
			QuantBits:    6,
			Labels: map[string]string{
				"format":            " GGUF ",
				"gemma4_size":       "E2B",
				"gemma4_quant_mode": "q6",
			},
		},
		"bf16": {
			Architecture: "gemma4_text",
			QuantBits:    16,
			Labels: map[string]string{
				"gemma4_size":       "E2B",
				"gemma4_quant_mode": "bf16",
			},
		},
		"status_only": {
			Architecture: "gemma4_text",
			QuantBits:    6,
			Labels: map[string]string{
				"gemma4_size":       "31b",
				"gemma4_quant_mode": "Q6",
			},
		},
		"load_only_label": {
			Architecture: "gemma4_text",
			QuantBits:    6,
			Labels: map[string]string{
				"gemma4_generate_status": " LOAD_ONLY ",
			},
		},
	} {
		features := Gemma4EngineFeaturesForIdentity(identity)
		if features.GenerateLinked() || features.DeviceKVState || !features.ModelContextWindow {
			t.Fatalf("%s identity features = %+v, want context-only load/status support", name, features)
		}
	}
}

func TestGemma4DeclaredFeaturesOfNativeConfig(t *testing.T) {
	features := Gemma4DeclaredFeaturesOfNativeConfig(nativeGemma4TextConfig{
		SlidingWindow:        1024,
		SlidingWindowPattern: 6,
		KVSharedLayers:       4,
		KVSharedLayersSet:    true,
		EnableMoEBlock:       true,
		NumExperts:           128,
		TopKExperts:          8,
		Vision:               true,
		Audio:                true,
	})
	if !features.Mixture ||
		features.NumExperts != 128 ||
		features.TopKExperts != 8 ||
		!features.Vision ||
		!features.Audio ||
		!features.Attention.Hybrid() ||
		features.Attention.SlidingWindow != 1024 ||
		features.Attention.SlidingPattern != 6 ||
		features.Attention.SharedKVLayers != 4 {
		t.Fatalf("declared features = %+v, want config-derived MoE, multimodal, and hybrid attention", features)
	}

	dense := Gemma4DeclaredFeaturesOfNativeConfig(nativeGemma4TextConfig{})
	if dense.Mixture || dense.Vision || dense.Audio || dense.Attention.Hybrid() || dense.Attention.SharedKVLayers != 0 {
		t.Fatalf("dense features = %+v, want zero feature surface when config declares none", dense)
	}
}

func TestGemma4EngineFeaturesCacheFromConfigLabels(t *testing.T) {
	base := inference.ModelIdentity{
		Architecture: "gemma4_text",
		QuantBits:    6,
		NumLayers:    26,
		HiddenSize:   2304,
		Labels: map[string]string{
			"gemma4_size":       "E4B",
			"gemma4_quant_mode": "q6",
		},
	}
	dense := Gemma4EngineFeaturesForIdentity(base)
	if !dense.GenerateLinked() ||
		!dense.DirectGreedyToken ||
		!dense.NativeMLPMatVec ||
		!dense.NativeLinearMatVec ||
		!dense.NativeQ6BitstreamMatVec ||
		!dense.NativeAttentionOMatVec ||
		!dense.GenerationStream ||
		!dense.AsyncDecodePrefetch ||
		dense.FixedSlidingCache ||
		dense.FixedSlidingCacheBound ||
		dense.NativeFixedSlidingAttention ||
		dense.CompiledLayerDecode ||
		dense.PipelinedDecode {
		t.Fatalf("dense features = %+v, want linked native fast paths without fixed-sliding/compiled/pipelined decode", dense)
	}

	for _, tc := range []struct {
		mode string
		bits int
	}{
		{mode: "q4", bits: 4},
		{mode: "q8", bits: 8},
	} {
		identity := base
		identity.QuantBits = tc.bits
		identity.Labels = cloneStringMap(base.Labels)
		identity.Labels["gemma4_quant_mode"] = tc.mode
		features := Gemma4EngineFeaturesForIdentity(identity)
		if !features.GenerateLinked() ||
			!features.NativeMLPMatVec ||
			!features.NativeLinearMatVec ||
			features.NativeQ6BitstreamMatVec ||
			!features.NativeAttentionOMatVec ||
			!features.GenerationStream ||
			!features.AsyncDecodePrefetch {
			t.Fatalf("linked %s features = %+v, want linked native paths without q6 bitstream", tc.mode, features)
		}
	}

	statusOnly := base
	statusOnly.QuantBits = 6
	statusOnly.Labels = cloneStringMap(base.Labels)
	statusOnly.Labels["gemma4_size"] = "31B"
	statusOnly.Labels["gemma4_quant_mode"] = "q6-status"
	statusOnlyFeatures := Gemma4EngineFeaturesForIdentity(statusOnly)
	if statusOnlyFeatures.GenerateLinked() || statusOnlyFeatures.NativeQ6BitstreamMatVec {
		t.Fatalf("status-only q6 features = %+v, want no linked generation or native q6 fast path", statusOnlyFeatures)
	}

	hybrid := base
	hybrid.Labels = cloneStringMap(base.Labels)
	hybrid.Labels["sliding_window"] = "1024"
	hybrid.Labels["sliding_window_pattern"] = "6"
	hybrid.Labels["attention_kv_shared_layers"] = "4"
	hybrid.Labels["gemma4_enable_moe_block"] = "true"
	hybrid.Labels["gemma4_num_experts"] = "128"
	hybrid.Labels["gemma4_top_k_experts"] = "8"
	hybridFeatures := Gemma4EngineFeaturesForIdentity(hybrid)
	declared := Gemma4DeclaredFeaturesForIdentity(hybrid)
	if !hybridFeatures.GenerateLinked() ||
		!hybridFeatures.DirectGreedyToken ||
		!hybridFeatures.NativeMLPMatVec ||
		!hybridFeatures.NativeLinearMatVec ||
		!hybridFeatures.NativeQ6BitstreamMatVec ||
		!hybridFeatures.NativeAttentionOMatVec ||
		!hybridFeatures.NativeFixedSlidingAttention ||
		!hybridFeatures.GenerationStream ||
		!hybridFeatures.AsyncDecodePrefetch ||
		!hybridFeatures.FixedSlidingCache ||
		!hybridFeatures.FixedSlidingCacheBound ||
		hybridFeatures.CompiledLayerDecode ||
		hybridFeatures.PipelinedDecode ||
		!declared.Mixture ||
		declared.NumExperts != 128 ||
		declared.TopKExperts != 8 ||
		declared.Attention.SlidingWindow != 1024 ||
		declared.Attention.SlidingPattern != 6 ||
		declared.Attention.SharedKVLayers != 4 {
		t.Fatalf("hybrid features = %+v declared=%+v, want config labels to select hybrid cache/MoE and linked native fast paths", hybridFeatures, declared)
	}
}

func TestGemma4DeclaredFeaturesForIdentityUsesMultimodalLabels(t *testing.T) {
	vision := Gemma4DeclaredFeaturesForIdentity(inference.ModelIdentity{
		Architecture: "gemma4",
		Labels: map[string]string{
			"multimodal_model":                  "true",
			"gemma4_multimodal":                 "true",
			"vision_model_type":                 "gemma4_vision",
			"image_token_id":                    "258880",
			"video_token_id":                    "258884",
			"vision_soft_tokens_per_image":      "280",
			"engine_multimodal_processor_audio": "false",
		},
	})
	if !vision.Vision || vision.Audio {
		t.Fatalf("vision declared features = %+v, want vision-only surface", vision)
	}
	visionLabels := map[string]string{}
	rocmApplyGemma4EngineFeatureLabels(visionLabels, Gemma4EngineFeatures{}, vision)
	if visionLabels["gemma4_multimodal"] != "true" ||
		visionLabels["gemma4_vision"] != "true" ||
		visionLabels["gemma4_audio"] != "" {
		t.Fatalf("vision labels = %+v, want Gemma4 multimodal/vision labels only", visionLabels)
	}

	audio := Gemma4DeclaredFeaturesForIdentity(inference.ModelIdentity{
		Architecture: "gemma4_unified",
		Labels: map[string]string{
			"engine_multimodal_processor_audio": "true",
			"audio_model_type":                  "gemma4_unified_audio",
			"audio_token_id":                    "258881",
			"audio_samples_per_token":           "640",
		},
	})
	if audio.Vision || !audio.Audio {
		t.Fatalf("audio declared features = %+v, want audio-only surface", audio)
	}
	audioLabels := map[string]string{}
	rocmApplyGemma4EngineFeatureLabels(audioLabels, Gemma4EngineFeatures{}, audio)
	if audioLabels["gemma4_multimodal"] != "true" ||
		audioLabels["gemma4_audio"] != "true" ||
		audioLabels["gemma4_vision"] != "" {
		t.Fatalf("audio labels = %+v, want Gemma4 multimodal/audio labels only", audioLabels)
	}

	empty := Gemma4DeclaredFeaturesForIdentity(inference.ModelIdentity{Architecture: "gemma4_text"})
	if empty.Vision || empty.Audio {
		t.Fatalf("empty declared features = %+v, want text-only surface", empty)
	}
}

func TestROCmModelRegistryGemma4ProfileReactsToLoadedConfig(t *testing.T) {
	factories := defaultROCmModelProfileRegistry().FactoryNames()
	if len(factories) != 2 || factories[0] != "gemma4" || factories[1] != "architecture-profile" {
		t.Fatalf("FactoryNames = %v, want Gemma4 and generic architecture-profile factories registered", factories)
	}
	factories[0] = "mutated"
	if next := defaultROCmModelProfileRegistry().FactoryNames(); len(next) != 2 || next[0] != "gemma4" || next[1] != "architecture-profile" {
		t.Fatalf("FactoryNames returned mutable registry state: %v", next)
	}

	profile, ok := defaultROCmModelProfileRegistry().Resolve(rocmModelProfileRequest{
		Path: "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		Model: inference.ModelIdentity{
			Path:         "/models/lmstudio-community-gemma-4-e4b-it-6bit",
			Architecture: "gemma4_text",
			QuantBits:    6,
			NumLayers:    26,
			HiddenSize:   2304,
			Labels:       linkedGemma4TestLabels("E4B", "q6"),
		},
		Gemma4TextConfig: nativeGemma4TextConfig{
			SlidingWindow:        1024,
			SlidingWindowPattern: 6,
			KVSharedLayers:       4,
			EnableMoEBlock:       true,
			NumExperts:           128,
			TopKExperts:          8,
			Vision:               true,
			Audio:                true,
		},
	})
	if !ok || !profile.Matched() || profile.Name != "gemma4" || profile.Registry != rocmModelRegistryName {
		t.Fatalf("profile = %+v ok=%v, want Gemma4 registry match", profile, ok)
	}
	if profile.Gemma4Settings.ID != "gemma4_text" ||
		profile.Gemma4Settings.ChatTemplate != "gemma4_hf_turn" ||
		!profile.Gemma4Settings.DefaultThinking ||
		!profile.Gemma4Settings.RequiresChatTemplate ||
		profile.Gemma4Settings.GenerationRole != "model" ||
		profile.Gemma4Settings.WeightWrapperPrefixes[0] != "model.language_model.model." {
		t.Fatalf("profile settings = %+v, want Gemma4 registry-owned architecture settings", profile.Gemma4Settings)
	}
	if strings.Join(profile.Gemma4LoRATargetPolicy.DefaultTargets, ",") != "q_proj,v_proj,o_proj" ||
		strings.Join(profile.Gemma4LoRATargetPolicy.SafeTargets, ",") != "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" ||
		strings.Join(profile.Gemma4LoRATargetPolicy.ExtendedTargets, ",") != "router.proj,per_layer_input_gate,per_layer_projection" {
		t.Fatalf("profile LoRA policy = %+v, want Gemma4 registry target policy", profile.Gemma4LoRATargetPolicy)
	}
	if !profile.Gemma4EngineFeatures.GenerateLinked() ||
		!profile.Gemma4EngineFeatures.DirectGreedyToken ||
		!profile.Gemma4EngineFeatures.NativeMLPMatVec ||
		!profile.Gemma4EngineFeatures.NativeLinearMatVec ||
		!profile.Gemma4EngineFeatures.NativeQ6BitstreamMatVec ||
		!profile.Gemma4EngineFeatures.NativeAttentionOMatVec ||
		!profile.Gemma4EngineFeatures.NativeFixedSlidingAttention ||
		!profile.Gemma4EngineFeatures.GenerationStream ||
		!profile.Gemma4EngineFeatures.AsyncDecodePrefetch ||
		!profile.Gemma4EngineFeatures.FixedSlidingCache ||
		!profile.Gemma4EngineFeatures.FixedSlidingCacheBound ||
		profile.Gemma4EngineFeatures.CompiledLayerDecode ||
		profile.Gemma4EngineFeatures.PipelinedDecode ||
		!profile.Gemma4DeclaredFeatures.Mixture ||
		!profile.Gemma4DeclaredFeatures.Vision ||
		!profile.Gemma4DeclaredFeatures.Audio ||
		profile.Gemma4DeclaredFeatures.NumExperts != 128 ||
		profile.Gemma4DeclaredFeatures.TopKExperts != 8 ||
		profile.Gemma4DeclaredFeatures.Attention.SlidingWindow != 1024 ||
		profile.Gemma4DeclaredFeatures.Attention.SlidingPattern != 6 ||
		profile.Gemma4DeclaredFeatures.Attention.SharedKVLayers != 4 {
		t.Fatalf("profile features = %+v declared=%+v, want config-owned engine profile", profile.Gemma4EngineFeatures, profile.Gemma4DeclaredFeatures)
	}
	labels := rocmApplyModelProfileLabels(nil, profile)
	if labels["engine_profile"] != "gemma4" ||
		labels["engine_profile_reactive"] != "true" ||
		labels["engine_text_generate"] != "true" ||
		labels["engine_direct_greedy_token"] != "true" ||
		labels["engine_native_mlp_matvec"] != "true" ||
		labels["engine_native_linear_matvec"] != "true" ||
		labels["engine_native_q6_bitstream_matvec"] != "true" ||
		labels["engine_native_attention_o_matvec"] != "true" ||
		labels["engine_native_fixed_sliding_attention"] != "true" ||
		labels["engine_generation_stream"] != "true" ||
		labels["engine_async_decode_prefetch"] != "true" ||
		labels["engine_fixed_sliding_cache"] != "true" ||
		labels["engine_compiled_layer_decode"] != "false" ||
		labels["engine_pipelined_decode"] != "false" ||
		labels["gemma4_attention_sliding_window"] != "1024" ||
		labels["gemma4_multimodal"] != "true" ||
		labels["gemma4_vision"] != "true" ||
		labels["gemma4_audio"] != "true" ||
		labels["engine_architecture_profile"] != "gemma4_text" ||
		labels["engine_architecture_runtime_status"] != string(inference.FeatureRuntimeNative) ||
		labels["engine_architecture_reasoning_parser"] != "gemma" ||
		labels["engine_architecture_tool_parser"] != "gemma" ||
		labels["engine_architecture_quantization_hints"] != "bf16,q8,q6,q4,mxfp8,mxfp4" ||
		labels["engine_architecture_cache_hints"] != "q8,paged,k-q8-v-q4,retained-state" ||
		labels["engine_chat_template"] != "gemma4_hf_turn" ||
		labels["chat_template"] != "gemma4_hf_turn" ||
		labels["engine_default_thinking"] != "true" ||
		labels["gemma4_weight_policy"] != "model_registry" ||
		labels["engine_lora_policy_source"] != "model_registry" ||
		labels["gemma4_lora_default_targets"] != "q_proj,v_proj,o_proj" ||
		labels["gemma4_lora_extended_targets_require_opt_in"] != "true" {
		t.Fatalf("profile labels = %+v, want engine registry and Gemma4 feature labels", labels)
	}
}

func TestROCmModelRegistryGemma4AssistantProfileIsAttachedOnly(t *testing.T) {
	assistant := rocmGemma4MTPAssistantIdentityForTarget(inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		Architecture: "gemma4_text",
		HiddenSize:   2304,
	})
	profile, ok := defaultROCmModelProfileRegistry().Resolve(rocmModelProfileRequest{
		Path:  assistant.Path,
		Model: assistant,
	})
	if !ok || !profile.Matched() || profile.Name != "gemma4" || profile.Architecture != "gemma4_assistant" {
		t.Fatalf("profile = %+v ok=%v, want Gemma4 assistant registry match", profile, ok)
	}
	if profile.Gemma4Settings.ID != "gemma4_assistant" ||
		!profile.Gemma4Settings.AttachedOnly ||
		profile.Gemma4Settings.Generation ||
		profile.Gemma4Settings.Chat ||
		profile.Gemma4Settings.ChatTemplate != "" ||
		len(profile.Gemma4LoRATargetPolicy.DefaultTargets) != 0 ||
		profile.Gemma4EngineFeatures.GenerateLinked() {
		t.Fatalf("profile = %+v, want attached-only assistant settings with no target generation/LoRA policy", profile)
	}
	labels := rocmApplyModelProfileLabels(nil, profile)
	if labels["engine_profile"] != "gemma4" ||
		labels["engine_architecture_profile"] != "gemma4_assistant" ||
		labels["engine_architecture_runtime_status"] != string(inference.FeatureRuntimeNative) ||
		labels["engine_architecture_reasoning_parser"] != "gemma" ||
		labels["engine_architecture_tool_parser"] != "gemma" ||
		labels["engine_architecture_attached_only"] != "true" ||
		labels["engine_architecture_generation"] != "false" ||
		labels["engine_architecture_cache_hints"] != "retained-state,attached-drafter" ||
		!strings.Contains(labels["engine_architecture_notes"], "attached MTP drafter") ||
		labels["engine_chat_template"] != "" ||
		labels["gemma4_lora_default_targets"] != "" ||
		labels["gemma4_weight_policy"] != "" {
		t.Fatalf("profile labels = %+v, want attached-only assistant registry labels", labels)
	}
}

func TestHIPLoadModelGemma4NativeConfigLabelsDriveEngineFeatures(t *testing.T) {
	driver := &fakeHIPDriver{
		available: true,
		device:    nativeDeviceInfo{Name: "gfx1100", MemoryBytes: 16 * memoryGiB, FreeBytes: 12 * memoryGiB, Driver: "fake"},
	}
	path, dataOffset := nativeHIPTensorGGUF(t)
	cfg := validHIPDriverFakeLoadConfigWithOffset(dataOffset)
	cfg.ModelInfo = inference.ModelInfo{
		Architecture: "gemma4_text",
		NumLayers:    26,
		HiddenSize:   2304,
		VocabSize:    262144,
		QuantBits:    6,
		QuantGroup:   64,
	}
	cfg.ModelLabels = map[string]string{
		"gemma4_size":       "E4B",
		"gemma4_quant_mode": "q6",
	}
	cfg.Gemma4TextConfig = nativeGemma4TextConfig{
		NumLayers:            4,
		LayerTypes:           []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"},
		SlidingWindow:        1024,
		SlidingWindowPattern: 6,
		KVSharedLayers:       2,
		EnableMoEBlock:       true,
		NumExperts:           16,
		TopKExperts:          2,
		Vision:               true,
		Audio:                true,
	}

	model, err := newHIPRuntime(driver).LoadModel(path, cfg)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	if !ok {
		t.Fatalf("model = %T, want *hipLoadedModel", model)
	}

	labels := loaded.modelLabels
	if labels["gemma4_sliding_window"] != "1024" ||
		labels["gemma4_sliding_window_pattern"] != "6" ||
		labels["gemma4_attention_kv_shared_layers"] != "2" ||
		labels["attention_layer_count"] != "4" ||
		labels["attention_cache_owner_by_layer"] != "0,1,0,1" ||
		labels["attention_cache_index_by_layer"] != "0,1,-1,-1" ||
		labels["attention_cache_shared_layers"] != "2" ||
		labels["gemma4_fixed_sliding_prefill_chunk_limit"] != "1024" ||
		labels["attention_window_policy"] != "sliding_causal" ||
		labels["gemma4_attention_mask_cached_offset_causal"] != "true" ||
		labels["gemma4_speculative_verify_proposal_window_limit"] != "1023" ||
		labels["gemma4_enable_moe_block"] != "true" ||
		labels["gemma4_num_experts"] != "16" ||
		labels["gemma4_top_k_experts"] != "2" ||
		labels["gemma4_multimodal"] != "true" ||
		labels["gemma4_vision"] != "true" ||
		labels["gemma4_audio"] != "true" {
		t.Fatalf("loaded labels = %+v, want Gemma4 native config feature labels", labels)
	}
	features := Gemma4EngineFeaturesForIdentity(inference.ModelIdentity{
		Architecture: loaded.modelInfo.Architecture,
		QuantBits:    loaded.modelInfo.QuantBits,
		QuantGroup:   loaded.modelInfo.QuantGroup,
		Labels:       labels,
	})
	if !features.GenerateLinked() || !features.FixedSlidingCache || !features.FixedSlidingCacheBound {
		t.Fatalf("features = %+v, want loaded config labels to drive linked hybrid engine features", features)
	}
}

func TestGemma4CapabilityReportEngineFeaturesFollowConfigLabels(t *testing.T) {
	model := inference.ModelIdentity{
		Architecture: "gemma4_text",
		QuantBits:    6,
		NumLayers:    26,
		HiddenSize:   2304,
		Labels: map[string]string{
			"gemma4_size":                       "E4B",
			"gemma4_quant_mode":                 "q6",
			"gemma4_sliding_window":             "1024",
			"gemma4_sliding_window_pattern":     "6",
			"gemma4_attention_kv_shared_layers": "4",
		},
	}
	report := rocmCapabilityReport(nativeDeviceInfo{}, model, inference.AdapterIdentity{}, true, defaultHIPKernelStatus())
	if report.Labels["engine_fixed_sliding_cache"] != "true" ||
		report.Labels["engine_fixed_sliding_cache_bound"] != "true" ||
		report.Labels["engine_native_q6_bitstream_matvec"] != "true" ||
		report.Labels["engine_native_fixed_sliding_attention"] != "true" ||
		report.Labels["engine_generation_stream"] != "true" ||
		report.Labels["engine_async_decode_prefetch"] != "true" ||
		report.Labels["engine_compiled_layer_decode"] != "false" ||
		report.Labels["engine_pipelined_decode"] != "false" ||
		report.Labels["gemma4_attention_sliding_window"] != "1024" ||
		report.Labels["gemma4_attention_sliding_pattern"] != "6" ||
		report.Labels["gemma4_attention_kv_shared_layers"] != "4" {
		t.Fatalf("report labels = %+v, want Gemma4 engine features from config labels", report.Labels)
	}
	modelLoad, ok := report.Capability(inference.CapabilityModelLoad)
	if !ok ||
		modelLoad.Labels["engine_fixed_sliding_cache"] != "true" ||
		modelLoad.Labels["engine_native_q6_bitstream_matvec"] != "true" ||
		modelLoad.Labels["engine_native_fixed_sliding_attention"] != "true" ||
		modelLoad.Labels["engine_async_decode_prefetch"] != "true" ||
		modelLoad.Labels["gemma4_attention_sliding_window"] != "1024" {
		t.Fatalf("model-load capability = %+v ok=%v, want engine feature labels propagated", modelLoad, ok)
	}

	dense := model
	dense.Labels = map[string]string{
		"gemma4_size":       "E4B",
		"gemma4_quant_mode": "q6",
	}
	denseReport := rocmCapabilityReport(nativeDeviceInfo{}, dense, inference.AdapterIdentity{}, true, defaultHIPKernelStatus())
	if denseReport.Labels["engine_fixed_sliding_cache"] != "false" ||
		denseReport.Labels["engine_native_q6_bitstream_matvec"] != "true" ||
		denseReport.Labels["engine_native_fixed_sliding_attention"] != "false" ||
		denseReport.Labels["engine_generation_stream"] != "true" ||
		denseReport.Labels["engine_async_decode_prefetch"] != "true" ||
		denseReport.Labels["gemma4_attention_sliding_window"] != "" {
		t.Fatalf("dense report labels = %+v, want no fixed-sliding cache without declared sliding window", denseReport.Labels)
	}
}

func TestHipLoadedGemma4Q4GenerateLinkedRejectsGGUFSource(t *testing.T) {
	info := inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6, NumLayers: 35, HiddenSize: 1536}
	if !hipLoadedGemma4Q4GenerateLinked(&hipLoadedModel{modelInfo: info, modelLabels: map[string]string{
		"gemma4_size":       "E2B",
		"gemma4_quant_mode": "q6",
	}}) {
		t.Fatalf("Gemma4 declared E2B q6 safetensors source should be linked")
	}
	if hipLoadedGemma4Q4GenerateLinked(&hipLoadedModel{modelInfo: info, modelLabels: map[string]string{"format": " GGUF "}}) {
		t.Fatalf("Gemma4 GGUF source must not claim linked MLX-affine generation")
	}
	if hipLoadedGemma4Q4GenerateLinked(&hipLoadedModel{modelInfo: info, modelLabels: map[string]string{"gemma4_source_format": " GGUF "}}) {
		t.Fatalf("Gemma4 GGUF source label must not claim linked MLX-affine generation")
	}
}

func TestHipLoadedGemma4Q4GenerateLinkedUsesEngineProfile(t *testing.T) {
	info := inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6, NumLayers: 26, HiddenSize: 2304}
	model := &hipLoadedModel{
		modelInfo:   info,
		modelLabels: linkedGemma4TestLabels("E4B", "q6"),
		engineProfile: ROCmModelProfile{
			Name:                 "gemma4",
			Family:               "gemma4",
			Registry:             rocmModelRegistryName,
			Gemma4EngineFeatures: Gemma4EngineFeatures{ModelContextWindow: true},
		},
	}
	if hipLoadedGemma4Q4GenerateLinked(model) {
		t.Fatalf("loaded model should follow engine profile feature decision before label fallback")
	}
	model.engineProfile.Gemma4EngineFeatures.TextGenerate = true
	model.engineProfile.Gemma4EngineFeatures.MLXAffineDecode = true
	if !hipLoadedGemma4Q4GenerateLinked(model) {
		t.Fatalf("loaded model should expose linked generation when engine profile enables it")
	}
}

func TestHipLoadedGemma4Q4GenerateLinkedUsesSizeQuantLabels(t *testing.T) {
	info := inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6, NumLayers: 64, HiddenSize: 4096}
	if hipLoadedGemma4Q4GenerateLinked(&hipLoadedModel{modelInfo: info, modelLabels: map[string]string{
		"gemma4_size":       "31B",
		"gemma4_quant_mode": "q6-status",
	}}) {
		t.Fatalf("Gemma4 31B q6-status loaded model must remain status-only")
	}
	if hipLoadedGemma4Q4GenerateLinked(&hipLoadedModel{modelInfo: info, modelLabels: map[string]string{
		"gemma4_size":       "31b",
		"gemma4_quant_mode": "Q6",
	}}) {
		t.Fatalf("Gemma4 31B carried q6 labels must normalize to status-only")
	}
	e4bInfo := inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6, NumLayers: 26, HiddenSize: 2304}
	if !hipLoadedGemma4Q4GenerateLinked(&hipLoadedModel{modelInfo: e4bInfo, modelLabels: map[string]string{
		"gemma4_size":       "E4B",
		"gemma4_quant_mode": "Q6",
	}}) {
		t.Fatalf("Gemma4 E4B carried q6 labels should remain linked")
	}
	if hipLoadedGemma4Q4GenerateLinked(&hipLoadedModel{modelInfo: e4bInfo, modelLabels: map[string]string{
		"gemma4_size":             "E4B",
		"gemma4_quant_mode":       "q6",
		"gemma4_runnable_on_card": "false",
	}}) {
		t.Fatalf("Gemma4 runnable-on-card=false label must veto linked generation")
	}
}

func TestGemma4CapabilityReportGenericLinkedKernelsStillUseMatrix(t *testing.T) {
	kernelStatus := hipKernelStatus{
		Decode:  hipKernelStatusLinked,
		Prefill: hipKernelStatusLinked,
		Reason:  "generic linked fixture",
	}
	for name, model := range map[string]inference.ModelIdentity{
		"gguf": {
			Architecture: "gemma4_text",
			QuantBits:    6,
			Labels: map[string]string{
				"format":            " GGUF ",
				"gemma4_size":       "E2B",
				"gemma4_quant_mode": "q6",
			},
		},
		"status_only": {
			Architecture: "gemma4_text",
			QuantBits:    6,
			Labels: map[string]string{
				"gemma4_size":       "31b",
				"gemma4_quant_mode": "Q6",
			},
		},
	} {
		report := rocmCapabilityReport(nativeDeviceInfo{}, model, inference.AdapterIdentity{}, true, kernelStatus, rocmCapabilityReportOption{
			ClassifyLinked:         true,
			Gemma4Q4GenerateLinked: true,
		})
		if report.Labels["decode_kernel"] == hipKernelStatusLinked ||
			report.Labels["prefill_kernel"] == hipKernelStatusLinked {
			t.Fatalf("%s report labels = %+v, want generic decode/prefill link hidden by Gemma4 matrix veto", name, report.Labels)
		}
		for _, id := range []inference.CapabilityID{
			inference.CapabilityGenerate,
			inference.CapabilityChat,
			inference.CapabilityBatchGenerate,
			inference.CapabilityClassify,
			inference.CapabilitySpeculativeDecode,
			inference.CapabilityPromptLookupDecode,
		} {
			capability, ok := report.Capability(id)
			if !ok || capability.Status == inference.CapabilityStatusExperimental ||
				capability.Labels["kernel_scope"] == "toy_tiny_fixture" ||
				capability.Labels["kernel_scope"] == "loaded_gemma4_q4_experimental_generate" {
				t.Fatalf("%s capability %s = %+v ok=%v, want Gemma4 matrix to veto linked generic promotion", name, id, capability, ok)
			}
		}
		benchmark, ok := report.Capability(inference.CapabilityBenchmark)
		if !ok ||
			benchmark.Labels["kernel_scope"] == "toy_tiny_fixture" ||
			benchmark.Labels["kernel_scope"] == "loaded_gemma4_q4_experimental_benchmark" ||
			benchmark.Labels["decode_kernel"] == hipKernelStatusLinked {
			t.Fatalf("%s benchmark capability = %+v ok=%v, want benchmark without linked decode labels", name, benchmark, ok)
		}
	}
}

func TestGemma4ReportKernelStatusUsesMatrix(t *testing.T) {
	status := hipKernelStatus{
		CrossEntropy: hipKernelStatusLinked,
		Decode:       hipKernelStatusLinked,
		Prefill:      hipKernelStatusLinked,
		Projection:   hipKernelStatusLinked,
		Reason:       "generic linked fixture",
	}
	blocked := rocmReportKernelStatusForModel(status, inference.ModelIdentity{
		Architecture: "gemma4_text",
		QuantBits:    6,
		Labels: map[string]string{
			"gemma4_size":       "31b",
			"gemma4_quant_mode": "q6",
		},
	})
	if blocked.Decode == hipKernelStatusLinked || blocked.Prefill == hipKernelStatusLinked ||
		blocked.CrossEntropy != hipKernelStatusLinked || blocked.Projection != hipKernelStatusLinked {
		t.Fatalf("blocked Gemma4 report status = %+v, want only generic decode/prefill hidden", blocked)
	}

	linked := rocmReportKernelStatusForModel(status, inference.ModelIdentity{
		Architecture: "gemma4_text",
		QuantBits:    6,
		Labels: map[string]string{
			"gemma4_size":       "E4B",
			"gemma4_quant_mode": "q6",
		},
	})
	if linked.Decode != hipKernelStatusLinked || linked.Prefill != hipKernelStatusLinked {
		t.Fatalf("linked Gemma4 report status = %+v, want generic decode/prefill preserved", linked)
	}

	nonGemma := rocmReportKernelStatusForModel(status, inference.ModelIdentity{Architecture: "tiny"})
	if nonGemma.Decode != hipKernelStatusLinked || nonGemma.Prefill != hipKernelStatusLinked {
		t.Fatalf("non-Gemma report status = %+v, want generic decode/prefill preserved", nonGemma)
	}
}

func TestGemma4BenchmarkHelperStatusUsesReportKernelStatus(t *testing.T) {
	raw := hipKernelStatus{Decode: hipKernelStatusLinked, Prefill: hipKernelStatusLinked}
	statusOnly := rocmReportKernelStatusForModel(raw, inference.ModelIdentity{
		Architecture: "gemma4_text",
		QuantBits:    6,
		Labels: map[string]string{
			"gemma4_size":       "31b",
			"gemma4_quant_mode": "q6",
		},
	})
	if got := rocmDecodeHelperStatusLabel(statusOnly, false); got != "planned" {
		t.Fatalf("status-only Gemma4 helper status = %q, want planned after report-kernel filtering", got)
	}
	linked := rocmReportKernelStatusForModel(raw, inference.ModelIdentity{
		Architecture: "gemma4_text",
		QuantBits:    6,
		Labels: map[string]string{
			"gemma4_size":       "E4B",
			"gemma4_quant_mode": "q6",
		},
	})
	if got := rocmDecodeHelperStatusLabel(linked, false); got != "experimental" {
		t.Fatalf("linked Gemma4 helper status = %q, want experimental", got)
	}
	if got := rocmDecodeHelperStatusLabel(statusOnly, true); got != "experimental" {
		t.Fatalf("explicit linked Gemma4 helper status = %q, want experimental", got)
	}
}

func TestPlanModelFitGemma4UsesSizeQuantMatrix(t *testing.T) {
	runtime := &fakeNativeRuntime{device: nativeDeviceInfo{MemoryBytes: 16 * memoryGiB, Name: "gfx1100"}}
	linked, err := newROCmBackendWithRuntime(runtime).PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture:  "gemma4_text",
		QuantBits:     6,
		ContextLength: 32768,
		NumLayers:     26,
		HiddenSize:    2304,
		Labels: map[string]string{
			"gemma4_size":       "E4B",
			"gemma4_quant_mode": "q6",
		},
	}, 0)
	if err != nil {
		t.Fatalf("PlanModelFit linked Gemma4: %v", err)
	}
	if linked == nil || !linked.Fits || !linked.QuantizationOK ||
		linked.MemoryPlan.Labels["gemma4_generate_status"] != Gemma4GenerateLinked {
		t.Fatalf("linked Gemma4 fit = %+v, want fitting linked q6 plan", linked)
	}

	for _, tc := range []struct {
		name        string
		model       inference.ModelIdentity
		wantFit     bool
		wantQuantOK bool
		wantStatus  string
	}{
		{
			name: "bf16_load_only",
			model: inference.ModelIdentity{
				Architecture:  "gemma4_text",
				QuantBits:     16,
				ContextLength: 32768,
				NumLayers:     35,
				HiddenSize:    1536,
				Labels: map[string]string{
					"gemma4_size":       "E2B",
					"gemma4_quant_mode": "bf16",
				},
			},
			wantFit:     true,
			wantQuantOK: true,
			wantStatus:  Gemma4GenerateLoadOnly,
		},
		{
			name: "status_only",
			model: inference.ModelIdentity{
				Architecture:  "gemma4_text",
				QuantBits:     6,
				ContextLength: 32768,
				NumLayers:     64,
				HiddenSize:    4096,
				Labels: map[string]string{
					"gemma4_size":       "31b",
					"gemma4_quant_mode": "q6",
				},
			},
			wantFit:     false,
			wantQuantOK: false,
			wantStatus:  Gemma4GeneratePlannedOnly,
		},
	} {
		report, err := newROCmBackendWithRuntime(runtime).PlanModelFit(context.Background(), tc.model, 0)
		if err != nil {
			t.Fatalf("%s PlanModelFit: %v", tc.name, err)
		}
		if report == nil || report.Fits != tc.wantFit || report.QuantizationOK != tc.wantQuantOK ||
			report.MemoryPlan.Labels["gemma4_generate_status"] != tc.wantStatus {
			t.Fatalf("%s fit = %+v, want fit=%t quantOK=%t status %s", tc.name, report, tc.wantFit, tc.wantQuantOK, tc.wantStatus)
		}
		foundNote := false
		for _, note := range report.MemoryPlan.Notes {
			if strings.Contains(note, "Gemma4 size/quant support matrix") {
				foundNote = true
				break
			}
		}
		if foundNote == tc.wantQuantOK {
			t.Fatalf("%s notes = %v, want Gemma4 matrix note only when quantization is not OK", tc.name, report.MemoryPlan.Notes)
		}
	}
}

func TestPlanModelFitGemma4InfersPathOnlyQuantMatrix(t *testing.T) {
	runtime := &fakeNativeRuntime{device: nativeDeviceInfo{MemoryBytes: 16 * memoryGiB, Name: "gfx1100"}}
	for _, tc := range []struct {
		name           string
		path           string
		hiddenSize     int
		layers         int
		wantQuantType  string
		wantQuantBits  int
		wantQuantGroup int
		wantMode       string
		wantStatus     string
		wantQuantOK    bool
		wantTrainingOK bool
	}{
		{name: "e4b_q8", path: "/models/lmstudio-community-gemma-4-e4b-it-8bit", hiddenSize: 2304, layers: 26, wantQuantType: "q8", wantQuantBits: 8, wantQuantGroup: 64, wantMode: "q8", wantStatus: Gemma4GenerateLinked, wantQuantOK: true, wantTrainingOK: true},
		{name: "e4b_q6", path: "/models/lmstudio-community-gemma-4-e4b-it-6bit", hiddenSize: 2304, layers: 26, wantQuantType: "q6", wantQuantBits: 6, wantQuantGroup: 64, wantMode: "q6", wantStatus: Gemma4GenerateLinked, wantQuantOK: true, wantTrainingOK: true},
		{name: "e4b_q4", path: "/models/lmstudio-community-gemma-4-e4b-it-4bit", hiddenSize: 2304, layers: 26, wantQuantType: "q4", wantQuantBits: 4, wantQuantGroup: 64, wantMode: "q4", wantStatus: Gemma4GenerateLinked, wantQuantOK: true, wantTrainingOK: true},
		{name: "e4b_mxfp8", path: "/models/lmstudio-community-gemma-4-e4b-it-mxfp8", hiddenSize: 2304, layers: 26, wantQuantType: "mxfp8", wantQuantBits: 8, wantQuantGroup: 32, wantMode: "mxfp8", wantStatus: Gemma4GeneratePlannedOnly},
		{name: "e4b_mxfp4", path: "/models/lmstudio-community-gemma-4-e4b-it-mxfp4", hiddenSize: 2304, layers: 26, wantQuantType: "mxfp4", wantQuantBits: 4, wantQuantGroup: 32, wantMode: "mxfp4", wantStatus: Gemma4GeneratePlannedOnly},
		{name: "e4b_bf16", path: "/models/lmstudio-community-gemma-4-e4b-it-bf16", hiddenSize: 2304, layers: 26, wantQuantType: "bf16", wantQuantBits: 16, wantMode: "bf16", wantStatus: Gemma4GenerateLoadOnly, wantQuantOK: true},
		{name: "12b_q6", path: "/models/lmstudio-community-gemma-4-12b-it-6bit", hiddenSize: 3840, layers: 48, wantQuantType: "q6", wantQuantBits: 6, wantQuantGroup: 64, wantMode: "q6", wantStatus: Gemma4GenerateLinked, wantQuantOK: true, wantTrainingOK: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			report, err := newROCmBackendWithRuntime(runtime).PlanModelFit(context.Background(), inference.ModelIdentity{
				Path:          tc.path,
				Architecture:  "gemma4_text",
				ContextLength: 8192,
				NumLayers:     tc.layers,
				HiddenSize:    tc.hiddenSize,
			}, 0)
			if err != nil {
				t.Fatalf("PlanModelFit: %v", err)
			}
			if report.Model.QuantType != tc.wantQuantType ||
				report.Model.QuantBits != tc.wantQuantBits ||
				report.Model.QuantGroup != tc.wantQuantGroup ||
				report.QuantizationOK != tc.wantQuantOK ||
				report.MemoryPlan.TrainingFeasible != tc.wantTrainingOK ||
				report.MemoryPlan.Labels["gemma4_quant_mode"] != tc.wantMode ||
				report.MemoryPlan.Labels["gemma4_generate_status"] != tc.wantStatus {
				t.Fatalf("fit = %+v labels=%+v, want path-only %s/%d group %d status %s", report, report.MemoryPlan.Labels, tc.wantQuantType, tc.wantQuantBits, tc.wantQuantGroup, tc.wantStatus)
			}
		})
	}
}

func TestROCmModelGemma4TextPromptSupportUsesLoadedMatrix(t *testing.T) {
	tokenText := &hipTokenTextDecoder{}
	linked := &rocmModel{native: &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6, NumLayers: productionLaneGemma4E2BLayers, HiddenSize: productionLaneGemma4E2BHiddenSize},
		modelLabels: map[string]string{
			"gemma4_size":       "E2B",
			"gemma4_quant_mode": "q6",
		},
		tokenText: tokenText,
	}}
	if !linked.gemma4Q4TextPromptSupported() {
		t.Fatalf("linked Gemma4 E2B q6 should support text prompt auto-routing")
	}
	gguf := &rocmModel{native: &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 4, NumLayers: productionLaneGemma4E2BLayers, HiddenSize: productionLaneGemma4E2BHiddenSize},
		modelLabels: map[string]string{
			"format":            "gguf",
			"gemma4_size":       "E2B",
			"gemma4_quant_mode": "q4",
		},
		tokenText: tokenText,
	}}
	if gguf.gemma4Q4TextPromptSupported() {
		t.Fatalf("Gemma4 GGUF load-only model must not auto-route plain text into linked generation")
	}
	statusOnly := &rocmModel{native: &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6, NumLayers: 64, HiddenSize: 4096},
		modelLabels: map[string]string{
			"gemma4_size":       "31B",
			"gemma4_quant_mode": "q6-status",
		},
		tokenText: tokenText,
	}}
	if statusOnly.gemma4Q4TextPromptSupported() {
		t.Fatalf("Gemma4 31B status-only model must not auto-route plain text into linked generation")
	}
}

func TestHIPGemma4PackageBranchesUseLoadedMatrix(t *testing.T) {
	kernels := hipNativeProjectionKernelSet{}
	linked := &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6},
		modelLabels: map[string]string{
			"gemma4_size":       "E2B",
			"gemma4_quant_mode": "q6",
		},
	}
	_, linkedErr := kernels.BatchGenerate(context.Background(), linked, []string{"tokens:1"}, inference.GenerateConfig{MaxTokens: 1})
	if linkedErr == nil || !strings.Contains(linkedErr.Error(), "layer count") {
		t.Fatalf("linked Gemma4 package branch should surface missing layer-count config error")
	}

	statusOnly := &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6},
		modelLabels: map[string]string{
			"gemma4_size":       "31B",
			"gemma4_quant_mode": "q6-status",
		},
	}
	results, err := kernels.BatchGenerate(context.Background(), statusOnly, []string{"tokens:1"}, inference.GenerateConfig{MaxTokens: 1})
	if err == nil || strings.Contains(err.Error(), "layer count") {
		t.Fatalf("status-only batch = %+v err=%v, want fallback not-linked error instead of Gemma4 package config error", results, err)
	}

	gguf := &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 4},
		modelLabels: map[string]string{
			"format":            "gguf",
			"gemma4_size":       "E2B",
			"gemma4_quant_mode": "q4",
		},
	}
	results, err = kernels.BatchGenerate(context.Background(), gguf, []string{"tokens:1"}, inference.GenerateConfig{MaxTokens: 1})
	if err == nil || strings.Contains(err.Error(), "layer count") {
		t.Fatalf("GGUF batch = %+v err=%v, want fallback not-linked error instead of Gemma4 package config error", results, err)
	}
}

func TestHIPGemma4PackageForwardConfigUsesLoadedMatrix(t *testing.T) {
	linked := &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6},
		modelLabels: map[string]string{
			"gemma4_size":       "E2B",
			"gemma4_quant_mode": "q6",
		},
	}
	_, ok, err := linked.loadedGemma4Q4PackageForwardConfig()
	if !ok || err == nil || !strings.Contains(err.Error(), "layer count") {
		t.Fatalf("linked package config ok=%v err=%v, want linked candidate with missing layer-count error", ok, err)
	}
	statusOnly := &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6},
		modelLabels: map[string]string{
			"gemma4_size":       "31B",
			"gemma4_quant_mode": "q6-status",
		},
	}
	_, ok, err = statusOnly.loadedGemma4Q4PackageForwardConfig()
	if ok || err != nil {
		t.Fatalf("status-only package config ok=%v err=%v, want not a linked package candidate", ok, err)
	}
	gguf := &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 4},
		modelLabels: map[string]string{
			"format":            "gguf",
			"gemma4_size":       "E2B",
			"gemma4_quant_mode": "q4",
		},
	}
	_, ok, err = gguf.loadedGemma4Q4PackageForwardConfig()
	if ok || err != nil {
		t.Fatalf("GGUF package config ok=%v err=%v, want not a linked package candidate", ok, err)
	}
}

func TestROCmGGUFNativeLoadLabelsGemma4LoadOnly(t *testing.T) {
	labels := rocmGGUFNativeLoadLabels(inference.ModelInfo{Architecture: "gemma4", QuantBits: 4}, "gemma-4-e2b-it-q4.gguf")
	if labels["format"] != "gguf" ||
		labels["gemma4_source_format"] != "gguf" ||
		labels["gemma4_size"] != "E2B" ||
		labels["gemma4_quant_mode"] != "q4" ||
		labels["gemma4_generate_status"] != Gemma4GenerateLoadOnly {
		t.Fatalf("labels = %+v, want Gemma4 GGUF load-only labels", labels)
	}
}

func TestLoadModelGemma4GGUFForwardsLoadOnlyLabels(t *testing.T) {
	runtime := &fakeNativeRuntime{available: true}
	model, err := resultValue[inference.TextModel](newROCmBackendWithRuntime(runtime).LoadModel(writeGemma4ModelPackGGUF(t)))
	if err != nil {
		t.Fatalf("LoadModel Gemma4 GGUF: %v", err)
	}
	defer model.Close()
	if runtime.loadConfig.ModelInfo.Architecture != "gemma4" ||
		runtime.loadConfig.ModelInfo.NumLayers != productionLaneGemma4E2BLayers ||
		runtime.loadConfig.ModelInfo.QuantBits != 4 {
		t.Fatalf("load model info = %+v, want Gemma4 E2B q4 GGUF identity", runtime.loadConfig.ModelInfo)
	}
	if runtime.loadConfig.ModelLabels["format"] != "gguf" ||
		runtime.loadConfig.ModelLabels["gemma4_source_format"] != "gguf" ||
		runtime.loadConfig.ModelLabels["gemma4_size"] != "E2B" ||
		runtime.loadConfig.ModelLabels["gemma4_quant_mode"] != "q4" ||
		runtime.loadConfig.ModelLabels["gemma4_generate_status"] != Gemma4GenerateLoadOnly {
		t.Fatalf("load labels = %+v, want Gemma4 GGUF load-only labels", runtime.loadConfig.ModelLabels)
	}
	if !runtime.loadConfig.EngineProfile.Matched() ||
		runtime.loadConfig.EngineProfile.Name != "gemma4" ||
		runtime.loadConfig.EngineProfile.Gemma4EngineFeatures.GenerateLinked() ||
		runtime.loadConfig.ModelLabels["engine_profile"] != "gemma4" ||
		runtime.loadConfig.ModelLabels["engine_text_generate"] != "false" {
		t.Fatalf("engine profile = %+v labels=%+v, want Gemma4 GGUF load-only registry profile", runtime.loadConfig.EngineProfile, runtime.loadConfig.ModelLabels)
	}
}

func TestLoadModelWithConfigForwardsDeviceKVMode(t *testing.T) {
	runtime := &fakeNativeRuntime{available: true}
	model, err := newROCmBackendWithRuntime(runtime).LoadModelWithConfig(writeGemma4ModelPackGGUF(t), ROCmLoadConfig{CacheMode: "q8"})
	if err != nil {
		t.Fatalf("LoadModelWithConfig Gemma4 GGUF: %v", err)
	}
	defer model.Close()
	if runtime.loadConfig.DeviceKVMode != rocmKVCacheModeQ8 ||
		runtime.loadConfig.ModelLabels["kv_cache_mode"] != rocmKVCacheModeQ8 ||
		runtime.loadConfig.ModelLabels["device_kv_mode"] != rocmKVCacheModeQ8 ||
		runtime.loadConfig.ModelLabels["kv_cache_source"] != "load_config" {
		t.Fatalf("load config device KV = %q labels=%+v, want q8 load-config binding", runtime.loadConfig.DeviceKVMode, runtime.loadConfig.ModelLabels)
	}
	rocmLoaded, ok := model.(*rocmModel)
	if !ok {
		t.Fatalf("model = %T, want *rocmModel", model)
	}
	report := rocmLoaded.Capabilities()
	if report.Model.Labels["kv_cache_mode"] != rocmKVCacheModeQ8 ||
		report.Model.Labels["device_kv_mode"] != rocmKVCacheModeQ8 ||
		report.Model.Labels["kv_cache_source"] != "load_config" ||
		report.Model.Labels["engine_profile"] != "gemma4" ||
		report.Model.Labels["engine_profile_reactive"] != "true" ||
		report.Model.Labels["gemma4_source_format"] != "gguf" {
		t.Fatalf("capability model labels = %+v, want ROCm load config and registry labels without hipLoadedModel", report.Model.Labels)
	}
}

func TestLoadModelWithConfigRejectsPlannedKVMode(t *testing.T) {
	runtime := &fakeNativeRuntime{available: true}
	_, err := newROCmBackendWithRuntime(runtime).LoadModelWithConfig("unused", ROCmLoadConfig{CacheMode: "turboquant"})
	if err == nil || !strings.Contains(err.Error(), `unsupported ROCm device KV cache mode "turboquant"`) {
		t.Fatalf("LoadModelWithConfig err = %v, want unsupported planned mode", err)
	}
}

func TestLoadModelGemma4GGUFHIPIdentityKeepsModelContext(t *testing.T) {
	runtime := &gemma4LoadConfigHIPRuntime{available: true}
	path := writeGemma4ModelPackGGUF(t)
	model, err := resultValue[inference.TextModel](newROCmBackendWithRuntime(runtime).LoadModel(path))
	if err != nil {
		t.Fatalf("LoadModel Gemma4 GGUF: %v", err)
	}
	defer model.Close()
	rocmModel, ok := model.(*rocmModel)
	if !ok {
		t.Fatalf("model = %T, want *rocmModel", model)
	}
	report := rocmModel.Capabilities()
	if report.Model.Path != path ||
		report.Model.ContextLength != 131072 ||
		report.Model.QuantType != "q4" ||
		report.Model.Labels["gemma4_size"] != "E2B" ||
		report.Model.Labels["gemma4_generate_status"] != Gemma4GenerateLoadOnly {
		t.Fatalf("capability model identity = %+v, want loaded Gemma4 GGUF path, context, quant, and load-only labels", report.Model)
	}
}

func TestROCmModelCapabilitiesInferLoadedGemma4PathQuant(t *testing.T) {
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    26,
			HiddenSize:   2304,
			VocabSize:    262144,
		},
		native: &fakeNativeModel{},
	}

	info := model.Info()
	if info.QuantBits != 6 {
		t.Fatalf("model info = %+v, want path-inferred q6 bits", info)
	}
	report := model.Capabilities()
	if report.Model.QuantType != "q6" || report.Model.QuantBits != 6 {
		t.Fatalf("report model = %+v, want loaded Gemma4 path-inferred q6 identity", report.Model)
	}
	modelLoad, ok := report.Capability(inference.CapabilityModelLoad)
	if !ok ||
		modelLoad.Labels["gemma4_size"] != "E4B" ||
		modelLoad.Labels["gemma4_quant_mode"] != "q6" ||
		modelLoad.Labels["gemma4_generate_status"] != Gemma4GenerateLinked {
		t.Fatalf("model-load capability = %+v ok=%v, want loaded Gemma4 path-inferred support labels", modelLoad, ok)
	}
}

func TestGemma4PlanningCapabilitiesCarryProductionMatrixLabels(t *testing.T) {
	tests := []struct {
		name         string
		path         string
		bits         int
		wantSize     string
		wantMode     string
		wantTier     string
		wantStatus   string
		wantRunnable string
	}{
		{name: "e2b_q6", path: "/models/lmstudio-community-gemma-4-e2b-it-6bit", bits: 6, wantSize: "E2B", wantMode: "q6", wantTier: "default", wantStatus: Gemma4GenerateLinked, wantRunnable: "true"},
		{name: "e4b_q4", path: "/models/lmstudio-community-gemma-4-e4b-it-4bit", bits: 4, wantSize: "E4B", wantMode: "q4", wantTier: "constrained", wantStatus: Gemma4GenerateLinked, wantRunnable: "true"},
		{name: "12b_q6", path: "/models/lmstudio-community-gemma-4-12b-it-6bit", bits: 6, wantSize: "12B", wantMode: "q6", wantTier: "largest-local-target", wantStatus: Gemma4GenerateLinked, wantRunnable: "true"},
		{name: "26b_a4b_q6", path: "/models/lmstudio-community-gemma-4-26b-a4b-it-6bit", bits: 6, wantSize: "26B-A4B", wantMode: "q6-status", wantTier: "status-only", wantStatus: Gemma4GeneratePlannedOnly, wantRunnable: "false"},
		{name: "31b_q4", path: "/models/lmstudio-community-gemma-4-31b-it-4bit", bits: 4, wantSize: "31B", wantMode: "q4-status", wantTier: "status-only", wantStatus: Gemma4GeneratePlannedOnly, wantRunnable: "false"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report := rocmCapabilityReport(nativeDeviceInfo{}, inference.ModelIdentity{
				Architecture: "gemma4_text",
				Path:         tt.path,
				QuantBits:    tt.bits,
				QuantGroup:   64,
			}, inference.AdapterIdentity{}, true, defaultHIPKernelStatus())

			for _, id := range []inference.CapabilityID{
				inference.CapabilityModelLoad,
				inference.CapabilityModelFit,
				inference.CapabilityMemoryPlanning,
				inference.CapabilityKVCachePlanning,
			} {
				capability, ok := report.Capability(id)
				if !ok {
					t.Fatalf("capability %s missing", id)
				}
				labels := capability.Labels
				if labels["gemma4_size"] != tt.wantSize ||
					labels["gemma4_quant_mode"] != tt.wantMode ||
					labels["gemma4_generate_status"] != tt.wantStatus ||
					labels["gemma4_runnable_on_card"] != tt.wantRunnable ||
					labels["production_quant_policy"] != "gemma4_mlx_affine" ||
					labels["production_quant_pack_sizes"] != "E2B,E4B,12B,26B-A4B,31B" ||
					labels["production_quant_size"] != tt.wantSize ||
					labels["production_quant_mode"] != tt.wantMode ||
					labels["production_quant_tier"] != tt.wantTier ||
					labels["production_quant_generate_status"] != tt.wantStatus ||
					labels["production_quant_runnable_on_card"] != tt.wantRunnable {
					t.Fatalf("capability %s labels = %+v, want Gemma4 %s/%s production matrix labels", id, labels, tt.wantSize, tt.wantMode)
				}
			}
		})
	}
}

func TestGemma4AssistantPlanningCapabilitiesCarryMTPLabels(t *testing.T) {
	report := rocmCapabilityReport(nativeDeviceInfo{}, inference.ModelIdentity{
		Architecture: "gemma4_assistant",
		Path:         "google/gemma-4-E4B-it-assistant",
		QuantBits:    16,
		QuantType:    "bf16",
	}, inference.AdapterIdentity{}, true, defaultHIPKernelStatus())

	for _, id := range []inference.CapabilityID{
		inference.CapabilityModelLoad,
		inference.CapabilityModelFit,
		inference.CapabilityMemoryPlanning,
		inference.CapabilityKVCachePlanning,
	} {
		capability, ok := report.Capability(id)
		if !ok {
			t.Fatalf("capability %s missing", id)
		}
		labels := capability.Labels
		if labels["gemma4_size"] != "E4B" ||
			labels["gemma4_quant_mode"] != "bf16" ||
			labels["gemma4_runtime"] != Gemma4RuntimeBF16 ||
			labels["gemma4_generate_status"] != Gemma4GenerateLoadOnly ||
			labels["gemma4_pack_supported"] != "true" ||
			labels["gemma4_runnable_on_card"] != "true" ||
			labels["engine_profile"] != "gemma4" ||
			labels["engine_profile_architecture"] != "gemma4_assistant" ||
			labels["engine_architecture_profile"] != "gemma4_assistant" ||
			labels["engine_architecture_runtime_status"] != string(inference.FeatureRuntimeNative) ||
			labels["engine_architecture_cache_hints"] != "retained-state,attached-drafter" ||
			labels["engine_architecture_attached_only"] != "true" ||
			labels["engine_architecture_generation"] != "false" ||
			labels["engine_architecture_chat"] != "false" ||
			labels["engine_chat_template"] != "" ||
			labels["gemma4_lora_default_targets"] != "" ||
			labels["gemma4_weight_policy"] != "" ||
			labels["attached_drafter_role"] != "gemma4_assistant" ||
			labels["attached_drafter_retained_state_entrypoint"] != hipKernelStatusLinked ||
			labels["attached_drafter_retained_state_required"] != "true" ||
			labels["attached_drafter_prompt_replay_fallback"] != "forbidden" ||
			labels["mtp_role"] != "drafter" ||
			labels["mtp_target_family"] != "gemma4" ||
			labels["production_quant_policy"] != "gemma4_mlx_affine" ||
			labels["production_quant_size"] != "E4B" ||
			labels["production_quant_mode"] != "bf16" ||
			labels["production_quant_bits"] != "16" ||
			labels["production_quant_tier"] != "mtp-assistant" ||
			labels["production_quant_pack"] != "E4B:assistant-bf16" ||
			labels["production_quant_model"] != "google/gemma-4-E4B-it-assistant" ||
			labels["production_quant_assistant_model"] != "google/gemma-4-E4B-it-assistant" ||
			labels["production_quant_mtp_assistant"] != "true" {
			t.Fatalf("capability %s labels = %+v, want Gemma4 E4B BF16 assistant planning labels", id, labels)
		}
	}
}

func TestROCmModelInfoPreservesGemma4MXFPQuantGroup(t *testing.T) {
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e4b-it-mxfp8",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    26,
			HiddenSize:   2304,
			VocabSize:    262144,
		},
		native: &fakeNativeModel{},
	}

	info := model.Info()
	if info.QuantBits != 8 || info.QuantGroup != 32 {
		t.Fatalf("model info = %+v, want Gemma4 E4B MXFP8 q8 group-32 identity", info)
	}
	report := model.Capabilities()
	if report.Model.QuantType != "mxfp8" ||
		report.Model.QuantBits != 8 ||
		report.Model.QuantGroup != 32 ||
		report.Model.Labels["gemma4_quant_mode"] != "mxfp8" ||
		report.Model.Labels["gemma4_generate_status"] != Gemma4GeneratePlannedOnly {
		t.Fatalf("capability model = %+v, want Gemma4 MXFP8 planned-only group-32 identity", report.Model)
	}
}

func TestLoadModelGemma4GGUFDoesNotExposeLinkedCapabilities(t *testing.T) {
	loaded := &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4", QuantBits: 4, NumLayers: productionLaneGemma4E2BLayers},
		modelLabels: map[string]string{
			"format":                 "gguf",
			"gemma4_source_format":   "gguf",
			"gemma4_size":            "E2B",
			"gemma4_quant_mode":      "q4",
			"gemma4_generate_status": Gemma4GenerateLoadOnly,
		},
	}
	model := &rocmModel{native: loaded, modelInfo: loaded.modelInfo}
	report := model.Capabilities()
	modelLoad, ok := report.Capability(inference.CapabilityModelLoad)
	if !ok ||
		modelLoad.Labels["gemma4_size"] != "E2B" ||
		modelLoad.Labels["gemma4_quant_mode"] != "q4" ||
		modelLoad.Labels["gemma4_source_format"] != "gguf" ||
		modelLoad.Labels["gemma4_generate_status"] != Gemma4GenerateLoadOnly {
		t.Fatalf("model-load capability = %+v ok=%v, want Gemma4 GGUF load-only labels", modelLoad, ok)
	}
	chatTemplate, ok := report.Capability(inference.CapabilityChatTemplate)
	if !ok ||
		chatTemplate.Labels["chat_template"] != "gemma4_hf_turn" ||
		chatTemplate.Labels["gemma4_size"] != "E2B" ||
		chatTemplate.Labels["gemma4_source_format"] != "gguf" ||
		chatTemplate.Labels["gemma4_generate_status"] != Gemma4GenerateLoadOnly {
		t.Fatalf("chat-template capability = %+v ok=%v, want Gemma4 GGUF template labels", chatTemplate, ok)
	}
	if generate, ok := report.Capability(inference.CapabilityGenerate); !ok ||
		generate.Status == inference.CapabilityStatusExperimental ||
		generate.Labels["gemma4_size"] != "E2B" ||
		generate.Labels["gemma4_quant_mode"] != "q4" ||
		generate.Labels["gemma4_source_format"] != "gguf" ||
		generate.Labels["gemma4_generate_status"] != Gemma4GenerateLoadOnly ||
		generate.Labels["kernel_scope"] == "loaded_gemma4_q4_experimental_generate" {
		t.Fatalf("generate capability = %+v ok=%v, Gemma4 GGUF load must not expose linked generation", generate, ok)
	}
	for _, id := range []inference.CapabilityID{
		inference.CapabilityModelFit,
		inference.CapabilityMemoryPlanning,
		inference.CapabilityKVCachePlanning,
		inference.CapabilityTokenizer,
		inference.CapabilityClassify,
		inference.CapabilityBenchmark,
		inference.CapabilityEvaluation,
		inference.CapabilitySpeculativeDecode,
		inference.CapabilityPromptLookupDecode,
	} {
		capability, ok := report.Capability(id)
		if !ok ||
			capability.Labels["gemma4_size"] != "E2B" ||
			capability.Labels["gemma4_quant_mode"] != "q4" ||
			capability.Labels["gemma4_source_format"] != "gguf" ||
			capability.Labels["gemma4_generate_status"] != Gemma4GenerateLoadOnly {
			t.Fatalf("capability %s = %+v ok=%v, want Gemma4 GGUF load-only labels", id, capability, ok)
		}
	}
}

func TestLoadModelGemma4GGUFSkipsLinkedWarmup(t *testing.T) {
	runtime := &gemma4LoadConfigHIPRuntime{available: true}
	model, err := resultValue[inference.TextModel](newROCmBackendWithRuntime(runtime).LoadModel(writeGemma4ModelPackGGUF(t)))
	if err != nil {
		t.Fatalf("LoadModel Gemma4 GGUF: %v", err)
	}
	defer model.Close()
	loaded, ok := model.(*rocmModel)
	if !ok {
		t.Fatalf("model = %T, want *rocmModel", model)
	}
	hipModel, ok := loaded.native.(*hipLoadedModel)
	if !ok {
		t.Fatalf("native = %T, want *hipLoadedModel", loaded.native)
	}
	if hipModel.q4ConfigOK {
		t.Fatalf("Gemma4 GGUF load prepared linked q4 config, want load-only warmup skipped")
	}
	if hipLoadedGemma4Q4GenerateLinked(hipModel) {
		t.Fatalf("Gemma4 GGUF loaded model must remain load-only")
	}
}

type gemma4LoadConfigHIPRuntime struct {
	available bool
	loadPath  string
	loadCfg   nativeLoadConfig
}

func (runtime *gemma4LoadConfigHIPRuntime) Available() bool { return runtime.available }

func (runtime *gemma4LoadConfigHIPRuntime) DeviceInfo() nativeDeviceInfo { return nativeDeviceInfo{} }

func (runtime *gemma4LoadConfigHIPRuntime) LoadModel(path string, cfg nativeLoadConfig) (nativeModel, error) {
	runtime.loadPath = path
	runtime.loadCfg = cfg
	return &hipLoadedModel{
		modelInfo:   cfg.ModelInfo,
		modelLabels: cloneStringMap(cfg.ModelLabels),
		contextSize: cfg.ContextSize,
		tensors:     map[string]hipTensor{},
	}, nil
}
