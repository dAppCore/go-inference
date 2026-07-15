// SPDX-Licence-Identifier: EUPL-1.2

//go:build !linux || !amd64 || rocm_legacy_server

package hip

import (
	"strconv"
	"strings"

	"dappco.re/go/inference"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
)

type Gemma4DeclaredFeatures struct {
	Mixture     bool                 `json:"mixture,omitempty"`
	NumExperts  int                  `json:"num_experts,omitempty"`
	TopKExperts int                  `json:"top_k_experts,omitempty"`
	Vision      bool                 `json:"vision,omitempty"`
	Audio       bool                 `json:"audio,omitempty"`
	Attention   Gemma4AttentionClass `json:"attention"`
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

func DefaultGemma4SizeQuantSupport() []Gemma4SizeQuantSupport {
	return modelgemma4.DefaultSizeQuantSupport()
}

func Gemma4SizeQuantSupportBySize(size string) (Gemma4SizeQuantSupport, bool) {
	return modelgemma4.SizeQuantSupportBySize(size)
}

func Gemma4QuantModeSupportBySize(size, mode string) (Gemma4QuantModeSupport, bool) {
	return modelgemma4.QuantModeSupportBySize(size, mode)
}

func DefaultProductionQuantizationPackSupport() []ProductionQuantizationPackSupport {
	return modelgemma4.DefaultProductionQuantizationPackSupport()
}

func ProductionQuantizationPacksBySize(size string) []ProductionQuantizationPackSupport {
	return modelgemma4.ProductionQuantizationPacksBySize(size)
}

func ProductionQuantizationPackByName(name string) (ProductionQuantizationPackSupport, bool) {
	return modelgemma4.ProductionQuantizationPackByName(name)
}

func ApplyProductionQuantizationPackSupportLabels(labels map[string]string) {
	modelgemma4.ApplyProductionQuantizationPackSupportLabels(labels)
}

func applyROCmPortableGemma4ModelPackSupportLabels(inspection *inference.ModelPackInspection) {
	if inspection == nil || !rocmIsGemma4SizeQuantIdentity(inspection.Model.Architecture) {
		return
	}
	model := inspection.Model
	model.Path = firstNonEmptyString(model.Path, inspection.Path)
	assistant := isROCmGemma4AssistantArchitecture(model.Architecture)
	size := rocmGemma4ModelPackSize(model, model.Path)
	mode := rocmGemma4ModelPackQuantModeForPath(model, model.Path)
	qatEntry, qatEntryOK := modelgemma4.QATCollectionEntryForModelID(model.Path)
	qatEntryOK = qatEntryOK && qatEntry.Assistant == assistant
	if qatEntryOK {
		size = qatEntry.Size
		mode = qatEntry.QuantMode
	} else if assistant {
		if support, ok := rocmGemma4MTPAssistantQuantModeSupport(size, mode); ok {
			mode = support.Mode
		}
	} else {
		mode = rocmGemma4NormalizeSizeQuantMode(size, mode)
	}
	if mode != "" {
		model = rocmGemma4ModelWithInferredQuantMode(model, mode)
	}
	inspection.Model = model
	if size != "" {
		inspection.Labels["gemma4_size"] = size
	}
	if mode != "" {
		inspection.Labels["gemma4_quant_mode"] = mode
	}
	model.Labels = inspection.Labels
	rocmApplyPortableGemma4RegistryLabels(inspection.Labels, model)
	if size == "" || mode == "" {
		return
	}
	var support Gemma4QuantModeSupport
	var ok bool
	if qatEntryOK {
		support = Gemma4QuantModeSupport{
			Mode:           qatEntry.QuantMode,
			Runtime:        qatEntry.Runtime,
			GenerateStatus: qatEntry.GenerateStatus,
		}
		ok = true
	} else if assistant {
		support, ok = rocmGemma4MTPAssistantQuantModeSupport(size, mode)
	} else {
		support, ok = Gemma4QuantModeSupportBySize(size, mode)
	}
	if !ok {
		inspection.Labels["gemma4_pack_supported"] = "false"
		inspection.Supported = false
		inspection.Notes = append(inspection.Notes, "Gemma4 "+size+" "+mode+" is not in the ROCm size/quant support matrix")
		return
	}
	sizeSupport, _ := Gemma4SizeQuantSupportBySize(size)
	if assistant {
		sizeSupport.RunnableOnCard = true
	}
	if qatEntryOK {
		sizeSupport.RunnableOnCard = qatEntry.RunnableOnCard
		inspection.Labels["gemma4_qat_collection"] = qatEntry.CollectionID
	}
	effectiveSupport := support
	if inspection.Format == "gguf" {
		effectiveSupport.Runtime = Gemma4RuntimeGGUF
		effectiveSupport.GenerateStatus = Gemma4GenerateLoadOnly
		inspection.Labels["gemma4_source_format"] = "gguf"
	}
	sizeSupport.RunnableOnCard = sizeSupport.RunnableOnCard && effectiveSupport.GenerateStatus != Gemma4GeneratePlannedOnly
	inspection.Labels["gemma4_pack_supported"] = "true"
	inspection.Labels["gemma4_runtime"] = effectiveSupport.Runtime
	inspection.Labels["gemma4_generate_status"] = effectiveSupport.GenerateStatus
	inspection.Labels["gemma4_runnable_on_card"] = strconv.FormatBool(sizeSupport.RunnableOnCard)
	model.Labels = inspection.Labels
	rocmApplyPortableGemma4RegistryLabels(inspection.Labels, model)
	rocmApplyGemma4ProductionQuantLabels(inspection.Labels, model)
	applyROCmPortableGemma4ModelPackSupportCapability(inspection, model, size, mode, effectiveSupport, sizeSupport, inspection.Labels["gemma4_source_format"])
	if !sizeSupport.RunnableOnCard || effectiveSupport.GenerateStatus == Gemma4GeneratePlannedOnly {
		inspection.Supported = false
	}
}

func applyROCmPortableGemma4ModelPackInspectionCapabilities(inspection *inference.ModelPackInspection) {
	if inspection == nil || !rocmIsGemma4SizeQuantIdentity(inspection.Model.Architecture) {
		return
	}
	model := inspection.Model
	model.Labels = inspection.Labels
	if isROCmGemma4Architecture(model.Architecture) {
		templateCapability := inference.ExperimentalCapability(inference.CapabilityChatTemplate, inference.CapabilityGroupModel, "Gemma4 HF-style turn template is available from the ROCm Gemma4 family profile")
		templateCapability.Labels = map[string]string{
			"chat_template":   "gemma4_hf_turn",
			"generation_role": "model",
			"runtime_status":  string(inference.FeatureRuntimeExperimental),
			"turn_end":        "<turn|>",
			"turn_start":      "<|turn>",
		}
		rocmApplyGemma4CapabilitySupportLabels(&templateCapability, model)
		appendROCmInspectionCapability(inspection, templateCapability)
	}
	for index := range inspection.Capabilities {
		rocmApplyGemma4CapabilitySupportLabels(&inspection.Capabilities[index], model)
		switch inspection.Capabilities[index].ID {
		case inference.CapabilityTokenizer, inference.CapabilityChatTemplate:
			inspection.Capabilities[index].Labels = rocmApplyROCmModelTokenizerCapabilityLabels(inspection.Capabilities[index].Labels, model)
		}
		if isROCmGemma4Architecture(model.Architecture) && inspection.Capabilities[index].ID == inference.CapabilityChatTemplate {
			labels := inspection.Capabilities[index].Labels
			if labels["chat_template"] == "" || labels["chat_template"] == "present" {
				labels["chat_template"] = "gemma4_hf_turn"
			}
			if labels["generation_role"] == "" {
				labels["generation_role"] = "model"
			}
			if labels["turn_start"] == "" {
				labels["turn_start"] = "<|turn>"
			}
			if labels["turn_end"] == "" {
				labels["turn_end"] = "<turn|>"
			}
			if labels["runtime_status"] == "" {
				labels["runtime_status"] = string(inference.FeatureRuntimeExperimental)
			}
		}
	}
}

func applyROCmPortableGemma4ModelPackSupportCapability(inspection *inference.ModelPackInspection, model inference.ModelIdentity, size, mode string, support Gemma4QuantModeSupport, sizeSupport Gemma4SizeQuantSupport, sourceFormat string) {
	labels := map[string]string{
		"gemma4_size":             size,
		"gemma4_quant_mode":       mode,
		"gemma4_runtime":          support.Runtime,
		"gemma4_generate_status":  support.GenerateStatus,
		"gemma4_pack_supported":   "true",
		"gemma4_runnable_on_card": strconv.FormatBool(sizeSupport.RunnableOnCard),
	}
	if sourceFormat != "" {
		labels["gemma4_source_format"] = sourceFormat
	}
	switch support.GenerateStatus {
	case Gemma4GenerateLinked:
		capability := inference.ExperimentalCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel, "Gemma4 "+size+" "+mode+" model-pack metadata matches the linked native generation path")
		capability.Labels = labels
		rocmApplyGemma4CapabilitySupportLabels(&capability, model)
		appendROCmInspectionCapability(inspection, capability)
	case Gemma4GenerateLoadOnly:
		capability := inference.SupportedCapability(inference.CapabilityModelLoad, inference.CapabilityGroupModel)
		capability.Detail = "Gemma4 " + size + " " + mode + " is recognised as load/metadata support; linked text generation is not claimed"
		capability.Labels = labels
		rocmApplyGemma4CapabilitySupportLabels(&capability, model)
		appendROCmInspectionCapability(inspection, capability)
	case Gemma4GeneratePlannedOnly:
		capability := inference.PlannedCapability(inference.CapabilityModelLoad, inference.CapabilityGroupModel, "Gemma4 "+size+" "+mode+" is recognised as status-only metadata; native load/generate is not claimed for this card")
		capability.Labels = labels
		rocmApplyGemma4CapabilitySupportLabels(&capability, model)
		appendROCmInspectionCapability(inspection, capability)
	}
}

func rocmApplyGemma4CapabilitySupportLabels(capability *inference.Capability, model inference.ModelIdentity) {
	if capability == nil || !rocmIsGemma4SizeQuantIdentity(model.Architecture) {
		return
	}
	if capability.Labels == nil {
		capability.Labels = map[string]string{}
	}
	rocmApplyResolvedModelProfileLabels(capability.Labels, model.Path, model)
	rocmApplyGemma4SizeQuantSupportLabels(capability.Labels, model)
	rocmApplyGemma4ProductionQuantLabels(capability.Labels, model)
	if isROCmGemma4AssistantArchitecture(model.Architecture) {
		capability.Labels["mtp_role"] = "drafter"
		capability.Labels["mtp_target_family"] = "gemma4"
	}
}

func rocmApplyPortableGemma4RegistryLabels(labels map[string]string, model inference.ModelIdentity) {
	if labels == nil || !rocmIsGemma4SizeQuantIdentity(model.Architecture) {
		return
	}
	model.Labels = labels
	profile := rocmResolvePortableModelProfile(model.Path, model)
	rocmApplyModelProfileLabels(labels, profile)
}

func rocmPortableAttentionConfigLabels(cfg rocmModelPackConfigProbe) map[string]string {
	out := map[string]string{}
	gemma4Architecture := isROCmGemma4Architecture(rocmConfigArchitecture(cfg))
	if slidingWindow := firstPositiveInt(cfg.SlidingWindow, cfg.TextConfig.SlidingWindow); slidingWindow > 0 {
		out["sliding_window"] = strconv.Itoa(slidingWindow)
	}
	if pattern := firstPositiveInt(cfg.SlidingWindowPattern, cfg.TextConfig.SlidingWindowPattern); pattern > 0 {
		out["sliding_window_pattern"] = strconv.Itoa(pattern)
	}
	if shared, ok := rocmConfigKVSharedLayers(cfg); ok {
		out["attention_kv_shared_layers"] = strconv.Itoa(shared)
	}
	if gemma4Architecture {
		rocmApplyGemma4ConfigLabels(out, rocmGemma4TextConfigFromProbe(cfg))
	}
	return out
}

func rocmConfigKVSharedLayers(cfg rocmModelPackConfigProbe) (int, bool) {
	switch {
	case cfg.NumKVSharedLayers != nil:
		return *cfg.NumKVSharedLayers, true
	case cfg.TextConfig.NumKVSharedLayers != nil:
		return *cfg.TextConfig.NumKVSharedLayers, true
	default:
		return 0, false
	}
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
	if rocmGemma4SupportMatrixGenerateLinked(identity) {
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

func gemma4EngineDenseBF16Linked(identity inference.ModelIdentity) bool {
	size := rocmGemma4ModelPackSize(identity, identity.Path)
	mode := rocmGemma4ModelPackQuantModeForPath(identity, identity.Path)
	return rocmGemma4NormalizeSizeQuantMode(size, mode) == "bf16"
}

func Gemma4DeclaredFeaturesForIdentity(identity inference.ModelIdentity) Gemma4DeclaredFeatures {
	return rocmGemma4DeclaredFeaturesForModel(identity)
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

func rocmApplyGemma4SizeQuantSupportLabels(labels map[string]string, model inference.ModelIdentity) {
	if labels == nil || !rocmIsGemma4SizeQuantIdentity(model.Architecture) {
		return
	}
	assistant := isROCmGemma4AssistantArchitecture(model.Architecture)
	size := rocmGemma4ModelPackSize(model, model.Path)
	mode := rocmGemma4ModelPackQuantModeForPath(model, model.Path)
	qatEntry, qatEntryOK := modelgemma4.QATCollectionEntryForModelID(model.Path)
	qatEntryOK = qatEntryOK && qatEntry.Assistant == assistant
	if qatEntryOK {
		size = qatEntry.Size
		mode = qatEntry.QuantMode
	} else if assistant {
		if support, ok := rocmGemma4MTPAssistantQuantModeSupport(size, mode); ok {
			mode = support.Mode
		}
	} else {
		mode = rocmGemma4NormalizeSizeQuantMode(size, mode)
	}
	if size != "" {
		labels["gemma4_size"] = size
	}
	if mode != "" {
		labels["gemma4_quant_mode"] = mode
	}
	if size == "" || mode == "" {
		return
	}
	var support Gemma4QuantModeSupport
	var ok bool
	if qatEntryOK {
		support = Gemma4QuantModeSupport{
			Mode:           qatEntry.QuantMode,
			Runtime:        qatEntry.Runtime,
			GenerateStatus: qatEntry.GenerateStatus,
		}
		ok = true
	} else if assistant {
		support, ok = rocmGemma4MTPAssistantQuantModeSupport(size, mode)
	} else {
		support, ok = Gemma4QuantModeSupportBySize(size, mode)
	}
	if !ok {
		labels["gemma4_pack_supported"] = "false"
		return
	}
	if rocmGemma4ModelSourceFormatGGUF(model) {
		support.Runtime = Gemma4RuntimeGGUF
		support.GenerateStatus = Gemma4GenerateLoadOnly
		labels["gemma4_source_format"] = "gguf"
	}
	sizeSupport, _ := Gemma4SizeQuantSupportBySize(size)
	if assistant {
		sizeSupport.RunnableOnCard = true
	}
	if qatEntryOK {
		sizeSupport.RunnableOnCard = qatEntry.RunnableOnCard
		labels["gemma4_qat_collection"] = qatEntry.CollectionID
	}
	sizeSupport.RunnableOnCard = sizeSupport.RunnableOnCard && support.GenerateStatus != Gemma4GeneratePlannedOnly
	labels["gemma4_pack_supported"] = "true"
	labels["gemma4_runtime"] = support.Runtime
	labels["gemma4_generate_status"] = support.GenerateStatus
	labels["gemma4_runnable_on_card"] = strconv.FormatBool(sizeSupport.RunnableOnCard)
}

func rocmGemma4SupportMatrixGenerateLinked(model inference.ModelIdentity) bool {
	if !isROCmGemma4Architecture(model.Architecture) {
		return false
	}
	if rocmGemma4ModelSourceFormatGGUF(model) || rocmGemma4LabelsVetoGenerateLinked(model.Labels) {
		return false
	}
	size := rocmGemma4ModelPackSize(model, model.Path)
	mode := rocmGemma4ModelPackQuantModeForPath(model, model.Path)
	if entry, ok := modelgemma4.QATCollectionEntryForModelID(model.Path); ok && !entry.Assistant {
		return entry.RunnableOnCard && entry.GenerateStatus == Gemma4GenerateLinked
	}
	mode = rocmGemma4NormalizeSizeQuantMode(size, mode)
	if size == "" || mode == "" {
		return false
	}
	support, ok := Gemma4QuantModeSupportBySize(size, mode)
	return ok && support.GenerateStatus == Gemma4GenerateLinked
}

func rocmGemma4MTPAssistantQuantModeSupport(size, mode string) (Gemma4QuantModeSupport, bool) {
	return modelgemma4.MTPAssistantQuantModeSupport(size, mode)
}

func rocmGemma4LabelValue(labels map[string]string, key string) string {
	return strings.ToLower(strings.TrimSpace(labels[key]))
}

func rocmGemma4SourceFormatGGUF(labels map[string]string) bool {
	return rocmGemma4LabelValue(labels, "gemma4_source_format") == "gguf" ||
		rocmGemma4LabelValue(labels, "format") == "gguf"
}

func rocmGemma4ModelSourceFormatGGUF(model inference.ModelIdentity) bool {
	return rocmGemma4SourceFormatGGUF(model.Labels) || strings.Contains(strings.ToLower(strings.TrimSpace(model.Path)), "gguf")
}

func rocmGemma4LabelsVetoGenerateLinked(labels map[string]string) bool {
	status := rocmGemma4LabelValue(labels, "gemma4_generate_status")
	moeUnlinked := rocmGemma4LabelValue(labels, "gemma4_enable_moe_block") == "true" &&
		rocmGemma4LabelValue(labels, "moe_text_runtime") != hipKernelStatusLinked
	return rocmGemma4LabelValue(labels, "gemma4_pack_supported") == "false" ||
		rocmGemma4LabelValue(labels, "gemma4_runnable_on_card") == "false" ||
		moeUnlinked ||
		status == Gemma4GenerateLoadOnly ||
		status == Gemma4GeneratePlannedOnly
}

func rocmGemma4ModelPackSize(model inference.ModelIdentity, path string) string {
	return modelgemma4.ModelPackSizeWithGeometry(model, path)
}

func rocmGemma4CanonicalSize(size string) string {
	return modelgemma4.CanonicalSize(size)
}

func rocmGemma4NormalizeSizeQuantMode(size, mode string) string {
	return modelgemma4.NormalizeSizeQuantMode(size, mode)
}

func rocmGemma4ModelPackQuantMode(model inference.ModelIdentity) string {
	return modelgemma4.ModelPackQuantModeWithGeometry(model)
}

func rocmGemma4ModelPackQuantModeForPath(model inference.ModelIdentity, path string) string {
	return modelgemma4.ModelPackQuantModeForPathWithGeometry(model, path)
}

func rocmGemma4ModelWithInferredPathQuant(model inference.ModelIdentity) inference.ModelIdentity {
	if !rocmIsGemma4SizeQuantIdentity(model.Architecture) {
		return model
	}
	mode := rocmGemma4ModelPackQuantModeForPath(model, model.Path)
	if !isROCmGemma4AssistantArchitecture(model.Architecture) {
		mode = rocmGemma4NormalizeSizeQuantMode(rocmGemma4ModelPackSize(model, model.Path), mode)
	}
	model = rocmGemma4ModelWithInferredQuantMode(model, mode)
	labels := cloneStringMap(model.Labels)
	if labels == nil {
		labels = map[string]string{}
	}
	rocmApplyGemma4SizeQuantSupportLabels(labels, model)
	if isROCmGemma4AssistantArchitecture(model.Architecture) {
		size := firstNonEmptyString(labels["gemma4_size"], rocmGemma4ModelPackSize(model, model.Path))
		mode := firstNonEmptyString(labels["gemma4_quant_mode"], rocmGemma4ModelPackQuantModeForPath(model, model.Path))
		if size != "" {
			if support, ok := rocmGemma4MTPAssistantQuantModeSupport(size, mode); ok && support.Mode == modelgemma4.AssistantQuantMode {
				labels = rocmGemma4MTPAssistantLabels(size, labels)
			}
		}
	}
	if len(labels) > 0 {
		model.Labels = labels
	}
	return model
}

func rocmGemma4PathQuantMode(path string) string {
	return modelgemma4.PathQuantMode(path)
}

func rocmGemma4ModelWithInferredQuantMode(model inference.ModelIdentity, mode string) inference.ModelIdentity {
	return modelgemma4.ModelWithInferredQuantMode(model, mode)
}

func rocmGemma4CanonicalQuantMode(size, mode string) string {
	return modelgemma4.CanonicalQuantMode(size, mode)
}

func rocmIsGemma4SizeQuantIdentity(architecture string) bool {
	return modelgemma4.IsSizeQuantIdentity(architecture)
}

func rocmApplyGemma4ProductionQuantLabels(labels map[string]string, model inference.ModelIdentity) {
	if labels == nil {
		return
	}
	labels["quant_family"] = "mlx_affine"
	labels["quant_default_tier"] = "q6"
	labels["quant_ladder"] = productionQuantizationLadderLabel
	labels["production_quant_policy"] = "gemma4_mlx_affine"
	labels["production_quant_default_bits"] = "6"
	labels["production_quant_quality_bits"] = "8"
	labels["production_quant_constrained_bits"] = "4"
	labels["production_quant_min_visible_tokens_per_sec"] = "100"
	ApplyProductionQuantizationPackSupportLabels(labels)

	model = rocmGemma4ModelWithInferredPathQuant(model)
	if pack, ok := rocmGemma4ProductionQuantPackForModel(model); ok {
		rocmApplyGemma4ProductionQuantPackLabels(labels, pack)
		rocmApplyGemma4EffectiveProductionQuantLabels(labels, model)
		return
	}
	bits := rocmModelQuantBits(model)
	if bits > 0 {
		if tier := rocmGemma4ProductionQuantTierForBits(bits); tier != "" {
			labels["production_quant_tier"] = tier
			rocmApplyGemma4StaticProductionQuantTierLabels(labels, bits)
		} else {
			labels["production_quant_bits"] = strconv.Itoa(bits)
			labels["production_quant_tier"] = "custom"
		}
		if size := rocmGemma4ModelPackSize(model, model.Path); size != "" {
			labels["production_quant_size"] = size
		}
		if mode := rocmGemma4ModelPackQuantModeForPath(model, model.Path); mode != "" {
			labels["production_quant_mode"] = rocmGemma4NormalizeSizeQuantMode(rocmGemma4ModelPackSize(model, model.Path), mode)
		}
	}
	rocmApplyGemma4EffectiveProductionQuantLabels(labels, model)
}

func rocmApplyGemma4EffectiveProductionQuantLabels(labels map[string]string, model inference.ModelIdentity) {
	if labels == nil {
		return
	}
	if value := model.Labels["gemma4_runtime"]; value != "" {
		labels["production_quant_runtime"] = value
	}
	if value := model.Labels["gemma4_generate_status"]; value != "" {
		labels["production_quant_generate_status"] = value
	}
	if value := model.Labels["gemma4_pack_supported"]; value != "" {
		labels["production_quant_supported"] = value
	}
	if value := model.Labels["gemma4_runnable_on_card"]; value != "" {
		labels["production_quant_runnable_on_card"] = value
	}
}

func rocmGemma4ProductionQuantPackForModel(model inference.ModelIdentity) (ProductionQuantizationPackSupport, bool) {
	return modelgemma4.ProductionQuantizationPackForModel(model)
}

func rocmApplyGemma4ProductionQuantPackLabels(labels map[string]string, pack ProductionQuantizationPackSupport) {
	if labels == nil {
		return
	}
	labels["production_quant_size"] = pack.Size
	labels["production_quant_pack"] = productionQuantizationPackLabelName(pack)
	labels["production_quant_pack_name"] = pack.Name
	labels["production_quant_tier"] = pack.ProductRole
	labels["production_quant_model"] = pack.ModelID
	if pack.SourceCollection != "" {
		labels["production_quant_collection"] = pack.SourceCollection
	}
	if pack.LockedModelID != "" {
		labels["production_quant_locked_model"] = pack.LockedModelID
	}
	labels["production_quant_mode"] = rocmGemma4ProductionQuantPackMode(pack)
	labels["production_quant_bits"] = strconv.Itoa(pack.Bits)
	if pack.QuantGroup > 0 {
		labels["production_quant_group"] = strconv.Itoa(pack.QuantGroup)
	}
	if pack.Runtime != "" {
		labels["production_quant_runtime"] = pack.Runtime
	}
	if pack.GenerateStatus != "" {
		labels["production_quant_generate_status"] = pack.GenerateStatus
	}
	labels["production_quant_supported"] = strconv.FormatBool(pack.Supported)
	labels["production_quant_runnable_on_card"] = strconv.FormatBool(pack.RunnableOnCard)
	if pack.RequiresBench {
		labels["production_quant_requires_bench"] = "true"
	}
	if pack.RequiresNative {
		labels["production_quant_requires_native"] = "true"
	}
	if pack.ProductRole != "mtp-assistant" {
		if target, ok := rocmGemma4ProductionQuantPackBySizeRole(pack.Size, "default"); ok {
			labels["production_quant_target_model"] = target.ModelID
		} else if pack.ProductRole == "largest-local-target" {
			labels["production_quant_target_model"] = pack.ModelID
		}
		if quality, ok := rocmGemma4ProductionQuantPackBySizeRole(pack.Size, "quality"); ok {
			labels["production_quant_quality_model"] = quality.ModelID
		}
		if constrained, ok := rocmGemma4ProductionQuantPackBySizeRole(pack.Size, "constrained"); ok {
			labels["production_quant_archived_baseline"] = constrained.ModelID
		}
	}
	switch pack.ProductRole {
	case "quality":
		labels["production_quant_quality_first"] = "true"
		if pack.Size == "E2B" {
			rocmApplyGemma4StaticProductionQuantTierLabels(labels, pack.Bits)
		}
	case "default":
		labels["production_quant_product_default"] = "true"
		labels["production_quant_size_default"] = "true"
		if pack.Size == "E2B" {
			rocmApplyGemma4StaticProductionQuantTierLabels(labels, pack.Bits)
		}
	case "constrained":
		labels["production_quant_constrained_only"] = "true"
		if pack.ModelID == ProductionLaneArchivedBaselineModelID || pack.ModelID == ProductionLaneCurrentConstrainedModelID {
			labels["production_quant_archived_control"] = "true"
			rocmApplyGemma4StaticProductionQuantTierLabels(labels, pack.Bits)
		}
	case "largest-local-target":
		labels["production_quant_size_default"] = "true"
	case "mtp-assistant":
		labels["production_quant_mtp_assistant"] = "true"
		labels["production_quant_assistant_model"] = pack.ModelID
		labels["production_quant_target_family"] = "gemma4"
	}
}

func rocmGemma4ProductionQuantPackMode(pack ProductionQuantizationPackSupport) string {
	return modelgemma4.ProductionQuantizationPackMode(pack)
}

func productionQuantizationPackLabelName(pack ProductionQuantizationPackSupport) string {
	return modelgemma4.ProductionQuantizationPackLabelName(pack)
}

func rocmGemma4ProductionQuantPackBySizeRole(size, role string) (ProductionQuantizationPackSupport, bool) {
	return modelgemma4.ProductionQuantizationPackBySizeRole(size, role)
}

func rocmGemma4ProductionQuantTierForBits(bits int) string {
	switch bits {
	case ProductionLaneQualityQuantBits:
		return "quality"
	case ProductionLaneProductDefaultQuantBits:
		return "default"
	case ProductionLaneConstrainedQuantBits:
		return "constrained"
	default:
		return ""
	}
}

func rocmApplyGemma4StaticProductionQuantTierLabels(labels map[string]string, bits int) {
	switch bits {
	case ProductionLaneQualityQuantBits:
		labels["production_quant_bits"] = "8"
		labels["production_quant_group"] = "64"
		labels["production_quant_active_weight_read_bytes_per_token"] = "2300000000"
		labels["production_quant_step_down_to_bits"] = "6"
	case ProductionLaneProductDefaultQuantBits:
		labels["production_quant_bits"] = "6"
		labels["production_quant_group"] = "64"
		labels["production_quant_active_weight_read_bytes_per_token"] = "1725000000"
		labels["production_quant_step_down_to_bits"] = "4"
	case ProductionLaneConstrainedQuantBits:
		labels["production_quant_bits"] = "4"
		labels["production_quant_group"] = "64"
		labels["production_quant_active_weight_read_bytes_per_token"] = "1150000000"
	}
}

func rocmModelQuantBits(model inference.ModelIdentity) int {
	if model.QuantBits > 0 {
		return model.QuantBits
	}
	quantType := strings.TrimPrefix(strings.ToLower(model.QuantType), "mlx_")
	quantType = strings.TrimPrefix(quantType, "affine_")
	quantType = strings.TrimPrefix(quantType, "q")
	bits, err := strconv.Atoi(quantType)
	if err != nil {
		return 0
	}
	return bits
}

func rocmGemma4MTPAssistantLabels(size string, labels map[string]string) map[string]string {
	out := modelgemma4.MTPAssistantLabels(size, labels)
	out = rocmApplyStaticGemma4ModelProfileLabels(out, portableOfficialGemma4E2BAssistantArchitecture)
	return out
}

func rocmMTPAssistantPackName(size string) string {
	return modelgemma4.MTPAssistantPackName(size)
}

func rocmGemma4MTPAssistantPath(size, mode string) string {
	return modelgemma4.MTPAssistantPath(size, mode)
}
