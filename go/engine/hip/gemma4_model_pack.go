// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/gguf"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
)

func applyROCmGemma4ModelPackSupportLabels(inspection *inference.ModelPackInspection, path string) {
	if inspection == nil || !rocmIsGemma4SizeQuantIdentity(inspection.Model.Architecture) {
		return
	}
	model := inspection.Model
	assistant := isROCmGemma4AssistantArchitecture(model.Architecture)
	size := rocmGemma4ModelPackSize(model, path)
	mode := rocmGemma4ModelPackQuantModeForPath(model, path)
	qatEntry, qatEntryOK := modelgemma4.QATCollectionEntryForModelID(path)
	qatEntryOK = qatEntryOK && qatEntry.Assistant == assistant
	if qatEntryOK {
		size = qatEntry.Size
		mode = qatEntry.QuantMode
	} else if !assistant {
		mode = rocmGemma4NormalizeSizeQuantMode(size, mode)
	}
	if mode != "" {
		model = rocmGemma4ModelWithInferredQuantMode(model, mode)
		inspection.Model = model
	}
	if size != "" {
		inspection.Labels["gemma4_size"] = size
	}
	if mode != "" {
		inspection.Labels["gemma4_quant_mode"] = mode
	}
	model.Labels = inspection.Labels
	if profile, ok := defaultROCmModelProfileRegistry().Resolve(rocmModelProfileRequest{
		Path:  path,
		Model: model,
	}); ok {
		rocmApplyModelProfileLabels(inspection.Labels, profile)
		model.Labels = inspection.Labels
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
	inspection.Labels["gemma4_pack_supported"] = "true"
	inspection.Labels["gemma4_runtime"] = effectiveSupport.Runtime
	inspection.Labels["gemma4_generate_status"] = effectiveSupport.GenerateStatus
	inspection.Labels["gemma4_runnable_on_card"] = core.Sprintf("%t", sizeSupport.RunnableOnCard)
	model.Labels = inspection.Labels
	if profile, ok := defaultROCmModelProfileRegistry().Resolve(rocmModelProfileRequest{
		Path:  path,
		Model: model,
	}); ok {
		rocmApplyModelProfileLabels(inspection.Labels, profile)
		model.Labels = inspection.Labels
	}
	applyROCmGemma4ModelPackSupportCapability(inspection, model, size, mode, effectiveSupport, sizeSupport, inspection.Labels["gemma4_source_format"])
	if !sizeSupport.RunnableOnCard || effectiveSupport.GenerateStatus == Gemma4GeneratePlannedOnly {
		inspection.Supported = false
	}
}

func rocmGemma4MTPAssistantQuantModeSupport(size, mode string) (Gemma4QuantModeSupport, bool) {
	return modelgemma4.MTPAssistantQuantModeSupport(size, mode)
}

func applyROCmGemma4ModelPackSupportCapability(inspection *inference.ModelPackInspection, model inference.ModelIdentity, size, mode string, support Gemma4QuantModeSupport, sizeSupport Gemma4SizeQuantSupport, sourceFormat string) {
	labels := map[string]string{
		"gemma4_size":             size,
		"gemma4_quant_mode":       mode,
		"gemma4_runtime":          support.Runtime,
		"gemma4_generate_status":  support.GenerateStatus,
		"gemma4_pack_supported":   "true",
		"gemma4_runnable_on_card": core.Sprintf("%t", sizeSupport.RunnableOnCard),
	}
	if sourceFormat != "" {
		labels["gemma4_source_format"] = sourceFormat
	}
	switch support.GenerateStatus {
	case Gemma4GenerateLinked:
		capability := inference.ExperimentalCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel, "Gemma4 "+size+" "+mode+" model-pack metadata matches the linked MLX-affine generation path")
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

func applyROCmGemma4ModelPackInspectionCapabilities(inspection *inference.ModelPackInspection) {
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
		if inspection.Capabilities[index].Labels == nil {
			inspection.Capabilities[index].Labels = map[string]string{}
		}
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

func rocmApplyGemma4SizeQuantSupportLabels(labels map[string]string, model inference.ModelIdentity) {
	if labels == nil || !rocmIsGemma4SizeQuantIdentity(model.Architecture) {
		return
	}
	assistant := isROCmGemma4AssistantArchitecture(model.Architecture)
	explicitStatus := rocmGemma4LabelValue(model.Labels, "gemma4_generate_status")
	explicitRuntime := strings.TrimSpace(model.Labels["gemma4_runtime"])
	explicitSupported := rocmGemma4LabelValue(model.Labels, "gemma4_pack_supported")
	explicitRunnable := rocmGemma4LabelValue(model.Labels, "gemma4_runnable_on_card")
	size := firstNonEmptyString(model.Labels["gemma4_size"], rocmGemma4ModelPackSize(model, model.Path))
	mode := firstNonEmptyString(model.Labels["gemma4_quant_mode"], rocmGemma4ModelPackQuantModeForPath(model, model.Path))
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
	if canonicalSize := rocmGemma4CanonicalSize(size); canonicalSize != "" {
		size = canonicalSize
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
		labels["gemma4_source_format"] = "gguf"
	}
	if explicitRuntime != "" {
		support.Runtime = explicitRuntime
	}
	if explicitStatus == Gemma4GenerateLinked || explicitStatus == Gemma4GenerateLoadOnly || explicitStatus == Gemma4GeneratePlannedOnly {
		support.GenerateStatus = explicitStatus
	}
	sizeSupport, _ := Gemma4SizeQuantSupportBySize(size)
	if assistant {
		sizeSupport.RunnableOnCard = true
	}
	if qatEntryOK {
		sizeSupport.RunnableOnCard = qatEntry.RunnableOnCard
		labels["gemma4_qat_collection"] = qatEntry.CollectionID
	}
	if explicitSupported == "false" {
		labels["gemma4_pack_supported"] = "false"
	} else {
		labels["gemma4_pack_supported"] = "true"
	}
	labels["gemma4_runtime"] = support.Runtime
	labels["gemma4_generate_status"] = support.GenerateStatus
	if explicitRunnable == "false" {
		labels["gemma4_runnable_on_card"] = "false"
	} else {
		labels["gemma4_runnable_on_card"] = core.Sprintf("%t", sizeSupport.RunnableOnCard)
	}
}

func rocmGemma4SupportMatrixGenerateLinked(model inference.ModelIdentity) bool {
	if !isROCmGemma4Architecture(model.Architecture) {
		return false
	}
	if rocmGemma4LabelsVetoGenerateLinked(model.Labels) {
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
	// The parallelised MoE router and GGUF expert kernels are now linked in
	// this engine, so an MoE block defaults to linked. Only an explicit
	// not-linked/planned runtime status still vetoes generate-linked; an unset
	// moe_text_runtime label no longer implies an unintegrated MoE runtime.
	moeRuntime := rocmGemma4LabelValue(labels, "moe_text_runtime")
	moeUnlinked := rocmGemma4LabelValue(labels, "gemma4_enable_moe_block") == "true" &&
		moeRuntime != "" && moeRuntime != hipKernelStatusLinked
	return rocmGemma4LabelValue(labels, "gemma4_pack_supported") == "false" ||
		rocmGemma4LabelValue(labels, "gemma4_runnable_on_card") == "false" ||
		moeUnlinked ||
		status == Gemma4GenerateLoadOnly ||
		status == Gemma4GeneratePlannedOnly
}

func rocmGemma4ModelPackSize(model inference.ModelIdentity, path string) string {
	return modelgemma4.ModelPackSize(model, path)
}

func rocmGemma4CanonicalSize(size string) string {
	return modelgemma4.CanonicalSize(size)
}

func rocmGemma4NormalizeSizeQuantMode(size, mode string) string {
	return modelgemma4.NormalizeSizeQuantMode(size, mode)
}

func rocmGemma4ModelPackQuantMode(model inference.ModelIdentity) string {
	return modelgemma4.ModelPackQuantMode(model)
}

func rocmGemma4ModelPackQuantModeForPath(model inference.ModelIdentity, path string) string {
	return modelgemma4.ModelPackQuantModeForPath(model, path)
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

func rocmGemma4ModelInfoIdentity(info inference.ModelInfo, path string) inference.ModelIdentity {
	return rocmGemma4ModelWithInferredPathQuant(inference.ModelIdentity{
		Architecture: info.Architecture,
		Path:         path,
		VocabSize:    info.VocabSize,
		NumLayers:    info.NumLayers,
		HiddenSize:   info.HiddenSize,
		QuantBits:    info.QuantBits,
		QuantGroup:   info.QuantGroup,
	})
}

func rocmGGUFNativeLoadLabels(info inference.ModelInfo, path string, metadata gguf.Metadata) map[string]string {
	labels := map[string]string{"format": "gguf"}
	if isROCmGemma4Architecture(info.Architecture) {
		labels["gemma4_source_format"] = "gguf"
		identity := inference.ModelIdentity{
			Architecture: info.Architecture,
			NumLayers:    info.NumLayers,
			HiddenSize:   info.HiddenSize,
			VocabSize:    info.VocabSize,
			QuantBits:    info.QuantBits,
			QuantGroup:   info.QuantGroup,
		}
		if size := rocmGemma4ModelPackSize(identity, path); size != "" {
			labels["gemma4_size"] = size
		}
		if mode := rocmGemma4ModelPackQuantModeForPath(identity, path); mode != "" {
			labels["gemma4_quant_mode"] = mode
		}
		identity.Path = path
		identity.Labels = labels
		if rocmGemma4SupportMatrixGenerateLinked(identity) {
			labels["gemma4_generate_status"] = Gemma4GenerateLinked
		} else {
			labels["gemma4_generate_status"] = Gemma4GenerateLoadOnly
		}
	}
	if isROCmGemma4AssistantArchitecture(info.Architecture) && metadata.EmbeddingLengthOut > 0 {
		var contradictsOfficial bool
		labels, contradictsOfficial = modelgemma4.ApplyAssistantConfigLabels(labels, modelgemma4.AssistantConfig{
			BackboneHiddenSize: int(metadata.EmbeddingLengthOut),
		})
		if contradictsOfficial {
			labels["attached_drafter_official_pair_verified"] = "false"
			labels["attached_drafter_gemma4_family_pair_verified"] = "false"
		}
	}
	return labels
}

func rocmIsGemma4SizeQuantIdentity(architecture string) bool {
	return modelgemma4.IsSizeQuantIdentity(architecture)
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
