// SPDX-Licence-Identifier: EUPL-1.2

package hip

import "dappco.re/go/inference"

func applyROCmInspectionModelProfile(inspection *inference.ModelPackInspection) {
	if inspection == nil {
		return
	}
	if inspection.Labels == nil {
		inspection.Labels = map[string]string{}
	}
	model := inspection.Model
	if model.Path == "" {
		model.Path = inspection.Path
	}
	sidecarChatTemplate := inspection.Labels["chat_template"]
	model.Labels = cloneStringMap(inspection.Labels)
	profile, ok := ResolveROCmModelProfile(inspection.Path, model)
	if !ok {
		return
	}
	labels := rocmApplyModelProfileLabels(inspection.Labels, profile)
	if profile.Family != "gemma4" && sidecarChatTemplate == "present" {
		labels["chat_template"] = sidecarChatTemplate
	}
	resolvedModel := profile.Model
	if profile.Family == "gemma4" &&
		labels["architecture_resolution_source"] == "model_type_text_tower" &&
		model.Architecture != "" {
		resolvedModel.Architecture = model.Architecture
	}
	model = resolvedModel
	model.Labels = cloneStringMap(labels)
	inspection.Model = model
	inspection.Labels = labels
	if tokenizerRoute, ok := ROCmModelTokenizerRouteForInspection(inspection); ok {
		inspection.Labels = rocmApplyROCmModelTokenizerRouteLabels(inspection.Labels, tokenizerRoute)
		inspection.Model.Labels = cloneStringMap(inspection.Labels)
	}
	applyROCmInspectionModelLoadCapability(inspection, profile)
	applyROCmInspectionEngineFeatureCapabilities(inspection, profile)
}

func applyROCmInspectionModelLoadCapability(inspection *inference.ModelPackInspection, profile ROCmModelProfile) {
	if inspection == nil || profile.Family == "gemma4" {
		return
	}
	status := profile.LoadStatus
	if status.empty() {
		status = ROCmModelLoadStatusForProfile(profile)
	}
	if status.empty() {
		return
	}
	var capability inference.Capability
	switch status.Status {
	case ROCmModelLoadStandaloneNative, ROCmModelLoadAttachedOnly:
		capability = inference.SupportedCapability(inference.CapabilityModelLoad, inference.CapabilityGroupModel)
	case ROCmModelLoadStagedNative:
		capability = inference.ExperimentalCapability(inference.CapabilityModelLoad, inference.CapabilityGroupModel, "model pack matches a staged native ROCm loader profile; standalone generation may remain pending")
	default:
		capability = inference.PlannedCapability(inference.CapabilityModelLoad, inference.CapabilityGroupModel, "model pack is recognised by the ROCm registry but native model loading is pending")
	}
	if capability.Detail == "" {
		capability.Detail = status.Reason
	}
	capability.Labels = rocmInspectionModelLoadCapabilityLabels(inspection, status)
	appendROCmInspectionCapabilityIfMissing(inspection, capability)
}

func applyROCmInspectionEngineFeatureCapabilities(inspection *inference.ModelPackInspection, profile ROCmModelProfile) {
	features := profile.EngineFeatures
	if features.empty() {
		features = ROCmEngineFeaturesForProfile(profile)
	}
	if features.ReasoningParse {
		capability := inference.SupportedCapability(inference.CapabilityReasoningParse, inference.CapabilityGroupModel)
		capability.Detail = "reasoning parser is resolved from the ROCm model registry"
		capability.Labels = rocmInspectionEngineFeatureCapabilityLabels(inspection, features)
		appendROCmInspectionCapabilityIfMissing(inspection, capability)
	}
	if features.ToolParse {
		capability := inference.SupportedCapability(inference.CapabilityToolParse, inference.CapabilityGroupModel)
		capability.Detail = "tool parser is resolved from the ROCm model registry"
		capability.Labels = rocmInspectionEngineFeatureCapabilityLabels(inspection, features)
		appendROCmInspectionCapabilityIfMissing(inspection, capability)
	}
	if features.ChatTemplate && profile.Family != "gemma4" {
		capability := inference.ExperimentalCapability(inference.CapabilityChatTemplate, inference.CapabilityGroupModel, "chat template family is resolved from the ROCm model registry")
		capability.Labels = rocmInspectionEngineFeatureCapabilityLabels(inspection, features)
		appendROCmInspectionCapabilityIfMissing(inspection, capability)
	}
	if features.Embeddings && inspection.Labels["embedding_model"] == "true" {
		capability := inference.PlannedCapability(inference.CapabilityEmbeddings, inference.CapabilityGroupModel, "embedding model-pack metadata is recognised by the ROCm model registry; native embedding kernels are pending")
		capability.Labels = rocmInspectionEngineFeatureCapabilityLabels(inspection, features)
		appendROCmInspectionCapabilityIfMissing(inspection, capability)
	}
	if features.Rerank && inspection.Labels["rerank_model"] == "true" {
		capability := inference.PlannedCapability(inference.CapabilityRerank, inference.CapabilityGroupModel, "rerank model-pack metadata is recognised by the ROCm model registry; native scorer kernels are pending")
		capability.Labels = rocmInspectionEngineFeatureCapabilityLabels(inspection, features)
		appendROCmInspectionCapabilityIfMissing(inspection, capability)
	}
}

func rocmInspectionModelLoadCapabilityLabels(inspection *inference.ModelPackInspection, status ROCmModelLoadStatus) map[string]string {
	labels := cloneStringMap(inspection.Labels)
	rocmApplyROCmModelLoadStatusLabels(labels, status)
	return labels
}

func rocmInspectionEngineFeatureCapabilityLabels(inspection *inference.ModelPackInspection, features ROCmEngineFeatures) map[string]string {
	labels := cloneStringMap(inspection.Labels)
	rocmApplyROCmEngineFeatureLabels(labels, features)
	return labels
}

func appendROCmInspectionCapabilityIfMissing(inspection *inference.ModelPackInspection, capability inference.Capability) {
	if inspection == nil || capability.ID == "" {
		return
	}
	for _, existing := range inspection.Capabilities {
		if existing.ID == capability.ID && existing.Group == capability.Group {
			return
		}
	}
	appendROCmInspectionCapability(inspection, capability)
}
