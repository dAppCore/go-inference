// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"dappco.re/go/inference"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
)

func rocmGemma4MTPAssistantIdentityForTarget(target inference.ModelIdentity) inference.ModelIdentity {
	target = rocmGemma4ModelWithInferredPathQuant(target)
	size := target.Labels["gemma4_size"]
	if size == "" {
		return officialGemma4E2BBF16AssistantIdentity()
	}
	assistantMode := modelgemma4.AssistantQuantMode
	assistantPath := rocmGemma4MTPAssistantPath(size, assistantMode)
	if entry, ok := modelgemma4.QATCollectionEntryForModelID(target.Path); ok && !entry.Assistant {
		assistantMode = modelgemma4.DenormalizedQuantModeForCollection(entry.QuantMode)
		assistantPath = modelgemma4.QATCollectionModelID(size, assistantMode, true)
	}
	assistantQuant := modelgemma4.ModelWithInferredQuantMode(inference.ModelIdentity{}, assistantMode)
	assistant := inference.ModelIdentity{
		Path:         assistantPath,
		Architecture: modelgemma4.AssistantArchitecture,
		VocabSize:    modelgemma4.AssistantTokenOrderingVocabSize,
		NumLayers:    modelgemma4.AssistantLayerCount,
		HiddenSize:   rocmGemma4MTPAssistantHiddenSizeForTarget(size, target.HiddenSize),
		QuantBits:    assistantQuant.QuantBits,
		QuantGroup:   assistantQuant.QuantGroup,
		QuantType:    assistantQuant.QuantType,
	}
	assistant = rocmGemma4ModelWithInferredPathQuant(assistant)
	assistant.Labels = rocmGemma4MTPAssistantLabelsForModel(size, assistantMode, assistantPath, assistant.Labels)
	return assistant
}

func rocmGemma4MTPAssistantHiddenSizeForTarget(size string, targetHidden int) int {
	return modelgemma4.MTPAssistantHiddenSizeForTarget(size, targetHidden)
}

func rocmGemma4MTPAssistantPath(size, mode string) string {
	return modelgemma4.MTPAssistantPath(size, mode)
}

func rocmGemma4MTPAssistantLabels(size string, labels map[string]string) map[string]string {
	out := modelgemma4.MTPAssistantLabels(size, labels)
	out = rocmApplyStaticGemma4ModelProfileLabels(out, officialGemma4E2BAssistantArchitecture)
	return out
}

func rocmGemma4MTPAssistantLabelsForModel(size, mode, modelID string, labels map[string]string) map[string]string {
	out := modelgemma4.MTPAssistantLabelsForModel(size, mode, modelID, labels)
	out = rocmApplyStaticGemma4ModelProfileLabels(out, officialGemma4E2BAssistantArchitecture)
	return out
}

func rocmMTPAssistantPackName(size string) string {
	return modelgemma4.MTPAssistantPackName(size)
}
