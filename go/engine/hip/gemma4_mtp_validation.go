// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func checkROCmGemma4AttachedDrafterTargetIdentity(operation string, identity inference.ModelIdentity) error {
	identity = rocmGemma4ModelWithInferredPathQuant(identity)
	labels := identity.Labels
	if rocmGemma4LabelValue(labels, "gemma4_pack_supported") == "false" {
		return core.E(operation, "target Gemma4 pack is unsupported", nil)
	}
	if labels["gemma4_size"] == "" || labels["gemma4_quant_mode"] == "" || labels["gemma4_generate_status"] == "" {
		return core.E(operation, "target Gemma4 pack identity is incomplete", nil)
	}
	if rocmGemma4LabelValue(labels, "gemma4_runnable_on_card") == "false" {
		return core.E(operation, "target Gemma4 pack is not runnable on this card", nil)
	}
	if status := labels["gemma4_generate_status"]; status != "" && status != Gemma4GenerateLinked {
		return core.E(operation, "target Gemma4 pack is not linked for generation", nil)
	}
	return nil
}

func checkROCmGemma4AttachedDrafterAssistantIdentity(operation string, identity inference.ModelIdentity) error {
	identity = rocmGemma4ModelWithInferredPathQuant(identity)
	labels := identity.Labels
	if rocmGemma4LabelValue(labels, "gemma4_pack_supported") == "false" {
		return core.E(operation, "draft Gemma4 assistant pack is unsupported", nil)
	}
	if labels["gemma4_size"] == "" || labels["gemma4_quant_mode"] == "" || labels["gemma4_generate_status"] == "" {
		return core.E(operation, "draft Gemma4 assistant pack identity is incomplete", nil)
	}
	if rocmGemma4LabelValue(labels, "gemma4_runnable_on_card") == "false" {
		return core.E(operation, "draft Gemma4 assistant pack is not runnable on this card", nil)
	}
	if labels["gemma4_size"] != "" && labels["gemma4_quant_mode"] != "" {
		if _, ok := rocmGemma4MTPAssistantQuantModeSupport(labels["gemma4_size"], labels["gemma4_quant_mode"]); !ok {
			return core.E(operation, "draft Gemma4 assistant quant mode is unsupported", nil)
		}
	}
	if status := labels["gemma4_generate_status"]; status != "" && status != Gemma4GenerateLoadOnly {
		return core.E(operation, "draft Gemma4 assistant pack must be load-only", nil)
	}
	return nil
}

func checkROCmGemma4AttachedDrafterFamilyPair(operation string, target, assistant inference.ModelIdentity) error {
	target = rocmGemma4ModelWithInferredPathQuant(target)
	assistant = rocmGemma4ModelWithInferredPathQuant(assistant)
	targetSize := target.Labels["gemma4_size"]
	assistantSize := assistant.Labels["gemma4_size"]
	if targetSize == "" || assistantSize == "" {
		return core.E(operation, "Gemma4 target and assistant family pair identity is incomplete", nil)
	}
	if targetSize != assistantSize {
		return core.E(operation, "Gemma4 target and assistant sizes must match", nil)
	}
	if !rocmGemma4AttachedDrafterFamilyPairVerified(target, assistant) {
		return core.E(operation, "Gemma4 target and assistant family pair is unsupported", nil)
	}
	return nil
}
