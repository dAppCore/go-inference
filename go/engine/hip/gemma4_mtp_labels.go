// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
)

func rocmAddGemma4AttachedDrafterModelLabels(labels map[string]string, prefix string, identity inference.ModelIdentity) {
	if labels == nil || prefix == "" {
		return
	}
	identity = rocmGemma4ModelWithInferredPathQuant(identity)
	rocmAddGemma4AttachedDrafterRegistryLabels(labels, prefix, "_", identity)
	rocmAddGemma4AttachedDrafterProductionQuantLabels(labels, prefix, "_", identity)
	size := identity.Labels["gemma4_size"]
	mode := identity.Labels["gemma4_quant_mode"]
	status := identity.Labels["gemma4_generate_status"]
	runtime := identity.Labels["gemma4_runtime"]
	supported := identity.Labels["gemma4_pack_supported"]
	runnable := identity.Labels["gemma4_runnable_on_card"]
	if identity.QuantGroup > 0 {
		labels[prefix+"_gemma4_quant_group"] = core.Sprintf("%d", identity.QuantGroup)
	}
	if size != "" {
		labels[prefix+"_gemma4_size"] = size
	}
	if mode != "" {
		labels[prefix+"_gemma4_quant_mode"] = mode
	}
	if status != "" {
		labels[prefix+"_gemma4_generate_status"] = status
	}
	if runtime != "" {
		labels[prefix+"_gemma4_runtime"] = runtime
	}
	if supported != "" {
		labels[prefix+"_gemma4_pack_supported"] = supported
	}
	if runnable != "" {
		labels[prefix+"_gemma4_runnable_on_card"] = runnable
	}
}

func rocmAddGemma4AttachedDrafterDottedModelLabels(labels map[string]string, prefix string, identity inference.ModelIdentity) {
	if labels == nil || prefix == "" {
		return
	}
	identity = rocmGemma4ModelWithInferredPathQuant(identity)
	rocmAddGemma4AttachedDrafterRegistryLabels(labels, prefix, ".", identity)
	rocmAddGemma4AttachedDrafterProductionQuantLabels(labels, prefix, ".", identity)
	size := identity.Labels["gemma4_size"]
	mode := identity.Labels["gemma4_quant_mode"]
	status := identity.Labels["gemma4_generate_status"]
	runtime := identity.Labels["gemma4_runtime"]
	supported := identity.Labels["gemma4_pack_supported"]
	runnable := identity.Labels["gemma4_runnable_on_card"]
	if identity.QuantGroup > 0 {
		labels[prefix+".gemma4_quant_group"] = core.Sprintf("%d", identity.QuantGroup)
	}
	if size != "" {
		labels[prefix+".gemma4_size"] = size
	}
	if mode != "" {
		labels[prefix+".gemma4_quant_mode"] = mode
	}
	if status != "" {
		labels[prefix+".gemma4_generate_status"] = status
	}
	if runtime != "" {
		labels[prefix+".gemma4_runtime"] = runtime
	}
	if supported != "" {
		labels[prefix+".gemma4_pack_supported"] = supported
	}
	if runnable != "" {
		labels[prefix+".gemma4_runnable_on_card"] = runnable
	}
}

func rocmAddGemma4AttachedDrafterRegistryLabels(labels map[string]string, prefix, separator string, identity inference.ModelIdentity) {
	if labels == nil || prefix == "" || !rocmIsGemma4SizeQuantIdentity(identity.Architecture) {
		return
	}
	profileLabels := map[string]string{}
	rocmApplyResolvedModelProfileLabels(profileLabels, identity.Path, identity)
	for _, key := range rocmGemma4AttachedDrafterRegistryLabelKeys {
		if value := profileLabels[key]; value != "" {
			labels[prefix+separator+key] = value
		}
	}
}

var rocmGemma4AttachedDrafterRegistryLabelKeys = []string{
	"engine_registry",
	"engine_profile",
	"engine_profile_family",
	"engine_profile_source",
	"engine_profile_matched",
	"engine_profile_reactive",
	"engine_profile_architecture",
	"engine_architecture_profile",
	"engine_architecture_family",
	"engine_architecture_native_runtime",
	"engine_architecture_generation",
	"engine_architecture_chat",
	"engine_architecture_runtime_status",
	"engine_architecture_reasoning_parser",
	"engine_architecture_tool_parser",
	"engine_architecture_embeddings",
	"engine_architecture_rerank",
	"engine_architecture_moe",
	"engine_architecture_attached_only",
	"engine_architecture_quantization_hints",
	"engine_architecture_cache_hints",
	"engine_architecture_notes",
	"engine_architecture_aliases",
	"engine_text_tower",
	"engine_generation_role",
	"engine_default_thinking",
	"engine_requires_chat_template",
	"engine_chat_template",
	"engine_model_context_window",
	"engine_text_generate",
	"engine_mlx_affine_decode",
	"engine_device_kv_state",
	"engine_fixed_sliding_cache",
	"engine_fixed_sliding_cache_bound",
	"engine_weight_policy",
	"engine_weight_policy_source",
	"engine_weight_wrapper_prefixes",
	"engine_weight_skip_prefixes",
	"engine_weight_skip_substrings",
	"engine_weight_model_prefixes",
	"engine_lora_policy",
	"engine_lora_policy_source",
	"engine_lora_target_family",
	"engine_lora_targets",
	"engine_lora_default_targets",
	"engine_lora_safe_targets",
	"engine_lora_extended_targets",
	"engine_lora_extended_targets_require_opt_in",
	"gemma4_weight_policy",
	"gemma4_weight_wrapper_prefixes",
	"gemma4_weight_skip_prefixes",
	"gemma4_weight_skip_substrings",
	"gemma4_weight_model_prefixes",
	"gemma4_lora_policy",
	"gemma4_lora_targets",
	"gemma4_lora_default_targets",
	"gemma4_lora_safe_targets",
	"gemma4_lora_extended_targets",
	"gemma4_lora_extended_targets_require_opt_in",
	"chat_template",
}

func rocmAddGemma4AttachedDrafterProductionQuantLabels(labels map[string]string, prefix, separator string, identity inference.ModelIdentity) {
	if labels == nil || prefix == "" || !rocmIsGemma4SizeQuantIdentity(identity.Architecture) {
		return
	}
	quantLabels := map[string]string{}
	rocmApplyGemma4ProductionQuantLabels(quantLabels, identity)
	for source, suffix := range map[string]string{
		"production_quant_collection":      "production_quant_collection",
		"production_quant_assistant_model": "production_quant_assistant_model",
		"production_quant_locked_model":    "production_quant_locked_model",
		"production_quant_model":           "production_quant_model",
		"production_quant_mtp_assistant":   "production_quant_mtp_assistant",
		"production_quant_pack":            "production_quant_pack",
		"production_quant_target_family":   "production_quant_target_family",
		"production_quant_tier":            "production_quant_tier",
	} {
		if value := quantLabels[source]; value != "" {
			key := prefix + separator + suffix
			if labels[key] == "" {
				labels[key] = value
			}
		}
	}
}

func rocmApplyGemma4AttachedDrafterOfficialPairVerification(labels map[string]string, target, assistant inference.ModelIdentity, dotted bool) {
	target = rocmGemma4ModelWithInferredPathQuant(target)
	assistant = rocmGemma4ModelWithInferredPathQuant(assistant)
	modelgemma4.ApplyPairVerificationLabels(labels, target, assistant, dotted)
}

func rocmAddGemma4AttachedDrafterIdentityLabel(labels map[string]string, prefix string, identity inference.ModelIdentity) {
	if labels == nil || prefix == "" || identity.Path == "" {
		return
	}
	labels[prefix+"_model_id"] = identity.Path
}

func rocmAddGemma4AttachedDrafterDottedIdentityLabel(labels map[string]string, prefix string, identity inference.ModelIdentity) {
	if labels == nil || prefix == "" || identity.Path == "" {
		return
	}
	labels[prefix+".model_id"] = identity.Path
}

func rocmAddGemma4AttachedDrafterOfficialLockLabels(labels map[string]string, target, assistant inference.ModelIdentity, dotted bool) {
	target = rocmGemma4ModelWithInferredPathQuant(target)
	assistant = rocmGemma4ModelWithInferredPathQuant(assistant)
	modelgemma4.ApplyOfficialPairLockLabels(labels, target, assistant, dotted)
}

func rocmGemma4AttachedDrafterOfficialPairVerified(target, assistant inference.ModelIdentity) bool {
	target = rocmGemma4ModelWithInferredPathQuant(target)
	assistant = rocmGemma4ModelWithInferredPathQuant(assistant)
	return modelgemma4.OfficialPairVerified(target, assistant)
}

func rocmGemma4AttachedDrafterFamilyPairVerified(target, assistant inference.ModelIdentity) bool {
	target = rocmGemma4ModelWithInferredPathQuant(target)
	assistant = rocmGemma4ModelWithInferredPathQuant(assistant)
	return modelgemma4.FamilyPairVerified(target, assistant)
}

func rocmAddGemma4AttachedDrafterBenchmarkBaseLabels(labels map[string]string) {
	if labels == nil {
		return
	}
	labels["attached.drafter.decode"] = "experimental"
	labels["attached.drafter.native_attachment"] = hipKernelStatusNotLinked
	labels["attached.drafter.native_handoff"] = attachedDrafterNativeHandoffTargetDecodeOnly
	labels["attached.drafter.role"] = "gemma4_assistant"
	labels["attached.drafter.source"] = "gemma4_mlx_affine_generate"
	labels["attached.drafter.retained_state_entrypoint"] = hipKernelStatusLinked
	labels["attached.drafter.retained_state_required"] = "true"
	labels["attached.drafter.state_source"] = "rocm_state_session_runtime_kv"
	labels["attached.drafter.prompt_replay_fallback"] = "forbidden"
	labels["attached.drafter.target_retained_decode"] = hipKernelStatusLinked
	labels["attached.drafter.target_retained_state_decode"] = hipKernelStatusLinked
	labels["attached.drafter.assistant_verify"] = hipKernelStatusNotLinked
	labels["attached.drafter.assistant_state_verify"] = hipKernelStatusNotLinked
	labels["attached.drafter.assistant_architecture"] = officialGemma4E2BAssistantArchitecture
	labels["attached.drafter.assistant_centroid_intermediate_top_k"] = productionMTPAssistantCentroidIntermediateTopKLabel
	labels["attached.drafter.assistant_centroids"] = productionMTPAssistantOrderedEmbeddingCentroidsLabel
	labels["attached.drafter.assistant_four_layer_drafter"] = "true"
	labels["attached.drafter.assistant_ordered_embeddings"] = "true"
	labels["attached.drafter.assistant_token_ordering_dtype"] = "int64"
	labels["attached.drafter.assistant_token_ordering_shape"] = productionMTPAssistantTokenOrderingShapeLabel
	labels["attached.drafter.speculative_draft_tokens"] = productionMTPDefaultDraftTokensLabel
}

func rocmAddGemma4AttachedDrafterBenchmarkLabels(labels map[string]string, identities ...inference.ModelIdentity) {
	if labels == nil {
		return
	}
	target := officialGemma4E2BQ6TargetIdentity()
	if len(identities) > 0 && !modelIdentityIsZero(identities[0]) {
		target = identities[0]
	}
	assistant := rocmGemma4MTPAssistantIdentityForTarget(target)
	if len(identities) > 1 && !modelIdentityIsZero(identities[1]) {
		assistant = identities[1]
	}
	rocmAddGemma4AttachedDrafterBenchmarkBaseLabels(labels)
	rocmAddGemma4AttachedDrafterDottedIdentityLabel(labels, "attached.drafter.target", target)
	rocmAddGemma4AttachedDrafterDottedIdentityLabel(labels, "attached.drafter.assistant", assistant)
	rocmAddGemma4AttachedDrafterDottedModelLabels(labels, "attached.drafter.target", target)
	rocmAddGemma4AttachedDrafterDottedModelLabels(labels, "attached.drafter.assistant", assistant)
	rocmAddGemma4AttachedDrafterOfficialLockLabels(labels, target, assistant, true)
}

func rocmAddGemma4AttachedDrafterCapabilityBaseLabels(labels map[string]string) {
	if labels == nil {
		return
	}
	setDefault := func(key, value string) {
		if labels[key] == "" {
			labels[key] = value
		}
	}
	setDefault("attached_drafter_helper", hipKernelStatusLinked)
	setDefault("attached_drafter_native_attachment", hipKernelStatusNotLinked)
	setDefault("attached_drafter_native_handoff", attachedDrafterNativeHandoffTargetDecodeOnly)
	setDefault("attached_drafter_role", "gemma4_assistant")
	setDefault("attached_drafter_source", "gemma4_mlx_affine_generate")
	setDefault("attached_drafter_retained_state_entrypoint", hipKernelStatusLinked)
	setDefault("attached_drafter_retained_state_required", "true")
	setDefault("attached_drafter_state_source", "rocm_state_session_runtime_kv")
	setDefault("attached_drafter_prompt_replay_fallback", "forbidden")
	setDefault("attached_drafter_target_retained_decode", hipKernelStatusLinked)
	setDefault("attached_drafter_target_retained_state_decode", hipKernelStatusLinked)
	setDefault("attached_drafter_assistant_verify", hipKernelStatusNotLinked)
	setDefault("attached_drafter_assistant_state_verify", hipKernelStatusNotLinked)
	setDefault("attached_drafter_assistant_architecture", officialGemma4E2BAssistantArchitecture)
	setDefault("attached_drafter_assistant_centroid_intermediate_top_k", productionMTPAssistantCentroidIntermediateTopKLabel)
	setDefault("attached_drafter_assistant_centroids", productionMTPAssistantOrderedEmbeddingCentroidsLabel)
	setDefault("attached_drafter_assistant_four_layer_drafter", "true")
	setDefault("attached_drafter_assistant_ordered_embeddings", "true")
	setDefault("attached_drafter_assistant_token_ordering_dtype", "int64")
	setDefault("attached_drafter_assistant_token_ordering_shape", productionMTPAssistantTokenOrderingShapeLabel)
	setDefault("attached_drafter_speculative_draft_tokens", productionMTPDefaultDraftTokensLabel)
}

func rocmAddGemma4AttachedDrafterCapabilityLabels(labels map[string]string, identities ...inference.ModelIdentity) {
	if labels == nil {
		return
	}
	target := officialGemma4E2BQ6TargetIdentity()
	if len(identities) > 0 && !modelIdentityIsZero(identities[0]) {
		target = identities[0]
	}
	assistant := rocmGemma4MTPAssistantIdentityForTarget(target)
	if len(identities) > 1 && !modelIdentityIsZero(identities[1]) {
		assistant = identities[1]
	}
	rocmAddGemma4AttachedDrafterCapabilityBaseLabels(labels)
	rocmAddGemma4AttachedDrafterIdentityLabel(labels, "attached_drafter_target", target)
	rocmAddGemma4AttachedDrafterIdentityLabel(labels, "attached_drafter_assistant", assistant)
	rocmAddGemma4AttachedDrafterModelLabels(labels, "attached_drafter_target", target)
	rocmAddGemma4AttachedDrafterModelLabels(labels, "attached_drafter_assistant", assistant)
	rocmAddGemma4AttachedDrafterOfficialLockLabels(labels, target, assistant, false)
}
