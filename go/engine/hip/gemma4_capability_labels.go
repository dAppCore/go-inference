// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func rocmGemma4Q4GenerateCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := make(map[string]string, 64)
	labels["attention_kv_backing"] = "hip_device_descriptor"
	labels["attention_kv_mode"] = rocmKVCacheModeKQ8VQ4
	labels["decode_architecture"] = "gemma4"
	labels["decode_quant"] = rocmGemma4MLXAffineQuantLabel(model.QuantBits)
	labels["gemma4_q4_device_kv_state"] = "forward_returned_device_state"
	labels["gemma4_q4_decode_kernel"] = hipKernelStatusLinked
	labels["gemma4_q4_decode_name"] = "rocm_gemma4_q4_greedy_decode_smoke"
	labels["gemma4_mlx_affine_bits"] = rocmGemma4MLXAffineBitsLabel(model.QuantBits)
	labels["gemma4_mlx_affine_decode"] = hipKernelStatusLinked
	labels["gemma4_mlx_affine_kv_state"] = "forward_returned_device_state"
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_generate"
	labels["production_decode"] = hipKernelStatusNotLinked
	labels["production_kv_cache_backing"] = hipKernelStatusNotLinked
	labels["production_prefill"] = hipKernelStatusNotLinked
	labels["prompt_modes"] = "tokens,text"
	labels["runtime_status"] = string(inference.FeatureRuntimeExperimental)
	rocmApplyGemma4SizeQuantSupportLabels(labels, model)
	rocmApplyGemma4ProductionQuantLabels(labels, model)
	rocmApplyGemma4StateContextCapabilityLabels(labels, model)
	if model.NumLayers > 0 {
		labels["decode_layers"] = rocmGemma4E2BShapeIntLabel(model.NumLayers, productionLaneGemma4E2BLayers, productionLaneGemma4E2BLayersLabel)
	}
	if model.VocabSize > 0 {
		labels["decode_vocab_size"] = rocmGemma4E2BShapeIntLabel(model.VocabSize, productionLaneGemma4E2BVocabSize, productionLaneGemma4E2BVocabSizeLabel)
	}
	if model.HiddenSize > 0 {
		labels["decode_hidden_size"] = rocmGemma4E2BShapeIntLabel(model.HiddenSize, productionLaneGemma4E2BHiddenSize, productionLaneGemma4E2BHiddenSizeLabel)
	}
	return labels
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
		rocmAddGemma4AttachedDrafterCapabilityBaseLabels(capability.Labels)
		capability.Labels["mtp_role"] = "drafter"
		capability.Labels["mtp_target_family"] = "gemma4"
	}
}

func rocmApplyGemma4StateContextCapabilityLabels(labels map[string]string, model inference.ModelIdentity) map[string]string {
	if route, ok := ROCmStateContextRouteForIdentity(model.Path, model); ok {
		return rocmApplyROCmStateContextRouteLabels(labels, route)
	}
	if route, ok := ROCmStateContextRouteForArchitecture(model.Architecture); ok {
		return rocmApplyROCmStateContextRouteLabels(labels, route)
	}
	return labels
}

func rocmApplyGemma4LoRAAdapterCapabilityLabels(labels map[string]string, model inference.ModelIdentity) map[string]string {
	if !isROCmGemma4Architecture(model.Architecture) {
		return labels
	}
	if route, ok := ROCmLoRAAdapterRouteForIdentity(model.Path, model); ok {
		return rocmApplyROCmLoRAAdapterRouteLabels(labels, route)
	}
	if route, ok := ROCmLoRAAdapterRouteForArchitecture(model.Architecture); ok {
		return rocmApplyROCmLoRAAdapterRouteLabels(labels, route)
	}
	return labels
}

func rocmApplyGemma4AttachedDrafterCapabilityLabels(labels map[string]string, model inference.ModelIdentity) map[string]string {
	if route, ok := ROCmAttachedDrafterRouteForIdentity(model.Path, model); ok {
		return rocmApplyROCmAttachedDrafterRouteLabels(labels, route)
	}
	if route, ok := ROCmAttachedDrafterRouteForArchitecture(model.Architecture); ok {
		return rocmApplyROCmAttachedDrafterRouteLabels(labels, route)
	}
	return labels
}

func rocmApplyGemma4StateArtifactLabels(labels map[string]string, model inference.ModelIdentity) map[string]string {
	if !isROCmGemma4Architecture(model.Architecture) && !isROCmGemma4AssistantArchitecture(model.Architecture) {
		return labels
	}
	rocmApplyGemma4SizeQuantSupportLabels(labels, model)
	rocmApplyGemma4ProductionQuantLabels(labels, model)
	labels = rocmApplyGemma4StateContextCapabilityLabels(labels, model)
	labels = rocmApplyGemma4LoRAAdapterCapabilityLabels(labels, model)
	labels = rocmApplyGemma4AttachedDrafterCapabilityLabels(labels, model)
	return labels
}

func rocmGemma4E2BShapeIntLabel(value, productionValue int, productionLabel string) string {
	if value == productionValue {
		return productionLabel
	}
	return core.Sprintf("%d", value)
}

func rocmGemma4MLXAffineQuantLabel(bits int) string {
	switch hipMLXQ4ProjectionBitsOrDefault(bits) {
	case 4:
		return "mlx_q4"
	case 6:
		return "mlx_q6"
	case 8:
		return "mlx_q8"
	default:
		return core.Sprintf("mlx_q%d", hipMLXQ4ProjectionBitsOrDefault(bits))
	}
}

func rocmGemma4MLXAffineBitsLabel(bits int) string {
	switch hipMLXQ4ProjectionBitsOrDefault(bits) {
	case 4:
		return "4"
	case 6:
		return "6"
	case 8:
		return "8"
	default:
		return core.Sprintf("%d", hipMLXQ4ProjectionBitsOrDefault(bits))
	}
}

func rocmGemma4Q4BatchGenerateCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := rocmGemma4Q4GenerateCapabilityLabels(model)
	labels["batch_generate_kernel"] = hipKernelStatusLinked
	labels["batch_generate_name"] = "rocm_gemma4_q4_batch_generate_experimental"
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_batch_generate"
	return labels
}

func rocmGemma4Q4ChatCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := rocmGemma4Q4GenerateCapabilityLabels(model)
	labels["chat_kernel"] = hipKernelStatusLinked
	labels["chat_name"] = "rocm_gemma4_q4_chat_generate_experimental"
	labels["chat_template"] = "gemma4_hf_turn"
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_chat"
	return labels
}

func rocmGemma4Q4EvaluationCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := rocmGemma4Q4GenerateCapabilityLabels(model)
	labels["eval_loss_logits_source"] = "gemma4_mlx_affine_package_prefill"
	labels["eval_prefill_kernel"] = hipKernelStatusLinked
	labels["eval_prefill_name"] = "rocm_gemma4_q4_package_prefill_experimental"
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_eval"
	labels["production_prefill"] = hipKernelStatusNotLinked
	return labels
}

func rocmGemma4Q4BenchmarkCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := rocmGemma4Q4GenerateCapabilityLabels(model)
	rocmAddGemma4AttachedDrafterCapabilityLabels(labels, model)
	labels = rocmApplyGemma4StateArtifactLabels(labels, model)
	labels["benchmark_kernel"] = hipKernelStatusLinked
	labels["benchmark_name"] = "rocm_gemma4_q4_benchmark_experimental"
	labels["benchmark_prompt_mode"] = "explicit_text"
	labels["benchmark_retained_state_book"] = "BenchmarkInferenceGemma4Q4Book10Turn_RetainedState"
	labels["benchmark_replay_baseline"] = "BenchmarkInferenceGemma4Q4Book10Turn_ReplayBaseline"
	labels["benchmark_retained_state_required"] = "true"
	labels["benchmark_prompt_replay_fallback"] = "forbidden"
	labels["benchmark_state_source"] = "rocm_state_session_runtime_kv"
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_benchmark"
	labels["production_book_policy"] = "retained_state_required"
	labels["production_book_decision_source"] = "benchmark_metrics"
	labels["production_book_gate_wall_seconds"] = productionLaneBookWallSecondsLabel
	labels["production_book_gate_turns"] = productionLaneBookTurnCountLabel
	labels["production_book_gate_raw_decode_tokens_per_sec"] = productionLaneRetainedVisibleTokensSecLabel
	labels["production_book_gate_metrics"] = productionBookGateMetricsLabel
	labels["production_book_gate_reason_codes"] = productionBookGateReasonCodesLabel
	labels["production_book_retained_route_metrics"] = productionBookRetainedRouteMetricsLabel
	labels["production_book_retained_artifact_labels"] = productionBookRetainedArtifactLabelsLabel
	labels["production_book_long_output_quality_flags"] = "0"
	labels["production_book_required_metrics"] = productionQuantizationRequiredMetricsLabel
	labels["production_model_source"] = "model_identity_or_pack"
	labels["production_mtp_required_metrics"] = strings.Join(defaultProductionMTPRequiredMetrics, ",")
	labels["production_quant_decision_source"] = "gemma4_family_matrix"
	return labels
}

func rocmGemma4Q4ClassifyCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := rocmGemma4Q4GenerateCapabilityLabels(model)
	labels["classify_kernel"] = hipKernelStatusLinked
	labels["classify_name"] = "rocm_gemma4_q4_classify_experimental"
	labels["classify_logits_source"] = "gemma4_mlx_affine_package_prefill"
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_classify"
	labels["production_prefill"] = hipKernelStatusNotLinked
	return labels
}

func rocmGemma4Q4LogitProbeCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := rocmGemma4Q4ClassifyCapabilityLabels(model)
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_logit_probe"
	labels["logit_probe_kernel"] = hipKernelStatusLinked
	labels["logit_probe_affine_source"] = "gemma4_mlx_affine_classify_logits"
	labels["logit_probe_source"] = "gemma4_q4_classify_logits"
	return labels
}

func rocmGemma4Q4SpeculativeDecodeCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := rocmGemma4Q4GenerateCapabilityLabels(model)
	rocmAddGemma4AttachedDrafterCapabilityLabels(labels, model)
	labels = rocmApplyGemma4AttachedDrafterCapabilityLabels(labels, model)
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_speculative_decode"
	labels["speculative_decode_affine_source"] = "gemma4_mlx_affine_generate"
	labels["speculative_decode_helper"] = hipKernelStatusLinked
	labels["speculative_decode_source"] = "gemma4_q4_generate"
	return labels
}

func rocmGemma4Q4PromptLookupDecodeCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := rocmGemma4Q4GenerateCapabilityLabels(model)
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_prompt_lookup_decode"
	labels["prompt_lookup_decode_affine_source"] = "gemma4_mlx_affine_generate"
	labels["prompt_lookup_decode_helper"] = hipKernelStatusLinked
	labels["prompt_lookup_decode_source"] = "gemma4_q4_generate"
	return labels
}
