// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"errors"
	"iter"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

func TestNativeContract_RocmBackendImplementsSharedPlanner_Good(t *testing.T) {
	var _ inference.ModelFitPlanner = (*rocmBackend)(nil)
	var _ inference.CapabilityReporter = (*rocmBackend)(nil)
	var _ inference.ModelPackInspector = (*rocmBackend)(nil)
}

func TestNativeContract_RocmModelImplementsSharedContracts_Good(t *testing.T) {
	var _ inference.TokenizerModel = (*rocmModel)(nil)
	var _ inference.AdapterModel = (*rocmModel)(nil)
	var _ inference.EmbeddingModel = (*rocmModel)(nil)
	var _ inference.RerankModel = (*rocmModel)(nil)
	var _ inference.ProbeableModel = (*rocmModel)(nil)
	var _ inference.BenchableModel = (*rocmModel)(nil)
	var _ inference.Evaluator = (*rocmModel)(nil)
	var _ inference.CapabilityReporter = (*rocmModel)(nil)
	var _ ROCmModelIdentityReporter = (*rocmModel)(nil)
	var _ ROCmModelProfileReporter = (*rocmModel)(nil)
	var _ ROCmModelRoutePlanReporter = (*rocmModel)(nil)
}

func TestNativeContract_RocmModelReactiveRegistryReporters_Good(t *testing.T) {
	model := &rocmModel{
		native: &hipLoadedModel{
			contextSize: 8192,
			modelLabels: map[string]string{
				"runtime_label": "loaded",
			},
		},
		modelPath: "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			VocabSize:    262144,
			NumLayers:    26,
			HiddenSize:   2304,
			QuantBits:    6,
			QuantGroup:   64,
		},
		modelLabels: map[string]string{
			"gemma4_size":       "E4B",
			"gemma4_quant_mode": "q6",
			"model_label":       "base",
		},
		engineProfile: ROCmModelProfile{
			Name:     "gemma4",
			Family:   "gemma4",
			Registry: rocmModelRegistryName,
			EngineFeatures: ROCmEngineFeatures{
				Contract:     rocmEngineFeaturesContract,
				Capabilities: []inference.CapabilityID{inference.CapabilityGenerate},
				Labels:       map[string]string{"engine_feature_text_generate": "true"},
			},
			Gemma4EngineFeatures: Gemma4EngineFeatures{
				ModelContextWindow: true,
				TextGenerate:       true,
				MLXAffineDecode:    true,
				DeviceKVState:      true,
			},
			Labels: map[string]string{"engine_profile": "gemma4"},
		},
	}

	_, err := model.WarmCache(context.Background(), inference.CacheWarmRequest{
		Mode:   rocmKVCacheModeQ8,
		Tokens: []int32{1, 2, 3},
	})
	core.RequireNoError(t, err)

	identity := model.ModelIdentity()
	if identity.Path != model.modelPath ||
		identity.Architecture != "gemma4_text" ||
		identity.ContextLength != 8192 ||
		identity.QuantType != "q6" ||
		identity.Labels["model_label"] != "base" ||
		identity.Labels["runtime_label"] != "loaded" {
		t.Fatalf("ModelIdentity = %+v, want loaded context and merged model labels", identity)
	}
	identity.Labels["model_label"] = "mutated"
	if next := model.ModelIdentity(); next.Labels["model_label"] == "mutated" {
		t.Fatalf("ModelIdentity returned aliased labels: %+v", next.Labels)
	}

	profile := model.ModelProfile()
	if !profile.Matched() ||
		profile.Model.ContextLength != 8192 ||
		profile.Model.Labels["runtime_label"] != "loaded" ||
		!profile.Gemma4EngineFeatures.GenerateLinked() ||
		!slices.Contains(profile.EngineFeatures.Capabilities, inference.CapabilityGenerate) {
		t.Fatalf("ModelProfile = %+v, want loaded reactive registry profile", profile)
	}
	profile.Model.Labels["runtime_label"] = "mutated"
	profile.EngineFeatures.Capabilities[0] = inference.CapabilityChat
	profile.EngineFeatures.Labels["engine_feature_text_generate"] = "mutated"
	profile.Labels["engine_profile"] = "mutated"
	nextProfile := model.ModelProfile()
	if nextProfile.Model.Labels["runtime_label"] == "mutated" ||
		nextProfile.EngineFeatures.Capabilities[0] == inference.CapabilityChat ||
		nextProfile.EngineFeatures.Labels["engine_feature_text_generate"] == "mutated" ||
		nextProfile.Labels["engine_profile"] == "mutated" {
		t.Fatalf("ModelProfile returned aliased profile data: %+v", nextProfile)
	}

	plan := model.ModelRoutePlan()
	if !plan.Matched() ||
		plan.Contract != ROCmModelRoutePlanContract ||
		plan.Architecture != "gemma4_text" ||
		plan.Model.ContextLength != 8192 ||
		plan.Model.Labels["runtime_label"] != "loaded" ||
		!plan.FeatureRoute.Matched() ||
		plan.Labels["engine_route_plan_contract"] != ROCmModelRoutePlanContract ||
		plan.Labels["engine_route_plan_cache_profile"] != "true" ||
		plan.Labels["engine_route_plan_cache_profile_contract"] != rocmmodel.CacheProfileContract ||
		plan.Labels["engine_route_plan_cache_profile_max_cache_tokens"] != "3" ||
		plan.CacheProfile.MaxCacheTokens != 3 ||
		plan.Labels["engine_route_plan_feature"] != "true" {
		t.Fatalf("ModelRoutePlan = %+v, want loaded model-owned route plan with live cache profile", plan)
	}
	plan.Model.Labels["runtime_label"] = "mutated"
	plan.FeatureRoute.Labels["engine_feature_route_contract"] = "mutated"
	nextPlan := model.ModelRoutePlan()
	if nextPlan.Model.Labels["runtime_label"] == "mutated" ||
		nextPlan.FeatureRoute.Labels["engine_feature_route_contract"] == "mutated" {
		t.Fatalf("ModelRoutePlan returned aliased route data: %+v", nextPlan)
	}
	resolvedPlan, ok := ROCmModelRoutePlanForModel(model)
	if !ok || !resolvedPlan.Matched() || resolvedPlan.Architecture != "gemma4_text" {
		t.Fatalf("ROCmModelRoutePlanForModel = %+v ok=%v, want loaded model route plan", resolvedPlan, ok)
	}
	report := model.Capabilities()
	if report.Labels["engine_route_plan_contract"] != ROCmModelRoutePlanContract ||
		report.Labels["engine_route_plan_cache_profile"] != "true" ||
		report.Labels["engine_route_plan_cache_profile_max_cache_tokens"] != "3" ||
		report.Labels["engine_route_plan_feature"] != "true" {
		t.Fatalf("Capabilities labels = %+v, want live route-plan labels", report.Labels)
	}
}

func TestNativeContract_RocmBackendCapabilities_Good(t *testing.T) {
	runtime := &fakeNativeRuntime{
		available: true,
		device:    nativeDeviceInfo{Name: "gfx1100", MemoryBytes: 16 * memoryGiB, FreeBytes: 8 * memoryGiB, Driver: "hip-test"},
	}

	report := newROCmBackendWithRuntime(runtime).Capabilities()

	if report.Runtime.Backend != "rocm" || !report.Runtime.NativeRuntime || report.Runtime.Device != "gfx1100" {
		t.Fatalf("runtime = %+v, want native ROCm device", report.Runtime)
	}
	if !report.Available {
		t.Fatalf("Available = false, want true")
	}
	if report.Labels["runtime_status"] != "available" ||
		report.Labels["kernel_status"] != hipKernelStatusNotLinked ||
		report.Labels["cross_entropy_kernel"] != hipKernelStatusNotLinked ||
		report.Labels["decode_kernel"] != hipKernelStatusNotLinked ||
		report.Labels["distillation_kernel"] != hipKernelStatusNotLinked ||
		report.Labels["grpo_kernel"] != hipKernelStatusNotLinked ||
		report.Labels["prefill_kernel"] != hipKernelStatusNotLinked ||
		report.Labels["projection_kernel"] != hipKernelStatusNotLinked {
		t.Fatalf("labels = %+v, want runtime and kernel status labels", report.Labels)
	}
	if !report.Supports(inference.CapabilityModelLoad) || !report.Supports(inference.CapabilityModelFit) {
		t.Fatalf("capabilities = %+v, want load and fit planning", report.CapabilityIDs())
	}
	if report.Supports(inference.CapabilityGenerate) {
		t.Fatalf("generate should be planned until native decode kernels are linked: %+v", report.CapabilityIDs())
	}
	if !report.Supports(inference.CapabilityTokenizer) || !report.Supports(inference.CapabilityProbeEvents) {
		t.Fatalf("capabilities = %+v, want fallback tokenizer and probe stream", report.CapabilityIDs())
	}
	if cap, ok := report.Capability(inference.CapabilityQuantization); !ok || cap.Status != inference.CapabilityStatusExperimental ||
		cap.Labels["kv_compression"] != rocmTurboQuantKVMode ||
		cap.Labels["kv_compression_bits"] != "3.5" ||
		cap.Labels["kv_compression_default"] != "true" ||
		cap.Labels["kv_compression_group_size"] != rocmTurboQuantKVDefaultGroupLabel ||
		cap.Labels["kv_compression_runtime"] != "cpu_reference" ||
		cap.Labels["autoround_algorithms"] != productionAutoRoundAlgorithmsLabel ||
		cap.Labels["autoround_formats"] != productionAutoRoundFormatsLabel ||
		cap.Labels["autoround_weight_schemes"] != productionAutoRoundSchemesLabel ||
		cap.Labels["autoround_float_formats"] != productionAutoRoundFloatFormatsLabel ||
		cap.Labels["autoround_group_sizes"] != productionAutoRoundGroupSizesLabel ||
		cap.Labels["autoround_profiles"] != productionAutoRoundProfilesLabel ||
		cap.Labels["autoround_calibration_evidence_helper"] != "ApplyProductionAutoRoundCalibrationLabelEvidence" ||
		cap.Labels["autoround_calibration_decision_helper"] != "EvaluateProductionAutoRoundCalibrationEvidence" ||
		cap.Labels["autoround_calibration_decision_labels"] != productionAutoRoundCalibrationDecisionLabelsLabel ||
		cap.Labels["autoround_calibration_decision_label_evidence_helper"] != "ApplyProductionAutoRoundCalibrationDecisionLabelEvidence" ||
		cap.Labels["autoround_calibration_decision_label_evaluator"] != "EvaluateProductionAutoRoundCalibrationDecisionLabels" ||
		cap.Labels["autoround_calibration_decision_validator"] != "ValidateProductionAutoRoundCalibrationDecisionLabels" ||
		cap.Labels["autoround_calibration_evidence_decision_label_helper"] != "ApplyProductionAutoRoundCalibrationEvidenceDecisionLabels" ||
		cap.Labels["autoround_calibration_evidence_decision_validator"] != "ValidateProductionAutoRoundCalibrationEvidenceDecisionLabels" ||
		cap.Labels["autoround_calibration_labels"] != productionAutoRoundCalibrationLabelsLabel ||
		cap.Labels["autoround_calibration_knobs"] != "nsamples,seqlen,iters" ||
		cap.Labels["autoround_calibration_validator"] != "ValidateProductionAutoRoundCalibrationLabels" ||
		cap.Labels["autoround_runtime"] != "planned_hip" ||
		cap.Labels["autoround_hip_kernel"] != hipKernelStatusNotLinked ||
		cap.Labels["production_candidate_gate"] != "linked" ||
		cap.Labels["production_explicit_opt_in_required"] != "false" ||
		cap.Labels["production_fast_lane_default"] != "true" ||
		cap.Labels["production_requires_cli_flag"] != "false" ||
		cap.Labels["production_requires_env_gate"] != "false" ||
		cap.Labels["production_hip_integration"] != hipKernelStatusNotLinked {
		t.Fatalf("quantization capability = %+v ok=%v, want production TurboQuant KV fast-lane labels", cap, ok)
	}
	cap, ok := report.Capability(inference.CapabilityQuantization)
	if !ok ||
		cap.Labels["production_required_layout_version"] != ProductionTurboQuantKVLayoutVersion ||
		cap.Labels["production_required_key_algorithm"] != ProductionTurboQuantKeyAlgorithm ||
		cap.Labels["production_required_value_algorithm"] != ProductionTurboQuantValueAlgorithm ||
		cap.Labels["production_required_outlier_policy"] != ProductionTurboQuantOutlierPolicy ||
		cap.Labels["production_combined_gate"] != ProductionCombinedMTPAndTurboQuantMode ||
		!strings.Contains(cap.Labels["production_compare_cache_modes"], rocmKVCacheModeKQ8VQ4) {
		t.Fatalf("quantization capability = %+v ok=%v, want TurboQuant production gate evidence labels", cap, ok)
	}
	assertCSVLabelContainsAll(t, "production_required_metrics", cap.Labels["production_required_metrics"], defaultProductionTurboQuantRequiredMetrics)
	assertCSVLabelContainsAll(t, "production_combined_required_metrics", cap.Labels["production_combined_required_metrics"], defaultProductionCombinedMTPAndTurboQuantRequiredMetrics)
	if !stringSliceContains(report.CacheModes, rocmTurboQuantKVMode) {
		t.Fatalf("cache modes = %+v, want research TurboQuant KV mode advertised", report.CacheModes)
	}
	metadataFixtures := nativeContractMetadataFixtureKernels()
	for _, id := range []inference.CapabilityID{inference.CapabilityMoERouting, inference.CapabilityMoELazyExperts, inference.CapabilityJANGTQ, inference.CapabilityCodebookVQ} {
		if cap, ok := report.Capability(id); !ok || cap.Status != inference.CapabilityStatusExperimental ||
			cap.Labels["runtime_status"] != string(inference.FeatureRuntimeExperimental) ||
			cap.Labels["fixture_kernel"] != hipKernelStatusLinked ||
			cap.Labels["fixture_kernel_name"] != metadataFixtures[id] ||
			cap.Labels["required_integration"] == "" ||
			cap.Labels["production_integration"] != "pending" {
			t.Fatalf("fixture capability %s = %+v ok=%v, want linked fixture kernel with production pending", id, cap, ok)
		}
	}
	if cap, ok := report.Capability(inference.CapabilityScheduler); !ok || cap.Status != inference.CapabilityStatusSupported {
		t.Fatalf("scheduler capability = %+v ok=%v, want supported scheduler wrapper", cap, ok)
	}
	if cap, ok := report.Capability(inference.CapabilityRequestCancel); !ok || cap.Status != inference.CapabilityStatusSupported {
		t.Fatalf("request cancel capability = %+v ok=%v, want supported scheduler cancellation", cap, ok)
	}
	if cap, ok := report.Capability(inference.CapabilityReasoningParse); !ok || cap.Status != inference.CapabilityStatusSupported {
		t.Fatalf("reasoning parser capability = %+v ok=%v, want supported parser registry", cap, ok)
	}
	if cap, ok := report.Capability(inference.CapabilityToolParse); !ok || cap.Status != inference.CapabilityStatusSupported {
		t.Fatalf("tool parser capability = %+v ok=%v, want supported parser registry", cap, ok)
	}
	if cap, ok := report.Capability(inference.CapabilityCacheBlocks); !ok || cap.Status != inference.CapabilityStatusExperimental ||
		!strings.Contains(cap.Detail, "HIP device remirror") ||
		cap.Labels["kv_device_backing"] != "best_effort_remirror" ||
		cap.Labels["fully_hip_owned"] != "pending" {
		t.Fatalf("cache blocks capability = %+v ok=%v, want experimental cache with device-remirror labels", cap, ok)
	}
	if cap, ok := report.Capability(inference.CapabilityCacheWarm); !ok || cap.Status != inference.CapabilityStatusExperimental ||
		!strings.Contains(cap.Detail, "optional HIP device remirror") ||
		cap.Labels["native_prefill_reuse"] != "pending" ||
		cap.Labels["kv_cache_snapshot"] != "portable" {
		t.Fatalf("cache warm capability = %+v ok=%v, want experimental warm with portable remirror labels", cap, ok)
	}
	if cap, ok := report.Capability(inference.CapabilityCacheDisk); !ok || cap.Status != inference.CapabilityStatusExperimental ||
		!strings.Contains(cap.Detail, "state") ||
		!strings.Contains(cap.Detail, "KV snapshots") ||
		!strings.Contains(cap.Detail, "HIP device remirror") ||
		cap.Labels["disk_cache_restore"] != "exact_cold_ref" {
		t.Fatalf("cache disk capability = %+v ok=%v, want experimental state-backed KV snapshot disk refs with remirror labels", cap, ok)
	}
	if cap, ok := report.Capability(inference.CapabilityKVSnapshot); !ok || cap.Status != inference.CapabilityStatusExperimental ||
		!strings.Contains(cap.Detail, "package-local KV snapshots") ||
		!strings.Contains(cap.Detail, "device-mirror") ||
		!strings.Contains(cap.Detail, "block-cache") ||
		cap.Labels["kv_backing"] != "package_local" ||
		cap.Labels["kv_device_backing"] != "best_effort_remirror" {
		t.Fatalf("KV snapshot capability = %+v ok=%v, want experimental package-local snapshots plus state/cache device remirror", cap, ok)
	}
	if cap, ok := report.Capability(inference.CapabilityPromptCache); !ok || cap.Status != inference.CapabilityStatusExperimental ||
		!strings.Contains(cap.Detail, "best-effort HIP device remirror") ||
		!strings.Contains(cap.Detail, "native prefill reuse remains pending") ||
		cap.Labels["native_prefill_reuse"] != "pending" {
		t.Fatalf("prompt cache capability = %+v ok=%v, want experimental package-local cache with remirror and native prefill caveat", cap, ok)
	}
	if cap, ok := report.Capability(inference.CapabilityStateBundle); !ok || cap.Status != inference.CapabilityStatusExperimental || !strings.Contains(cap.Detail, "metadata-only") {
		t.Fatalf("state bundle capability = %+v ok=%v, want experimental metadata-only bundle surface", cap, ok)
	}
	for _, id := range []inference.CapabilityID{inference.CapabilitySpeculativeDecode, inference.CapabilityPromptLookupDecode} {
		if cap, ok := report.Capability(id); !ok || cap.Status != inference.CapabilityStatusPlanned {
			t.Fatalf("decode helper capability %s = %+v ok=%v, want planned until decode kernel is linked", id, cap, ok)
		}
	}
	for _, id := range []inference.CapabilityID{inference.CapabilityAgentMemory, inference.CapabilityStateWake, inference.CapabilityStateSleep, inference.CapabilityStateFork} {
		if cap, ok := report.Capability(id); !ok || cap.Status != inference.CapabilityStatusExperimental {
			t.Fatalf("state capability %s = %+v ok=%v, want experimental state lifecycle groundwork", id, cap, ok)
		}
	}
	if cap, ok := report.Capability(inference.CapabilityAgentMemory); !ok ||
		cap.Labels["hierarchical_memory_pretraining"] != "experimental" ||
		cap.Labels["memory_pretraining_package"] != "dappco.re/go/rocm/memorypretrain" ||
		cap.Labels["memory_bank_builder"] != "hierarchical_kmeans" ||
		cap.Labels["memory_pretraining_retrieval"] != "leaf_cluster_topk" ||
		cap.Labels["memory_pretraining_injection"] != "additive" ||
		cap.Labels["memory_pretraining_runtime"] != "cpu_native" ||
		cap.Labels["memory_pretraining_hip_injection"] != "pending" ||
		cap.Labels["memory_pretraining_training_bridge"] != "RunModelNativeSimpleSelfDistillationMemoryPretraining" ||
		cap.Labels["memory_pretraining_optimizer_track"] != "append_only_adamw" ||
		cap.Labels["memory_pretraining_optimizer_track_containers"] != "kv,mp4,binary" ||
		cap.Labels["memory_pretraining_optimizer_track_frames"] != "propagated" ||
		cap.Labels["memory_pretraining_optimizer_track_finder"] != "FindNativeAdamWStateTrackStep" ||
		cap.Labels["memory_pretraining_optimizer_track_lister"] != "ListNativeAdamWStateTrack" ||
		cap.Labels["memory_pretraining_optimizer_track_loader"] != "LoadNativeAdamWStateTrackStep" ||
		cap.Labels["memory_pretraining_hot_path_benchmarks"] != "present" {
		t.Fatalf("agent memory capability = %+v ok=%v, want hierarchical-memory pretraining labels", cap, ok)
	}
	if cap, ok := report.Capability(inference.CapabilityBenchmark); !ok || cap.Status != inference.CapabilityStatusExperimental {
		t.Fatalf("benchmark capability = %+v ok=%v, want experimental benchmark wrapper", cap, ok)
	}
	if cap, ok := report.Capability(inference.CapabilityResponsesAPI); !ok || cap.Status != inference.CapabilityStatusExperimental {
		t.Fatalf("responses capability = %+v ok=%v, want experimental streaming handler", cap, ok)
	} else if !strings.Contains(cap.Detail, "SSE streaming") || strings.Contains(cap.Detail, "streaming is pending") {
		t.Fatalf("responses capability = %+v, want streaming advertised", cap)
	}
	for _, id := range []inference.CapabilityID{inference.CapabilityAnthropicMessages, inference.CapabilityOllamaCompat} {
		if cap, ok := report.Capability(id); !ok || cap.Status != inference.CapabilityStatusExperimental {
			t.Fatalf("wire capability %s = %+v ok=%v, want experimental handler", id, cap, ok)
		}
	}
	if cap, ok := report.Capability(inference.CapabilityAnthropicMessages); !ok ||
		!strings.Contains(cap.Detail, "SSE streaming") ||
		strings.Contains(cap.Detail, "streaming is pending") {
		t.Fatalf("Anthropic capability = %+v ok=%v, want streaming advertised", cap, ok)
	}
	if cap, ok := report.Capability(inference.CapabilityOllamaCompat); !ok ||
		!strings.Contains(cap.Detail, "streaming") ||
		!strings.Contains(cap.Detail, "/api/tags") ||
		!strings.Contains(cap.Detail, "/api/show") ||
		strings.Contains(cap.Detail, "streaming remains pending") ||
		strings.Contains(cap.Detail, "model registry endpoints are pending") {
		t.Fatalf("Ollama capability = %+v ok=%v, want streaming registry tags/show advertised", cap, ok)
	}
	requiredTrainingKernels := map[inference.CapabilityID]string{
		inference.CapabilityLoRATraining: "lora_backward",
		inference.CapabilityDistillation: "distillation_forward_loss",
		inference.CapabilityGRPO:         "grpo_rollout_policy",
	}
	trainingFixtureKernels := map[inference.CapabilityID]string{
		inference.CapabilityDistillation: hipKernelNameDistillKL,
		inference.CapabilityGRPO:         hipKernelNameGRPOAdvantage,
	}
	for _, id := range []inference.CapabilityID{inference.CapabilityLoRATraining, inference.CapabilityDistillation, inference.CapabilityGRPO} {
		cap, ok := report.Capability(id)
		if !ok || cap.Status != inference.CapabilityStatusPlanned ||
			cap.Labels["runtime_status"] != string(inference.FeatureRuntimePlanned) ||
			cap.Labels["training_kernel"] != hipKernelStatusNotLinked ||
			cap.Labels["training_interface"] != "not_implemented" ||
			cap.Labels["required_kernel"] != requiredTrainingKernels[id] ||
			cap.Labels["optimizer_status"] != "update_only" ||
			cap.Labels["optimizer_backend"] != "reference" ||
			cap.Labels["optimizer_kernel"] != hipKernelStatusNotLinked ||
			cap.Labels["optimizer_direct_helper"] != "RunNativeAdamWUpdate" ||
			cap.Labels["optimizer_helper"] != "RunNativeAdamWUpdatePass" ||
			cap.Labels["optimizer_launch_args"] != "hipAdamWUpdateLaunchArgs" ||
			cap.Labels["optimizer_launch_args_bytes"] != "128" ||
			cap.Labels["optimizer_layout"] != "packed_contiguous_parameters_m_v" ||
			cap.Labels["optimizer_track"] != "append_only" ||
			cap.Labels["optimizer_track_containers"] != "kv,mp4,binary" ||
			cap.Labels["optimizer_track_helper"] != "AppendNativeAdamWStateTrack" ||
			cap.Labels["optimizer_track_list_helper"] != "ListNativeAdamWStateTrack" ||
			cap.Labels["optimizer_track_find_helper"] != "FindNativeAdamWStateTrackStep" ||
			cap.Labels["optimizer_track_load_step_helper"] != "LoadNativeAdamWStateTrackStep" {
			t.Fatalf("training capability %s = %+v ok=%v, want planned/not-linked training labels", id, cap, ok)
		}
		if fixture := trainingFixtureKernels[id]; fixture != "" {
			if cap.Labels["fixture_kernel"] != hipKernelStatusNotLinked || cap.Labels["fixture_kernel_name"] != fixture {
				t.Fatalf("training capability %s labels = %+v, want not-linked toy fixture kernel %s without native kernels", id, cap.Labels, fixture)
			}
		}
		if id == inference.CapabilityLoRATraining &&
			(cap.Labels["lora_update_helper"] != "RunNativeLoRAAdamWUpdatePass" ||
				cap.Labels["lora_backward_backend"] != "reference" ||
				cap.Labels["lora_adapter_snapshot_helper"] != "SaveNativeLoRAAdapterSnapshot" ||
				cap.Labels["lora_adapter_track_latest_snapshot_helper"] != "SaveNativeLoRAAdapterSnapshotTrackLast" ||
				cap.Labels["lora_adapter_track_snapshot_helper"] != "SaveNativeLoRAAdapterSnapshotTrackStep") {
			t.Fatalf("LoRA training capability labels = %+v, want reference LoRA update helper", cap.Labels)
		}
		if id == inference.CapabilityDistillation &&
			(cap.Labels["distillation_update_helper"] != "RunNativeDistillationAdamWUpdatePass" ||
				cap.Labels["distillation_track_helper"] != "RunNativeDistillationAdamWUpdateTrackPass") {
			t.Fatalf("distillation training capability labels = %+v, want distillation update and track helpers", cap.Labels)
		}
		if id == inference.CapabilityGRPO &&
			(cap.Labels["advantage_update_helper"] != "RunNativeGRPOAdamWUpdatePass" ||
				cap.Labels["advantage_track_helper"] != "RunNativeGRPOAdamWUpdateTrackPass" ||
				cap.Labels["policy_loss_helper"] != "RunNativeGRPOPolicyLossPass" ||
				cap.Labels["policy_update_helper"] != "RunNativeGRPOPolicyAdamWUpdatePass" ||
				cap.Labels["policy_track_helper"] != "RunNativeGRPOPolicyAdamWUpdateTrackPass" ||
				cap.Labels["policy_rollout_group_label"] != "group_id" ||
				cap.Labels["policy_rollout_group_result_labels"] != "grpo_rollout_group_source,grpo_rollout_groups" ||
				cap.Labels["policy_rollout_identity_labels"] != "rollout_id,sample_id,trajectory_id,turn_id,completion_id,episode_id" ||
				cap.Labels["policy_rollout_identity_result_labels"] != "grpo_rollouts,grpo_rollout_samples,grpo_rollout_trajectories,grpo_rollout_turns,grpo_rollout_completions,grpo_rollout_episodes" ||
				cap.Labels["policy_rollout_prompt_labels"] != "prompt_id,query_id" ||
				cap.Labels["policy_rollout_prompt_result_labels"] != "grpo_rollout_prompt_source,grpo_rollout_prompts" ||
				cap.Labels["policy_loss_backend"] != "reference") {
			t.Fatalf("GRPO training capability labels = %+v, want reference advantage and policy helpers", cap.Labels)
		}
	}
	for _, id := range nativeContractSharedCapabilityIDs() {
		if _, ok := report.Capability(id); !ok {
			t.Fatalf("capability %q missing from ROCm report: %+v", id, report.CapabilityIDs())
		}
	}
	if len(report.Architectures) == 0 || len(report.Quantizations) == 0 || len(report.CacheModes) == 0 {
		t.Fatalf("report = %+v, want architecture/quant/cache metadata", report)
	}
}

func TestNativeContract_RocmBackendCapabilitiesUseRuntimeKernelStatus_Good(t *testing.T) {
	runtime := &fakeNativeRuntime{
		available: true,
		device:    nativeDeviceInfo{Name: "gfx1100", MemoryBytes: 16 * memoryGiB, FreeBytes: 8 * memoryGiB, Driver: "hip-test"},
		kernelStatus: hipKernelStatus{
			Decode:     hipKernelStatusNotLinked,
			Optimizer:  hipKernelStatusLinked,
			Prefill:    hipKernelStatusNotLinked,
			Projection: hipKernelStatusLinked,
			KVCache:    hipKernelStatusPlanned,
		},
	}

	report := newROCmBackendWithRuntime(runtime).Capabilities()

	if report.Labels["kernel_status"] != hipKernelStatusLinked ||
		report.Labels["cross_entropy_kernel"] != hipKernelStatusNotLinked ||
		report.Labels["decode_kernel"] != hipKernelStatusNotLinked ||
		report.Labels["optimizer_kernel"] != hipKernelStatusLinked ||
		report.Labels["prefill_kernel"] != hipKernelStatusNotLinked ||
		report.Labels["projection_kernel"] != hipKernelStatusLinked {
		t.Fatalf("labels = %+v, want runtime kernel status", report.Labels)
	}
	if capability, ok := report.Capability(inference.CapabilityLoRATraining); !ok || capability.Labels["optimizer_kernel"] != hipKernelStatusLinked {
		t.Fatalf("LoRA training capability = %+v ok=%v, want linked optimizer kernel status", capability, ok)
	}
	if capability, ok := report.Capability(inference.CapabilityEvaluation); !ok || capability.Labels["loss_kernel"] != hipKernelStatusNotLinked {
		t.Fatalf("evaluation capability = %+v ok=%v, want loss fixture not linked for projection-only status", capability, ok)
	}
	if report.Supports(inference.CapabilityGenerate) {
		t.Fatalf("generate should remain planned without linked decode kernel: %+v", report.CapabilityIDs())
	}
}

func TestNativeContract_RocmBackendUnavailableRuntime_Bad(t *testing.T) {
	backend := newROCmBackendWithRuntime(&fakeNativeRuntime{})

	if backend.Available() {
		t.Fatalf("Available = true, want false for unavailable fake native runtime")
	}
	report := backend.Capabilities()
	if report.Available {
		t.Fatalf("report.Available = true, want false")
	}
	if report.Labels["runtime_status"] != "unavailable" || report.Labels["kernel_status"] != hipKernelStatusNotLinked {
		t.Fatalf("labels = %+v, want unavailable runtime and not-linked kernel status", report.Labels)
	}
	_, err := resultValue[inference.TextModel](backend.LoadModel(nativeContractGGUF(t)))
	if err == nil || !core.Contains(err.Error(), "native ROCm runtime is not available") {
		t.Fatalf("LoadModel error = %v, want clear native runtime unavailable error", err)
	}
}

func TestNativeContract_RocmModelCapabilities_Ugly(t *testing.T) {
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3", NumLayers: 28, QuantBits: 4},
		native:    &fakeNativeModel{adapter: inference.AdapterIdentity{Path: "domain.safetensors", Format: "lora"}, kernelStatus: defaultHIPKernelStatus()},
	}

	report := model.Capabilities()

	if !report.Available || report.Model.Architecture != "qwen3" || report.Adapter.Path != "domain.safetensors" {
		t.Fatalf("report = %+v, want loaded model and adapter identity", report)
	}
	if report.Supports(inference.CapabilityLoRAInference) {
		t.Fatalf("LoRA inference should be planned until HIP adapter application is linked")
	}
	if report.Supports(inference.CapabilityEmbeddings) || report.Supports(inference.CapabilityRerank) {
		t.Fatalf("embeddings/rerank should remain planned until HIP kernels are linked")
	}
	if !report.Supports(inference.CapabilityEvaluation) {
		t.Fatalf("evaluation should be experimentally available: %+v", report.CapabilityIDs())
	}
}

func TestNativeContract_CapabilityReportGenericReactiveRegistryLabels_Good(t *testing.T) {
	report := rocmCapabilityReport(nativeDeviceInfo{}, inference.ModelIdentity{
		Path:         "/models/qwen",
		Architecture: "Qwen3_5MoeForConditionalGeneration",
		QuantBits:    4,
	}, inference.AdapterIdentity{}, true, defaultHIPKernelStatus())

	if report.Labels["engine_feature_architecture"] != "qwen3_6_moe" ||
		report.Labels["engine_feature_family"] != "qwen" ||
		report.Labels["engine_feature_chat_template_id"] != "qwen" ||
		report.Labels["engine_feature_reasoning_parser"] != "qwen" ||
		report.Labels["engine_feature_tool_parser"] != "qwen" ||
		report.Labels["engine_feature_text_generate"] != "false" ||
		report.Labels["engine_feature_capabilities"] != "chat.template,reasoning.parse,tool.parse" ||
		report.Labels["engine_load_status"] != string(ROCmModelLoadStagedNative) ||
		report.Labels["engine_load_target"] != "standalone" ||
		report.Labels["engine_load_staged"] != "true" ||
		report.Labels["engine_load_text_generate"] != "false" {
		t.Fatalf("report labels = %+v, want generic registry-derived Qwen engine feature labels", report.Labels)
	}
	modelLoad, ok := report.Capability(inference.CapabilityModelLoad)
	if !ok ||
		modelLoad.Labels["engine_load_status"] != string(ROCmModelLoadStagedNative) ||
		modelLoad.Labels["engine_load_target"] != "standalone" ||
		modelLoad.Labels["engine_load_staged"] != "true" ||
		modelLoad.Labels["engine_load_text_generate"] != "false" {
		t.Fatalf("model-load capability = %+v ok=%v, want staged Qwen load-status labels", modelLoad, ok)
	}
	if cap, ok := report.Capability(inference.CapabilityGenerate); !ok || cap.Status != inference.CapabilityStatusPlanned {
		t.Fatalf("generate capability = %+v ok=%v, staged Qwen must not claim linked generation", cap, ok)
	}
	chatTemplate, ok := report.Capability(inference.CapabilityChatTemplate)
	if !ok ||
		chatTemplate.Labels["chat_template"] != "qwen" ||
		chatTemplate.Labels["engine_feature_chat_template_id"] != "qwen" ||
		chatTemplate.Labels["engine_feature_reasoning_parser"] != "qwen" ||
		chatTemplate.Labels["engine_feature_tool_parser"] != "qwen" {
		t.Fatalf("chat template capability = %+v ok=%v, want Qwen registry template labels", chatTemplate, ok)
	}
	for _, id := range []inference.CapabilityID{inference.CapabilityReasoningParse, inference.CapabilityToolParse} {
		capability, ok := report.Capability(id)
		if !ok || capability.Status != inference.CapabilityStatusSupported ||
			capability.Labels["engine_feature_architecture"] != "qwen3_6_moe" ||
			capability.Labels["engine_feature_reasoning_parser"] != "qwen" ||
			capability.Labels["engine_feature_tool_parser"] != "qwen" {
			t.Fatalf("parser capability %s = %+v ok=%v, want registry parser labels", id, capability, ok)
		}
	}
}

func TestNativeContract_CapabilityReportClonesIdentityMetadata_Good(t *testing.T) {
	modelIdentity := inference.ModelIdentity{
		Architecture: "qwen3",
		Labels:       map[string]string{"model": "source"},
	}
	adapterIdentity := inference.AdapterIdentity{
		Path:       "domain.safetensors",
		Format:     "lora",
		TargetKeys: []string{"lm_head"},
		Labels:     map[string]string{"adapter": "source"},
	}

	report := rocmCapabilityReport(nativeDeviceInfo{}, modelIdentity, adapterIdentity, true, defaultHIPKernelStatus())
	report.Model.Labels["model"] = "report-mutated"
	report.Adapter.TargetKeys[0] = "report-mutated"
	report.Adapter.Labels["adapter"] = "report-mutated"
	core.AssertEqual(t, "source", modelIdentity.Labels["model"])
	core.AssertEqual(t, "lm_head", adapterIdentity.TargetKeys[0])
	core.AssertEqual(t, "source", adapterIdentity.Labels["adapter"])

	modelIdentity.Labels["model"] = "input-mutated"
	adapterIdentity.TargetKeys[0] = "input-mutated"
	adapterIdentity.Labels["adapter"] = "input-mutated"

	core.AssertEqual(t, "report-mutated", report.Model.Labels["model"])
	core.AssertEqual(t, "report-mutated", report.Adapter.TargetKeys[0])
	core.AssertEqual(t, "report-mutated", report.Adapter.Labels["adapter"])
	core.AssertEqual(t, "input-mutated", modelIdentity.Labels["model"])
	core.AssertEqual(t, "input-mutated", adapterIdentity.TargetKeys[0])
	core.AssertEqual(t, "input-mutated", adapterIdentity.Labels["adapter"])
}

func TestNativeContract_RocmModelDoesNotImplementTrainingSurfaces_Ugly(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{}}

	if _, ok := any(model).(inference.TrainableModel); ok {
		t.Fatalf("rocmModel unexpectedly implements TrainableModel before native training kernels exist")
	}
	if _, ok := any(model).(inference.SFTTrainer); ok {
		t.Fatalf("rocmModel unexpectedly implements SFTTrainer before native training kernels exist")
	}
	if _, ok := any(model).(inference.DistillTrainer); ok {
		t.Fatalf("rocmModel unexpectedly implements DistillTrainer before native training kernels exist")
	}
	if _, ok := any(model).(inference.GRPOTrainer); ok {
		t.Fatalf("rocmModel unexpectedly implements GRPOTrainer before rollout kernels exist")
	}
	report := model.Capabilities()
	requiredTrainingKernels := map[inference.CapabilityID]string{
		inference.CapabilityLoRATraining: "lora_backward",
		inference.CapabilityDistillation: "distillation_forward_loss",
		inference.CapabilityGRPO:         "grpo_rollout_policy",
	}
	trainingFixtureKernels := map[inference.CapabilityID]string{
		inference.CapabilityDistillation: hipKernelNameDistillKL,
		inference.CapabilityGRPO:         hipKernelNameGRPOAdvantage,
	}
	for _, id := range []inference.CapabilityID{inference.CapabilityLoRATraining, inference.CapabilityDistillation, inference.CapabilityGRPO} {
		capability, ok := report.Capability(id)
		if !ok || capability.Status != inference.CapabilityStatusPlanned ||
			capability.Labels["runtime_status"] != string(inference.FeatureRuntimePlanned) ||
			capability.Labels["training_kernel"] != hipKernelStatusNotLinked ||
			capability.Labels["training_interface"] != "not_implemented" ||
			capability.Labels["required_kernel"] != requiredTrainingKernels[id] ||
			capability.Labels["optimizer_status"] != "update_only" ||
			capability.Labels["optimizer_backend"] != "reference" ||
			capability.Labels["optimizer_kernel"] != hipKernelStatusNotLinked ||
			capability.Labels["optimizer_direct_helper"] != "RunNativeAdamWUpdate" ||
			capability.Labels["optimizer_helper"] != "RunNativeAdamWUpdatePass" ||
			capability.Labels["optimizer_launch_args"] != "hipAdamWUpdateLaunchArgs" ||
			capability.Labels["optimizer_launch_args_bytes"] != "128" ||
			capability.Labels["optimizer_layout"] != "packed_contiguous_parameters_m_v" ||
			capability.Labels["optimizer_track"] != "append_only" ||
			capability.Labels["optimizer_track_containers"] != "kv,mp4,binary" ||
			capability.Labels["optimizer_track_helper"] != "AppendNativeAdamWStateTrack" ||
			capability.Labels["optimizer_track_list_helper"] != "ListNativeAdamWStateTrack" ||
			capability.Labels["optimizer_track_find_helper"] != "FindNativeAdamWStateTrackStep" ||
			capability.Labels["optimizer_track_load_step_helper"] != "LoadNativeAdamWStateTrackStep" {
			t.Fatalf("training capability %s = %+v ok=%v, want planned/not-linked model report", id, capability, ok)
		}
		if fixture := trainingFixtureKernels[id]; fixture != "" {
			if capability.Labels["fixture_kernel"] != hipKernelStatusNotLinked || capability.Labels["fixture_kernel_name"] != fixture {
				t.Fatalf("training capability %s labels = %+v, want not-linked toy fixture kernel %s without native kernels", id, capability.Labels, fixture)
			}
		}
		if id == inference.CapabilityLoRATraining &&
			(capability.Labels["lora_update_helper"] != "RunNativeLoRAAdamWUpdatePass" ||
				capability.Labels["lora_backward_backend"] != "reference" ||
				capability.Labels["lora_adapter_snapshot_helper"] != "SaveNativeLoRAAdapterSnapshot" ||
				capability.Labels["lora_adapter_track_latest_snapshot_helper"] != "SaveNativeLoRAAdapterSnapshotTrackLast" ||
				capability.Labels["lora_adapter_track_snapshot_helper"] != "SaveNativeLoRAAdapterSnapshotTrackStep") {
			t.Fatalf("LoRA training capability labels = %+v, want reference LoRA update helper", capability.Labels)
		}
		if id == inference.CapabilityDistillation &&
			(capability.Labels["distillation_update_helper"] != "RunNativeDistillationAdamWUpdatePass" ||
				capability.Labels["distillation_track_helper"] != "RunNativeDistillationAdamWUpdateTrackPass") {
			t.Fatalf("distillation training capability labels = %+v, want distillation update and track helpers", capability.Labels)
		}
		if id == inference.CapabilityGRPO &&
			(capability.Labels["advantage_update_helper"] != "RunNativeGRPOAdamWUpdatePass" ||
				capability.Labels["advantage_track_helper"] != "RunNativeGRPOAdamWUpdateTrackPass" ||
				capability.Labels["policy_loss_helper"] != "RunNativeGRPOPolicyLossPass" ||
				capability.Labels["policy_update_helper"] != "RunNativeGRPOPolicyAdamWUpdatePass" ||
				capability.Labels["policy_track_helper"] != "RunNativeGRPOPolicyAdamWUpdateTrackPass" ||
				capability.Labels["policy_rollout_group_label"] != "group_id" ||
				capability.Labels["policy_rollout_group_result_labels"] != "grpo_rollout_group_source,grpo_rollout_groups" ||
				capability.Labels["policy_rollout_identity_labels"] != "rollout_id,sample_id,trajectory_id,turn_id,completion_id,episode_id" ||
				capability.Labels["policy_rollout_identity_result_labels"] != "grpo_rollouts,grpo_rollout_samples,grpo_rollout_trajectories,grpo_rollout_turns,grpo_rollout_completions,grpo_rollout_episodes" ||
				capability.Labels["policy_rollout_prompt_labels"] != "prompt_id,query_id" ||
				capability.Labels["policy_rollout_prompt_result_labels"] != "grpo_rollout_prompt_source,grpo_rollout_prompts" ||
				capability.Labels["policy_loss_backend"] != "reference") {
			t.Fatalf("GRPO training capability labels = %+v, want reference advantage and policy helpers", capability.Labels)
		}
	}
}

func TestNativeContract_RocmModelCapabilitiesUseNativeKernelStatus_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3", NumLayers: 28, QuantBits: 4},
		native: &fakeNativeModel{kernelStatus: hipKernelStatus{
			CrossEntropy: hipKernelStatusLinked,
			Decode:       hipKernelStatusLinked,
			Distillation: hipKernelStatusLinked,
			GRPO:         hipKernelStatusLinked,
			Prefill:      hipKernelStatusLinked,
			Projection:   hipKernelStatusPlanned,
			KVCache:      hipKernelStatusPlanned,
			Reason:       "fake deterministic kernel fixture",
		}},
	}

	report := model.Capabilities()

	if report.Labels["kernel_status"] != hipKernelStatusLinked || report.Labels["cross_entropy_kernel"] != hipKernelStatusLinked || report.Labels["decode_kernel"] != hipKernelStatusLinked || report.Labels["distillation_kernel"] != hipKernelStatusLinked || report.Labels["grpo_kernel"] != hipKernelStatusLinked || report.Labels["prefill_kernel"] != hipKernelStatusLinked || report.Labels["projection_kernel"] != hipKernelStatusPlanned {
		t.Fatalf("labels = %+v, want linked decode/prefill and planned projection kernel status", report.Labels)
	}
	for _, id := range []inference.CapabilityID{inference.CapabilityGenerate, inference.CapabilityChat, inference.CapabilityClassify, inference.CapabilityBatchGenerate} {
		capability, ok := report.Capability(id)
		if !ok || capability.Status != inference.CapabilityStatusExperimental {
			t.Fatalf("capability %s = %+v ok=%v, want experimental with linked fake kernels", id, capability, ok)
		}
		if id == inference.CapabilityClassify {
			if capability.Labels["prefill_kernel_name"] != hipKernelNamePrefill || capability.Labels["kernel_scope"] != "native_prefill" {
				t.Fatalf("classify capability labels = %+v, want production prefill kernel labels", capability.Labels)
			}
			continue
		}
		if capability.Labels["decode_kernel"] != hipKernelStatusLinked ||
			capability.Labels["decode_kernel_name"] != hipKernelNameDecode ||
			capability.Labels["prefill_kernel_name"] != hipKernelNamePrefill ||
			capability.Labels["kernel_scope"] != "native_decode" {
			t.Fatalf("capability %s labels = %+v, want production decode kernel labels", id, capability.Labels)
		}
	}
	if capability, ok := report.Capability(inference.CapabilityBenchmark); !ok || capability.Status != inference.CapabilityStatusExperimental || capability.Labels["decode_kernel"] != hipKernelStatusLinked || strings.Contains(capability.Detail, "not linked") {
		t.Fatalf("benchmark capability = %+v ok=%v, want linked decode-aware experimental detail", capability, ok)
	}
	if capability, ok := report.Capability(inference.CapabilityEvaluation); !ok ||
		capability.Status != inference.CapabilityStatusExperimental ||
		capability.Labels["prefill_kernel"] != hipKernelStatusLinked ||
		capability.Labels["loss_kernel"] != hipKernelStatusLinked ||
		capability.Labels["loss_kernel_name"] != hipKernelNameCrossEntropy ||
		strings.Contains(capability.Detail, "before prefill") {
		t.Fatalf("evaluation capability = %+v ok=%v, want linked prefill/loss-aware experimental detail", capability, ok)
	}
	if capability, ok := report.Capability(inference.CapabilityLogitProbe); !ok || capability.Status != inference.CapabilityStatusExperimental || capability.Labels["prefill_kernel"] != hipKernelStatusLinked {
		t.Fatalf("logit probe capability = %+v ok=%v, want experimental with linked prefill kernel", capability, ok)
	}
	for _, id := range []inference.CapabilityID{inference.CapabilityDistillation, inference.CapabilityGRPO} {
		capability, ok := report.Capability(id)
		if !ok ||
			capability.Labels["fixture_kernel"] != hipKernelStatusLinked ||
			capability.Labels["optimizer_status"] != "update_only" ||
			capability.Labels["optimizer_kernel"] != hipKernelStatusNotLinked {
			t.Fatalf("training capability %s = %+v ok=%v, want linked toy fixture label and update-only optimizer metadata when native kernels are configured", id, capability, ok)
		}
	}
	for _, id := range []inference.CapabilityID{inference.CapabilitySpeculativeDecode, inference.CapabilityPromptLookupDecode} {
		if capability, ok := report.Capability(id); !ok || capability.Status != inference.CapabilityStatusExperimental || capability.Labels["decode_kernel"] != hipKernelStatusLinked {
			t.Fatalf("decode helper capability %s = %+v ok=%v, want experimental with linked decode kernel", id, capability, ok)
		}
	}
	if capability, ok := report.Capability(inference.CapabilityAttentionProbe); !ok || capability.Status != inference.CapabilityStatusPlanned {
		t.Fatalf("attention probe capability = %+v ok=%v, want planned until native attention probes are emitted", capability, ok)
	}
}

func TestNativeContract_RocmTinyFixtureCapabilitiesLabelProductionPending_Good(t *testing.T) {
	report := rocmCapabilityReport(nativeDeviceInfo{}, inference.ModelIdentity{
		Architecture: "tiny",
		VocabSize:    3,
		HiddenSize:   2,
	}, inference.AdapterIdentity{}, true, hipKernelStatus{
		Decode:     hipKernelStatusLinked,
		Prefill:    hipKernelStatusLinked,
		Projection: hipKernelStatusLinked,
		KVCache:    hipKernelStatusPlanned,
		Reason:     "fake tiny fixture",
	})

	for _, id := range []inference.CapabilityID{
		inference.CapabilityGenerate,
		inference.CapabilityChat,
		inference.CapabilityBatchGenerate,
		inference.CapabilityBenchmark,
		inference.CapabilitySpeculativeDecode,
		inference.CapabilityPromptLookupDecode,
	} {
		capability, ok := report.Capability(id)
		if !ok || capability.Status != inference.CapabilityStatusExperimental ||
			capability.Labels["kernel_scope"] != "toy_tiny_fixture" ||
			capability.Labels["decode_kernel_name"] != hipKernelNameTinyDecode ||
			capability.Labels["prefill_kernel_name"] != hipKernelNameTinyPrefill ||
			capability.Labels["production_decode"] != hipKernelStatusNotLinked ||
			capability.Labels["production_prefill"] != hipKernelStatusNotLinked {
			t.Fatalf("capability %s = %+v ok=%v, want linked toy fixture labels with production pending", id, capability, ok)
		}
	}
	classify, ok := report.Capability(inference.CapabilityClassify)
	if !ok || classify.Status != inference.CapabilityStatusExperimental ||
		classify.Labels["kernel_scope"] != "toy_tiny_fixture" ||
		classify.Labels["prefill_kernel_name"] != hipKernelNameTinyPrefill ||
		classify.Labels["production_prefill"] != hipKernelStatusNotLinked {
		t.Fatalf("classify capability = %+v ok=%v, want linked toy prefill labels with production pending", classify, ok)
	}
}

func TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability_Good(t *testing.T) {
	report := rocmCapabilityReport(nativeDeviceInfo{}, inference.ModelIdentity{
		Path:          "/models/lmstudio-community-gemma-4-e2b-it-4bit",
		Architecture:  "gemma4",
		VocabSize:     262144,
		NumLayers:     35,
		HiddenSize:    1536,
		QuantBits:     4,
		QuantGroup:    64,
		ContextLength: 131072,
	}, inference.AdapterIdentity{}, true, defaultHIPKernelStatus(), rocmCapabilityReportOption{Gemma4Q4GenerateLinked: true})

	generate, ok := report.Capability(inference.CapabilityGenerate)
	if !ok || generate.Status != inference.CapabilityStatusExperimental ||
		generate.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_generate" ||
		generate.Labels["gemma4_q4_decode_kernel"] != hipKernelStatusLinked ||
		generate.Labels["gemma4_q4_decode_name"] != "rocm_gemma4_q4_greedy_decode_smoke" ||
		generate.Labels["attention_kv_backing"] != "hip_device_descriptor" ||
		generate.Labels["attention_kv_mode"] != rocmKVCacheModeKQ8VQ4 ||
		generate.Labels["gemma4_q4_device_kv_state"] != "forward_returned_device_state" ||
		generate.Labels["decode_architecture"] != "gemma4" ||
		generate.Labels["decode_quant"] != "mlx_q4" ||
		generate.Labels["gemma4_mlx_affine_bits"] != "4" ||
		generate.Labels["gemma4_mlx_affine_decode"] != hipKernelStatusLinked ||
		generate.Labels["gemma4_mlx_affine_kv_state"] != "forward_returned_device_state" ||
		generate.Labels["gemma4_size"] != "E2B" ||
		generate.Labels["gemma4_quant_mode"] != "q4" ||
		generate.Labels["gemma4_pack_supported"] != "true" ||
		generate.Labels["gemma4_runtime"] != Gemma4RuntimeMLXAffine ||
		generate.Labels["gemma4_generate_status"] != Gemma4GenerateLinked ||
		generate.Labels["gemma4_runnable_on_card"] != "true" ||
		generate.Labels["quant_default_tier"] != "q6" ||
		generate.Labels["quant_family"] != "mlx_affine" ||
		generate.Labels["quant_ladder"] != "bf16,q8,q6,q4" ||
		generate.Labels["production_quant_policy"] != "gemma4_mlx_affine" ||
		generate.Labels["production_quant_tier"] != "constrained" ||
		generate.Labels["production_quant_pack_count"] != "20" ||
		generate.Labels["production_quant_pack_sizes"] != "E2B,E4B,12B,26B-A4B,31B" ||
		!strings.Contains(generate.Labels["production_quant_linked_generate_packs"], "E4B:q6") ||
		!strings.Contains(generate.Labels["production_quant_linked_generate_packs"], "12B:q6") ||
		!strings.Contains(generate.Labels["production_quant_load_only_packs"], "E2B:bf16") ||
		!strings.Contains(generate.Labels["production_quant_planned_packs"], "E4B:mxfp4") ||
		generate.Labels["production_quant_active_weight_read_bytes_per_token"] == "" ||
		generate.Labels["decode_layers"] != "35" ||
		generate.Labels["decode_vocab_size"] != "262144" ||
		generate.Labels["decode_hidden_size"] != "1536" ||
		generate.Labels["production_prefill"] != hipKernelStatusNotLinked ||
		generate.Labels["production_decode"] != hipKernelStatusNotLinked ||
		generate.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked ||
		generate.Labels["runtime_status"] != string(inference.FeatureRuntimeExperimental) ||
		!strings.Contains(generate.Labels["prompt_modes"], "tokens") ||
		!strings.Contains(generate.Labels["prompt_modes"], "text") {
		t.Fatalf("generate capability = %+v ok=%v, want experimental Gemma4 q4 labels with production prefill/decode pending", generate, ok)
	}
	if !strings.Contains(generate.Detail, "production native prefill/decode remain pending") {
		t.Fatalf("generate detail = %q, want production prefill/decode caveat", generate.Detail)
	}
	if generate.Labels["engine_state_context_route_contract"] != ROCmStateContextRegistryContract ||
		generate.Labels["engine_state_context_window"] != "131072" ||
		generate.Labels["engine_state_context_prompt_replay_refused"] != "true" ||
		generate.Labels["engine_state_context_remaining_context_default"] != "true" ||
		generate.Labels["engine_state_context_runtime_owned_kv"] != "true" ||
		generate.Labels["engine_state_context_gemma4_size"] != "E2B" ||
		generate.Labels["engine_state_context_gemma4_quant_mode"] != "q4" {
		t.Fatalf("generate state/context labels = %+v, want Gemma4 route labels with model context window", generate.Labels)
	}
	batch, ok := report.Capability(inference.CapabilityBatchGenerate)
	if !ok || batch.Status != inference.CapabilityStatusExperimental ||
		batch.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_batch_generate" ||
		batch.Labels["batch_generate_kernel"] != hipKernelStatusLinked ||
		batch.Labels["batch_generate_name"] != "rocm_gemma4_q4_batch_generate_experimental" ||
		batch.Labels["gemma4_q4_decode_kernel"] != hipKernelStatusLinked ||
		batch.Labels["attention_kv_mode"] != rocmKVCacheModeKQ8VQ4 ||
		batch.Labels["production_prefill"] != hipKernelStatusNotLinked ||
		batch.Labels["production_decode"] != hipKernelStatusNotLinked ||
		batch.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked ||
		batch.Labels["runtime_status"] != string(inference.FeatureRuntimeExperimental) {
		t.Fatalf("batch capability = %+v ok=%v, want experimental Gemma4 q4 batch labels with production prefill/decode pending", batch, ok)
	}
	if !strings.Contains(batch.Detail, "production native prefill/decode remain pending") {
		t.Fatalf("batch detail = %q, want production prefill/decode caveat", batch.Detail)
	}
	chat, ok := report.Capability(inference.CapabilityChat)
	if !ok || chat.Status != inference.CapabilityStatusExperimental ||
		chat.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_chat" ||
		chat.Labels["chat_kernel"] != hipKernelStatusLinked ||
		chat.Labels["chat_name"] != "rocm_gemma4_q4_chat_generate_experimental" ||
		chat.Labels["chat_template"] != "gemma4_hf_turn" ||
		chat.Labels["gemma4_q4_decode_kernel"] != hipKernelStatusLinked ||
		chat.Labels["attention_kv_mode"] != rocmKVCacheModeKQ8VQ4 ||
		chat.Labels["production_prefill"] != hipKernelStatusNotLinked ||
		chat.Labels["production_decode"] != hipKernelStatusNotLinked ||
		chat.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked ||
		chat.Labels["runtime_status"] != string(inference.FeatureRuntimeExperimental) {
		t.Fatalf("chat capability = %+v ok=%v, want experimental Gemma4 q4 chat labels with production prefill/decode pending", chat, ok)
	}
	if !strings.Contains(chat.Detail, "production native prefill/decode remain pending") {
		t.Fatalf("chat detail = %q, want production prefill/decode caveat", chat.Detail)
	}
	chatTemplate, ok := report.Capability(inference.CapabilityChatTemplate)
	if !ok || chatTemplate.Status != inference.CapabilityStatusExperimental ||
		chatTemplate.Labels["chat_template"] != "gemma4_hf_turn" ||
		chatTemplate.Labels["turn_start"] != "<|turn>" ||
		chatTemplate.Labels["turn_end"] != "<turn|>" ||
		chatTemplate.Labels["generation_role"] != "model" ||
		chatTemplate.Labels["runtime_status"] != string(inference.FeatureRuntimeExperimental) {
		t.Fatalf("chat template capability = %+v ok=%v, want Gemma4 HF turn template labels", chatTemplate, ok)
	}
	if chatTemplate.Labels["engine_tokenizer_route_contract"] != ROCmModelTokenizerRegistryContract ||
		chatTemplate.Labels["engine_tokenizer_kind"] != "GemmaTokenizer" ||
		chatTemplate.Labels["engine_tokenizer_chat_template_id"] != "gemma4_hf_turn" ||
		chatTemplate.Labels["engine_tokenizer_generation_role"] != "model" ||
		chatTemplate.Labels["engine_tokenizer_model_owned_template"] != "true" {
		t.Fatalf("chat template tokenizer route labels = %+v, want Gemma4 tokenizer route labels", chatTemplate.Labels)
	}
	evaluation, ok := report.Capability(inference.CapabilityEvaluation)
	if !ok || evaluation.Status != inference.CapabilityStatusExperimental ||
		evaluation.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_eval" ||
		evaluation.Labels["eval_loss_logits_source"] != "gemma4_mlx_affine_package_prefill" ||
		evaluation.Labels["eval_prefill_kernel"] != hipKernelStatusLinked ||
		evaluation.Labels["eval_prefill_name"] != "rocm_gemma4_q4_package_prefill_experimental" ||
		evaluation.Labels["attention_kv_mode"] != rocmKVCacheModeKQ8VQ4 ||
		evaluation.Labels["production_prefill"] != hipKernelStatusNotLinked ||
		evaluation.Labels["production_decode"] != hipKernelStatusNotLinked ||
		evaluation.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked ||
		evaluation.Labels["runtime_status"] != string(inference.FeatureRuntimeExperimental) {
		t.Fatalf("evaluation capability = %+v ok=%v, want experimental Gemma4 q4 eval labels with production prefill/decode pending", evaluation, ok)
	}
	if !strings.Contains(evaluation.Detail, "production native prefill/decode remain pending") {
		t.Fatalf("evaluation detail = %q, want production prefill/decode caveat", evaluation.Detail)
	}
	if !strings.Contains(evaluation.Detail, "MLX affine 4/6/8-bit") {
		t.Fatalf("evaluation detail = %q, want bit-aware MLX affine detail", evaluation.Detail)
	}
	benchmark, ok := report.Capability(inference.CapabilityBenchmark)
	if !ok || benchmark.Status != inference.CapabilityStatusExperimental ||
		benchmark.Labels["attached_drafter_helper"] != hipKernelStatusLinked ||
		benchmark.Labels["attached_drafter_native_attachment"] != hipKernelStatusNotLinked ||
		benchmark.Labels["attached_drafter_role"] != "gemma4_assistant" ||
		benchmark.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_benchmark" ||
		benchmark.Labels["benchmark_kernel"] != hipKernelStatusLinked ||
		benchmark.Labels["benchmark_name"] != "rocm_gemma4_q4_benchmark_experimental" ||
		benchmark.Labels["benchmark_prompt_mode"] != "explicit_text" ||
		benchmark.Labels["benchmark_retained_state_book"] != "BenchmarkInferenceGemma4Q4Book10Turn_RetainedState" ||
		benchmark.Labels["benchmark_replay_baseline"] != "BenchmarkInferenceGemma4Q4Book10Turn_ReplayBaseline" ||
		benchmark.Labels["benchmark_retained_state_required"] != "true" ||
		benchmark.Labels["benchmark_prompt_replay_fallback"] != "forbidden" ||
		benchmark.Labels["benchmark_state_source"] != "rocm_state_session_runtime_kv" ||
		benchmark.Labels["production_book_policy"] != "retained_state_required" ||
		benchmark.Labels["production_book_decision_source"] != "benchmark_metrics" ||
		benchmark.Labels["production_book_gate_wall_seconds"] != strconv.Itoa(ProductionLaneBookWallSeconds) ||
		benchmark.Labels["production_book_gate_turns"] != strconv.Itoa(ProductionLaneBookTurnCount) ||
		benchmark.Labels["production_book_gate_raw_decode_tokens_per_sec"] != strconv.Itoa(DefaultProductionQuantizationPolicy().MinimumVisibleTokensPerSec) ||
		benchmark.Labels["production_book_gate_metrics"] == "" ||
		benchmark.Labels["production_book_gate_reason_codes"] != productionBookGateReasonCodesLabel ||
		benchmark.Labels["production_book_retained_route_metrics"] == "" ||
		benchmark.Labels["production_book_retained_artifact_labels"] == "" ||
		benchmark.Labels["production_book_long_output_quality_flags"] != "0" ||
		benchmark.Labels["production_model_source"] != "model_identity_or_pack" ||
		benchmark.Labels["production_mtp_required_metrics"] == "" ||
		benchmark.Labels["production_quant_decision_source"] != "gemma4_family_matrix" ||
		benchmark.Labels["attention_kv_mode"] != rocmKVCacheModeKQ8VQ4 ||
		benchmark.Labels["production_prefill"] != hipKernelStatusNotLinked ||
		benchmark.Labels["production_decode"] != hipKernelStatusNotLinked ||
		benchmark.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked ||
		benchmark.Labels["runtime_status"] != string(inference.FeatureRuntimeExperimental) {
		t.Fatalf("benchmark capability = %+v ok=%v, want experimental Gemma4 q4 benchmark labels with production prefill/decode pending", benchmark, ok)
	}
	if !strings.Contains(benchmark.Detail, "production native prefill/decode remain pending") {
		t.Fatalf("benchmark detail = %q, want production prefill/decode caveat", benchmark.Detail)
	}
	if !strings.Contains(benchmark.Detail, "retained-state 10-turn book gate") ||
		!strings.Contains(benchmark.Detail, "prompt replay forbidden") {
		t.Fatalf("benchmark detail = %q, want retained-state book gate with prompt replay forbidden", benchmark.Detail)
	}
	if !strings.Contains(benchmark.Detail, "MLX affine 4/6/8-bit") {
		t.Fatalf("benchmark detail = %q, want bit-aware MLX affine detail", benchmark.Detail)
	}
	if benchmark.Labels["engine_state_context_route_contract"] != ROCmStateContextRegistryContract ||
		benchmark.Labels["engine_state_context_prompt_replay_refused"] != "true" ||
		benchmark.Labels["engine_state_context_runtime_owned_kv"] != "true" ||
		benchmark.Labels["engine_lora_route_contract"] != ROCmLoRAAdapterRegistryContract ||
		benchmark.Labels["engine_lora_target_policy"] != "gemma4" ||
		benchmark.Labels["engine_attached_drafter_route_contract"] != ROCmAttachedDrafterRegistryContract ||
		benchmark.Labels["engine_attached_drafter_role"] != "target" ||
		benchmark.Labels["engine_attached_drafter_native_attachment"] != hipKernelStatusNotLinked ||
		benchmark.Labels["engine_attached_drafter_retained_state_required"] != "true" ||
		benchmark.Labels["engine_attached_drafter_prompt_replay_fallback"] != "forbidden" {
		t.Fatalf("benchmark route labels = %+v, want Gemma4 registry route labels", benchmark.Labels)
	}
	assertCSVLabelContainsAll(t, "production_book_gate_metrics", benchmark.Labels["production_book_gate_metrics"], productionBookGateMetrics)
	assertCSVLabelContainsAll(t, "production_book_retained_route_metrics", benchmark.Labels["production_book_retained_route_metrics"], productionBookRetainedRouteMetrics)
	assertCSVLabelContainsAll(t, "production_book_retained_artifact_labels", benchmark.Labels["production_book_retained_artifact_labels"], productionBookRetainedArtifactLabels)
	for _, metric := range DefaultProductionQuantizationPolicy().RequiredBenchmarkMetrics {
		if !strings.Contains(benchmark.Labels["production_book_required_metrics"], metric) {
			t.Fatalf("benchmark required metrics = %q, missing %q", benchmark.Labels["production_book_required_metrics"], metric)
		}
	}
	assertCSVLabelContainsAll(t, "production_mtp_required_metrics", benchmark.Labels["production_mtp_required_metrics"], defaultProductionMTPRequiredMetrics)
	classify, ok := report.Capability(inference.CapabilityClassify)
	if !ok || classify.Status != inference.CapabilityStatusExperimental ||
		classify.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_classify" ||
		classify.Labels["classify_kernel"] != hipKernelStatusLinked ||
		classify.Labels["classify_name"] != "rocm_gemma4_q4_classify_experimental" ||
		classify.Labels["classify_logits_source"] != "gemma4_mlx_affine_package_prefill" ||
		classify.Labels["attention_kv_mode"] != rocmKVCacheModeKQ8VQ4 ||
		classify.Labels["production_prefill"] != hipKernelStatusNotLinked ||
		classify.Labels["production_decode"] != hipKernelStatusNotLinked ||
		classify.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked ||
		classify.Labels["runtime_status"] != string(inference.FeatureRuntimeExperimental) {
		t.Fatalf("classify capability = %+v ok=%v, want experimental Gemma4 q4 classify labels with production prefill pending", classify, ok)
	}
	if !strings.Contains(classify.Detail, "production native prefill remains pending") {
		t.Fatalf("classify detail = %q, want production prefill caveat", classify.Detail)
	}
	logitProbe, ok := report.Capability(inference.CapabilityLogitProbe)
	if !ok || logitProbe.Status != inference.CapabilityStatusExperimental ||
		logitProbe.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_logit_probe" ||
		logitProbe.Labels["logit_probe_kernel"] != hipKernelStatusLinked ||
		logitProbe.Labels["logit_probe_affine_source"] != "gemma4_mlx_affine_classify_logits" ||
		logitProbe.Labels["logit_probe_source"] != "gemma4_q4_classify_logits" ||
		logitProbe.Labels["classify_logits_source"] != "gemma4_mlx_affine_package_prefill" ||
		logitProbe.Labels["attention_kv_mode"] != rocmKVCacheModeKQ8VQ4 ||
		logitProbe.Labels["production_prefill"] != hipKernelStatusNotLinked ||
		logitProbe.Labels["production_decode"] != hipKernelStatusNotLinked ||
		logitProbe.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked ||
		logitProbe.Labels["runtime_status"] != string(inference.FeatureRuntimeExperimental) {
		t.Fatalf("logit probe capability = %+v ok=%v, want experimental Gemma4 q4 classify-logit probe labels with production prefill pending", logitProbe, ok)
	}
	if !strings.Contains(logitProbe.Detail, "Gemma4 MLX affine 4/6/8-bit classification logits") {
		t.Fatalf("logit probe detail = %q, want MLX affine classify-logit source", logitProbe.Detail)
	}
	speculative, ok := report.Capability(inference.CapabilitySpeculativeDecode)
	if !ok || speculative.Status != inference.CapabilityStatusExperimental ||
		speculative.Labels["attached_drafter_helper"] != hipKernelStatusLinked ||
		speculative.Labels["attached_drafter_native_attachment"] != hipKernelStatusNotLinked ||
		speculative.Labels["attached_drafter_role"] != "gemma4_assistant" ||
		speculative.Labels["attached_drafter_retained_state_entrypoint"] != hipKernelStatusLinked ||
		speculative.Labels["attached_drafter_retained_state_required"] != "true" ||
		speculative.Labels["attached_drafter_state_source"] != "rocm_state_session_runtime_kv" ||
		speculative.Labels["attached_drafter_prompt_replay_fallback"] != "forbidden" ||
		speculative.Labels["engine_attached_drafter_route_contract"] != ROCmAttachedDrafterRegistryContract ||
		speculative.Labels["engine_attached_drafter_native_attachment"] != hipKernelStatusNotLinked ||
		speculative.Labels["engine_attached_drafter_retained_state_required"] != "true" ||
		speculative.Labels["engine_attached_drafter_state_source"] != "rocm_state_session_runtime_kv" ||
		speculative.Labels["engine_attached_drafter_prompt_replay_fallback"] != "forbidden" ||
		speculative.Labels["engine_attached_drafter_assistant_architecture"] != officialGemma4E2BAssistantArchitecture ||
		speculative.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_speculative_decode" ||
		speculative.Labels["speculative_decode_helper"] != hipKernelStatusLinked ||
		speculative.Labels["speculative_decode_affine_source"] != "gemma4_mlx_affine_generate" ||
		speculative.Labels["speculative_decode_source"] != "gemma4_q4_generate" ||
		speculative.Labels["gemma4_q4_decode_kernel"] != hipKernelStatusLinked ||
		speculative.Labels["production_prefill"] != hipKernelStatusNotLinked ||
		speculative.Labels["production_decode"] != hipKernelStatusNotLinked ||
		speculative.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked ||
		speculative.Labels["runtime_status"] != string(inference.FeatureRuntimeExperimental) {
		t.Fatalf("speculative capability = %+v ok=%v, want experimental Gemma4 q4 helper labels with production prefill/decode pending", speculative, ok)
	}
	if !strings.Contains(speculative.Detail, "native HIP drafter attachment") ||
		!strings.Contains(speculative.Detail, "production native prefill/decode remain pending") {
		t.Fatalf("speculative detail = %q, want attached-drafter and production prefill/decode caveats", speculative.Detail)
	}
	if !strings.Contains(speculative.Detail, "MLX affine 4/6/8-bit") {
		t.Fatalf("speculative detail = %q, want bit-aware MLX affine source", speculative.Detail)
	}
	for _, id := range []inference.CapabilityID{inference.CapabilityStateBundle, inference.CapabilityStateWake, inference.CapabilityStateSleep, inference.CapabilityStateFork} {
		stateCapability, ok := report.Capability(id)
		if !ok ||
			stateCapability.Labels["engine_state_context_route_contract"] != ROCmStateContextRegistryContract ||
			stateCapability.Labels["engine_state_context_window"] != "131072" ||
			stateCapability.Labels["engine_state_context_prompt_replay_refused"] != "true" ||
			stateCapability.Labels["engine_state_context_remaining_context_default"] != "true" ||
			stateCapability.Labels["engine_state_context_runtime_owned_kv"] != "true" ||
			stateCapability.Labels["engine_state_context_gemma4_size"] != "E2B" ||
			stateCapability.Labels["engine_state_context_gemma4_quant_mode"] != "q4" {
			t.Fatalf("state capability %s = %+v ok=%v, want Gemma4 state/context route labels", id, stateCapability, ok)
		}
	}
	tokenizerRouteCapability, ok := report.Capability(inference.CapabilityTokenizer)
	if !ok ||
		tokenizerRouteCapability.Labels["engine_tokenizer_route_contract"] != ROCmModelTokenizerRegistryContract ||
		tokenizerRouteCapability.Labels["engine_tokenizer_kind"] != "GemmaTokenizer" ||
		tokenizerRouteCapability.Labels["engine_tokenizer_chat_template_id"] != "gemma4_hf_turn" ||
		tokenizerRouteCapability.Labels["engine_tokenizer_generation_role"] != "model" {
		t.Fatalf("tokenizer capability = %+v ok=%v, want Gemma4 tokenizer route labels", tokenizerRouteCapability, ok)
	}
	for _, id := range []inference.CapabilityID{inference.CapabilityLoRAInference, inference.CapabilityLoRATraining, inference.CapabilityModelMerge} {
		loraRouteCapability, ok := report.Capability(id)
		if !ok ||
			loraRouteCapability.Labels["engine_lora_route_contract"] != ROCmLoRAAdapterRegistryContract ||
			loraRouteCapability.Labels["engine_lora_target_policy"] != "gemma4" ||
			loraRouteCapability.Labels["engine_lora_default_targets"] != "q_proj,v_proj,o_proj" ||
			loraRouteCapability.Labels["engine_lora_safe_targets"] != "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" ||
			loraRouteCapability.Labels["engine_lora_extended_targets"] != "router.proj,per_layer_input_gate,per_layer_projection" ||
			loraRouteCapability.Labels["engine_lora_extended_targets_require_opt"] != "true" ||
			loraRouteCapability.Labels["engine_lora_apply_supported"] != "true" ||
			loraRouteCapability.Labels["engine_lora_training_supported"] != "true" ||
			!strings.Contains(loraRouteCapability.Labels["engine_lora_capabilities"], string(inference.CapabilityModelMerge)) ||
			!strings.Contains(loraRouteCapability.Labels["engine_lora_target_paths"], "q_proj=self_attn.q_proj") {
			t.Fatalf("LoRA route capability %s = %+v ok=%v, want Gemma4 adapter route labels", id, loraRouteCapability, ok)
		}
	}
	promptLookup, ok := report.Capability(inference.CapabilityPromptLookupDecode)
	if !ok || promptLookup.Status != inference.CapabilityStatusExperimental ||
		promptLookup.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_prompt_lookup_decode" ||
		promptLookup.Labels["prompt_lookup_decode_helper"] != hipKernelStatusLinked ||
		promptLookup.Labels["prompt_lookup_decode_affine_source"] != "gemma4_mlx_affine_generate" ||
		promptLookup.Labels["prompt_lookup_decode_source"] != "gemma4_q4_generate" ||
		promptLookup.Labels["gemma4_q4_decode_kernel"] != hipKernelStatusLinked ||
		promptLookup.Labels["production_prefill"] != hipKernelStatusNotLinked ||
		promptLookup.Labels["production_decode"] != hipKernelStatusNotLinked ||
		promptLookup.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked ||
		promptLookup.Labels["runtime_status"] != string(inference.FeatureRuntimeExperimental) {
		t.Fatalf("prompt lookup capability = %+v ok=%v, want experimental Gemma4 q4 helper labels with production prefill/decode pending", promptLookup, ok)
	}
	if !strings.Contains(promptLookup.Detail, "production native prefill/decode remain pending") {
		t.Fatalf("prompt lookup detail = %q, want production prefill/decode caveat", promptLookup.Detail)
	}
	if !strings.Contains(promptLookup.Detail, "MLX affine 4/6/8-bit") {
		t.Fatalf("prompt lookup detail = %q, want bit-aware MLX affine source", promptLookup.Detail)
	}
}

func TestNativeContract_RocmGemma4Q6CapabilityLabels_Good(t *testing.T) {
	report := rocmCapabilityReport(nativeDeviceInfo{}, inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-e2b-it-6bit",
		Architecture: "gemma4_text",
		VocabSize:    262144,
		NumLayers:    35,
		HiddenSize:   1536,
		QuantBits:    6,
		QuantGroup:   64,
	}, inference.AdapterIdentity{}, true, defaultHIPKernelStatus(), rocmCapabilityReportOption{Gemma4Q4GenerateLinked: true})

	generate, ok := report.Capability(inference.CapabilityGenerate)
	if !ok || generate.Labels["decode_quant"] != "mlx_q6" ||
		generate.Labels["gemma4_mlx_affine_bits"] != "6" ||
		generate.Labels["gemma4_size"] != "E2B" ||
		generate.Labels["gemma4_quant_mode"] != "q6" ||
		generate.Labels["gemma4_runtime"] != Gemma4RuntimeMLXAffine ||
		generate.Labels["gemma4_generate_status"] != Gemma4GenerateLinked ||
		generate.Labels["quant_default_tier"] != "q6" ||
		generate.Labels["quant_family"] != "mlx_affine" ||
		generate.Labels["quant_ladder"] != "bf16,q8,q6,q4" ||
		generate.Labels["production_quant_tier"] != "default" ||
		generate.Labels["production_quant_product_default"] != "true" ||
		generate.Labels["production_quant_model"] != ProductionLaneCurrentModelID ||
		generate.Labels["production_quant_min_visible_tokens_per_sec"] != "100" ||
		generate.Labels["production_quant_runnable_pack_count"] != "14" ||
		!strings.Contains(generate.Labels["production_quant_load_only_packs"], "E4B:bf16") ||
		!strings.Contains(generate.Labels["production_quant_planned_packs"], "E2B:mxfp8") ||
		!strings.Contains(generate.Labels["production_quant_planned_packs"], "E4B:mxfp8") {
		t.Fatalf("generate capability = %+v ok=%v, want Gemma4 q6 MLX affine labels", generate, ok)
	}
	if !strings.Contains(generate.Detail, "MLX affine 4/6/8-bit") {
		t.Fatalf("generate detail = %q, want bit-aware MLX affine detail", generate.Detail)
	}
	chat, ok := report.Capability(inference.CapabilityChat)
	if !ok || chat.Labels["decode_quant"] != "mlx_q6" || chat.Labels["chat_template"] != "gemma4_hf_turn" {
		t.Fatalf("chat capability = %+v ok=%v, want q6 Gemma4 template labels", chat, ok)
	}
	benchmark, ok := report.Capability(inference.CapabilityBenchmark)
	if !ok ||
		benchmark.Labels["decode_quant"] != "mlx_q6" ||
		benchmark.Labels["benchmark_retained_state_book"] != "BenchmarkInferenceGemma4Q4Book10Turn_RetainedState" ||
		benchmark.Labels["benchmark_prompt_replay_fallback"] != "forbidden" ||
		benchmark.Labels["production_book_policy"] != "retained_state_required" ||
		benchmark.Labels["production_book_decision_source"] != "benchmark_metrics" ||
		benchmark.Labels["production_book_gate_raw_decode_tokens_per_sec"] != "100" ||
		benchmark.Labels["production_book_gate_wall_seconds"] != strconv.Itoa(ProductionLaneBookWallSeconds) ||
		benchmark.Labels["production_book_gate_metrics"] == "" ||
		benchmark.Labels["production_book_gate_reason_codes"] != productionBookGateReasonCodesLabel ||
		benchmark.Labels["production_book_retained_route_metrics"] == "" ||
		benchmark.Labels["production_book_retained_artifact_labels"] == "" ||
		benchmark.Labels["production_book_required_metrics"] == "" ||
		benchmark.Labels["production_model_source"] != "model_identity_or_pack" ||
		benchmark.Labels["production_quant_decision_source"] != "gemma4_family_matrix" {
		t.Fatalf("benchmark capability = %+v ok=%v, want q6 retained-state production book labels", benchmark, ok)
	}
	if benchmark.Labels["engine_state_context_route_contract"] != ROCmStateContextRegistryContract ||
		benchmark.Labels["engine_lora_route_contract"] != ROCmLoRAAdapterRegistryContract ||
		benchmark.Labels["engine_attached_drafter_route_contract"] != ROCmAttachedDrafterRegistryContract ||
		benchmark.Labels["engine_attached_drafter_role"] != "target" {
		t.Fatalf("benchmark route labels = %+v, want q6 Gemma4 registry route labels", benchmark.Labels)
	}
	assertCSVLabelContainsAll(t, "production_book_gate_metrics", benchmark.Labels["production_book_gate_metrics"], productionBookGateMetrics)
	assertCSVLabelContainsAll(t, "production_book_retained_route_metrics", benchmark.Labels["production_book_retained_route_metrics"], productionBookRetainedRouteMetrics)
	assertCSVLabelContainsAll(t, "production_book_retained_artifact_labels", benchmark.Labels["production_book_retained_artifact_labels"], productionBookRetainedArtifactLabels)
}

func TestNativeContract_RocmGemma4E4BQ6CapabilityLabels_Good(t *testing.T) {
	report := rocmCapabilityReport(nativeDeviceInfo{}, inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		Architecture: "gemma4_text",
		VocabSize:    262144,
		NumLayers:    26,
		HiddenSize:   2304,
		QuantBits:    6,
		QuantGroup:   64,
	}, inference.AdapterIdentity{}, true, defaultHIPKernelStatus(), rocmCapabilityReportOption{Gemma4Q4GenerateLinked: true})

	generate, ok := report.Capability(inference.CapabilityGenerate)
	if !ok ||
		generate.Labels["decode_quant"] != "mlx_q6" ||
		generate.Labels["gemma4_size"] != "E4B" ||
		generate.Labels["gemma4_quant_mode"] != "q6" ||
		generate.Labels["gemma4_pack_supported"] != "true" ||
		generate.Labels["gemma4_runtime"] != Gemma4RuntimeMLXAffine ||
		generate.Labels["gemma4_generate_status"] != Gemma4GenerateLinked ||
		generate.Labels["gemma4_runnable_on_card"] != "true" ||
		generate.Labels["decode_layers"] != "26" ||
		generate.Labels["decode_hidden_size"] != "2304" {
		t.Fatalf("generate capability = %+v ok=%v, want Gemma4 E4B q6 size/quant labels from path metadata", generate, ok)
	}
}

func TestNativeContract_RocmGemma4CapabilityLabelsInferPathQuant_Good(t *testing.T) {
	report := rocmCapabilityReport(nativeDeviceInfo{}, inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		Architecture: "gemma4_text",
		VocabSize:    262144,
		NumLayers:    26,
		HiddenSize:   2304,
	}, inference.AdapterIdentity{}, true, defaultHIPKernelStatus(), rocmCapabilityReportOption{Gemma4Q4GenerateLinked: true})

	if report.Model.QuantType != "q6" || report.Model.QuantBits != 6 {
		t.Fatalf("report model = %+v, want path-inferred q6 identity", report.Model)
	}
	generate, ok := report.Capability(inference.CapabilityGenerate)
	if !ok ||
		generate.Labels["decode_quant"] != "mlx_q6" ||
		generate.Labels["gemma4_mlx_affine_bits"] != "6" ||
		generate.Labels["gemma4_size"] != "E4B" ||
		generate.Labels["gemma4_quant_mode"] != "q6" ||
		generate.Labels["gemma4_generate_status"] != Gemma4GenerateLinked {
		t.Fatalf("generate capability = %+v ok=%v, want path-inferred E4B q6 labels", generate, ok)
	}
}

func TestNativeContract_RocmGemma4TwelveBQ6CapabilityLabels_Good(t *testing.T) {
	report := rocmCapabilityReport(nativeDeviceInfo{}, inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-12b-it-6bit",
		Architecture: "gemma4_text",
		VocabSize:    262144,
		NumLayers:    48,
		HiddenSize:   3840,
		QuantBits:    6,
		QuantGroup:   64,
	}, inference.AdapterIdentity{}, true, defaultHIPKernelStatus(), rocmCapabilityReportOption{Gemma4Q4GenerateLinked: true})

	generate, ok := report.Capability(inference.CapabilityGenerate)
	if !ok ||
		generate.Labels["decode_quant"] != "mlx_q6" ||
		generate.Labels["gemma4_size"] != "12B" ||
		generate.Labels["gemma4_quant_mode"] != "q6" ||
		generate.Labels["gemma4_pack_supported"] != "true" ||
		generate.Labels["gemma4_runtime"] != Gemma4RuntimeMLXAffine ||
		generate.Labels["gemma4_generate_status"] != Gemma4GenerateLinked ||
		generate.Labels["gemma4_runnable_on_card"] != "true" ||
		generate.Labels["decode_layers"] != "48" ||
		generate.Labels["decode_hidden_size"] != "3840" {
		t.Fatalf("generate capability = %+v ok=%v, want Gemma4 12B q6 size/quant labels", generate, ok)
	}
	benchmark, ok := report.Capability(inference.CapabilityBenchmark)
	if !ok ||
		benchmark.Labels["attached_drafter_target_gemma4_size"] != "12B" ||
		benchmark.Labels["attached_drafter_target_gemma4_quant_mode"] != "q6" ||
		benchmark.Labels["attached_drafter_target_gemma4_quant_group"] != "64" ||
		benchmark.Labels["attached_drafter_assistant_gemma4_size"] != "12B" ||
		benchmark.Labels["attached_drafter_assistant_gemma4_quant_mode"] != "bf16" ||
		benchmark.Labels["attached_drafter_official_pair_verified"] != "false" {
		t.Fatalf("benchmark capability = %+v ok=%v, want non-official 12B MTP pair labels", benchmark, ok)
	}
}

func TestNativeContract_RocmGemma4Unified12BQ4ExposesLinkedCapability_Good(t *testing.T) {
	report := rocmCapabilityReport(nativeDeviceInfo{}, inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-12b-it-4bit",
		Architecture: "gemma4_text",
		VocabSize:    262144,
		NumLayers:    48,
		HiddenSize:   3840,
		QuantBits:    4,
		QuantGroup:   64,
	}, inference.AdapterIdentity{}, true, defaultHIPKernelStatus(), rocmCapabilityReportOption{Gemma4Q4GenerateLinked: true})

	generate, ok := report.Capability(inference.CapabilityGenerate)
	if !ok || generate.Status != inference.CapabilityStatusExperimental ||
		generate.Labels["gemma4_size"] != "12B" ||
		generate.Labels["gemma4_quant_mode"] != "q4" ||
		generate.Labels["gemma4_pack_supported"] != "true" ||
		generate.Labels["gemma4_runtime"] != Gemma4RuntimeMLXAffine ||
		generate.Labels["gemma4_generate_status"] != Gemma4GenerateLinked ||
		generate.Labels["gemma4_runnable_on_card"] != "true" ||
		generate.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_generate" {
		t.Fatalf("generate capability = %+v ok=%v, want linked Gemma4 12B q4 generation", generate, ok)
	}
	chat, ok := report.Capability(inference.CapabilityChat)
	if !ok || chat.Status != inference.CapabilityStatusExperimental ||
		chat.Labels["gemma4_size"] != "12B" ||
		chat.Labels["gemma4_quant_mode"] != "q4" ||
		chat.Labels["gemma4_pack_supported"] != "true" ||
		chat.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_chat" {
		t.Fatalf("chat capability = %+v ok=%v, want linked Gemma4 12B q4 chat", chat, ok)
	}
	modelLoad, ok := report.Capability(inference.CapabilityModelLoad)
	if !ok ||
		modelLoad.Labels["gemma4_size"] != "12B" ||
		modelLoad.Labels["gemma4_quant_mode"] != "q4" ||
		modelLoad.Labels["gemma4_pack_supported"] != "true" {
		t.Fatalf("model-load capability = %+v ok=%v, want supported Gemma4 12B q4 labels", modelLoad, ok)
	}
	chatTemplate, ok := report.Capability(inference.CapabilityChatTemplate)
	if !ok ||
		chatTemplate.Labels["chat_template"] != "gemma4_hf_turn" ||
		chatTemplate.Labels["gemma4_size"] != "12B" ||
		chatTemplate.Labels["gemma4_quant_mode"] != "q4" ||
		chatTemplate.Labels["gemma4_pack_supported"] != "true" ||
		chatTemplate.Labels["engine_tokenizer_route_contract"] != ROCmModelTokenizerRegistryContract ||
		chatTemplate.Labels["engine_tokenizer_chat_template_id"] != "gemma4_hf_turn" {
		t.Fatalf("chat-template capability = %+v ok=%v, want Gemma4 12B q4 template labels", chatTemplate, ok)
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
			capability.Labels["gemma4_size"] != "12B" ||
			capability.Labels["gemma4_quant_mode"] != "q4" ||
			capability.Labels["gemma4_pack_supported"] != "true" {
			t.Fatalf("capability %s = %+v ok=%v, want supported Gemma4 12B q4 labels", id, capability, ok)
		}
	}
}

func TestNativeContract_RocmGemma4LargestPacksStatusOnly_Bad(t *testing.T) {
	for _, tc := range []struct {
		name   string
		size   string
		path   string
		labels map[string]string
	}{
		{name: "26b-a4b", size: "26B-A4B", path: "gemma-4-26b-a4b-it-6bit"},
		{name: "31b", size: "31B", path: "gemma-4-31b-it-6bit"},
		{name: "31b-carried-labels", size: "31B", path: "generic-local-pack", labels: map[string]string{
			"gemma4_size":       "31b",
			"gemma4_quant_mode": "Q6",
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			report := rocmCapabilityReport(nativeDeviceInfo{}, inference.ModelIdentity{
				Architecture: "gemma4_text",
				Path:         tc.path,
				Labels:       tc.labels,
				VocabSize:    262144,
				NumLayers:    64,
				HiddenSize:   4096,
				QuantBits:    6,
				QuantGroup:   64,
			}, inference.AdapterIdentity{}, true, defaultHIPKernelStatus(), rocmCapabilityReportOption{Gemma4Q4GenerateLinked: true})

			generate, ok := report.Capability(inference.CapabilityGenerate)
			if !ok ||
				generate.Status == inference.CapabilityStatusExperimental ||
				generate.Labels["gemma4_size"] != tc.size ||
				generate.Labels["gemma4_quant_mode"] != "q6-status" ||
				generate.Labels["gemma4_pack_supported"] != "true" ||
				generate.Labels["gemma4_runtime"] != Gemma4RuntimePlanned ||
				generate.Labels["gemma4_generate_status"] != Gemma4GeneratePlannedOnly ||
				generate.Labels["gemma4_runnable_on_card"] != "false" ||
				generate.Labels["kernel_scope"] == "loaded_gemma4_q4_experimental_generate" {
				t.Fatalf("generate capability = %+v ok=%v, want %s q6 status-only planned labels", generate, ok, tc.size)
			}
			if chat, ok := report.Capability(inference.CapabilityChat); !ok ||
				chat.Status == inference.CapabilityStatusExperimental ||
				chat.Labels["gemma4_size"] != tc.size ||
				chat.Labels["gemma4_quant_mode"] != "q6-status" ||
				chat.Labels["gemma4_generate_status"] != Gemma4GeneratePlannedOnly ||
				chat.Labels["kernel_scope"] == "loaded_gemma4_q4_experimental_chat" {
				t.Fatalf("chat capability = %+v ok=%v, want %s q6 status-only planned labels", chat, ok, tc.size)
			}
		})
	}
}

var nativeContractGemma4BenchmarkCapabilityLabelsSink map[string]string
var nativeContractQuantizationCapabilityLabelsSink map[string]string

func BenchmarkNativeContract_RocmGemma4Q6BenchmarkCapabilityLabels(b *testing.B) {
	model := inference.ModelIdentity{
		Architecture: "gemma4_text",
		VocabSize:    262144,
		NumLayers:    35,
		HiddenSize:   1536,
		QuantBits:    6,
		QuantGroup:   64,
	}
	b.ReportAllocs()
	for b.Loop() {
		nativeContractGemma4BenchmarkCapabilityLabelsSink = rocmGemma4Q4BenchmarkCapabilityLabels(model)
	}
}

func BenchmarkNativeContract_RocmQuantizationCapabilityLabels(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		nativeContractQuantizationCapabilityLabelsSink = rocmQuantizationCapabilityLabels()
	}
}

func BenchmarkNativeContract_RocmQuantizationCapabilityLabelsApply(b *testing.B) {
	labels := make(map[string]string, 32)
	b.ReportAllocs()
	for b.Loop() {
		clear(labels)
		rocmApplyQuantizationCapabilityLabels(labels)
	}
	if labels["autoround_calibration_evidence_helper"] != "ApplyProductionAutoRoundCalibrationLabelEvidence" ||
		labels["autoround_calibration_validator"] != "ValidateProductionAutoRoundCalibrationLabels" ||
		labels["autoround_calibration_decision_validator"] != "ValidateProductionAutoRoundCalibrationDecisionLabels" ||
		labels["autoround_calibration_evidence_decision_validator"] != "ValidateProductionAutoRoundCalibrationEvidenceDecisionLabels" ||
		labels["production_required_metrics"] != defaultProductionTurboQuantRequiredMetricsLabel {
		b.Fatalf("labels = %+v, want quantization capability labels", labels)
	}
}

func TestNativeContract_RocmModelCapabilitiesUseEmbeddingRerankKernelStatus_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "bert",
		modelInfo: inference.ModelInfo{Architecture: "bert", HiddenSize: 2, VocabSize: 3, QuantBits: 32},
		native: &fakeNativeEmbeddingModel{fakeNativeModel: &fakeNativeModel{kernelStatus: hipKernelStatus{
			Embedding: hipKernelStatusLinked,
			Rerank:    hipKernelStatusLinked,
			KVCache:   hipKernelStatusPlanned,
			Reason:    "fake embedding/rerank fixture",
		}}},
	}

	report := model.Capabilities()

	if report.Labels["embedding_kernel"] != hipKernelStatusLinked || report.Labels["rerank_kernel"] != hipKernelStatusLinked {
		t.Fatalf("labels = %+v, want linked embedding/rerank status", report.Labels)
	}
	for _, id := range []inference.CapabilityID{inference.CapabilityEmbeddings, inference.CapabilityRerank} {
		capability, ok := report.Capability(id)
		if !ok || capability.Status != inference.CapabilityStatusExperimental || capability.Labels["runtime_status"] != string(inference.FeatureRuntimeExperimental) {
			t.Fatalf("capability %s = %+v ok=%v, want experimental model-level kernel fixture", id, capability, ok)
		}
	}
	embedding, ok := report.Capability(inference.CapabilityEmbeddings)
	if !ok || embedding.Labels["kernel_name"] != hipKernelNameEmbedMean ||
		embedding.Labels["embedding_kernel"] != hipKernelStatusLinked ||
		embedding.Labels["embedding_kernel_name"] != hipKernelNameEmbedMean ||
		embedding.Labels["kernel_scope"] != "loaded_embedding_fixtures" ||
		embedding.Labels["supported_embedding_scopes"] != "tiny_token_embeddings,bert_word_embeddings" ||
		embedding.Labels["production_embedding_models"] != hipKernelStatusNotLinked {
		t.Fatalf("embedding capability = %+v ok=%v, want loaded embedding fixture labels with production pending", embedding, ok)
	}
	rerank, ok := report.Capability(inference.CapabilityRerank)
	if !ok || rerank.Labels["kernel_name"] != hipKernelNameRerank ||
		rerank.Labels["rerank_kernel"] != hipKernelStatusLinked ||
		rerank.Labels["rerank_kernel_name"] != hipKernelNameRerank ||
		rerank.Labels["embedding_kernel"] != hipKernelStatusLinked ||
		rerank.Labels["embedding_kernel_name"] != hipKernelNameEmbedMean ||
		rerank.Labels["kernel_scope"] != "loaded_rerank_fixtures" ||
		rerank.Labels["supported_rerank_scopes"] != "embedding_cosine,bert_sequence_classifier" ||
		rerank.Labels["production_rerank_models"] != hipKernelStatusNotLinked {
		t.Fatalf("rerank capability = %+v ok=%v, want loaded rerank fixture labels with production pending", rerank, ok)
	}
}

func TestNativeContract_RocmModelCapabilitiesUseLoRAKernelStatus_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", HiddenSize: 2, VocabSize: 3, QuantBits: 32},
		native: &fakeNativeModel{kernelStatus: hipKernelStatus{
			LoRA:    hipKernelStatusLinked,
			KVCache: hipKernelStatusPlanned,
			Reason:  "fake tiny LoRA fixture",
		}},
	}

	report := model.Capabilities()

	if report.Labels["lora_kernel"] != hipKernelStatusLinked {
		t.Fatalf("labels = %+v, want linked LoRA status", report.Labels)
	}
	capability, ok := report.Capability(inference.CapabilityLoRAInference)
	if !ok || capability.Status != inference.CapabilityStatusExperimental ||
		capability.Labels["runtime_status"] != string(inference.FeatureRuntimeExperimental) ||
		capability.Labels["kernel_name"] != hipKernelNameLoRA ||
		capability.Labels["lora_kernel"] != hipKernelStatusLinked ||
		capability.Labels["kernel_scope"] != "loaded_adapter_fixtures" ||
		capability.Labels["supported_adapter_scopes"] != "tiny_output_head,qwen_gemma_dense_small_lm_head,bert_sequence_classifier" ||
		capability.Labels["production_adapter_application"] != hipKernelStatusNotLinked {
		t.Fatalf("LoRA capability = %+v ok=%v, want experimental tiny LoRA kernel fixture", capability, ok)
	}
}

func TestNativeContract_RocmModelEmbeddingsAndRerankDispatch_Good(t *testing.T) {
	native := &fakeNativeEmbeddingModel{fakeNativeModel: &fakeNativeModel{}}
	model := &rocmModel{
		modelType: "bert",
		modelInfo: inference.ModelInfo{Architecture: "bert", HiddenSize: 2, VocabSize: 3, QuantBits: 32},
		native:    native,
	}

	embedded, err := model.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"core"}, Normalize: true})
	core.RequireNoError(t, err)
	if embedded.Model.Architecture != "bert" || len(embedded.Vectors) != 1 || embedded.Labels["backend"] != "fake" {
		t.Fatalf("embedding result = %+v, want ROCm model identity and native labels", embedded)
	}
	reranked, err := model.Rerank(context.Background(), inference.RerankRequest{Query: "core", Documents: []string{"a", "b"}, TopN: 1})
	core.RequireNoError(t, err)
	if reranked.Model.Architecture != "bert" || len(reranked.Results) != 1 || reranked.Results[0].Index != 1 {
		t.Fatalf("rerank result = %+v, want native top result and ROCm model identity", reranked)
	}
}

func TestNativeContract_RocmModelEmbeddingsAndRerankNotLinked_Bad(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{kernelStatus: hipKernelStatus{
		Embedding: hipKernelStatusLinked,
		Rerank:    hipKernelStatusLinked,
	}}}

	_, err := model.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"core"}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native embedding kernels are not linked yet")
	_, err = model.Rerank(context.Background(), inference.RerankRequest{Query: "core", Documents: []string{"doc"}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native rerank kernels are not linked yet")
	if report := model.Capabilities(); report.Supports(inference.CapabilityEmbeddings) || report.Supports(inference.CapabilityRerank) {
		t.Fatalf("capabilities = %+v, want embedding/rerank planned without native optional methods", report.CapabilityIDs())
	}
}

func TestNativeContract_RocmModelEmbeddingsAndRerankPreflightBeforeNotLinked_Bad(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{kernelStatus: hipKernelStatus{
		Embedding: hipKernelStatusLinked,
		Rerank:    hipKernelStatusLinked,
	}}}

	_, err := model.Embed(context.Background(), inference.EmbeddingRequest{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input text is required")

	_, err = model.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"core", " "}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input 1 is empty")

	_, err = model.Rerank(context.Background(), inference.RerankRequest{Documents: []string{"doc"}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "query is required")

	_, err = model.Rerank(context.Background(), inference.RerankRequest{Query: "core"})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "documents are required")

	_, err = model.Rerank(context.Background(), inference.RerankRequest{Query: "core", Documents: []string{"doc", ""}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "document 1 is empty")
}

func TestNativeContract_LoadModelUsesNativeRuntimeWithoutServer_Good(t *testing.T) {
	runtime := &fakeNativeRuntime{
		available: true,
		model:     &fakeNativeModel{tokens: []inference.Token{{ID: 17, Text: "ok"}}},
	}
	backend := newROCmBackendWithRuntime(runtime)
	t.Setenv("PATH", "")
	t.Setenv("ROCM_LLAMA_SERVER_PATH", "")

	model, err := resultValue[inference.TextModel](backend.LoadModel(nativeContractGGUF(t), inference.WithContextLen(8192), inference.WithAdapterPath("adapter.safetensors")))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer model.Close()

	if runtime.loadPath == "" || runtime.loadConfig.ContextSize != 8192 {
		t.Fatalf("native runtime load = path %q config %+v, want direct native load", runtime.loadPath, runtime.loadConfig)
	}
	if runtime.loadConfig.AdapterPath != "adapter.safetensors" {
		t.Fatalf("adapter path = %q, want load-time adapter path forwarded", runtime.loadConfig.AdapterPath)
	}
	if model.ModelType() != "qwen3" {
		t.Fatalf("ModelType = %q, want qwen3", model.ModelType())
	}
}

func TestNativeContract_LoadModelBadAdapterFailureClosesNativeModel_Bad(t *testing.T) {
	native := &fakeNativeModel{adapterErr: core.NewError("adapter failed")}
	runtime := &fakeNativeRuntime{available: true, model: native}
	backend := newROCmBackendWithRuntime(runtime)

	model, err := resultValue[inference.TextModel](backend.LoadModel(nativeContractGGUF(t), inference.WithAdapterPath("adapter.safetensors")))

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "load adapter")
	core.AssertNil(t, model)
	core.AssertEqual(t, 1, native.closeCalls)
	core.AssertEqual(t, []string{"adapter.safetensors"}, native.adapterLoads)
}

func TestNativeContract_LoadModelBadEmptyAdapterPathDoesNotLoadNativeModel_Bad(t *testing.T) {
	runtime := &fakeNativeRuntime{available: true, model: &fakeNativeModel{}}
	backend := newROCmBackendWithRuntime(runtime)

	model, err := resultValue[inference.TextModel](backend.LoadModel(nativeContractGGUF(t), inference.WithAdapterPath(" \t")))

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "adapter path is required")
	core.AssertNil(t, model)
	core.AssertEqual(t, "", runtime.loadPath)
}

func TestNativeContract_LoadModelSafetensorsGemma4UsesNativeRuntime_Good(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures":["Gemma4ForConditionalGeneration"],
		"model_type":"gemma4",
		"tie_word_embeddings":true,
		"quantization_config":{"bits":6,"group_size":64,"mode":"affine"},
		"text_config":{
			"model_type":"gemma4_text",
			"hidden_size":16,
			"num_hidden_layers":1,
			"max_position_embeddings":8192,
			"vocab_size":8
		}
	}`)
	header := `{"language_model.model.embed_tokens.weight":{"dtype":"U32","shape":[8,2],"data_offsets":[0,64]},"language_model.model.layers.0.input_layernorm.weight":{"dtype":"BF16","shape":[16],"data_offsets":[64,96]}}`
	writeNativeContractSafetensorsHeaderWithPayload(t, core.PathJoin(dir, "model.safetensors"), header, 96)
	runtime := &fakeNativeRuntime{
		available: true,
		device:    nativeDeviceInfo{Name: "AMD Radeon RX 7800 XT", MemoryBytes: 16 * memoryGiB, FreeBytes: 12 * memoryGiB, Driver: "hip-test"},
		model:     &fakeNativeModel{},
	}

	model, err := resultValue[inference.TextModel](newROCmBackendWithRuntime(runtime).LoadModel(dir, inference.WithContextLen(128)))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer model.Close()

	if core.PathBase(runtime.loadPath) != "model.safetensors" {
		t.Fatalf("load path = %q, want safetensors weight file", runtime.loadPath)
	}
	if runtime.loadConfig.ModelInfo.Architecture != "gemma4" ||
		runtime.loadConfig.ModelInfo.HiddenSize != 16 ||
		runtime.loadConfig.ModelInfo.VocabSize != 8 ||
		runtime.loadConfig.ModelInfo.NumLayers != 1 ||
		runtime.loadConfig.ModelInfo.QuantBits != 6 ||
		runtime.loadConfig.ModelInfo.QuantGroup != 64 {
		t.Fatalf("load config model = %+v, want Gemma4 text_config identity", runtime.loadConfig.ModelInfo)
	}
	if !runtime.loadConfig.TiedWordEmbeddings || runtime.loadConfig.ContextSize != 128 || len(runtime.loadConfig.Tensors) != 2 {
		t.Fatalf("load config = %+v, want tied Gemma4 safetensors tensor plan", runtime.loadConfig)
	}
	if runtime.loadConfig.DataOffset != int64(8+len(header)) {
		t.Fatalf("data offset = %d, want %d", runtime.loadConfig.DataOffset, 8+len(header))
	}
	for _, tensor := range runtime.loadConfig.Tensors {
		if tensor.SourcePath == "" || tensor.DataOffset != runtime.loadConfig.DataOffset {
			t.Fatalf("tensor = %+v, want safetensors source path and per-tensor data offset", tensor)
		}
	}
}

func TestNativeContract_LoadModelSafetensorsGemma4PropagatesTextRuntimeConfig_Good(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures":["Gemma4ForConditionalGeneration"],
		"model_type":"gemma4",
		"tie_word_embeddings":true,
		"quantization_config":{"bits":6,"group_size":64,"mode":"affine"},
		"text_config":{
			"model_type":"gemma4_text",
			"hidden_size":16,
			"num_hidden_layers":6,
			"num_attention_heads":8,
			"num_key_value_heads":1,
			"num_global_key_value_heads":1,
			"head_dim":512,
			"global_head_dim":1024,
			"attention_k_eq_v":true,
			"num_kv_shared_layers":2,
			"hidden_size_per_layer_input":4,
			"vocab_size_per_layer_input":8,
			"final_logit_softcapping":42.0,
			"use_double_wide_mlp":true,
			"enable_moe_block":true,
			"num_experts":16,
			"top_k_experts":2,
			"moe_intermediate_size":32,
			"max_position_embeddings":131072,
			"sliding_window":1024,
			"sliding_window_pattern":5,
			"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention"],
			"rope_parameters":{
				"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"},
				"full_attention":{"partial_rotary_factor":0.25,"rope_theta":1000000.0,"rope_type":"proportional"}
			},
			"vocab_size":8
		}
	}`)
	writeNativeContractSafetensorsHeaderWithPayload(t, core.PathJoin(dir, "model.safetensors"), `{"language_model.model.embed_tokens.weight":{"dtype":"U32","shape":[8,2],"data_offsets":[0,64]}}`, 64)
	runtime := &fakeNativeRuntime{
		available: true,
		device:    nativeDeviceInfo{Name: "AMD Radeon RX 7800 XT", MemoryBytes: 16 * memoryGiB, FreeBytes: 12 * memoryGiB, Driver: "hip-test"},
		model:     &fakeNativeModel{},
	}

	model, err := resultValue[inference.TextModel](newROCmBackendWithRuntime(runtime).LoadModel(dir))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer model.Close()

	cfg := runtime.loadConfig.Gemma4TextConfig
	core.AssertEqual(t, 6, cfg.NumLayers)
	core.AssertEqual(t, []string{"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "full_attention"}, cfg.LayerTypes)
	core.AssertEqual(t, true, cfg.KVSharedLayersSet)
	core.AssertEqual(t, 2, cfg.KVSharedLayers)
	core.AssertEqual(t, 1024, cfg.SlidingWindow)
	core.AssertEqual(t, 5, cfg.SlidingWindowPattern)
	core.AssertEqual(t, 512, cfg.HeadDim)
	core.AssertEqual(t, 1024, cfg.GlobalHeadDim)
	core.AssertEqual(t, 4, cfg.HiddenSizePerLayerInput)
	core.AssertEqual(t, 8, cfg.VocabSizePerLayerInput)
	core.AssertEqual(t, true, cfg.AttentionKEqV)
	core.AssertEqual(t, float64(42), cfg.FinalLogitSoftcap)
	core.AssertEqual(t, true, cfg.UseDoubleWideMLP)
	core.AssertEqual(t, true, cfg.EnableMoEBlock)
	core.AssertEqual(t, 16, cfg.NumExperts)
	core.AssertEqual(t, 2, cfg.TopKExperts)
	core.AssertEqual(t, 32, cfg.MoEIntermediateSize)
	core.AssertEqual(t, float64(10000), cfg.RoPEParameters["sliding_attention"].RopeTheta)
	core.AssertEqual(t, float64(1000000), cfg.RoPEParameters["full_attention"].RopeTheta)
	core.AssertEqual(t, float64(0.25), cfg.RoPEParameters["full_attention"].PartialRotaryFactor)
	core.AssertEqual(t, float64(1), cfg.RoPEParameters["full_attention"].Factor)
	if runtime.loadConfig.ModelLabels["attention_layer_types"] == "" ||
		runtime.loadConfig.ModelLabels["sliding_window"] != "1024" ||
		runtime.loadConfig.ModelLabels["gemma4_sliding_window"] != "1024" ||
		runtime.loadConfig.ModelLabels["sliding_window_pattern"] != "5" ||
		runtime.loadConfig.ModelLabels["gemma4_sliding_window_pattern"] != "5" ||
		runtime.loadConfig.ModelLabels["attention_kv_shared_layers"] != "2" ||
		runtime.loadConfig.ModelLabels["gemma4_attention_kv_shared_layers"] != "2" ||
		runtime.loadConfig.ModelLabels["attention_layer_count"] != "6" ||
		runtime.loadConfig.ModelLabels["gemma4_attention_layer_count"] != "6" ||
		runtime.loadConfig.ModelLabels["attention_cache_owner_by_layer"] != "0,1,2,3,4,4" ||
		runtime.loadConfig.ModelLabels["attention_cache_index_by_layer"] != "0,1,2,3,4,-1" ||
		runtime.loadConfig.ModelLabels["attention_cache_owner_count"] != "5" ||
		runtime.loadConfig.ModelLabels["attention_cache_shared_layers"] != "1" ||
		runtime.loadConfig.ModelLabels["gemma4_fixed_sliding_prefill_chunk_limit"] != "1024" ||
		runtime.loadConfig.ModelLabels["attention_window_policy"] != "sliding_causal" ||
		runtime.loadConfig.ModelLabels["attention_mask_cached_offset_causal"] != "true" ||
		runtime.loadConfig.ModelLabels["attention_mask_fixed_single_token"] != "true" ||
		runtime.loadConfig.ModelLabels["gemma4_speculative_verify_proposal_window_limit"] != "1023" ||
		runtime.loadConfig.ModelLabels["gemma4_hidden_size_per_layer_input"] != "4" ||
		runtime.loadConfig.ModelLabels["gemma4_vocab_size_per_layer_input"] != "8" ||
		runtime.loadConfig.ModelLabels["gemma4_use_double_wide_mlp"] != "true" ||
		runtime.loadConfig.ModelLabels["gemma4_enable_moe_block"] != "true" ||
		runtime.loadConfig.ModelLabels["gemma4_num_experts"] != "16" ||
		runtime.loadConfig.ModelLabels["gemma4_top_k_experts"] != "2" ||
		runtime.loadConfig.ModelLabels["gemma4_moe_intermediate_size"] != "32" ||
		runtime.loadConfig.ModelLabels["final_logit_softcapping"] != "42" ||
		runtime.loadConfig.ModelLabels["attention_k_eq_v"] != "true" ||
		runtime.loadConfig.ModelLabels["attention_rope_full_theta"] != "1e+06" ||
		runtime.loadConfig.ModelLabels["attention_rope_full_factor"] != "1" {
		t.Fatalf("model labels = %+v, want Gemma4 attention metadata propagated", runtime.loadConfig.ModelLabels)
	}
	if !runtime.loadConfig.EngineProfile.Matched() ||
		runtime.loadConfig.EngineProfile.Name != "gemma4" ||
		!runtime.loadConfig.EngineProfile.Gemma4EngineFeatures.FixedSlidingCache ||
		!runtime.loadConfig.EngineProfile.Gemma4EngineFeatures.FixedSlidingCacheBound ||
		runtime.loadConfig.EngineProfile.Gemma4DeclaredFeatures.Attention.SlidingWindow != 1024 ||
		runtime.loadConfig.EngineProfile.Gemma4DeclaredFeatures.Attention.SlidingPattern != 5 ||
		runtime.loadConfig.ModelLabels["engine_profile"] != "gemma4" ||
		runtime.loadConfig.ModelLabels["engine_fixed_sliding_cache"] != "true" {
		t.Fatalf("engine profile = %+v labels=%+v, want config-owned Gemma4 registry profile", runtime.loadConfig.EngineProfile, runtime.loadConfig.ModelLabels)
	}
}

func TestNativeContract_Gemma4GlobalPartialRotaryFallback_Good(t *testing.T) {
	cfg := rocmModelPackConfigProbe{
		ModelType: "gemma4",
		TextConfig: rocmModelPackTextConfigProbe{
			ModelType:           "gemma4_text",
			NumHiddenLayers:     2,
			GlobalPartialRotary: 0.125,
		},
	}

	runtime := rocmNativeGemma4TextConfigFromProbe(cfg)
	full := runtime.RoPEParameters["full_attention"]
	core.AssertEqual(t, float64(0.125), full.PartialRotaryFactor)
	core.AssertEqual(t, float64(1000000), full.RopeTheta)
	core.AssertEqual(t, "proportional", full.RopeType)
	core.AssertEqual(t, float64(1), full.Factor)

	labels := rocmAttentionConfigLabels(cfg)
	core.AssertEqual(t, "0.125", labels["attention_rope_full_partial_rotary_factor"])
	core.AssertEqual(t, "1e+06", labels["attention_rope_full_theta"])
	core.AssertEqual(t, "proportional", labels["attention_rope_full_type"])
	core.AssertEqual(t, "1", labels["attention_rope_full_factor"])
}

func TestNativeContract_Gemma4TieWordEmbeddingsDefaultsTrue_Good(t *testing.T) {
	cfg := rocmModelPackConfigProbe{
		ModelType: "gemma4",
		TextConfig: rocmModelPackTextConfigProbe{
			ModelType: "gemma4_text",
		},
	}
	core.AssertEqual(t, true, rocmConfigTiedWordEmbeddings(cfg))

	explicitFalse := false
	cfg.TextConfig.TieWordEmbeddings = &explicitFalse
	core.AssertEqual(t, false, rocmConfigTiedWordEmbeddings(cfg))

	explicitTrue := true
	cfg.TieWordEmbeddings = &explicitTrue
	core.AssertEqual(t, true, rocmConfigTiedWordEmbeddings(cfg))
}

func TestNativeContract_Gemma4NativeConfigDeclaresMultimodalTowers_Good(t *testing.T) {
	cfg := rocmNativeGemma4TextConfigFromProbe(rocmModelPackConfigProbe{
		ModelType:                "gemma4",
		ImageTokenID:             258880,
		AudioTokenID:             258881,
		VisionSoftTokensPerImage: 280,
		VisionConfig: rocmModelPackVisionConfigProbe{
			ModelType:       "gemma4_vision",
			HiddenSize:      1152,
			NumHiddenLayers: 27,
		},
		AudioConfig: rocmModelPackAudioConfigProbe{
			ModelType:       "gemma4_audio",
			HiddenSize:      1024,
			NumHiddenLayers: 24,
			AudioEmbedDim:   768,
		},
	})

	core.AssertEqual(t, true, cfg.Vision)
	core.AssertEqual(t, true, cfg.Audio)
	features := Gemma4DeclaredFeaturesOfNativeConfig(cfg)
	core.AssertEqual(t, true, features.Vision)
	core.AssertEqual(t, true, features.Audio)
	labels := rocmApplyGemma4NativeConfigFeatureLabels(nil, cfg)
	core.AssertEqual(t, "true", labels["gemma4_multimodal"])
	core.AssertEqual(t, "true", labels["gemma4_vision"])
	core.AssertEqual(t, "true", labels["gemma4_audio"])

	textOnly := rocmNativeGemma4TextConfigFromProbe(rocmModelPackConfigProbe{ModelType: "gemma4"})
	core.AssertEqual(t, false, textOnly.Vision)
	core.AssertEqual(t, false, textOnly.Audio)
}

func TestNativeContract_Gemma4LayerTypesDefaultPatternForcesFinalFull_Good(t *testing.T) {
	cfg := rocmNativeGemma4TextConfigFromProbe(rocmModelPackConfigProbe{
		TextConfig: rocmModelPackTextConfigProbe{
			NumHiddenLayers:      7,
			SlidingWindowPattern: 3,
		},
	})

	core.AssertEqual(t, []string{
		"sliding_attention",
		"sliding_attention",
		"full_attention",
		"sliding_attention",
		"sliding_attention",
		"full_attention",
		"full_attention",
	}, cfg.LayerTypes)
}

func TestNativeContract_Gemma4PreservesE2BLayerMetadata_Good(t *testing.T) {
	kvShared := 20
	layerTypes := []string{
		"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
		"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
		"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
		"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
		"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
		"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
		"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
	}
	cfg := rocmNativeGemma4TextConfigFromProbe(rocmModelPackConfigProbe{
		ModelType: "gemma4",
		TextConfig: rocmModelPackTextConfigProbe{
			ModelType:         "gemma4_text",
			NumHiddenLayers:   35,
			SlidingWindow:     512,
			NumKVSharedLayers: &kvShared,
			LayerTypes:        layerTypes,
			RoPEParameters: map[string]rocmRoPEProbe{
				"sliding_attention": {RopeTheta: 10000, RopeType: "default"},
				"full_attention":    {PartialRotaryFactor: 0.25, RopeTheta: 1000000, RopeType: "proportional"},
			},
		},
	})

	core.AssertEqual(t, 35, len(cfg.LayerTypes))
	core.AssertEqual(t, layerTypes, cfg.LayerTypes)
	core.AssertEqual(t, true, cfg.KVSharedLayersSet)
	core.AssertEqual(t, 20, cfg.KVSharedLayers)
	core.AssertEqual(t, 512, cfg.SlidingWindow)
	core.AssertEqual(t, float64(10000), cfg.RoPEParameters["sliding_attention"].RopeTheta)
	core.AssertEqual(t, "default", cfg.RoPEParameters["sliding_attention"].RopeType)
	core.AssertEqual(t, float64(1000000), cfg.RoPEParameters["full_attention"].RopeTheta)
	core.AssertEqual(t, float64(0.25), cfg.RoPEParameters["full_attention"].PartialRotaryFactor)
	core.AssertEqual(t, "proportional", cfg.RoPEParameters["full_attention"].RopeType)

	layers := make([]hipGemma4Q4Layer0Config, len(cfg.LayerTypes))
	slidingLayers := 0
	fullLayers := 0
	for index, layerType := range cfg.LayerTypes {
		layers[index] = hipGemma4Q4Layer0Config{Layer: index, LayerType: layerType}
		switch layerType {
		case "sliding_attention":
			slidingLayers++
		case "full_attention":
			fullLayers++
		}
	}
	sources := hipGemma4Q4BuildSharedKVSourceByLayer(hipGemma4Q4ForwardConfig{
		Layers:         layers,
		KVSharedLayers: cfg.KVSharedLayers,
	})
	ownerCount := 0
	for index, source := range sources {
		if source == index {
			ownerCount++
		}
	}
	core.AssertEqual(t, 28, slidingLayers)
	core.AssertEqual(t, 7, fullLayers)
	core.AssertEqual(t, 15, ownerCount)
	core.AssertEqual(t, 13, sources[15])
	core.AssertEqual(t, 14, sources[19])
	core.AssertEqual(t, 14, sources[34])
}

func TestNativeContract_LoadModelSafetensorsShardedPackUsesNativeRuntime_Good(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type":"gemma4",
		"tie_word_embeddings":true,
		"quantization_config":{"bits":4,"group_size":64},
		"text_config":{"hidden_size":16,"num_hidden_layers":1,"vocab_size":8}
	}`)
	writeNativeContractSafetensorsHeaderWithPayload(t, core.PathJoin(dir, "model-00001-of-00002.safetensors"), `{"language_model.model.embed_tokens.weight":{"dtype":"U32","shape":[8,2],"data_offsets":[0,64]}}`, 64)
	writeNativeContractSafetensorsHeaderWithPayload(t, core.PathJoin(dir, "model-00002-of-00002.safetensors"), `{"language_model.model.layers.0.input_layernorm.weight":{"dtype":"BF16","shape":[16],"data_offsets":[0,32]}}`, 32)
	runtime := &fakeNativeRuntime{
		available: true,
		device:    nativeDeviceInfo{Name: "AMD Radeon RX 7800 XT", MemoryBytes: 16 * memoryGiB, FreeBytes: 12 * memoryGiB, Driver: "hip-test"},
		model:     &fakeNativeModel{},
	}

	model, err := resultValue[inference.TextModel](newROCmBackendWithRuntime(runtime).LoadModel(dir))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer model.Close()

	if runtime.loadPath != dir {
		t.Fatalf("loadPath = %q, want sharded model-pack dir", runtime.loadPath)
	}
	if runtime.loadConfig.DataOffset != 0 || !runtime.loadConfig.TiedWordEmbeddings || len(runtime.loadConfig.Tensors) != 2 {
		t.Fatalf("load config = %+v, want sharded safetensors tensor plan", runtime.loadConfig)
	}
	sourcePaths := map[string]bool{}
	for _, tensor := range runtime.loadConfig.Tensors {
		if tensor.SourcePath == "" || tensor.DataOffset <= 0 {
			t.Fatalf("tensor = %+v, want per-shard source path and data offset", tensor)
		}
		sourcePaths[core.PathBase(tensor.SourcePath)] = true
	}
	if !sourcePaths["model-00001-of-00002.safetensors"] || !sourcePaths["model-00002-of-00002.safetensors"] {
		t.Fatalf("source paths = %+v, want both safetensors shards", sourcePaths)
	}
}

func TestNativeContract_PlanModelFit_Good(t *testing.T) {
	runtime := &fakeNativeRuntime{device: nativeDeviceInfo{MemoryBytes: 16 * memoryGiB, Name: "gfx1100"}}
	report, err := newROCmBackendWithRuntime(runtime).PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture:  "qwen3",
		QuantBits:     4,
		ContextLength: 32768,
		NumLayers:     28,
		HiddenSize:    2048,
	}, 0)
	if err != nil {
		t.Fatalf("PlanModelFit: %v", err)
	}
	if report == nil || !report.Fits || !report.ArchitectureOK || !report.QuantizationOK {
		t.Fatalf("fit report = %+v, want supported fitting qwen3 q4", report)
	}
	if report.MemoryPlan.CacheMode == "" || report.MemoryPlan.KVCacheBytes == 0 {
		t.Fatalf("memory plan = %+v, want cache sizing", report.MemoryPlan)
	}
}

func TestNativeContract_PlanModelFit_Q6QuantTypeOnly_Good(t *testing.T) {
	runtime := &fakeNativeRuntime{device: nativeDeviceInfo{MemoryBytes: 16 * memoryGiB, Name: "gfx1100"}}
	report, err := newROCmBackendWithRuntime(runtime).PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture:  "gemma4_text",
		Path:          "/models/lmstudio-community-gemma-4-e2b-it-6bit",
		QuantType:     "q6",
		ContextLength: 32768,
		NumLayers:     35,
		HiddenSize:    1536,
	}, 0)
	if err != nil {
		t.Fatalf("PlanModelFit: %v", err)
	}
	if report == nil || !report.Fits || !report.ArchitectureOK || !report.QuantizationOK {
		t.Fatalf("fit report = %+v, want string-only q6 quantization accepted", report)
	}
	if report.Model.QuantType != "q6" {
		t.Fatalf("model quant type = %q, want q6", report.Model.QuantType)
	}
	if report.MemoryPlan.Labels["production_quant_policy"] != "gemma4_mlx_affine" ||
		report.MemoryPlan.Labels["production_quant_tier"] != "default" ||
		report.MemoryPlan.Labels["production_quant_active_weight_read_bytes_per_token"] == "" ||
		report.MemoryPlan.Labels["production_quant_min_visible_tokens_per_sec"] != "100" ||
		report.MemoryPlan.Labels["production_quant_pack_sizes"] != "E2B,E4B,12B,26B-A4B,31B" {
		t.Fatalf("memory plan labels = %+v, want q6 production quant policy labels", report.MemoryPlan.Labels)
	}
	if report.MemoryPlan.Labels["engine_state_context_route_contract"] != ROCmStateContextRegistryContract ||
		report.MemoryPlan.Labels["engine_state_context_window"] != "32768" ||
		report.MemoryPlan.Labels["engine_state_context_prompt_replay_refused"] != "true" ||
		report.MemoryPlan.Labels["engine_state_context_remaining_context_default"] != "true" ||
		report.MemoryPlan.Labels["engine_state_context_runtime_owned_kv"] != "true" ||
		report.MemoryPlan.Labels["engine_lora_route_contract"] != ROCmLoRAAdapterRegistryContract ||
		report.MemoryPlan.Labels["engine_lora_target_policy"] != "gemma4" ||
		report.MemoryPlan.Labels["engine_lora_default_targets"] != "q_proj,v_proj,o_proj" ||
		report.MemoryPlan.Labels["engine_lora_extended_targets_require_opt"] != "true" ||
		report.MemoryPlan.Labels["engine_attached_drafter_route_contract"] != ROCmAttachedDrafterRegistryContract ||
		report.MemoryPlan.Labels["engine_attached_drafter_role"] != "target" ||
		report.MemoryPlan.Labels["engine_attached_drafter_native_attachment"] != hipKernelStatusNotLinked ||
		report.MemoryPlan.Labels["engine_attached_drafter_retained_state_required"] != "true" ||
		report.MemoryPlan.Labels["engine_attached_drafter_prompt_replay_fallback"] != "forbidden" {
		t.Fatalf("memory plan labels = %+v, want registry route labels", report.MemoryPlan.Labels)
	}
}

func TestNativeContract_PlanModelFit_DenseAndMTPRouteLabels_Good(t *testing.T) {
	runtime := &fakeNativeRuntime{device: nativeDeviceInfo{MemoryBytes: 16 * memoryGiB, Name: "gfx1100"}}
	for _, architecture := range []string{"gemma3", "qwen3", "qwen3_6", "mistral"} {
		t.Run("dense_"+architecture, func(t *testing.T) {
			dense, err := newROCmBackendWithRuntime(runtime).PlanModelFit(context.Background(), inference.ModelIdentity{
				Architecture:  architecture,
				QuantBits:     6,
				ContextLength: 32768,
				NumLayers:     32,
				HiddenSize:    4096,
			}, 0)
			if err != nil {
				t.Fatalf("PlanModelFit dense: %v", err)
			}
			if dense == nil || dense.MemoryPlan.Labels["dense_route_candidate"] != "true" ||
				dense.MemoryPlan.Labels["dense_route_status"] != "experimental" ||
				dense.MemoryPlan.Labels["dense_route_family"] != "loader_neutral" ||
				dense.MemoryPlan.Labels["dense_route_backend"] != "hip_small_decode" ||
				dense.MemoryPlan.Labels["dense_route_reference"] != "gemma4_mlx_affine_matvec" {
				t.Fatalf("dense fit report = %+v, want dense route candidate labels", dense)
			}
		})
	}

	assistant, err := newROCmBackendWithRuntime(runtime).PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture:  "gemma4_assistant",
		QuantBits:     6,
		ContextLength: 32768,
		NumLayers:     35,
		HiddenSize:    1536,
	}, 0)
	if err != nil {
		t.Fatalf("PlanModelFit assistant: %v", err)
	}
	if assistant == nil || assistant.MemoryPlan.Labels["attached_drafter"] != "experimental_retained_plan" ||
		assistant.MemoryPlan.Labels["attached_drafter_native_attachment"] != hipKernelStatusNotLinked ||
		assistant.MemoryPlan.Labels["attached_drafter_retained_state_entrypoint"] != hipKernelStatusLinked ||
		assistant.MemoryPlan.Labels["attached_drafter_retained_state_required"] != "true" ||
		assistant.MemoryPlan.Labels["attached_drafter_state_source"] != "rocm_state_session_runtime_kv" ||
		assistant.MemoryPlan.Labels["attached_drafter_prompt_replay_fallback"] != "forbidden" ||
		assistant.MemoryPlan.Labels["mtp_role"] != "drafter" ||
		assistant.MemoryPlan.Labels["mtp_target_family"] != "gemma4" {
		t.Fatalf("assistant fit report = %+v, want MTP drafter labels", assistant)
	}
	if assistant.MemoryPlan.Labels["engine_state_context_route_contract"] != ROCmStateContextRegistryContract ||
		assistant.MemoryPlan.Labels["engine_state_context_attached_only"] != "true" ||
		assistant.MemoryPlan.Labels["engine_state_context_attached_drafter_state"] != "true" ||
		assistant.MemoryPlan.Labels["engine_state_context_runtime_owned_kv"] != "true" ||
		assistant.MemoryPlan.Labels["engine_attached_drafter_route_contract"] != ROCmAttachedDrafterRegistryContract ||
		assistant.MemoryPlan.Labels["engine_attached_drafter_role"] != "assistant" ||
		assistant.MemoryPlan.Labels["engine_attached_drafter_attached_only"] != "true" ||
		assistant.MemoryPlan.Labels["engine_attached_drafter_assistant"] != "true" ||
		assistant.MemoryPlan.Labels["engine_attached_drafter_prompt_replay_fallback"] != "forbidden" {
		t.Fatalf("assistant fit report labels = %+v, want registry route labels", assistant.MemoryPlan.Labels)
	}
}

func TestNativeContract_PlanModelFit_Rocm16GBMoELazyExperts_Good(t *testing.T) {
	runtime := &fakeNativeRuntime{device: nativeDeviceInfo{MemoryBytes: 16 * memoryGiB, Name: "gfx1100"}}
	report, err := newROCmBackendWithRuntime(runtime).PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture:  "Qwen3MoeForCausalLM",
		QuantBits:     2,
		QuantType:     "jangtq",
		QuantGroup:    64,
		ContextLength: 32768,
		NumLayers:     24,
		HiddenSize:    2048,
	}, 0)
	if err != nil {
		t.Fatalf("PlanModelFit: %v", err)
	}
	if report == nil || !report.Fits || report.MemoryPlan.MachineClass != "rocm-16gb" {
		t.Fatalf("fit report = %+v, want fitting ROCm 16GB MoE plan", report)
	}
	if report.MemoryPlan.CacheMode != "k-q8-v-q4" || report.MemoryPlan.Labels["moe_lazy_experts"] != "true" || report.MemoryPlan.Labels["prefill_chunk_tokens"] != "512" {
		t.Fatalf("memory plan = %+v, want compact KV, lazy experts, and chunked prefill", report.MemoryPlan)
	}
}

func TestNativeContract_PlanModelFit_MemoryClassesAndCacheModes_Good(t *testing.T) {
	cases := []struct {
		name             string
		memoryBytes      uint64
		contextLength    int
		wantMachineClass string
		wantCacheMode    string
		wantBatchSize    int
		wantTraining     bool
	}{
		{name: "Small", memoryBytes: 8 * memoryGiB, contextLength: 4096, wantMachineClass: "rocm-small", wantCacheMode: "q8", wantBatchSize: 1, wantTraining: false},
		{name: "RX7800XTReported16GB", memoryBytes: 17163091968, contextLength: 131072, wantMachineClass: "rocm-16gb", wantCacheMode: "k-q8-v-q4", wantBatchSize: 1, wantTraining: true},
		{name: "TwentyFourGB", memoryBytes: 24 * memoryGiB, contextLength: 4096, wantMachineClass: "rocm-24gb", wantCacheMode: "q8", wantBatchSize: 4, wantTraining: true},
		{name: "SixtyFourGB", memoryBytes: 64 * memoryGiB, contextLength: 4096, wantMachineClass: "rocm-64gb-plus", wantCacheMode: "fp16", wantBatchSize: 8, wantTraining: true},
		{name: "LongContext", memoryBytes: 64 * memoryGiB, contextLength: 32768, wantMachineClass: "rocm-64gb-plus", wantCacheMode: "q8", wantBatchSize: 8, wantTraining: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			report, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).PlanModelFit(context.Background(), inference.ModelIdentity{
				Architecture:  "qwen3",
				QuantBits:     4,
				ContextLength: tc.contextLength,
				NumLayers:     16,
				HiddenSize:    2048,
			}, tc.memoryBytes)
			if err != nil {
				t.Fatalf("PlanModelFit: %v", err)
			}
			if report.MemoryPlan.MachineClass != tc.wantMachineClass || report.MemoryPlan.CacheMode != tc.wantCacheMode || report.MemoryPlan.BatchSize != tc.wantBatchSize || report.MemoryPlan.TrainingFeasible != tc.wantTraining {
				t.Fatalf("memory plan = %+v, want class=%s cache=%s batch=%d training=%t", report.MemoryPlan, tc.wantMachineClass, tc.wantCacheMode, tc.wantBatchSize, tc.wantTraining)
			}
			if report.MemoryPlan.Labels["recommended_cache_mode"] != tc.wantCacheMode || report.MemoryPlan.Labels["allocator_limit_bytes"] == "" {
				t.Fatalf("memory plan labels = %+v, want cache mode and allocator labels", report.MemoryPlan.Labels)
			}
		})
	}
}

func TestNativeContract_PlanModelFit_UsesKnownWeightBytes_Bad(t *testing.T) {
	report, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture:  "gemma4",
		QuantType:     "bf16",
		ContextLength: 131072,
		NumLayers:     35,
		HiddenSize:    1536,
		Labels:        map[string]string{"weight_bytes": "9294899782"},
	}, 17163091968)
	if err != nil {
		t.Fatalf("PlanModelFit: %v", err)
	}
	if report == nil || report.Fits || report.MemoryPlan.MachineClass != "rocm-16gb" || report.MemoryPlan.CacheMode != rocmKVCacheModeKQ8VQ4 {
		t.Fatalf("fit report = %+v, want non-fitting native-context Gemma4 BF16 plan on RX 7800 XT", report)
	}
	if report.MemoryPlan.Labels["weight_bytes"] != "9294899782" || report.MemoryPlan.Labels["estimated_runtime_bytes"] == "" {
		t.Fatalf("memory plan labels = %+v, want known weight bytes and total estimate", report.MemoryPlan.Labels)
	}
	if !nativeContractHasNoteContaining(report.Notes, "weight and KV cache estimate leaves too little memory") {
		t.Fatalf("notes = %+v, want known-weight memory pressure note", report.Notes)
	}
}

func TestNativeContract_PlanModelFit_Gemma4SlidingAttentionWeightBytes_Good(t *testing.T) {
	report, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture:  "gemma4",
		QuantType:     "bf16",
		ContextLength: 131072,
		NumLayers:     35,
		HiddenSize:    1536,
		Labels: map[string]string{
			"weight_bytes":              "9294899782",
			"attention_full_layers":     "7",
			"attention_sliding_layers":  "28",
			"sliding_window":            "512",
			"attention_kv_width":        "256",
			"attention_global_kv_width": "512",
		},
	}, 17163091968)
	if err != nil {
		t.Fatalf("PlanModelFit: %v", err)
	}
	if report == nil || !report.Fits || report.MemoryPlan.MachineClass != "rocm-16gb" || report.MemoryPlan.CacheMode != rocmKVCacheModeKQ8VQ4 {
		t.Fatalf("fit report = %+v, want fitting Gemma4 BF16 sliding-attention plan on RX 7800 XT", report)
	}
	if report.MemoryPlan.Labels["weight_bytes"] != "9294899782" ||
		report.MemoryPlan.Labels["estimated_runtime_bytes"] == "" ||
		report.MemoryPlan.Labels["kv_cache_bytes"] != "710148096" ||
		report.MemoryPlan.Labels["kv_key_width"] != "10752" ||
		report.MemoryPlan.Labels["kv_value_width"] != "10752" ||
		report.MemoryPlan.Labels["attention_full_layers"] != "7" ||
		report.MemoryPlan.Labels["attention_sliding_layers"] != "28" ||
		report.MemoryPlan.Labels["attention_kv_width"] != "256" ||
		report.MemoryPlan.Labels["attention_global_kv_width"] != "512" ||
		report.MemoryPlan.Labels["sliding_window"] != "512" ||
		report.MemoryPlan.Labels["production_quant_policy"] != "gemma4_mlx_affine" ||
		report.MemoryPlan.Labels["production_quant_default_bits"] != "6" ||
		report.MemoryPlan.Labels["production_quant_quality_bits"] != "8" ||
		report.MemoryPlan.Labels["production_quant_constrained_bits"] != "4" {
		t.Fatalf("memory plan labels = %+v, want known weights and sliding-attention metadata", report.MemoryPlan.Labels)
	}
	if nativeContractHasNoteContaining(report.Notes, "weight and KV cache estimate leaves too little memory") {
		t.Fatalf("notes = %+v, sliding-attention plan should not report known-weight memory pressure", report.Notes)
	}
}

func TestNativeContract_PlanModelFit_CacheModesAreConstructible_Good(t *testing.T) {
	cases := []struct {
		name        string
		model       inference.ModelIdentity
		memoryBytes uint64
		wantMode    string
	}{
		{
			name:        "Q8Small",
			memoryBytes: 8 * memoryGiB,
			wantMode:    rocmKVCacheModeQ8,
			model:       inference.ModelIdentity{Architecture: "qwen3", QuantBits: 4, ContextLength: 4096, NumLayers: 16, HiddenSize: 2048},
		},
		{
			name:        "FP16Large",
			memoryBytes: 64 * memoryGiB,
			wantMode:    rocmKVCacheModeFP16,
			model:       inference.ModelIdentity{Architecture: "qwen3", QuantBits: 4, ContextLength: 4096, NumLayers: 16, HiddenSize: 2048},
		},
		{
			name:        "CompactMoE",
			memoryBytes: 16 * memoryGiB,
			wantMode:    rocmKVCacheModeKQ8VQ4,
			model:       inference.ModelIdentity{Architecture: "Qwen3MoeForCausalLM", QuantBits: 2, QuantType: "jangtq", QuantGroup: 64, ContextLength: 32768, NumLayers: 24, HiddenSize: 2048},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			report, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).PlanModelFit(context.Background(), tc.model, tc.memoryBytes)
			core.RequireNoError(t, err)
			core.AssertEqual(t, tc.wantMode, report.MemoryPlan.CacheMode)

			cache := NewBlockCacheService(BlockCacheConfig{CacheMode: report.MemoryPlan.CacheMode})
			warmed, err := cache.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4, 5, 6, 7, 8}, Labels: report.MemoryPlan.Labels})

			core.RequireNoError(t, err)
			core.AssertEqual(t, tc.wantMode, warmed.Blocks[0].Encoding)
			core.AssertEqual(t, "true", warmed.Blocks[0].Labels["kv_cache_constructible"])
			core.AssertEqual(t, report.MemoryPlan.Labels["kv_key_width"], warmed.Blocks[0].Labels["kv_key_width"])
			core.AssertEqual(t, report.MemoryPlan.Labels["kv_value_width"], warmed.Blocks[0].Labels["kv_value_width"])
			core.AssertGreater(t, warmed.Blocks[0].SizeBytes, uint64(0))
		})
	}
}

func TestNativeContract_PlanModelFit_Bad(t *testing.T) {
	report, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture: "unknown",
		QuantBits:    16,
	}, 8*memoryGiB)
	if err != nil {
		t.Fatalf("PlanModelFit: %v", err)
	}
	if report == nil || report.ArchitectureOK || report.QuantizationOK || report.Fits {
		t.Fatalf("fit report = %+v, want unsupported model", report)
	}
}

func TestNativeContract_PlanModelFit_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	report, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).PlanModelFit(ctx, inference.ModelIdentity{Architecture: "qwen3"}, 0)
	if err == nil {
		t.Fatalf("PlanModelFit cancelled error = nil, report=%+v", report)
	}
}

func TestNativeContract_ProbeSinkReceivesGeneratedTokens_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &fakeNativeModel{tokens: []inference.Token{{ID: 9, Text: "hi"}}},
	}
	var got inference.ProbeEvent
	model.SetProbeSink(inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
		got = event
	}))

	for range model.Generate(context.Background(), "hello") {
	}

	if got.Kind != inference.ProbeEventToken || got.Token == nil || got.Token.ID != 9 || got.Token.Text != "hi" {
		t.Fatalf("probe event = %+v, want generated token event", got)
	}
}

func TestNativeContract_GeneratePassesStopTokens_Good(t *testing.T) {
	native := &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "ok"}}}
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    native,
	}

	core.AssertEqual(t, []string{"ok"}, collectTokenText(model.Generate(context.Background(), "hello", inference.WithStopTokens(2, 3))))

	core.AssertEqual(t, []int32{2, 3}, native.generateConfigs[0].StopTokens)
}

func TestNativeContract_TokenizerBoundariesCloneMutableSlices_Good(t *testing.T) {
	native := &fakeNativeModel{
		encodeResult:             []int32{1, 2},
		decodeMutatesInput:       true,
		chatTemplateMutatesInput: true,
	}
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    native,
	}

	encoded := model.Encode("hello")
	encoded[0] = 99
	core.AssertEqual(t, int32(1), native.encodeResult[0])

	ids := []int32{1, 2}
	_ = model.Decode(ids)
	core.AssertEqual(t, []int32{1, 2}, ids)

	messages := []inference.Message{{Role: "user", Content: "hello"}}
	_, err := model.ApplyChatTemplate(messages)
	core.RequireNoError(t, err)
	core.AssertEqual(t, "user", messages[0].Role)
	core.AssertEqual(t, "hello", messages[0].Content)
}

func TestNativeContract_ApplyChatTemplateBadRecordsErrAndSuccessClears_Bad(t *testing.T) {
	native := &fakeNativeModel{chatTemplateErr: core.NewError("template failed")}
	model := &rocmModel{native: native}

	_, err := model.ApplyChatTemplate([]inference.Message{{Role: "user", Content: "hello"}})

	core.AssertError(t, err)
	if resultError(model.Err()) == nil {
		t.Fatal("ApplyChatTemplate failure Err() = nil")
	}
	core.AssertContains(t, resultError(model.Err()).Error(), "template failed")

	native.chatTemplateErr = nil
	native.chatTemplateResult = "user:hello\n"
	prompt, err := model.ApplyChatTemplate([]inference.Message{{Role: "user", Content: "hello"}})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "user:hello\n", prompt)
	if resultError(model.Err()) != nil {
		t.Fatalf("ApplyChatTemplate success Err() = %v, want nil", resultError(model.Err()))
	}
}

func TestNativeContract_GenerateStopTokensSurviveNativeConfigMutation_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native: &fakeNativeModel{
			mutateGenerateConfig: true,
			tokens: []inference.Token{
				{ID: 1, Text: "hello "},
				{ID: 2, Text: "EN"},
				{ID: 3, Text: "D hidden"},
			},
		},
	}

	text := strings.Join(collectTokenText(model.Generate(context.Background(), "hello", inference.WithStopTokens(2))), "")

	core.AssertEqual(t, "hello END hidden", text)
}

func TestNativeContract_ClassifyWithLogitsEmitsLogitAndEntropyProbes_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 3},
		native: &fakeNativeModel{
			classLogits: [][]float32{{0, 3, 1}},
		},
	}
	var events []inference.ProbeEvent
	model.SetProbeSink(inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
		events = append(events, event)
	}))

	results, err := resultValue[[]inference.ClassifyResult](model.Classify(context.Background(), []string{"hello"}, inference.WithLogits()))

	core.RequireNoError(t, err)
	if len(results) != 1 || len(results[0].Logits) != 3 {
		t.Fatalf("classify results = %+v, want logits returned when requested", results)
	}
	logitEvent, ok := nativeContractProbeEvent(events, inference.ProbeEventLogits)
	if !ok || logitEvent.Logits == nil || len(logitEvent.Logits.Top) == 0 || logitEvent.Logits.Top[0].ID != 1 || logitEvent.Labels["source"] != "classification" || logitEvent.Step != 1 {
		t.Fatalf("probe events = %+v, want compact classification logit event", events)
	}
	entropyEvent, ok := nativeContractProbeEvent(events, inference.ProbeEventEntropy)
	if !ok || entropyEvent.Entropy == nil || entropyEvent.Entropy.Unit != "nats" || entropyEvent.Labels["classify_prompt_index"] != "0" {
		t.Fatalf("probe events = %+v, want classification entropy event", events)
	}
}

func TestNativeContract_ClassifyWithoutLogitsStripsNativeLogits_Bad(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2},
		native: &fakeNativeModel{
			classLogits:       [][]float32{{2, 0}},
			classLogitsAlways: true,
		},
	}
	var events []inference.ProbeEvent
	model.SetProbeSink(inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
		events = append(events, event)
	}))

	results, err := resultValue[[]inference.ClassifyResult](model.Classify(context.Background(), []string{"hello"}))

	core.RequireNoError(t, err)
	if len(results) != 1 || len(results[0].Logits) != 0 {
		t.Fatalf("classify results = %+v, want logits stripped unless WithLogits is requested", results)
	}
	if len(events) != 0 {
		t.Fatalf("probe events = %+v, want no logit probes without WithLogits", events)
	}
}

func TestNativeContract_ClassifyResultsClonedAtPublicBoundary_Good(t *testing.T) {
	nativeResults := []inference.ClassifyResult{{
		Token:  inference.Token{ID: 7, Text: "native"},
		Logits: []float32{1, 2, 3},
	}}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 3},
		native: &fakeNativeModel{
			classifyResults: nativeResults,
		},
	}

	results, err := resultValue[[]inference.ClassifyResult](model.Classify(context.Background(), []string{"hello"}, inference.WithLogits()))
	core.RequireNoError(t, err)
	if len(results) != 1 || len(results[0].Logits) != 3 {
		t.Fatalf("Classify() = %+v, want native logits", results)
	}
	results[0].Logits[0] = 9

	core.AssertEqual(t, float32(1), nativeResults[0].Logits[0])
}

func TestNativeContract_ClassifyWithoutLogitsDoesNotMutateNativeResult_Bad(t *testing.T) {
	nativeResults := []inference.ClassifyResult{{
		Token:  inference.Token{ID: 7, Text: "native"},
		Logits: []float32{1, 2, 3},
	}}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 3},
		native: &fakeNativeModel{
			classifyResults: nativeResults,
		},
	}

	results, err := resultValue[[]inference.ClassifyResult](model.Classify(context.Background(), []string{"hello"}))
	core.RequireNoError(t, err)
	if len(results) != 1 || len(results[0].Logits) != 0 {
		t.Fatalf("Classify() = %+v, want returned logits stripped", results)
	}

	core.AssertEqual(t, 3, len(nativeResults[0].Logits))
	core.AssertEqual(t, float32(1), nativeResults[0].Logits[0])
}

func TestNativeContract_TextBatchPreflightRejectsEmptyPrompts_Bad(t *testing.T) {
	native := &fakeNativeModel{}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny"},
		native:    native,
	}

	_, err := resultValue[[]inference.ClassifyResult](model.Classify(context.Background(), nil))
	if err == nil {
		t.Fatal("Classify(nil) error = nil, want prompts-required error")
	}
	core.AssertContains(t, err.Error(), "prompts are required")

	_, err = resultValue[[]inference.ClassifyResult](model.Classify(context.Background(), []string{"hello", " "}))
	if err == nil {
		t.Fatal("Classify(empty prompt) error = nil, want prompt-empty error")
	}
	core.AssertContains(t, err.Error(), "prompt 1 is empty")

	_, err = resultValue[[]inference.BatchResult](model.BatchGenerate(context.Background(), nil))
	if err == nil {
		t.Fatal("BatchGenerate(nil) error = nil, want prompts-required error")
	}
	core.AssertContains(t, err.Error(), "prompts are required")

	_, err = resultValue[[]inference.BatchResult](model.BatchGenerate(context.Background(), []string{"hello", ""}))
	if err == nil {
		t.Fatal("BatchGenerate(empty prompt) error = nil, want prompt-empty error")
	}
	core.AssertContains(t, err.Error(), "prompt 1 is empty")

	if len(native.classifyPrompts) != 0 {
		t.Fatalf("classify prompts = %+v, want no native dispatch for invalid prompt batches", native.classifyPrompts)
	}
}

func TestNativeContract_NonStreamingNilNativeRecordsErr_Bad(t *testing.T) {
	model := &rocmModel{modelType: "tiny", modelInfo: inference.ModelInfo{Architecture: "tiny"}}

	_, err := resultValue[[]inference.ClassifyResult](model.Classify(context.Background(), []string{"hello"}))
	if err == nil {
		t.Fatal("Classify(nil native) error = nil")
	}
	core.AssertContains(t, err.Error(), "native model is nil")
	core.AssertContains(t, resultError(model.Err()).Error(), "native model is nil")

	_, err = resultValue[[]inference.BatchResult](model.BatchGenerate(context.Background(), []string{"hello"}))
	if err == nil {
		t.Fatal("BatchGenerate(nil native) error = nil")
	}
	core.AssertContains(t, err.Error(), "native model is nil")
	core.AssertContains(t, resultError(model.Err()).Error(), "native model is nil")
}

func TestNativeContract_ChatPreflightRejectsInvalidMessages_Bad(t *testing.T) {
	native := &fakeNativeModel{}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny"},
		native:    native,
	}

	for range model.Chat(context.Background(), nil) {
		t.Fatal("Chat(nil) yielded token, want empty stream")
	}
	if err := resultError(model.Err()); err == nil {
		t.Fatal("Chat(nil) Err() = nil, want messages-required error")
	} else {
		core.AssertContains(t, err.Error(), "messages are required")
	}

	for range model.Chat(context.Background(), []inference.Message{{Role: "moderator", Content: "hello"}}) {
		t.Fatal("Chat(invalid role) yielded token, want empty stream")
	}
	if err := resultError(model.Err()); err == nil {
		t.Fatal("Chat(invalid role) Err() = nil, want role validation error")
	} else {
		core.AssertContains(t, err.Error(), "message 0 role")
	}

	for range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: " "}}) {
		t.Fatal("Chat(empty content) yielded token, want empty stream")
	}
	if err := resultError(model.Err()); err == nil {
		t.Fatal("Chat(empty content) Err() = nil, want content validation error")
	} else {
		core.AssertContains(t, err.Error(), "at least one message must contain content")
	}

	if len(native.generatePrompts) != 0 {
		t.Fatalf("generate prompts = %+v, want no native dispatch for invalid chat messages", native.generatePrompts)
	}
}

func TestNativeContract_BatchGenerateRecordsNativeError_Bad(t *testing.T) {
	nativeErr := core.NewError("native batch failure")
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny"},
		native:    &fakeNativeModel{batchErr: nativeErr},
	}

	results, err := resultValue[[]inference.BatchResult](model.BatchGenerate(context.Background(), []string{"hello"}))

	if err == nil {
		t.Fatalf("BatchGenerate error = nil, results=%+v", results)
	}
	core.AssertContains(t, err.Error(), "native batch failure")
	if resultError(model.Err()) == nil {
		t.Fatal("model.Err() = nil, want native batch failure")
	}
	core.AssertContains(t, resultError(model.Err()).Error(), "native batch failure")
}

func TestNativeContract_BatchGenerateRecordsPerPromptError_Bad(t *testing.T) {
	promptErr := core.NewError("prompt 1 failed")
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny"},
		native: &fakeNativeModel{
			batchResults: []inference.BatchResult{
				{Tokens: []inference.Token{{ID: 1, Text: "ok"}}},
				{Err: promptErr},
			},
		},
	}

	results, err := resultValue[[]inference.BatchResult](model.BatchGenerate(context.Background(), []string{"ok", "bad"}))

	core.RequireNoError(t, err)
	if len(results) != 2 {
		t.Fatalf("BatchGenerate results = %+v, want 2 results", results)
	}
	if results[1].Err == nil {
		t.Fatalf("BatchGenerate result error = nil, results=%+v", results)
	}
	core.AssertContains(t, results[1].Err.Error(), "prompt 1 failed")
	if resultError(model.Err()) == nil {
		t.Fatal("model.Err() = nil, want per-prompt batch failure")
	}
	core.AssertContains(t, resultError(model.Err()).Error(), "prompt 1 failed")
	core.AssertEqual(t, 1, model.Metrics().GeneratedTokens)
}

func TestNativeContract_BatchGenerateResultsClonedAtPublicBoundary_Good(t *testing.T) {
	nativeResults := []inference.BatchResult{{
		Tokens: []inference.Token{{ID: 7, Text: "native"}},
	}}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny"},
		native: &fakeNativeModel{
			batchResults: nativeResults,
		},
	}

	results, err := resultValue[[]inference.BatchResult](model.BatchGenerate(context.Background(), []string{"hello"}))
	core.RequireNoError(t, err)
	if len(results) != 1 || len(results[0].Tokens) != 1 {
		t.Fatalf("BatchGenerate() = %+v, want native tokens", results)
	}
	results[0].Tokens[0].Text = "mutated"

	core.AssertEqual(t, "native", nativeResults[0].Tokens[0].Text)
}

func TestNativeContract_NonStreamingPromptInputsClonedAtNativeBoundary_Good(t *testing.T) {
	prompts := []string{"hello"}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny"},
		native: &fakeNativeModel{
			tokens:             []inference.Token{{ID: 1, Text: "ok"}},
			mutatePromptInputs: true,
		},
	}

	_, err := resultValue[[]inference.ClassifyResult](model.Classify(context.Background(), prompts))
	core.RequireNoError(t, err)
	core.AssertEqual(t, "hello", prompts[0])

	_, err = resultValue[[]inference.BatchResult](model.BatchGenerate(context.Background(), prompts))
	core.RequireNoError(t, err)
	core.AssertEqual(t, "hello", prompts[0])
}

func TestNativeContract_BatchGeneratePassesStopTokens_Good(t *testing.T) {
	nativeResults := []inference.BatchResult{{
		Tokens: []inference.Token{
			{ID: 1, Text: "hello "},
			{ID: 2, Text: "EN"},
			{ID: 3, Text: "D hidden"},
		},
	}}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny"},
		native: &fakeNativeModel{
			batchResults: nativeResults,
		},
	}

	results, err := resultValue[[]inference.BatchResult](model.BatchGenerate(context.Background(), []string{"hello"}, inference.WithStopTokens(2, 3)))
	core.RequireNoError(t, err)
	var text string
	for _, token := range results[0].Tokens {
		text += token.Text
	}

	core.AssertEqual(t, "hello END hidden", text)
	core.AssertEqual(t, "D hidden", nativeResults[0].Tokens[2].Text)
	core.AssertEqual(t, []int32{2, 3}, model.native.(*fakeNativeModel).generateConfigs[0].StopTokens)
	core.AssertEqual(t, 3, model.Metrics().GeneratedTokens)
}

func TestNativeContract_BatchGenerateStopTokensSurviveNativeConfigMutation_Good(t *testing.T) {
	nativeResults := []inference.BatchResult{{
		Tokens: []inference.Token{
			{ID: 1, Text: "hello "},
			{ID: 2, Text: "EN"},
			{ID: 3, Text: "D hidden"},
		},
	}}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny"},
		native: &fakeNativeModel{
			batchResults:         nativeResults,
			mutateGenerateConfig: true,
		},
	}

	results, err := resultValue[[]inference.BatchResult](model.BatchGenerate(context.Background(), []string{"hello"}, inference.WithStopTokens(2)))
	core.RequireNoError(t, err)
	var text string
	for _, token := range results[0].Tokens {
		text += token.Text
	}

	core.AssertEqual(t, "hello END hidden", text)
}

func TestNativeContract_NonStreamingTextMetricsUseTokenizerPromptCounts_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny"},
		native: &fakeNativeModel{
			tokens:       []inference.Token{{ID: 7, Text: "ok"}},
			encodeResult: []int32{10, 11, 12},
		},
	}

	classify, err := resultValue[[]inference.ClassifyResult](model.Classify(context.Background(), []string{"a", "b"}))
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, len(classify))
	core.AssertEqual(t, 6, model.Metrics().PromptTokens)
	core.AssertEqual(t, 2, model.Metrics().GeneratedTokens)

	batch, err := resultValue[[]inference.BatchResult](model.BatchGenerate(context.Background(), []string{"a", "b"}))
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, len(batch))
	core.AssertEqual(t, 6, model.Metrics().PromptTokens)
	core.AssertEqual(t, 2, model.Metrics().GeneratedTokens)
}

func TestNativeContract_ChatMetricsUseTemplateTokenizerPromptCount_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny"},
		native: &fakeNativeModel{
			tokens:       []inference.Token{{ID: 7, Text: "ok"}},
			encodeResult: []int32{10, 11, 12, 13},
		},
	}
	messages := []inference.Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "hello"},
	}

	core.AssertEqual(t, []string{"ok"}, collectTokenText(model.Chat(context.Background(), messages)))
	core.AssertEqual(t, 4, model.Metrics().PromptTokens)
	core.AssertEqual(t, 1, model.Metrics().GeneratedTokens)
	core.AssertEqual(t, []inference.Message{{Role: "system", Content: "sys"}, {Role: "user", Content: "hello"}}, messages)
}

func TestNativeContract_ChatMessagesClonedAtNativeBoundary_Good(t *testing.T) {
	messages := []inference.Message{{Role: "user", Content: "hello"}}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny"},
		native: &fakeNativeModel{
			tokens:           []inference.Token{{ID: 1, Text: "ok"}},
			chatMutatesInput: true,
		},
	}

	core.AssertEqual(t, []string{"ok"}, collectTokenText(model.Chat(context.Background(), messages)))

	core.AssertEqual(t, []inference.Message{{Role: "user", Content: "hello"}}, messages)
}

func TestNativeContract_EvaluateMetricsUseTemplateTokenizerTokenCounts_Good(t *testing.T) {
	native := &fakeNativeModel{
		chatTemplateResult: "templated prompt",
		encodeByText: map[string][]int32{
			"templated prompt": []int32{1, 2, 3, 4},
			"user: hello\n":    []int32{9},
		},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny"},
		native:    native,
	}

	eval, err := model.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{
		Messages: []inference.Message{{Role: "user", Content: "hello"}},
	}}, inference.EvalConfig{MaxSamples: 1})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, eval.Metrics.Samples)
	core.AssertEqual(t, 4, eval.Metrics.Tokens)
	core.AssertEqual(t, "4", eval.Labels["eval.tokens"])
}

func TestNativeContract_NonStreamingTextSuccessClearsLastError_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny"},
		native:    &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "ok"}}},
	}

	_, err := resultValue[[]inference.ClassifyResult](model.Classify(context.Background(), nil))
	if err == nil {
		t.Fatal("Classify(nil) error = nil, want validation failure")
	}
	if resultError(model.Err()) == nil {
		t.Fatal("model.Err() after invalid Classify = nil")
	}
	_, err = resultValue[[]inference.ClassifyResult](model.Classify(context.Background(), []string{"hello"}))
	core.RequireNoError(t, err)
	if resultError(model.Err()) != nil {
		t.Fatalf("model.Err() after successful Classify = %v, want nil", resultError(model.Err()))
	}

	_, err = resultValue[[]inference.BatchResult](model.BatchGenerate(context.Background(), nil))
	if err == nil {
		t.Fatal("BatchGenerate(nil) error = nil, want validation failure")
	}
	if resultError(model.Err()) == nil {
		t.Fatal("model.Err() after invalid BatchGenerate = nil")
	}
	_, err = resultValue[[]inference.BatchResult](model.BatchGenerate(context.Background(), []string{"hello"}))
	core.RequireNoError(t, err)
	if resultError(model.Err()) != nil {
		t.Fatalf("model.Err() after successful BatchGenerate = %v, want nil", resultError(model.Err()))
	}
}

func TestNativeContract_PublicWrappersPreferCancelledContext_Ugly(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny"},
		native: &fakeNativeEmbeddingModel{
			fakeNativeModel: &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "ok"}}},
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	for range model.Generate(ctx, "hello") {
		t.Fatal("Generate(cancelled) yielded token, want empty stream")
	}
	if !errors.Is(resultError(model.Err()), context.Canceled) {
		t.Fatalf("Generate Err() = %v, want context.Canceled", resultError(model.Err()))
	}

	for range model.Chat(ctx, nil) {
		t.Fatal("Chat(cancelled) yielded token, want empty stream")
	}
	if !errors.Is(resultError(model.Err()), context.Canceled) {
		t.Fatalf("Chat Err() = %v, want context.Canceled", resultError(model.Err()))
	}

	_, err := resultValue[[]inference.ClassifyResult](model.Classify(ctx, nil))
	if !errors.Is(err, context.Canceled) || !errors.Is(resultError(model.Err()), context.Canceled) {
		t.Fatalf("Classify error=%v Err()=%v, want context.Canceled", err, resultError(model.Err()))
	}

	_, err = resultValue[[]inference.BatchResult](model.BatchGenerate(ctx, nil))
	if !errors.Is(err, context.Canceled) || !errors.Is(resultError(model.Err()), context.Canceled) {
		t.Fatalf("BatchGenerate error=%v Err()=%v, want context.Canceled", err, resultError(model.Err()))
	}

	_, err = model.Embed(ctx, inference.EmbeddingRequest{})
	if !errors.Is(err, context.Canceled) || !errors.Is(resultError(model.Err()), context.Canceled) {
		t.Fatalf("Embed error=%v Err()=%v, want context.Canceled", err, resultError(model.Err()))
	}

	_, err = model.Rerank(ctx, inference.RerankRequest{})
	if !errors.Is(err, context.Canceled) || !errors.Is(resultError(model.Err()), context.Canceled) {
		t.Fatalf("Rerank error=%v Err()=%v, want context.Canceled", err, resultError(model.Err()))
	}

	if len(model.native.(*fakeNativeEmbeddingModel).generatePrompts) != 0 {
		t.Fatalf("generate prompts = %+v, want no native dispatch after cancelled context", model.native.(*fakeNativeEmbeddingModel).generatePrompts)
	}
}

func TestNativeContract_EmbeddingResultClonedAtPublicBoundary_Good(t *testing.T) {
	nativeResult := &inference.EmbeddingResult{
		Model:   inference.ModelIdentity{Architecture: "native", Labels: map[string]string{"source": "native"}},
		Vectors: [][]float32{{1, 2}},
		Labels:  map[string]string{"backend": "fake"},
	}
	model := &rocmModel{
		modelType: "bert",
		modelInfo: inference.ModelInfo{Architecture: "bert", HiddenSize: 2},
		native: &fakeNativeEmbeddingModel{
			fakeNativeModel: &fakeNativeModel{},
			embedResult:     nativeResult,
		},
	}

	result, err := model.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"hello"}})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "bert", result.Model.Architecture)
	result.Vectors[0][0] = 9
	result.Labels["backend"] = "mutated"

	core.AssertEqual(t, float32(1), nativeResult.Vectors[0][0])
	core.AssertEqual(t, "fake", nativeResult.Labels["backend"])
	core.AssertEqual(t, "native", nativeResult.Model.Architecture)
	core.AssertEqual(t, "native", nativeResult.Model.Labels["source"])
}

func TestNativeContract_RerankResultClonedAtPublicBoundary_Good(t *testing.T) {
	nativeResult := &inference.RerankResult{
		Model: inference.ModelIdentity{Architecture: "native", Labels: map[string]string{"source": "native"}},
		Results: []inference.RerankScore{{
			Index:  0,
			Score:  0.75,
			Text:   "doc",
			Labels: map[string]string{"ranker": "native"},
		}},
		Labels: map[string]string{"backend": "fake"},
	}
	model := &rocmModel{
		modelType: "bert",
		modelInfo: inference.ModelInfo{Architecture: "bert", HiddenSize: 2},
		native: &fakeNativeEmbeddingModel{
			fakeNativeModel: &fakeNativeModel{},
			rerankResult:    nativeResult,
		},
	}

	result, err := model.Rerank(context.Background(), inference.RerankRequest{Query: "hello", Documents: []string{"doc"}})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "bert", result.Model.Architecture)
	result.Results[0].Labels["ranker"] = "mutated"
	result.Labels["backend"] = "mutated"

	core.AssertEqual(t, "native", nativeResult.Results[0].Labels["ranker"])
	core.AssertEqual(t, "fake", nativeResult.Labels["backend"])
	core.AssertEqual(t, "native", nativeResult.Model.Architecture)
	core.AssertEqual(t, "native", nativeResult.Model.Labels["source"])
}

func TestNativeContract_AdapterLifecycle_Good(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{}}
	identity, err := model.LoadAdapter("domain.safetensors")
	if err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}
	if identity.Path != "domain.safetensors" || identity.Format != "lora" {
		t.Fatalf("adapter identity = %+v, want lora path", identity)
	}
	if model.ActiveAdapter().Path != "domain.safetensors" {
		t.Fatalf("active adapter = %+v, want loaded adapter", model.ActiveAdapter())
	}
	if err := model.UnloadAdapter(); err != nil {
		t.Fatalf("UnloadAdapter: %v", err)
	}
	if !adapterIdentityIsZero(model.ActiveAdapter()) {
		t.Fatalf("active adapter after unload = %+v, want zero", model.ActiveAdapter())
	}
}

func TestNativeContract_AdapterIdentityClonedAtPublicBoundary_Good(t *testing.T) {
	native := &fakeNativeModel{
		loadAdapterIdentity: inference.AdapterIdentity{
			Path:       "domain.safetensors",
			Format:     "lora",
			TargetKeys: []string{"output.weight"},
			Labels:     map[string]string{"adapter_runtime": "hip_tiny_loaded"},
		},
	}
	model := &rocmModel{native: native}

	loaded, err := model.LoadAdapter("domain.safetensors")
	core.RequireNoError(t, err)
	loaded.TargetKeys[0] = "mutated"
	loaded.Labels["adapter_runtime"] = "mutated"
	native.adapter.TargetKeys[0] = "native-mutated"
	native.adapter.Labels["adapter_runtime"] = "native-mutated"

	active := model.ActiveAdapter()
	core.AssertEqual(t, "output.weight", active.TargetKeys[0])
	core.AssertEqual(t, "hip_tiny_loaded", active.Labels["adapter_runtime"])

	active.TargetKeys[0] = "active-mutated"
	active.Labels["adapter_runtime"] = "active-mutated"
	again := model.ActiveAdapter()
	core.AssertEqual(t, "output.weight", again.TargetKeys[0])
	core.AssertEqual(t, "hip_tiny_loaded", again.Labels["adapter_runtime"])
}

func TestNativeContract_ActiveAdapterClonesNativeFallback_Good(t *testing.T) {
	native := &fakeNativeModel{adapter: inference.AdapterIdentity{
		Path:       "native.safetensors",
		Format:     "lora",
		TargetKeys: []string{"score.weight"},
		Labels:     map[string]string{"adapter_runtime": "hip_bert_classifier"},
	}}
	model := &rocmModel{native: native}

	active := model.ActiveAdapter()
	active.TargetKeys[0] = "mutated"
	active.Labels["adapter_runtime"] = "mutated"

	core.AssertEqual(t, "score.weight", native.adapter.TargetKeys[0])
	core.AssertEqual(t, "hip_bert_classifier", native.adapter.Labels["adapter_runtime"])
}

func TestNativeContract_HIPLoadedModelActiveAdapterClonesIdentity_Good(t *testing.T) {
	loaded := &hipLoadedModel{adapter: inference.AdapterIdentity{
		Path:       "classifier-lora.json",
		Format:     rocmClassifierLoRAFormat,
		TargetKeys: []string{"classifier.weight"},
		Labels:     map[string]string{"adapter_runtime": "hip_bert_classifier"},
	}}

	active := loaded.ActiveAdapter()
	active.TargetKeys[0] = "mutated"
	active.Labels["adapter_runtime"] = "mutated"

	again := loaded.ActiveAdapter()
	core.AssertEqual(t, "classifier.weight", again.TargetKeys[0])
	core.AssertEqual(t, "hip_bert_classifier", again.Labels["adapter_runtime"])
}

func TestNativeContract_CloseGoodIdempotentClearsRuntimeState(t *testing.T) {
	native := &fakeNativeModel{adapter: inference.AdapterIdentity{Path: "domain.safetensors", Format: "lora"}}
	model := &rocmModel{
		native:    native,
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		adapter:   inference.AdapterIdentity{Path: "domain.safetensors", Format: "lora"},
		cache:     NewBlockCacheService(BlockCacheConfig{}),
	}
	model.setLastFailure(core.NewError("stale failure"))

	core.AssertNoError(t, resultError(model.Close()))
	core.AssertNoError(t, resultError(model.Close()))

	core.AssertEqual(t, 1, native.closeCalls)
	if !adapterIdentityIsZero(model.ActiveAdapter()) {
		t.Fatalf("active adapter = %+v, want zero after close", model.ActiveAdapter())
	}
	if model.cache != nil {
		t.Fatalf("cache service should be cleared after close")
	}
	if report := model.Capabilities(); report.Available {
		t.Fatalf("capability report = %+v, want unavailable after close", report)
	}
	if resultError(model.Err()) != nil {
		t.Fatalf("Close success Err() = %v, want nil", resultError(model.Err()))
	}
}

func TestNativeContract_CloseBadStateCloseFailureKeepsRuntime_Bad(t *testing.T) {
	native := &fakeNativeModel{}
	state := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, &failingStateRuntime{err: core.NewError("close failed")})
	model := &rocmModel{
		native: native,
		state:  state,
	}

	err := resultError(model.Close())

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "close failed")
	if resultError(model.Err()) == nil {
		t.Fatal("Close state failure Err() = nil")
	}
	core.AssertContains(t, resultError(model.Err()).Error(), "close failed")
	core.AssertEqual(t, 0, native.closeCalls)
	if model.native != native {
		t.Fatal("native model was cleared after state close failure")
	}
	if model.state != state {
		t.Fatal("state session was cleared after state close failure")
	}
}

func TestNativeContract_CloseBadNativeCloseFailureKeepsRuntime_Bad(t *testing.T) {
	native := &fakeNativeModel{closeErr: core.NewError("native close failed")}
	model := &rocmModel{native: native}

	err := resultError(model.Close())

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native close failed")
	if resultError(model.Err()) == nil {
		t.Fatal("Close native failure Err() = nil")
	}
	core.AssertContains(t, resultError(model.Err()).Error(), "native close failed")
	core.AssertEqual(t, 1, native.closeCalls)
	if model.native != native {
		t.Fatal("native model was cleared after native close failure")
	}
}

func TestNativeContract_LoadAdapterBadEmptyPathDoesNotCallNative_Bad(t *testing.T) {
	native := &fakeNativeModel{}
	model := &rocmModel{native: native}

	identity, err := model.LoadAdapter(" \t")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "adapter path is required")
	if !adapterIdentityIsZero(identity) {
		t.Fatalf("identity = %+v, want zero", identity)
	}
	core.AssertEqual(t, 0, len(native.adapterLoads))
	if !adapterIdentityIsZero(model.ActiveAdapter()) {
		t.Fatalf("active adapter = %+v, want zero", model.ActiveAdapter())
	}
}

func TestNativeContract_LoadAdapterBadNativeFailureKeepsActiveAdapter_Bad(t *testing.T) {
	native := &fakeNativeModel{adapterErr: core.NewError("adapter failed")}
	model := &rocmModel{
		native:  native,
		adapter: inference.AdapterIdentity{Path: "previous.safetensors", Format: "lora"},
	}

	identity, err := model.LoadAdapter("next.safetensors")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "adapter failed")
	if !adapterIdentityIsZero(identity) {
		t.Fatalf("identity = %+v, want zero", identity)
	}
	core.AssertEqual(t, []string{"next.safetensors"}, native.adapterLoads)
	if got := model.ActiveAdapter(); got.Path != "previous.safetensors" || got.Format != "lora" {
		t.Fatalf("active adapter = %+v, want previous adapter", got)
	}
}

func TestNativeContract_LoadAdapterBadRecordsErrAndSuccessClears_Bad(t *testing.T) {
	native := &fakeNativeModel{adapterErr: core.NewError("adapter failed")}
	model := &rocmModel{native: native}

	_, err := model.LoadAdapter("broken.safetensors")

	core.AssertError(t, err)
	if resultError(model.Err()) == nil {
		t.Fatal("LoadAdapter failure Err() = nil")
	}
	core.AssertContains(t, resultError(model.Err()).Error(), "adapter failed")

	native.adapterErr = nil
	identity, err := model.LoadAdapter("domain.safetensors")

	core.RequireNoError(t, err)
	core.AssertEqual(t, "domain.safetensors", identity.Path)
	if resultError(model.Err()) != nil {
		t.Fatalf("LoadAdapter success Err() = %v, want nil", resultError(model.Err()))
	}
}

func TestNativeContract_LoadAdapterBadStateCloseFailureDoesNotCallNative_Bad(t *testing.T) {
	native := &fakeNativeModel{}
	state := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, &failingStateRuntime{err: core.NewError("close failed")})
	model := &rocmModel{
		native:  native,
		adapter: inference.AdapterIdentity{Path: "previous.safetensors", Format: "lora"},
		state:   state,
	}

	identity, err := model.LoadAdapter("next.safetensors")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "close state runtime")
	if !adapterIdentityIsZero(identity) {
		t.Fatalf("identity = %+v, want zero", identity)
	}
	core.AssertEqual(t, 0, len(native.adapterLoads))
	if got := model.ActiveAdapter(); got.Path != "previous.safetensors" || got.Format != "lora" {
		t.Fatalf("active adapter = %+v, want previous adapter", got)
	}
	if model.state != state {
		t.Fatal("state session was cleared after load-adapter state close failure")
	}
}

func TestNativeContract_UnloadAdapterBadStateCloseFailureDoesNotCallNative_Bad(t *testing.T) {
	native := &fakeNativeModel{adapter: inference.AdapterIdentity{Path: "previous.safetensors", Format: "lora"}}
	state := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, &failingStateRuntime{err: core.NewError("close failed")})
	model := &rocmModel{
		native:  native,
		adapter: inference.AdapterIdentity{Path: "previous.safetensors", Format: "lora"},
		state:   state,
	}

	err := model.UnloadAdapter()

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "close state runtime")
	core.AssertEqual(t, 0, native.unloadCalls)
	if got := model.ActiveAdapter(); got.Path != "previous.safetensors" || got.Format != "lora" {
		t.Fatalf("active adapter = %+v, want previous adapter", got)
	}
	if model.state != state {
		t.Fatal("state session was cleared after unload-adapter state close failure")
	}
}

func TestNativeContract_UnloadAdapterBadRecordsErrAndSuccessClears_Bad(t *testing.T) {
	adapter := inference.AdapterIdentity{Path: "previous.safetensors", Format: "lora"}
	native := &fakeNativeModel{adapter: adapter, unloadAdapterErr: core.NewError("unload failed")}
	model := &rocmModel{native: native, adapter: adapter}

	err := model.UnloadAdapter()

	core.AssertError(t, err)
	if resultError(model.Err()) == nil {
		t.Fatal("UnloadAdapter failure Err() = nil")
	}
	core.AssertContains(t, resultError(model.Err()).Error(), "unload failed")
	if got := model.ActiveAdapter(); got.Path != "previous.safetensors" {
		t.Fatalf("active adapter = %+v, want previous adapter after failed unload", got)
	}

	native.unloadAdapterErr = nil
	err = model.UnloadAdapter()

	core.RequireNoError(t, err)
	if resultError(model.Err()) != nil {
		t.Fatalf("UnloadAdapter success Err() = %v, want nil", resultError(model.Err()))
	}
	if !adapterIdentityIsZero(model.ActiveAdapter()) {
		t.Fatalf("active adapter = %+v, want zero after successful unload", model.ActiveAdapter())
	}
}

func TestNativeContract_BenchmarkAndEvaluateUseModelSurface_Ugly(t *testing.T) {
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "a"}, {ID: 2, Text: "b"}}},
	}
	bench, err := model.Benchmark(context.Background(), inference.BenchConfig{Prompts: []string{"hi"}, MaxTokens: 2, MeasuredRuns: 1})
	if err != nil {
		t.Fatalf("Benchmark: %v", err)
	}
	if bench.GeneratedTokens != 2 || bench.DecodeTokensPerSec == 0 {
		t.Fatalf("bench = %+v, want generated token throughput", bench)
	}
	if bench.PromptCacheHitRate < 0 || bench.KVRestoreMilliseconds < 0 {
		t.Fatalf("bench = %+v, want shared cache fields populated", bench)
	}
	if bench.Labels["scheduler"] != "supported" || bench.Labels["cache.blocks"] != "experimental" || bench.Labels["cache.disk"] != "experimental" || bench.Labels["prompt.cache"] != "experimental" || bench.Labels["probe.events"] != "stream_tokens" || bench.Labels["queue_latency_ms"] == "" || bench.Labels["first_token_latency_ms"] == "" || bench.Labels["kernel_status"] != hipKernelStatusNotLinked {
		t.Fatalf("bench labels = %+v, want ROCm parity probe/cache/scheduler fields", bench.Labels)
	}

	eval, err := model.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{Text: "hello world"}}, inference.EvalConfig{MaxSamples: 1})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if eval.Metrics.Samples != 1 || eval.Metrics.Tokens == 0 {
		t.Fatalf("eval = %+v, want token counts", eval)
	}
	if eval.Labels["loss"] != "unsupported_until_prefill_kernels" ||
		eval.Labels["perplexity"] != "unsupported_until_prefill_kernels" ||
		eval.Labels["kernel_status"] != hipKernelStatusNotLinked ||
		eval.Labels["loss_kernel"] != hipKernelStatusNotLinked ||
		eval.Labels["loss_kernel_name"] != hipKernelNameCrossEntropy ||
		eval.Labels["loss_scope"] != "toy_cross_entropy" {
		t.Fatalf("eval labels = %+v, want explicit unsupported loss/perplexity labels", eval.Labels)
	}
}

func TestNativeContract_BenchmarkWarmupRunsAllPromptsWithoutMeasuredCounters_Good(t *testing.T) {
	native := &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "a"}}}
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    native,
	}

	bench, err := model.Benchmark(context.Background(), inference.BenchConfig{
		Prompts:      []string{"first", "second"},
		MaxTokens:    1,
		WarmupRuns:   2,
		MeasuredRuns: 1,
	})

	core.RequireNoError(t, err)
	if bench.PromptTokens != 2 || bench.GeneratedTokens != 2 {
		t.Fatalf("bench = %+v, want only measured prompt/token counters", bench)
	}
	if got := native.generatePrompts; len(got) != 6 || got[0] != "first" || got[1] != "second" || got[4] != "first" || got[5] != "second" {
		t.Fatalf("generate prompts = %+v, want warmup and measured runs across all prompts", got)
	}
	if bench.Labels["warmup_runs"] != "2" || bench.Labels["measured_runs"] != "1" || bench.Labels["prompt_count"] != "2" {
		t.Fatalf("bench labels = %+v, want benchmark run-shape labels", bench.Labels)
	}
	if bench.Labels["probe_count"] != "4" || bench.Labels["probe_count_status"] != "measured" {
		t.Fatalf("bench labels = %+v, want measured probe count excluding warmups", bench.Labels)
	}
	if metrics := model.Metrics(); metrics.GeneratedTokens != 2 {
		t.Fatalf("metrics = %+v, want measured aggregate after warmups", metrics)
	}
}

func TestNativeContract_GeneratedPromptUsesExplicitGemma4Q4TextMode_Good(t *testing.T) {
	model := &rocmModel{
		native: &hipLoadedModel{
			modelInfo:   inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 4},
			modelLabels: linkedGemma4TestLabels("E2B", "q4"),
			tokenText:   &hipTokenTextDecoder{},
		},
	}

	core.AssertEqual(t, "text:hello", model.generatedPrompt("hello"))
	core.AssertEqual(t, "text:hello", model.generatedPrompt("text:hello"))
	core.AssertEqual(t, "tokens:1", model.generatedPrompt("tokens:1"))
	core.AssertEqual(t, "hello", (&rocmModel{}).generatedPrompt("hello"))
}

func TestNativeContract_BenchmarkReportsMeasuredLatencyLabels_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native: &fakeNativeModel{
			tokens:         []inference.Token{{ID: 1, Text: "a"}, {ID: 2, Text: "b"}},
			baseTokenDelay: 2 * time.Millisecond,
		},
	}

	bench, err := model.Benchmark(context.Background(), inference.BenchConfig{
		Prompts:      []string{"first prompt", "second prompt"},
		MaxTokens:    2,
		MeasuredRuns: 2,
	})

	core.RequireNoError(t, err)
	if bench.Labels["operation_count"] != "4" {
		t.Fatalf("bench labels = %+v, want operation count across prompts and measured runs", bench.Labels)
	}
	for _, key := range []string{"first_token_latency_ms", "prefill_duration_ms", "decode_duration_ms", "total_duration_ms"} {
		if got := positiveFloatLabel(t, bench.Labels, key); got <= 0 {
			t.Fatalf("bench labels[%s] = %q, want positive measured latency/duration", key, bench.Labels[key])
		}
	}
	if got := floatLabel(t, bench.Labels, "queue_latency_ms"); got != 0 {
		t.Fatalf("queue latency = %v, want direct benchmark path to report no scheduler queue", got)
	}
}

func TestNativeContract_BenchmarkDecodeHelperStatusUsesQ4Generate_Good(t *testing.T) {
	core.AssertEqual(t, "planned", rocmDecodeHelperStatusLabel(defaultHIPKernelStatus(), false))
	core.AssertEqual(t, "experimental", rocmDecodeHelperStatusLabel(defaultHIPKernelStatus(), true))
	core.AssertEqual(t, "experimental", rocmDecodeHelperStatusLabel(hipKernelStatus{Decode: hipKernelStatusLinked}, false))
}

func TestNativeContract_BenchmarkLabelsAttachedDrafterHelperForGemma4Affine_Good(t *testing.T) {
	labels := map[string]string{}

	rocmAddGemma4AttachedDrafterBenchmarkLabels(labels)

	core.AssertEqual(t, "experimental", labels["attached.drafter.decode"])
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["attached.drafter.native_attachment"])
	core.AssertEqual(t, "gemma4_assistant", labels["attached.drafter.role"])
	core.AssertEqual(t, "gemma4_mlx_affine_generate", labels["attached.drafter.source"])
	core.AssertEqual(t, hipKernelStatusLinked, labels["attached.drafter.retained_state_entrypoint"])
	core.AssertEqual(t, "true", labels["attached.drafter.retained_state_required"])
	core.AssertEqual(t, "rocm_state_session_runtime_kv", labels["attached.drafter.state_source"])
	core.AssertEqual(t, "forbidden", labels["attached.drafter.prompt_replay_fallback"])
	core.AssertEqual(t, hipKernelStatusLinked, labels["attached.drafter.target_retained_decode"])
	core.AssertEqual(t, hipKernelStatusLinked, labels["attached.drafter.target_retained_state_decode"])
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["attached.drafter.assistant_verify"])
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["attached.drafter.assistant_state_verify"])
	core.AssertEqual(t, attachedDrafterNativeHandoffTargetDecodeOnly, labels["attached.drafter.native_handoff"])
	core.AssertEqual(t, officialGemma4E2BAssistantArchitecture, labels["attached.drafter.assistant_architecture"])
	core.AssertEqual(t, "true", labels["attached.drafter.assistant_ordered_embeddings"])
	core.AssertEqual(t, "true", labels["attached.drafter.assistant_four_layer_drafter"])
	core.AssertEqual(t, "2048", labels["attached.drafter.assistant_centroids"])
	core.AssertEqual(t, "32", labels["attached.drafter.assistant_centroid_intermediate_top_k"])
	core.AssertEqual(t, "int64", labels["attached.drafter.assistant_token_ordering_dtype"])
	core.AssertEqual(t, "2048x128", labels["attached.drafter.assistant_token_ordering_shape"])
	core.AssertEqual(t, officialGemma4E2BAssistantModelID, labels["attached.drafter.official_assistant_model_id"])
	core.AssertEqual(t, officialGemma4E2BAssistantRevision, labels["attached.drafter.official_assistant_revision"])
	core.AssertEqual(t, officialGemma4E2BTargetModelID, labels["attached.drafter.official_target_model_id"])
	core.AssertEqual(t, officialGemma4E2BTargetRevision, labels["attached.drafter.official_target_revision"])
	core.AssertEqual(t, "true", labels["attached.drafter.official_pair_verified"])
	core.AssertEqual(t, "true", labels["attached.drafter.gemma4_family_pair_verified"])
	core.AssertEqual(t, ProductionLaneCurrentModelID, labels["attached.drafter.target.production_quant_model"])
	core.AssertEqual(t, ProductionLaneModelID, labels["attached.drafter.target.production_quant_locked_model"])
	core.AssertEqual(t, "gemma4", labels["attached.drafter.target.engine_profile"])
	core.AssertEqual(t, "gemma4_text", labels["attached.drafter.target.engine_architecture_profile"])
	core.AssertEqual(t, string(inference.FeatureRuntimeNative), labels["attached.drafter.target.engine_architecture_runtime_status"])
	core.AssertEqual(t, "gemma", labels["attached.drafter.target.engine_architecture_reasoning_parser"])
	core.AssertEqual(t, "q8,paged,k-q8-v-q4,retained-state", labels["attached.drafter.target.engine_architecture_cache_hints"])
	core.AssertEqual(t, "gemma4_hf_turn", labels["attached.drafter.target.engine_chat_template"])
	core.AssertEqual(t, "q_proj,v_proj,o_proj", labels["attached.drafter.target.gemma4_lora_default_targets"])
	core.AssertEqual(t, "model_registry", labels["attached.drafter.target.gemma4_weight_policy"])
	core.AssertEqual(t, officialGemma4E2BAssistantModelID, labels["attached.drafter.assistant.production_quant_model"])
	core.AssertEqual(t, officialGemma4E2BAssistantModelID, labels["attached.drafter.assistant.production_quant_assistant_model"])
	core.AssertEqual(t, "E2B:assistant-bf16", labels["attached.drafter.assistant.production_quant_pack"])
	core.AssertEqual(t, "mtp-assistant", labels["attached.drafter.assistant.production_quant_tier"])
	core.AssertEqual(t, "true", labels["attached.drafter.assistant.production_quant_mtp_assistant"])
	core.AssertEqual(t, "gemma4", labels["attached.drafter.assistant.production_quant_target_family"])
	core.AssertEqual(t, "gemma4", labels["attached.drafter.assistant.engine_profile"])
	core.AssertEqual(t, "gemma4_assistant", labels["attached.drafter.assistant.engine_architecture_profile"])
	core.AssertEqual(t, string(inference.FeatureRuntimeNative), labels["attached.drafter.assistant.engine_architecture_runtime_status"])
	core.AssertEqual(t, "retained-state,attached-drafter", labels["attached.drafter.assistant.engine_architecture_cache_hints"])
	core.AssertEqual(t, "true", labels["attached.drafter.assistant.engine_architecture_attached_only"])
	core.AssertEqual(t, "false", labels["attached.drafter.assistant.engine_architecture_generation"])
	core.AssertEqual(t, "", labels["attached.drafter.assistant.gemma4_lora_default_targets"])
	core.AssertEqual(t, "", labels["attached.drafter.assistant.gemma4_weight_policy"])
	core.AssertEqual(t, productionMTPDefaultDraftTokensLabel, labels["attached.drafter.speculative_draft_tokens"])
}

func TestNativeContract_BenchmarkLabelsAttachedDrafterRejectsNonOfficialPair_Bad(t *testing.T) {
	labels := map[string]string{}

	rocmAddGemma4AttachedDrafterBenchmarkLabels(labels, inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-12b-it-6bit",
		Architecture: "gemma4_text",
		NumLayers:    48,
		HiddenSize:   3840,
		VocabSize:    262144,
		QuantBits:    6,
	}, officialGemma4E2BBF16AssistantIdentity())

	core.AssertEqual(t, "12B", labels["attached.drafter.target.gemma4_size"])
	core.AssertEqual(t, "q6", labels["attached.drafter.target.gemma4_quant_mode"])
	core.AssertEqual(t, "64", labels["attached.drafter.target.gemma4_quant_group"])
	core.AssertEqual(t, "mlx-community/gemma-4-12b-it-6bit", labels["attached.drafter.target.production_quant_model"])
	core.AssertEqual(t, "", labels["attached.drafter.target.production_quant_locked_model"])
	core.AssertEqual(t, "E2B", labels["attached.drafter.assistant.gemma4_size"])
	core.AssertEqual(t, "bf16", labels["attached.drafter.assistant.gemma4_quant_mode"])
	core.AssertEqual(t, officialGemma4E2BAssistantModelID, labels["attached.drafter.assistant.production_quant_model"])
	core.AssertEqual(t, "E2B:assistant-bf16", labels["attached.drafter.assistant.production_quant_pack"])
	core.AssertEqual(t, "true", labels["attached.drafter.assistant.production_quant_mtp_assistant"])
	core.AssertEqual(t, "false", labels["attached.drafter.official_pair_verified"])
	core.AssertEqual(t, "false", labels["attached.drafter.gemma4_family_pair_verified"])
}

func TestNativeContract_BenchmarkLabelsAttachedDrafterInfersAssistantFromTarget_Good(t *testing.T) {
	labels := map[string]string{}

	rocmAddGemma4AttachedDrafterBenchmarkLabels(labels, inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-12b-it-6bit",
		Architecture: "gemma4_text",
		NumLayers:    48,
		HiddenSize:   3840,
		VocabSize:    262144,
		QuantBits:    6,
	})

	core.AssertEqual(t, "12B", labels["attached.drafter.target.gemma4_size"])
	core.AssertEqual(t, "q6", labels["attached.drafter.target.gemma4_quant_mode"])
	core.AssertEqual(t, "64", labels["attached.drafter.target.gemma4_quant_group"])
	core.AssertEqual(t, "mlx-community/gemma-4-12b-it-6bit", labels["attached.drafter.target.production_quant_model"])
	core.AssertEqual(t, "", labels["attached.drafter.target.production_quant_locked_model"])
	core.AssertEqual(t, "12B", labels["attached.drafter.assistant.gemma4_size"])
	core.AssertEqual(t, "bf16", labels["attached.drafter.assistant.gemma4_quant_mode"])
	core.AssertEqual(t, rocmGemma4MTPAssistantPath("12B", "bf16"), labels["attached.drafter.assistant.production_quant_model"])
	core.AssertEqual(t, "12B:assistant-bf16", labels["attached.drafter.assistant.production_quant_pack"])
	core.AssertEqual(t, "true", labels["attached.drafter.assistant.production_quant_mtp_assistant"])
	core.AssertEqual(t, "false", labels["attached.drafter.official_pair_verified"])
	core.AssertEqual(t, "true", labels["attached.drafter.gemma4_family_pair_verified"])
}

func TestNativeContract_CapabilityLabelsAttachedDrafterEvidence_Good(t *testing.T) {
	labels := map[string]string{}

	rocmAddGemma4AttachedDrafterCapabilityLabels(labels)

	core.AssertEqual(t, hipKernelStatusLinked, labels["attached_drafter_helper"])
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["attached_drafter_native_attachment"])
	core.AssertEqual(t, "gemma4_assistant", labels["attached_drafter_role"])
	core.AssertEqual(t, "gemma4_mlx_affine_generate", labels["attached_drafter_source"])
	core.AssertEqual(t, hipKernelStatusLinked, labels["attached_drafter_retained_state_entrypoint"])
	core.AssertEqual(t, "true", labels["attached_drafter_retained_state_required"])
	core.AssertEqual(t, "rocm_state_session_runtime_kv", labels["attached_drafter_state_source"])
	core.AssertEqual(t, "forbidden", labels["attached_drafter_prompt_replay_fallback"])
	core.AssertEqual(t, hipKernelStatusLinked, labels["attached_drafter_target_retained_decode"])
	core.AssertEqual(t, hipKernelStatusLinked, labels["attached_drafter_target_retained_state_decode"])
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["attached_drafter_assistant_verify"])
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["attached_drafter_assistant_state_verify"])
	core.AssertEqual(t, attachedDrafterNativeHandoffTargetDecodeOnly, labels["attached_drafter_native_handoff"])
	core.AssertEqual(t, officialGemma4E2BAssistantArchitecture, labels["attached_drafter_assistant_architecture"])
	core.AssertEqual(t, "true", labels["attached_drafter_assistant_ordered_embeddings"])
	core.AssertEqual(t, "true", labels["attached_drafter_assistant_four_layer_drafter"])
	core.AssertEqual(t, "2048", labels["attached_drafter_assistant_centroids"])
	core.AssertEqual(t, "32", labels["attached_drafter_assistant_centroid_intermediate_top_k"])
	core.AssertEqual(t, "int64", labels["attached_drafter_assistant_token_ordering_dtype"])
	core.AssertEqual(t, "2048x128", labels["attached_drafter_assistant_token_ordering_shape"])
	core.AssertEqual(t, officialGemma4E2BAssistantModelID, labels["attached_drafter_official_assistant_model_id"])
	core.AssertEqual(t, officialGemma4E2BAssistantRevision, labels["attached_drafter_official_assistant_revision"])
	core.AssertEqual(t, officialGemma4E2BTargetModelID, labels["attached_drafter_official_target_model_id"])
	core.AssertEqual(t, officialGemma4E2BTargetRevision, labels["attached_drafter_official_target_revision"])
	core.AssertEqual(t, "true", labels["attached_drafter_official_pair_verified"])
	core.AssertEqual(t, "true", labels["attached_drafter_gemma4_family_pair_verified"])
	core.AssertEqual(t, ProductionLaneCurrentModelID, labels["attached_drafter_target_production_quant_model"])
	core.AssertEqual(t, ProductionLaneModelID, labels["attached_drafter_target_production_quant_locked_model"])
	core.AssertEqual(t, officialGemma4E2BAssistantModelID, labels["attached_drafter_assistant_production_quant_model"])
	core.AssertEqual(t, officialGemma4E2BAssistantModelID, labels["attached_drafter_assistant_production_quant_assistant_model"])
	core.AssertEqual(t, "E2B:assistant-bf16", labels["attached_drafter_assistant_production_quant_pack"])
	core.AssertEqual(t, "mtp-assistant", labels["attached_drafter_assistant_production_quant_tier"])
	core.AssertEqual(t, "true", labels["attached_drafter_assistant_production_quant_mtp_assistant"])
	core.AssertEqual(t, "gemma4", labels["attached_drafter_assistant_production_quant_target_family"])
	core.AssertEqual(t, "gemma4", labels["attached_drafter_target_engine_profile"])
	core.AssertEqual(t, "gemma4_text", labels["attached_drafter_target_engine_architecture_profile"])
	core.AssertEqual(t, string(inference.FeatureRuntimeNative), labels["attached_drafter_target_engine_architecture_runtime_status"])
	core.AssertEqual(t, "gemma", labels["attached_drafter_target_engine_architecture_reasoning_parser"])
	core.AssertEqual(t, "q8,paged,k-q8-v-q4,retained-state", labels["attached_drafter_target_engine_architecture_cache_hints"])
	core.AssertEqual(t, "gemma4_hf_turn", labels["attached_drafter_target_engine_chat_template"])
	core.AssertEqual(t, "q_proj,v_proj,o_proj", labels["attached_drafter_target_gemma4_lora_default_targets"])
	core.AssertEqual(t, "model_registry", labels["attached_drafter_target_gemma4_weight_policy"])
	core.AssertEqual(t, "gemma4", labels["attached_drafter_assistant_engine_profile"])
	core.AssertEqual(t, "gemma4_assistant", labels["attached_drafter_assistant_engine_architecture_profile"])
	core.AssertEqual(t, string(inference.FeatureRuntimeNative), labels["attached_drafter_assistant_engine_architecture_runtime_status"])
	core.AssertEqual(t, "retained-state,attached-drafter", labels["attached_drafter_assistant_engine_architecture_cache_hints"])
	core.AssertEqual(t, "true", labels["attached_drafter_assistant_engine_architecture_attached_only"])
	core.AssertEqual(t, "false", labels["attached_drafter_assistant_engine_architecture_generation"])
	core.AssertEqual(t, "", labels["attached_drafter_assistant_gemma4_lora_default_targets"])
	core.AssertEqual(t, "", labels["attached_drafter_assistant_gemma4_weight_policy"])
	core.AssertEqual(t, productionMTPDefaultDraftTokensLabel, labels["attached_drafter_speculative_draft_tokens"])

	speculative := rocmGemma4Q4SpeculativeDecodeCapabilityLabels(productionMTPE2BQ6TargetModel().modelIdentity())
	core.AssertEqual(t, ROCmAttachedDrafterRegistryContract, speculative["engine_attached_drafter_route_contract"])
	core.AssertEqual(t, hipKernelStatusNotLinked, speculative["engine_attached_drafter_native_attachment"])
	core.AssertEqual(t, "true", speculative["engine_attached_drafter_retained_state_required"])
	core.AssertEqual(t, "rocm_state_session_runtime_kv", speculative["engine_attached_drafter_state_source"])
	core.AssertEqual(t, "forbidden", speculative["engine_attached_drafter_prompt_replay_fallback"])
	core.AssertEqual(t, officialGemma4E2BAssistantArchitecture, speculative["engine_attached_drafter_assistant_architecture"])
	core.AssertEqual(t, productionMTPAssistantTokenOrderingShapeLabel, speculative["engine_attached_drafter_assistant_token_ordering_shape"])
}

func TestNativeContract_CapabilityLabelsAttachedDrafterInfersAssistantFromTarget_Good(t *testing.T) {
	labels := map[string]string{}

	rocmAddGemma4AttachedDrafterCapabilityLabels(labels, inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-e4b-it-8bit",
		Architecture: "gemma4_text",
		NumLayers:    26,
		HiddenSize:   2304,
		VocabSize:    262144,
		QuantBits:    8,
	})

	core.AssertEqual(t, "E4B", labels["attached_drafter_target_gemma4_size"])
	core.AssertEqual(t, "q8", labels["attached_drafter_target_gemma4_quant_mode"])
	core.AssertEqual(t, "64", labels["attached_drafter_target_gemma4_quant_group"])
	core.AssertEqual(t, "lmstudio-community/gemma-4-E4B-it-MLX-8bit", labels["attached_drafter_target_production_quant_model"])
	core.AssertEqual(t, "", labels["attached_drafter_target_production_quant_locked_model"])
	core.AssertEqual(t, "E4B", labels["attached_drafter_assistant_gemma4_size"])
	core.AssertEqual(t, "bf16", labels["attached_drafter_assistant_gemma4_quant_mode"])
	core.AssertEqual(t, rocmGemma4MTPAssistantPath("E4B", "bf16"), labels["attached_drafter_assistant_production_quant_model"])
	core.AssertEqual(t, "E4B:assistant-bf16", labels["attached_drafter_assistant_production_quant_pack"])
	core.AssertEqual(t, "true", labels["attached_drafter_assistant_production_quant_mtp_assistant"])
	core.AssertEqual(t, "gemma4_text", labels["attached_drafter_target_engine_architecture_profile"])
	core.AssertEqual(t, "gemma4_assistant", labels["attached_drafter_assistant_engine_architecture_profile"])
	core.AssertEqual(t, string(inference.FeatureRuntimeNative), labels["attached_drafter_assistant_engine_architecture_runtime_status"])
	core.AssertEqual(t, "retained-state,attached-drafter", labels["attached_drafter_assistant_engine_architecture_cache_hints"])
	core.AssertEqual(t, "true", labels["attached_drafter_assistant_engine_architecture_attached_only"])
	core.AssertEqual(t, "false", labels["attached_drafter_official_pair_verified"])
	core.AssertEqual(t, "true", labels["attached_drafter_gemma4_family_pair_verified"])
}

func TestNativeContract_BenchmarkAndEvaluateTinyFixtureLabelsProductionPending_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 3, HiddenSize: 2},
		native: &fakeNativeModel{
			tokens: []inference.Token{{ID: 1, Text: "a"}},
			kernelStatus: hipKernelStatus{
				Decode:    hipKernelStatusLinked,
				Prefill:   hipKernelStatusLinked,
				KVCache:   hipKernelStatusPlanned,
				Reason:    "fake tiny fixture",
				LoRA:      hipKernelStatusLinked,
				Rerank:    hipKernelStatusNotLinked,
				Embedding: hipKernelStatusNotLinked,
			},
		},
	}

	bench, err := model.Benchmark(context.Background(), inference.BenchConfig{Prompts: []string{"hi"}, MaxTokens: 1, MeasuredRuns: 1})
	core.RequireNoError(t, err)
	if bench.Labels["kernel_scope"] != "toy_tiny_fixture" ||
		bench.Labels["decode_kernel_name"] != hipKernelNameTinyDecode ||
		bench.Labels["prefill_kernel_name"] != hipKernelNameTinyPrefill ||
		bench.Labels["production_decode"] != hipKernelStatusNotLinked ||
		bench.Labels["production_prefill"] != hipKernelStatusNotLinked {
		t.Fatalf("bench labels = %+v, want tiny fixture kernel scope and production pending labels", bench.Labels)
	}

	eval, err := model.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{Text: "hello world"}}, inference.EvalConfig{
		MaxSamples: 1,
		Probes:     []inference.QualityProbe{{Name: "tiny-decode", Prompt: "hi"}},
	})
	core.RequireNoError(t, err)
	if eval.Labels["kernel_scope"] != "toy_tiny_fixture" ||
		eval.Labels["decode_kernel_name"] != hipKernelNameTinyDecode ||
		eval.Labels["prefill_kernel_name"] != hipKernelNameTinyPrefill ||
		eval.Labels["production_decode"] != hipKernelStatusNotLinked ||
		eval.Labels["production_prefill"] != hipKernelStatusNotLinked ||
		eval.Labels["loss_status"] != "not_requested" ||
		eval.Labels["quality_probe_status"] != "passed" {
		t.Fatalf("eval labels = %+v, want tiny fixture decode/prefill scope and production pending labels", eval.Labels)
	}
}

func TestNativeContract_BenchmarkAndEvaluateUgly_NilContext(t *testing.T) {
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "a"}}},
	}

	bench, err := model.Benchmark(nil, inference.BenchConfig{Prompts: []string{"hi"}, MaxTokens: 1, MeasuredRuns: 1})
	if err != nil {
		t.Fatalf("Benchmark(nil context): %v", err)
	}
	if bench.GeneratedTokens != 1 {
		t.Fatalf("bench = %+v, want generated token", bench)
	}
	eval, err := model.Evaluate(nil, &singleInferenceSample{sample: inference.DatasetSample{Text: "hello"}}, inference.EvalConfig{
		MaxSamples: 1,
		Probes:     []inference.QualityProbe{{Name: "nil-context", Prompt: "hi"}},
	})
	if err != nil {
		t.Fatalf("Evaluate(nil context): %v", err)
	}
	if eval.Metrics.Samples != 1 || len(eval.Probes) != 1 {
		t.Fatalf("eval = %+v, want sample and probe", eval)
	}
}

func TestNativeContract_BenchmarkBad_PropagatesCacheStatsError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native: &fakeNativeModel{
			tokens:      []inference.Token{{ID: 1, Text: "a"}},
			afterStream: cancel,
		},
	}

	bench, err := model.Benchmark(ctx, inference.BenchConfig{Prompts: []string{"hi"}, MaxTokens: 1, MeasuredRuns: 1})

	core.AssertError(t, err)
	core.AssertNil(t, bench)
	core.AssertContains(t, err.Error(), "context canceled")
	if !errors.Is(resultError(model.Err()), context.Canceled) {
		t.Fatalf("Benchmark Err() = %v, want context.Canceled", resultError(model.Err()))
	}
}

func TestNativeContract_BenchmarkMeasuresActiveLoRAOverhead_Good(t *testing.T) {
	native := &fakeNativeModel{
		tokens: []inference.Token{{ID: 1, Text: "a"}},
		adapter: inference.AdapterIdentity{
			Path:   "tiny-lora.json",
			Hash:   "adapter-hash",
			Format: rocmTinyLoRAFormat,
			Rank:   1,
			Alpha:  1,
		},
		adapterTokenDelay: 2 * time.Millisecond,
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 3, HiddenSize: 2},
		native:    native,
	}
	var events []inference.ProbeEvent
	model.SetProbeSink(inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
		events = append(events, event)
	}))

	bench, err := model.Benchmark(context.Background(), inference.BenchConfig{Prompts: []string{"hi"}, MaxTokens: 1, MeasuredRuns: 1})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "adapter-hash", bench.Adapter.Hash)
	core.AssertEqual(t, "measured", bench.Labels["lora_overhead"])
	core.AssertEqual(t, "measured", bench.Labels["lora_overhead_status"])
	core.AssertEqual(t, rocmTinyLoRAFormat, bench.Labels["lora_adapter_format"])
	core.AssertEqual(t, "adapter-hash", bench.Labels["lora_adapter_hash"])
	core.AssertEqual(t, "1", bench.Labels["lora_adapter_rank"])
	if bench.Labels["lora_overhead_ms"] == "" || bench.Labels["lora_baseline_duration_ms"] == "" || bench.Labels["lora_adapter_duration_ms"] == "" {
		t.Fatalf("bench labels = %+v, want measured LoRA timing labels", bench.Labels)
	}
	if native.adapter.Path != "tiny-lora.json" || len(native.adapterLoads) == 0 {
		t.Fatalf("native adapter = %+v loads=%+v, want adapter restored after overhead measurement", native.adapter, native.adapterLoads)
	}
	if bench.Labels["probe_count"] != "3" || bench.Labels["probe_count_status"] != "measured" {
		t.Fatalf("bench labels = %+v, want measured token/cache/memory probes excluding LoRA baseline", bench.Labels)
	}
	tokenEvents := 0
	for _, event := range events {
		if event.Kind == inference.ProbeEventToken {
			tokenEvents++
		}
	}
	if tokenEvents != 1 {
		t.Fatalf("events = %+v, want only the active measured token event forwarded", events)
	}
	if got := model.ActiveAdapter(); got.Hash != "adapter-hash" || got.Format != rocmTinyLoRAFormat {
		t.Fatalf("active adapter = %+v, want original adapter identity preserved", got)
	}
	if metrics := model.Metrics(); metrics.GeneratedTokens != 1 {
		t.Fatalf("metrics = %+v, want active benchmark metrics restored after baseline measurement", metrics)
	}
}

func TestNativeContract_BenchmarkLoRARestoreFailureClearsActiveAdapter_Bad(t *testing.T) {
	native := &fakeNativeModel{
		tokens: []inference.Token{{ID: 1, Text: "a"}},
		adapter: inference.AdapterIdentity{
			Path:   "tiny-lora.json",
			Hash:   "adapter-hash",
			Format: rocmTinyLoRAFormat,
		},
		restoreAdapterErr: core.NewError("restore failed"),
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 3, HiddenSize: 2},
		native:    native,
		adapter: inference.AdapterIdentity{
			Path:   "tiny-lora.json",
			Hash:   "adapter-hash",
			Format: rocmTinyLoRAFormat,
		},
	}

	bench, err := model.Benchmark(context.Background(), inference.BenchConfig{Prompts: []string{"hi"}, MaxTokens: 1, MeasuredRuns: 1})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "restore_failed", bench.Labels["lora_overhead_status"])
	core.AssertContains(t, bench.Labels["lora_overhead_error"], "restore failed")
	if !adapterIdentityIsZero(model.ActiveAdapter()) {
		t.Fatalf("active adapter = %+v, want cleared after failed restore", model.ActiveAdapter())
	}
}

func TestNativeContract_BenchmarkLoRAStateCloseFailureSkipsNativeUnload_Bad(t *testing.T) {
	adapter := inference.AdapterIdentity{Path: "tiny-lora.json", Hash: "adapter-hash", Format: rocmTinyLoRAFormat}
	native := &fakeNativeModel{
		tokens:  []inference.Token{{ID: 1, Text: "a"}},
		adapter: adapter,
	}
	state := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, &failingStateRuntime{err: core.NewError("close failed")})
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 3, HiddenSize: 2},
		native:    native,
		adapter:   adapter,
		state:     state,
	}

	bench, err := model.Benchmark(context.Background(), inference.BenchConfig{Prompts: []string{"hi"}, MaxTokens: 1, MeasuredRuns: 1})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "state_close_failed", bench.Labels["lora_overhead_status"])
	core.AssertContains(t, bench.Labels["lora_overhead_error"], "close failed")
	core.AssertEqual(t, 0, native.unloadCalls)
	if model.state != state {
		t.Fatal("benchmark LoRA overhead cleared state after close failure")
	}
	if got := model.ActiveAdapter(); got.Hash != "adapter-hash" || got.Format != rocmTinyLoRAFormat {
		t.Fatalf("active adapter = %+v, want original adapter identity", got)
	}
	if native.adapter.Hash != "adapter-hash" {
		t.Fatalf("native adapter = %+v, want original adapter", native.adapter)
	}
}

func TestNativeContract_EvaluateQualityProbes_Good(t *testing.T) {
	native := &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "a"}, {ID: 2, Text: "b"}}}
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    native,
	}

	eval, err := model.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{Text: "hello world"}}, inference.EvalConfig{
		MaxSamples: 1,
		MaxSeqLen:  4,
		Probes:     []inference.QualityProbe{{Name: "sanity", Prompt: "say hi"}},
	})

	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if len(eval.Probes) != 1 || eval.Probes[0].Name != "sanity" || !eval.Probes[0].Passed || eval.Probes[0].Text != "ab" || eval.Probes[0].Score != 1 {
		t.Fatalf("eval probes = %+v, want generated qualitative probe result", eval.Probes)
	}
	if eval.Labels["eval.samples"] != "1" || eval.Labels["eval.tokens"] != "2" || eval.Labels["loss_status"] != "unsupported" || eval.Labels["perplexity_status"] != "unsupported" {
		t.Fatalf("eval labels = %+v, want token-count eval and unsupported loss/perplexity labels", eval.Labels)
	}
	if eval.Labels["quality_probe_count"] != "1" || eval.Labels["quality_probe_passes"] != "1" || eval.Labels["quality_probe_failures"] != "0" || eval.Labels["quality_probe_status"] != "passed" {
		t.Fatalf("eval labels = %+v, want completed quality probe labels with pass/fail counts", eval.Labels)
	}
}

func TestNativeContract_EvaluateUsesClassifyLogitsForLoss_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native: &fakeNativeModel{
			classLogits: [][]float32{{2, 0}},
		},
	}

	eval, err := model.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{
		Prompt: "hello",
		Labels: map[string]string{"target_token_id": "0"},
	}}, inference.EvalConfig{MaxSamples: 1})

	core.RequireNoError(t, err)
	assertFloat64Near(t, 0.1269, eval.Metrics.Loss, 0.0001)
	assertFloat64Near(t, 1.1353, eval.Metrics.Perplexity, 0.0001)
	if eval.Labels["loss_status"] != "experimental" || eval.Labels["perplexity_status"] != "experimental" || eval.Labels["eval.loss_tokens"] != "1" {
		t.Fatalf("eval labels = %+v, want experimental loss/perplexity labels", eval.Labels)
	}
	core.AssertEqual(t, "reference", eval.Labels["loss_backend"])
	core.AssertEqual(t, hipKernelStatusNotLinked, eval.Labels["loss_kernel"])
	core.AssertEqual(t, hipKernelNameCrossEntropy, eval.Labels["loss_kernel_name"])
}

func TestNativeContract_EvaluateUsesNativeCrossEntropyLossKernel_Good(t *testing.T) {
	native := &fakeNativeModel{
		classLogits:       [][]float32{{0, 3}},
		evalLossKernelOK:  true,
		evalLossKernelOut: hipCrossEntropyLossResult{Loss: 0.25, Perplexity: 1.284025},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}

	eval, err := model.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{
		Prompt: "hello",
		Labels: map[string]string{"target_token_id": "1"},
	}}, inference.EvalConfig{MaxSamples: 1})

	core.RequireNoError(t, err)
	assertFloat64Near(t, 0.25, eval.Metrics.Loss, 0.0001)
	assertFloat64Near(t, 1.284025, eval.Metrics.Perplexity, 0.0001)
	core.AssertEqual(t, 1, native.evalLossKernelCalls)
	core.AssertEqual(t, "hip", eval.Labels["loss_backend"])
	core.AssertEqual(t, hipKernelStatusLinked, eval.Labels["loss_kernel"])
	core.AssertEqual(t, hipKernelNameCrossEntropy, eval.Labels["loss_kernel_name"])
	core.AssertEqual(t, "experimental", eval.Labels["loss_status"])
}

func TestNativeContract_EvaluateLossKernelErrorDoesNotFailEval_Bad(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native: &fakeNativeModel{
			classLogits:       [][]float32{{0, 3}},
			evalLossKernelOK:  true,
			evalLossKernelErr: core.NewError("loss kernel failed"),
		},
	}

	eval, err := model.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{
		Prompt: "hello",
		Labels: map[string]string{"target_token_id": "1"},
	}}, inference.EvalConfig{MaxSamples: 1})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "error", eval.Labels["loss_status"])
	core.AssertEqual(t, "error", eval.Labels["perplexity_status"])
	core.AssertEqual(t, "hip", eval.Labels["loss_backend"])
	core.AssertEqual(t, hipKernelNameCrossEntropy, eval.Labels["loss_kernel_name"])
	core.AssertContains(t, eval.Labels["loss_error"], "loss kernel failed")
}

func TestNativeContract_EvaluateLinkedPrefillWithoutLossTargetsLabelsNotRequested_Good(t *testing.T) {
	native := &fakeNativeModel{
		kernelStatus: hipKernelStatus{Prefill: hipKernelStatusLinked},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}

	eval, err := model.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{Text: "hello world"}}, inference.EvalConfig{MaxSamples: 1})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "not_requested", eval.Labels["loss"])
	core.AssertEqual(t, "not_requested", eval.Labels["loss_status"])
	core.AssertEqual(t, "not_requested", eval.Labels["perplexity"])
	core.AssertEqual(t, "not_requested", eval.Labels["perplexity_status"])
	core.AssertEqual(t, hipKernelStatusLinked, eval.Labels["prefill_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, eval.Labels["loss_kernel"])
	core.AssertEqual(t, hipKernelNameCrossEntropy, eval.Labels["loss_kernel_name"])
	core.AssertEqual(t, 0, len(native.classifyPrompts))
}

func TestNativeContract_EvaluateBatchesClassifyLogitLoss_Good(t *testing.T) {
	native := &fakeNativeModel{
		classLogits: [][]float32{{2, 0}, {2, 0}},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}

	eval, err := model.Evaluate(context.Background(), &sliceInferenceSamples{samples: []inference.DatasetSample{
		{Prompt: "one", Labels: map[string]string{"target_token_id": "0"}},
		{Prompt: "two", Labels: map[string]string{"target_token_id": "0"}},
		{Prompt: "three", Labels: map[string]string{"target_token_id": "0"}},
	}}, inference.EvalConfig{MaxSamples: 3, BatchSize: 2})

	core.RequireNoError(t, err)
	assertFloat64Near(t, 0.1269, eval.Metrics.Loss, 0.0001)
	assertFloat64Near(t, 1.1353, eval.Metrics.Perplexity, 0.0001)
	if len(native.classifyPrompts) != 2 || len(native.classifyPrompts[0]) != 2 || len(native.classifyPrompts[1]) != 1 {
		t.Fatalf("classify prompts = %+v, want batched loss classification", native.classifyPrompts)
	}
	if eval.Labels["eval.batch_size"] != "2" || eval.Labels["eval.loss_batch_size"] != "2" || eval.Labels["eval.loss_batches"] != "2" || eval.Labels["eval.loss_candidates"] != "3" || eval.Labels["eval.loss_tokens"] != "3" {
		t.Fatalf("eval labels = %+v, want batched loss accounting", eval.Labels)
	}
}

func TestNativeContract_EvaluateBadLossTargetWithoutLogitsDoesNotFailEval_Bad(t *testing.T) {
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &fakeNativeModel{},
	}

	eval, err := model.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{
		Prompt: "hello",
		Labels: map[string]string{"target_token_id": "0"},
	}}, inference.EvalConfig{MaxSamples: 1})

	core.RequireNoError(t, err)
	if eval.Metrics.Loss != 0 || eval.Metrics.Perplexity != 0 {
		t.Fatalf("eval metrics = %+v, want no loss/perplexity without logits", eval.Metrics)
	}
	if eval.Labels["loss_status"] != "logits_unavailable" || eval.Labels["perplexity_status"] != "logits_unavailable" {
		t.Fatalf("eval labels = %+v, want logits unavailable status without failing token-count eval", eval.Labels)
	}
}

func TestNativeContract_EvaluateQualityProbes_Bad_RecordsUnavailableGeneration(t *testing.T) {
	model := &rocmModel{modelType: "qwen3", modelInfo: inference.ModelInfo{Architecture: "qwen3"}}

	eval, err := model.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{Text: "hello world"}}, inference.EvalConfig{
		MaxSamples: 1,
		Probes:     []inference.QualityProbe{{Name: "native-decode", Prompt: "say hi"}},
	})

	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if len(eval.Probes) != 1 || eval.Probes[0].Passed || eval.Probes[0].Score != 0 {
		t.Fatalf("eval probes = %+v, want failed qualitative probe without failing token-count eval", eval.Probes)
	}
	if eval.Labels["quality_probe_count"] != "1" || eval.Labels["quality_probe_passes"] != "0" || eval.Labels["quality_probe_failures"] != "1" || eval.Labels["quality_probe_status"] != "generation_unavailable" {
		t.Fatalf("eval labels = %+v, want unavailable generation recorded", eval.Labels)
	}
	if !core.Contains(eval.Labels["quality_probe_error"], "native model is nil") {
		t.Fatalf("eval labels = %+v, want first quality probe error preserved", eval.Labels)
	}
	if resultError(model.Err()) != nil {
		t.Fatalf("Evaluate success with failed quality probe Err() = %v, want nil", resultError(model.Err()))
	}
}

func TestNativeContract_EvaluateQualityProbes_Bad_RecordsEmptyGeneration(t *testing.T) {
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: ""}}},
	}

	eval, err := model.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{Text: "hello world"}}, inference.EvalConfig{
		MaxSamples: 1,
		Probes:     []inference.QualityProbe{{Name: "empty", Prompt: "say hi"}},
	})

	core.RequireNoError(t, err)
	if len(eval.Probes) != 1 || eval.Probes[0].Passed || eval.Probes[0].Score != 0 {
		t.Fatalf("eval probes = %+v, want empty qualitative probe recorded as failed", eval.Probes)
	}
	if eval.Labels["quality_probe_count"] != "1" || eval.Labels["quality_probe_passes"] != "0" || eval.Labels["quality_probe_failures"] != "1" || eval.Labels["quality_probe_status"] != "generation_unavailable" {
		t.Fatalf("eval labels = %+v, want empty generation counted as unavailable", eval.Labels)
	}
	core.AssertContains(t, eval.Labels["quality_probe_error"], "empty response")
}

func TestNativeContract_EvaluateSuccessClearsLastError_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "a"}}},
	}
	model.setLastFailure(core.NewError("stale failure"))

	eval, err := model.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{Text: "hello world"}}, inference.EvalConfig{MaxSamples: 1})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, eval.Metrics.Samples)
	if resultError(model.Err()) != nil {
		t.Fatalf("Evaluate success Err() = %v, want nil", resultError(model.Err()))
	}
}

func TestNativeContract_EvaluateBadRecordsFailure(t *testing.T) {
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "a"}}},
	}

	eval, err := model.Evaluate(context.Background(), &errorInferenceSamples{err: core.NewError("dataset read failed")}, inference.EvalConfig{MaxSamples: 1})

	core.AssertNil(t, eval)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dataset read failed")
	if resultError(model.Err()) == nil {
		t.Fatal("Evaluate failure Err() = nil, want dataset error")
	}
	core.AssertContains(t, resultError(model.Err()).Error(), "dataset read failed")
}

func TestNativeContract_EvaluateBadRejectsEmptyDataset(t *testing.T) {
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "a"}}},
	}

	eval, err := model.Evaluate(context.Background(), &sliceInferenceSamples{}, inference.EvalConfig{MaxSamples: 1})

	core.AssertNil(t, eval)
	core.AssertError(t, err)
	if err != nil {
		core.AssertContains(t, err.Error(), "dataset produced no samples")
	}
	if resultError(model.Err()) == nil {
		t.Fatal("Evaluate empty dataset Err() = nil, want eval failure")
	}
	core.AssertContains(t, resultError(model.Err()).Error(), "dataset produced no samples")
}

func TestNativeContract_BenchmarkEmitsCacheAndMemoryProbeEvents_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native: &fakeNativeModel{
			tokens:  []inference.Token{{ID: 1, Text: "a"}},
			metrics: inference.GenerateMetrics{GeneratedTokens: 1, DecodeDuration: time.Millisecond, PeakMemoryBytes: 64, ActiveMemoryBytes: 32},
		},
	}
	var events []inference.ProbeEvent
	model.SetProbeSink(inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
		events = append(events, event)
	}))
	_, err := model.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2, 3},
		Mode:   rocmKVCacheModeQ8,
		Labels: map[string]string{
			"kv_key_width":   "2",
			"kv_value_width": "2",
		},
	})
	core.RequireNoError(t, err)

	bench, err := model.Benchmark(context.Background(), inference.BenchConfig{Prompts: []string{"hi"}, MaxTokens: 1, MeasuredRuns: 1})

	if err != nil {
		t.Fatalf("Benchmark: %v", err)
	}
	if bench.PeakMemoryBytes < 64 {
		t.Fatalf("bench = %+v, want peak native memory propagated", bench)
	}
	if bench.Labels["memory_active_bytes"] != "32" || bench.Labels["memory_peak_bytes"] != core.Sprintf("%d", bench.PeakMemoryBytes) || floatLabel(t, bench.Labels, "memory_peak_bytes") < 64 {
		t.Fatalf("bench labels = %+v, want active and peak memory byte labels", bench.Labels)
	}
	if bench.Labels["cache.mode"] != rocmKVCacheModeQ8 || bench.Labels["cache.cached_tokens"] != "3" || bench.Labels["cache.kv_cache_block_size"] == "" || bench.Labels["cache.kv_key_width"] != "2" || bench.Labels["cache.kv_value_width"] != "2" {
		t.Fatalf("bench labels = %+v, want cache stats labels with KV shape", bench.Labels)
	}
	if bench.Labels["probe_count"] != "3" || bench.Labels["probe_count_status"] != "measured" {
		t.Fatalf("bench labels = %+v, want token/cache/memory probe count", bench.Labels)
	}
	cacheEvent, ok := nativeContractProbeEvent(events, inference.ProbeEventCachePressure)
	if !ok || cacheEvent.Cache == nil || cacheEvent.Cache.CacheMode != rocmKVCacheModeQ8 {
		t.Fatalf("events = %+v, want cache pressure probe", events)
	}
	if cacheEvent.Cache.CachedTokens != 3 || cacheEvent.Labels["cached_tokens"] != "3" || cacheEvent.Labels["kv_key_width"] != "2" {
		t.Fatalf("cache event = %+v, want cached token count and KV width labels", cacheEvent)
	}
	memoryEvent, ok := nativeContractProbeEvent(events, inference.ProbeEventMemoryPressure)
	if !ok || memoryEvent.Memory == nil || memoryEvent.Memory.ActiveBytes != 32 || memoryEvent.Memory.PeakBytes != bench.PeakMemoryBytes {
		t.Fatalf("events = %+v, want memory pressure probe", events)
	}
}

func TestNativeContract_BenchmarkMirrorsDeviceCacheLabels_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native: &fakeNativeModel{
			tokens:  []inference.Token{{ID: 1, Text: "a"}},
			metrics: inference.GenerateMetrics{GeneratedTokens: 1, DecodeDuration: time.Millisecond, PeakMemoryBytes: 64, ActiveMemoryBytes: 32},
		},
		cache: NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, deviceDriver: driver}),
	}
	var events []inference.ProbeEvent
	model.SetProbeSink(inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
		events = append(events, event)
	}))
	_, err := model.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2, 3},
		Labels: map[string]string{
			"kv_cache_block_size": "2",
			"kv_key_width":        "2",
			"kv_value_width":      "2",
		},
	})
	core.RequireNoError(t, err)

	bench, err := model.Benchmark(context.Background(), inference.BenchConfig{Prompts: []string{"hi"}, MaxTokens: 1, MeasuredRuns: 1})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "mirrored", bench.Labels["cache.kv_device_backing"])
	core.AssertEqual(t, "2", bench.Labels["cache.kv_device_pages"])
	core.AssertEqual(t, "3", bench.Labels["cache.kv_device_tokens"])
	core.AssertNotEmpty(t, bench.Labels["cache.kv_device_bytes"])
	cacheEvent, ok := nativeContractProbeEvent(events, inference.ProbeEventCachePressure)
	if !ok || cacheEvent.Cache == nil {
		t.Fatalf("events = %+v, want cache pressure probe", events)
	}
	core.AssertEqual(t, "mirrored", cacheEvent.Labels["kv_device_backing"])
	core.AssertEqual(t, "2", cacheEvent.Labels["kv_device_pages"])
	core.AssertEqual(t, "3", cacheEvent.Labels["kv_device_tokens"])
	core.AssertNotEmpty(t, cacheEvent.Labels["kv_device_bytes"])
}

func TestNativeContract_ModelPackInspectorRejectsMalformedCodebook_Bad(t *testing.T) {
	dir := nativeContractSafetensorsPack(t, `{"model_type":"Qwen3ForCausalLM"}`)
	writeNativeContractFile(t, core.PathJoin(dir, "codebook_config.json"), `{
		"type":"codebook",
		"format":"vq",
		"codebook_size":16,
		"code_dim":0,
		"tensors":[{"name":"model.layers.0.mlp.down_proj.weight","shape":[2,4]}]
	}`)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)

	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if nativeInspectionHasCapability(inspection, inference.CapabilityCodebookVQ) {
		t.Fatalf("capabilities = %+v, malformed codebook should not report codebook capability", inspection.Capabilities)
	}
	if !nativeContractHasNoteContaining(inspection.Notes, "codebook_config.json could not be parsed") {
		t.Fatalf("notes = %+v, want codebook parse note", inspection.Notes)
	}
}

func TestNativeContract_ModelPackInspectorUgly_CancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(ctx, t.TempDir())

	if err == nil {
		t.Fatalf("InspectModelPack cancelled error = nil, inspection=%+v", inspection)
	}
}

func TestNativeContract_ModelPackInspectorReadsTokenizerSidecars_Good(t *testing.T) {
	dir := nativeContractSafetensorsPack(t, `{
		"model_type":"Qwen3ForCausalLM",
		"hidden_size":1024,
		"num_hidden_layers":8,
		"vocab_size":151936,
		"max_position_embeddings":32768
	}`)
	writeNativeContractFile(t, core.PathJoin(dir, "tokenizer.json"), `{"model":{"type":"BPE"}}`)
	writeNativeContractFile(t, core.PathJoin(dir, "tokenizer_config.json"), `{
		"tokenizer_class":"Qwen2Tokenizer",
		"chat_template":"{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}",
		"bos_token_id":151643,
		"eos_token_id":151645,
		"pad_token_id":151643,
		"model_max_length":32768
	}`)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Tokenizer.Kind != "Qwen2Tokenizer" || inspection.Tokenizer.ChatTemplate == "" || inspection.Tokenizer.EOSID != 151645 {
		t.Fatalf("tokenizer = %+v, want tokenizer_config/chat template metadata", inspection.Tokenizer)
	}
	if inspection.Labels["tokenizer_json_model"] != "BPE" || inspection.Labels["chat_template"] != "present" {
		t.Fatalf("labels = %+v, want tokenizer sidecar labels", inspection.Labels)
	}
	if !nativeInspectionHasCapability(inspection, inference.CapabilityTokenizer) || !nativeInspectionHasCapability(inspection, inference.CapabilityChatTemplate) {
		t.Fatalf("capabilities = %+v, want tokenizer and chat template capabilities", inspection.Capabilities)
	}
}

func TestNativeContract_ModelPackInspectorTokenizerMaxLengthFillsContext_Good(t *testing.T) {
	dir := nativeContractSafetensorsPack(t, `{
		"model_type":"Qwen3ForCausalLM",
		"hidden_size":1024,
		"num_hidden_layers":8,
		"vocab_size":151936
	}`)
	writeNativeContractFile(t, core.PathJoin(dir, "tokenizer_config.json"), `{
		"tokenizer_class":"Qwen2Tokenizer",
		"model_max_length":8192
	}`)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.ContextLength != 8192 {
		t.Fatalf("context length = %d, want tokenizer model_max_length", inspection.Model.ContextLength)
	}
	if inspection.Labels["tokenizer_model_max_length"] != "8192" {
		t.Fatalf("labels = %+v, want tokenizer model max length label", inspection.Labels)
	}
}

func TestNativeContract_ModelPackInspectorTokenizerConfigAcceptsArrayTokenIDs_Good(t *testing.T) {
	dir := nativeContractSafetensorsPack(t, `{"model_type":"Qwen3ForCausalLM"}`)
	writeNativeContractFile(t, core.PathJoin(dir, "tokenizer_config.json"), `{
		"tokenizer_class":"Qwen2Tokenizer",
		"chat_template":"{{ .Prompt }}",
		"eos_token_id":[151645,151643],
		"pad_token_id":151643
	}`)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Tokenizer.EOSID != 151645 || inspection.Tokenizer.PADID != 151643 || inspection.Tokenizer.ChatTemplate == "" {
		t.Fatalf("tokenizer = %+v, want array EOS ID and scalar PAD ID", inspection.Tokenizer)
	}
	if !nativeInspectionHasCapability(inspection, inference.CapabilityChatTemplate) {
		t.Fatalf("capabilities = %+v, want chat template capability", inspection.Capabilities)
	}
}

func TestNativeContract_ModelPackInspectorRerankTaskParamsAllowNonStrings_Good(t *testing.T) {
	dir := nativeContractSafetensorsPack(t, `{
		"architectures":["BertForSequenceClassification"],
		"task_specific_params":{"rerank":{"top_k":10,"normalize":true}}
	}`)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "bert_rerank" || inspection.Labels["rerank_model"] != "true" || inspection.Labels["classifier_model"] != "true" {
		t.Fatalf("inspection = %+v labels=%+v, want BERT classifier/rerank metadata", inspection, inspection.Labels)
	}
	if !nativeInspectionHasCapability(inspection, inference.CapabilityRerank) {
		t.Fatalf("capabilities = %+v, want rerank metadata capability", inspection.Capabilities)
	}
	classifyCapability, ok := nativeInspectionCapability(inspection, inference.CapabilityClassify)
	if !ok || classifyCapability.Status != inference.CapabilityStatusPlanned || classifyCapability.Labels["classify_path"] != "bert_sequence_classifier" {
		t.Fatalf("classify capability = %+v ok=%v, want planned BERT classifier metadata", classifyCapability, ok)
	}
}

func TestNativeContract_ModelPackInspectorTextClassificationTaskParamsPreferClassifier_Good(t *testing.T) {
	dir := nativeContractSafetensorsPack(t, `{
		"model_type":"BertModel",
		"task_specific_params":{"text-classification":{"return_all_scores":true}}
	}`)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "bert" || inspection.Labels["classifier_model"] != "true" {
		t.Fatalf("inspection = %+v labels=%+v, want BERT classifier metadata", inspection, inspection.Labels)
	}
	if _, ok := inspection.Labels["embedding_model"]; ok {
		t.Fatalf("labels = %+v, text-classification metadata should not be labelled as embedding_model", inspection.Labels)
	}
	classifyCapability, ok := nativeInspectionCapability(inspection, inference.CapabilityClassify)
	if !ok || classifyCapability.Labels["classify_path"] != "bert_sequence_classifier" {
		t.Fatalf("classify capability = %+v ok=%v, want BERT classifier path", classifyCapability, ok)
	}
	if nativeInspectionHasCapability(inspection, inference.CapabilityEmbeddings) {
		t.Fatalf("capabilities = %+v, text-classification metadata should not report embedding capability", inspection.Capabilities)
	}
}

func TestNativeContract_ModelPackInspectorSequenceClassificationWithoutRerankIsClassifierOnly_Good(t *testing.T) {
	dir := nativeContractSafetensorsPack(t, `{
		"architectures":["BertForSequenceClassification"],
		"task_specific_params":{"text-classification":{"return_all_scores":true}}
	}`)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "bert_rerank" || inspection.Labels["classifier_model"] != "true" {
		t.Fatalf("inspection = %+v labels=%+v, want BERT classifier metadata", inspection, inspection.Labels)
	}
	if _, ok := inspection.Labels["rerank_model"]; ok {
		t.Fatalf("labels = %+v, sequence-classification metadata without rerank task should not be labelled as rerank_model", inspection.Labels)
	}
	if nativeInspectionHasCapability(inspection, inference.CapabilityRerank) {
		t.Fatalf("capabilities = %+v, sequence-classification metadata without rerank task should not report rerank capability", inspection.Capabilities)
	}
	if !nativeInspectionHasCapability(inspection, inference.CapabilityClassify) {
		t.Fatalf("capabilities = %+v, want classifier capability", inspection.Capabilities)
	}
}

func TestNativeContract_ModelPackInspectorTaskParamsAllowScalarValues_Good(t *testing.T) {
	cases := []struct {
		name           string
		config         string
		wantCapability inference.CapabilityID
		wantLabel      string
	}{
		{
			name:           "rerank_bool",
			config:         `{"model_type":"BertModel","task_specific_params":{"rerank":true}}`,
			wantCapability: inference.CapabilityRerank,
			wantLabel:      "rerank_model",
		},
		{
			name:           "classification_string",
			config:         `{"model_type":"BertModel","task_specific_params":{"pipeline_tag":"text-classification"}}`,
			wantCapability: inference.CapabilityClassify,
			wantLabel:      "classifier_model",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, tc.config))
			if err != nil {
				t.Fatalf("InspectModelPack: %v", err)
			}
			if nativeContractHasNoteContaining(inspection.Notes, "config.json could not be parsed") {
				t.Fatalf("notes = %+v, scalar task-specific params should not reject config.json", inspection.Notes)
			}
			if inspection.Model.Architecture != "bert" || inspection.Labels[tc.wantLabel] != "true" {
				t.Fatalf("inspection = %+v labels=%+v, want BERT %s metadata", inspection, inspection.Labels, tc.wantLabel)
			}
			if !nativeInspectionHasCapability(inspection, tc.wantCapability) {
				t.Fatalf("capabilities = %+v, want %s", inspection.Capabilities, tc.wantCapability)
			}
		})
	}
}

func TestNativeContract_ModelPackInspectorArchitectureFixtures_Good(t *testing.T) {
	cases := []struct {
		name         string
		config       string
		architecture string
		quantType    string
		capability   inference.CapabilityID
		dense        bool
		mtpDrafter   bool
	}{
		{name: "Qwen3", architecture: "qwen3", dense: true, config: `{"model_type":"Qwen3ForCausalLM","max_position_embeddings":32768}`},
		{name: "Qwen3MoE", architecture: "qwen3_moe", capability: inference.CapabilityMoERouting, config: `{"model_type":"Qwen3MoeForCausalLM","num_local_experts":128,"num_experts_per_tok":8}`},
		{name: "Qwen3Next", architecture: "qwen3_next", config: `{"architectures":["Qwen3NextForCausalLM"],"max_position_embeddings":262144}`},
		{name: "Qwen3.6", architecture: "qwen3_6", dense: true, config: `{"architectures":["Qwen3_5ForConditionalGeneration"],"max_position_embeddings":262144}`},
		{name: "Qwen3.6MoE", architecture: "qwen3_6_moe", capability: inference.CapabilityMoERouting, config: `{"architectures":["Qwen3_5MoeForConditionalGeneration"],"num_local_experts":128,"num_experts_per_tok":8}`},
		{name: "Gemma", architecture: "gemma", config: `{"model_type":"GemmaForCausalLM","max_position_embeddings":8192}`},
		{name: "Gemma3", architecture: "gemma3", dense: true, config: `{"model_type":"Gemma3ForCausalLM","max_position_embeddings":131072}`},
		{name: "Mistral", architecture: "mistral", dense: true, config: `{"model_type":"MistralForCausalLM","sliding_window":4096}`},
		{name: "Mixtral", architecture: "mixtral", config: `{"model_type":"MixtralForCausalLM","num_local_experts":8,"num_experts_per_tok":2}`},
		{name: "Phi", architecture: "phi", dense: true, config: `{"model_type":"Phi3ForCausalLM","max_position_embeddings":4096}`},
		{name: "DeepSeek", architecture: "deepseek", config: `{"model_type":"DeepseekV3ForCausalLM","num_hidden_layers":61}`},
		{name: "DeepSeekR1", architecture: "deepseek_r1", config: `{"architectures":["DeepSeekR1ForCausalLM"],"num_hidden_layers":61}`},
		{name: "GPTOSS", architecture: "gpt-oss", quantType: "mxfp4", config: `{"architectures":["GptOssForCausalLM"],"max_position_embeddings":131072,"quantization_config":{"quant_method":"mxfp4"}}`},
		{name: "Kimi", architecture: "kimi", quantType: "nvfp4", config: `{"architectures":["KimiK2ForCausalLM"],"max_position_embeddings":131072,"quantization_config":{"format":"nvfp4"}}`},
		{name: "Gemma4Text", architecture: "gemma4_text", config: `{"model_type":"gemma4_text","max_position_embeddings":131072}`},
		{name: "Gemma4CausalLM", architecture: "gemma4_text", config: `{"architectures":["Gemma4ForCausalLM"],"max_position_embeddings":131072}`},
		{name: "Gemma4Assistant", architecture: "gemma4_assistant", mtpDrafter: true, config: `{"architectures":["Gemma4AssistantForCausalLM"],"max_position_embeddings":131072}`},
		{name: "MiniMax", architecture: "minimax", config: `{"model_type":"MiniMaxForCausalLM","max_position_embeddings":32768}`},
		{name: "Llama", architecture: "llama", config: `{"model_type":"LlamaForCausalLM","max_position_embeddings":8192}`},
		{name: "GLM4", architecture: "glm4", dense: true, config: `{"model_type":"ChatGLM4ForCausalLM","max_position_embeddings":32768}`},
		{name: "Hermes", architecture: "hermes", dense: true, config: `{"architectures":["NousHermesForCausalLM"],"max_position_embeddings":32768}`},
		{name: "Granite", architecture: "granite", dense: true, config: `{"model_type":"GraniteForCausalLM","max_position_embeddings":8192}`},
		{name: "BERTEmbeddings", architecture: "bert", capability: inference.CapabilityEmbeddings, config: `{"model_type":"BertModel","max_position_embeddings":512}`},
		{name: "BERTReranker", architecture: "bert_rerank", capability: inference.CapabilityRerank, config: `{"architectures":["BertForSequenceClassification"],"task_specific_params":{"rerank":{"task":"rerank"}}}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, tc.config))
			if err != nil {
				t.Fatalf("InspectModelPack: %v", err)
			}
			if inspection.Model.Architecture != tc.architecture || !inspection.Supported {
				t.Fatalf("inspection = %+v, want supported architecture %q", inspection, tc.architecture)
			}
			if tc.quantType != "" && inspection.Model.QuantType != tc.quantType {
				t.Fatalf("inspection quantization = %q, want %q", inspection.Model.QuantType, tc.quantType)
			}
			if tc.capability != "" && !nativeInspectionHasCapability(inspection, tc.capability) {
				t.Fatalf("capabilities = %+v, want %s", inspection.Capabilities, tc.capability)
			}
			if tc.dense {
				if inspection.Labels["dense_route_candidate"] != "true" ||
					inspection.Labels["dense_route_status"] != "experimental" ||
					inspection.Labels["dense_route_family"] != "loader_neutral" ||
					inspection.Labels["dense_route_backend"] != "hip_small_decode" ||
					inspection.Labels["dense_route_reference"] != "gemma4_mlx_affine_matvec" {
					t.Fatalf("labels = %+v, want dense quick-win route candidate labels", inspection.Labels)
				}
			} else if inspection.Labels["dense_route_candidate"] == "true" {
				t.Fatalf("labels = %+v, non-dense fixture should not be labelled as dense route candidate", inspection.Labels)
			}
			if tc.mtpDrafter {
				if inspection.Labels["attached_drafter"] != "experimental_retained_plan" ||
					inspection.Labels["attached_drafter_native_attachment"] != hipKernelStatusNotLinked ||
					inspection.Labels["attached_drafter_retained_state_entrypoint"] != hipKernelStatusLinked ||
					inspection.Labels["attached_drafter_retained_state_required"] != "true" ||
					inspection.Labels["attached_drafter_state_source"] != "rocm_state_session_runtime_kv" ||
					inspection.Labels["attached_drafter_prompt_replay_fallback"] != "forbidden" ||
					inspection.Labels["attached_drafter_assistant_architecture"] != officialGemma4E2BAssistantArchitecture ||
					inspection.Labels["attached_drafter_assistant_ordered_embeddings"] != "true" ||
					inspection.Labels["attached_drafter_assistant_centroids"] != productionMTPAssistantOrderedEmbeddingCentroidsLabel ||
					inspection.Labels["attached_drafter_assistant_centroid_intermediate_top_k"] != productionMTPAssistantCentroidIntermediateTopKLabel ||
					inspection.Labels["attached_drafter_assistant_four_layer_drafter"] != "true" ||
					inspection.Labels["attached_drafter_assistant_token_ordering_dtype"] != "int64" ||
					inspection.Labels["attached_drafter_assistant_token_ordering_shape"] != productionMTPAssistantTokenOrderingShapeLabel ||
					inspection.Labels["attached_drafter_official_pair_verified"] != "false" ||
					inspection.Labels["attached_drafter_speculative_draft_tokens"] != productionMTPDefaultDraftTokensLabel ||
					inspection.Labels["mtp_role"] != "drafter" ||
					inspection.Labels["mtp_target_family"] != "gemma4" ||
					!nativeContractHasNoteContaining(inspection.Notes, "attached MTP drafter") {
					t.Fatalf("inspection = %+v labels=%+v notes=%+v, want Gemma4 assistant MTP drafter metadata", inspection, inspection.Labels, inspection.Notes)
				}
			} else if inspection.Labels["mtp_role"] != "" {
				t.Fatalf("labels = %+v, non-assistant fixture should not be labelled as MTP drafter", inspection.Labels)
			}
		})
	}
}

func TestNativeContract_ModelPackInspectorAutoRoundQuantization_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, `{
		"architectures":["Qwen3ForCausalLM"],
		"max_position_embeddings":32768,
		"quantization_config":{
			"quant_method":"auto-round-light",
			"format":"native",
			"weight_format":"mxfp4",
			"scheme":"W4A16",
			"bits":4,
			"group_size":128,
			"iters":200,
			"nsamples":512,
			"seqlen":2048,
			"sym":true,
			"asym":false
		}
	}`))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.QuantType != "auto_round_light" || inspection.Model.QuantBits != 4 || inspection.Model.QuantGroup != 128 {
		t.Fatalf("model quantization = %+v, want AutoRound q4 group-128 identity", inspection.Model)
	}
	if inspection.Labels["autoround_quantization"] != "true" ||
		inspection.Labels["autoround_algorithm"] != "auto_round_light" ||
		inspection.Labels["autoround_format"] != "native" ||
		inspection.Labels["autoround_weight_format"] != "mxfp4" ||
		inspection.Labels["autoround_scheme"] != "W4A16" ||
		inspection.Labels["autoround_bits"] != "4" ||
		inspection.Labels["autoround_group_size"] != "128" ||
		inspection.Labels["autoround_iters"] != "200" ||
		inspection.Labels["autoround_nsamples"] != "512" ||
		inspection.Labels["autoround_seqlen"] != "2048" ||
		inspection.Labels["autoround_sym"] != "true" ||
		inspection.Labels["autoround_asym"] != "false" ||
		inspection.Labels["autoround_profile"] != "w4a16-mxfp4-g128" ||
		inspection.Labels["autoround_profile_role"] != "rocm-fp4-planning" ||
		inspection.Labels["autoround_profile_matched"] != "true" ||
		inspection.Labels["autoround_profile_requires_bench"] != "true" ||
		inspection.Labels["autoround_profile_requires_calibration"] != "true" ||
		inspection.Labels["autoround_calibration_profile"] != "w4a16-mxfp4-g128" ||
		inspection.Labels["autoround_calibration_format"] != "native" ||
		inspection.Labels["autoround_calibration_weight_scheme"] != "W4A16" ||
		inspection.Labels["autoround_calibration_float_format"] != "mxfp4" ||
		inspection.Labels["autoround_calibration_bits"] != "4" ||
		inspection.Labels["autoround_calibration_group_size"] != "128" ||
		inspection.Labels["autoround_calibration_nsamples"] != "512" ||
		inspection.Labels["autoround_calibration_seqlen"] != "2048" ||
		inspection.Labels["autoround_calibration_iters"] != "200" ||
		inspection.Labels["autoround_calibration_runtime"] != "planned_hip" ||
		inspection.Labels["autoround_calibration_hip_kernel"] != hipKernelStatusNotLinked ||
		inspection.Labels["autoround_calibration_requires_bench"] != "true" ||
		inspection.Labels["autoround_calibration_required"] != "true" ||
		inspection.Labels["autoround_runtime"] != "planned_hip" ||
		inspection.Labels["autoround_hip_kernel"] != hipKernelStatusNotLinked {
		t.Fatalf("labels = %+v, want AutoRound metadata labels", inspection.Labels)
	}

	mxfp8Inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, `{
		"architectures":["Qwen3ForCausalLM"],
		"max_position_embeddings":32768,
		"quantization_config":{
			"quant_method":"auto-round",
			"format":"native",
			"weight_format":"mxfp8",
			"scheme":"W8A16",
			"bits":8,
			"group_size":64,
			"iters":220,
			"nsamples":640,
			"seqlen":3072
		}
	}`))
	if err != nil {
		t.Fatalf("InspectModelPack MXFP8: %v", err)
	}
	if mxfp8Inspection.Model.QuantType != "auto_round" || mxfp8Inspection.Model.QuantBits != 8 || mxfp8Inspection.Model.QuantGroup != 64 {
		t.Fatalf("MXFP8 model quantization = %+v, want AutoRound q8 group-64 identity", mxfp8Inspection.Model)
	}
	if mxfp8Inspection.Labels["autoround_weight_format"] != "mxfp8" ||
		mxfp8Inspection.Labels["autoround_scheme"] != "W8A16" ||
		mxfp8Inspection.Labels["autoround_profile"] != "w8a16-mxfp8-g64" ||
		mxfp8Inspection.Labels["autoround_profile_role"] != "rocm-fp8-planning" ||
		mxfp8Inspection.Labels["autoround_profile_matched"] != "true" ||
		mxfp8Inspection.Labels["autoround_calibration_profile"] != "w8a16-mxfp8-g64" ||
		mxfp8Inspection.Labels["autoround_calibration_weight_scheme"] != "W8A16" ||
		mxfp8Inspection.Labels["autoround_calibration_float_format"] != "mxfp8" ||
		mxfp8Inspection.Labels["autoround_calibration_bits"] != "8" ||
		mxfp8Inspection.Labels["autoround_calibration_group_size"] != "64" ||
		mxfp8Inspection.Labels["autoround_calibration_nsamples"] != "640" ||
		mxfp8Inspection.Labels["autoround_calibration_seqlen"] != "3072" ||
		mxfp8Inspection.Labels["autoround_calibration_iters"] != "220" ||
		mxfp8Inspection.Labels["autoround_calibration_runtime"] != "planned_hip" ||
		mxfp8Inspection.Labels["autoround_calibration_hip_kernel"] != hipKernelStatusNotLinked {
		t.Fatalf("MXFP8 labels = %+v, want AutoRound MXFP8 calibration labels", mxfp8Inspection.Labels)
	}

	int2Inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, `{
		"architectures":["Qwen3ForCausalLM"],
		"max_position_embeddings":32768,
		"quantization_config":{
			"quant_method":"auto-round",
			"format":"native",
			"weight_format":"int2",
			"scheme":"W2A16",
			"bits":2,
			"group_size":128,
			"iters":240,
			"nsamples":768,
			"seqlen":4096
		}
	}`))
	if err != nil {
		t.Fatalf("InspectModelPack INT2: %v", err)
	}
	if int2Inspection.Model.QuantType != "auto_round" || int2Inspection.Model.QuantBits != 2 || int2Inspection.Model.QuantGroup != 128 {
		t.Fatalf("INT2 model quantization = %+v, want AutoRound q2 group-128 identity", int2Inspection.Model)
	}
	if int2Inspection.Labels["autoround_weight_format"] != "int2" ||
		int2Inspection.Labels["autoround_scheme"] != "W2A16" ||
		int2Inspection.Labels["autoround_profile"] != "w2a16-int2-g128" ||
		int2Inspection.Labels["autoround_profile_role"] != "rocm-int2-planning" ||
		int2Inspection.Labels["autoround_profile_matched"] != "true" ||
		int2Inspection.Labels["autoround_calibration_profile"] != "w2a16-int2-g128" ||
		int2Inspection.Labels["autoround_calibration_weight_scheme"] != "W2A16" ||
		int2Inspection.Labels["autoround_calibration_float_format"] != "int2" ||
		int2Inspection.Labels["autoround_calibration_bits"] != "2" ||
		int2Inspection.Labels["autoround_calibration_group_size"] != "128" ||
		int2Inspection.Labels["autoround_calibration_nsamples"] != "768" ||
		int2Inspection.Labels["autoround_calibration_seqlen"] != "4096" ||
		int2Inspection.Labels["autoround_calibration_iters"] != "240" ||
		int2Inspection.Labels["autoround_calibration_runtime"] != "planned_hip" ||
		int2Inspection.Labels["autoround_calibration_hip_kernel"] != hipKernelStatusNotLinked {
		t.Fatalf("INT2 labels = %+v, want AutoRound W2A16 INT2 calibration labels", int2Inspection.Labels)
	}
}

func TestNativeContract_ModelPackInspectorGemma4NestedTextConfig_Good(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures":["Gemma4ForConditionalGeneration"],
		"model_type":"gemma4",
		"tie_word_embeddings":true,
		"quantization_config":{"bits":6,"group_size":64,"mode":"affine"},
		"text_config":{
			"model_type":"gemma4_text",
			"hidden_size":1536,
			"num_hidden_layers":35,
			"num_attention_heads":8,
			"num_key_value_heads":1,
			"num_global_key_value_heads":1,
			"head_dim":256,
			"global_head_dim":512,
			"hidden_size_per_layer_input":256,
			"vocab_size_per_layer_input":262144,
			"max_position_embeddings":131072,
			"sliding_window":512,
			"layer_types":["full_attention","sliding_attention"],
			"use_double_wide_mlp":true,
			"rms_norm_eps":0.000001,
			"final_logit_softcapping":30.0,
			"vocab_size":262144
		}
	}`)
	writeNativeContractFile(t, core.PathJoin(dir, "tokenizer_config.json"), `{
		"tokenizer_class":"GemmaTokenizer",
		"model_max_length":1000000000000000019884624838656
	}`)
	writeNativeContractSafetensors(t, core.PathJoin(dir, "model.safetensors"))

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{
		available: true,
		device:    nativeDeviceInfo{Name: "AMD Radeon RX 7800 XT", MemoryBytes: 16 * memoryGiB, FreeBytes: 12 * memoryGiB, Driver: "hip-test"},
	}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !inspection.Supported || inspection.Format != "safetensors" || inspection.Model.Architecture != "gemma4" {
		t.Fatalf("inspection = %+v labels=%+v, want supported Gemma4 safetensors pack", inspection, inspection.Labels)
	}
	if inspection.Model.ContextLength != 131072 ||
		inspection.Model.NumLayers != 35 ||
		inspection.Model.HiddenSize != 1536 ||
		inspection.Model.VocabSize != 262144 ||
		inspection.Model.QuantBits != 6 ||
		inspection.Model.QuantGroup != 64 {
		t.Fatalf("model = %+v, want Gemma4 text_config dimensions and quantization", inspection.Model)
	}
	if inspection.Labels["tokenizer_config"] != "present" || inspection.Tokenizer.Kind != "GemmaTokenizer" {
		t.Fatalf("tokenizer = %+v labels=%+v, want Gemma tokenizer config", inspection.Tokenizer, inspection.Labels)
	}
	if inspection.Labels["tied_word_embeddings"] != "true" {
		t.Fatalf("labels = %+v, want tied Gemma4 embedding metadata", inspection.Labels)
	}
	if inspection.Labels["sliding_window"] != "512" ||
		inspection.Labels["attention_full_layers"] != "1" ||
		inspection.Labels["attention_sliding_layers"] != "1" ||
		inspection.Labels["memory_plan_sliding_window"] != "512" {
		t.Fatalf("labels = %+v, want Gemma4 sliding-attention metadata", inspection.Labels)
	}
	if inspection.Labels["attention_heads"] != "8" ||
		inspection.Labels["attention_kv_heads"] != "1" ||
		inspection.Labels["attention_global_kv_heads"] != "1" ||
		inspection.Labels["attention_head_dim"] != "256" ||
		inspection.Labels["attention_global_head_dim"] != "512" ||
		inspection.Labels["gemma4_hidden_size_per_layer_input"] != "256" ||
		inspection.Labels["gemma4_vocab_size_per_layer_input"] != "262144" ||
		inspection.Labels["attention_query_width"] != "2048" ||
		inspection.Labels["attention_kv_width"] != "256" ||
		inspection.Labels["attention_global_kv_width"] != "512" ||
		inspection.Labels["attention_gqa"] != "true" ||
		inspection.Labels["gemma4_use_double_wide_mlp"] != "true" ||
		inspection.Labels["rms_norm_eps"] != "1e-06" ||
		inspection.Labels["final_logit_softcapping"] != "30" ||
		inspection.Labels["memory_plan_attention_query_width"] != "2048" {
		t.Fatalf("labels = %+v, want Gemma4 GQA/head-dimension metadata", inspection.Labels)
	}
	if _, ok := inspection.Labels["tokenizer_model_max_length"]; ok {
		t.Fatalf("labels = %+v, sentinel tokenizer model_max_length should not override Gemma4 text context", inspection.Labels)
	}
	if inspection.Labels["memory_fit"] != "true" || inspection.Labels["memory_plan_machine_class"] != "rocm-16gb" {
		t.Fatalf("labels = %+v notes=%+v, want 16GB Gemma4 memory-fit plan", inspection.Labels, inspection.Notes)
	}
}

func TestNativeContract_ModelPackInspectorGemma4BF16DType_Good(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures":["Gemma4ForConditionalGeneration"],
		"dtype":"bfloat16",
		"model_type":"gemma4",
		"tie_word_embeddings":true,
		"text_config":{
			"model_type":"gemma4_text",
			"hidden_size":16,
			"num_hidden_layers":1,
			"max_position_embeddings":8192,
			"vocab_size":8
		}
	}`)
	writeNativeContractSafetensorsHeaderWithPayload(t, core.PathJoin(dir, "model-00001-of-00002.safetensors"), `{"language_model.model.embed_tokens.weight":{"dtype":"BF16","shape":[8,2],"data_offsets":[0,32]}}`, 32)
	writeNativeContractSafetensorsHeaderWithPayload(t, core.PathJoin(dir, "model-00002-of-00002.safetensors"), `{"language_model.model.layers.0.input_layernorm.weight":{"dtype":"BF16","shape":[16],"data_offsets":[0,32]}}`, 32)
	writeNativeContractFile(t, core.PathJoin(dir, "model.safetensors.index.json"), `{
		"metadata":{"total_size":64,"total_parameters":16},
		"weight_map":{
			"language_model.model.embed_tokens.weight":"model-00001-of-00002.safetensors",
			"language_model.model.layers.0.input_layernorm.weight":"model-00002-of-00002.safetensors"
		}
	}`)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{
		available: true,
		device:    nativeDeviceInfo{Name: "AMD Radeon RX 7800 XT", MemoryBytes: 16 * memoryGiB, FreeBytes: 12 * memoryGiB, Driver: "hip-test"},
	}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !inspection.Supported || inspection.Model.Architecture != "gemma4" || inspection.Model.QuantType != "bf16" || inspection.Model.QuantBits != 16 {
		t.Fatalf("inspection = %+v labels=%+v, want supported BF16 Gemma4 safetensors pack", inspection, inspection.Labels)
	}
	if inspection.Labels["weight_files"] != "2" || inspection.Labels["safetensors_dtypes"] != "BF16" || inspection.Labels["tied_word_embeddings"] != "true" {
		t.Fatalf("labels = %+v, want two-shard BF16 tied Gemma4 metadata", inspection.Labels)
	}
	if inspection.Labels["safetensors_index"] != "present" ||
		inspection.Labels["safetensors_index_tensors"] != "2" ||
		inspection.Labels["safetensors_index_shards"] != "2" ||
		inspection.Labels["safetensors_index_total_size"] != "64" ||
		inspection.Labels["safetensors_index_total_parameters"] != "16" ||
		inspection.Labels["weight_bytes"] != "64" ||
		inspection.Labels["memory_plan_weight_bytes"] != "64" ||
		inspection.Labels["sharded_safetensors"] != "true" {
		t.Fatalf("labels = %+v, want safetensors index metadata", inspection.Labels)
	}
}

func TestNativeContract_ModelPackInspectorSafetensorsIndexMissingShard_Bad(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"gemma4","text_config":{"hidden_size":16,"num_hidden_layers":1,"vocab_size":8}}`)
	writeNativeContractSafetensorsHeaderWithPayload(t, core.PathJoin(dir, "model-00001-of-00002.safetensors"), `{"language_model.model.embed_tokens.weight":{"dtype":"BF16","shape":[8,2],"data_offsets":[0,32]}}`, 32)
	writeNativeContractFile(t, core.PathJoin(dir, "model.safetensors.index.json"), `{
		"metadata":{"total_size":64},
		"weight_map":{
			"language_model.model.embed_tokens.weight":"model-00001-of-00002.safetensors",
			"language_model.model.layers.0.input_layernorm.weight":"model-00002-of-00002.safetensors"
		}
	}`)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, stale safetensors index should not be supported", inspection, inspection.Labels)
	}
	if !nativeContractHasNoteContaining(inspection.Notes, "references missing shard") {
		t.Fatalf("notes = %+v, want missing shard note", inspection.Notes)
	}
	if _, ok := inspection.Labels["safetensors_index"]; ok {
		t.Fatalf("labels = %+v, invalid safetensors index metadata should be cleared", inspection.Labels)
	}
}

func TestNativeContract_ModelPackInspectorAggregatesSafetensorsShardSummaries_Good(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"Qwen3ForCausalLM"}`)
	header1 := `{"model.layers.0.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[0,16]}}`
	header2 := `{"model.layers.1.weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]}}`
	writeNativeContractSafetensorsHeader(t, core.PathJoin(dir, "model-00001-of-00002.safetensors"), header1)
	writeNativeContractSafetensorsHeader(t, core.PathJoin(dir, "model-00002-of-00002.safetensors"), header2)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !inspection.Supported || inspection.Labels["weight_metadata_valid"] != "true" {
		t.Fatalf("inspection = %+v labels=%+v, sharded safetensors should be supported", inspection, inspection.Labels)
	}
	if inspection.Labels["safetensors_tensors"] != "2" ||
		inspection.Labels["safetensors_payload_bytes"] != "32" ||
		inspection.Labels["safetensors_header_bytes"] != core.Sprintf("%d", len(header1)+len(header2)) ||
		inspection.Labels["safetensors_dtypes"] != "BF16,F16" {
		t.Fatalf("labels = %+v, want aggregate safetensors shard summaries", inspection.Labels)
	}
}

func TestNativeContract_ModelPackInspectorMalformedSafetensors_Bad(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"Qwen3ForCausalLM"}`)
	path := core.PathJoin(dir, "model.safetensors")
	buf := core.NewBuffer()
	core.RequireNoError(t, binary.Write(buf, binary.LittleEndian, uint64(maxSafetensorsHeaderBytes+1)))
	result := core.WriteFile(path, buf.Bytes(), 0o644)
	core.RequireTrue(t, result.OK)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if len(inspection.Notes) == 0 || !core.Contains(core.Join("\n", inspection.Notes...), "outside supported bounds") {
		t.Fatalf("notes = %+v, want bounded safetensors header error", inspection.Notes)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, malformed safetensors should not be supported", inspection, inspection.Labels)
	}
	if _, ok := inspection.Labels["memory_fit"]; ok {
		t.Fatalf("labels = %+v, unsupported model pack should not report memory fit", inspection.Labels)
	}
	if _, ok := inspection.Labels["safetensors_tensors"]; ok {
		t.Fatalf("labels = %+v, malformed safetensors should not report tensor count", inspection.Labels)
	}
}

func TestNativeContract_ModelPackInspectorMalformedSafetensorsOffsets_Bad(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"Qwen3ForCausalLM"}`)
	path := core.PathJoin(dir, "model.safetensors")
	writeNativeContractSafetensorsHeader(t, path, `{"model.layers.0.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[16,0]}}`)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !nativeContractHasNoteContaining(inspection.Notes, "reversed data_offsets") {
		t.Fatalf("notes = %+v, want malformed data_offsets note", inspection.Notes)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, malformed safetensors should not be supported", inspection, inspection.Labels)
	}
	if _, ok := inspection.Labels["safetensors_tensors"]; ok {
		t.Fatalf("labels = %+v, malformed safetensors should not report tensor count", inspection.Labels)
	}
}

func TestNativeContract_ModelPackInspectorTruncatedSafetensorsPayload_Bad(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"Qwen3ForCausalLM"}`)
	path := core.PathJoin(dir, "model.safetensors")
	writeNativeContractSafetensorsHeader(t, path, `{"model.layers.0.weight":{"dtype":"F16","shape":[2,16],"data_offsets":[0,64]}}`)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !nativeContractHasNoteContaining(inspection.Notes, "exceeds payload bytes") {
		t.Fatalf("notes = %+v, want truncated safetensors payload note", inspection.Notes)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, truncated safetensors should not be supported", inspection, inspection.Labels)
	}
	if _, ok := inspection.Labels["safetensors_tensors"]; ok {
		t.Fatalf("labels = %+v, truncated safetensors should not report tensor count", inspection.Labels)
	}
}

func TestNativeContract_ModelPackInspectorSafetensorsMissingRequiredFields_Bad(t *testing.T) {
	cases := []struct {
		name   string
		header string
		note   string
	}{
		{
			name:   "missing_dtype",
			header: `{"model.layers.0.weight":{"shape":[2,4],"data_offsets":[0,16]}}`,
			note:   "missing dtype",
		},
		{
			name:   "missing_shape",
			header: `{"model.layers.0.weight":{"dtype":"F16","data_offsets":[0,16]}}`,
			note:   "missing shape",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"Qwen3ForCausalLM"}`)
			writeNativeContractSafetensorsHeader(t, core.PathJoin(dir, "model.safetensors"), tc.header)

			inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
			if err != nil {
				t.Fatalf("InspectModelPack: %v", err)
			}
			if !nativeContractHasNoteContaining(inspection.Notes, tc.note) {
				t.Fatalf("notes = %+v, want %q", inspection.Notes, tc.note)
			}
			if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
				t.Fatalf("inspection = %+v labels=%+v, malformed safetensors should not be supported", inspection, inspection.Labels)
			}
			if _, ok := inspection.Labels["safetensors_tensors"]; ok {
				t.Fatalf("labels = %+v, malformed safetensors should not report tensor count", inspection.Labels)
			}
		})
	}
}

func TestNativeContract_ModelPackInspectorSafetensorsRequiresTensorEntries_Bad(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"Qwen3ForCausalLM"}`)
	writeNativeContractSafetensorsHeader(t, core.PathJoin(dir, "model.safetensors"), `{"__metadata__":{"format":"pt"}}`)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !nativeContractHasNoteContaining(inspection.Notes, "contains no tensor entries") {
		t.Fatalf("notes = %+v, want no tensor entries note", inspection.Notes)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, metadata-only safetensors should not be supported", inspection, inspection.Labels)
	}
	if _, ok := inspection.Labels["safetensors_tensors"]; ok {
		t.Fatalf("labels = %+v, metadata-only safetensors should not report tensor count", inspection.Labels)
	}
}

func TestNativeContract_ModelPackInspectorSafetensorsValidatesNonMetadataDoubleUnderscoreKeys_Bad(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"Qwen3ForCausalLM"}`)
	writeNativeContractSafetensorsHeader(t, core.PathJoin(dir, "model.safetensors"), `{
		"model.layers.0.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[0,16]},
		"__not_metadata__":{"dtype":"FLOAT9000","shape":[2,4],"data_offsets":[0,16]},
		"__metadata__":{"format":"pt"}
	}`)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !nativeContractHasNoteContaining(inspection.Notes, "unsupported dtype") {
		t.Fatalf("notes = %+v, want non-metadata double-underscore tensor validation note", inspection.Notes)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, malformed safetensors should not be supported", inspection, inspection.Labels)
	}
	if _, ok := inspection.Labels["safetensors_tensors"]; ok {
		t.Fatalf("labels = %+v, malformed safetensors should not report tensor count", inspection.Labels)
	}
}

func TestNativeContract_ModelPackInspectorSafetensorsRejectsDuplicateTensorKeys_Bad(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"Qwen3ForCausalLM"}`)
	writeNativeContractSafetensorsHeaderWithPayload(t, core.PathJoin(dir, "model.safetensors"), `{
		"model.layers.0.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[0,16]},
		"model.layers.0.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[16,32]}
	}`, 32)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !nativeContractHasNoteContaining(inspection.Notes, "duplicate tensor key") {
		t.Fatalf("notes = %+v, want duplicate safetensors key note", inspection.Notes)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, duplicate-key safetensors should not be supported", inspection, inspection.Labels)
	}
	if _, ok := inspection.Labels["safetensors_tensors"]; ok {
		t.Fatalf("labels = %+v, duplicate-key safetensors should not report tensor count", inspection.Labels)
	}
}

func TestNativeContract_ModelPackInspectorSafetensorsValidatesDTypeShapeByteSpan_Bad(t *testing.T) {
	cases := []struct {
		name   string
		header string
		note   string
	}{
		{
			name:   "unknown_dtype",
			header: `{"model.layers.0.weight":{"dtype":"FLOAT9000","shape":[2,4],"data_offsets":[0,16]}}`,
			note:   "unsupported dtype",
		},
		{
			name:   "shape_byte_mismatch",
			header: `{"model.layers.0.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[0,12]}}`,
			note:   "byte span 12 does not match shape bytes 16",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"Qwen3ForCausalLM"}`)
			writeNativeContractSafetensorsHeader(t, core.PathJoin(dir, "model.safetensors"), tc.header)

			inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
			if err != nil {
				t.Fatalf("InspectModelPack: %v", err)
			}
			if !nativeContractHasNoteContaining(inspection.Notes, tc.note) {
				t.Fatalf("notes = %+v, want %q", inspection.Notes, tc.note)
			}
			if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
				t.Fatalf("inspection = %+v labels=%+v, malformed safetensors should not be supported", inspection, inspection.Labels)
			}
			if _, ok := inspection.Labels["safetensors_tensors"]; ok {
				t.Fatalf("labels = %+v, malformed safetensors should not report tensor count", inspection.Labels)
			}
		})
	}
}

func TestNativeContract_ModelPackInspectorSafetensorsRejectsOverlappingTensorOffsets_Bad(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"Qwen3ForCausalLM"}`)
	writeNativeContractSafetensorsHeaderWithPayload(t, core.PathJoin(dir, "model.safetensors"), `{
		"model.layers.0.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[0,16]},
		"model.layers.1.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[8,24]}
	}`, 24)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !nativeContractHasNoteContaining(inspection.Notes, "overlaps") {
		t.Fatalf("notes = %+v, want overlapping tensor offsets note", inspection.Notes)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, overlapping safetensors should not be supported", inspection, inspection.Labels)
	}
	if _, ok := inspection.Labels["safetensors_tensors"]; ok {
		t.Fatalf("labels = %+v, overlapping safetensors should not report tensor count", inspection.Labels)
	}
}

func TestNativeContract_ModelPackInspectorMalformedSafetensorsShardClearsWeightLabels_Bad(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"Qwen3ForCausalLM"}`)
	writeNativeContractSafetensors(t, core.PathJoin(dir, "model-00001-of-00002.safetensors"))
	writeNativeContractSafetensorsHeader(t, core.PathJoin(dir, "model-00002-of-00002.safetensors"), `{"model.layers.1.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[16,0]}}`)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, malformed shard should not be supported", inspection, inspection.Labels)
	}
	if _, ok := inspection.Labels["safetensors_tensors"]; ok {
		t.Fatalf("labels = %+v, malformed shard should clear partial safetensors summaries", inspection.Labels)
	}
	if _, ok := inspection.Labels["safetensors_dtypes"]; ok {
		t.Fatalf("labels = %+v, malformed shard should clear partial safetensors dtype summaries", inspection.Labels)
	}
}

func TestNativeContract_ModelPackInspectorMalformedGGUF_Bad(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"Qwen3ForCausalLM"}`)
	writeNativeContractFile(t, core.PathJoin(dir, "model.gguf"), "not a gguf file")

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !nativeContractHasNoteContaining(inspection.Notes, "GGUF metadata could not be parsed") {
		t.Fatalf("notes = %+v, want malformed GGUF parse note", inspection.Notes)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, malformed GGUF should not be supported", inspection, inspection.Labels)
	}
	if _, ok := inspection.Labels["gguf_tensors"]; ok {
		t.Fatalf("labels = %+v, malformed GGUF should not report tensor count", inspection.Labels)
	}
}

func TestNativeContract_ModelPackInspectorMissingSafetensorsArchitecture_Bad(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractSafetensors(t, core.PathJoin(dir, "model.safetensors"))

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "true" || inspection.Labels["architecture_detected"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, missing architecture should not be supported", inspection, inspection.Labels)
	}
	if _, ok := inspection.Labels["memory_fit"]; ok {
		t.Fatalf("labels = %+v, unsupported model pack should not report memory fit", inspection.Labels)
	}
	if !nativeContractHasNoteContaining(inspection.Notes, "model architecture could not be detected") {
		t.Fatalf("notes = %+v, want missing architecture note", inspection.Notes)
	}
}

func TestNativeContract_ModelPackInspectorMalformedConfigDoesNotImplySupport_Bad(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{"model_type":`)
	writeNativeContractSafetensors(t, core.PathJoin(dir, "model.safetensors"))

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !nativeContractHasNoteContaining(inspection.Notes, "config.json could not be parsed") {
		t.Fatalf("notes = %+v, want malformed config parse note", inspection.Notes)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "true" || inspection.Labels["architecture_detected"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, malformed config should not be supported", inspection, inspection.Labels)
	}
	if _, ok := inspection.Labels["memory_fit"]; ok {
		t.Fatalf("labels = %+v, unsupported model pack should not report memory fit", inspection.Labels)
	}
}

type fakeNativeRuntime struct {
	available    bool
	device       nativeDeviceInfo
	model        nativeModel
	loadPath     string
	loadPaths    []string
	loadConfig   nativeLoadConfig
	loadConfigs  []nativeLoadConfig
	kernelStatus hipKernelStatus
}

func (runtime *fakeNativeRuntime) Available() bool { return runtime.available }
func (runtime *fakeNativeRuntime) DeviceInfo() nativeDeviceInfo {
	return runtime.device
}
func (runtime *fakeNativeRuntime) KernelStatus() hipKernelStatus {
	if runtime == nil || runtime.kernelStatus == (hipKernelStatus{}) {
		return defaultHIPKernelStatus()
	}
	return normalizeHIPKernelStatus(runtime.kernelStatus)
}
func (runtime *fakeNativeRuntime) LoadModel(path string, cfg nativeLoadConfig) (nativeModel, error) {
	runtime.loadPath = path
	runtime.loadPaths = append(runtime.loadPaths, path)
	runtime.loadConfig = cfg
	runtime.loadConfigs = append(runtime.loadConfigs, cfg)
	if runtime.model == nil {
		runtime.model = &fakeNativeModel{}
	}
	return runtime.model, nil
}

type fakeNativeModel struct {
	tokens                   []inference.Token
	adapter                  inference.AdapterIdentity
	loadAdapterIdentity      inference.AdapterIdentity
	adapterLoads             []string
	unloadCalls              int
	adapterErr               error
	unloadAdapterErr         error
	restoreAdapterErr        error
	kernelStatus             hipKernelStatus
	metrics                  inference.GenerateMetrics
	closeCalls               int
	closeErr                 error
	afterStream              func()
	baseTokenDelay           time.Duration
	adapterTokenDelay        time.Duration
	classLogits              [][]float32
	classLogitsAlways        bool
	evalLossKernelOK         bool
	evalLossKernelOut        hipCrossEntropyLossResult
	evalLossKernelErr        error
	evalLossKernelCalls      int
	distillKernelOK          bool
	distillKernelOut         hipDistillationKLLossResult
	distillKernelErr         error
	distillKernelCalls       int
	grpoKernelOK             bool
	grpoKernelOut            []float64
	grpoKernelErr            error
	grpoKernelCalls          int
	classifyResults          []inference.ClassifyResult
	classifyPrompts          [][]string
	generatePrompts          []string
	generateConfigs          []inference.GenerateConfig
	mutateGenerateConfig     bool
	batchErr                 error
	batchResults             []inference.BatchResult
	encodeResult             []int32
	encodeByText             map[string][]int32
	chatTemplateResult       string
	chatTemplateErr          error
	decodeMutatesInput       bool
	chatMutatesInput         bool
	chatTemplateMutatesInput bool
	mutatePromptInputs       bool
}

func (model *fakeNativeModel) Generate(ctx context.Context, prompt string, cfg inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if model.mutateGenerateConfig {
		mutateGenerateConfig(&cfg)
	}
	model.generatePrompts = append(model.generatePrompts, prompt)
	model.generateConfigs = append(model.generateConfigs, cfg)
	return func(yield func(inference.Token) bool) {
		defer func() {
			if model.afterStream != nil {
				model.afterStream()
			}
		}()
		delay := model.baseTokenDelay
		if !adapterIdentityIsZero(model.adapter) && model.adapterTokenDelay > 0 {
			delay = model.adapterTokenDelay
		}
		for _, token := range model.tokens {
			if delay > 0 {
				select {
				case <-ctx.Done():
					return
				case <-time.After(delay):
				}
			}
			if !yield(token) {
				return
			}
		}
	}, func() error { return nil }
}
func (model *fakeNativeModel) Chat(ctx context.Context, messages []inference.Message, cfg inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	if model.chatMutatesInput && len(messages) > 0 {
		messages[0].Role = "mutated"
		messages[0].Content = "mutated"
	}
	return model.Generate(ctx, "", cfg)
}
func (model *fakeNativeModel) Classify(_ context.Context, prompts []string, cfg inference.GenerateConfig) ([]inference.ClassifyResult, error) {
	if model.mutatePromptInputs && len(prompts) > 0 {
		prompts[0] = "mutated"
	}
	if model.mutateGenerateConfig {
		mutateGenerateConfig(&cfg)
	}
	model.classifyPrompts = append(model.classifyPrompts, append([]string(nil), prompts...))
	if model.classifyResults != nil {
		return model.classifyResults, nil
	}
	out := make([]inference.ClassifyResult, len(prompts))
	for i := range prompts {
		out[i] = inference.ClassifyResult{Token: inference.Token{ID: int32(i + 1), Text: "ok"}}
		if (cfg.ReturnLogits || model.classLogitsAlways) && i < len(model.classLogits) {
			out[i].Logits = append([]float32(nil), model.classLogits[i]...)
		}
	}
	return out, nil
}
func (model *fakeNativeModel) BatchGenerate(_ context.Context, prompts []string, cfg inference.GenerateConfig) ([]inference.BatchResult, error) {
	if model.mutatePromptInputs && len(prompts) > 0 {
		prompts[0] = "mutated"
	}
	if model.mutateGenerateConfig {
		mutateGenerateConfig(&cfg)
	}
	model.generateConfigs = append(model.generateConfigs, cfg)
	if model.batchErr != nil {
		return nil, model.batchErr
	}
	if model.batchResults != nil {
		return model.batchResults, nil
	}
	out := make([]inference.BatchResult, len(prompts))
	for i := range prompts {
		out[i] = inference.BatchResult{Tokens: append([]inference.Token(nil), model.tokens...)}
	}
	return out, nil
}

func mutateGenerateConfig(cfg *inference.GenerateConfig) {
	if cfg == nil {
		return
	}
	if len(cfg.StopTokens) > 0 {
		cfg.StopTokens[0] = 99
	}
}

func (model *fakeNativeModel) Encode(text string) []int32 {
	if model.encodeByText != nil {
		if tokens, ok := model.encodeByText[text]; ok {
			return append([]int32(nil), tokens...)
		}
	}
	if model.encodeResult != nil {
		return model.encodeResult
	}
	if core.Trim(text) == "" {
		return nil
	}
	parts := core.Split(core.Trim(text), " ")
	ids := make([]int32, len(parts))
	for i := range parts {
		ids[i] = int32(i + 1)
	}
	return ids
}
func (model *fakeNativeModel) Decode(ids []int32) string {
	if model.decodeMutatesInput && len(ids) > 0 {
		ids[0] = 99
	}
	return core.Sprintf("%d tokens", len(ids))
}
func (model *fakeNativeModel) ApplyChatTemplate(messages []inference.Message) (string, error) {
	if model.chatTemplateMutatesInput && len(messages) > 0 {
		messages[0].Role = "mutated"
		messages[0].Content = "mutated"
	}
	if model.chatTemplateErr != nil {
		return "", model.chatTemplateErr
	}
	if model.chatTemplateResult != "" {
		return model.chatTemplateResult, nil
	}
	var text string
	for _, message := range messages {
		text += message.Role + ":" + message.Content + "\n"
	}
	return text, nil
}
func (model *fakeNativeModel) LoadAdapter(path string) (inference.AdapterIdentity, error) {
	model.adapterLoads = append(model.adapterLoads, path)
	if adapterIdentityIsZero(model.adapter) && model.restoreAdapterErr != nil {
		return inference.AdapterIdentity{}, model.restoreAdapterErr
	}
	if model.adapterErr != nil {
		return inference.AdapterIdentity{}, model.adapterErr
	}
	if !adapterIdentityIsZero(model.loadAdapterIdentity) {
		model.adapter = cloneAdapterIdentity(model.loadAdapterIdentity)
		if model.adapter.Path == "" {
			model.adapter.Path = path
		}
		if model.adapter.Format == "" {
			model.adapter.Format = "lora"
		}
		return cloneAdapterIdentity(model.adapter), nil
	}
	model.adapter = inference.AdapterIdentity{Path: path, Format: "lora"}
	return cloneAdapterIdentity(model.adapter), nil
}
func (model *fakeNativeModel) UnloadAdapter() error {
	model.unloadCalls++
	if model.unloadAdapterErr != nil {
		return model.unloadAdapterErr
	}
	model.adapter = inference.AdapterIdentity{}
	return nil
}
func (model *fakeNativeModel) ActiveAdapter() inference.AdapterIdentity {
	return cloneAdapterIdentity(model.adapter)
}
func (model *fakeNativeModel) KernelStatus() hipKernelStatus {
	if model == nil {
		return defaultHIPKernelStatus()
	}
	return normalizeHIPKernelStatus(model.kernelStatus)
}
func (model *fakeNativeModel) RunEvalCrossEntropyLoss(_ context.Context, _ [][]float32, _ []int) (hipCrossEntropyLossResult, bool, error) {
	model.evalLossKernelCalls++
	if !model.evalLossKernelOK {
		return hipCrossEntropyLossResult{}, false, nil
	}
	return model.evalLossKernelOut, true, model.evalLossKernelErr
}
func (model *fakeNativeModel) RunDistillationKLLoss(_ context.Context, _, _ [][]float32, _ float64) (hipDistillationKLLossResult, bool, error) {
	model.distillKernelCalls++
	if !model.distillKernelOK {
		return hipDistillationKLLossResult{}, false, nil
	}
	return model.distillKernelOut, true, model.distillKernelErr
}
func (model *fakeNativeModel) RunGRPOAdvantage(_ context.Context, _ []float64) ([]float64, bool, error) {
	model.grpoKernelCalls++
	if !model.grpoKernelOK {
		return nil, false, nil
	}
	return append([]float64(nil), model.grpoKernelOut...), true, model.grpoKernelErr
}
func (model *fakeNativeModel) Metrics() inference.GenerateMetrics {
	if model.metrics != (inference.GenerateMetrics{}) {
		return model.metrics
	}
	return inference.GenerateMetrics{GeneratedTokens: len(model.tokens), DecodeDuration: time.Millisecond}
}
func (model *fakeNativeModel) Close() error {
	model.closeCalls++
	if model.closeErr != nil {
		return model.closeErr
	}
	return nil
}

type fakeNativeEmbeddingModel struct {
	*fakeNativeModel
	embedResult  *inference.EmbeddingResult
	rerankResult *inference.RerankResult
}

func positiveFloatLabel(t *testing.T, labels map[string]string, key string) float64 {
	t.Helper()
	value := floatLabel(t, labels, key)
	if value <= 0 {
		t.Fatalf("labels[%s] = %q, want positive float", key, labels[key])
	}
	return value
}

func floatLabel(t *testing.T, labels map[string]string, key string) float64 {
	t.Helper()
	raw := labels[key]
	if raw == "" {
		t.Fatalf("labels[%s] is empty", key)
	}
	value, err := strconv.ParseFloat(raw, 64)
	core.RequireNoError(t, err)
	return value
}

func (model *fakeNativeEmbeddingModel) Embed(_ context.Context, req inference.EmbeddingRequest) (*inference.EmbeddingResult, error) {
	if model.embedResult != nil {
		return model.embedResult, nil
	}
	return &inference.EmbeddingResult{
		Vectors: [][]float32{{1, 0}},
		Usage:   inference.EmbeddingUsage{PromptTokens: len(req.Input), TotalTokens: len(req.Input)},
		Labels:  map[string]string{"backend": "fake"},
	}, nil
}

func (model *fakeNativeEmbeddingModel) Rerank(_ context.Context, req inference.RerankRequest) (*inference.RerankResult, error) {
	if model.rerankResult != nil {
		return model.rerankResult, nil
	}
	return &inference.RerankResult{
		Results: []inference.RerankScore{{Index: 1, Score: 0.9, Text: req.Documents[1]}},
		Labels:  map[string]string{"backend": "fake"},
	}, nil
}

func nativeContractProbeEvent(events []inference.ProbeEvent, kind inference.ProbeEventKind) (inference.ProbeEvent, bool) {
	for _, event := range events {
		if event.Kind == kind {
			return event, true
		}
	}
	return inference.ProbeEvent{}, false
}

type singleInferenceSample struct {
	sample inference.DatasetSample
	done   bool
}

func (stream *singleInferenceSample) Next() (inference.DatasetSample, bool, error) {
	if stream.done {
		return inference.DatasetSample{}, false, nil
	}
	stream.done = true
	return stream.sample, true, nil
}

type sliceInferenceSamples struct {
	samples []inference.DatasetSample
	index   int
}

func (stream *sliceInferenceSamples) Next() (inference.DatasetSample, bool, error) {
	if stream == nil || stream.index >= len(stream.samples) {
		return inference.DatasetSample{}, false, nil
	}
	sample := stream.samples[stream.index]
	stream.index++
	return sample, true, nil
}

type errorInferenceSamples struct {
	err error
}

func (stream *errorInferenceSamples) Next() (inference.DatasetSample, bool, error) {
	return inference.DatasetSample{}, false, stream.err
}

func nativeContractGGUF(t *testing.T) string {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "native-contract.gguf")
	buf := core.NewBuffer()
	writeUint32 := func(v uint32) { core.RequireNoError(t, binary.Write(buf, binary.LittleEndian, v)) }
	writeUint64 := func(v uint64) { core.RequireNoError(t, binary.Write(buf, binary.LittleEndian, v)) }
	writeString := func(v string) {
		writeUint64(uint64(len(v)))
		_, err := buf.Write([]byte(v))
		core.RequireNoError(t, err)
	}
	writeKVString := func(key, value string) {
		writeString(key)
		writeUint32(8)
		writeString(value)
	}
	writeKVUint32 := func(key string, value uint32) {
		writeString(key)
		writeUint32(4)
		writeUint32(value)
	}

	writeUint32(0x46554747)
	writeUint32(3)
	writeUint64(0)
	writeUint64(6)
	writeKVString("general.architecture", "qwen3")
	writeKVString("general.name", "native-test")
	writeKVString("general.size_label", "0B")
	writeKVUint32("general.file_type", 15)
	writeKVUint32("qwen3.context_length", 32768)
	writeKVUint32("qwen3.block_count", 28)

	result := core.WriteFile(path, buf.Bytes(), 0o644)
	core.RequireTrue(t, result.OK)
	return path
}

func writeNativeContractFile(t *testing.T, path, content string) {
	t.Helper()
	result := core.WriteFile(path, []byte(content), 0o644)
	core.RequireTrue(t, result.OK)
}

func writeNativeContractSafetensors(t *testing.T, path string) {
	t.Helper()
	header := []byte(`{"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[0,16]},"__metadata__":{"format":"pt"}}`)
	writeNativeContractSafetensorsHeader(t, path, string(header))
}

func writeNativeContractSafetensorsHeader(t *testing.T, path, headerText string) {
	t.Helper()
	writeNativeContractSafetensorsHeaderWithPayload(t, path, headerText, 16)
}

func writeNativeContractSafetensorsHeaderWithPayload(t *testing.T, path, headerText string, payloadBytes int) {
	t.Helper()
	header := []byte(headerText)
	buf := core.NewBuffer()
	core.RequireNoError(t, binary.Write(buf, binary.LittleEndian, uint64(len(header))))
	_, err := buf.Write(header)
	core.RequireNoError(t, err)
	_, err = buf.Write(make([]byte, payloadBytes))
	core.RequireNoError(t, err)
	result := core.WriteFile(path, buf.Bytes(), 0o644)
	core.RequireTrue(t, result.OK)
}

func nativeContractSafetensorsPack(t *testing.T, config string) string {
	t.Helper()
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), config)
	writeNativeContractSafetensors(t, core.PathJoin(dir, "model.safetensors"))
	return dir
}

func nativeInspectionHasCapability(inspection *inference.ModelPackInspection, id inference.CapabilityID) bool {
	_, ok := nativeInspectionCapability(inspection, id)
	return ok
}

func nativeInspectionCapability(inspection *inference.ModelPackInspection, id inference.CapabilityID) (inference.Capability, bool) {
	if inspection == nil {
		return inference.Capability{}, false
	}
	for _, capability := range inspection.Capabilities {
		if capability.ID == id {
			return capability, true
		}
	}
	return inference.Capability{}, false
}

func nativeContractHasNoteContaining(notes []string, needle string) bool {
	for _, note := range notes {
		if strings.Contains(note, needle) {
			return true
		}
	}
	return false
}

func nativeContractMetadataFixtureKernels() map[inference.CapabilityID]string {
	return map[inference.CapabilityID]string{
		inference.CapabilityMoERouting:     hipKernelNameMoERouter,
		inference.CapabilityMoELazyExperts: hipKernelNameMoELazy,
		inference.CapabilityJANGTQ:         hipKernelNameJANGTQ,
		inference.CapabilityCodebookVQ:     hipKernelNameCodebook,
	}
}

func assertCSVLabelContainsAll(t *testing.T, label string, value string, required []string) {
	t.Helper()
	values := splitProductionCSVLabel(value)
	for _, metric := range required {
		if !stringSliceContains(values, metric) {
			t.Fatalf("%s = %q, missing %q", label, value, metric)
		}
	}
}

func nativeContractSharedCapabilityIDs() []inference.CapabilityID {
	return []inference.CapabilityID{
		inference.CapabilityModelLoad,
		inference.CapabilityGenerate,
		inference.CapabilityChat,
		inference.CapabilityClassify,
		inference.CapabilityBatchGenerate,
		inference.CapabilityTokenizer,
		inference.CapabilityChatTemplate,
		inference.CapabilityLoRAInference,
		inference.CapabilityLoRATraining,
		inference.CapabilityStateBundle,
		inference.CapabilityKVSnapshot,
		inference.CapabilityPromptCache,
		inference.CapabilityKVCachePlanning,
		inference.CapabilityMemoryPlanning,
		inference.CapabilityModelFit,
		inference.CapabilityBenchmark,
		inference.CapabilityEvaluation,
		inference.CapabilityDistillation,
		inference.CapabilityGRPO,
		inference.CapabilityQuantization,
		inference.CapabilityModelMerge,
		inference.CapabilityProbeEvents,
		inference.CapabilityAttentionProbe,
		inference.CapabilityLogitProbe,
		inference.CapabilityResponsesAPI,
		inference.CapabilityAnthropicMessages,
		inference.CapabilityOllamaCompat,
		inference.CapabilityEmbeddings,
		inference.CapabilityRerank,
		inference.CapabilityScheduler,
		inference.CapabilityRequestCancel,
		inference.CapabilityCacheBlocks,
		inference.CapabilityCacheDisk,
		inference.CapabilityCacheWarm,
		inference.CapabilityToolParse,
		inference.CapabilityReasoningParse,
		inference.CapabilitySpeculativeDecode,
		inference.CapabilityPromptLookupDecode,
		inference.CapabilityMoERouting,
		inference.CapabilityMoELazyExperts,
		inference.CapabilityJANGTQ,
		inference.CapabilityCodebookVQ,
		inference.CapabilityAgentMemory,
		inference.CapabilityStateWake,
		inference.CapabilityStateSleep,
		inference.CapabilityStateFork,
	}
}
