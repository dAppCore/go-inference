// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"strconv"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	inferdecode "dappco.re/go/inference/decode"
)

var productionMTPSink ProductionMTPPromotionDecision
var productionMTPEvidenceSink ProductionMTPPromotionEvidence

func TestProductionMTPPolicy_Defaults_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	core.AssertEqual(t, officialGemma4E2BTargetModelID, policy.TargetModelID)
	core.AssertEqual(t, officialGemma4E2BAssistantModelID, policy.AssistantModelID)
	core.AssertEqual(t, "mtp_attached_drafter", policy.Mode)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, policy.DefaultDraftTokens)
	core.AssertEqual(t, ProductionMTPPromotionMinRetainedTurns, policy.MinimumRetainedTurns)
	core.AssertEqual(t, ProductionLaneBookTurnCount, policy.MinimumRetainedTurns)
	core.AssertEqual(t, float64(productionLaneRetainedVisibleTokensSec), policy.MinimumVisibleTokensPerSec)
	core.AssertEqual(t, true, policy.EnabledByDefault)
	core.AssertEqual(t, true, policy.RequiresRetainedWorkflow)
	core.AssertEqual(t, true, policy.RequiresGreedyParity)
	core.AssertEqual(t, true, policy.RequiresSideBySideBenchmark)
	core.AssertEqual(t, strconv.Itoa(ProductionMTPAssistantCentroidIntermediateTopK), productionMTPAssistantCentroidIntermediateTopKLabel)
	core.AssertEqual(t, strconv.Itoa(ProductionMTPAssistantOrderedEmbeddingCentroids), productionMTPAssistantOrderedEmbeddingCentroidsLabel)
	core.AssertEqual(t, strconv.Itoa(ProductionMTPDefaultDraftTokens), productionMTPDefaultDraftTokensLabel)
	core.AssertEqual(t, productionMTPAssistantOrderedEmbeddingCentroidsLabel+"x"+strconv.Itoa(ProductionMTPAssistantTokenOrderingVocabSize/ProductionMTPAssistantOrderedEmbeddingCentroids), productionMTPAssistantTokenOrderingShapeLabel)
	if !intSliceEqual(policy.RequiredDraftTokenSweeps, []int{1, 2, 4}) {
		t.Fatalf("RequiredDraftTokenSweeps = %v, want 1/2/4", policy.RequiredDraftTokenSweeps)
	}
	for _, metric := range []string{
		"retained_workflow",
		"turns",
		"greedy_output_matches",
		"quality_flags",
		"speculative_draft_model_path",
		"speculative_draft_tokens",
		"target_only_visible_tokens_per_sec",
		"mtp_visible_tokens_per_sec",
		"mtp_target_tokens_per_sec",
		"mtp_warm_decode_tokens_per_sec",
		"target_only_wall_duration",
		"mtp_wall_duration",
		"target_only_restore_duration",
		"mtp_restore_duration",
		"target_only_peak_memory_bytes",
		"mtp_peak_memory_bytes",
		"target_only_active_plus_cache_memory_bytes",
		"mtp_active_plus_cache_memory_bytes",
		"target_only_energy_joules",
		"mtp_energy_joules",
		"same_load_policy",
		"target_only_cache_mode",
		"mtp_cache_mode",
		"mtp_observed_draft_token_sweeps",
		"mtp_proposed_tokens",
		"mtp_accepted_tokens",
		"mtp_rejected_tokens",
		"mtp_target_verify_calls",
		"mtp_draft_calls",
		"attached_drafter_retained_state_entrypoint",
		"attached_drafter_retained_state_required",
		"attached_drafter_state_source",
		"attached_drafter_prompt_replay_fallback",
		"attached_drafter_native_attachment",
		"attached_drafter_native_handoff",
		"attached_drafter_target_retained_decode",
		"attached_drafter_target_retained_state_decode",
		"attached_drafter_assistant_verify",
		"attached_drafter_assistant_state_verify",
		"attached_drafter_assistant_draft_step_input_bridge",
		"attached_drafter_assistant_draft_step_hidden_runtime",
		"attached_drafter_assistant_draft_step_proposal_runtime",
		"attached_drafter_target_gemma4_size",
		"attached_drafter_target_gemma4_quant_mode",
		"attached_drafter_target_gemma4_quant_group",
		"attached_drafter_target_gemma4_runtime",
		"attached_drafter_target_gemma4_generate_status",
		"attached_drafter_target_production_quant_model",
		"attached_drafter_assistant_gemma4_size",
		"attached_drafter_assistant_gemma4_quant_mode",
		"attached_drafter_assistant_gemma4_runtime",
		"attached_drafter_assistant_gemma4_generate_status",
		"attached_drafter_assistant_production_quant_model",
		"attached_drafter_assistant_production_quant_pack",
		"attached_drafter_assistant_production_quant_tier",
		"attached_drafter_assistant_production_quant_mtp_assistant",
		"assistant_architecture",
		"assistant_ordered_embeddings",
		"assistant_centroids",
		"assistant_centroid_intermediate_top_k",
		"assistant_four_layer_drafter",
		"assistant_token_ordering_dtype",
		"assistant_token_ordering_shape",
		"gemma4_family_pair_verified",
	} {
		if !stringSliceContains(policy.RequiredMetrics, metric) {
			t.Fatalf("RequiredMetrics = %v, missing %q", policy.RequiredMetrics, metric)
		}
	}

	policy.RequiredDraftTokenSweeps[0] = 99
	policy.RequiredMetrics[0] = "mutated"
	next := DefaultProductionMTPPolicy()
	if next.RequiredDraftTokenSweeps[0] == 99 || next.RequiredMetrics[0] == "mutated" {
		t.Fatalf("DefaultProductionMTPPolicy leaked mutable slices: %+v", next)
	}
}

func TestProductionMTPPromotion_Good_AcceptsFasterRetainedOfficialPair(t *testing.T) {
	decision := EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), productionMTPPassingEvidence())

	if !decision.EnableByDefault {
		t.Fatalf("decision = %+v, want MTP promotion", decision)
	}
	core.AssertGreater(t, decision.WallSpeedup, float64(1))
	core.AssertGreater(t, decision.VisibleSpeedup, float64(1))
	core.AssertGreater(t, decision.RestoreSpeedup, float64(1))
	core.AssertGreater(t, decision.EnergySavings, float64(0))
	core.AssertEqual(t, 0.75, decision.AcceptanceRate)
}

func TestProductionMTPAttachedDrafterEvidence_Good_FillsStaticEvidenceOnly(t *testing.T) {
	plan, err := PlanAttachedDrafter(
		productionMTPE2BQ6TargetModel(),
		productionMTPE2BBF16AssistantModel(),
	)
	core.RequireNoError(t, err)
	evidence := ProductionMTPPromotionEvidence{
		MTPDraftTokenSchedule: []int{1, 2, 4},
	}

	err = ApplyProductionMTPAttachedDrafterPlanEvidence(&evidence, plan)

	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmGemma4MTPAssistantPath("E2B", "bf16"), evidence.SpeculativeDraftModelPath)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, evidence.SpeculativeDraftTokens)
	core.AssertEqual(t, true, evidence.AttachedDrafterRetainedStateEntrypoint)
	core.AssertEqual(t, true, evidence.AttachedDrafterRetainedStateRequired)
	core.AssertEqual(t, "rocm_state_session_runtime_kv", evidence.AttachedDrafterStateSource)
	core.AssertEqual(t, "forbidden", evidence.AttachedDrafterPromptReplayFallback)
	core.AssertEqual(t, hipKernelStatusNotLinked, evidence.AttachedDrafterNativeAttachment)
	core.AssertEqual(t, attachedDrafterNativeHandoffTargetDecodeOnly, evidence.AttachedDrafterNativeHandoff)
	core.AssertEqual(t, hipKernelStatusLinked, evidence.AttachedDrafterTargetRetainedDecode)
	core.AssertEqual(t, hipKernelStatusLinked, evidence.AttachedDrafterTargetRetainedState)
	core.AssertEqual(t, hipKernelStatusNotLinked, evidence.AttachedDrafterAssistantVerify)
	core.AssertEqual(t, hipKernelStatusNotLinked, evidence.AttachedDrafterAssistantStateVerify)
	core.AssertEqual(t, "E2B", evidence.TargetGemma4Size)
	core.AssertEqual(t, "q6", evidence.TargetGemma4QuantMode)
	core.AssertEqual(t, 64, evidence.TargetGemma4QuantGroup)
	core.AssertEqual(t, Gemma4RuntimeMLXAffine, evidence.TargetGemma4Runtime)
	core.AssertEqual(t, Gemma4GenerateLinked, evidence.TargetGemma4GenerateStatus)
	core.AssertEqual(t, ProductionLaneCurrentModelID, evidence.TargetProductionQuantModelID)
	core.AssertEqual(t, ProductionLaneModelID, evidence.TargetProductionQuantLockedModelID)
	core.AssertEqual(t, "E2B", evidence.AssistantGemma4Size)
	core.AssertEqual(t, "bf16", evidence.AssistantGemma4QuantMode)
	core.AssertEqual(t, Gemma4RuntimeBF16, evidence.AssistantGemma4Runtime)
	core.AssertEqual(t, Gemma4GenerateLoadOnly, evidence.AssistantGemma4GenerateStatus)
	core.AssertEqual(t, officialGemma4E2BAssistantModelID, evidence.AssistantProductionQuantModelID)
	core.AssertEqual(t, "E2B:assistant-bf16", evidence.AssistantProductionQuantPack)
	core.AssertEqual(t, "mtp-assistant", evidence.AssistantProductionQuantTier)
	core.AssertEqual(t, true, evidence.AssistantProductionQuantMTPAssistant)
	core.AssertEqual(t, "gemma4", evidence.AssistantProductionQuantTargetFamily)
	core.AssertEqual(t, officialGemma4E2BAssistantArchitecture, evidence.AssistantArchitecture)
	core.AssertEqual(t, true, evidence.AssistantOrderedEmbeddings)
	core.AssertEqual(t, ProductionMTPAssistantOrderedEmbeddingCentroids, evidence.AssistantCentroids)
	core.AssertEqual(t, ProductionMTPAssistantCentroidIntermediateTopK, evidence.AssistantCentroidIntermediateTopK)
	core.AssertEqual(t, true, evidence.AssistantFourLayerDrafter)
	core.AssertEqual(t, "int64", evidence.AssistantTokenOrderingDType)
	if !intSliceEqual(evidence.AssistantTokenOrderingShape, []int{ProductionMTPAssistantOrderedEmbeddingCentroids, ProductionMTPAssistantTokenOrderingVocabSize / ProductionMTPAssistantOrderedEmbeddingCentroids}) {
		t.Fatalf("AssistantTokenOrderingShape = %v, want ordered centroid shape", evidence.AssistantTokenOrderingShape)
	}
	core.AssertEqual(t, true, evidence.Gemma4FamilyPairVerified)
	core.AssertEqual(t, true, evidence.OfficialPairVerified)
	core.AssertEqual(t, officialGemma4E2BTargetModelID, evidence.OfficialTargetModelID)
	core.AssertEqual(t, officialGemma4E2BTargetRevision, evidence.OfficialTargetRevision)
	core.AssertEqual(t, officialGemma4E2BAssistantModelID, evidence.OfficialAssistantModelID)
	core.AssertEqual(t, officialGemma4E2BAssistantRevision, evidence.OfficialAssistantRevision)
	if !intSliceEqual(evidence.MTPDraftTokenSchedule, []int{1, 2, 4}) {
		t.Fatalf("MTPDraftTokenSchedule = %v, want existing measured schedule preserved", evidence.MTPDraftTokenSchedule)
	}
	decision := EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "retained workflow")
}

func TestProductionMTPAttachedDrafterPlanInfersPathOnlyQuant(t *testing.T) {
	for _, tc := range []struct {
		name             string
		targetPath       string
		targetBits       int
		targetGroup      int
		wantSize         string
		wantMode         string
		wantModel        string
		wantLockedModel  string
		wantOfficialPair string
		wantFamilyPair   string
	}{
		{name: "e2b_official", targetPath: "/models/lmstudio-community-gemma-4-e2b-it-6bit", targetBits: 6, targetGroup: 64, wantSize: "E2B", wantMode: "q6", wantModel: ProductionLaneCurrentModelID, wantLockedModel: ProductionLaneModelID, wantOfficialPair: "true", wantFamilyPair: "true"},
		{name: "e4b_path_only", targetPath: "/models/lmstudio-community-gemma-4-e4b-it-8bit", targetBits: 8, targetGroup: 64, wantSize: "E4B", wantMode: "q8", wantModel: "lmstudio-community/gemma-4-E4B-it-MLX-8bit", wantOfficialPair: "false", wantFamilyPair: "true"},
		{name: "12b_path_only", targetPath: "/models/lmstudio-community-gemma-4-12b-it-6bit", targetBits: 6, targetGroup: 64, wantSize: "12B", wantMode: "q6", wantModel: "mlx-community/gemma-4-12b-it-6bit", wantOfficialPair: "false", wantFamilyPair: "true"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := PlanAttachedDrafter(
				&rocmModel{
					modelPath: tc.targetPath,
					modelInfo: inference.ModelInfo{
						Architecture: "gemma4_text",
					},
				},
				&rocmModel{
					modelPath: rocmGemma4MTPAssistantPath(tc.wantSize, "bf16"),
					modelInfo: inference.ModelInfo{
						Architecture: officialGemma4E2BAssistantArchitecture,
					},
				},
			)

			core.RequireNoError(t, err)
			core.AssertEqual(t, tc.targetBits, plan.Target.QuantBits)
			core.AssertEqual(t, tc.targetGroup, plan.Target.QuantGroup)
			core.AssertEqual(t, 16, plan.Draft.QuantBits)
			core.AssertEqual(t, "gemma4_text", plan.Target.Architecture)
			core.AssertEqual(t, officialGemma4E2BAssistantArchitecture, plan.Draft.Architecture)
			core.AssertEqual(t, tc.wantSize, plan.Labels["attached_drafter_target_gemma4_size"])
			core.AssertEqual(t, tc.wantMode, plan.Labels["attached_drafter_target_gemma4_quant_mode"])
			core.AssertEqual(t, strconv.Itoa(tc.targetGroup), plan.Labels["attached_drafter_target_gemma4_quant_group"])
			core.AssertEqual(t, Gemma4GenerateLinked, plan.Labels["attached_drafter_target_gemma4_generate_status"])
			core.AssertEqual(t, "true", plan.Labels["attached_drafter_target_gemma4_pack_supported"])
			core.AssertEqual(t, "true", plan.Labels["attached_drafter_target_gemma4_runnable_on_card"])
			core.AssertEqual(t, tc.wantModel, plan.Labels["attached_drafter_target_production_quant_model"])
			core.AssertEqual(t, tc.wantLockedModel, plan.Labels["attached_drafter_target_production_quant_locked_model"])
			core.AssertEqual(t, tc.wantSize, plan.Labels["attached_drafter_assistant_gemma4_size"])
			core.AssertEqual(t, "bf16", plan.Labels["attached_drafter_assistant_gemma4_quant_mode"])
			core.AssertEqual(t, Gemma4GenerateLoadOnly, plan.Labels["attached_drafter_assistant_gemma4_generate_status"])
			core.AssertEqual(t, "true", plan.Labels["attached_drafter_assistant_gemma4_pack_supported"])
			core.AssertEqual(t, "true", plan.Labels["attached_drafter_assistant_gemma4_runnable_on_card"])
			core.AssertEqual(t, rocmGemma4MTPAssistantPath(tc.wantSize, "bf16"), plan.Labels["attached_drafter_assistant_production_quant_model"])
			core.AssertEqual(t, rocmGemma4MTPAssistantPath(tc.wantSize, "bf16"), plan.Labels["attached_drafter_assistant_production_quant_assistant_model"])
			core.AssertEqual(t, tc.wantSize+":assistant-bf16", plan.Labels["attached_drafter_assistant_production_quant_pack"])
			core.AssertEqual(t, "mtp-assistant", plan.Labels["attached_drafter_assistant_production_quant_tier"])
			core.AssertEqual(t, "true", plan.Labels["attached_drafter_assistant_production_quant_mtp_assistant"])
			core.AssertEqual(t, "gemma4", plan.Labels["attached_drafter_assistant_production_quant_target_family"])
			core.AssertEqual(t, tc.wantOfficialPair, plan.Labels["attached_drafter_official_pair_verified"])
			core.AssertEqual(t, tc.wantFamilyPair, plan.Labels["attached_drafter_gemma4_family_pair_verified"])
		})
	}
}

func TestProductionMTPAttachedDrafterPlan_Good_AcceptsGGUFTargetPath(t *testing.T) {
	plan, err := PlanAttachedDrafter(
		&rocmModel{
			modelPath: "/models/lmstudio-community/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q6_K.gguf",
			modelInfo: inference.ModelInfo{
				Architecture: "gemma4_text",
			},
		},
		&rocmModel{
			modelPath: rocmGemma4MTPAssistantPath("E2B", "bf16"),
			modelInfo: inference.ModelInfo{
				Architecture: officialGemma4E2BAssistantArchitecture,
			},
		},
	)

	core.RequireNoError(t, err)
	core.AssertEqual(t, 6, plan.Target.QuantBits)
	core.AssertEqual(t, 64, plan.Target.QuantGroup)
	core.AssertEqual(t, "E2B", plan.Labels["attached_drafter_target_gemma4_size"])
	core.AssertEqual(t, "q6", plan.Labels["attached_drafter_target_gemma4_quant_mode"])
	core.AssertEqual(t, Gemma4RuntimeGGUF, plan.Labels["attached_drafter_target_gemma4_runtime"])
	core.AssertEqual(t, Gemma4GenerateLinked, plan.Labels["attached_drafter_target_gemma4_generate_status"])
	core.AssertEqual(t, "true", plan.Labels["attached_drafter_target_gemma4_pack_supported"])
	core.AssertEqual(t, "true", plan.Labels["attached_drafter_target_gemma4_runnable_on_card"])
	core.AssertEqual(t, "E2B", plan.Labels["attached_drafter_assistant_gemma4_size"])
	core.AssertEqual(t, "bf16", plan.Labels["attached_drafter_assistant_gemma4_quant_mode"])
	core.AssertEqual(t, Gemma4GenerateLoadOnly, plan.Labels["attached_drafter_assistant_gemma4_generate_status"])
	core.AssertEqual(t, "true", plan.Labels["attached_drafter_gemma4_family_pair_verified"])
}

func TestProductionMTPAttachedDrafterEvidence_Good_PreservesNonOfficialGemma4PairLabels(t *testing.T) {
	plan, err := PlanAttachedDrafter(
		productionMTP12BQ6TargetModel(),
		productionMTP12BBF16AssistantModel(),
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, "false", plan.Labels["attached_drafter_official_pair_verified"])
	evidence := productionMTPPassingEvidence()
	clearProductionMTPAttachedDrafterStaticEvidence(&evidence)

	err = ApplyProductionMTPAttachedDrafterPlanEvidence(&evidence, plan)
	markProductionMTPNativeHandoffEvidenceLinked(&evidence)
	decision := EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)

	core.RequireNoError(t, err)
	core.AssertEqual(t, "12B", evidence.TargetGemma4Size)
	core.AssertEqual(t, "q6", evidence.TargetGemma4QuantMode)
	core.AssertEqual(t, 64, evidence.TargetGemma4QuantGroup)
	core.AssertEqual(t, Gemma4RuntimeMLXAffine, evidence.TargetGemma4Runtime)
	core.AssertEqual(t, Gemma4GenerateLinked, evidence.TargetGemma4GenerateStatus)
	core.AssertEqual(t, "mlx-community/gemma-4-12b-it-6bit", evidence.TargetProductionQuantModelID)
	core.AssertEqual(t, "", evidence.TargetProductionQuantLockedModelID)
	core.AssertEqual(t, "12B", evidence.AssistantGemma4Size)
	core.AssertEqual(t, "bf16", evidence.AssistantGemma4QuantMode)
	core.AssertEqual(t, rocmGemma4MTPAssistantPath("12B", "bf16"), evidence.AssistantProductionQuantModelID)
	core.AssertEqual(t, "12B:assistant-bf16", evidence.AssistantProductionQuantPack)
	core.AssertEqual(t, "mtp-assistant", evidence.AssistantProductionQuantTier)
	core.AssertEqual(t, true, evidence.AssistantProductionQuantMTPAssistant)
	core.AssertEqual(t, "gemma4", evidence.AssistantProductionQuantTargetFamily)
	core.AssertEqual(t, true, evidence.Gemma4FamilyPairVerified)
	core.AssertEqual(t, false, evidence.OfficialPairVerified)
	core.AssertEqual(t, true, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "MTP retained workflow")
}

func TestProductionMTPAttachedDrafterEvidence_Good_AllowsMTPQATAssistantPack(t *testing.T) {
	plan, err := PlanAttachedDrafter(
		productionMTP12BQ6QATTargetModel(),
		productionMTP12BQ6QATAssistantModel(),
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, "false", plan.Labels["attached_drafter_official_pair_verified"])
	evidence := productionMTPPassingEvidence()
	clearProductionMTPAttachedDrafterStaticEvidence(&evidence)

	err = ApplyProductionMTPAttachedDrafterPlanEvidence(&evidence, plan)
	markProductionMTPNativeHandoffEvidenceLinked(&evidence)
	decision := EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)

	core.RequireNoError(t, err)
	core.AssertEqual(t, "12B", evidence.TargetGemma4Size)
	core.AssertEqual(t, "q6", evidence.TargetGemma4QuantMode)
	core.AssertEqual(t, "mlx-community/gemma-4-12B-it-qat-6bit", evidence.TargetProductionQuantModelID)
	core.AssertEqual(t, "12B", evidence.AssistantGemma4Size)
	core.AssertEqual(t, "q6", evidence.AssistantGemma4QuantMode)
	core.AssertEqual(t, 64, evidence.AssistantGemma4QuantGroup)
	core.AssertEqual(t, Gemma4RuntimeMLXAffine, evidence.AssistantGemma4Runtime)
	core.AssertEqual(t, Gemma4GenerateLoadOnly, evidence.AssistantGemma4GenerateStatus)
	core.AssertEqual(t, rocmGemma4MTPAssistantPath("12B", "q6"), evidence.SpeculativeDraftModelPath)
	core.AssertEqual(t, rocmGemma4MTPAssistantPath("12B", "q6"), evidence.AssistantProductionQuantModelID)
	core.AssertEqual(t, "12B:assistant-q6", evidence.AssistantProductionQuantPack)
	core.AssertEqual(t, "mtp-assistant", evidence.AssistantProductionQuantTier)
	core.AssertEqual(t, true, evidence.AssistantProductionQuantMTPAssistant)
	core.AssertEqual(t, "gemma4", evidence.AssistantProductionQuantTargetFamily)
	core.AssertEqual(t, true, evidence.Gemma4FamilyPairVerified)
	core.AssertEqual(t, false, evidence.OfficialPairVerified)
	core.AssertEqual(t, true, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "MTP retained workflow")
}

func TestProductionMTPAttachedDrafterEvidence_Good_PreservesSameSizeAssistantLabels(t *testing.T) {
	plan, err := PlanAttachedDrafter(
		productionMTPE4BQ8TargetModel(),
		productionMTPE4BBF16AssistantModel(),
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, "false", plan.Labels["attached_drafter_official_pair_verified"])
	evidence := productionMTPPassingEvidence()
	clearProductionMTPAttachedDrafterStaticEvidence(&evidence)

	err = ApplyProductionMTPAttachedDrafterPlanEvidence(&evidence, plan)
	markProductionMTPNativeHandoffEvidenceLinked(&evidence)
	decision := EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)

	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmGemma4MTPAssistantPath("E4B", "bf16"), evidence.SpeculativeDraftModelPath)
	core.AssertEqual(t, "E4B", evidence.TargetGemma4Size)
	core.AssertEqual(t, "q8", evidence.TargetGemma4QuantMode)
	core.AssertEqual(t, 64, evidence.TargetGemma4QuantGroup)
	core.AssertEqual(t, Gemma4RuntimeMLXAffine, evidence.TargetGemma4Runtime)
	core.AssertEqual(t, Gemma4GenerateLinked, evidence.TargetGemma4GenerateStatus)
	core.AssertEqual(t, "lmstudio-community/gemma-4-E4B-it-MLX-8bit", evidence.TargetProductionQuantModelID)
	core.AssertEqual(t, "", evidence.TargetProductionQuantLockedModelID)
	core.AssertEqual(t, "E4B", evidence.AssistantGemma4Size)
	core.AssertEqual(t, "bf16", evidence.AssistantGemma4QuantMode)
	core.AssertEqual(t, Gemma4RuntimeBF16, evidence.AssistantGemma4Runtime)
	core.AssertEqual(t, Gemma4GenerateLoadOnly, evidence.AssistantGemma4GenerateStatus)
	core.AssertEqual(t, true, evidence.Gemma4FamilyPairVerified)
	core.AssertEqual(t, false, evidence.OfficialPairVerified)
	core.AssertEqual(t, "", evidence.OfficialTargetModelID)
	core.AssertEqual(t, "", evidence.OfficialAssistantModelID)
	core.AssertEqual(t, true, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "MTP retained workflow")
}

func TestProductionMTPAttachedDrafterEvidence_Good_AllowsHigherPrecisionGGUFAssistant(t *testing.T) {
	plan, err := PlanAttachedDrafter(
		productionMTPE2BQ4GGUFTargetModel(),
		productionMTPE2BQ8GGUFAssistantModel(),
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, "false", plan.Labels["attached_drafter_official_pair_verified"])
	evidence := productionMTPPassingEvidence()
	clearProductionMTPAttachedDrafterStaticEvidence(&evidence)

	err = ApplyProductionMTPAttachedDrafterPlanEvidence(&evidence, plan)

	core.RequireNoError(t, err)
	core.AssertEqual(t, "E2B", evidence.TargetGemma4Size)
	core.AssertEqual(t, "q4", evidence.TargetGemma4QuantMode)
	core.AssertEqual(t, 64, evidence.TargetGemma4QuantGroup)
	core.AssertEqual(t, Gemma4RuntimeGGUF, evidence.TargetGemma4Runtime)
	core.AssertEqual(t, Gemma4GenerateLinked, evidence.TargetGemma4GenerateStatus)
	core.AssertEqual(t, "E2B", evidence.AssistantGemma4Size)
	core.AssertEqual(t, "q8", evidence.AssistantGemma4QuantMode)
	core.AssertEqual(t, 64, evidence.AssistantGemma4QuantGroup)
	core.AssertEqual(t, Gemma4RuntimeGGUF, evidence.AssistantGemma4Runtime)
	core.AssertEqual(t, Gemma4GenerateLoadOnly, evidence.AssistantGemma4GenerateStatus)
	core.AssertEqual(t, "/models/unsloth-gemma-4-E2B-it-GGUF/MTP/gemma-4-E2B-it-Q8_0-MTP.gguf", evidence.SpeculativeDraftModelPath)
	core.AssertEqual(t, "/models/unsloth-gemma-4-E2B-it-GGUF/MTP/gemma-4-E2B-it-Q8_0-MTP.gguf", evidence.AssistantProductionQuantModelID)
	core.AssertEqual(t, "E2B:assistant-q8", evidence.AssistantProductionQuantPack)
	core.AssertEqual(t, true, evidence.Gemma4FamilyPairVerified)
	core.AssertEqual(t, false, evidence.OfficialPairVerified)
}

func TestProductionMTPAttachedDrafterEvidence_Good_CompletesMeasuredEvidence(t *testing.T) {
	plan, err := PlanAttachedDrafter(
		productionMTPE2BQ6TargetModel(),
		productionMTPE2BBF16AssistantModel(),
	)
	core.RequireNoError(t, err)
	evidence := productionMTPPassingEvidence()
	clearProductionMTPAttachedDrafterStaticEvidence(&evidence)

	err = ApplyProductionMTPAttachedDrafterPlanEvidence(&evidence, plan)
	markProductionMTPNativeHandoffEvidenceLinked(&evidence)
	decision := EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)

	core.RequireNoError(t, err)
	if !decision.EnableByDefault {
		t.Fatalf("decision = %+v, want static plan evidence plus measured counters to pass", decision)
	}
}

func TestProductionMTPAttachedDrafterEvidence_Good_FillsRetainedRouteFromCapabilityLabels(t *testing.T) {
	evidence := ProductionMTPPromotionEvidence{}
	labels := map[string]string{}
	rocmAddGemma4AttachedDrafterCapabilityLabels(labels)

	err := ApplyProductionMTPAttachedDrafterLabelEvidence(&evidence, labels)

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, evidence.AttachedDrafterRetainedStateEntrypoint)
	core.AssertEqual(t, true, evidence.AttachedDrafterRetainedStateRequired)
	core.AssertEqual(t, "rocm_state_session_runtime_kv", evidence.AttachedDrafterStateSource)
	core.AssertEqual(t, "forbidden", evidence.AttachedDrafterPromptReplayFallback)
	core.AssertEqual(t, hipKernelStatusNotLinked, evidence.AttachedDrafterNativeAttachment)
	core.AssertEqual(t, attachedDrafterNativeHandoffTargetDecodeOnly, evidence.AttachedDrafterNativeHandoff)
	core.AssertEqual(t, hipKernelStatusLinked, evidence.AttachedDrafterTargetRetainedDecode)
	core.AssertEqual(t, hipKernelStatusLinked, evidence.AttachedDrafterTargetRetainedState)
	core.AssertEqual(t, hipKernelStatusNotLinked, evidence.AttachedDrafterAssistantVerify)
	core.AssertEqual(t, hipKernelStatusNotLinked, evidence.AttachedDrafterAssistantStateVerify)
	core.AssertEqual(t, "E2B", evidence.TargetGemma4Size)
	core.AssertEqual(t, "q6", evidence.TargetGemma4QuantMode)
	core.AssertEqual(t, 64, evidence.TargetGemma4QuantGroup)
	core.AssertEqual(t, Gemma4RuntimeMLXAffine, evidence.TargetGemma4Runtime)
	core.AssertEqual(t, Gemma4GenerateLinked, evidence.TargetGemma4GenerateStatus)
	core.AssertEqual(t, ProductionLaneCurrentModelID, evidence.TargetProductionQuantModelID)
	core.AssertEqual(t, ProductionLaneModelID, evidence.TargetProductionQuantLockedModelID)
	core.AssertEqual(t, "E2B", evidence.AssistantGemma4Size)
	core.AssertEqual(t, "bf16", evidence.AssistantGemma4QuantMode)
	core.AssertEqual(t, Gemma4RuntimeBF16, evidence.AssistantGemma4Runtime)
	core.AssertEqual(t, Gemma4GenerateLoadOnly, evidence.AssistantGemma4GenerateStatus)
	core.AssertEqual(t, officialGemma4E2BAssistantModelID, evidence.AssistantProductionQuantModelID)
	core.AssertEqual(t, "E2B:assistant-bf16", evidence.AssistantProductionQuantPack)
	core.AssertEqual(t, "mtp-assistant", evidence.AssistantProductionQuantTier)
	core.AssertEqual(t, true, evidence.AssistantProductionQuantMTPAssistant)
	core.AssertEqual(t, "gemma4", evidence.AssistantProductionQuantTargetFamily)
	core.AssertEqual(t, rocmGemma4MTPAssistantPath("E2B", "bf16"), evidence.SpeculativeDraftModelPath)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, evidence.SpeculativeDraftTokens)
	core.AssertEqual(t, officialGemma4E2BAssistantArchitecture, evidence.AssistantArchitecture)
	core.AssertEqual(t, true, evidence.AssistantOrderedEmbeddings)
	core.AssertEqual(t, ProductionMTPAssistantOrderedEmbeddingCentroids, evidence.AssistantCentroids)
	core.AssertEqual(t, ProductionMTPAssistantCentroidIntermediateTopK, evidence.AssistantCentroidIntermediateTopK)
	core.AssertEqual(t, true, evidence.AssistantFourLayerDrafter)
	core.AssertEqual(t, "int64", evidence.AssistantTokenOrderingDType)
	if !intSliceEqual(evidence.AssistantTokenOrderingShape, []int{ProductionMTPAssistantOrderedEmbeddingCentroids, ProductionMTPAssistantTokenOrderingVocabSize / ProductionMTPAssistantOrderedEmbeddingCentroids}) {
		t.Fatalf("AssistantTokenOrderingShape = %v, want ordered centroid shape", evidence.AssistantTokenOrderingShape)
	}
	core.AssertEqual(t, true, evidence.Gemma4FamilyPairVerified)
	core.AssertEqual(t, true, evidence.OfficialPairVerified)
	core.AssertEqual(t, officialGemma4E2BTargetModelID, evidence.OfficialTargetModelID)
	core.AssertEqual(t, officialGemma4E2BTargetRevision, evidence.OfficialTargetRevision)
	core.AssertEqual(t, officialGemma4E2BAssistantModelID, evidence.OfficialAssistantModelID)
	core.AssertEqual(t, officialGemma4E2BAssistantRevision, evidence.OfficialAssistantRevision)
}

func TestProductionMTPAttachedDrafterEvidence_Good_AcceptsRouteCapabilityLabels(t *testing.T) {
	evidence := ProductionMTPPromotionEvidence{}
	labels := rocmGemma4Q4SpeculativeDecodeCapabilityLabels(productionMTPE2BQ6TargetModel().modelIdentity())
	core.AssertEqual(t, ROCmAttachedDrafterRegistryContract, labels["engine_attached_drafter_route_contract"])
	core.AssertEqual(t, "forbidden", labels["engine_attached_drafter_prompt_replay_fallback"])
	core.AssertEqual(t, "rocm_state_session_runtime_kv", labels["engine_attached_drafter_state_source"])

	err := ApplyProductionMTPAttachedDrafterLabelEvidence(&evidence, labels)

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, evidence.AttachedDrafterRetainedStateEntrypoint)
	core.AssertEqual(t, true, evidence.AttachedDrafterRetainedStateRequired)
	core.AssertEqual(t, "rocm_state_session_runtime_kv", evidence.AttachedDrafterStateSource)
	core.AssertEqual(t, "forbidden", evidence.AttachedDrafterPromptReplayFallback)
	core.AssertEqual(t, Gemma4RuntimeMLXAffine, evidence.TargetGemma4Runtime)
	core.AssertEqual(t, Gemma4GenerateLinked, evidence.TargetGemma4GenerateStatus)
	core.AssertEqual(t, Gemma4RuntimeBF16, evidence.AssistantGemma4Runtime)
	core.AssertEqual(t, Gemma4GenerateLoadOnly, evidence.AssistantGemma4GenerateStatus)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, evidence.SpeculativeDraftTokens)
	core.AssertEqual(t, officialGemma4E2BAssistantArchitecture, evidence.AssistantArchitecture)
	core.AssertEqual(t, true, evidence.AssistantOrderedEmbeddings)
	core.AssertEqual(t, ProductionMTPAssistantOrderedEmbeddingCentroids, evidence.AssistantCentroids)
	core.AssertEqual(t, ProductionMTPAssistantCentroidIntermediateTopK, evidence.AssistantCentroidIntermediateTopK)
	core.AssertEqual(t, true, evidence.AssistantFourLayerDrafter)
	core.AssertEqual(t, "int64", evidence.AssistantTokenOrderingDType)
	if !intSliceEqual(evidence.AssistantTokenOrderingShape, []int{ProductionMTPAssistantOrderedEmbeddingCentroids, ProductionMTPAssistantTokenOrderingVocabSize / ProductionMTPAssistantOrderedEmbeddingCentroids}) {
		t.Fatalf("AssistantTokenOrderingShape = %v, want ordered centroid shape", evidence.AssistantTokenOrderingShape)
	}
	core.AssertEqual(t, true, evidence.Gemma4FamilyPairVerified)
	core.AssertEqual(t, true, evidence.OfficialPairVerified)
}

func TestProductionMTPAttachedDrafterEvidence_Good_FillsRetainedRouteFromBenchmarkLabels(t *testing.T) {
	evidence := ProductionMTPPromotionEvidence{}
	labels := map[string]string{}
	rocmAddGemma4AttachedDrafterBenchmarkLabels(labels)
	core.AssertEqual(t, "true", labels["attached.drafter.target.gemma4_pack_supported"])
	core.AssertEqual(t, "true", labels["attached.drafter.target.gemma4_runnable_on_card"])
	core.AssertEqual(t, ProductionLaneCurrentModelID, labels["attached.drafter.target.production_quant_model"])
	core.AssertEqual(t, ProductionLaneModelID, labels["attached.drafter.target.production_quant_locked_model"])
	core.AssertEqual(t, "true", labels["attached.drafter.assistant.gemma4_pack_supported"])
	core.AssertEqual(t, "true", labels["attached.drafter.assistant.gemma4_runnable_on_card"])

	err := ApplyProductionMTPAttachedDrafterLabelEvidence(&evidence, labels)

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, evidence.AttachedDrafterRetainedStateEntrypoint)
	core.AssertEqual(t, true, evidence.AttachedDrafterRetainedStateRequired)
	core.AssertEqual(t, "rocm_state_session_runtime_kv", evidence.AttachedDrafterStateSource)
	core.AssertEqual(t, "forbidden", evidence.AttachedDrafterPromptReplayFallback)
	core.AssertEqual(t, "E2B", evidence.TargetGemma4Size)
	core.AssertEqual(t, "q6", evidence.TargetGemma4QuantMode)
	core.AssertEqual(t, 64, evidence.TargetGemma4QuantGroup)
	core.AssertEqual(t, Gemma4RuntimeMLXAffine, evidence.TargetGemma4Runtime)
	core.AssertEqual(t, Gemma4GenerateLinked, evidence.TargetGemma4GenerateStatus)
	core.AssertEqual(t, ProductionLaneCurrentModelID, evidence.TargetProductionQuantModelID)
	core.AssertEqual(t, ProductionLaneModelID, evidence.TargetProductionQuantLockedModelID)
	core.AssertEqual(t, "E2B", evidence.AssistantGemma4Size)
	core.AssertEqual(t, "bf16", evidence.AssistantGemma4QuantMode)
	core.AssertEqual(t, Gemma4RuntimeBF16, evidence.AssistantGemma4Runtime)
	core.AssertEqual(t, Gemma4GenerateLoadOnly, evidence.AssistantGemma4GenerateStatus)
	core.AssertEqual(t, rocmGemma4MTPAssistantPath("E2B", "bf16"), evidence.SpeculativeDraftModelPath)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, evidence.SpeculativeDraftTokens)
	core.AssertEqual(t, officialGemma4E2BAssistantArchitecture, evidence.AssistantArchitecture)
	core.AssertEqual(t, true, evidence.AssistantOrderedEmbeddings)
	core.AssertEqual(t, true, evidence.AssistantFourLayerDrafter)
	core.AssertEqual(t, true, evidence.Gemma4FamilyPairVerified)
	core.AssertEqual(t, true, evidence.OfficialPairVerified)
	core.AssertEqual(t, officialGemma4E2BTargetModelID, evidence.OfficialTargetModelID)
	core.AssertEqual(t, officialGemma4E2BAssistantModelID, evidence.OfficialAssistantModelID)
}

func TestProductionMTPLabelEvidence_Good_MeasuredLabelsPromote(t *testing.T) {
	var evidence ProductionMTPPromotionEvidence
	labels := productionMTPPassingLabels()

	err := ApplyProductionMTPLabelEvidence(&evidence, labels)
	decision := EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)

	core.RequireNoError(t, err)
	if !decision.EnableByDefault {
		t.Fatalf("decision = %+v evidence=%+v, want MTP labels to produce passing evidence", decision, evidence)
	}
	core.AssertEqual(t, "forbidden", evidence.AttachedDrafterPromptReplayFallback)
	core.AssertEqual(t, "E2B", evidence.TargetGemma4Size)
	core.AssertEqual(t, "q6", evidence.TargetGemma4QuantMode)
	core.AssertEqual(t, 64, evidence.TargetGemma4QuantGroup)
	core.AssertEqual(t, "E2B", evidence.AssistantGemma4Size)
	core.AssertEqual(t, "bf16", evidence.AssistantGemma4QuantMode)
	core.AssertEqual(t, true, evidence.Gemma4FamilyPairVerified)
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, evidence.MTPCacheMode)
	core.AssertEqual(t, 0.75, decision.AcceptanceRate)
	if !intSliceEqual(evidence.MTPObservedDraftTokenSweeps, []int{1, 2, 4}) {
		t.Fatalf("MTPObservedDraftTokenSweeps = %v, want 1/2/4", evidence.MTPObservedDraftTokenSweeps)
	}
}

func TestProductionMTPPromotionMetricLabels_Good_EvaluatesPassingLabels(t *testing.T) {
	decision, err := EvaluateProductionMTPPromotionMetricLabels(productionMTPPassingLabels())

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, decision.EnableByDefault)
	core.AssertEqual(t, 0.75, decision.AcceptanceRate)
	core.AssertContains(t, decision.Reason, "MTP retained workflow")
}

func TestProductionMTPPromotionMetricLabels_Good_EvaluatesFamilyPairWithoutOfficialIDs(t *testing.T) {
	plan, err := PlanAttachedDrafter(
		productionMTPE4BQ8TargetModel(),
		productionMTPE4BBF16AssistantModel(),
	)
	core.RequireNoError(t, err)
	labels := productionMTPPassingLabels()
	for _, key := range []string{
		"speculative_draft_model_path",
		"official_pair_verified",
		"official_target_model_id",
		"official_target_revision",
		"official_assistant_model_id",
		"official_assistant_revision",
		"attached_drafter_official_pair_verified",
		"attached_drafter_official_target_model_id",
		"attached_drafter_official_target_revision",
		"attached_drafter_official_assistant_model_id",
		"attached_drafter_official_assistant_revision",
		"attached.drafter.official_pair_verified",
		"attached.drafter.official_target_model_id",
		"attached.drafter.official_target_revision",
		"attached.drafter.official_assistant_model_id",
		"attached.drafter.official_assistant_revision",
	} {
		delete(labels, key)
	}
	for key, value := range plan.Labels {
		labels[key] = value
	}
	markProductionMTPNativeHandoffLabelsLinked(labels)

	decision, err := EvaluateProductionMTPPromotionMetricLabels(labels)

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, decision.EnableByDefault)
	core.AssertEqual(t, 0.75, decision.AcceptanceRate)
	core.AssertEqual(t, "google/gemma-4-E4B-it-assistant", labels["attached_drafter_assistant_model_id"])
	core.AssertEqual(t, "google/gemma-4-E4B-it-assistant", labels["attached_drafter_assistant_production_quant_model"])
	core.AssertEqual(t, "E4B:assistant-bf16", labels["attached_drafter_assistant_production_quant_pack"])
	core.AssertEqual(t, "mtp-assistant", labels["attached_drafter_assistant_production_quant_tier"])
	core.AssertEqual(t, "true", labels["attached_drafter_assistant_production_quant_mtp_assistant"])
	core.AssertEqual(t, "false", labels["attached_drafter_official_pair_verified"])
	core.AssertEqual(t, "true", labels["attached_drafter_gemma4_family_pair_verified"])
}

func TestProductionMTPPromotionMetricLabels_Good_EvaluatesValidNonPromotingLabels(t *testing.T) {
	labels := productionMTPPassingLabels()
	labels["mtp_visible_tokens_per_sec"] = "99"

	decision, err := EvaluateProductionMTPPromotionMetricLabels(labels)

	core.RequireNoError(t, err)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "below the ROCm production minimum")
}

func TestProductionMTPPromotionMetricLabels_Bad_RejectsMismatchedAssistantProductionPack(t *testing.T) {
	labels := productionMTPPassingLabels()
	labels["attached_drafter_assistant_production_quant_pack"] = "E4B:assistant-bf16"

	decision, err := EvaluateProductionMTPPromotionMetricLabels(labels)

	core.RequireNoError(t, err)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "assistant production pack")
}

func TestProductionMTPPromotionMetricLabels_Bad_RejectsMissingRequiredMetric(t *testing.T) {
	labels := productionMTPPassingLabels()
	delete(labels, "mtp_target_tokens_per_sec")

	err := ValidateProductionMTPPromotionMetricLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "mtp_target_tokens_per_sec")
}

func TestProductionMTPPromotionMetricLabels_Bad_RejectsMalformedMetric(t *testing.T) {
	labels := productionMTPPassingLabels()
	labels["mtp_proposed_tokens"] = "forty"

	_, err := EvaluateProductionMTPPromotionMetricLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "mtp_proposed_tokens")
}

func TestProductionMTPDecodeRunEvidence_Good_MeasuredResultPromotesWithStaticPlan(t *testing.T) {
	plan, err := PlanAttachedDrafter(
		productionMTPE2BQ6TargetModel(),
		productionMTPE2BBF16AssistantModel(),
	)
	core.RequireNoError(t, err)
	var evidence ProductionMTPPromotionEvidence
	core.RequireNoError(t, ApplyProductionMTPAttachedDrafterPlanEvidence(&evidence, plan))
	result := inferdecode.Result{
		Mode:   inferdecode.ModeSpeculative,
		Prompt: "new turn only",
		Metrics: inferdecode.Metrics{
			TargetTokens:   880,
			DraftTokens:    40,
			AcceptedTokens: 30,
			RejectedTokens: 10,
			EmittedTokens:  1000,
			TargetCalls:    20,
			DraftCalls:     20,
			Duration:       8 * time.Second,
			TargetDuration: 8 * time.Second,
			DraftDuration:  500 * time.Millisecond,
		},
	}

	err = ApplyProductionMTPDecodeRunEvidence(&evidence, result, productionMTPPassingDecodeRunEvidence())
	decision := EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)

	core.RequireNoError(t, err)
	if !decision.EnableByDefault {
		t.Fatalf("decision = %+v evidence=%+v, want decode-result evidence plus static plan to promote", decision, evidence)
	}
	core.AssertEqual(t, 40, evidence.MTPProposedTokens)
	core.AssertEqual(t, 30, evidence.MTPAcceptedTokens)
	core.AssertEqual(t, 10, evidence.MTPRejectedTokens)
	core.AssertEqual(t, float64(125), evidence.MTPVisibleTokensPerSec)
	core.AssertEqual(t, float64(110), evidence.MTPTargetTokensPerSec)
	core.AssertEqual(t, "forbidden", evidence.AttachedDrafterPromptReplayFallback)
}

func TestProductionMTPAttachedDrafterEvidence_Bad_InvalidStaticLabel(t *testing.T) {
	evidence := ProductionMTPPromotionEvidence{}
	labels := map[string]string{}
	rocmAddGemma4AttachedDrafterCapabilityLabels(labels)
	labels["attached_drafter_speculative_draft_tokens"] = "two"

	err := ApplyProductionMTPAttachedDrafterLabelEvidence(&evidence, labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "attached_drafter_speculative_draft_tokens")

	evidence = ProductionMTPPromotionEvidence{}
	labels = map[string]string{}
	rocmAddGemma4AttachedDrafterCapabilityLabels(labels)
	labels["attached_drafter_target_gemma4_quant_group"] = "sixty-four"

	err = ApplyProductionMTPAttachedDrafterLabelEvidence(&evidence, labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "attached_drafter_target_gemma4_quant_group")
}

func TestProductionMTPLabelEvidence_Bad_InvalidMeasuredValue(t *testing.T) {
	var evidence ProductionMTPPromotionEvidence
	labels := productionMTPPassingLabels()
	labels["mtp_proposed_tokens"] = "forty"

	err := ApplyProductionMTPLabelEvidence(&evidence, labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "mtp_proposed_tokens")
}

func TestProductionMTPDecodeRunEvidence_Bad_RejectsInconsistentCounters(t *testing.T) {
	var evidence ProductionMTPPromotionEvidence
	result := inferdecode.Result{
		Mode: inferdecode.ModeSpeculative,
		Metrics: inferdecode.Metrics{
			DraftTokens:    40,
			AcceptedTokens: 30,
			RejectedTokens: 9,
		},
	}

	err := ApplyProductionMTPDecodeRunEvidence(&evidence, result, ProductionMTPDecodeRunEvidence{})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "accepted/rejected")
}

func TestProductionMTPDecodeRunEvidence_Bad_RejectsNonMTPResultMode(t *testing.T) {
	var evidence ProductionMTPPromotionEvidence

	err := ApplyProductionMTPDecodeRunEvidence(&evidence, inferdecode.Result{Mode: inferdecode.ModePromptLookup}, ProductionMTPDecodeRunEvidence{})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "speculative MTP")
}

func TestProductionMTPAttachedDrafterEvidence_Bad_RetainedRouteLabelsDoNotHidePromptReplay(t *testing.T) {
	evidence := productionMTPPassingEvidence()
	labels := map[string]string{}
	rocmAddGemma4AttachedDrafterCapabilityLabels(labels)
	labels["attached_drafter_prompt_replay_fallback"] = "allowed"
	clearProductionMTPAttachedDrafterRouteEvidence(&evidence)

	err := ApplyProductionMTPAttachedDrafterLabelEvidence(&evidence, labels)
	decision := EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)

	core.RequireNoError(t, err)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "retained attached-drafter route")
}

func TestProductionMTPAttachedDrafterEvidence_Bad_RejectsInvalidPlan(t *testing.T) {
	plan := AttachedDrafterPlan{
		Mode:             "mtp_attached_drafter",
		Target:           inference.ModelInfo{Architecture: "gemma4_text"},
		Draft:            inference.ModelInfo{Architecture: "qwen3"},
		DraftTokens:      ProductionMTPDefaultDraftTokens,
		HelperStatus:     hipKernelStatusLinked,
		NativeAttachment: hipKernelStatusNotLinked,
	}
	err := ApplyProductionMTPAttachedDrafterPlanEvidence(&ProductionMTPPromotionEvidence{}, plan)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "draft model")

	plan.Draft = inference.ModelInfo{Architecture: "gemma4_assistant"}
	plan.NativeAttachment = hipKernelStatusLinked
	err = ApplyProductionMTPAttachedDrafterPlanEvidence(&ProductionMTPPromotionEvidence{}, plan)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "not_linked")

	err = ApplyProductionMTPAttachedDrafterPlanEvidence(nil, plan)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "evidence")
}

func TestProductionMTPAttachedDrafterEvidence_Bad_RejectsPlanWithoutGemma4SupportLabels(t *testing.T) {
	plan := AttachedDrafterPlan{
		Mode: "mtp_attached_drafter",
		Target: inference.ModelInfo{
			Architecture: "gemma4_text",
		},
		Draft: inference.ModelInfo{
			Architecture: officialGemma4E2BAssistantArchitecture,
		},
		DraftTokens:      ProductionMTPDefaultDraftTokens,
		HelperStatus:     hipKernelStatusLinked,
		NativeAttachment: hipKernelStatusNotLinked,
	}

	err := ApplyProductionMTPAttachedDrafterPlanEvidence(&ProductionMTPPromotionEvidence{}, plan)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "target Gemma4 pack identity is incomplete")
}

func TestProductionMTPAttachedDrafterEvidence_Bad_RejectsPlanWithLoadOnlyTarget(t *testing.T) {
	plan, err := PlanAttachedDrafter(productionMTPE2BQ6TargetModel(), productionMTPE2BBF16AssistantModel())
	core.RequireNoError(t, err)
	plan.Target.QuantBits = 16
	plan.Target.QuantGroup = 0
	plan.Labels["attached_drafter_target_gemma4_quant_mode"] = "bf16"
	plan.Labels["attached_drafter_target_gemma4_runtime"] = Gemma4RuntimeBF16
	plan.Labels["attached_drafter_target_gemma4_generate_status"] = Gemma4GenerateLoadOnly

	err = ApplyProductionMTPAttachedDrafterPlanEvidence(&ProductionMTPPromotionEvidence{}, plan)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "target Gemma4 pack is not linked for generation")
}

func TestProductionMTPPromotion_Bad_RejectsMissingEvidence(t *testing.T) {
	evidence := productionMTPPassingEvidence()
	evidence.RetainedWorkflow = false
	decision := EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "retained workflow")

	evidence = productionMTPPassingEvidence()
	evidence.MTPAcceptedTokens = 0
	evidence.MTPRejectedTokens = evidence.MTPProposedTokens
	decision = EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "accepted draft tokens")

	evidence = productionMTPPassingEvidence()
	evidence.MTPObservedDraftTokenSweeps = []int{2}
	decision = EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "draft-token sweep")

	evidence = productionMTPPassingEvidence()
	evidence.AttachedDrafterPromptReplayFallback = "allowed"
	decision = EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "retained attached-drafter route")

	evidence = productionMTPPassingEvidence()
	evidence.AttachedDrafterNativeAttachment = hipKernelStatusNotLinked
	evidence.AttachedDrafterNativeHandoff = attachedDrafterNativeHandoffTargetDecodeOnly
	decision = EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "native attached-drafter handoff")

	evidence = productionMTPPassingEvidence()
	evidence.AttachedDrafterAssistantStateVerify = hipKernelStatusNotLinked
	decision = EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "assistant verifier")

	evidence = productionMTPPassingEvidence()
	evidence.AssistantTokenOrderingShape = []int{2048, 64}
	decision = EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "token-ordering")

	evidence = productionMTPPassingEvidence()
	evidence.Gemma4FamilyPairVerified = false
	decision = EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "Gemma 4 family")

	evidence = productionMTPPassingEvidence()
	evidence.AssistantGemma4Size = "E4B"
	evidence.Gemma4FamilyPairVerified = false
	decision = EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "Gemma 4 family")

	evidence = productionMTPPassingEvidence()
	evidence.AssistantProductionQuantPack = "E4B:assistant-bf16"
	decision = EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "assistant production pack")

	evidence = productionMTPPassingEvidence()
	evidence.AssistantProductionQuantMTPAssistant = false
	decision = EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "assistant production pack")
}

func TestProductionMTPPromotion_Bad_RejectsSlowerOrSub100(t *testing.T) {
	evidence := productionMTPPassingEvidence()
	evidence.MTPWallDuration = 12 * time.Second
	decision := EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "faster")

	evidence = productionMTPPassingEvidence()
	evidence.MTPVisibleTokensPerSec = 99
	decision = EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), evidence)
	core.AssertEqual(t, false, decision.EnableByDefault)
	core.AssertContains(t, decision.Reason, "production minimum")
}

func TestOfficialGemma4E2BLocks_Good(t *testing.T) {
	target := OfficialGemma4E2BTargetLock()
	assistant := OfficialGemma4E2BAssistantLock()

	core.AssertEqual(t, OfficialGemma4E2BRoleTarget, target.Role)
	core.AssertEqual(t, officialGemma4E2BTargetModelID, target.ModelID)
	core.AssertEqual(t, OfficialGemma4E2BRoleAssistant, assistant.Role)
	core.AssertEqual(t, officialGemma4E2BAssistantArchitecture, assistant.ModelType)
	core.AssertEqual(t, officialGemma4E2BSourceCheckedAt, assistant.SourceCheckedAt)
	if assistant.ConfigSHA256 == "" || target.ConfigSHA256 == "" {
		t.Fatalf("locks = %+v %+v, want config hashes", target, assistant)
	}
}

func BenchmarkProductionMTPPromotion_PassingEvidence(b *testing.B) {
	policy := DefaultProductionMTPPolicy()
	evidence := productionMTPPassingEvidence()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		productionMTPSink = EvaluateProductionMTPPromotion(policy, evidence)
	}
}

func BenchmarkProductionMTPAttachedDrafterEvidence_ApplyPlan(b *testing.B) {
	plan, err := PlanAttachedDrafter(
		productionMTPE2BQ6TargetModel(),
		productionMTPE2BBF16AssistantModel(),
	)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var evidence ProductionMTPPromotionEvidence
		if err := ApplyProductionMTPAttachedDrafterPlanEvidence(&evidence, plan); err != nil {
			b.Fatal(err)
		}
		productionMTPEvidenceSink = evidence
	}
}

func BenchmarkProductionMTPAttachedDrafterEvidence_ApplyLabels(b *testing.B) {
	labels := map[string]string{}
	rocmAddGemma4AttachedDrafterCapabilityLabels(labels)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var evidence ProductionMTPPromotionEvidence
		if err := ApplyProductionMTPAttachedDrafterLabelEvidence(&evidence, labels); err != nil {
			b.Fatal(err)
		}
		productionMTPEvidenceSink = evidence
	}
}

func BenchmarkProductionMTPLabelEvidence_ApplyMeasuredLabels(b *testing.B) {
	labels := productionMTPPassingLabels()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var evidence ProductionMTPPromotionEvidence
		if err := ApplyProductionMTPLabelEvidence(&evidence, labels); err != nil {
			b.Fatal(err)
		}
		productionMTPEvidenceSink = evidence
	}
}

func BenchmarkProductionMTPPromotionMetricLabels_EvaluatePassing(b *testing.B) {
	labels := productionMTPPassingLabels()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		decision, err := EvaluateProductionMTPPromotionMetricLabels(labels)
		if err != nil {
			b.Fatal(err)
		}
		productionMTPSink = decision
	}
}

func BenchmarkProductionMTPDecodeRunEvidence_ApplyMeasuredResult(b *testing.B) {
	run := productionMTPPassingDecodeRunEvidence()
	result := inferdecode.Result{
		Mode: inferdecode.ModeSpeculative,
		Metrics: inferdecode.Metrics{
			TargetTokens:   880,
			DraftTokens:    40,
			AcceptedTokens: 30,
			RejectedTokens: 10,
			EmittedTokens:  1000,
			TargetCalls:    20,
			DraftCalls:     20,
			Duration:       8 * time.Second,
			TargetDuration: 8 * time.Second,
			DraftDuration:  500 * time.Millisecond,
		},
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var evidence ProductionMTPPromotionEvidence
		if err := ApplyProductionMTPDecodeRunEvidence(&evidence, result, run); err != nil {
			b.Fatal(err)
		}
		productionMTPEvidenceSink = evidence
	}
}

func productionMTPPassingEvidence() ProductionMTPPromotionEvidence {
	return ProductionMTPPromotionEvidence{
		RetainedWorkflow:                       true,
		Turns:                                  ProductionMTPPromotionMinRetainedTurns,
		GreedyOutputMatches:                    true,
		TargetOnlyVisibleTokensPerSec:          105,
		MTPVisibleTokensPerSec:                 125,
		MTPTargetTokensPerSec:                  110,
		MTPWarmDecodeTokensPerSec:              123,
		TargetOnlyWallDuration:                 10 * time.Second,
		MTPWallDuration:                        8 * time.Second,
		TargetOnlyRestoreDuration:              100 * time.Millisecond,
		MTPRestoreDuration:                     80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:              4096,
		MTPPeakMemoryBytes:                     3584,
		TargetOnlyActivePlusCacheMemoryBytes:   2560,
		MTPActivePlusCacheMemoryBytes:          2304,
		TargetOnlyEnergyJoules:                 1000,
		MTPEnergyJoules:                        760,
		SameLoadPolicy:                         true,
		TargetOnlyCacheMode:                    rocmKVCacheModeKQ8VQ4,
		MTPCacheMode:                           rocmKVCacheModeKQ8VQ4,
		SpeculativeDraftModelPath:              rocmGemma4MTPAssistantPath("E2B", "bf16"),
		SpeculativeDraftTokens:                 ProductionMTPDefaultDraftTokens,
		AttachedDrafterRetainedStateEntrypoint: true,
		AttachedDrafterRetainedStateRequired:   true,
		AttachedDrafterStateSource:             "rocm_state_session_runtime_kv",
		AttachedDrafterPromptReplayFallback:    "forbidden",
		AttachedDrafterNativeAttachment:        hipKernelStatusLinked,
		AttachedDrafterNativeHandoff:           attachedDrafterNativeHandoffRetainedStateVerifier,
		AttachedDrafterTargetRetainedDecode:    hipKernelStatusLinked,
		AttachedDrafterTargetRetainedState:     hipKernelStatusLinked,
		AttachedDrafterAssistantVerify:         hipKernelStatusLinked,
		AttachedDrafterAssistantStateVerify:    hipKernelStatusLinked,
		TargetGemma4Size:                       "E2B",
		TargetGemma4QuantMode:                  "q6",
		TargetGemma4QuantGroup:                 64,
		TargetGemma4Runtime:                    Gemma4RuntimeMLXAffine,
		TargetGemma4GenerateStatus:             Gemma4GenerateLinked,
		TargetProductionQuantModelID:           ProductionLaneCurrentModelID,
		TargetProductionQuantLockedModelID:     ProductionLaneModelID,
		AssistantGemma4Size:                    "E2B",
		AssistantGemma4QuantMode:               "bf16",
		AssistantGemma4Runtime:                 Gemma4RuntimeBF16,
		AssistantGemma4GenerateStatus:          Gemma4GenerateLoadOnly,
		AssistantProductionQuantModelID:        officialGemma4E2BAssistantModelID,
		AssistantProductionQuantPack:           "E2B:assistant-bf16",
		AssistantProductionQuantTier:           "mtp-assistant",
		AssistantProductionQuantMTPAssistant:   true,
		AssistantProductionQuantTargetFamily:   "gemma4",
		AssistantArchitecture:                  officialGemma4E2BAssistantArchitecture,
		AssistantOrderedEmbeddings:             true,
		AssistantCentroids:                     ProductionMTPAssistantOrderedEmbeddingCentroids,
		AssistantCentroidIntermediateTopK:      ProductionMTPAssistantCentroidIntermediateTopK,
		AssistantFourLayerDrafter:              true,
		AssistantTokenOrderingDType:            "int64",
		AssistantTokenOrderingShape:            []int{ProductionMTPAssistantOrderedEmbeddingCentroids, ProductionMTPAssistantTokenOrderingVocabSize / ProductionMTPAssistantOrderedEmbeddingCentroids},
		Gemma4FamilyPairVerified:               true,
		OfficialPairVerified:                   true,
		OfficialTargetModelID:                  officialGemma4E2BTargetModelID,
		OfficialTargetRevision:                 officialGemma4E2BTargetRevision,
		OfficialAssistantModelID:               officialGemma4E2BAssistantModelID,
		OfficialAssistantRevision:              officialGemma4E2BAssistantRevision,
		MTPDraftTokenSchedule:                  []int{ProductionMTPDefaultDraftTokens, ProductionMTPDefaultDraftTokens},
		MTPObservedDraftTokenSweeps:            []int{1, 2, 4},
		MTPProposedTokens:                      40,
		MTPAcceptedTokens:                      30,
		MTPRejectedTokens:                      10,
		MTPTargetVerifyCalls:                   20,
		MTPDraftCalls:                          20,
	}
}

func productionMTPPassingLabels() map[string]string {
	labels := map[string]string{}
	rocmAddGemma4AttachedDrafterCapabilityLabels(labels)
	labels["retained_workflow"] = "true"
	labels["turns"] = strconv.Itoa(ProductionMTPPromotionMinRetainedTurns)
	labels["greedy_output_matches"] = "true"
	labels["quality_flags"] = ""
	labels["target_only_visible_tokens_per_sec"] = "105"
	labels["mtp_visible_tokens_per_sec"] = "125"
	labels["mtp_target_tokens_per_sec"] = "110"
	labels["mtp_warm_decode_tokens_per_sec"] = "123"
	labels["target_only_wall_duration"] = "10s"
	labels["mtp_wall_duration"] = "8s"
	labels["target_only_restore_duration"] = "100ms"
	labels["mtp_restore_duration"] = "80ms"
	labels["target_only_peak_memory_bytes"] = "4096"
	labels["mtp_peak_memory_bytes"] = "3584"
	labels["target_only_active_plus_cache_memory_bytes"] = "2560"
	labels["mtp_active_plus_cache_memory_bytes"] = "2304"
	labels["target_only_energy_joules"] = "1000"
	labels["mtp_energy_joules"] = "760"
	labels["same_load_policy"] = "true"
	labels["target_only_cache_mode"] = rocmKVCacheModeKQ8VQ4
	labels["mtp_cache_mode"] = rocmKVCacheModeKQ8VQ4
	markProductionMTPNativeHandoffLabelsLinked(labels)
	labels["mtp_draft_token_schedule"] = "2,2"
	labels["mtp_observed_draft_token_sweeps"] = "1,2,4"
	labels["mtp_proposed_tokens"] = "40"
	labels["mtp_accepted_tokens"] = "30"
	labels["mtp_rejected_tokens"] = "10"
	labels["mtp_target_verify_calls"] = "20"
	labels["mtp_draft_calls"] = "20"
	return labels
}

func productionMTPPassingDecodeRunEvidence() ProductionMTPDecodeRunEvidence {
	return ProductionMTPDecodeRunEvidence{
		RetainedWorkflow:                     true,
		Turns:                                ProductionMTPPromotionMinRetainedTurns,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        105,
		TargetOnlyWallDuration:               10 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		SameLoadPolicy:                       true,
		TargetOnlyCacheMode:                  rocmKVCacheModeKQ8VQ4,
		MTPCacheMode:                         rocmKVCacheModeKQ8VQ4,
		AttachedDrafterNativeAttachment:      hipKernelStatusLinked,
		AttachedDrafterNativeHandoff:         attachedDrafterNativeHandoffRetainedStateVerifier,
		AttachedDrafterTargetRetainedDecode:  hipKernelStatusLinked,
		AttachedDrafterTargetRetainedState:   hipKernelStatusLinked,
		AttachedDrafterAssistantVerify:       hipKernelStatusLinked,
		AttachedDrafterAssistantStateVerify:  hipKernelStatusLinked,
		DraftTokenSchedule:                   []int{ProductionMTPDefaultDraftTokens, ProductionMTPDefaultDraftTokens},
		ObservedDraftTokenSweeps:             []int{1, 2, 4},
	}
}

func clearProductionMTPAttachedDrafterStaticEvidence(evidence *ProductionMTPPromotionEvidence) {
	evidence.SpeculativeDraftModelPath = ""
	evidence.SpeculativeDraftTokens = 0
	clearProductionMTPAttachedDrafterRouteEvidence(evidence)
	evidence.TargetGemma4Size = ""
	evidence.TargetGemma4QuantMode = ""
	evidence.TargetGemma4QuantGroup = 0
	evidence.TargetGemma4Runtime = ""
	evidence.TargetGemma4GenerateStatus = ""
	evidence.TargetProductionQuantModelID = ""
	evidence.TargetProductionQuantLockedModelID = ""
	evidence.AssistantGemma4Size = ""
	evidence.AssistantGemma4QuantMode = ""
	evidence.AssistantGemma4QuantGroup = 0
	evidence.AssistantGemma4Runtime = ""
	evidence.AssistantGemma4GenerateStatus = ""
	evidence.AssistantProductionQuantModelID = ""
	evidence.AssistantProductionQuantPack = ""
	evidence.AssistantProductionQuantTier = ""
	evidence.AssistantProductionQuantMTPAssistant = false
	evidence.AssistantProductionQuantTargetFamily = ""
	evidence.AssistantArchitecture = ""
	evidence.AssistantOrderedEmbeddings = false
	evidence.AssistantCentroids = 0
	evidence.AssistantCentroidIntermediateTopK = 0
	evidence.AssistantFourLayerDrafter = false
	evidence.AssistantTokenOrderingDType = ""
	evidence.AssistantTokenOrderingShape = nil
	evidence.Gemma4FamilyPairVerified = false
	evidence.OfficialPairVerified = false
	evidence.OfficialTargetModelID = ""
	evidence.OfficialTargetRevision = ""
	evidence.OfficialAssistantModelID = ""
	evidence.OfficialAssistantRevision = ""
}

func clearProductionMTPAttachedDrafterRouteEvidence(evidence *ProductionMTPPromotionEvidence) {
	evidence.AttachedDrafterRetainedStateEntrypoint = false
	evidence.AttachedDrafterRetainedStateRequired = false
	evidence.AttachedDrafterStateSource = ""
	evidence.AttachedDrafterPromptReplayFallback = ""
	evidence.AttachedDrafterNativeAttachment = ""
	evidence.AttachedDrafterNativeHandoff = ""
	evidence.AttachedDrafterTargetRetainedDecode = ""
	evidence.AttachedDrafterTargetRetainedState = ""
	evidence.AttachedDrafterAssistantVerify = ""
	evidence.AttachedDrafterAssistantStateVerify = ""
}

func markProductionMTPNativeHandoffEvidenceLinked(evidence *ProductionMTPPromotionEvidence) {
	evidence.AttachedDrafterNativeAttachment = hipKernelStatusLinked
	evidence.AttachedDrafterNativeHandoff = attachedDrafterNativeHandoffRetainedStateVerifier
	evidence.AttachedDrafterTargetRetainedDecode = hipKernelStatusLinked
	evidence.AttachedDrafterTargetRetainedState = hipKernelStatusLinked
	evidence.AttachedDrafterAssistantVerify = hipKernelStatusLinked
	evidence.AttachedDrafterAssistantStateVerify = hipKernelStatusLinked
}

func markProductionMTPNativeHandoffLabelsLinked(labels map[string]string) {
	labels["attached_drafter_native_attachment"] = hipKernelStatusLinked
	labels["attached_drafter_native_handoff"] = attachedDrafterNativeHandoffRetainedStateVerifier
	labels["attached_drafter_target_retained_decode"] = hipKernelStatusLinked
	labels["attached_drafter_target_retained_state_decode"] = hipKernelStatusLinked
	labels["attached_drafter_assistant_verify"] = hipKernelStatusLinked
	labels["attached_drafter_assistant_state_verify"] = hipKernelStatusLinked
	labels["attached_drafter_assistant_draft_step_input_bridge"] = hipKernelStatusLinked
	labels["attached_drafter_assistant_draft_step_hidden_runtime"] = hipKernelStatusLinked
	labels["attached_drafter_assistant_draft_step_proposal_runtime"] = hipKernelStatusLinked
}

func productionMTPE2BQ6TargetModel() *rocmModel {
	return &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e2b-it-6bit",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    productionLaneGemma4E2BLayers,
			HiddenSize:   productionLaneGemma4E2BHiddenSize,
			VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
		},
	}
}

func productionMTPE2BBF16AssistantModel() *rocmModel {
	return &rocmModel{
		modelPath: rocmGemma4MTPAssistantPath("E2B", "bf16"),
		modelInfo: inference.ModelInfo{
			Architecture: officialGemma4E2BAssistantArchitecture,
			NumLayers:    4,
			HiddenSize:   productionLaneGemma4E2BHiddenSize,
			QuantBits:    16,
			VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
		},
	}
}

func productionMTPE2BQ4GGUFTargetModel() *rocmModel {
	return &rocmModel{
		modelPath: "/models/unsloth-gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    productionLaneGemma4E2BLayers,
			HiddenSize:   productionLaneGemma4E2BHiddenSize,
			QuantBits:    4,
			QuantGroup:   64,
			VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
		},
	}
}

func productionMTPE2BQ8GGUFAssistantModel() *rocmModel {
	return &rocmModel{
		modelPath: "/models/unsloth-gemma-4-E2B-it-GGUF/MTP/gemma-4-E2B-it-Q8_0-MTP.gguf",
		modelInfo: inference.ModelInfo{
			Architecture: officialGemma4E2BAssistantArchitecture,
			NumLayers:    4,
			HiddenSize:   productionLaneGemma4E2BHiddenSize,
			QuantBits:    8,
			QuantGroup:   64,
			VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
		},
	}
}

func productionMTPE4BQ8TargetModel() *rocmModel {
	return &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e4b-it-8bit",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    26,
			HiddenSize:   2304,
			VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
		},
	}
}

func productionMTPE4BBF16AssistantModel() *rocmModel {
	return &rocmModel{
		modelPath: rocmGemma4MTPAssistantPath("E4B", "bf16"),
		modelInfo: inference.ModelInfo{
			Architecture: officialGemma4E2BAssistantArchitecture,
			NumLayers:    4,
			HiddenSize:   2304,
			QuantBits:    16,
			VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
		},
	}
}

func productionMTP12BQ6TargetModel() *rocmModel {
	return &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-12b-it-6bit",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    48,
			HiddenSize:   3840,
			VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
		},
	}
}

func productionMTP12BQ6QATTargetModel() *rocmModel {
	return &rocmModel{
		modelPath: "mlx-community/gemma-4-12B-it-qat-6bit",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    48,
			HiddenSize:   3840,
			QuantBits:    6,
			VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
		},
	}
}

func productionMTP12BQ6QATAssistantModel() *rocmModel {
	return &rocmModel{
		modelPath: rocmGemma4MTPAssistantPath("12B", "q6"),
		modelInfo: inference.ModelInfo{
			Architecture: officialGemma4E2BAssistantArchitecture,
			NumLayers:    4,
			HiddenSize:   3840,
			QuantBits:    6,
			QuantGroup:   64,
			VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
		},
	}
}

func productionMTP12BBF16AssistantModel() *rocmModel {
	return &rocmModel{
		modelPath: rocmGemma4MTPAssistantPath("12B", "bf16"),
		modelInfo: inference.ModelInfo{
			Architecture: officialGemma4E2BAssistantArchitecture,
			NumLayers:    4,
			HiddenSize:   3840,
			QuantBits:    16,
			VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
		},
	}
}

func intSliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func stringSliceContains(values []string, needle string) bool {
	for _, value := range values {
		if value == needle {
			return true
		}
	}
	return false
}
