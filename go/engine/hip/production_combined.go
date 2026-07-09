// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"strings"

	core "dappco.re/go"
)

const ProductionCombinedMTPAndTurboQuantMode = "mtp+turboquant-kv"

var defaultProductionCombinedMTPAndTurboQuantRequiredMetrics = []string{
	"retained_workflow",
	"turns",
	"quality_matches",
	"mtp_greedy_output_matches",
	"quality_flags",
	"mtp_target_only_cache_mode",
	"mtp_cache_mode",
	"mtp_target_only_visible_tokens_per_sec",
	"mtp_visible_tokens_per_sec",
	"mtp_target_tokens_per_sec",
	"mtp_warm_decode_tokens_per_sec",
	"mtp_target_only_wall_duration",
	"mtp_wall_duration",
	"mtp_target_only_restore_duration",
	"mtp_restore_duration",
	"mtp_target_only_peak_memory_bytes",
	"mtp_peak_memory_bytes",
	"mtp_target_only_active_plus_cache_memory_bytes",
	"mtp_active_plus_cache_memory_bytes",
	"mtp_target_only_energy_joules",
	"mtp_energy_joules",
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
	"attached_drafter_target_gemma4_size",
	"attached_drafter_target_gemma4_quant_mode",
	"attached_drafter_target_gemma4_quant_group",
	"attached_drafter_target_gemma4_runtime",
	"attached_drafter_target_gemma4_generate_status",
	"attached_drafter_assistant_gemma4_size",
	"attached_drafter_assistant_gemma4_quant_mode",
	"attached_drafter_assistant_gemma4_runtime",
	"attached_drafter_assistant_gemma4_generate_status",
	"assistant_architecture",
	"assistant_ordered_embeddings",
	"assistant_centroids",
	"assistant_centroid_intermediate_top_k",
	"assistant_four_layer_drafter",
	"assistant_token_ordering_dtype",
	"assistant_token_ordering_shape",
	"gemma4_family_pair_verified",
	"baseline_cache_mode",
	"turboquant_candidate_cache_mode",
	"same_load_policy",
	"baseline_cache_policy",
	"turboquant_candidate_cache_policy",
	"baseline_context_length",
	"candidate_context_length",
	"compared_cache_modes",
	"turboquant_normal_context_validated",
	"turboquant_stress_context_validated",
	"turboquant_candidate_layout_version",
	"turboquant_candidate_key_algorithm",
	"turboquant_candidate_value_algorithm",
	"turboquant_candidate_outlier_policy",
	"turboquant_candidate_effective_bits_milli",
	"turboquant_candidate_qjl_residual",
	"turboquant_candidate_metadata_bytes",
	"turboquant_quality_flags",
	"baseline_visible_tokens_per_sec",
	"turboquant_candidate_visible_tokens_per_sec",
	"baseline_input_output_tokens_per_sec",
	"turboquant_candidate_input_output_tokens_per_sec",
	"baseline_wall_duration",
	"turboquant_candidate_wall_duration",
	"baseline_restore_duration",
	"turboquant_candidate_restore_duration",
	"baseline_peak_memory_bytes",
	"turboquant_candidate_peak_memory_bytes",
	"baseline_active_plus_cache_memory_bytes",
	"turboquant_candidate_active_plus_cache_memory_bytes",
	"baseline_energy_joules",
	"turboquant_candidate_energy_joules",
	"estimated_power_watts",
	"turboquant_active_plus_cache_memory_savings",
}

var defaultProductionCombinedMTPAndTurboQuantRequiredMetricsLabel = strings.Join(defaultProductionCombinedMTPAndTurboQuantRequiredMetrics, ",")

var defaultProductionCombinedMTPAndTurboQuantPolicy = ProductionCombinedMTPAndTurboQuantPolicy{
	TargetModelID:                   officialGemma4E2BTargetModelID,
	AssistantModelID:                officialGemma4E2BAssistantModelID,
	Mode:                            ProductionCombinedMTPAndTurboQuantMode,
	CacheMode:                       rocmTurboQuantKVMode,
	EnabledByDefault:                true,
	RequiresExplicitOptIn:           false,
	RequiresRetainedWorkflow:        true,
	RequiresGreedyParity:            true,
	RequiresTurboQuantQualityParity: true,
	RequiresMTPPromotion:            true,
	RequiresTurboQuantPromotion:     true,
	MinimumRetainedTurns:            ProductionMTPPromotionMinRetainedTurns,
	RequiredMetrics:                 defaultProductionCombinedMTPAndTurboQuantRequiredMetrics,
}

type ProductionCombinedMTPAndTurboQuantPolicy struct {
	TargetModelID                   string   `json:"target_model_id"`
	AssistantModelID                string   `json:"assistant_model_id"`
	Mode                            string   `json:"mode"`
	CacheMode                       string   `json:"cache_mode"`
	EnabledByDefault                bool     `json:"enabled_by_default"`
	RequiresExplicitOptIn           bool     `json:"requires_explicit_opt_in"`
	RequiresRetainedWorkflow        bool     `json:"requires_retained_workflow"`
	RequiresGreedyParity            bool     `json:"requires_greedy_parity"`
	RequiresTurboQuantQualityParity bool     `json:"requires_turboquant_quality_parity"`
	RequiresMTPPromotion            bool     `json:"requires_mtp_promotion"`
	RequiresTurboQuantPromotion     bool     `json:"requires_turboquant_promotion"`
	MinimumRetainedTurns            int      `json:"minimum_retained_turns"`
	RequiredMetrics                 []string `json:"required_metrics,omitempty"`
}

type ProductionCombinedMTPAndTurboQuantDecision struct {
	ProductionCandidate          bool    `json:"production_candidate"`
	EnableByDefault              bool    `json:"enable_by_default"`
	Reason                       string  `json:"reason"`
	MTPEligible                  bool    `json:"mtp_eligible"`
	TurboQuantEligible           bool    `json:"turboquant_eligible"`
	MTPWallSpeedup               float64 `json:"mtp_wall_speedup,omitempty"`
	MTPVisibleSpeedup            float64 `json:"mtp_visible_speedup,omitempty"`
	MTPAcceptanceRate            float64 `json:"mtp_acceptance_rate,omitempty"`
	TurboQuantMemorySavingsRatio float64 `json:"turboquant_memory_savings_ratio,omitempty"`
	TurboQuantEnergySavingsRatio float64 `json:"turboquant_energy_savings_ratio,omitempty"`
}

func DefaultProductionCombinedMTPAndTurboQuantPolicy() ProductionCombinedMTPAndTurboQuantPolicy {
	policy := defaultProductionCombinedMTPAndTurboQuantPolicy
	policy.RequiredMetrics = append([]string(nil), policy.RequiredMetrics...)
	return policy
}

func ApplyProductionCombinedMTPAndTurboQuantLabelEvidence(mtpEvidence *ProductionMTPPromotionEvidence, turboEvidence *ProductionTurboQuantPromotionEvidence, labels map[string]string) error {
	if mtpEvidence == nil || turboEvidence == nil {
		return core.E("rocm.ApplyProductionCombinedMTPAndTurboQuantLabelEvidence", "MTP and TurboQuant evidence are required", nil)
	}
	if labels == nil {
		return core.E("rocm.ApplyProductionCombinedMTPAndTurboQuantLabelEvidence", "labels are required", nil)
	}
	if err := ApplyProductionMTPLabelEvidence(mtpEvidence, labels); err != nil {
		return err
	}
	if err := ApplyProductionTurboQuantLabelEvidence(turboEvidence, labels); err != nil {
		return err
	}
	return nil
}

func ValidateProductionCombinedMTPAndTurboQuantPromotionMetricLabels(labels map[string]string) error {
	_, err := EvaluateProductionCombinedMTPAndTurboQuantPromotionMetricLabels(labels)
	return err
}

func EvaluateProductionCombinedMTPAndTurboQuantPromotionMetricLabels(labels map[string]string) (ProductionCombinedMTPAndTurboQuantDecision, error) {
	return EvaluateProductionCombinedMTPAndTurboQuantPromotionMetricLabelsWithPolicy(DefaultProductionCombinedMTPAndTurboQuantPolicy(), labels)
}

func EvaluateProductionCombinedMTPAndTurboQuantPromotionMetricLabelsWithPolicy(policy ProductionCombinedMTPAndTurboQuantPolicy, labels map[string]string) (ProductionCombinedMTPAndTurboQuantDecision, error) {
	if err := ValidateProductionCombinedMTPAndTurboQuantRequiredMetricLabels(labels); err != nil {
		return ProductionCombinedMTPAndTurboQuantDecision{}, err
	}
	var mtpEvidence ProductionMTPPromotionEvidence
	var turboEvidence ProductionTurboQuantPromotionEvidence
	if err := ApplyProductionCombinedMTPAndTurboQuantLabelEvidence(&mtpEvidence, &turboEvidence, labels); err != nil {
		return ProductionCombinedMTPAndTurboQuantDecision{}, err
	}
	return EvaluateProductionCombinedMTPAndTurboQuantPromotion(policy, mtpEvidence, turboEvidence), nil
}

func EvaluateProductionCombinedMTPAndTurboQuantPromotion(policy ProductionCombinedMTPAndTurboQuantPolicy, mtpEvidence ProductionMTPPromotionEvidence, turboEvidence ProductionTurboQuantPromotionEvidence) ProductionCombinedMTPAndTurboQuantDecision {
	if policy.CacheMode == "" {
		policy = DefaultProductionCombinedMTPAndTurboQuantPolicy()
	}
	mtpDecision := EvaluateProductionMTPPromotion(defaultProductionMTPPolicy, mtpEvidence)
	turboDecision := EvaluateProductionTurboQuantPromotion(defaultProductionTurboQuantPolicy, turboEvidence)
	decision := ProductionCombinedMTPAndTurboQuantDecision{
		MTPEligible:                  mtpDecision.EnableByDefault,
		TurboQuantEligible:           turboDecision.ProductionCandidate,
		MTPWallSpeedup:               mtpDecision.WallSpeedup,
		MTPVisibleSpeedup:            mtpDecision.VisibleSpeedup,
		MTPAcceptanceRate:            mtpDecision.AcceptanceRate,
		TurboQuantMemorySavingsRatio: turboDecision.MemorySavingsRatio,
		TurboQuantEnergySavingsRatio: turboDecision.EnergySavingsRatio,
	}
	if policy.RequiresRetainedWorkflow && (!mtpEvidence.RetainedWorkflow || !turboEvidence.RetainedWorkflow) {
		decision.Reason = "combined MTP+TurboQuant retained workflow evidence is required"
		return decision
	}
	if mtpEvidence.Turns < policy.MinimumRetainedTurns || turboEvidence.Turns < policy.MinimumRetainedTurns {
		decision.Reason = "combined MTP+TurboQuant retained workflow turn count is below the promotion minimum"
		return decision
	}
	if policy.RequiresGreedyParity && !mtpEvidence.GreedyOutputMatches {
		decision.Reason = "combined MTP+TurboQuant requires MTP greedy output parity"
		return decision
	}
	if policy.RequiresTurboQuantQualityParity && !turboEvidence.QualityMatches {
		decision.Reason = "combined MTP+TurboQuant requires TurboQuant quality parity"
		return decision
	}
	if mtpEvidence.TargetOnlyCacheMode != policy.CacheMode || mtpEvidence.MTPCacheMode != policy.CacheMode {
		decision.Reason = "combined MTP benchmark must run target-only and MTP with TurboQuant cache mode"
		return decision
	}
	if turboEvidence.CandidateCacheMode != policy.CacheMode {
		decision.Reason = "combined MTP+TurboQuant requires a TurboQuant candidate cache mode"
		return decision
	}
	if policy.RequiresMTPPromotion && !mtpDecision.EnableByDefault {
		decision.Reason = "MTP must pass target-only retained workflow under TurboQuant: " + mtpDecision.Reason
		return decision
	}
	if policy.RequiresTurboQuantPromotion && !turboDecision.ProductionCandidate {
		decision.Reason = "TurboQuant must pass retained quality/memory gates before combined promotion: " + turboDecision.Reason
		return decision
	}
	decision.ProductionCandidate = true
	decision.EnableByDefault = policy.EnabledByDefault
	decision.Reason = "combined MTP+TurboQuant retained workflow passes both lanes for the production fast lane"
	return decision
}
