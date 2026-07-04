// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"strconv"
	"strings"
	"time"

	core "dappco.re/go"
)

const (
	// ProductionTurboQuantKVLayoutVersion is the promoted physical K/V payload
	// schema expected by the explicit TurboQuant evidence gate.
	ProductionTurboQuantKVLayoutVersion = "turboquant-kv-v1"
	ProductionTurboQuantKeyAlgorithm    = "turboquantprod"
	ProductionTurboQuantValueAlgorithm  = "turboquantmse"
	ProductionTurboQuantOutlierPolicy   = "high-half-head-dim-v1"

	productionTurboQuantCacheModePaged = "paged"
)

var (
	defaultProductionTurboQuantCompareAgainstCacheModes = []string{
		rocmKVCacheModeFP16,
		productionTurboQuantCacheModePaged,
		rocmKVCacheModeQ8,
		rocmKVCacheModeKQ8VQ4,
	}
	defaultProductionTurboQuantRequiredMetrics = []string{
		"retained_workflow",
		"turns",
		"quality_matches",
		"quality_flags",
		"baseline_cache_mode",
		"candidate_cache_mode",
		"candidate_layout_version",
		"candidate_key_algorithm",
		"candidate_value_algorithm",
		"candidate_outlier_policy",
		"candidate_effective_bits_milli",
		"candidate_qjl_residual",
		"candidate_metadata_bytes",
		"same_load_policy",
		"baseline_cache_policy",
		"candidate_cache_policy",
		"baseline_context_length",
		"candidate_context_length",
		"normal_context_validated",
		"stress_context_validated",
		"candidate_peak_memory_bytes",
		"baseline_peak_memory_bytes",
		"candidate_active_plus_cache_memory_bytes",
		"baseline_active_plus_cache_memory_bytes",
		"candidate_wall_duration",
		"baseline_wall_duration",
		"candidate_restore_duration",
		"baseline_restore_duration",
		"candidate_visible_tokens_per_sec",
		"baseline_visible_tokens_per_sec",
		"candidate_input_output_tokens_per_sec",
		"baseline_input_output_tokens_per_sec",
		"candidate_energy_joules",
		"baseline_energy_joules",
		"estimated_power_watts",
	}
	defaultProductionTurboQuantCompareAgainstCacheModesLabel = strings.Join(defaultProductionTurboQuantCompareAgainstCacheModes, ",")
	defaultProductionTurboQuantRequiredMetricsLabel          = strings.Join(defaultProductionTurboQuantRequiredMetrics, ",")
	defaultProductionTurboQuantPolicy                        = ProductionTurboQuantPolicy{
		TargetModelID:                   ProductionLaneCurrentModelID,
		CacheMode:                       rocmTurboQuantKVMode,
		Mode:                            rocmTurboQuantKVMode,
		TargetEffectiveBitsMilli:        3500,
		RequiredLayoutVersion:           ProductionTurboQuantKVLayoutVersion,
		RequiredKeyAlgorithm:            ProductionTurboQuantKeyAlgorithm,
		RequiredValueAlgorithm:          ProductionTurboQuantValueAlgorithm,
		RequiredOutlierPolicy:           ProductionTurboQuantOutlierPolicy,
		RequiresQJLResidual:             true,
		RequiresMetadataAccounting:      true,
		EnabledByDefault:                true,
		RequiresExplicitOptIn:           false,
		RequiresRetainedWorkflow:        true,
		RequiresQualityParity:           true,
		RequiresSideBySideBenchmark:     true,
		RequiresNormalContextValidation: true,
		RequiresStressContextValidation: true,
		MinimumRetainedTurns:            ProductionMTPPromotionMinRetainedTurns,
		NormalContextLength:             ProductionLaneLongContextLength,
		StressContextLength:             ProductionLaneHyperLongContextLength,
		CompareAgainstCacheModes:        defaultProductionTurboQuantCompareAgainstCacheModes,
		RequiredMetrics:                 defaultProductionTurboQuantRequiredMetrics,
	}
)

// ProductionTurboQuantPolicy describes the evidence required before the
// explicit TurboQuant KV-cache mode can become a production candidate.
type ProductionTurboQuantPolicy struct {
	TargetModelID                   string   `json:"target_model_id"`
	CacheMode                       string   `json:"cache_mode"`
	Mode                            string   `json:"mode"`
	TargetEffectiveBitsMilli        int      `json:"target_effective_bits_milli"`
	RequiredLayoutVersion           string   `json:"required_layout_version"`
	RequiredKeyAlgorithm            string   `json:"required_key_algorithm"`
	RequiredValueAlgorithm          string   `json:"required_value_algorithm"`
	RequiredOutlierPolicy           string   `json:"required_outlier_policy"`
	RequiresQJLResidual             bool     `json:"requires_qjl_residual"`
	RequiresMetadataAccounting      bool     `json:"requires_metadata_accounting"`
	EnabledByDefault                bool     `json:"enabled_by_default"`
	RequiresExplicitOptIn           bool     `json:"requires_explicit_opt_in"`
	RequiresRetainedWorkflow        bool     `json:"requires_retained_workflow"`
	RequiresQualityParity           bool     `json:"requires_quality_parity"`
	RequiresSideBySideBenchmark     bool     `json:"requires_side_by_side_benchmark"`
	RequiresNormalContextValidation bool     `json:"requires_normal_context_validation"`
	RequiresStressContextValidation bool     `json:"requires_stress_context_validation"`
	MinimumRetainedTurns            int      `json:"minimum_retained_turns"`
	NormalContextLength             int      `json:"normal_context_length"`
	StressContextLength             int      `json:"stress_context_length"`
	CompareAgainstCacheModes        []string `json:"compare_against_cache_modes"`
	RequiredMetrics                 []string `json:"required_metrics"`
}

type ProductionTurboQuantPromotionEvidence struct {
	RetainedWorkflow                    bool          `json:"retained_workflow"`
	Turns                               int           `json:"turns"`
	QualityMatches                      bool          `json:"quality_matches"`
	QualityFlags                        []string      `json:"quality_flags,omitempty"`
	BaselineCacheMode                   string        `json:"baseline_cache_mode"`
	CandidateCacheMode                  string        `json:"candidate_cache_mode"`
	CandidateLayoutVersion              string        `json:"candidate_layout_version,omitempty"`
	CandidateKeyAlgorithm               string        `json:"candidate_key_algorithm,omitempty"`
	CandidateValueAlgorithm             string        `json:"candidate_value_algorithm,omitempty"`
	CandidateOutlierPolicy              string        `json:"candidate_outlier_policy,omitempty"`
	CandidateEffectiveBitsMilli         int           `json:"candidate_effective_bits_milli,omitempty"`
	CandidateQJLResidual                bool          `json:"candidate_qjl_residual"`
	CandidateMetadataBytes              uint64        `json:"candidate_metadata_bytes,omitempty"`
	SameLoadPolicy                      bool          `json:"same_load_policy"`
	BaselineCachePolicy                 string        `json:"baseline_cache_policy"`
	CandidateCachePolicy                string        `json:"candidate_cache_policy"`
	BaselineContextLength               int           `json:"baseline_context_length"`
	CandidateContextLength              int           `json:"candidate_context_length"`
	ComparedCacheModes                  []string      `json:"compared_cache_modes,omitempty"`
	NormalContextValidated              bool          `json:"normal_context_validated"`
	StressContextValidated              bool          `json:"stress_context_validated"`
	BaselineVisibleTokensPerSec         float64       `json:"baseline_visible_tokens_per_sec,omitempty"`
	CandidateVisibleTokensPerSec        float64       `json:"candidate_visible_tokens_per_sec,omitempty"`
	BaselineInputOutputTokensPerSec     float64       `json:"baseline_input_output_tokens_per_sec,omitempty"`
	CandidateInputOutputTokensPerSec    float64       `json:"candidate_input_output_tokens_per_sec,omitempty"`
	BaselineWallDuration                time.Duration `json:"baseline_wall_duration,omitempty"`
	CandidateWallDuration               time.Duration `json:"candidate_wall_duration,omitempty"`
	BaselineRestoreDuration             time.Duration `json:"baseline_restore_duration,omitempty"`
	CandidateRestoreDuration            time.Duration `json:"candidate_restore_duration,omitempty"`
	BaselinePeakMemoryBytes             uint64        `json:"baseline_peak_memory_bytes,omitempty"`
	CandidatePeakMemoryBytes            uint64        `json:"candidate_peak_memory_bytes,omitempty"`
	BaselineActivePlusCacheMemoryBytes  uint64        `json:"baseline_active_plus_cache_memory_bytes,omitempty"`
	CandidateActivePlusCacheMemoryBytes uint64        `json:"candidate_active_plus_cache_memory_bytes,omitempty"`
	BaselineEnergyJoules                float64       `json:"baseline_energy_joules,omitempty"`
	CandidateEnergyJoules               float64       `json:"candidate_energy_joules,omitempty"`
	EstimatedPowerWatts                 float64       `json:"estimated_power_watts,omitempty"`
}

type ProductionTurboQuantPromotionDecision struct {
	ProductionCandidate bool    `json:"production_candidate"`
	EnableByDefault     bool    `json:"enable_by_default"`
	Reason              string  `json:"reason"`
	WallSpeedup         float64 `json:"wall_speedup,omitempty"`
	VisibleSpeedup      float64 `json:"visible_speedup,omitempty"`
	RestoreSpeedup      float64 `json:"restore_speedup,omitempty"`
	MemorySavingsRatio  float64 `json:"memory_savings_ratio,omitempty"`
	EnergySavingsRatio  float64 `json:"energy_savings_ratio,omitempty"`
}

func DefaultProductionTurboQuantPolicy() ProductionTurboQuantPolicy {
	policy := defaultProductionTurboQuantPolicy
	policy.CompareAgainstCacheModes = append([]string(nil), policy.CompareAgainstCacheModes...)
	policy.RequiredMetrics = append([]string(nil), policy.RequiredMetrics...)
	return policy
}

// ApplyProductionTurboQuantLabelEvidence fills promotion evidence from
// benchmark/capability labels. It accepts the static runtime-report labels
// emitted by ROCm and measured benchmark-row labels with matching metric names.
func ApplyProductionTurboQuantLabelEvidence(evidence *ProductionTurboQuantPromotionEvidence, labels map[string]string) error {
	if evidence == nil {
		return core.E("rocm.ApplyProductionTurboQuantLabelEvidence", "evidence is required", nil)
	}
	if labels == nil {
		return core.E("rocm.ApplyProductionTurboQuantLabelEvidence", "labels are required", nil)
	}
	evidence.CandidateCacheMode = firstNonEmptyString(labels["candidate_cache_mode"], labels["turboquant_candidate_cache_mode"], labels["kv_compression"], labels["production_candidate_cache_mode"])
	evidence.BaselineCacheMode = firstNonEmptyString(labels["baseline_cache_mode"], labels["turboquant_baseline_cache_mode"])
	evidence.CandidateLayoutVersion = firstNonEmptyString(labels["candidate_layout_version"], labels["turboquant_candidate_layout_version"], labels["production_required_layout_version"])
	evidence.CandidateKeyAlgorithm = firstNonEmptyString(labels["candidate_key_algorithm"], labels["turboquant_candidate_key_algorithm"], labels["production_required_key_algorithm"])
	evidence.CandidateValueAlgorithm = firstNonEmptyString(labels["candidate_value_algorithm"], labels["turboquant_candidate_value_algorithm"], labels["production_required_value_algorithm"])
	evidence.CandidateOutlierPolicy = firstNonEmptyString(labels["candidate_outlier_policy"], labels["turboquant_candidate_outlier_policy"], labels["production_required_outlier_policy"])
	evidence.BaselineCachePolicy = firstNonEmptyString(labels["baseline_cache_policy"], labels["turboquant_baseline_cache_policy"])
	evidence.CandidateCachePolicy = firstNonEmptyString(labels["candidate_cache_policy"], labels["turboquant_candidate_cache_policy"])
	if value := firstNonEmptyString(labels["compared_cache_modes"], labels["production_compare_cache_modes"]); value != "" {
		evidence.ComparedCacheModes = splitProductionCSVLabel(value)
	}
	if value := firstNonEmptyString(labels["quality_flags"], labels["turboquant_quality_flags"]); value != "" {
		evidence.QualityFlags = splitProductionCSVLabel(value)
	}
	if err := productionTurboQuantApplyBoolLabel(labels, "retained_workflow", &evidence.RetainedWorkflow); err != nil {
		return err
	}
	if err := productionTurboQuantApplyBoolLabel(labels, "quality_matches", &evidence.QualityMatches); err != nil {
		return err
	}
	if err := productionTurboQuantApplyBoolLabel(labels, "candidate_qjl_residual", &evidence.CandidateQJLResidual, "turboquant_candidate_qjl_residual"); err != nil {
		return err
	}
	if err := productionTurboQuantApplyBoolLabel(labels, "same_load_policy", &evidence.SameLoadPolicy); err != nil {
		return err
	}
	if err := productionTurboQuantApplyBoolLabel(labels, "normal_context_validated", &evidence.NormalContextValidated, "turboquant_normal_context_validated"); err != nil {
		return err
	}
	if err := productionTurboQuantApplyBoolLabel(labels, "stress_context_validated", &evidence.StressContextValidated, "turboquant_stress_context_validated"); err != nil {
		return err
	}
	if err := productionTurboQuantApplyIntLabel(labels, []string{"turns"}, &evidence.Turns); err != nil {
		return err
	}
	if err := productionTurboQuantApplyIntLabel(labels, []string{"candidate_effective_bits_milli", "turboquant_candidate_effective_bits_milli", "production_target_effective_bits_milli"}, &evidence.CandidateEffectiveBitsMilli); err != nil {
		return err
	}
	if err := productionTurboQuantApplyIntLabel(labels, []string{"baseline_context_length"}, &evidence.BaselineContextLength); err != nil {
		return err
	}
	if err := productionTurboQuantApplyIntLabel(labels, []string{"candidate_context_length"}, &evidence.CandidateContextLength); err != nil {
		return err
	}
	if err := productionTurboQuantApplyUint64Label(labels, []string{"candidate_metadata_bytes", "turboquant_candidate_metadata_bytes"}, &evidence.CandidateMetadataBytes); err != nil {
		return err
	}
	if err := productionTurboQuantApplyUint64Label(labels, []string{"baseline_peak_memory_bytes"}, &evidence.BaselinePeakMemoryBytes); err != nil {
		return err
	}
	if err := productionTurboQuantApplyUint64Label(labels, []string{"candidate_peak_memory_bytes", "turboquant_candidate_peak_memory_bytes"}, &evidence.CandidatePeakMemoryBytes); err != nil {
		return err
	}
	if err := productionTurboQuantApplyUint64Label(labels, []string{"baseline_active_plus_cache_memory_bytes"}, &evidence.BaselineActivePlusCacheMemoryBytes); err != nil {
		return err
	}
	if err := productionTurboQuantApplyUint64Label(labels, []string{"candidate_active_plus_cache_memory_bytes", "turboquant_candidate_active_plus_cache_memory_bytes"}, &evidence.CandidateActivePlusCacheMemoryBytes); err != nil {
		return err
	}
	if err := productionTurboQuantApplyFloat64Label(labels, []string{"baseline_visible_tokens_per_sec"}, &evidence.BaselineVisibleTokensPerSec); err != nil {
		return err
	}
	if err := productionTurboQuantApplyFloat64Label(labels, []string{"candidate_visible_tokens_per_sec", "turboquant_candidate_visible_tokens_per_sec"}, &evidence.CandidateVisibleTokensPerSec); err != nil {
		return err
	}
	if err := productionTurboQuantApplyFloat64Label(labels, []string{"baseline_input_output_tokens_per_sec"}, &evidence.BaselineInputOutputTokensPerSec); err != nil {
		return err
	}
	if err := productionTurboQuantApplyFloat64Label(labels, []string{"candidate_input_output_tokens_per_sec", "turboquant_candidate_input_output_tokens_per_sec"}, &evidence.CandidateInputOutputTokensPerSec); err != nil {
		return err
	}
	if err := productionTurboQuantApplyFloat64Label(labels, []string{"baseline_energy_joules"}, &evidence.BaselineEnergyJoules); err != nil {
		return err
	}
	if err := productionTurboQuantApplyFloat64Label(labels, []string{"candidate_energy_joules", "turboquant_candidate_energy_joules"}, &evidence.CandidateEnergyJoules); err != nil {
		return err
	}
	if err := productionTurboQuantApplyFloat64Label(labels, []string{"estimated_power_watts"}, &evidence.EstimatedPowerWatts); err != nil {
		return err
	}
	if err := productionTurboQuantApplyDurationLabel(labels, []string{"baseline_wall_duration"}, &evidence.BaselineWallDuration); err != nil {
		return err
	}
	if err := productionTurboQuantApplyDurationLabel(labels, []string{"candidate_wall_duration", "turboquant_candidate_wall_duration"}, &evidence.CandidateWallDuration); err != nil {
		return err
	}
	if err := productionTurboQuantApplyDurationLabel(labels, []string{"baseline_restore_duration"}, &evidence.BaselineRestoreDuration); err != nil {
		return err
	}
	if err := productionTurboQuantApplyDurationLabel(labels, []string{"candidate_restore_duration", "turboquant_candidate_restore_duration"}, &evidence.CandidateRestoreDuration); err != nil {
		return err
	}
	return nil
}

func ValidateProductionTurboQuantPromotionMetricLabels(labels map[string]string) error {
	_, err := EvaluateProductionTurboQuantPromotionMetricLabels(labels)
	return err
}

func EvaluateProductionTurboQuantPromotionMetricLabels(labels map[string]string) (ProductionTurboQuantPromotionDecision, error) {
	return EvaluateProductionTurboQuantPromotionMetricLabelsWithPolicy(DefaultProductionTurboQuantPolicy(), labels)
}

func EvaluateProductionTurboQuantPromotionMetricLabelsWithPolicy(policy ProductionTurboQuantPolicy, labels map[string]string) (ProductionTurboQuantPromotionDecision, error) {
	if err := ValidateProductionTurboQuantRequiredMetricLabels(labels); err != nil {
		return ProductionTurboQuantPromotionDecision{}, err
	}
	var evidence ProductionTurboQuantPromotionEvidence
	if err := ApplyProductionTurboQuantLabelEvidence(&evidence, labels); err != nil {
		return ProductionTurboQuantPromotionDecision{}, err
	}
	return EvaluateProductionTurboQuantPromotion(policy, evidence), nil
}

func EvaluateProductionTurboQuantPromotion(policy ProductionTurboQuantPolicy, evidence ProductionTurboQuantPromotionEvidence) ProductionTurboQuantPromotionDecision {
	if policy.CacheMode == "" {
		policy = DefaultProductionTurboQuantPolicy()
	}
	policy = fillProductionTurboQuantPolicyDefaults(policy)
	decision := ProductionTurboQuantPromotionDecision{
		EnableByDefault:    false,
		WallSpeedup:        durationSpeedup(evidence.BaselineWallDuration, evidence.CandidateWallDuration),
		VisibleSpeedup:     ratioSpeedup(evidence.CandidateVisibleTokensPerSec, evidence.BaselineVisibleTokensPerSec),
		RestoreSpeedup:     durationSpeedup(evidence.BaselineRestoreDuration, evidence.CandidateRestoreDuration),
		MemorySavingsRatio: byteSavingsRatio(evidence.BaselineActivePlusCacheMemoryBytes, evidence.CandidateActivePlusCacheMemoryBytes),
		EnergySavingsRatio: ratioSavings(evidence.BaselineEnergyJoules, evidence.CandidateEnergyJoules),
	}
	peakMemorySavingsRatio := byteSavingsRatio(evidence.BaselinePeakMemoryBytes, evidence.CandidatePeakMemoryBytes)
	if evidence.CandidateCacheMode != policy.CacheMode {
		decision.Reason = "TurboQuant candidate cache mode is required"
		return decision
	}
	if evidence.BaselineCacheMode == "" || evidence.BaselineCacheMode == policy.CacheMode || !turboQuantModeInSlice(policy.CompareAgainstCacheModes, evidence.BaselineCacheMode) {
		decision.Reason = "TurboQuant baseline cache mode must be one of fp16, paged, q8, or k-q8-v-q4"
		return decision
	}
	if policy.RequiresRetainedWorkflow && !evidence.RetainedWorkflow {
		decision.Reason = "retained workflow evidence is required before TurboQuant promotion"
		return decision
	}
	if evidence.Turns < policy.MinimumRetainedTurns {
		decision.Reason = "retained workflow turn count is below the TurboQuant promotion minimum"
		return decision
	}
	if policy.RequiresQualityParity && !evidence.QualityMatches {
		decision.Reason = "quality parity is required before TurboQuant promotion"
		return decision
	}
	if len(evidence.QualityFlags) > 0 {
		decision.Reason = "quality flags must be empty before TurboQuant promotion"
		return decision
	}
	if policy.RequiresSideBySideBenchmark && !turboQuantComparedAllModes(policy.CompareAgainstCacheModes, evidence.ComparedCacheModes) {
		decision.Reason = "TurboQuant must be compared side by side against fp16, paged, q8, and k-q8-v-q4 cache modes"
		return decision
	}
	if policy.RequiresNormalContextValidation && !evidence.NormalContextValidated {
		decision.Reason = "normal 30k-40k retained-context validation is required before TurboQuant promotion"
		return decision
	}
	if policy.RequiresStressContextValidation && !evidence.StressContextValidated {
		decision.Reason = "100k stress-context validation is required before TurboQuant promotion"
		return decision
	}
	if evidence.BaselinePeakMemoryBytes == 0 || evidence.CandidatePeakMemoryBytes == 0 {
		decision.Reason = "TurboQuant peak memory evidence is required"
		return decision
	}
	if evidence.BaselineActivePlusCacheMemoryBytes == 0 || evidence.CandidateActivePlusCacheMemoryBytes == 0 {
		decision.Reason = "TurboQuant active+cache memory evidence is required"
		return decision
	}
	if decision.WallSpeedup == 0 || decision.EnergySavingsRatio <= 0 || evidence.EstimatedPowerWatts <= 0 {
		decision.Reason = "TurboQuant wall and estimated-energy evidence are required"
		return decision
	}
	if peakMemorySavingsRatio <= 0 {
		decision.Reason = "TurboQuant peak memory savings are required"
		return decision
	}
	if decision.MemorySavingsRatio <= 0 {
		decision.Reason = "TurboQuant active+cache memory savings are required"
		return decision
	}
	if evidence.BaselineVisibleTokensPerSec <= 0 || evidence.CandidateVisibleTokensPerSec <= 0 {
		decision.Reason = "TurboQuant visible throughput evidence is required"
		return decision
	}
	if !productionTurboQuantHasLoadPolicyEvidence(evidence) {
		decision.Reason = "TurboQuant load policy evidence is required"
		return decision
	}
	if evidence.BaselineInputOutputTokensPerSec <= 0 || evidence.CandidateInputOutputTokensPerSec <= 0 {
		decision.Reason = "TurboQuant input+output throughput evidence is required"
		return decision
	}
	if evidence.CandidateLayoutVersion != policy.RequiredLayoutVersion {
		decision.Reason = "TurboQuant layout version evidence must match " + policy.RequiredLayoutVersion
		return decision
	}
	if evidence.CandidateKeyAlgorithm != policy.RequiredKeyAlgorithm || evidence.CandidateValueAlgorithm != policy.RequiredValueAlgorithm {
		decision.Reason = "TurboQuant K/V algorithm evidence must use " + policy.RequiredKeyAlgorithm + " keys and " + policy.RequiredValueAlgorithm + " values"
		return decision
	}
	if evidence.CandidateOutlierPolicy != policy.RequiredOutlierPolicy {
		decision.Reason = "TurboQuant outlier policy evidence must match " + policy.RequiredOutlierPolicy
		return decision
	}
	if evidence.CandidateEffectiveBitsMilli != policy.TargetEffectiveBitsMilli {
		decision.Reason = "TurboQuant effective-bit evidence must match the 3.5 bits/channel target"
		return decision
	}
	if policy.RequiresQJLResidual && !evidence.CandidateQJLResidual {
		decision.Reason = "TurboQuant QJL residual evidence is required"
		return decision
	}
	if policy.RequiresMetadataAccounting && evidence.CandidateMetadataBytes == 0 {
		decision.Reason = "TurboQuant metadata byte accounting is required"
		return decision
	}
	if decision.WallSpeedup <= 1 && decision.RestoreSpeedup <= 1 {
		decision.Reason = "TurboQuant must improve retained wall time or restore time before promotion"
		return decision
	}
	decision.ProductionCandidate = true
	decision.EnableByDefault = policy.EnabledByDefault
	decision.Reason = "TurboQuant retained workflow saves memory/energy with quality parity"
	return decision
}

func fillProductionTurboQuantPolicyDefaults(policy ProductionTurboQuantPolicy) ProductionTurboQuantPolicy {
	defaults := defaultProductionTurboQuantPolicy
	if policy.TargetEffectiveBitsMilli == 0 {
		policy.TargetEffectiveBitsMilli = defaults.TargetEffectiveBitsMilli
	}
	if policy.RequiredLayoutVersion == "" {
		policy.RequiredLayoutVersion = ProductionTurboQuantKVLayoutVersion
	}
	if policy.RequiredKeyAlgorithm == "" {
		policy.RequiredKeyAlgorithm = ProductionTurboQuantKeyAlgorithm
	}
	if policy.RequiredValueAlgorithm == "" {
		policy.RequiredValueAlgorithm = ProductionTurboQuantValueAlgorithm
	}
	if policy.RequiredOutlierPolicy == "" {
		policy.RequiredOutlierPolicy = ProductionTurboQuantOutlierPolicy
	}
	if len(policy.CompareAgainstCacheModes) == 0 {
		policy.CompareAgainstCacheModes = defaults.CompareAgainstCacheModes
	}
	if policy.MinimumRetainedTurns == 0 {
		policy.MinimumRetainedTurns = defaults.MinimumRetainedTurns
	}
	return policy
}

func turboQuantComparedAllModes(required, actual []string) bool {
	for _, want := range required {
		if !turboQuantModeInSlice(actual, want) {
			return false
		}
	}
	return true
}

func turboQuantModeInSlice(values []string, needle string) bool {
	for _, value := range values {
		if value == needle {
			return true
		}
	}
	return false
}

func productionTurboQuantHasLoadPolicyEvidence(evidence ProductionTurboQuantPromotionEvidence) bool {
	return evidence.SameLoadPolicy &&
		evidence.BaselineCachePolicy != "" &&
		evidence.BaselineCachePolicy == evidence.CandidateCachePolicy &&
		evidence.BaselineContextLength > 0 &&
		evidence.BaselineContextLength == evidence.CandidateContextLength
}

func byteSavingsRatio(baseline, candidate uint64) float64 {
	if baseline == 0 || candidate == 0 || candidate >= baseline {
		return 0
	}
	return 1 - float64(candidate)/float64(baseline)
}

func productionTurboQuantApplyBoolLabel(labels map[string]string, key string, target *bool, aliases ...string) error {
	foundKey, value := productionFirstLabel1(labels, key, aliases...)
	if value == "" {
		return nil
	}
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return core.E("rocm.ApplyProductionTurboQuantLabelEvidence", "parse "+foundKey, err)
	}
	*target = parsed
	return nil
}

func productionTurboQuantApplyIntLabel(labels map[string]string, keys []string, target *int) error {
	key, value := productionFirstLabel(labels, keys)
	if value == "" {
		return nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return core.E("rocm.ApplyProductionTurboQuantLabelEvidence", "parse "+key, err)
	}
	*target = parsed
	return nil
}

func productionTurboQuantApplyUint64Label(labels map[string]string, keys []string, target *uint64) error {
	key, value := productionFirstLabel(labels, keys)
	if value == "" {
		return nil
	}
	parsed, err := strconv.ParseUint(value, 10, 64)
	if err != nil {
		return core.E("rocm.ApplyProductionTurboQuantLabelEvidence", "parse "+key, err)
	}
	*target = parsed
	return nil
}

func productionTurboQuantApplyFloat64Label(labels map[string]string, keys []string, target *float64) error {
	key, value := productionFirstLabel(labels, keys)
	if value == "" {
		return nil
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return core.E("rocm.ApplyProductionTurboQuantLabelEvidence", "parse "+key, err)
	}
	*target = parsed
	return nil
}

func productionTurboQuantApplyDurationLabel(labels map[string]string, keys []string, target *time.Duration) error {
	key, value := productionFirstLabel(labels, keys)
	if value == "" {
		return nil
	}
	parsed, err := time.ParseDuration(value)
	if err != nil {
		seconds, secondsErr := strconv.ParseFloat(value, 64)
		if secondsErr != nil {
			return core.E("rocm.ApplyProductionTurboQuantLabelEvidence", "parse "+key, err)
		}
		parsed = time.Duration(seconds * float64(time.Second))
	}
	*target = parsed
	return nil
}

func productionFirstLabel(labels map[string]string, keys []string) (string, string) {
	for _, key := range keys {
		if value := labels[key]; value != "" {
			return key, value
		}
	}
	return "", ""
}

func productionFirstLabel1(labels map[string]string, key string, aliases ...string) (string, string) {
	if value := labels[key]; value != "" {
		return key, value
	}
	for _, alias := range aliases {
		if value := labels[alias]; value != "" {
			return alias, value
		}
	}
	return "", ""
}

func splitProductionCSVLabel(value string) []string {
	if value == "" {
		return nil
	}
	out := make([]string, 0, 1+strings.Count(value, ","))
	for start := 0; start <= len(value); {
		end := start
		for end < len(value) && value[end] != ',' {
			end++
		}
		if trimmed := strings.TrimSpace(value[start:end]); trimmed != "" {
			out = append(out, trimmed)
		}
		start = end + 1
	}
	return out
}
