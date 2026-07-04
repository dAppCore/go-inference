// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	core "dappco.re/go"
)

var productionMTPPromotionDecisionLabels = []string{
	"production_mtp_enable_by_default",
	"production_mtp_reason",
	"production_mtp_wall_speedup",
	"production_mtp_visible_speedup",
	"production_mtp_restore_speedup",
	"production_mtp_energy_savings",
	"production_mtp_acceptance_rate",
}

var productionTurboQuantPromotionDecisionLabels = []string{
	"production_turboquant_candidate",
	"production_turboquant_enable_by_default",
	"production_turboquant_reason",
	"production_turboquant_wall_speedup",
	"production_turboquant_visible_speedup",
	"production_turboquant_restore_speedup",
	"production_turboquant_memory_savings_ratio",
	"production_turboquant_energy_savings_ratio",
}

var productionCombinedMTPAndTurboQuantDecisionLabels = []string{
	"production_combined_candidate",
	"production_combined_enable_by_default",
	"production_combined_reason",
	"production_combined_mtp_eligible",
	"production_combined_turboquant_eligible",
	"production_combined_mtp_wall_speedup",
	"production_combined_mtp_visible_speedup",
	"production_combined_mtp_acceptance_rate",
	"production_combined_turboquant_memory_savings_ratio",
	"production_combined_turboquant_energy_savings_ratio",
}

func ValidateProductionMTPRequiredMetricLabels(labels map[string]string) error {
	return validateProductionRequiredMetricLabels("rocm.ValidateProductionMTPRequiredMetricLabels", labels, defaultProductionMTPRequiredMetrics, productionMTPRequiredMetricAliases)
}

func ValidateProductionTurboQuantRequiredMetricLabels(labels map[string]string) error {
	return validateProductionRequiredMetricLabels("rocm.ValidateProductionTurboQuantRequiredMetricLabels", labels, defaultProductionTurboQuantRequiredMetrics, productionTurboQuantRequiredMetricAliases)
}

func ValidateProductionCombinedMTPAndTurboQuantRequiredMetricLabels(labels map[string]string) error {
	return validateProductionRequiredMetricLabels("rocm.ValidateProductionCombinedMTPAndTurboQuantRequiredMetricLabels", labels, defaultProductionCombinedMTPAndTurboQuantRequiredMetrics, productionCombinedRequiredMetricAliases)
}

func ValidateProductionAutoRoundCalibrationLabels(labels map[string]string) error {
	var evidence ProductionAutoRoundCalibrationEvidence
	return applyProductionAutoRoundRequiredCalibrationLabelEvidence("rocm.ValidateProductionAutoRoundCalibrationLabels", &evidence, labels)
}

func ValidateProductionAutoRoundCalibrationDecisionLabels(labels map[string]string) error {
	var decision ProductionAutoRoundCalibrationDecision
	return applyProductionAutoRoundRequiredCalibrationDecisionLabelEvidence("rocm.ValidateProductionAutoRoundCalibrationDecisionLabels", &decision, labels)
}

func ValidateProductionAutoRoundCalibrationEvidenceDecisionLabels(evidenceLabels, decisionLabels map[string]string) error {
	var evidence ProductionAutoRoundCalibrationEvidence
	if err := applyProductionAutoRoundRequiredCalibrationLabelEvidence("rocm.ValidateProductionAutoRoundCalibrationEvidenceDecisionLabels", &evidence, evidenceLabels); err != nil {
		return err
	}
	expected := EvaluateProductionAutoRoundCalibrationEvidence(evidence)
	var actual ProductionAutoRoundCalibrationDecision
	if err := applyProductionAutoRoundRequiredCalibrationDecisionLabelEvidence("rocm.ValidateProductionAutoRoundCalibrationEvidenceDecisionLabels", &actual, decisionLabels); err != nil {
		return err
	}
	if actual != expected {
		return core.E("rocm.ValidateProductionAutoRoundCalibrationEvidenceDecisionLabels", "decision labels do not match calibration evidence", nil)
	}
	return nil
}

func applyProductionAutoRoundRequiredCalibrationLabelEvidence(name string, evidence *ProductionAutoRoundCalibrationEvidence, labels map[string]string) error {
	if evidence == nil {
		return core.E(name, "evidence is required", nil)
	}
	if labels == nil {
		return core.E(name, "labels are required", nil)
	}
	var missing []string
	evidence.ProfileName = productionRequiredStringLabel(labels, "autoround_calibration_profile", &missing)
	evidence.Format = productionRequiredStringLabel(labels, "autoround_calibration_format", &missing)
	evidence.WeightScheme = productionRequiredStringLabel(labels, "autoround_calibration_weight_scheme", &missing)
	evidence.FloatFormat = productionRequiredStringLabel(labels, "autoround_calibration_float_format", &missing)
	evidence.Runtime = productionRequiredStringLabel(labels, "autoround_calibration_runtime", &missing)
	evidence.HIPKernel = productionRequiredStringLabel(labels, "autoround_calibration_hip_kernel", &missing)
	var err error
	if evidence.Bits, err = productionRequiredIntLabel(labels, "autoround_calibration_bits", &missing); err != nil {
		return err
	}
	if evidence.GroupSize, err = productionRequiredIntLabel(labels, "autoround_calibration_group_size", &missing); err != nil {
		return err
	}
	if evidence.NSamples, err = productionRequiredIntLabel(labels, "autoround_calibration_nsamples", &missing); err != nil {
		return err
	}
	if evidence.SeqLen, err = productionRequiredIntLabel(labels, "autoround_calibration_seqlen", &missing); err != nil {
		return err
	}
	if evidence.Iters, err = productionRequiredIntLabel(labels, "autoround_calibration_iters", &missing); err != nil {
		return err
	}
	if evidence.RequiresBench, err = productionRequiredBoolLabel(labels, "autoround_calibration_requires_bench", &missing); err != nil {
		return err
	}
	if evidence.RequiresCalibration, err = productionRequiredBoolLabel(labels, "autoround_calibration_required", &missing); err != nil {
		return err
	}
	if len(missing) > 0 {
		return core.E(name, "missing required production metric labels: "+strings.Join(missing, ","), nil)
	}
	return nil
}

func applyProductionAutoRoundRequiredCalibrationDecisionLabelEvidence(name string, decision *ProductionAutoRoundCalibrationDecision, labels map[string]string) error {
	if decision == nil {
		return core.E(name, "decision is required", nil)
	}
	if labels == nil {
		return core.E(name, "labels are required", nil)
	}
	var missing []string
	decision.Reason = productionRequiredStringLabel(labels, "autoround_calibration_decision_reason", &missing)
	decision.ProfileName = productionRequiredStringLabel(labels, "autoround_calibration_decision_profile", &missing)
	decision.FloatFormat = productionRequiredStringLabel(labels, "autoround_calibration_decision_float_format", &missing)
	decision.HIPKernel = productionRequiredStringLabel(labels, "autoround_calibration_decision_hip_kernel", &missing)
	var err error
	if decision.CalibrationCandidate, err = productionRequiredBoolLabel(labels, "autoround_calibration_candidate", &missing); err != nil {
		return err
	}
	if decision.RequiresBench, err = productionRequiredBoolLabel(labels, "autoround_calibration_decision_requires_bench", &missing); err != nil {
		return err
	}
	if len(missing) > 0 {
		return core.E(name, "missing required production metric labels: "+strings.Join(missing, ","), nil)
	}
	return nil
}

func productionRequiredStringLabel(labels map[string]string, key string, missing *[]string) string {
	value, ok := labels[key]
	if !ok {
		*missing = append(*missing, key)
	}
	return value
}

func productionRequiredIntLabel(labels map[string]string, key string, missing *[]string) (int, error) {
	value, ok := labels[key]
	if !ok {
		*missing = append(*missing, key)
		return 0, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return 0, core.E("rocm.ApplyProductionAutoRoundCalibrationLabelEvidence", "parse "+key, err)
	}
	return parsed, nil
}

func productionRequiredBoolLabel(labels map[string]string, key string, missing *[]string) (bool, error) {
	value, ok := labels[key]
	if !ok {
		*missing = append(*missing, key)
		return false, nil
	}
	switch value {
	case "true":
		return true, nil
	case "false":
		return false, nil
	}
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "true", "1", "yes":
		return true, nil
	case "false", "0", "no":
		return false, nil
	default:
		return false, core.E("rocm.ApplyProductionAutoRoundCalibrationLabelEvidence", "parse "+key, nil)
	}
}

func ValidateProductionBookGateMetricLabels(labels map[string]string) error {
	if err := validateProductionRequiredMetricLabels("rocm.ValidateProductionBookGateMetricLabels", labels, productionBookGateMetrics, nil); err != nil {
		return err
	}
	for _, metric := range productionBookGateMetrics {
		value := strings.TrimSpace(labels[metric])
		parsed, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return core.E("rocm.ValidateProductionBookGateMetricLabels", "parse "+metric, err)
		}
		if !productionBookGateFinite(parsed) {
			return core.E("rocm.ValidateProductionBookGateMetricLabels", metric+" must be finite", nil)
		}
	}
	return nil
}

func ValidateProductionBookRetainedArtifactDecisionLabels(labels map[string]string) error {
	_, err := EvaluateProductionBookRetainedArtifactDecisionLabels(labels)
	return err
}

func ValidateProductionMTPPromotionDecisionLabels(labels map[string]string) error {
	_, err := EvaluateProductionMTPPromotionDecisionLabels(labels)
	return err
}

func ValidateProductionTurboQuantPromotionDecisionLabels(labels map[string]string) error {
	_, err := EvaluateProductionTurboQuantPromotionDecisionLabels(labels)
	return err
}

func ValidateProductionCombinedMTPAndTurboQuantDecisionLabels(labels map[string]string) error {
	_, err := EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels(labels)
	return err
}

type ProductionBookGateReasonCode int

const (
	ProductionBookGateReasonPass ProductionBookGateReasonCode = iota
	ProductionBookGateReasonQuant
	ProductionBookGateReasonMetrics
	ProductionBookGateReasonTurns
	ProductionBookGateReasonWall
	ProductionBookGateReasonDecode
	ProductionBookGateReasonQuality
)

type ProductionBookGateMetricDecision struct {
	ProductionCandidate   bool
	Reason                string
	ReasonCode            ProductionBookGateReasonCode
	QuantAccepted         bool
	TurnsAccepted         bool
	WallAccepted          bool
	DecodeAccepted        bool
	QualityAccepted       bool
	RawDecodeTokensPerSec float64
	WallSeconds           float64
	QualityFlags          int
}

type ProductionBookRetainedArtifactDecision struct {
	RetainedRoute bool
	Gate          ProductionBookGateMetricDecision
}

func EvaluateProductionBookGateMetricLabels(labels map[string]string) (ProductionBookGateMetricDecision, error) {
	return EvaluateProductionBookGateMetricLabelsWithPolicy(defaultProductionBookGatePolicy(), labels)
}

func EvaluateProductionBookGateMetricLabelsWithPolicy(policy ProductionBookGatePolicy, labels map[string]string) (ProductionBookGateMetricDecision, error) {
	if err := ValidateProductionBookGateMetricLabels(labels); err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	candidate, err := productionBookGateBoolMetric(labels, "production_book_gate_candidate")
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	reasonCode, err := productionBookGateReasonCodeMetric(labels)
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	quant, err := productionBookGateBoolMetric(labels, "production_book_gate_q6")
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	turns, err := productionBookGateBoolMetric(labels, "production_book_gate_turns")
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	wall, err := productionBookGateBoolMetric(labels, "production_book_gate_wall")
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	decode, err := productionBookGateBoolMetric(labels, "production_book_gate_decode")
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	quality, err := productionBookGateBoolMetric(labels, "production_book_gate_quality")
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	rawDecode, err := productionBookGateFloatMetric(labels, "production_book_gate_raw_decode_tok/s")
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	wallSeconds, err := productionBookGateFloatMetric(labels, "production_book_gate_wall_s")
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	qualityFlags, err := productionBookGateIntMetric(labels, "production_book_gate_quality_flags")
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	decision := ProductionBookGateMetricDecision{
		ProductionCandidate:   candidate,
		ReasonCode:            reasonCode,
		QuantAccepted:         quant,
		TurnsAccepted:         turns,
		WallAccepted:          wall,
		DecodeAccepted:        decode,
		QualityAccepted:       quality,
		RawDecodeTokensPerSec: rawDecode,
		WallSeconds:           wallSeconds,
		QualityFlags:          qualityFlags,
	}
	if err := decision.validateProductionBookGateMetricDecision(policy); err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	decision.Reason = productionBookGateMetricDecisionReason(policy, decision)
	return decision, nil
}

func ValidateProductionBookGateMetrics(metrics map[string]float64) error {
	if metrics == nil {
		return core.E("rocm.ValidateProductionBookGateMetrics", "metrics are required", nil)
	}
	for _, metric := range productionBookGateMetrics {
		value, ok := metrics[metric]
		if !ok {
			return core.E("rocm.ValidateProductionBookGateMetrics", "missing production book gate metric "+metric, nil)
		}
		if !productionBookGateFinite(value) {
			return core.E("rocm.ValidateProductionBookGateMetrics", metric+" must be finite", nil)
		}
	}
	return nil
}

func ValidateProductionBookRetainedRouteMetrics(metrics map[string]float64) error {
	if metrics == nil {
		return core.E("rocm.ValidateProductionBookRetainedRouteMetrics", "metrics are required", nil)
	}
	for _, metric := range productionBookRetainedRouteMetrics {
		value, ok := metrics[metric]
		if !ok {
			return core.E("rocm.ValidateProductionBookRetainedRouteMetrics", "missing production book retained-route metric "+metric, nil)
		}
		if !productionBookGateFinite(value) {
			return core.E("rocm.ValidateProductionBookRetainedRouteMetrics", metric+" must be finite", nil)
		}
		accepted, err := productionBookGateBool(metric, value)
		if err != nil {
			return core.E("rocm.ValidateProductionBookRetainedRouteMetrics", "parse "+metric, err)
		}
		if metric == "book_replay_baseline" {
			if accepted {
				return core.E("rocm.ValidateProductionBookRetainedRouteMetrics", "book_replay_baseline must be 0 for retained-state production artifacts", nil)
			}
			continue
		}
		if !accepted {
			return core.E("rocm.ValidateProductionBookRetainedRouteMetrics", metric+" must be 1 for retained-state production artifacts", nil)
		}
	}
	return nil
}

func EvaluateProductionBookGateMetrics(metrics map[string]float64) (ProductionBookGateMetricDecision, error) {
	return EvaluateProductionBookGateMetricsWithPolicy(defaultProductionBookGatePolicy(), metrics)
}

func EvaluateProductionBookRetainedArtifactMetrics(metrics map[string]float64) (ProductionBookRetainedArtifactDecision, error) {
	return EvaluateProductionBookRetainedArtifactMetricsWithPolicy(defaultProductionBookGatePolicy(), metrics)
}

func EvaluateProductionBookRetainedArtifactDecisionLabels(labels map[string]string) (ProductionBookRetainedArtifactDecision, error) {
	return EvaluateProductionBookRetainedArtifactDecisionLabelsWithPolicy(defaultProductionBookGatePolicy(), labels)
}

func EvaluateProductionBookRetainedArtifactDecisionLabelsWithPolicy(policy ProductionBookGatePolicy, labels map[string]string) (ProductionBookRetainedArtifactDecision, error) {
	if err := validateProductionRequiredMetricLabels("rocm.EvaluateProductionBookRetainedArtifactDecisionLabels", labels, productionBookRetainedArtifactLabels, nil); err != nil {
		return ProductionBookRetainedArtifactDecision{}, err
	}
	candidate, err := productionBoolLabel(labels, "production_book_retained_artifact_candidate")
	if err != nil {
		return ProductionBookRetainedArtifactDecision{}, err
	}
	retainedRoute, err := productionBoolLabel(labels, "production_book_retained_artifact_retained_route")
	if err != nil {
		return ProductionBookRetainedArtifactDecision{}, err
	}
	if !retainedRoute {
		return ProductionBookRetainedArtifactDecision{}, core.E("rocm.EvaluateProductionBookRetainedArtifactDecisionLabels", "production_book_retained_artifact_retained_route must be true", nil)
	}
	reason := strings.TrimSpace(labels["production_book_retained_artifact_reason"])
	if reason == "" {
		return ProductionBookRetainedArtifactDecision{}, core.E("rocm.EvaluateProductionBookRetainedArtifactDecisionLabels", "production_book_retained_artifact_reason is required", nil)
	}
	gateCandidate, err := productionBoolLabel(labels, "production_book_retained_artifact_gate_candidate")
	if err != nil {
		return ProductionBookRetainedArtifactDecision{}, err
	}
	reasonCode, err := productionBookRetainedArtifactReasonCodeLabel(labels)
	if err != nil {
		return ProductionBookRetainedArtifactDecision{}, err
	}
	quant, err := productionBoolLabel(labels, "production_book_retained_artifact_gate_q6")
	if err != nil {
		return ProductionBookRetainedArtifactDecision{}, err
	}
	turns, err := productionBoolLabel(labels, "production_book_retained_artifact_gate_turns")
	if err != nil {
		return ProductionBookRetainedArtifactDecision{}, err
	}
	wall, err := productionBoolLabel(labels, "production_book_retained_artifact_gate_wall")
	if err != nil {
		return ProductionBookRetainedArtifactDecision{}, err
	}
	decode, err := productionBoolLabel(labels, "production_book_retained_artifact_gate_decode")
	if err != nil {
		return ProductionBookRetainedArtifactDecision{}, err
	}
	quality, err := productionBoolLabel(labels, "production_book_retained_artifact_gate_quality")
	if err != nil {
		return ProductionBookRetainedArtifactDecision{}, err
	}
	rawDecode, err := productionFloatLabel(labels, "production_book_retained_artifact_raw_decode_tok/s")
	if err != nil {
		return ProductionBookRetainedArtifactDecision{}, err
	}
	wallSeconds, err := productionFloatLabel(labels, "production_book_retained_artifact_wall_s")
	if err != nil {
		return ProductionBookRetainedArtifactDecision{}, err
	}
	qualityFlags, err := productionIntLabel(labels, "production_book_retained_artifact_quality_flags")
	if err != nil {
		return ProductionBookRetainedArtifactDecision{}, err
	}
	decision := ProductionBookRetainedArtifactDecision{
		RetainedRoute: true,
		Gate: ProductionBookGateMetricDecision{
			ProductionCandidate:   gateCandidate,
			ReasonCode:            reasonCode,
			QuantAccepted:         quant,
			TurnsAccepted:         turns,
			WallAccepted:          wall,
			DecodeAccepted:        decode,
			QualityAccepted:       quality,
			RawDecodeTokensPerSec: rawDecode,
			WallSeconds:           wallSeconds,
			QualityFlags:          qualityFlags,
		},
	}
	if err := decision.Gate.validateProductionBookGateMetricDecision(policy); err != nil {
		return ProductionBookRetainedArtifactDecision{}, err
	}
	decision.Gate.Reason = productionBookGateMetricDecisionReason(policy, decision.Gate)
	if candidate != (decision.RetainedRoute && decision.Gate.ProductionCandidate) {
		return ProductionBookRetainedArtifactDecision{}, core.E("rocm.EvaluateProductionBookRetainedArtifactDecisionLabels", "production_book_retained_artifact_candidate is inconsistent with route and gate candidate", nil)
	}
	if reason != productionBookRetainedArtifactDecisionReason(decision) {
		return ProductionBookRetainedArtifactDecision{}, core.E("rocm.EvaluateProductionBookRetainedArtifactDecisionLabels", "production_book_retained_artifact_reason is inconsistent with gate result", nil)
	}
	return decision, nil
}

func EvaluateProductionMTPPromotionDecisionLabels(labels map[string]string) (ProductionMTPPromotionDecision, error) {
	if err := validateProductionRequiredMetricLabels("rocm.EvaluateProductionMTPPromotionDecisionLabels", labels, productionMTPPromotionDecisionLabels, nil); err != nil {
		return ProductionMTPPromotionDecision{}, err
	}
	enabled, err := productionDecisionBoolLabel("rocm.EvaluateProductionMTPPromotionDecisionLabels", labels, "production_mtp_enable_by_default")
	if err != nil {
		return ProductionMTPPromotionDecision{}, err
	}
	reason := strings.TrimSpace(labels["production_mtp_reason"])
	if reason == "" {
		return ProductionMTPPromotionDecision{}, core.E("rocm.EvaluateProductionMTPPromotionDecisionLabels", "production_mtp_reason is required", nil)
	}
	wallSpeedup, err := productionDecisionNonNegativeFloatLabel("rocm.EvaluateProductionMTPPromotionDecisionLabels", labels, "production_mtp_wall_speedup")
	if err != nil {
		return ProductionMTPPromotionDecision{}, err
	}
	visibleSpeedup, err := productionDecisionNonNegativeFloatLabel("rocm.EvaluateProductionMTPPromotionDecisionLabels", labels, "production_mtp_visible_speedup")
	if err != nil {
		return ProductionMTPPromotionDecision{}, err
	}
	restoreSpeedup, err := productionDecisionNonNegativeFloatLabel("rocm.EvaluateProductionMTPPromotionDecisionLabels", labels, "production_mtp_restore_speedup")
	if err != nil {
		return ProductionMTPPromotionDecision{}, err
	}
	energySavings, err := productionDecisionNonNegativeFloatLabel("rocm.EvaluateProductionMTPPromotionDecisionLabels", labels, "production_mtp_energy_savings")
	if err != nil {
		return ProductionMTPPromotionDecision{}, err
	}
	acceptanceRate, err := productionDecisionNonNegativeFloatLabel("rocm.EvaluateProductionMTPPromotionDecisionLabels", labels, "production_mtp_acceptance_rate")
	if err != nil {
		return ProductionMTPPromotionDecision{}, err
	}
	return ProductionMTPPromotionDecision{
		EnableByDefault: enabled,
		Reason:          reason,
		WallSpeedup:     wallSpeedup,
		VisibleSpeedup:  visibleSpeedup,
		RestoreSpeedup:  restoreSpeedup,
		EnergySavings:   energySavings,
		AcceptanceRate:  acceptanceRate,
	}, nil
}

func EvaluateProductionTurboQuantPromotionDecisionLabels(labels map[string]string) (ProductionTurboQuantPromotionDecision, error) {
	if err := validateProductionRequiredMetricLabels("rocm.EvaluateProductionTurboQuantPromotionDecisionLabels", labels, productionTurboQuantPromotionDecisionLabels, nil); err != nil {
		return ProductionTurboQuantPromotionDecision{}, err
	}
	candidate, err := productionDecisionBoolLabel("rocm.EvaluateProductionTurboQuantPromotionDecisionLabels", labels, "production_turboquant_candidate")
	if err != nil {
		return ProductionTurboQuantPromotionDecision{}, err
	}
	enabled, err := productionDecisionBoolLabel("rocm.EvaluateProductionTurboQuantPromotionDecisionLabels", labels, "production_turboquant_enable_by_default")
	if err != nil {
		return ProductionTurboQuantPromotionDecision{}, err
	}
	reason := strings.TrimSpace(labels["production_turboquant_reason"])
	if reason == "" {
		return ProductionTurboQuantPromotionDecision{}, core.E("rocm.EvaluateProductionTurboQuantPromotionDecisionLabels", "production_turboquant_reason is required", nil)
	}
	wallSpeedup, err := productionDecisionNonNegativeFloatLabel("rocm.EvaluateProductionTurboQuantPromotionDecisionLabels", labels, "production_turboquant_wall_speedup")
	if err != nil {
		return ProductionTurboQuantPromotionDecision{}, err
	}
	visibleSpeedup, err := productionDecisionNonNegativeFloatLabel("rocm.EvaluateProductionTurboQuantPromotionDecisionLabels", labels, "production_turboquant_visible_speedup")
	if err != nil {
		return ProductionTurboQuantPromotionDecision{}, err
	}
	restoreSpeedup, err := productionDecisionNonNegativeFloatLabel("rocm.EvaluateProductionTurboQuantPromotionDecisionLabels", labels, "production_turboquant_restore_speedup")
	if err != nil {
		return ProductionTurboQuantPromotionDecision{}, err
	}
	memorySavings, err := productionDecisionNonNegativeFloatLabel("rocm.EvaluateProductionTurboQuantPromotionDecisionLabels", labels, "production_turboquant_memory_savings_ratio")
	if err != nil {
		return ProductionTurboQuantPromotionDecision{}, err
	}
	energySavings, err := productionDecisionNonNegativeFloatLabel("rocm.EvaluateProductionTurboQuantPromotionDecisionLabels", labels, "production_turboquant_energy_savings_ratio")
	if err != nil {
		return ProductionTurboQuantPromotionDecision{}, err
	}
	return ProductionTurboQuantPromotionDecision{
		ProductionCandidate: candidate,
		EnableByDefault:     enabled,
		Reason:              reason,
		WallSpeedup:         wallSpeedup,
		VisibleSpeedup:      visibleSpeedup,
		RestoreSpeedup:      restoreSpeedup,
		MemorySavingsRatio:  memorySavings,
		EnergySavingsRatio:  energySavings,
	}, nil
}

func EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels(labels map[string]string) (ProductionCombinedMTPAndTurboQuantDecision, error) {
	if err := validateProductionRequiredMetricLabels("rocm.EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels", labels, productionCombinedMTPAndTurboQuantDecisionLabels, nil); err != nil {
		return ProductionCombinedMTPAndTurboQuantDecision{}, err
	}
	candidate, err := productionDecisionBoolLabel("rocm.EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels", labels, "production_combined_candidate")
	if err != nil {
		return ProductionCombinedMTPAndTurboQuantDecision{}, err
	}
	enabled, err := productionDecisionBoolLabel("rocm.EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels", labels, "production_combined_enable_by_default")
	if err != nil {
		return ProductionCombinedMTPAndTurboQuantDecision{}, err
	}
	reason := strings.TrimSpace(labels["production_combined_reason"])
	if reason == "" {
		return ProductionCombinedMTPAndTurboQuantDecision{}, core.E("rocm.EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels", "production_combined_reason is required", nil)
	}
	mtpEligible, err := productionDecisionBoolLabel("rocm.EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels", labels, "production_combined_mtp_eligible")
	if err != nil {
		return ProductionCombinedMTPAndTurboQuantDecision{}, err
	}
	turboEligible, err := productionDecisionBoolLabel("rocm.EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels", labels, "production_combined_turboquant_eligible")
	if err != nil {
		return ProductionCombinedMTPAndTurboQuantDecision{}, err
	}
	if candidate && (!mtpEligible || !turboEligible) {
		return ProductionCombinedMTPAndTurboQuantDecision{}, core.E("rocm.EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels", "production_combined_candidate requires both component lanes to be eligible", nil)
	}
	mtpWallSpeedup, err := productionDecisionNonNegativeFloatLabel("rocm.EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels", labels, "production_combined_mtp_wall_speedup")
	if err != nil {
		return ProductionCombinedMTPAndTurboQuantDecision{}, err
	}
	mtpVisibleSpeedup, err := productionDecisionNonNegativeFloatLabel("rocm.EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels", labels, "production_combined_mtp_visible_speedup")
	if err != nil {
		return ProductionCombinedMTPAndTurboQuantDecision{}, err
	}
	mtpAcceptanceRate, err := productionDecisionNonNegativeFloatLabel("rocm.EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels", labels, "production_combined_mtp_acceptance_rate")
	if err != nil {
		return ProductionCombinedMTPAndTurboQuantDecision{}, err
	}
	turboMemorySavings, err := productionDecisionNonNegativeFloatLabel("rocm.EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels", labels, "production_combined_turboquant_memory_savings_ratio")
	if err != nil {
		return ProductionCombinedMTPAndTurboQuantDecision{}, err
	}
	turboEnergySavings, err := productionDecisionNonNegativeFloatLabel("rocm.EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels", labels, "production_combined_turboquant_energy_savings_ratio")
	if err != nil {
		return ProductionCombinedMTPAndTurboQuantDecision{}, err
	}
	return ProductionCombinedMTPAndTurboQuantDecision{
		ProductionCandidate:          candidate,
		EnableByDefault:              enabled,
		Reason:                       reason,
		MTPEligible:                  mtpEligible,
		TurboQuantEligible:           turboEligible,
		MTPWallSpeedup:               mtpWallSpeedup,
		MTPVisibleSpeedup:            mtpVisibleSpeedup,
		MTPAcceptanceRate:            mtpAcceptanceRate,
		TurboQuantMemorySavingsRatio: turboMemorySavings,
		TurboQuantEnergySavingsRatio: turboEnergySavings,
	}, nil
}

func EvaluateProductionBookRetainedArtifactMetricsWithPolicy(policy ProductionBookGatePolicy, metrics map[string]float64) (ProductionBookRetainedArtifactDecision, error) {
	if err := ValidateProductionBookRetainedRouteMetrics(metrics); err != nil {
		return ProductionBookRetainedArtifactDecision{}, err
	}
	gate, err := EvaluateProductionBookGateMetricsWithPolicy(policy, metrics)
	if err != nil {
		return ProductionBookRetainedArtifactDecision{}, err
	}
	return ProductionBookRetainedArtifactDecision{
		RetainedRoute: true,
		Gate:          gate,
	}, nil
}

func EvaluateProductionBookGateMetricsWithPolicy(policy ProductionBookGatePolicy, metrics map[string]float64) (ProductionBookGateMetricDecision, error) {
	if err := ValidateProductionBookGateMetrics(metrics); err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	candidate, err := productionBookGateBoolValue(metrics, "production_book_gate_candidate")
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	reasonCode, err := productionBookGateReasonCodeValue(metrics)
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	quant, err := productionBookGateBoolValue(metrics, "production_book_gate_q6")
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	turns, err := productionBookGateBoolValue(metrics, "production_book_gate_turns")
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	wall, err := productionBookGateBoolValue(metrics, "production_book_gate_wall")
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	decode, err := productionBookGateBoolValue(metrics, "production_book_gate_decode")
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	quality, err := productionBookGateBoolValue(metrics, "production_book_gate_quality")
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	qualityFlags, err := productionBookGateIntValue(metrics, "production_book_gate_quality_flags")
	if err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	decision := ProductionBookGateMetricDecision{
		ProductionCandidate:   candidate,
		ReasonCode:            reasonCode,
		QuantAccepted:         quant,
		TurnsAccepted:         turns,
		WallAccepted:          wall,
		DecodeAccepted:        decode,
		QualityAccepted:       quality,
		RawDecodeTokensPerSec: metrics["production_book_gate_raw_decode_tok/s"],
		WallSeconds:           metrics["production_book_gate_wall_s"],
		QualityFlags:          qualityFlags,
	}
	if err := decision.validateProductionBookGateMetricDecision(policy); err != nil {
		return ProductionBookGateMetricDecision{}, err
	}
	decision.Reason = productionBookGateMetricDecisionReason(policy, decision)
	return decision, nil
}

func ProductionBookGateMetricLabels(metrics map[string]float64) (map[string]string, error) {
	return AddProductionBookGateMetricLabels(make(map[string]string, len(productionBookGateMetrics)), metrics)
}

func AddProductionBookGateMetricLabels(labels map[string]string, metrics map[string]float64) (map[string]string, error) {
	if labels == nil {
		labels = make(map[string]string, len(productionBookGateMetrics))
	}
	if metrics == nil {
		return labels, core.E("rocm.AddProductionBookGateMetricLabels", "metrics are required", nil)
	}
	for _, metric := range productionBookGateMetrics {
		value, ok := metrics[metric]
		if !ok {
			return labels, core.E("rocm.AddProductionBookGateMetricLabels", "missing production book gate metric "+metric, nil)
		}
		labels[metric] = productionBookGateMetricLabel(metric, value)
	}
	return labels, nil
}

func ProductionBookRetainedArtifactDecisionLabels(decision ProductionBookRetainedArtifactDecision) map[string]string {
	return AddProductionBookRetainedArtifactDecisionLabels(make(map[string]string, 13), decision)
}

func AddProductionBookRetainedArtifactDecisionLabels(labels map[string]string, decision ProductionBookRetainedArtifactDecision) map[string]string {
	if labels == nil {
		labels = make(map[string]string, 13)
	}
	labels["production_book_retained_artifact_candidate"] = strconv.FormatBool(decision.RetainedRoute && decision.Gate.ProductionCandidate)
	labels["production_book_retained_artifact_retained_route"] = strconv.FormatBool(decision.RetainedRoute)
	labels["production_book_retained_artifact_reason"] = productionBookRetainedArtifactDecisionReason(decision)
	labels["production_book_retained_artifact_gate_candidate"] = strconv.FormatBool(decision.Gate.ProductionCandidate)
	labels["production_book_retained_artifact_gate_reason_code"] = strconv.Itoa(int(decision.Gate.ReasonCode))
	labels["production_book_retained_artifact_gate_q6"] = strconv.FormatBool(decision.Gate.QuantAccepted)
	labels["production_book_retained_artifact_gate_turns"] = strconv.FormatBool(decision.Gate.TurnsAccepted)
	labels["production_book_retained_artifact_gate_wall"] = strconv.FormatBool(decision.Gate.WallAccepted)
	labels["production_book_retained_artifact_gate_decode"] = strconv.FormatBool(decision.Gate.DecodeAccepted)
	labels["production_book_retained_artifact_gate_quality"] = strconv.FormatBool(decision.Gate.QualityAccepted)
	labels["production_book_retained_artifact_raw_decode_tok/s"] = productionMetricFloatLabel(decision.Gate.RawDecodeTokensPerSec)
	labels["production_book_retained_artifact_wall_s"] = productionMetricFloatLabel(decision.Gate.WallSeconds)
	labels["production_book_retained_artifact_quality_flags"] = strconv.Itoa(decision.Gate.QualityFlags)
	return labels
}

func ProductionBookRetainedArtifactMetricDecisionLabels(metrics map[string]float64) (map[string]string, error) {
	return AddProductionBookRetainedArtifactMetricDecisionLabels(make(map[string]string, 13), metrics)
}

func AddProductionBookRetainedArtifactMetricDecisionLabels(labels map[string]string, metrics map[string]float64) (map[string]string, error) {
	if labels == nil {
		labels = make(map[string]string, 13)
	}
	decision, err := EvaluateProductionBookRetainedArtifactMetrics(metrics)
	if err != nil {
		return labels, err
	}
	AddProductionBookRetainedArtifactDecisionLabels(labels, decision)
	return labels, nil
}

func ProductionMTPPromotionDecisionLabels(decision ProductionMTPPromotionDecision) map[string]string {
	return AddProductionMTPPromotionDecisionLabels(make(map[string]string, 8), decision)
}

func AddProductionMTPPromotionDecisionLabels(labels map[string]string, decision ProductionMTPPromotionDecision) map[string]string {
	if labels == nil {
		labels = make(map[string]string, 8)
	}
	labels["production_mtp_enable_by_default"] = strconv.FormatBool(decision.EnableByDefault)
	labels["production_mtp_reason"] = decision.Reason
	labels["production_mtp_wall_speedup"] = productionMetricFloatLabel(decision.WallSpeedup)
	labels["production_mtp_visible_speedup"] = productionMetricFloatLabel(decision.VisibleSpeedup)
	labels["production_mtp_restore_speedup"] = productionMetricFloatLabel(decision.RestoreSpeedup)
	labels["production_mtp_energy_savings"] = productionMetricFloatLabel(decision.EnergySavings)
	labels["production_mtp_acceptance_rate"] = productionMetricFloatLabel(decision.AcceptanceRate)
	return labels
}

func ProductionTurboQuantPromotionDecisionLabels(decision ProductionTurboQuantPromotionDecision) map[string]string {
	return AddProductionTurboQuantPromotionDecisionLabels(make(map[string]string, 8), decision)
}

func AddProductionTurboQuantPromotionDecisionLabels(labels map[string]string, decision ProductionTurboQuantPromotionDecision) map[string]string {
	if labels == nil {
		labels = make(map[string]string, 8)
	}
	labels["production_turboquant_candidate"] = strconv.FormatBool(decision.ProductionCandidate)
	labels["production_turboquant_enable_by_default"] = strconv.FormatBool(decision.EnableByDefault)
	labels["production_turboquant_reason"] = decision.Reason
	labels["production_turboquant_wall_speedup"] = productionMetricFloatLabel(decision.WallSpeedup)
	labels["production_turboquant_visible_speedup"] = productionMetricFloatLabel(decision.VisibleSpeedup)
	labels["production_turboquant_restore_speedup"] = productionMetricFloatLabel(decision.RestoreSpeedup)
	labels["production_turboquant_memory_savings_ratio"] = productionMetricFloatLabel(decision.MemorySavingsRatio)
	labels["production_turboquant_energy_savings_ratio"] = productionMetricFloatLabel(decision.EnergySavingsRatio)
	return labels
}

func ProductionCombinedMTPAndTurboQuantDecisionLabels(decision ProductionCombinedMTPAndTurboQuantDecision) map[string]string {
	return AddProductionCombinedMTPAndTurboQuantDecisionLabels(make(map[string]string, 10), decision)
}

func AddProductionCombinedMTPAndTurboQuantDecisionLabels(labels map[string]string, decision ProductionCombinedMTPAndTurboQuantDecision) map[string]string {
	if labels == nil {
		labels = make(map[string]string, 10)
	}
	labels["production_combined_candidate"] = strconv.FormatBool(decision.ProductionCandidate)
	labels["production_combined_enable_by_default"] = strconv.FormatBool(decision.EnableByDefault)
	labels["production_combined_reason"] = decision.Reason
	labels["production_combined_mtp_eligible"] = strconv.FormatBool(decision.MTPEligible)
	labels["production_combined_turboquant_eligible"] = strconv.FormatBool(decision.TurboQuantEligible)
	labels["production_combined_mtp_wall_speedup"] = productionMetricFloatLabel(decision.MTPWallSpeedup)
	labels["production_combined_mtp_visible_speedup"] = productionMetricFloatLabel(decision.MTPVisibleSpeedup)
	labels["production_combined_mtp_acceptance_rate"] = productionMetricFloatLabel(decision.MTPAcceptanceRate)
	labels["production_combined_turboquant_memory_savings_ratio"] = productionMetricFloatLabel(decision.TurboQuantMemorySavingsRatio)
	labels["production_combined_turboquant_energy_savings_ratio"] = productionMetricFloatLabel(decision.TurboQuantEnergySavingsRatio)
	return labels
}

var productionMTPRequiredMetricAliases = map[string][]string{
	"retained_workflow":                          {"mtp_retained_workflow"},
	"turns":                                      {"mtp_turns"},
	"greedy_output_matches":                      {"mtp_greedy_output_matches"},
	"speculative_draft_model_path":               {"attached_drafter_assistant_model_id", "attached.drafter.assistant.model_id", "attached_drafter_official_assistant_model_id", "attached.drafter.official_assistant_model_id"},
	"speculative_draft_tokens":                   {"attached_drafter_speculative_draft_tokens", "attached.drafter.speculative_draft_tokens"},
	"target_only_visible_tokens_per_sec":         {"mtp_target_only_visible_tokens_per_sec"},
	"target_only_wall_duration":                  {"mtp_target_only_wall_duration"},
	"target_only_restore_duration":               {"mtp_target_only_restore_duration"},
	"target_only_peak_memory_bytes":              {"mtp_target_only_peak_memory_bytes"},
	"target_only_active_plus_cache_memory_bytes": {"mtp_target_only_active_plus_cache_memory_bytes"},
	"target_only_energy_joules":                  {"mtp_target_only_energy_joules"},
	"same_load_policy":                           {"mtp_same_load_policy"},
	"target_only_cache_mode":                     {"mtp_target_only_cache_mode"},
	"attached_drafter_target_gemma4_size":        {"target_gemma4_size", "attached.drafter.target.gemma4_size"},
	"attached_drafter_target_gemma4_quant_mode":  {"target_gemma4_quant_mode", "attached.drafter.target.gemma4_quant_mode"},
	"attached_drafter_target_gemma4_quant_group": {"target_gemma4_quant_group", "attached.drafter.target.gemma4_quant_group"},
	"attached_drafter_target_gemma4_runtime":     {"target_gemma4_runtime", "attached.drafter.target.gemma4_runtime"},
	"attached_drafter_target_gemma4_generate_status": {
		"target_gemma4_generate_status",
		"attached.drafter.target.gemma4_generate_status",
	},
	"attached_drafter_target_production_quant_model": {"target_production_quant_model", "attached.drafter.target.production_quant_model"},
	"attached_drafter_assistant_gemma4_size":         {"assistant_gemma4_size", "draft_gemma4_size", "attached_drafter_draft_gemma4_size", "attached.drafter.assistant.gemma4_size", "attached.drafter.draft.gemma4_size"},
	"attached_drafter_assistant_gemma4_quant_mode":   {"assistant_gemma4_quant_mode", "draft_gemma4_quant_mode", "attached_drafter_draft_gemma4_quant_mode", "attached.drafter.assistant.gemma4_quant_mode", "attached.drafter.draft.gemma4_quant_mode"},
	"attached_drafter_assistant_gemma4_quant_group": {
		"assistant_gemma4_quant_group",
		"draft_gemma4_quant_group",
		"attached_drafter_draft_gemma4_quant_group",
		"attached.drafter.assistant.gemma4_quant_group",
		"attached.drafter.draft.gemma4_quant_group",
	},
	"attached_drafter_assistant_gemma4_runtime": {"assistant_gemma4_runtime", "draft_gemma4_runtime", "attached_drafter_draft_gemma4_runtime", "attached.drafter.assistant.gemma4_runtime", "attached.drafter.draft.gemma4_runtime"},
	"attached_drafter_assistant_gemma4_generate_status": {
		"assistant_gemma4_generate_status",
		"draft_gemma4_generate_status",
		"attached_drafter_draft_gemma4_generate_status",
		"attached.drafter.assistant.gemma4_generate_status",
		"attached.drafter.draft.gemma4_generate_status",
	},
	"attached_drafter_assistant_production_quant_model":         {"assistant_production_quant_model", "assistant_production_quant_assistant_model", "draft_production_quant_model", "attached_drafter_assistant_production_quant_assistant_model", "attached_drafter_draft_production_quant_model", "attached.drafter.assistant.production_quant_model", "attached.drafter.assistant.production_quant_assistant_model", "attached.drafter.draft.production_quant_model"},
	"attached_drafter_assistant_production_quant_pack":          {"assistant_production_quant_pack", "draft_production_quant_pack", "attached_drafter_draft_production_quant_pack", "attached.drafter.assistant.production_quant_pack", "attached.drafter.draft.production_quant_pack"},
	"attached_drafter_assistant_production_quant_tier":          {"assistant_production_quant_tier", "draft_production_quant_tier", "attached_drafter_draft_production_quant_tier", "attached.drafter.assistant.production_quant_tier", "attached.drafter.draft.production_quant_tier"},
	"attached_drafter_assistant_production_quant_mtp_assistant": {"assistant_production_quant_mtp_assistant", "draft_production_quant_mtp_assistant", "attached_drafter_draft_production_quant_mtp_assistant", "attached.drafter.assistant.production_quant_mtp_assistant", "attached.drafter.draft.production_quant_mtp_assistant"},
	"assistant_architecture":                                    {"attached_drafter_assistant_architecture", "attached.drafter.assistant_architecture"},
	"assistant_ordered_embeddings":                              {"attached_drafter_assistant_ordered_embeddings", "attached.drafter.assistant_ordered_embeddings"},
	"assistant_centroids":                                       {"attached_drafter_assistant_centroids", "attached.drafter.assistant_centroids"},
	"assistant_centroid_intermediate_top_k":                     {"attached_drafter_assistant_centroid_intermediate_top_k", "attached.drafter.assistant_centroid_intermediate_top_k"},
	"assistant_four_layer_drafter":                              {"attached_drafter_assistant_four_layer_drafter", "attached.drafter.assistant_four_layer_drafter"},
	"assistant_token_ordering_dtype":                            {"attached_drafter_assistant_token_ordering_dtype", "attached.drafter.assistant_token_ordering_dtype"},
	"assistant_token_ordering_shape":                            {"attached_drafter_assistant_token_ordering_shape", "attached.drafter.assistant_token_ordering_shape"},
	"gemma4_family_pair_verified":                               {"attached_drafter_gemma4_family_pair_verified", "attached.drafter.gemma4_family_pair_verified"},
	"official_pair_verified":                                    {"attached_drafter_official_pair_verified", "attached.drafter.official_pair_verified"},
	"official_target_model_id":                                  {"attached_drafter_official_target_model_id", "attached.drafter.official_target_model_id"},
	"official_target_revision":                                  {"attached_drafter_official_target_revision", "attached.drafter.official_target_revision"},
	"official_assistant_model_id":                               {"attached_drafter_official_assistant_model_id", "attached.drafter.official_assistant_model_id"},
	"official_assistant_revision":                               {"attached_drafter_official_assistant_revision", "attached.drafter.official_assistant_revision"},
}

var productionTurboQuantRequiredMetricAliases = map[string][]string{
	"candidate_cache_mode":                     {"turboquant_candidate_cache_mode", "kv_compression", "production_candidate_cache_mode"},
	"candidate_layout_version":                 {"turboquant_candidate_layout_version", "production_required_layout_version"},
	"candidate_key_algorithm":                  {"turboquant_candidate_key_algorithm", "production_required_key_algorithm"},
	"candidate_value_algorithm":                {"turboquant_candidate_value_algorithm", "production_required_value_algorithm"},
	"candidate_outlier_policy":                 {"turboquant_candidate_outlier_policy", "production_required_outlier_policy"},
	"candidate_effective_bits_milli":           {"turboquant_candidate_effective_bits_milli", "production_target_effective_bits_milli"},
	"candidate_qjl_residual":                   {"turboquant_candidate_qjl_residual"},
	"candidate_metadata_bytes":                 {"turboquant_candidate_metadata_bytes"},
	"candidate_cache_policy":                   {"turboquant_candidate_cache_policy"},
	"normal_context_validated":                 {"turboquant_normal_context_validated"},
	"stress_context_validated":                 {"turboquant_stress_context_validated"},
	"quality_flags":                            {"turboquant_quality_flags"},
	"candidate_peak_memory_bytes":              {"turboquant_candidate_peak_memory_bytes"},
	"candidate_active_plus_cache_memory_bytes": {"turboquant_candidate_active_plus_cache_memory_bytes"},
	"candidate_wall_duration":                  {"turboquant_candidate_wall_duration"},
	"candidate_restore_duration":               {"turboquant_candidate_restore_duration"},
	"candidate_visible_tokens_per_sec":         {"turboquant_candidate_visible_tokens_per_sec"},
	"candidate_input_output_tokens_per_sec":    {"turboquant_candidate_input_output_tokens_per_sec"},
	"candidate_energy_joules":                  {"turboquant_candidate_energy_joules"},
}

var productionCombinedRequiredMetricAliases = mergeProductionRequiredMetricAliases(
	productionMTPRequiredMetricAliases,
	productionTurboQuantRequiredMetricAliases,
	map[string][]string{
		"mtp_greedy_output_matches":                           {"greedy_output_matches"},
		"mtp_target_only_cache_mode":                          {"target_only_cache_mode"},
		"mtp_target_only_visible_tokens_per_sec":              {"target_only_visible_tokens_per_sec"},
		"mtp_target_only_wall_duration":                       {"target_only_wall_duration"},
		"mtp_target_only_restore_duration":                    {"target_only_restore_duration"},
		"mtp_target_only_peak_memory_bytes":                   {"target_only_peak_memory_bytes"},
		"mtp_target_only_active_plus_cache_memory_bytes":      {"target_only_active_plus_cache_memory_bytes"},
		"mtp_target_only_energy_joules":                       {"target_only_energy_joules"},
		"turboquant_candidate_cache_mode":                     {"candidate_cache_mode", "kv_compression", "production_candidate_cache_mode"},
		"turboquant_candidate_cache_policy":                   {"candidate_cache_policy"},
		"turboquant_normal_context_validated":                 {"normal_context_validated"},
		"turboquant_stress_context_validated":                 {"stress_context_validated"},
		"turboquant_candidate_layout_version":                 {"candidate_layout_version", "production_required_layout_version"},
		"turboquant_candidate_key_algorithm":                  {"candidate_key_algorithm", "production_required_key_algorithm"},
		"turboquant_candidate_value_algorithm":                {"candidate_value_algorithm", "production_required_value_algorithm"},
		"turboquant_candidate_outlier_policy":                 {"candidate_outlier_policy", "production_required_outlier_policy"},
		"turboquant_candidate_effective_bits_milli":           {"candidate_effective_bits_milli", "production_target_effective_bits_milli"},
		"turboquant_candidate_qjl_residual":                   {"candidate_qjl_residual"},
		"turboquant_candidate_metadata_bytes":                 {"candidate_metadata_bytes"},
		"turboquant_quality_flags":                            {"quality_flags"},
		"turboquant_candidate_visible_tokens_per_sec":         {"candidate_visible_tokens_per_sec"},
		"turboquant_candidate_input_output_tokens_per_sec":    {"candidate_input_output_tokens_per_sec"},
		"turboquant_candidate_wall_duration":                  {"candidate_wall_duration"},
		"turboquant_candidate_restore_duration":               {"candidate_restore_duration"},
		"turboquant_candidate_peak_memory_bytes":              {"candidate_peak_memory_bytes"},
		"turboquant_candidate_active_plus_cache_memory_bytes": {"candidate_active_plus_cache_memory_bytes"},
		"turboquant_candidate_energy_joules":                  {"candidate_energy_joules"},
	},
)

func validateProductionRequiredMetricLabels(name string, labels map[string]string, required []string, aliases map[string][]string) error {
	if labels == nil {
		return core.E(name, "labels are required", nil)
	}
	var missing []string
	for _, metric := range required {
		if productionLabelKeyPresent(labels, metric) {
			continue
		}
		found := false
		for _, alias := range aliases[metric] {
			if productionLabelKeyPresent(labels, alias) {
				found = true
				break
			}
		}
		if !found {
			missing = append(missing, metric)
		}
	}
	if len(missing) > 0 {
		return core.E(name, "missing required production metric labels: "+strings.Join(missing, ","), nil)
	}
	return nil
}

func validateProductionRequiredLabelKeys(name string, labels map[string]string, required []string) error {
	if labels == nil {
		return core.E(name, "labels are required", nil)
	}
	var missing []string
	for _, metric := range required {
		if productionLabelKeyPresent(labels, metric) {
			continue
		}
		missing = append(missing, metric)
	}
	if len(missing) > 0 {
		return core.E(name, "missing required production metric labels: "+strings.Join(missing, ","), nil)
	}
	return nil
}

func productionLabelKeyPresent(labels map[string]string, key string) bool {
	_, ok := labels[key]
	return ok
}

func mergeProductionRequiredMetricAliases(inputs ...map[string][]string) map[string][]string {
	merged := make(map[string][]string)
	for _, input := range inputs {
		for key, values := range input {
			merged[key] = append(merged[key], values...)
		}
	}
	return merged
}

func productionMetricFloatLabel(value float64) string {
	return strconv.FormatFloat(value, 'f', 6, 64)
}

func productionDecisionBoolLabel(context string, labels map[string]string, metric string) (bool, error) {
	value := strings.TrimSpace(labels[metric])
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return false, core.E(context, "parse "+metric, err)
	}
	return parsed, nil
}

func productionDecisionNonNegativeFloatLabel(context string, labels map[string]string, metric string) (float64, error) {
	value := strings.TrimSpace(labels[metric])
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0, core.E(context, "parse "+metric, err)
	}
	if !productionBookGateFinite(parsed) {
		return 0, core.E(context, metric+" must be finite", nil)
	}
	if parsed < 0 {
		return 0, core.E(context, metric+" must be non-negative", nil)
	}
	return parsed, nil
}

func productionBoolLabel(labels map[string]string, metric string) (bool, error) {
	value := strings.TrimSpace(labels[metric])
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return false, core.E("rocm.EvaluateProductionBookRetainedArtifactDecisionLabels", "parse "+metric, err)
	}
	return parsed, nil
}

func productionFloatLabel(labels map[string]string, metric string) (float64, error) {
	value := strings.TrimSpace(labels[metric])
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0, core.E("rocm.EvaluateProductionBookRetainedArtifactDecisionLabels", "parse "+metric, err)
	}
	if !productionBookGateFinite(parsed) {
		return 0, core.E("rocm.EvaluateProductionBookRetainedArtifactDecisionLabels", metric+" must be finite", nil)
	}
	return parsed, nil
}

func productionIntLabel(labels map[string]string, metric string) (int, error) {
	value := strings.TrimSpace(labels[metric])
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return 0, core.E("rocm.EvaluateProductionBookRetainedArtifactDecisionLabels", "parse "+metric, err)
	}
	return parsed, nil
}

func productionBookGateMetricLabel(metric string, value float64) string {
	switch metric {
	case "production_book_gate_candidate",
		"production_book_gate_q6",
		"production_book_gate_turns",
		"production_book_gate_wall",
		"production_book_gate_decode",
		"production_book_gate_quality",
		"production_book_gate_reason_code",
		"production_book_gate_quality_flags":
		return strconv.Itoa(int(value))
	}
	return strconv.FormatFloat(value, 'f', -1, 64)
}

func productionBookRetainedArtifactDecisionReason(decision ProductionBookRetainedArtifactDecision) string {
	if !decision.RetainedRoute {
		return "retained-state runtime KV route is required; prompt replay artifacts are rejected"
	}
	return decision.Gate.Reason
}

func productionBookGateFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func productionBookGateFloatMetric(labels map[string]string, metric string) (float64, error) {
	value := strings.TrimSpace(labels[metric])
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0, core.E("rocm.EvaluateProductionBookGateMetricLabels", "parse "+metric, err)
	}
	return parsed, nil
}

func productionBookGateBoolMetric(labels map[string]string, metric string) (bool, error) {
	value, err := productionBookGateFloatMetric(labels, metric)
	if err != nil {
		return false, err
	}
	return productionBookGateBool(metric, value)
}

func productionBookGateBoolValue(metrics map[string]float64, metric string) (bool, error) {
	return productionBookGateBool(metric, metrics[metric])
}

func productionBookGateBool(metric string, value float64) (bool, error) {
	switch value {
	case 0:
		return false, nil
	case 1:
		return true, nil
	default:
		return false, core.E("rocm.EvaluateProductionBookGateMetricLabels", metric+" must be 0 or 1", nil)
	}
}

func productionBookGateIntMetric(labels map[string]string, metric string) (int, error) {
	value, err := productionBookGateFloatMetric(labels, metric)
	if err != nil {
		return 0, err
	}
	parsed := int(value)
	if value != float64(parsed) {
		return 0, core.E("rocm.EvaluateProductionBookGateMetricLabels", metric+" must be an integer", nil)
	}
	return parsed, nil
}

func productionBookGateIntValue(metrics map[string]float64, metric string) (int, error) {
	value := metrics[metric]
	parsed := int(value)
	if value != float64(parsed) {
		return 0, core.E("rocm.EvaluateProductionBookGateMetrics", metric+" must be an integer", nil)
	}
	return parsed, nil
}

func productionBookGateReasonCodeMetric(labels map[string]string) (ProductionBookGateReasonCode, error) {
	value, err := productionBookGateIntMetric(labels, "production_book_gate_reason_code")
	if err != nil {
		return 0, err
	}
	code := ProductionBookGateReasonCode(value)
	if code < ProductionBookGateReasonPass || code > ProductionBookGateReasonQuality {
		return 0, core.E("rocm.EvaluateProductionBookGateMetricLabels", fmt.Sprintf("unknown production_book_gate_reason_code %d", value), nil)
	}
	return code, nil
}

func productionBookGateReasonCodeValue(metrics map[string]float64) (ProductionBookGateReasonCode, error) {
	value, err := productionBookGateIntValue(metrics, "production_book_gate_reason_code")
	if err != nil {
		return 0, err
	}
	code := ProductionBookGateReasonCode(value)
	if code < ProductionBookGateReasonPass || code > ProductionBookGateReasonQuality {
		return 0, core.E("rocm.EvaluateProductionBookGateMetrics", fmt.Sprintf("unknown production_book_gate_reason_code %d", value), nil)
	}
	return code, nil
}

func productionBookRetainedArtifactReasonCodeLabel(labels map[string]string) (ProductionBookGateReasonCode, error) {
	value, err := productionIntLabel(labels, "production_book_retained_artifact_gate_reason_code")
	if err != nil {
		return 0, err
	}
	code := ProductionBookGateReasonCode(value)
	if code < ProductionBookGateReasonPass || code > ProductionBookGateReasonQuality {
		return 0, core.E("rocm.EvaluateProductionBookRetainedArtifactDecisionLabels", fmt.Sprintf("unknown production_book_retained_artifact_gate_reason_code %d", value), nil)
	}
	return code, nil
}

func (decision ProductionBookGateMetricDecision) validateProductionBookGateMetricDecision(policy ProductionBookGatePolicy) error {
	if policy.MinimumRawDecodeTokensSec <= 0 {
		policy.MinimumRawDecodeTokensSec = float64(productionLaneRetainedVisibleTokensSec)
	}
	if policy.MaximumWallSeconds <= 0 {
		policy.MaximumWallSeconds = ProductionLaneBookWallSeconds
	}
	allChecksPass := decision.QuantAccepted &&
		decision.TurnsAccepted &&
		decision.WallAccepted &&
		decision.DecodeAccepted &&
		decision.QualityAccepted
	if decision.ProductionCandidate != (allChecksPass && decision.ReasonCode == ProductionBookGateReasonPass) {
		return core.E("rocm.EvaluateProductionBookGateMetricLabels", "production_book_gate_candidate is inconsistent with gate checks and reason code", nil)
	}
	if decision.QualityFlags < 0 {
		return core.E("rocm.EvaluateProductionBookGateMetricLabels", "production_book_gate_quality_flags must be non-negative", nil)
	}
	if decision.WallSeconds < 0 {
		return core.E("rocm.EvaluateProductionBookGateMetricLabels", "production_book_gate_wall_s must be non-negative", nil)
	}
	if decision.RawDecodeTokensPerSec < 0 {
		return core.E("rocm.EvaluateProductionBookGateMetricLabels", "production_book_gate_raw_decode_tok/s must be non-negative", nil)
	}
	expectedWallAccepted := decision.WallSeconds > 0 && decision.WallSeconds <= float64(policy.MaximumWallSeconds)
	if decision.WallAccepted != expectedWallAccepted {
		return core.E("rocm.EvaluateProductionBookGateMetricLabels", "production_book_gate_wall is inconsistent with production_book_gate_wall_s", nil)
	}
	expectedDecodeAccepted := decision.RawDecodeTokensPerSec >= policy.MinimumRawDecodeTokensSec
	if decision.DecodeAccepted != expectedDecodeAccepted {
		return core.E("rocm.EvaluateProductionBookGateMetricLabels", "production_book_gate_decode is inconsistent with production_book_gate_raw_decode_tok/s", nil)
	}
	expectedQualityAccepted := decision.QualityFlags <= policy.MaximumQualityFlags
	if decision.QualityAccepted != expectedQualityAccepted {
		return core.E("rocm.EvaluateProductionBookGateMetricLabels", "production_book_gate_quality is inconsistent with production_book_gate_quality_flags", nil)
	}
	expectedReason := productionBookGateExpectedReasonCode(decision)
	if decision.ReasonCode != expectedReason {
		return core.E("rocm.EvaluateProductionBookGateMetricLabels", fmt.Sprintf("production_book_gate_reason_code %d is inconsistent with first failing gate %d", decision.ReasonCode, expectedReason), nil)
	}
	return nil
}

func productionBookGateExpectedReasonCode(decision ProductionBookGateMetricDecision) ProductionBookGateReasonCode {
	if !decision.QuantAccepted {
		return ProductionBookGateReasonQuant
	}
	if !decision.TurnsAccepted {
		return ProductionBookGateReasonTurns
	}
	if !decision.WallAccepted {
		return ProductionBookGateReasonWall
	}
	if !decision.DecodeAccepted {
		return ProductionBookGateReasonDecode
	}
	if !decision.QualityAccepted {
		return ProductionBookGateReasonQuality
	}
	return ProductionBookGateReasonPass
}

func productionBookGateMetricDecisionReason(policy ProductionBookGatePolicy, decision ProductionBookGateMetricDecision) string {
	if policy.QuantBits <= 0 {
		policy.QuantBits = ProductionLaneProductDefaultQuantBits
	}
	if policy.MinimumTurns <= 0 {
		policy.MinimumTurns = ProductionLaneBookTurnCount
	}
	if policy.MaximumWallSeconds <= 0 {
		policy.MaximumWallSeconds = ProductionLaneBookWallSeconds
	}
	if policy.MinimumRawDecodeTokensSec <= 0 {
		policy.MinimumRawDecodeTokensSec = float64(productionLaneRetainedVisibleTokensSec)
	}
	switch decision.ReasonCode {
	case ProductionBookGateReasonPass:
		return "production book gate passes q6 retained-state throughput, wall, and quality checks"
	case ProductionBookGateReasonQuant:
		return fmt.Sprintf("production book gate requires q%d", policy.QuantBits)
	case ProductionBookGateReasonMetrics:
		return fmt.Sprintf("production book gate requires complete q%d metrics", policy.QuantBits)
	case ProductionBookGateReasonTurns:
		return fmt.Sprintf("production book gate requires %d turns", policy.MinimumTurns)
	case ProductionBookGateReasonWall:
		return fmt.Sprintf("production book gate wall %.3fs exceeds %ds candidate limit", decision.WallSeconds, policy.MaximumWallSeconds)
	case ProductionBookGateReasonDecode:
		return fmt.Sprintf("production book gate raw decode %.3f tok/s below %.0f tok/s", decision.RawDecodeTokensPerSec, policy.MinimumRawDecodeTokensSec)
	case ProductionBookGateReasonQuality:
		return fmt.Sprintf("production book gate quality flags = %d, want 0", decision.QualityFlags)
	default:
		return "production book gate reason is unknown"
	}
}
