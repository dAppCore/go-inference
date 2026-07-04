// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"strconv"
)

const ProductionFastLaneName = "rocm-gemma4-fast-lane"

// ProductionFastLane is the default CLI/API contract for applications that
// need a production ROCm route without hidden environment gates or opt-in flags.
type ProductionFastLane struct {
	Name                  string            `json:"name"`
	Backend               string            `json:"backend"`
	Library               string            `json:"library"`
	ReferenceBackend      string            `json:"reference_backend"`
	ModelID               string            `json:"model_id"`
	LockedModelID         string            `json:"locked_model_id"`
	OfficialTargetModelID string            `json:"official_target_model_id"`
	AssistantModelID      string            `json:"assistant_model_id"`
	Architecture          string            `json:"architecture"`
	ChatTemplate          string            `json:"chat_template"`
	QuantBits             int               `json:"quant_bits"`
	QuantMode             string            `json:"quant_mode"`
	QuantGroup            int               `json:"quant_group"`
	CacheMode             string            `json:"cache_mode"`
	ContextLength         int               `json:"context_length"`
	MaxTokens             int               `json:"max_tokens"`
	MTPDefaultDraftTokens int               `json:"mtp_default_draft_tokens"`
	EnabledByDefault      bool              `json:"enabled_by_default"`
	RequiresEnvGate       bool              `json:"requires_env_gate"`
	RequiresCLIFlag       bool              `json:"requires_cli_flag"`
	RequiredMetrics       []string          `json:"required_metrics,omitempty"`
	Labels                map[string]string `json:"labels,omitempty"`
}

func DefaultProductionFastLane() ProductionFastLane {
	lane := DefaultProductionLane()
	quant := DefaultProductionQuantizationPolicy()
	mtp := DefaultProductionMTPPolicy()
	turbo := DefaultProductionTurboQuantPolicy()
	combined := DefaultProductionCombinedMTPAndTurboQuantPolicy()
	defaultTier := productionQuantizationTierByBits(quant, quant.DefaultBits)
	if defaultTier.QuantMode == "" {
		defaultTier.QuantMode = "affine"
	}
	if defaultTier.QuantGroup == 0 {
		defaultTier.QuantGroup = 64
	}
	required := productionFastLaneRequiredMetrics(quant.RequiredBenchmarkMetrics, mtp.RequiredMetrics, turbo.RequiredMetrics, combined.RequiredMetrics)
	enabled := mtp.EnabledByDefault && turbo.EnabledByDefault && combined.EnabledByDefault
	labels := map[string]string{
		"backend":                          "rocm",
		"library":                          "go-rocm",
		"reference_backend":                "go-mlx",
		"production_lane":                  lane.Name,
		"production_fast_lane":             "true",
		"production_default":               boolLabel(enabled),
		"production_requires_env_gate":     "false",
		"production_requires_cli_flag":     "false",
		"production_quant_model":           lane.ModelID,
		"production_quant_locked_model":    ProductionLaneModelID,
		"production_quant_tier":            defaultTier.Name,
		"production_quant_mode":            defaultTier.QuantMode,
		"production_quant_group":           strconv.Itoa(defaultTier.QuantGroup),
		"production_quant_bits":            strconv.Itoa(defaultTier.Bits),
		"production_cache_mode":            turbo.CacheMode,
		"production_mtp_mode":              mtp.Mode,
		"production_combined_mode":         combined.Mode,
		"production_mtp_assistant_model":   mtp.AssistantModelID,
		"production_mtp_default_drafts":    strconv.Itoa(mtp.DefaultDraftTokens),
		"production_required_metric_count": strconv.Itoa(len(required)),
	}
	return ProductionFastLane{
		Name:                  ProductionFastLaneName,
		Backend:               "rocm",
		Library:               "go-rocm",
		ReferenceBackend:      "go-mlx",
		ModelID:               lane.ModelID,
		LockedModelID:         ProductionLaneModelID,
		OfficialTargetModelID: mtp.TargetModelID,
		AssistantModelID:      mtp.AssistantModelID,
		Architecture:          lane.Architecture,
		ChatTemplate:          lane.ChatTemplate,
		QuantBits:             defaultTier.Bits,
		QuantMode:             defaultTier.QuantMode,
		QuantGroup:            defaultTier.QuantGroup,
		CacheMode:             turbo.CacheMode,
		ContextLength:         lane.ContextLength,
		MaxTokens:             lane.MaxTokens,
		MTPDefaultDraftTokens: mtp.DefaultDraftTokens,
		EnabledByDefault:      enabled,
		RequiresEnvGate:       false,
		RequiresCLIFlag:       false,
		RequiredMetrics:       required,
		Labels:                labels,
	}
}

func productionFastLaneRequiredMetrics(groups ...[]string) []string {
	var out []string
	seen := make(map[string]struct{})
	for _, group := range groups {
		for _, metric := range group {
			if metric == "" {
				continue
			}
			if _, ok := seen[metric]; ok {
				continue
			}
			seen[metric] = struct{}{}
			out = append(out, metric)
		}
	}
	return out
}
