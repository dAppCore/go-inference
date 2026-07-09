// SPDX-Licence-Identifier: EUPL-1.2

//go:build !linux || !amd64 || rocm_legacy_server

package hip

import "strconv"

const ProductionFastLaneName = "rocm-gemma4-fast-lane"

const (
	portableProductionLaneName                    = "gemma4-e2b-it-q6"
	portableProductionLaneModelID                 = "mlx-community/gemma-4-e2b-it-6bit"
	portableProductionLaneCurrentModelID          = "lmstudio-community/gemma-4-E2B-it-MLX-6bit"
	portableProductionLaneArchitecture            = "gemma4_text"
	portableProductionLaneChatTemplate            = "gemma4"
	portableProductionLaneProductDefaultQuantBits = 6
	portableProductionFastLaneQuantMode           = "affine"
	portableProductionFastLaneQuantGroup          = 64
	portableProductionFastLaneCacheMode           = "turboquant-kv"
)

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
	mtp := DefaultProductionMTPPolicy()
	turbo := DefaultProductionTurboQuantPolicy()
	combined := DefaultProductionCombinedMTPAndTurboQuantPolicy()
	required := portableProductionFastLaneRequiredMetrics(mtp.RequiredMetrics, turbo.RequiredMetrics, combined.RequiredMetrics)
	enabled := mtp.EnabledByDefault && turbo.EnabledByDefault && combined.EnabledByDefault
	return ProductionFastLane{
		Name:                  ProductionFastLaneName,
		Backend:               "rocm",
		Library:               "go-rocm",
		ReferenceBackend:      "go-mlx",
		ModelID:               portableProductionLaneCurrentModelID,
		LockedModelID:         portableProductionLaneModelID,
		OfficialTargetModelID: mtp.TargetModelID,
		AssistantModelID:      mtp.AssistantModelID,
		Architecture:          portableProductionLaneArchitecture,
		ChatTemplate:          portableProductionLaneChatTemplate,
		QuantBits:             portableProductionLaneProductDefaultQuantBits,
		QuantMode:             portableProductionFastLaneQuantMode,
		QuantGroup:            portableProductionFastLaneQuantGroup,
		CacheMode:             turbo.CacheMode,
		ContextLength:         0,
		MaxTokens:             0,
		MTPDefaultDraftTokens: mtp.DefaultDraftTokens,
		EnabledByDefault:      enabled,
		RequiresEnvGate:       false,
		RequiresCLIFlag:       false,
		RequiredMetrics:       required,
		Labels: map[string]string{
			"backend":                          "rocm",
			"library":                          "go-rocm",
			"reference_backend":                "go-mlx",
			"production_lane":                  portableProductionLaneName,
			"production_fast_lane":             "true",
			"production_default":               strconv.FormatBool(enabled),
			"production_requires_env_gate":     "false",
			"production_requires_cli_flag":     "false",
			"production_quant_model":           portableProductionLaneCurrentModelID,
			"production_quant_locked_model":    portableProductionLaneModelID,
			"production_quant_tier":            "q6",
			"production_quant_mode":            portableProductionFastLaneQuantMode,
			"production_quant_group":           strconv.Itoa(portableProductionFastLaneQuantGroup),
			"production_quant_bits":            strconv.Itoa(portableProductionLaneProductDefaultQuantBits),
			"production_cache_mode":            turbo.CacheMode,
			"production_mtp_mode":              mtp.Mode,
			"production_combined_mode":         combined.Mode,
			"production_mtp_assistant_model":   mtp.AssistantModelID,
			"production_mtp_default_drafts":    strconv.Itoa(mtp.DefaultDraftTokens),
			"production_required_metric_count": strconv.Itoa(len(required)),
			"production_build":                 "portable",
		},
	}
}

func portableProductionFastLaneRequiredMetrics(groups ...[]string) []string {
	seed := []string{
		"load_duration",
		"candidate_cache_mode",
		"turboquant_candidate_cache_mode",
	}
	out := make([]string, 0, len(seed)+64)
	seen := make(map[string]struct{}, len(seed)+64)
	for _, metric := range seed {
		if metric == "" {
			continue
		}
		seen[metric] = struct{}{}
		out = append(out, metric)
	}
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
