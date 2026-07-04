// SPDX-Licence-Identifier: EUPL-1.2

//go:build !linux || !amd64 || rocm_legacy_server

package hip

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"math"
	"os"
	"strconv"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
)

const (
	ProductionMTPDefaultDraftTokens                 = 4
	ProductionMTPAssistantTokenOrderingVocabSize    = modelgemma4.AssistantTokenOrderingVocabSize
	ProductionMTPAssistantOrderedEmbeddingCentroids = modelgemma4.AssistantOrderedEmbeddingCentroids
	ProductionMTPAssistantCentroidIntermediateTopK  = modelgemma4.AssistantCentroidIntermediateTopK
	ProductionTurboQuantKVLayoutVersion             = "turboquant-kv-v1"
	ProductionTurboQuantKeyAlgorithm                = "turboquantprod"
	ProductionTurboQuantValueAlgorithm              = "turboquantmse"
	ProductionTurboQuantOutlierPolicy               = "high-half-head-dim-v1"
	ProductionCombinedMTPAndTurboQuantMode          = "mtp+turboquant-kv"
	OfficialGemma4E2BRoleTarget                     = "target"
	OfficialGemma4E2BRoleAssistant                  = "assistant"
	SimpleSelfDistillationRecipe4BInstruct          = "SimpleSD-4B-instruct"
	SimpleSelfDistillationRecipe4BThinking          = "SimpleSD-4B-thinking"
	SimpleSelfDistillationRecipe30BA3BInstruct      = "SimpleSD-30b-a3b-instruct"

	portableOfficialGemma4E2BTargetModelID                       = modelgemma4.OfficialE2BTargetModelID
	portableOfficialGemma4E2BTargetRevision                      = modelgemma4.OfficialE2BTargetRevision
	portableOfficialGemma4E2BAssistantModelID                    = modelgemma4.OfficialE2BAssistantModelID
	portableOfficialGemma4E2BAssistantRevision                   = modelgemma4.OfficialE2BAssistantRevision
	portableOfficialGemma4E2BAssistantArchitecture               = modelgemma4.AssistantArchitecture
	portableOfficialGemma4E2BSourceCheckedAt                     = modelgemma4.OfficialE2BSourceCheckedAt
	portableOfficialGemma4E2BTargetConfigSHA256                  = modelgemma4.OfficialE2BTargetConfigSHA256
	portableOfficialGemma4E2BAssistantConfigSHA256               = modelgemma4.OfficialE2BAssistantConfigSHA256
	portableProductionMTPAssistantCentroidIntermediateTopKLabel  = modelgemma4.AssistantCentroidIntermediateTopKLabel
	portableProductionMTPAssistantOrderedEmbeddingCentroidsLabel = modelgemma4.AssistantOrderedEmbeddingCentroidsLabel
	portableProductionMTPAssistantTokenOrderingShapeLabel        = modelgemma4.AssistantTokenOrderingShape
	portableProductionTurboQuantKVMode                           = "turboquant-kv"
	portableProductionTurboQuantCacheModePaged                   = "paged"
	portableProductionRetainedTurns                              = 10
	portableProductionLongContextLength                          = 32768
	portableProductionHyperLongContextLength                     = 131072

	simpleSelfDistillationDecodeTemperatureLabel = "ssd_decode_temperature"
	simpleSelfDistillationEvalTemperatureLabel   = "ssd_eval_temperature"
)

var (
	defaultPortableProductionTurboQuantCompareAgainstCacheModes = []string{
		"fp16",
		portableProductionTurboQuantCacheModePaged,
		"q8",
		"k-q8-v-q4",
	}
	defaultPortableProductionTurboQuantRequiredMetrics = []string{
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
	defaultPortableProductionCombinedMTPAndTurboQuantRequiredMetrics = []string{
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
)

// OfficialGemma4E2BLock records the pinned target/assistant pair the MTP CLI
// contract reports even on portable builds.
type OfficialGemma4E2BLock struct {
	Role            string `json:"role"`
	ModelID         string `json:"model_id"`
	Revision        string `json:"revision"`
	SourceCheckedAt string `json:"source_checked_at"`
	Architecture    string `json:"architecture"`
	ModelType       string `json:"model_type"`
	ConfigSHA256    string `json:"config_sha256"`
}

type ROCmLoadConfig struct {
	CacheMode    string `json:"cache_mode,omitempty"`
	DeviceKVMode string `json:"device_kv_mode,omitempty"`
}

func LoadModelWithConfig(string, ROCmLoadConfig, ...inference.LoadOption) (inference.TextModel, error) {
	return nil, core.E("rocm.LoadModelWithConfig", "native ROCm load config is not available in this build", nil)
}

type ProductionMTPPolicy struct {
	TargetModelID               string   `json:"target_model_id"`
	AssistantModelID            string   `json:"assistant_model_id"`
	Mode                        string   `json:"mode"`
	DefaultDraftTokens          int      `json:"default_draft_tokens"`
	RequiredDraftTokenSweeps    []int    `json:"required_draft_token_sweeps,omitempty"`
	MinimumRetainedTurns        int      `json:"minimum_retained_turns"`
	MinimumVisibleTokensPerSec  float64  `json:"minimum_visible_tokens_per_sec"`
	EnabledByDefault            bool     `json:"enabled_by_default"`
	RequiresRetainedWorkflow    bool     `json:"requires_retained_workflow"`
	RequiresGreedyParity        bool     `json:"requires_greedy_parity"`
	RequiresSideBySideBenchmark bool     `json:"requires_side_by_side_benchmark"`
	RequiredMetrics             []string `json:"required_metrics"`
}

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

// SimpleSelfDistillationConfig configures native self-distillation reports. The
// portable build keeps the schema available so CLI planning stays cross-arch.
type SimpleSelfDistillationConfig struct {
	SampleMaxTokens   int                      `json:"sample_max_tokens,omitempty"`
	SampleTemperature float32                  `json:"sample_temperature,omitempty"`
	SampleTopK        int                      `json:"sample_top_k,omitempty"`
	SampleTopP        float32                  `json:"sample_top_p,omitempty"`
	SampleMinP        float32                  `json:"sample_min_p,omitempty"`
	RepetitionPenalty float32                  `json:"repetition_penalty,omitempty"`
	FilterShortestPct float32                  `json:"filter_shortest_percent,omitempty"`
	DecodeTemperature float32                  `json:"decode_temperature,omitempty"`
	SFT               inference.TrainingConfig `json:"sft,omitempty"`
}

// SimpleSelfDistillationRunner supplies the generation step for portable CLI
// targets. Native ROCm builds provide the HIP-backed variant in
// simple_self_distillation.go.
type SimpleSelfDistillationRunner struct {
	Generate func(context.Context, string, inference.GenerateConfig) (string, error)
}

// SimpleSelfDistillationSample records one raw sampled response.
type SimpleSelfDistillationSample struct {
	Prompt   string            `json:"prompt"`
	Response string            `json:"response"`
	Labels   map[string]string `json:"labels,omitempty"`
}

// SimpleSelfDistillationResult records a portable SSD trace run.
type SimpleSelfDistillationResult struct {
	Samples           []SimpleSelfDistillationSample `json:"samples"`
	SFT               *inference.TrainingResult      `json:"-"`
	SampleTemperature float32                        `json:"sample_temperature"`
	DecodeTemperature float32                        `json:"decode_temperature"`
	SampleMaxTokens   int                            `json:"sample_max_tokens"`
	SampleTopK        int                            `json:"sample_top_k,omitempty"`
	SampleTopP        float32                        `json:"sample_top_p,omitempty"`
	SampleMinP        float32                        `json:"sample_min_p,omitempty"`
	RepetitionPenalty float32                        `json:"repetition_penalty,omitempty"`
	FilterShortestPct float32                        `json:"filter_shortest_percent,omitempty"`
}

type SimpleSelfDistillationRecipe struct {
	Name          string                                    `json:"name"`
	Model         string                                    `json:"model"`
	Dataset       string                                    `json:"dataset,omitempty"`
	DatasetConfig string                                    `json:"dataset_config,omitempty"`
	DatasetSplit  string                                    `json:"dataset_split,omitempty"`
	Train         SimpleSelfDistillationConfig              `json:"train"`
	Eval          SimpleSelfDistillationCodeBenchmarkConfig `json:"eval"`
	Notes         []string                                  `json:"notes,omitempty"`
}

type SimpleSelfDistillationCodeBenchmarkConfig struct {
	Benchmark  string                   `json:"benchmark,omitempty"`
	NRepeat    int                      `json:"n_repeat,omitempty"`
	Generate   inference.GenerateConfig `json:"generate"`
	Seeds      []uint64                 `json:"seeds,omitempty"`
	OutputPath string                   `json:"output_path,omitempty"`
}

type SimpleSelfDistillationCodeBenchmarkSample struct {
	ID     string            `json:"id,omitempty"`
	Prompt string            `json:"prompt"`
	Tests  []string          `json:"tests,omitempty"`
	Meta   map[string]string `json:"meta,omitempty"`
}

type portableSSDCodeBenchmarkJSONLRecord struct {
	ID               string            `json:"id"`
	QuestionID       string            `json:"question_id"`
	TaskID           string            `json:"task_id"`
	Prompt           string            `json:"prompt"`
	Question         string            `json:"question"`
	QuestionContent  string            `json:"question_content"`
	Problem          string            `json:"problem"`
	StarterCode      string            `json:"starter_code"`
	Test             string            `json:"test"`
	Tests            []string          `json:"tests"`
	PublicTestCases  []string          `json:"public_test_cases"`
	PrivateTestCases []string          `json:"private_test_cases"`
	Metadata         map[string]string `json:"metadata"`
	ContestDate      string            `json:"contest_date"`
	Difficulty       string            `json:"difficulty"`
	Platform         string            `json:"platform"`
}

func DefaultOfficialGemma4E2BLocks() []OfficialGemma4E2BLock {
	return []OfficialGemma4E2BLock{
		{
			Role:            OfficialGemma4E2BRoleTarget,
			ModelID:         portableOfficialGemma4E2BTargetModelID,
			Revision:        portableOfficialGemma4E2BTargetRevision,
			SourceCheckedAt: portableOfficialGemma4E2BSourceCheckedAt,
			Architecture:    "Gemma4ForConditionalGeneration",
			ModelType:       "gemma4",
			ConfigSHA256:    portableOfficialGemma4E2BTargetConfigSHA256,
		},
		{
			Role:            OfficialGemma4E2BRoleAssistant,
			ModelID:         portableOfficialGemma4E2BAssistantModelID,
			Revision:        portableOfficialGemma4E2BAssistantRevision,
			SourceCheckedAt: portableOfficialGemma4E2BSourceCheckedAt,
			Architecture:    "Gemma4AssistantForCausalLM",
			ModelType:       "gemma4_assistant",
			ConfigSHA256:    portableOfficialGemma4E2BAssistantConfigSHA256,
		},
	}
}

func DefaultProductionMTPPolicy() ProductionMTPPolicy {
	return ProductionMTPPolicy{
		TargetModelID:               portableOfficialGemma4E2BTargetModelID,
		AssistantModelID:            portableOfficialGemma4E2BAssistantModelID,
		Mode:                        "mtp_attached_drafter",
		DefaultDraftTokens:          ProductionMTPDefaultDraftTokens,
		RequiredDraftTokenSweeps:    []int{1, 2, 4},
		MinimumRetainedTurns:        portableProductionRetainedTurns,
		MinimumVisibleTokensPerSec:  100,
		EnabledByDefault:            true,
		RequiresRetainedWorkflow:    true,
		RequiresGreedyParity:        true,
		RequiresSideBySideBenchmark: true,
		RequiredMetrics: []string{
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
		},
	}
}

func DefaultProductionTurboQuantPolicy() ProductionTurboQuantPolicy {
	return ProductionTurboQuantPolicy{
		TargetModelID:                   portableProductionLaneCurrentModelID,
		CacheMode:                       portableProductionTurboQuantKVMode,
		Mode:                            portableProductionTurboQuantKVMode,
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
		MinimumRetainedTurns:            portableProductionRetainedTurns,
		NormalContextLength:             portableProductionLongContextLength,
		StressContextLength:             portableProductionHyperLongContextLength,
		CompareAgainstCacheModes:        append([]string(nil), defaultPortableProductionTurboQuantCompareAgainstCacheModes...),
		RequiredMetrics:                 append([]string(nil), defaultPortableProductionTurboQuantRequiredMetrics...),
	}
}

func DefaultProductionCombinedMTPAndTurboQuantPolicy() ProductionCombinedMTPAndTurboQuantPolicy {
	mtp := DefaultProductionMTPPolicy()
	return ProductionCombinedMTPAndTurboQuantPolicy{
		TargetModelID:                   mtp.TargetModelID,
		AssistantModelID:                mtp.AssistantModelID,
		Mode:                            ProductionCombinedMTPAndTurboQuantMode,
		CacheMode:                       portableProductionTurboQuantKVMode,
		EnabledByDefault:                true,
		RequiresExplicitOptIn:           false,
		RequiresRetainedWorkflow:        true,
		RequiresGreedyParity:            true,
		RequiresTurboQuantQualityParity: true,
		RequiresMTPPromotion:            true,
		RequiresTurboQuantPromotion:     true,
		MinimumRetainedTurns:            portableProductionRetainedTurns,
		RequiredMetrics:                 append([]string(nil), defaultPortableProductionCombinedMTPAndTurboQuantRequiredMetrics...),
	}
}

func DefaultSimpleSelfDistillationConfig() SimpleSelfDistillationConfig {
	return SimpleSelfDistillationConfig{
		SampleMaxTokens:   65536,
		SampleTemperature: 1.5,
		SampleTopK:        20,
		SampleTopP:        0.8,
		RepetitionPenalty: 1.0,
		FilterShortestPct: 10,
	}
}

func DefaultSimpleSelfDistillationCodeBenchmarkConfig() SimpleSelfDistillationCodeBenchmarkConfig {
	return SimpleSelfDistillationCodeBenchmarkConfig{
		Benchmark: "LiveCodeBench-v6",
		NRepeat:   20,
		Seeds:     []uint64{0, 1234, 1234, 1234},
		Generate: inference.GenerateConfig{
			MaxTokens:   32768,
			Temperature: 0.6,
			TopP:        0.95,
			TopK:        20,
		},
	}
}

// RunSimpleSelfDistillation samples raw outputs from a frozen model and stops
// at the generated trace. Training remains an explicit SFT step.
func RunSimpleSelfDistillation(ctx context.Context, runner SimpleSelfDistillationRunner, dataset inference.DatasetStream, cfg SimpleSelfDistillationConfig) (*SimpleSelfDistillationResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if dataset == nil {
		return nil, core.NewError("rocm: SSD dataset is nil")
	}
	if runner.Generate == nil {
		return nil, core.NewError("rocm: SSD generate function is nil")
	}
	cfg = normalizePortableSimpleSelfDistillationConfig(cfg)
	if err := validatePortableSimpleSelfDistillationConfig(cfg); err != nil {
		return nil, err
	}

	result := &SimpleSelfDistillationResult{
		Samples:           make([]SimpleSelfDistillationSample, 0, 16),
		SampleTemperature: cfg.SampleTemperature,
		DecodeTemperature: cfg.DecodeTemperature,
		SampleMaxTokens:   cfg.SampleMaxTokens,
		SampleTopK:        cfg.SampleTopK,
		SampleTopP:        cfg.SampleTopP,
		SampleMinP:        cfg.SampleMinP,
		RepetitionPenalty: cfg.RepetitionPenalty,
		FilterShortestPct: cfg.FilterShortestPct,
	}
	generateCfg := portableSimpleSelfDistillationGenerateConfig(cfg)
	for index := 0; ; index++ {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		sample, ok, err := dataset.Next()
		if err != nil {
			return result, err
		}
		if !ok {
			break
		}
		prompt := portableSimpleSelfDistillationPrompt(sample)
		if prompt == "" {
			continue
		}
		response, err := runner.Generate(ctx, prompt, generateCfg)
		if err != nil {
			return result, err
		}
		labels := cloneStringMap(sample.Labels)
		if labels == nil {
			labels = make(map[string]string, 4)
		}
		labels["ssd"] = "simple_self_distillation"
		labels["ssd_source_index"] = strconv.Itoa(index)
		labels["ssd_sample_temperature"] = formatPortableSimpleSelfDistillationFloat32(cfg.SampleTemperature)
		result.Samples = append(result.Samples, SimpleSelfDistillationSample{
			Prompt:   prompt,
			Response: response,
			Labels:   cloneStringMap(labels),
		})
	}
	if len(result.Samples) == 0 {
		return result, core.NewError("rocm: SSD dataset produced no prompts")
	}
	return result, nil
}

// RunModelSimpleSelfDistillation wires a TextModel into the portable SSD trace
// runner so CPU/CUDA targets keep the same CLI contract as the ROCm build.
func RunModelSimpleSelfDistillation(ctx context.Context, model inference.TextModel, dataset inference.DatasetStream, cfg SimpleSelfDistillationConfig) (*SimpleSelfDistillationResult, error) {
	if model == nil {
		return nil, core.NewError("rocm: SSD model is nil")
	}
	return RunSimpleSelfDistillation(ctx, SimpleSelfDistillationRunner{
		Generate: func(ctx context.Context, prompt string, cfg inference.GenerateConfig) (string, error) {
			return generatePortableSimpleSelfDistillationText(ctx, model, prompt, cfg)
		},
	}, dataset, cfg)
}

// SampleGenerateConfig returns the frozen-model sampling configuration used to
// create the raw SSD trace rows.
func (result *SimpleSelfDistillationResult) SampleGenerateConfig() inference.GenerateConfig {
	if result == nil {
		return inference.GenerateConfig{}
	}
	return inference.GenerateConfig{
		MaxTokens:     result.SampleMaxTokens,
		Temperature:   result.SampleTemperature,
		TopK:          result.SampleTopK,
		TopP:          result.SampleTopP,
		MinP:          result.SampleMinP,
		RepeatPenalty: result.RepetitionPenalty,
	}
}

// DecodeGenerateConfig returns the post-SSD decode configuration with the
// separately tuned decode temperature. The token budget remains caller-owned.
func (result *SimpleSelfDistillationResult) DecodeGenerateConfig(maxTokens int) inference.GenerateConfig {
	if result == nil {
		return inference.GenerateConfig{MaxTokens: maxTokens}
	}
	return inference.GenerateConfig{
		MaxTokens:   maxTokens,
		Temperature: result.DecodeTemperature,
	}
}

// SimpleSelfDistillationEvalGenerateConfig reconstructs the post-SSD eval
// generation config carried through TrainingConfig labels.
func SimpleSelfDistillationEvalGenerateConfig(labels map[string]string, maxTokens int) (inference.GenerateConfig, bool, error) {
	cfg := inference.GenerateConfig{MaxTokens: maxTokens}
	value := labels[simpleSelfDistillationEvalTemperatureLabel]
	if value == "" {
		value = labels[simpleSelfDistillationDecodeTemperatureLabel]
	}
	if value == "" {
		return cfg, false, nil
	}
	temperature, err := strconv.ParseFloat(value, 32)
	if err != nil || temperature < 0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
		return inference.GenerateConfig{}, false, core.NewError("rocm: SSD eval temperature label must be non-negative and finite")
	}
	cfg.Temperature = float32(temperature)
	return cfg, true, nil
}

func SimpleSelfDistillationRecipes() []SimpleSelfDistillationRecipe {
	train := DefaultSimpleSelfDistillationConfig()
	eval := DefaultSimpleSelfDistillationCodeBenchmarkConfig()
	return []SimpleSelfDistillationRecipe{
		portableSSDRecipe(SimpleSelfDistillationRecipe4BInstruct, "apple/SimpleSD-4B-instruct", train, eval),
		portableSSDRecipe(SimpleSelfDistillationRecipe4BThinking, "apple/SimpleSD-4B-thinking", train, eval),
		portableSSDRecipe(SimpleSelfDistillationRecipe30BA3BInstruct, "apple/SimpleSD-30b-a3b-instruct", train, eval),
	}
}

func normalizePortableSimpleSelfDistillationConfig(cfg SimpleSelfDistillationConfig) SimpleSelfDistillationConfig {
	defaults := DefaultSimpleSelfDistillationConfig()
	if cfg.SampleMaxTokens <= 0 {
		cfg.SampleMaxTokens = defaults.SampleMaxTokens
	}
	if cfg.SampleTemperature == 0 {
		cfg.SampleTemperature = defaults.SampleTemperature
	}
	if cfg.SampleTopK == 0 {
		cfg.SampleTopK = defaults.SampleTopK
	}
	if cfg.SampleTopP == 0 {
		cfg.SampleTopP = defaults.SampleTopP
	}
	if cfg.RepetitionPenalty == 0 {
		cfg.RepetitionPenalty = defaults.RepetitionPenalty
	}
	if cfg.FilterShortestPct == 0 {
		cfg.FilterShortestPct = defaults.FilterShortestPct
	}
	if cfg.DecodeTemperature != 0 && cfg.SFT.Labels == nil {
		cfg.SFT.Labels = map[string]string{}
	}
	if cfg.DecodeTemperature != 0 {
		formatted := formatPortableSimpleSelfDistillationFloat32(cfg.DecodeTemperature)
		cfg.SFT.Labels[simpleSelfDistillationDecodeTemperatureLabel] = formatted
		cfg.SFT.Labels[simpleSelfDistillationEvalTemperatureLabel] = formatted
	}
	return cfg
}

func validatePortableSimpleSelfDistillationConfig(cfg SimpleSelfDistillationConfig) error {
	if cfg.SampleTemperature <= 0 || math.IsNaN(float64(cfg.SampleTemperature)) || math.IsInf(float64(cfg.SampleTemperature), 0) {
		return core.NewError("rocm: SSD sample temperature must be positive and finite")
	}
	if cfg.SampleTemperature == 1 {
		return core.NewError("rocm: SSD sample temperature must be non-unit")
	}
	if cfg.DecodeTemperature < 0 || math.IsNaN(float64(cfg.DecodeTemperature)) || math.IsInf(float64(cfg.DecodeTemperature), 0) {
		return core.NewError("rocm: SSD decode temperature must be finite")
	}
	if cfg.SampleMaxTokens <= 0 {
		return core.NewError("rocm: SSD sample max tokens must be positive")
	}
	if cfg.RepetitionPenalty < 0 || math.IsNaN(float64(cfg.RepetitionPenalty)) || math.IsInf(float64(cfg.RepetitionPenalty), 0) {
		return core.NewError("rocm: SSD repetition penalty must be finite and non-negative")
	}
	if cfg.FilterShortestPct < 0 || cfg.FilterShortestPct > 100 || math.IsNaN(float64(cfg.FilterShortestPct)) || math.IsInf(float64(cfg.FilterShortestPct), 0) {
		return core.NewError("rocm: SSD filter shortest percent must be finite between 0 and 100")
	}
	return nil
}

func portableSimpleSelfDistillationPrompt(sample inference.DatasetSample) string {
	if prompt := strings.TrimSpace(sample.Prompt); prompt != "" {
		return prompt
	}
	if text := strings.TrimSpace(sample.Text); text != "" {
		return text
	}
	for _, message := range sample.Messages {
		if strings.TrimSpace(message.Role) == "system" {
			continue
		}
		if content := strings.TrimSpace(message.Content); content != "" {
			return content
		}
	}
	return ""
}

func portableSimpleSelfDistillationGenerateConfig(cfg SimpleSelfDistillationConfig) inference.GenerateConfig {
	return inference.GenerateConfig{
		MaxTokens:     cfg.SampleMaxTokens,
		Temperature:   cfg.SampleTemperature,
		TopK:          cfg.SampleTopK,
		TopP:          cfg.SampleTopP,
		MinP:          cfg.SampleMinP,
		RepeatPenalty: cfg.RepetitionPenalty,
	}
}

func generatePortableSimpleSelfDistillationText(ctx context.Context, model inference.TextModel, prompt string, cfg inference.GenerateConfig) (string, error) {
	builder := core.NewBuilder()
	if cfg.MaxTokens > 0 {
		builder.Grow(cfg.MaxTokens * 4)
	}
	for token := range model.Generate(ctx, prompt, portableSimpleSelfDistillationOptions(cfg)...) {
		builder.WriteString(token.Text)
	}
	if r := model.Err(); !r.OK {
		return "", r.Value.(error)
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return "", err
		}
	}
	return builder.String(), nil
}

func portableSimpleSelfDistillationOptions(cfg inference.GenerateConfig) []inference.GenerateOption {
	opts := []inference.GenerateOption{
		inference.WithMaxTokens(cfg.MaxTokens),
		inference.WithTemperature(cfg.Temperature),
		inference.WithTopK(cfg.TopK),
		inference.WithTopP(cfg.TopP),
	}
	if cfg.MinP != 0 {
		opts = append(opts, inference.WithMinP(cfg.MinP))
	}
	if cfg.RepeatPenalty != 0 {
		opts = append(opts, inference.WithRepeatPenalty(cfg.RepeatPenalty))
	}
	return opts
}

func formatPortableSimpleSelfDistillationFloat32(value float32) string {
	return strconv.FormatFloat(float64(value), 'f', -1, 32)
}

func LoadAttachedDrafterPairAsTextModel(targetPath, draftPath string, opts ...inference.LoadOption) (inference.TextModel, error) {
	return LoadAttachedDrafterPairAsTextModelBlock(targetPath, draftPath, 0, opts...)
}

func LoadAttachedDrafterPairAsTextModelWithConfig(targetPath, draftPath string, cfg ROCmLoadConfig, opts ...inference.LoadOption) (inference.TextModel, error) {
	return LoadAttachedDrafterPairAsTextModelBlockWithConfig(targetPath, draftPath, 0, cfg, opts...)
}

func LoadAttachedDrafterPairAsTextModelBlock(string, string, int, ...inference.LoadOption) (inference.TextModel, error) {
	return nil, core.E("rocm.LoadAttachedDrafterPairAsTextModelBlock", "native attached drafter execution is not available in this build", nil)
}

func LoadAttachedDrafterPairAsTextModelBlockWithConfig(string, string, int, ROCmLoadConfig, ...inference.LoadOption) (inference.TextModel, error) {
	return nil, core.E("rocm.LoadAttachedDrafterPairAsTextModelBlockWithConfig", "native attached drafter execution is not available in this build", nil)
}

func IsAttachedDrafterTextModel(inference.TextModel) bool {
	return false
}

func LoadSimpleSelfDistillationCodeBenchmarkJSONLFile(path string) ([]SimpleSelfDistillationCodeBenchmarkSample, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return LoadSimpleSelfDistillationCodeBenchmarkJSONL(data)
}

func LoadSimpleSelfDistillationLiveCodeBenchV6JSONLFile(path string) ([]SimpleSelfDistillationCodeBenchmarkSample, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return LoadSimpleSelfDistillationLiveCodeBenchV6JSONL(data)
}

func LoadSimpleSelfDistillationCodeBenchmarkJSONL(raw []byte) ([]SimpleSelfDistillationCodeBenchmarkSample, error) {
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	samples := make([]SimpleSelfDistillationCodeBenchmarkSample, 0, bytes.Count(raw, []byte{'\n'})+1)
	for index := 1; scanner.Scan(); index++ {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record portableSSDCodeBenchmarkJSONLRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return nil, core.Errorf("rocm: parse SSD code benchmark JSONL record %d: %w", index, err)
		}
		sample, ok := record.sample()
		if ok {
			samples = append(samples, sample)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(samples) == 0 {
		return nil, core.NewError("rocm: SSD code benchmark JSONL produced no samples")
	}
	return samples, nil
}

func LoadSimpleSelfDistillationLiveCodeBenchV6JSONL(raw []byte) ([]SimpleSelfDistillationCodeBenchmarkSample, error) {
	samples, err := LoadSimpleSelfDistillationCodeBenchmarkJSONL(raw)
	if err != nil {
		return nil, err
	}
	filtered := make([]SimpleSelfDistillationCodeBenchmarkSample, 0, len(samples))
	for _, sample := range samples {
		date := strings.TrimSpace(sample.Meta["contest_date"])
		if date >= "2025-02-01" && date < "2025-06-01" {
			filtered = append(filtered, sample)
		}
	}
	if len(filtered) == 0 {
		return nil, core.NewError("rocm: LiveCodeBench-v6 JSONL produced no samples")
	}
	return filtered, nil
}

func portableSSDRecipe(name, model string, train SimpleSelfDistillationConfig, eval SimpleSelfDistillationCodeBenchmarkConfig) SimpleSelfDistillationRecipe {
	return SimpleSelfDistillationRecipe{
		Name:          name,
		Model:         model,
		Dataset:       "microsoft/rStar-Coder",
		DatasetConfig: "seed_sft",
		DatasetSplit:  "train",
		Train:         train,
		Eval:          eval,
		Notes: []string{
			"Use the released model card for model-specific decode sampling when it differs from the upstream eval example.",
			"Portable builds expose the planning schema; native generation/training still requires the ROCm runtime build.",
		},
	}
}

func (record portableSSDCodeBenchmarkJSONLRecord) sample() (SimpleSelfDistillationCodeBenchmarkSample, bool) {
	prompt := firstNonEmptyPortableString(record.Prompt, record.QuestionContent, record.Question, record.Problem)
	if prompt == "" {
		return SimpleSelfDistillationCodeBenchmarkSample{}, false
	}
	if starterCode := strings.TrimSpace(record.StarterCode); starterCode != "" {
		prompt += "\n\nstarter code:\n" + starterCode
	}
	tests := appendPortableSSDTests(nil, record.Tests...)
	tests = appendPortableSSDTests(tests, record.Test)
	tests = appendPortableSSDTests(tests, record.PublicTestCases...)
	tests = appendPortableSSDTests(tests, record.PrivateTestCases...)
	meta := clonePortableSSDMeta(record.Metadata)
	if meta == nil {
		meta = map[string]string{}
	}
	if record.ContestDate != "" {
		meta["contest_date"] = record.ContestDate
	}
	if record.Difficulty != "" {
		meta["difficulty"] = record.Difficulty
	}
	if record.Platform != "" {
		meta["platform"] = record.Platform
	}
	return SimpleSelfDistillationCodeBenchmarkSample{
		ID:     firstNonEmptyPortableString(record.ID, record.QuestionID, record.TaskID),
		Prompt: prompt,
		Tests:  tests,
		Meta:   meta,
	}, true
}

func firstNonEmptyPortableString(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func appendPortableSSDTests(dst []string, values ...string) []string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			dst = append(dst, value)
		}
	}
	return dst
}

func clonePortableSSDMeta(src map[string]string) map[string]string {
	if len(src) == 0 {
		return nil
	}
	dst := make(map[string]string, len(src))
	for key, value := range src {
		dst[key] = value
	}
	return dst
}
