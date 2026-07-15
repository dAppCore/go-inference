// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"strconv"
	"strings"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	inferdecode "dappco.re/go/inference/decode"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
)

const (
	ProductionMTPDefaultDraftTokens                      = 4
	ProductionMTPFallbackDraftTokens                     = 2
	ProductionMTPPromotionMinRetainedTurns               = ProductionLaneBookTurnCount
	ProductionMTPAssistantTokenOrderingVocabSize         = modelgemma4.AssistantTokenOrderingVocabSize
	ProductionMTPAssistantOrderedEmbeddingCentroids      = modelgemma4.AssistantOrderedEmbeddingCentroids
	ProductionMTPAssistantCentroidIntermediateTopK       = modelgemma4.AssistantCentroidIntermediateTopK
	OfficialGemma4E2BRoleTarget                          = "target"
	OfficialGemma4E2BRoleAssistant                       = "assistant"
	officialGemma4E2BTargetModelID                       = modelgemma4.OfficialE2BTargetModelID
	officialGemma4E2BTargetRevision                      = modelgemma4.OfficialE2BTargetRevision
	officialGemma4E2BAssistantModelID                    = modelgemma4.OfficialE2BAssistantModelID
	officialGemma4E2BAssistantRevision                   = modelgemma4.OfficialE2BAssistantRevision
	officialGemma4E2BAssistantArchitecture               = modelgemma4.AssistantArchitecture
	productionMTPAssistantCentroidIntermediateTopKLabel  = modelgemma4.AssistantCentroidIntermediateTopKLabel
	productionMTPAssistantOrderedEmbeddingCentroidsLabel = modelgemma4.AssistantOrderedEmbeddingCentroidsLabel
	productionMTPAssistantTokenOrderingShapeLabel        = modelgemma4.AssistantTokenOrderingShape
	productionMTPDefaultDraftTokensLabel                 = "4"
	officialGemma4E2BSourceCheckedAt                     = modelgemma4.OfficialE2BSourceCheckedAt
	officialGemma4E2BTargetConfigSHA256                  = modelgemma4.OfficialE2BTargetConfigSHA256
	officialGemma4E2BAssistantConfigSHA256               = modelgemma4.OfficialE2BAssistantConfigSHA256
	productionMTPTargetRetainedVisibleTokensPerSecond    = productionLaneRetainedVisibleTokensSec
)

var (
	defaultProductionMTPDraftTokenSweepsValue = []int{1, 2, 4}
	defaultProductionMTPRequiredMetrics       = []string{
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
	}
	defaultProductionMTPPolicy = ProductionMTPPolicy{
		TargetModelID:               officialGemma4E2BTargetModelID,
		AssistantModelID:            officialGemma4E2BAssistantModelID,
		Mode:                        "mtp_attached_drafter",
		DefaultDraftTokens:          ProductionMTPDefaultDraftTokens,
		RequiredDraftTokenSweeps:    defaultProductionMTPDraftTokenSweepsValue,
		MinimumRetainedTurns:        ProductionMTPPromotionMinRetainedTurns,
		MinimumVisibleTokensPerSec:  productionMTPTargetRetainedVisibleTokensPerSecond,
		EnabledByDefault:            true,
		RequiresRetainedWorkflow:    true,
		RequiresGreedyParity:        true,
		RequiresSideBySideBenchmark: true,
		RequiredMetrics:             defaultProductionMTPRequiredMetrics,
	}
)

type OfficialGemma4E2BLock struct {
	Role            string `json:"role"`
	ModelID         string `json:"model_id"`
	Revision        string `json:"revision"`
	SourceCheckedAt string `json:"source_checked_at"`
	Architecture    string `json:"architecture"`
	ModelType       string `json:"model_type"`
	ConfigSHA256    string `json:"config_sha256"`
}

func DefaultOfficialGemma4E2BLocks() []OfficialGemma4E2BLock {
	return []OfficialGemma4E2BLock{
		{
			Role:            OfficialGemma4E2BRoleTarget,
			ModelID:         officialGemma4E2BTargetModelID,
			Revision:        officialGemma4E2BTargetRevision,
			SourceCheckedAt: officialGemma4E2BSourceCheckedAt,
			Architecture:    "Gemma4ForConditionalGeneration",
			ModelType:       "gemma4",
			ConfigSHA256:    officialGemma4E2BTargetConfigSHA256,
		},
		{
			Role:            OfficialGemma4E2BRoleAssistant,
			ModelID:         officialGemma4E2BAssistantModelID,
			Revision:        officialGemma4E2BAssistantRevision,
			SourceCheckedAt: officialGemma4E2BSourceCheckedAt,
			Architecture:    "Gemma4AssistantForCausalLM",
			ModelType:       officialGemma4E2BAssistantArchitecture,
			ConfigSHA256:    officialGemma4E2BAssistantConfigSHA256,
		},
	}
}

func OfficialGemma4E2BTargetLock() OfficialGemma4E2BLock {
	lock, _ := OfficialGemma4E2BLockByRole(OfficialGemma4E2BRoleTarget)
	return lock
}

func OfficialGemma4E2BAssistantLock() OfficialGemma4E2BLock {
	lock, _ := OfficialGemma4E2BLockByRole(OfficialGemma4E2BRoleAssistant)
	return lock
}

func officialGemma4E2BQ6TargetIdentity() inference.ModelIdentity {
	return inference.ModelIdentity{
		Path:         officialGemma4E2BTargetModelID + "-6bit",
		Architecture: "gemma4_text",
		VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
		NumLayers:    productionLaneGemma4E2BLayers,
		HiddenSize:   productionLaneGemma4E2BHiddenSize,
		QuantBits:    6,
	}
}

func officialGemma4E2BBF16AssistantIdentity() inference.ModelIdentity {
	assistant := inference.ModelIdentity{
		Path:         rocmGemma4MTPAssistantPath("E2B", "bf16"),
		Architecture: officialGemma4E2BAssistantArchitecture,
		VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
		NumLayers:    4,
		HiddenSize:   productionLaneGemma4E2BHiddenSize,
		QuantBits:    16,
		QuantType:    "bf16",
	}
	assistant.Labels = rocmGemma4MTPAssistantLabels("E2B", assistant.Labels)
	return assistant
}

func OfficialGemma4E2BLockByRole(role string) (OfficialGemma4E2BLock, bool) {
	for _, lock := range DefaultOfficialGemma4E2BLocks() {
		if lock.Role == role {
			return lock, true
		}
	}
	return OfficialGemma4E2BLock{}, false
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

type ProductionMTPPromotionEvidence struct {
	RetainedWorkflow                       bool          `json:"retained_workflow"`
	Turns                                  int           `json:"turns"`
	GreedyOutputMatches                    bool          `json:"greedy_output_matches"`
	QualityFlags                           []string      `json:"quality_flags,omitempty"`
	TargetOnlyVisibleTokensPerSec          float64       `json:"target_only_visible_tokens_per_sec,omitempty"`
	MTPVisibleTokensPerSec                 float64       `json:"mtp_visible_tokens_per_sec,omitempty"`
	MTPTargetTokensPerSec                  float64       `json:"mtp_target_tokens_per_sec,omitempty"`
	MTPWarmDecodeTokensPerSec              float64       `json:"mtp_warm_decode_tokens_per_sec,omitempty"`
	TargetOnlyWallDuration                 time.Duration `json:"target_only_wall_duration,omitempty"`
	MTPWallDuration                        time.Duration `json:"mtp_wall_duration,omitempty"`
	TargetOnlyRestoreDuration              time.Duration `json:"target_only_restore_duration,omitempty"`
	MTPRestoreDuration                     time.Duration `json:"mtp_restore_duration,omitempty"`
	TargetOnlyPeakMemoryBytes              uint64        `json:"target_only_peak_memory_bytes,omitempty"`
	MTPPeakMemoryBytes                     uint64        `json:"mtp_peak_memory_bytes,omitempty"`
	TargetOnlyActivePlusCacheMemoryBytes   uint64        `json:"target_only_active_plus_cache_memory_bytes,omitempty"`
	MTPActivePlusCacheMemoryBytes          uint64        `json:"mtp_active_plus_cache_memory_bytes,omitempty"`
	TargetOnlyEnergyJoules                 float64       `json:"target_only_energy_joules,omitempty"`
	MTPEnergyJoules                        float64       `json:"mtp_energy_joules,omitempty"`
	SameLoadPolicy                         bool          `json:"same_load_policy"`
	TargetOnlyCacheMode                    string        `json:"target_only_cache_mode"`
	MTPCacheMode                           string        `json:"mtp_cache_mode"`
	SpeculativeDraftModelPath              string        `json:"speculative_draft_model_path,omitempty"`
	SpeculativeDraftTokens                 int           `json:"speculative_draft_tokens,omitempty"`
	AttachedDrafterRetainedStateEntrypoint bool          `json:"attached_drafter_retained_state_entrypoint"`
	AttachedDrafterRetainedStateRequired   bool          `json:"attached_drafter_retained_state_required"`
	AttachedDrafterStateSource             string        `json:"attached_drafter_state_source,omitempty"`
	AttachedDrafterPromptReplayFallback    string        `json:"attached_drafter_prompt_replay_fallback,omitempty"`
	AttachedDrafterNativeAttachment        string        `json:"attached_drafter_native_attachment,omitempty"`
	AttachedDrafterNativeHandoff           string        `json:"attached_drafter_native_handoff,omitempty"`
	AttachedDrafterTargetRetainedDecode    string        `json:"attached_drafter_target_retained_decode,omitempty"`
	AttachedDrafterTargetRetainedState     string        `json:"attached_drafter_target_retained_state_decode,omitempty"`
	AttachedDrafterAssistantVerify         string        `json:"attached_drafter_assistant_verify,omitempty"`
	AttachedDrafterAssistantStateVerify    string        `json:"attached_drafter_assistant_state_verify,omitempty"`
	TargetGemma4Size                       string        `json:"target_gemma4_size,omitempty"`
	TargetGemma4QuantMode                  string        `json:"target_gemma4_quant_mode,omitempty"`
	TargetGemma4QuantGroup                 int           `json:"target_gemma4_quant_group,omitempty"`
	TargetGemma4Runtime                    string        `json:"target_gemma4_runtime,omitempty"`
	TargetGemma4GenerateStatus             string        `json:"target_gemma4_generate_status,omitempty"`
	TargetProductionQuantModelID           string        `json:"target_production_quant_model_id,omitempty"`
	TargetProductionQuantLockedModelID     string        `json:"target_production_quant_locked_model_id,omitempty"`
	AssistantGemma4Size                    string        `json:"assistant_gemma4_size,omitempty"`
	AssistantGemma4QuantMode               string        `json:"assistant_gemma4_quant_mode,omitempty"`
	AssistantGemma4QuantGroup              int           `json:"assistant_gemma4_quant_group,omitempty"`
	AssistantGemma4Runtime                 string        `json:"assistant_gemma4_runtime,omitempty"`
	AssistantGemma4GenerateStatus          string        `json:"assistant_gemma4_generate_status,omitempty"`
	AssistantProductionQuantModelID        string        `json:"assistant_production_quant_model_id,omitempty"`
	AssistantProductionQuantPack           string        `json:"assistant_production_quant_pack,omitempty"`
	AssistantProductionQuantTier           string        `json:"assistant_production_quant_tier,omitempty"`
	AssistantProductionQuantMTPAssistant   bool          `json:"assistant_production_quant_mtp_assistant"`
	AssistantProductionQuantTargetFamily   string        `json:"assistant_production_quant_target_family,omitempty"`
	AssistantArchitecture                  string        `json:"assistant_architecture,omitempty"`
	AssistantOrderedEmbeddings             bool          `json:"assistant_ordered_embeddings"`
	AssistantCentroids                     int           `json:"assistant_centroids,omitempty"`
	AssistantCentroidIntermediateTopK      int           `json:"assistant_centroid_intermediate_top_k,omitempty"`
	AssistantFourLayerDrafter              bool          `json:"assistant_four_layer_drafter"`
	AssistantTokenOrderingDType            string        `json:"assistant_token_ordering_dtype,omitempty"`
	AssistantTokenOrderingShape            []int         `json:"assistant_token_ordering_shape,omitempty"`
	Gemma4FamilyPairVerified               bool          `json:"gemma4_family_pair_verified"`
	OfficialPairVerified                   bool          `json:"official_pair_verified"`
	OfficialTargetModelID                  string        `json:"official_target_model_id,omitempty"`
	OfficialTargetRevision                 string        `json:"official_target_revision,omitempty"`
	OfficialAssistantModelID               string        `json:"official_assistant_model_id,omitempty"`
	OfficialAssistantRevision              string        `json:"official_assistant_revision,omitempty"`
	MTPDraftTokenSchedule                  []int         `json:"mtp_draft_token_schedule,omitempty"`
	MTPObservedDraftTokenSweeps            []int         `json:"mtp_observed_draft_token_sweeps,omitempty"`
	MTPProposedTokens                      int           `json:"mtp_proposed_tokens,omitempty"`
	MTPAcceptedTokens                      int           `json:"mtp_accepted_tokens,omitempty"`
	MTPRejectedTokens                      int           `json:"mtp_rejected_tokens,omitempty"`
	MTPTargetVerifyCalls                   int           `json:"mtp_target_verify_calls,omitempty"`
	MTPDraftCalls                          int           `json:"mtp_draft_calls,omitempty"`
}

// ProductionMTPDecodeRunEvidence carries measured retained-run context that is
// not present in go-inference/decode metrics. It is intentionally scalar
// metadata; historical prompt text never belongs here.
type ProductionMTPDecodeRunEvidence struct {
	RetainedWorkflow                     bool
	Turns                                int
	GreedyOutputMatches                  bool
	QualityFlags                         []string
	TargetOnlyVisibleTokensPerSec        float64
	TargetOnlyWallDuration               time.Duration
	TargetOnlyRestoreDuration            time.Duration
	MTPRestoreDuration                   time.Duration
	TargetOnlyPeakMemoryBytes            uint64
	MTPPeakMemoryBytes                   uint64
	TargetOnlyActivePlusCacheMemoryBytes uint64
	MTPActivePlusCacheMemoryBytes        uint64
	TargetOnlyEnergyJoules               float64
	MTPEnergyJoules                      float64
	SameLoadPolicy                       bool
	TargetOnlyCacheMode                  string
	MTPCacheMode                         string
	AttachedDrafterNativeAttachment      string
	AttachedDrafterNativeHandoff         string
	AttachedDrafterTargetRetainedDecode  string
	AttachedDrafterTargetRetainedState   string
	AttachedDrafterAssistantVerify       string
	AttachedDrafterAssistantStateVerify  string
	DraftTokenSchedule                   []int
	ObservedDraftTokenSweeps             []int
}

type ProductionMTPPromotionDecision struct {
	EnableByDefault bool    `json:"enable_by_default"`
	Reason          string  `json:"reason"`
	WallSpeedup     float64 `json:"wall_speedup,omitempty"`
	VisibleSpeedup  float64 `json:"visible_speedup,omitempty"`
	RestoreSpeedup  float64 `json:"restore_speedup,omitempty"`
	EnergySavings   float64 `json:"energy_savings_ratio,omitempty"`
	AcceptanceRate  float64 `json:"acceptance_rate,omitempty"`
}

// ApplyProductionMTPAttachedDrafterPlanEvidence fills the static identity and
// assistant-layout evidence proven by a validated attached-drafter plan. It
// intentionally leaves retained workflow, timing, memory, energy, and
// acceptance counters untouched; those must come from the measured benchmark.
func ApplyProductionMTPAttachedDrafterPlanEvidence(evidence *ProductionMTPPromotionEvidence, plan AttachedDrafterPlan) error {
	if evidence == nil {
		return core.E("rocm.ApplyProductionMTPAttachedDrafterPlanEvidence", "evidence is required", nil)
	}
	if err := validateProductionMTPAttachedDrafterPlan(plan); err != nil {
		return core.E("rocm.ApplyProductionMTPAttachedDrafterPlanEvidence", "attached drafter plan is invalid", err)
	}
	evidence.SpeculativeDraftTokens = plan.DraftTokens
	evidence.AttachedDrafterRetainedStateEntrypoint = true
	evidence.AttachedDrafterRetainedStateRequired = true
	evidence.AttachedDrafterStateSource = "rocm_state_session_runtime_kv"
	evidence.AttachedDrafterPromptReplayFallback = "forbidden"
	evidence.AttachedDrafterNativeAttachment = plan.NativeAttachment
	labels := cloneStringMap(plan.Labels)
	if labels == nil {
		labels = map[string]string{}
	}
	rocmAddGemma4AttachedDrafterModelLabels(labels, "attached_drafter_target", productionMTPPlanTargetIdentity(plan))
	rocmAddGemma4AttachedDrafterModelLabels(labels, "attached_drafter_assistant", productionMTPPlanDraftIdentity(plan))
	productionMTPApplyAttachedDrafterNativeLabelEvidence(evidence, labels)
	productionMTPApplyGemma4PairLabelEvidence(evidence, labels)
	evidence.SpeculativeDraftModelPath = firstNonEmptyString(
		labels["attached_drafter_assistant_model_id"],
		labels["attached.drafter.assistant.model_id"],
		evidence.SpeculativeDraftModelPath,
	)
	evidence.AssistantArchitecture = normalizeROCmArchitecture(plan.Draft.Architecture)
	evidence.AssistantOrderedEmbeddings = true
	evidence.AssistantCentroids = ProductionMTPAssistantOrderedEmbeddingCentroids
	evidence.AssistantCentroidIntermediateTopK = ProductionMTPAssistantCentroidIntermediateTopK
	evidence.AssistantFourLayerDrafter = true
	evidence.AssistantTokenOrderingDType = "int64"
	evidence.AssistantTokenOrderingShape = []int{
		ProductionMTPAssistantOrderedEmbeddingCentroids,
		ProductionMTPAssistantTokenOrderingVocabSize / ProductionMTPAssistantOrderedEmbeddingCentroids,
	}
	productionMTPApplyOfficialPairLockEvidence(evidence)
	productionMTPApplyGemma4FamilyPairEvidence(evidence)
	if evidence.SpeculativeDraftModelPath == "" {
		evidence.SpeculativeDraftModelPath = evidence.OfficialAssistantModelID
	}
	if len(evidence.MTPDraftTokenSchedule) == 0 {
		evidence.MTPDraftTokenSchedule = []int{plan.DraftTokens}
	}
	return nil
}

// ApplyProductionMTPAttachedDrafterLabelEvidence fills retained-route and
// static assistant-layout evidence from benchmark/capability labels. It accepts
// both capability-style underscore labels and benchmark-style dotted labels.
func ApplyProductionMTPAttachedDrafterLabelEvidence(evidence *ProductionMTPPromotionEvidence, labels map[string]string) error {
	if evidence == nil {
		return core.E("rocm.ApplyProductionMTPAttachedDrafterLabelEvidence", "evidence is required", nil)
	}
	if labels == nil {
		return core.E("rocm.ApplyProductionMTPAttachedDrafterLabelEvidence", "labels are required", nil)
	}
	entrypoint := firstNonEmptyString(labels["attached_drafter_retained_state_entrypoint"], labels["attached.drafter.retained_state_entrypoint"], labels["engine_attached_drafter_retained_state_entrypoint"])
	required := firstNonEmptyString(labels["attached_drafter_retained_state_required"], labels["attached.drafter.retained_state_required"], labels["engine_attached_drafter_retained_state_required"])
	source := firstNonEmptyString(labels["attached_drafter_state_source"], labels["attached.drafter.state_source"], labels["engine_attached_drafter_state_source"])
	fallback := firstNonEmptyString(labels["attached_drafter_prompt_replay_fallback"], labels["attached.drafter.prompt_replay_fallback"], labels["engine_attached_drafter_prompt_replay_fallback"])
	if fallback == "" && labels["engine_attached_drafter_prompt_replay_refused"] == "true" {
		fallback = "forbidden"
	}
	evidence.AttachedDrafterRetainedStateEntrypoint = entrypoint == hipKernelStatusLinked
	evidence.AttachedDrafterRetainedStateRequired = required == "true"
	evidence.AttachedDrafterStateSource = source
	evidence.AttachedDrafterPromptReplayFallback = fallback
	productionMTPApplyAttachedDrafterNativeLabelEvidence(evidence, labels)
	productionMTPApplyGemma4PairLabelEvidence(evidence, labels)
	if err := productionMTPApplyBoolAlias(labels, []string{"assistant_production_quant_mtp_assistant", "draft_production_quant_mtp_assistant", "attached_drafter_assistant_production_quant_mtp_assistant", "attached_drafter_draft_production_quant_mtp_assistant", "attached.drafter.assistant.production_quant_mtp_assistant", "attached.drafter.draft.production_quant_mtp_assistant"}, &evidence.AssistantProductionQuantMTPAssistant); err != nil {
		return err
	}
	if err := productionMTPApplyIntAlias(labels, []string{"target_gemma4_quant_group", "attached_drafter_target_gemma4_quant_group", "attached.drafter.target.gemma4_quant_group"}, &evidence.TargetGemma4QuantGroup); err != nil {
		return err
	}
	if err := productionMTPApplyIntAlias(labels, []string{"assistant_gemma4_quant_group", "draft_gemma4_quant_group", "attached_drafter_assistant_gemma4_quant_group", "attached_drafter_draft_gemma4_quant_group", "attached.drafter.assistant.gemma4_quant_group", "attached.drafter.draft.gemma4_quant_group"}, &evidence.AssistantGemma4QuantGroup); err != nil {
		return err
	}
	evidence.SpeculativeDraftModelPath = firstNonEmptyString(
		labels["speculative_draft_model_path"],
		labels["attached_drafter_assistant_model_id"],
		labels["attached.drafter.assistant.model_id"],
		labels["attached_drafter_official_assistant_model_id"],
		labels["attached.drafter.official_assistant_model_id"],
		evidence.SpeculativeDraftModelPath,
	)
	evidence.AssistantArchitecture = firstNonEmptyString(labels["assistant_architecture"], labels["attached_drafter_assistant_architecture"], labels["attached.drafter.assistant_architecture"], labels["engine_attached_drafter_assistant_architecture"], evidence.AssistantArchitecture)
	evidence.AssistantTokenOrderingDType = firstNonEmptyString(labels["assistant_token_ordering_dtype"], labels["attached_drafter_assistant_token_ordering_dtype"], labels["attached.drafter.assistant_token_ordering_dtype"], labels["engine_attached_drafter_assistant_token_ordering_dtype"], evidence.AssistantTokenOrderingDType)
	evidence.OfficialTargetModelID = firstNonEmptyString(labels["official_target_model_id"], labels["attached_drafter_official_target_model_id"], labels["attached.drafter.official_target_model_id"], evidence.OfficialTargetModelID)
	evidence.OfficialTargetRevision = firstNonEmptyString(labels["official_target_revision"], labels["attached_drafter_official_target_revision"], labels["attached.drafter.official_target_revision"], evidence.OfficialTargetRevision)
	evidence.OfficialAssistantModelID = firstNonEmptyString(labels["official_assistant_model_id"], labels["attached_drafter_official_assistant_model_id"], labels["attached.drafter.official_assistant_model_id"], evidence.OfficialAssistantModelID)
	evidence.OfficialAssistantRevision = firstNonEmptyString(labels["official_assistant_revision"], labels["attached_drafter_official_assistant_revision"], labels["attached.drafter.official_assistant_revision"], evidence.OfficialAssistantRevision)
	if err := productionMTPApplyBoolAlias(labels, []string{"assistant_ordered_embeddings", "attached_drafter_assistant_ordered_embeddings", "attached.drafter.assistant_ordered_embeddings", "engine_attached_drafter_ordered_embeddings"}, &evidence.AssistantOrderedEmbeddings); err != nil {
		return err
	}
	if err := productionMTPApplyBoolAlias(labels, []string{"assistant_four_layer_drafter", "attached_drafter_assistant_four_layer_drafter", "attached.drafter.assistant_four_layer_drafter", "engine_attached_drafter_four_layer_drafter"}, &evidence.AssistantFourLayerDrafter); err != nil {
		return err
	}
	if err := productionMTPApplyBoolAlias(labels, []string{"official_pair_verified", "attached_drafter_official_pair_verified", "attached.drafter.official_pair_verified"}, &evidence.OfficialPairVerified); err != nil {
		return err
	}
	if err := productionMTPApplyBoolAlias(labels, []string{"gemma4_family_pair_verified", "attached_drafter_gemma4_family_pair_verified", "attached.drafter.gemma4_family_pair_verified"}, &evidence.Gemma4FamilyPairVerified); err != nil {
		return err
	}
	if err := productionMTPApplyIntAlias(labels, []string{"speculative_draft_tokens", "attached_drafter_speculative_draft_tokens", "attached.drafter.speculative_draft_tokens", "engine_attached_drafter_default_draft_tokens"}, &evidence.SpeculativeDraftTokens); err != nil {
		return err
	}
	if err := productionMTPApplyIntAlias(labels, []string{"assistant_centroids", "attached_drafter_assistant_centroids", "attached.drafter.assistant_centroids", "engine_attached_drafter_assistant_centroids"}, &evidence.AssistantCentroids); err != nil {
		return err
	}
	if err := productionMTPApplyIntAlias(labels, []string{"assistant_centroid_intermediate_top_k", "attached_drafter_assistant_centroid_intermediate_top_k", "attached.drafter.assistant_centroid_intermediate_top_k", "engine_attached_drafter_assistant_centroid_intermediate_top_k"}, &evidence.AssistantCentroidIntermediateTopK); err != nil {
		return err
	}
	if value := firstNonEmptyString(labels["assistant_token_ordering_shape"], labels["attached_drafter_assistant_token_ordering_shape"], labels["attached.drafter.assistant_token_ordering_shape"], labels["engine_attached_drafter_assistant_token_ordering_shape"]); value != "" {
		shape, err := parseProductionMTPShape(value)
		if err != nil {
			return core.E("rocm.ApplyProductionMTPAttachedDrafterLabelEvidence", "parse assistant_token_ordering_shape", err)
		}
		evidence.AssistantTokenOrderingShape = shape
	}
	productionMTPApplyGemma4FamilyPairEvidence(evidence)
	return nil
}

// ApplyProductionMTPLabelEvidence fills complete MTP promotion evidence from a
// measured benchmark/capability label row. Static attached-drafter identity is
// parsed by ApplyProductionMTPAttachedDrafterLabelEvidence; measured counters
// and timings must still be present in the row before promotion can pass.
func ApplyProductionMTPLabelEvidence(evidence *ProductionMTPPromotionEvidence, labels map[string]string) error {
	if evidence == nil {
		return core.E("rocm.ApplyProductionMTPLabelEvidence", "evidence is required", nil)
	}
	if labels == nil {
		return core.E("rocm.ApplyProductionMTPLabelEvidence", "labels are required", nil)
	}
	if err := ApplyProductionMTPAttachedDrafterLabelEvidence(evidence, labels); err != nil {
		return err
	}
	if err := productionMTPApplyBoolLabel(labels, []string{"retained_workflow", "mtp_retained_workflow"}, &evidence.RetainedWorkflow); err != nil {
		return err
	}
	if err := productionMTPApplyBoolLabel(labels, []string{"greedy_output_matches", "mtp_greedy_output_matches"}, &evidence.GreedyOutputMatches); err != nil {
		return err
	}
	if err := productionMTPApplyBoolLabel(labels, []string{"same_load_policy", "mtp_same_load_policy"}, &evidence.SameLoadPolicy); err != nil {
		return err
	}
	if err := productionMTPApplyIntLabel(labels, []string{"turns", "mtp_turns"}, &evidence.Turns); err != nil {
		return err
	}
	if err := productionMTPApplyIntLabel(labels, []string{"mtp_proposed_tokens"}, &evidence.MTPProposedTokens); err != nil {
		return err
	}
	if err := productionMTPApplyIntLabel(labels, []string{"mtp_accepted_tokens"}, &evidence.MTPAcceptedTokens); err != nil {
		return err
	}
	if err := productionMTPApplyIntLabel(labels, []string{"mtp_rejected_tokens"}, &evidence.MTPRejectedTokens); err != nil {
		return err
	}
	if err := productionMTPApplyIntLabel(labels, []string{"mtp_target_verify_calls"}, &evidence.MTPTargetVerifyCalls); err != nil {
		return err
	}
	if err := productionMTPApplyIntLabel(labels, []string{"mtp_draft_calls"}, &evidence.MTPDraftCalls); err != nil {
		return err
	}
	if err := productionMTPApplyUint64Label(labels, []string{"target_only_peak_memory_bytes", "mtp_target_only_peak_memory_bytes"}, &evidence.TargetOnlyPeakMemoryBytes); err != nil {
		return err
	}
	if err := productionMTPApplyUint64Label(labels, []string{"mtp_peak_memory_bytes"}, &evidence.MTPPeakMemoryBytes); err != nil {
		return err
	}
	if err := productionMTPApplyUint64Label(labels, []string{"target_only_active_plus_cache_memory_bytes", "mtp_target_only_active_plus_cache_memory_bytes"}, &evidence.TargetOnlyActivePlusCacheMemoryBytes); err != nil {
		return err
	}
	if err := productionMTPApplyUint64Label(labels, []string{"mtp_active_plus_cache_memory_bytes"}, &evidence.MTPActivePlusCacheMemoryBytes); err != nil {
		return err
	}
	if err := productionMTPApplyFloat64Label(labels, []string{"target_only_visible_tokens_per_sec", "mtp_target_only_visible_tokens_per_sec"}, &evidence.TargetOnlyVisibleTokensPerSec); err != nil {
		return err
	}
	if err := productionMTPApplyFloat64Label(labels, []string{"mtp_visible_tokens_per_sec"}, &evidence.MTPVisibleTokensPerSec); err != nil {
		return err
	}
	if err := productionMTPApplyFloat64Label(labels, []string{"mtp_target_tokens_per_sec"}, &evidence.MTPTargetTokensPerSec); err != nil {
		return err
	}
	if err := productionMTPApplyFloat64Label(labels, []string{"mtp_warm_decode_tokens_per_sec"}, &evidence.MTPWarmDecodeTokensPerSec); err != nil {
		return err
	}
	if err := productionMTPApplyFloat64Label(labels, []string{"target_only_energy_joules", "mtp_target_only_energy_joules"}, &evidence.TargetOnlyEnergyJoules); err != nil {
		return err
	}
	if err := productionMTPApplyFloat64Label(labels, []string{"mtp_energy_joules"}, &evidence.MTPEnergyJoules); err != nil {
		return err
	}
	if err := productionMTPApplyDurationLabel(labels, []string{"target_only_wall_duration", "mtp_target_only_wall_duration"}, &evidence.TargetOnlyWallDuration); err != nil {
		return err
	}
	if err := productionMTPApplyDurationLabel(labels, []string{"mtp_wall_duration"}, &evidence.MTPWallDuration); err != nil {
		return err
	}
	if err := productionMTPApplyDurationLabel(labels, []string{"target_only_restore_duration", "mtp_target_only_restore_duration"}, &evidence.TargetOnlyRestoreDuration); err != nil {
		return err
	}
	if err := productionMTPApplyDurationLabel(labels, []string{"mtp_restore_duration"}, &evidence.MTPRestoreDuration); err != nil {
		return err
	}
	if _, value := productionFirstLabel(labels, []string{"target_only_cache_mode", "mtp_target_only_cache_mode"}); value != "" {
		evidence.TargetOnlyCacheMode = value
	}
	if _, value := productionFirstLabel(labels, []string{"mtp_cache_mode"}); value != "" {
		evidence.MTPCacheMode = value
	}
	if value := labels["mtp_draft_token_schedule"]; value != "" {
		parsed, err := parseProductionMTPIntList(value)
		if err != nil {
			return err
		}
		evidence.MTPDraftTokenSchedule = parsed
	}
	if value := labels["mtp_observed_draft_token_sweeps"]; value != "" {
		parsed, err := parseProductionMTPIntList(value)
		if err != nil {
			return err
		}
		evidence.MTPObservedDraftTokenSweeps = parsed
	}
	if value := labels["quality_flags"]; value != "" {
		evidence.QualityFlags = splitProductionCSVLabel(value)
	}
	return nil
}

func ValidateProductionMTPPromotionMetricLabels(labels map[string]string) error {
	_, err := EvaluateProductionMTPPromotionMetricLabels(labels)
	return err
}

func EvaluateProductionMTPPromotionMetricLabels(labels map[string]string) (ProductionMTPPromotionDecision, error) {
	return EvaluateProductionMTPPromotionMetricLabelsWithPolicy(DefaultProductionMTPPolicy(), labels)
}

func EvaluateProductionMTPPromotionMetricLabelsWithPolicy(policy ProductionMTPPolicy, labels map[string]string) (ProductionMTPPromotionDecision, error) {
	if err := ValidateProductionMTPRequiredMetricLabels(labels); err != nil {
		return ProductionMTPPromotionDecision{}, err
	}
	var evidence ProductionMTPPromotionEvidence
	if err := ApplyProductionMTPLabelEvidence(&evidence, labels); err != nil {
		return ProductionMTPPromotionDecision{}, err
	}
	return EvaluateProductionMTPPromotion(policy, evidence), nil
}

// ApplyProductionMTPDecodeRunEvidence fills measured MTP counters and timings
// from a retained attached-drafter decode result plus scalar benchmark context.
// It does not inspect or replay result.Prompt; callers must pass only measured
// runtime state and new-turn metadata.
func ApplyProductionMTPDecodeRunEvidence(evidence *ProductionMTPPromotionEvidence, result inferdecode.Result, run ProductionMTPDecodeRunEvidence) error {
	if evidence == nil {
		return core.E("rocm.ApplyProductionMTPDecodeRunEvidence", "evidence is required", nil)
	}
	if result.Mode != inferdecode.ModeSpeculative {
		return core.E("rocm.ApplyProductionMTPDecodeRunEvidence", "decode result must be speculative MTP", nil)
	}
	metrics := result.Metrics
	proposed := metrics.DraftTokens
	if proposed == 0 {
		proposed = metrics.AcceptedTokens + metrics.RejectedTokens
	}
	if proposed < 0 || metrics.AcceptedTokens < 0 || metrics.RejectedTokens < 0 || metrics.TargetCalls < 0 || metrics.DraftCalls < 0 {
		return core.E("rocm.ApplyProductionMTPDecodeRunEvidence", "decode metrics must be non-negative", nil)
	}
	if proposed > 0 && metrics.AcceptedTokens+metrics.RejectedTokens > 0 && metrics.AcceptedTokens+metrics.RejectedTokens != proposed {
		return core.E("rocm.ApplyProductionMTPDecodeRunEvidence", "accepted/rejected tokens must account for proposed draft tokens", nil)
	}
	evidence.RetainedWorkflow = run.RetainedWorkflow
	evidence.Turns = run.Turns
	evidence.GreedyOutputMatches = run.GreedyOutputMatches
	evidence.QualityFlags = append([]string(nil), run.QualityFlags...)
	evidence.TargetOnlyVisibleTokensPerSec = run.TargetOnlyVisibleTokensPerSec
	evidence.TargetOnlyWallDuration = run.TargetOnlyWallDuration
	evidence.TargetOnlyRestoreDuration = run.TargetOnlyRestoreDuration
	evidence.MTPRestoreDuration = run.MTPRestoreDuration
	evidence.TargetOnlyPeakMemoryBytes = run.TargetOnlyPeakMemoryBytes
	evidence.MTPPeakMemoryBytes = run.MTPPeakMemoryBytes
	evidence.TargetOnlyActivePlusCacheMemoryBytes = run.TargetOnlyActivePlusCacheMemoryBytes
	evidence.MTPActivePlusCacheMemoryBytes = run.MTPActivePlusCacheMemoryBytes
	evidence.TargetOnlyEnergyJoules = run.TargetOnlyEnergyJoules
	evidence.MTPEnergyJoules = run.MTPEnergyJoules
	evidence.SameLoadPolicy = run.SameLoadPolicy
	evidence.TargetOnlyCacheMode = run.TargetOnlyCacheMode
	evidence.MTPCacheMode = run.MTPCacheMode
	evidence.AttachedDrafterNativeAttachment = firstNonEmptyString(run.AttachedDrafterNativeAttachment, evidence.AttachedDrafterNativeAttachment)
	evidence.AttachedDrafterNativeHandoff = firstNonEmptyString(run.AttachedDrafterNativeHandoff, evidence.AttachedDrafterNativeHandoff)
	evidence.AttachedDrafterTargetRetainedDecode = firstNonEmptyString(run.AttachedDrafterTargetRetainedDecode, evidence.AttachedDrafterTargetRetainedDecode)
	evidence.AttachedDrafterTargetRetainedState = firstNonEmptyString(run.AttachedDrafterTargetRetainedState, evidence.AttachedDrafterTargetRetainedState)
	evidence.AttachedDrafterAssistantVerify = firstNonEmptyString(run.AttachedDrafterAssistantVerify, evidence.AttachedDrafterAssistantVerify)
	evidence.AttachedDrafterAssistantStateVerify = firstNonEmptyString(run.AttachedDrafterAssistantStateVerify, evidence.AttachedDrafterAssistantStateVerify)
	evidence.MTPDraftTokenSchedule = append([]int(nil), run.DraftTokenSchedule...)
	evidence.MTPObservedDraftTokenSweeps = append([]int(nil), run.ObservedDraftTokenSweeps...)
	evidence.MTPProposedTokens = proposed
	evidence.MTPAcceptedTokens = metrics.AcceptedTokens
	evidence.MTPRejectedTokens = metrics.RejectedTokens
	evidence.MTPTargetVerifyCalls = metrics.TargetCalls
	evidence.MTPDraftCalls = metrics.DraftCalls
	evidence.MTPWallDuration = metrics.Duration
	if evidence.MTPWallDuration == 0 {
		evidence.MTPWallDuration = metrics.TargetDuration + metrics.DraftDuration
	}
	evidence.MTPVisibleTokensPerSec = tokensPerSecond(metrics.EmittedTokens, evidence.MTPWallDuration)
	evidence.MTPTargetTokensPerSec = tokensPerSecond(metrics.TargetTokens, metrics.TargetDuration)
	if evidence.MTPTargetTokensPerSec == 0 {
		evidence.MTPTargetTokensPerSec = tokensPerSecond(metrics.EmittedTokens, metrics.TargetDuration)
	}
	evidence.MTPWarmDecodeTokensPerSec = tokensPerSecond(metrics.EmittedTokens, evidence.MTPWallDuration)
	return nil
}

func DefaultProductionMTPPolicy() ProductionMTPPolicy {
	policy := defaultProductionMTPPolicy
	policy.RequiredDraftTokenSweeps = append([]int(nil), policy.RequiredDraftTokenSweeps...)
	policy.RequiredMetrics = append([]string(nil), policy.RequiredMetrics...)
	return policy
}

func EvaluateProductionMTPPromotion(policy ProductionMTPPolicy, evidence ProductionMTPPromotionEvidence) ProductionMTPPromotionDecision {
	if policy.MinimumRetainedTurns == 0 {
		policy = DefaultProductionMTPPolicy()
	}
	decision := ProductionMTPPromotionDecision{
		WallSpeedup:     durationSpeedup(evidence.TargetOnlyWallDuration, evidence.MTPWallDuration),
		VisibleSpeedup:  ratioSpeedup(evidence.MTPVisibleTokensPerSec, evidence.TargetOnlyVisibleTokensPerSec),
		RestoreSpeedup:  durationSpeedup(evidence.TargetOnlyRestoreDuration, evidence.MTPRestoreDuration),
		EnergySavings:   ratioSavings(evidence.TargetOnlyEnergyJoules, evidence.MTPEnergyJoules),
		AcceptanceRate:  ratioSpeedup(float64(evidence.MTPAcceptedTokens), float64(evidence.MTPProposedTokens)),
		EnableByDefault: false,
	}
	if policy.RequiresRetainedWorkflow && !evidence.RetainedWorkflow {
		decision.Reason = "retained workflow evidence is required before MTP promotion"
		return decision
	}
	if evidence.Turns < policy.MinimumRetainedTurns {
		decision.Reason = "retained workflow turn count is below the MTP promotion minimum"
		return decision
	}
	if policy.RequiresGreedyParity && !evidence.GreedyOutputMatches {
		decision.Reason = "greedy output parity is required before MTP promotion"
		return decision
	}
	if len(evidence.QualityFlags) > 0 {
		decision.Reason = "quality flags must be empty before MTP promotion"
		return decision
	}
	if policy.RequiresSideBySideBenchmark && (decision.WallSpeedup == 0 || decision.VisibleSpeedup == 0) {
		decision.Reason = "side-by-side target-only and MTP wall/visible metrics are required"
		return decision
	}
	if evidence.MTPVisibleTokensPerSec < policy.MinimumVisibleTokensPerSec {
		decision.Reason = "MTP visible throughput is below the ROCm production minimum"
		return decision
	}
	if evidence.SpeculativeDraftModelPath == "" || evidence.SpeculativeDraftTokens <= 0 || len(evidence.MTPDraftTokenSchedule) == 0 {
		decision.Reason = "MTP draft model, draft token count, and schedule evidence are required"
		return decision
	}
	if !productionMTPHasRetainedRouteEvidence(evidence) {
		decision.Reason = "MTP retained attached-drafter route evidence is required"
		return decision
	}
	if issue := productionMTPNativeHandoffEvidenceIssue(evidence); issue != "" {
		decision.Reason = issue
		return decision
	}
	for _, draftTokens := range evidence.MTPDraftTokenSchedule {
		if draftTokens <= 0 {
			decision.Reason = "MTP draft token schedule must contain positive draft counts"
			return decision
		}
	}
	if !productionMTPObservedDraftTokenSweepsCover(requiredProductionMTPDraftTokenSweeps(policy), evidence.MTPObservedDraftTokenSweeps) {
		decision.Reason = "MTP draft-token sweep evidence is incomplete"
		return decision
	}
	if evidence.MTPTargetTokensPerSec <= 0 || evidence.MTPWarmDecodeTokensPerSec <= 0 {
		decision.Reason = "MTP target-verify and warm-decode throughput evidence are required"
		return decision
	}
	if evidence.MTPProposedTokens <= 0 || evidence.MTPTargetVerifyCalls <= 0 || evidence.MTPDraftCalls <= 0 {
		decision.Reason = "MTP proposed-token, target-verify, and draft-call counters are required"
		return decision
	}
	if evidence.MTPAcceptedTokens < 0 || evidence.MTPRejectedTokens < 0 || evidence.MTPAcceptedTokens+evidence.MTPRejectedTokens != evidence.MTPProposedTokens {
		decision.Reason = "MTP accepted/rejected counters must account for every proposed token"
		return decision
	}
	if evidence.MTPAcceptedTokens == 0 {
		decision.Reason = "MTP accepted draft tokens are required before promotion"
		return decision
	}
	if evidence.TargetOnlyRestoreDuration <= 0 || evidence.MTPRestoreDuration <= 0 ||
		evidence.TargetOnlyPeakMemoryBytes == 0 || evidence.MTPPeakMemoryBytes == 0 ||
		evidence.TargetOnlyEnergyJoules <= 0 || evidence.MTPEnergyJoules <= 0 {
		decision.Reason = "MTP restore, memory, and energy evidence are required"
		return decision
	}
	if evidence.TargetOnlyActivePlusCacheMemoryBytes == 0 || evidence.MTPActivePlusCacheMemoryBytes == 0 {
		decision.Reason = "MTP active+cache memory evidence is required"
		return decision
	}
	if decision.WallSpeedup <= 1 || decision.VisibleSpeedup <= 1 {
		decision.Reason = "MTP must be faster than target-only on retained wall time and visible throughput"
		return decision
	}
	if decision.EnergySavings <= 0 {
		decision.Reason = "MTP must not increase estimated energy before promotion"
		return decision
	}
	if !productionMTPHasLoadPolicyEvidence(evidence) {
		decision.Reason = "MTP load policy evidence is required"
		return decision
	}
	if issue := productionMTPAssistantLayoutEvidenceIssue(evidence); issue != "" {
		decision.Reason = issue
		return decision
	}
	if !productionMTPHasGemma4FamilyPairEvidence(policy, evidence) {
		decision.Reason = "verified Gemma 4 family target+assistant pair evidence is required"
		return decision
	}
	if !productionMTPHasGemma4AssistantProductionPackEvidence(evidence) {
		decision.Reason = "Gemma 4 MTP assistant production pack evidence is required"
		return decision
	}
	decision.EnableByDefault = policy.EnabledByDefault
	decision.Reason = "MTP retained workflow is faster than target-only with greedy parity"
	return decision
}

func durationSpeedup(baseline, candidate time.Duration) float64 {
	if baseline <= 0 || candidate <= 0 {
		return 0
	}
	return float64(baseline) / float64(candidate)
}

func ratioSpeedup(candidate, baseline float64) float64 {
	if baseline <= 0 || candidate <= 0 {
		return 0
	}
	return candidate / baseline
}

func ratioSavings(baseline, candidate float64) float64 {
	if baseline <= 0 || candidate <= 0 || candidate >= baseline {
		return 0
	}
	return 1 - candidate/baseline
}

func productionMTPHasLoadPolicyEvidence(evidence ProductionMTPPromotionEvidence) bool {
	return evidence.SameLoadPolicy &&
		evidence.TargetOnlyCacheMode != "" &&
		evidence.TargetOnlyCacheMode == evidence.MTPCacheMode
}

func productionMTPApplyBoolAlias(labels map[string]string, keys []string, target *bool) error {
	key, value := productionFirstLabel(labels, keys)
	if value == "" {
		return nil
	}
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return core.E("rocm.ApplyProductionMTPAttachedDrafterLabelEvidence", "parse "+key, err)
	}
	*target = parsed
	return nil
}

func productionMTPApplyIntAlias(labels map[string]string, keys []string, target *int) error {
	key, value := productionFirstLabel(labels, keys)
	if value == "" {
		return nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return core.E("rocm.ApplyProductionMTPAttachedDrafterLabelEvidence", "parse "+key, err)
	}
	*target = parsed
	return nil
}

func productionMTPApplyBoolLabel(labels map[string]string, keys []string, target *bool) error {
	key, value := productionFirstLabel(labels, keys)
	if value == "" {
		return nil
	}
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return core.E("rocm.ApplyProductionMTPLabelEvidence", "parse "+key, err)
	}
	*target = parsed
	return nil
}

func productionMTPApplyIntLabel(labels map[string]string, keys []string, target *int) error {
	key, value := productionFirstLabel(labels, keys)
	if value == "" {
		return nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return core.E("rocm.ApplyProductionMTPLabelEvidence", "parse "+key, err)
	}
	*target = parsed
	return nil
}

func productionMTPApplyUint64Label(labels map[string]string, keys []string, target *uint64) error {
	key, value := productionFirstLabel(labels, keys)
	if value == "" {
		return nil
	}
	parsed, err := strconv.ParseUint(value, 10, 64)
	if err != nil {
		return core.E("rocm.ApplyProductionMTPLabelEvidence", "parse "+key, err)
	}
	*target = parsed
	return nil
}

func productionMTPApplyFloat64Label(labels map[string]string, keys []string, target *float64) error {
	key, value := productionFirstLabel(labels, keys)
	if value == "" {
		return nil
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return core.E("rocm.ApplyProductionMTPLabelEvidence", "parse "+key, err)
	}
	*target = parsed
	return nil
}

func productionMTPApplyDurationLabel(labels map[string]string, keys []string, target *time.Duration) error {
	key, value := productionFirstLabel(labels, keys)
	if value == "" {
		return nil
	}
	parsed, err := time.ParseDuration(value)
	if err != nil {
		seconds, secondsErr := strconv.ParseFloat(value, 64)
		if secondsErr != nil {
			return core.E("rocm.ApplyProductionMTPLabelEvidence", "parse "+key, err)
		}
		parsed = time.Duration(seconds * float64(time.Second))
	}
	*target = parsed
	return nil
}

func parseProductionMTPIntList(value string) ([]int, error) {
	parts := splitProductionCSVLabel(value)
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		parsed, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil {
			return nil, core.E("rocm.ApplyProductionMTPLabelEvidence", "parse int list", err)
		}
		out = append(out, parsed)
	}
	return out, nil
}

func parseProductionMTPShape(value string) ([]int, error) {
	return parseProductionMTPIntList(strings.ReplaceAll(value, "x", ","))
}

func productionMTPApplyAttachedDrafterNativeLabelEvidence(evidence *ProductionMTPPromotionEvidence, labels map[string]string) {
	if evidence == nil || labels == nil {
		return
	}
	evidence.AttachedDrafterNativeAttachment = firstNonEmptyString(
		labels["attached_drafter_native_attachment"],
		labels["attached.drafter.native_attachment"],
		labels["engine_attached_drafter_native_attachment"],
		evidence.AttachedDrafterNativeAttachment,
	)
	evidence.AttachedDrafterNativeHandoff = firstNonEmptyString(
		labels["attached_drafter_native_handoff"],
		labels["attached.drafter.native_handoff"],
		labels["engine_attached_drafter_native_handoff"],
		evidence.AttachedDrafterNativeHandoff,
	)
	evidence.AttachedDrafterTargetRetainedDecode = firstNonEmptyString(
		labels["attached_drafter_target_retained_decode"],
		labels["attached.drafter.target_retained_decode"],
		labels["engine_attached_drafter_target_retained_decode"],
		evidence.AttachedDrafterTargetRetainedDecode,
	)
	evidence.AttachedDrafterTargetRetainedState = firstNonEmptyString(
		labels["attached_drafter_target_retained_state_decode"],
		labels["attached.drafter.target_retained_state_decode"],
		labels["engine_attached_drafter_target_retained_state_decode"],
		evidence.AttachedDrafterTargetRetainedState,
	)
	if evidence.AttachedDrafterTargetRetainedState == "" {
		evidence.AttachedDrafterTargetRetainedState = evidence.AttachedDrafterTargetRetainedDecode
	}
	evidence.AttachedDrafterAssistantVerify = firstNonEmptyString(
		labels["attached_drafter_assistant_verify"],
		labels["attached.drafter.assistant_verify"],
		labels["engine_attached_drafter_assistant_verify"],
		evidence.AttachedDrafterAssistantVerify,
	)
	evidence.AttachedDrafterAssistantStateVerify = firstNonEmptyString(
		labels["attached_drafter_assistant_state_verify"],
		labels["attached.drafter.assistant_state_verify"],
		labels["engine_attached_drafter_assistant_state_verify"],
		evidence.AttachedDrafterAssistantStateVerify,
	)
	if evidence.AttachedDrafterAssistantStateVerify == "" {
		evidence.AttachedDrafterAssistantStateVerify = evidence.AttachedDrafterAssistantVerify
	}
}

func productionMTPHasRetainedRouteEvidence(evidence ProductionMTPPromotionEvidence) bool {
	return evidence.AttachedDrafterRetainedStateEntrypoint &&
		evidence.AttachedDrafterRetainedStateRequired &&
		evidence.AttachedDrafterStateSource == "rocm_state_session_runtime_kv" &&
		evidence.AttachedDrafterPromptReplayFallback == "forbidden"
}

func productionMTPNativeHandoffEvidenceIssue(evidence ProductionMTPPromotionEvidence) string {
	if evidence.AttachedDrafterNativeAttachment != hipKernelStatusLinked ||
		evidence.AttachedDrafterNativeHandoff == "" ||
		evidence.AttachedDrafterNativeHandoff == attachedDrafterNativeHandoffPendingTargetDecode ||
		evidence.AttachedDrafterNativeHandoff == attachedDrafterNativeHandoffTargetDecodeOnly {
		return "MTP native attached-drafter handoff evidence is required"
	}
	if evidence.AttachedDrafterTargetRetainedDecode != hipKernelStatusLinked ||
		evidence.AttachedDrafterTargetRetainedState != hipKernelStatusLinked {
		return "MTP retained target decode evidence is required"
	}
	if evidence.AttachedDrafterAssistantVerify != hipKernelStatusLinked ||
		evidence.AttachedDrafterAssistantStateVerify != hipKernelStatusLinked {
		return "MTP retained assistant verifier evidence is required"
	}
	return ""
}

func productionMTPAssistantLayoutEvidenceIssue(evidence ProductionMTPPromotionEvidence) string {
	if evidence.AssistantArchitecture != officialGemma4E2BAssistantArchitecture {
		return "official Gemma 4 assistant architecture evidence is required"
	}
	if !evidence.AssistantOrderedEmbeddings ||
		evidence.AssistantCentroids != ProductionMTPAssistantOrderedEmbeddingCentroids ||
		evidence.AssistantCentroidIntermediateTopK != ProductionMTPAssistantCentroidIntermediateTopK {
		return "official Gemma 4 assistant ordered-embedding evidence is required"
	}
	if !evidence.AssistantFourLayerDrafter {
		return "official Gemma 4 assistant four-layer drafter evidence is required"
	}
	if !productionMTPHasAssistantTokenOrderingEvidence(evidence) {
		return "official Gemma 4 assistant token-ordering evidence is required"
	}
	return ""
}

func productionMTPHasAssistantTokenOrderingEvidence(evidence ProductionMTPPromotionEvidence) bool {
	if evidence.AssistantTokenOrderingDType != "int64" && evidence.AssistantTokenOrderingDType != "I64" {
		return false
	}
	tokensPerCentroid := ProductionMTPAssistantTokenOrderingVocabSize / ProductionMTPAssistantOrderedEmbeddingCentroids
	shape := evidence.AssistantTokenOrderingShape
	return len(shape) == 1 && shape[0] == ProductionMTPAssistantTokenOrderingVocabSize ||
		len(shape) == 2 && shape[0] == ProductionMTPAssistantOrderedEmbeddingCentroids && shape[1] == tokensPerCentroid
}

func productionMTPHasOfficialPairEvidence(policy ProductionMTPPolicy, evidence ProductionMTPPromotionEvidence) bool {
	return evidence.OfficialPairVerified &&
		evidence.OfficialTargetModelID == policy.TargetModelID &&
		evidence.OfficialTargetRevision == officialGemma4E2BTargetRevision &&
		evidence.OfficialAssistantModelID == policy.AssistantModelID &&
		evidence.OfficialAssistantRevision == officialGemma4E2BAssistantRevision &&
		productionMTPHasOfficialGemma4PairLabels(evidence)
}

func productionMTPHasGemma4FamilyPairEvidence(_ ProductionMTPPolicy, evidence ProductionMTPPromotionEvidence) bool {
	return evidence.Gemma4FamilyPairVerified && productionMTPHasGemma4FamilyPairLabels(evidence)
}

func productionMTPHasGemma4AssistantProductionPackEvidence(evidence ProductionMTPPromotionEvidence) bool {
	size := rocmGemma4CanonicalSize(evidence.AssistantGemma4Size)
	if size == "" || size != rocmGemma4CanonicalSize(evidence.TargetGemma4Size) {
		return false
	}
	mode := modelgemma4.DenormalizedQuantModeForCollection(evidence.AssistantGemma4QuantMode)
	if mode == "" {
		mode = modelgemma4.AssistantQuantMode
	}
	support, ok := rocmGemma4MTPAssistantQuantModeSupport(size, mode)
	if !ok {
		return false
	}
	mode = support.Mode
	return evidence.AssistantProductionQuantModelID == rocmGemma4MTPAssistantPath(size, mode) &&
		evidence.AssistantProductionQuantPack == size+":assistant-"+mode &&
		evidence.AssistantProductionQuantTier == "mtp-assistant" &&
		evidence.AssistantProductionQuantMTPAssistant &&
		evidence.AssistantProductionQuantTargetFamily == "gemma4"
}

func requiredProductionMTPDraftTokenSweeps(policy ProductionMTPPolicy) []int {
	if len(policy.RequiredDraftTokenSweeps) == 0 {
		return append([]int(nil), defaultProductionMTPDraftTokenSweepsValue...)
	}
	return policy.RequiredDraftTokenSweeps
}

func productionMTPObservedDraftTokenSweepsCover(required, observed []int) bool {
	for _, want := range required {
		if want <= 0 {
			continue
		}
		found := false
		for _, got := range observed {
			if got == want {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func productionMTPModelInfoIdentity(info inference.ModelInfo) inference.ModelIdentity {
	return rocmGemma4ModelInfoIdentity(info, "")
}

func productionMTPApplyGemma4PairLabelEvidence(evidence *ProductionMTPPromotionEvidence, labels map[string]string) {
	if evidence == nil || labels == nil {
		return
	}
	evidence.TargetGemma4Size = firstNonEmptyString(
		labels["target_gemma4_size"],
		labels["attached_drafter_target_gemma4_size"],
		labels["attached.drafter.target.gemma4_size"],
		evidence.TargetGemma4Size,
	)
	evidence.TargetGemma4QuantMode = firstNonEmptyString(
		labels["target_gemma4_quant_mode"],
		labels["attached_drafter_target_gemma4_quant_mode"],
		labels["attached.drafter.target.gemma4_quant_mode"],
		evidence.TargetGemma4QuantMode,
	)
	evidence.TargetGemma4QuantGroup = productionMTPFirstNonZeroIntLabel(labels, []string{
		"target_gemma4_quant_group",
		"attached_drafter_target_gemma4_quant_group",
		"attached.drafter.target.gemma4_quant_group",
	}, evidence.TargetGemma4QuantGroup)
	evidence.TargetGemma4Runtime = firstNonEmptyString(
		labels["target_gemma4_runtime"],
		labels["attached_drafter_target_gemma4_runtime"],
		labels["attached.drafter.target.gemma4_runtime"],
		labels["engine_attached_drafter_target_runtime"],
		evidence.TargetGemma4Runtime,
	)
	evidence.TargetGemma4GenerateStatus = firstNonEmptyString(
		labels["target_gemma4_generate_status"],
		labels["attached_drafter_target_gemma4_generate_status"],
		labels["attached.drafter.target.gemma4_generate_status"],
		labels["engine_attached_drafter_target_generate_status"],
		evidence.TargetGemma4GenerateStatus,
	)
	evidence.TargetProductionQuantModelID = firstNonEmptyString(
		labels["target_production_quant_model"],
		labels["attached_drafter_target_production_quant_model"],
		labels["attached.drafter.target.production_quant_model"],
		evidence.TargetProductionQuantModelID,
	)
	evidence.TargetProductionQuantLockedModelID = firstNonEmptyString(
		labels["target_production_quant_locked_model"],
		labels["attached_drafter_target_production_quant_locked_model"],
		labels["attached.drafter.target.production_quant_locked_model"],
		evidence.TargetProductionQuantLockedModelID,
	)
	evidence.AssistantGemma4Size = firstNonEmptyString(
		labels["assistant_gemma4_size"],
		labels["draft_gemma4_size"],
		labels["attached_drafter_assistant_gemma4_size"],
		labels["attached_drafter_draft_gemma4_size"],
		labels["attached.drafter.assistant.gemma4_size"],
		labels["attached.drafter.draft.gemma4_size"],
		evidence.AssistantGemma4Size,
	)
	evidence.AssistantGemma4QuantMode = firstNonEmptyString(
		labels["assistant_gemma4_quant_mode"],
		labels["draft_gemma4_quant_mode"],
		labels["attached_drafter_assistant_gemma4_quant_mode"],
		labels["attached_drafter_draft_gemma4_quant_mode"],
		labels["attached.drafter.assistant.gemma4_quant_mode"],
		labels["attached.drafter.draft.gemma4_quant_mode"],
		evidence.AssistantGemma4QuantMode,
	)
	evidence.AssistantGemma4QuantGroup = productionMTPFirstNonZeroIntLabel(labels, []string{
		"assistant_gemma4_quant_group",
		"draft_gemma4_quant_group",
		"attached_drafter_assistant_gemma4_quant_group",
		"attached_drafter_draft_gemma4_quant_group",
		"attached.drafter.assistant.gemma4_quant_group",
		"attached.drafter.draft.gemma4_quant_group",
	}, evidence.AssistantGemma4QuantGroup)
	evidence.AssistantGemma4Runtime = firstNonEmptyString(
		labels["assistant_gemma4_runtime"],
		labels["draft_gemma4_runtime"],
		labels["attached_drafter_assistant_gemma4_runtime"],
		labels["attached_drafter_draft_gemma4_runtime"],
		labels["attached.drafter.assistant.gemma4_runtime"],
		labels["attached.drafter.draft.gemma4_runtime"],
		labels["engine_attached_drafter_assistant_runtime"],
		evidence.AssistantGemma4Runtime,
	)
	evidence.AssistantGemma4GenerateStatus = firstNonEmptyString(
		labels["assistant_gemma4_generate_status"],
		labels["draft_gemma4_generate_status"],
		labels["attached_drafter_assistant_gemma4_generate_status"],
		labels["attached_drafter_draft_gemma4_generate_status"],
		labels["attached.drafter.assistant.gemma4_generate_status"],
		labels["attached.drafter.draft.gemma4_generate_status"],
		labels["engine_attached_drafter_assistant_generate_status"],
		evidence.AssistantGemma4GenerateStatus,
	)
	evidence.AssistantProductionQuantModelID = firstNonEmptyString(
		labels["assistant_production_quant_model"],
		labels["assistant_production_quant_assistant_model"],
		labels["draft_production_quant_model"],
		labels["attached_drafter_assistant_production_quant_model"],
		labels["attached_drafter_assistant_production_quant_assistant_model"],
		labels["attached_drafter_draft_production_quant_model"],
		labels["attached.drafter.assistant.production_quant_model"],
		labels["attached.drafter.assistant.production_quant_assistant_model"],
		labels["attached.drafter.draft.production_quant_model"],
		evidence.AssistantProductionQuantModelID,
	)
	evidence.AssistantProductionQuantPack = firstNonEmptyString(
		labels["assistant_production_quant_pack"],
		labels["draft_production_quant_pack"],
		labels["attached_drafter_assistant_production_quant_pack"],
		labels["attached_drafter_draft_production_quant_pack"],
		labels["attached.drafter.assistant.production_quant_pack"],
		labels["attached.drafter.draft.production_quant_pack"],
		evidence.AssistantProductionQuantPack,
	)
	evidence.AssistantProductionQuantTier = firstNonEmptyString(
		labels["assistant_production_quant_tier"],
		labels["draft_production_quant_tier"],
		labels["attached_drafter_assistant_production_quant_tier"],
		labels["attached_drafter_draft_production_quant_tier"],
		labels["attached.drafter.assistant.production_quant_tier"],
		labels["attached.drafter.draft.production_quant_tier"],
		evidence.AssistantProductionQuantTier,
	)
	evidence.AssistantProductionQuantTargetFamily = firstNonEmptyString(
		labels["assistant_production_quant_target_family"],
		labels["draft_production_quant_target_family"],
		labels["attached_drafter_assistant_production_quant_target_family"],
		labels["attached_drafter_draft_production_quant_target_family"],
		labels["attached.drafter.assistant.production_quant_target_family"],
		labels["attached.drafter.draft.production_quant_target_family"],
		evidence.AssistantProductionQuantTargetFamily,
	)
	evidence.AssistantProductionQuantMTPAssistant = productionMTPFirstBoolLabel(labels, []string{
		"assistant_production_quant_mtp_assistant",
		"draft_production_quant_mtp_assistant",
		"attached_drafter_assistant_production_quant_mtp_assistant",
		"attached_drafter_draft_production_quant_mtp_assistant",
		"attached.drafter.assistant.production_quant_mtp_assistant",
		"attached.drafter.draft.production_quant_mtp_assistant",
	}, evidence.AssistantProductionQuantMTPAssistant)
}

func productionMTPFirstNonZeroIntLabel(labels map[string]string, keys []string, fallback int) int {
	_, value := productionFirstLabel(labels, keys)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || parsed <= 0 {
		return fallback
	}
	return parsed
}

func productionMTPFirstBoolLabel(labels map[string]string, keys []string, fallback bool) bool {
	_, value := productionFirstLabel(labels, keys)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func productionMTPApplyOfficialPairLockEvidence(evidence *ProductionMTPPromotionEvidence) {
	if evidence == nil {
		return
	}
	evidence.OfficialPairVerified = false
	evidence.OfficialTargetModelID = ""
	evidence.OfficialTargetRevision = ""
	evidence.OfficialAssistantModelID = ""
	evidence.OfficialAssistantRevision = ""
	if !productionMTPHasOfficialGemma4PairLabels(*evidence) {
		return
	}
	evidence.OfficialPairVerified = true
	evidence.OfficialTargetModelID = officialGemma4E2BTargetModelID
	evidence.OfficialTargetRevision = officialGemma4E2BTargetRevision
	evidence.OfficialAssistantModelID = officialGemma4E2BAssistantModelID
	evidence.OfficialAssistantRevision = officialGemma4E2BAssistantRevision
}

func productionMTPApplyGemma4FamilyPairEvidence(evidence *ProductionMTPPromotionEvidence) {
	if evidence == nil {
		return
	}
	evidence.Gemma4FamilyPairVerified = productionMTPHasGemma4FamilyPairLabels(*evidence)
}

func productionMTPHasGemma4FamilyPairLabels(evidence ProductionMTPPromotionEvidence) bool {
	return modelgemma4.FamilyPairEvidenceVerified(productionMTPGemma4PairEvidence(evidence))
}

func productionMTPHasOfficialGemma4PairLabels(evidence ProductionMTPPromotionEvidence) bool {
	return modelgemma4.OfficialPairEvidenceVerified(productionMTPGemma4PairEvidence(evidence))
}

func productionMTPGemma4PairEvidence(evidence ProductionMTPPromotionEvidence) modelgemma4.PairEvidence {
	return modelgemma4.PairEvidence{
		TargetSize:              evidence.TargetGemma4Size,
		TargetQuantMode:         evidence.TargetGemma4QuantMode,
		TargetQuantGroup:        evidence.TargetGemma4QuantGroup,
		TargetRuntime:           evidence.TargetGemma4Runtime,
		TargetGenerateStatus:    evidence.TargetGemma4GenerateStatus,
		AssistantSize:           evidence.AssistantGemma4Size,
		AssistantQuantMode:      evidence.AssistantGemma4QuantMode,
		AssistantQuantGroup:     evidence.AssistantGemma4QuantGroup,
		AssistantRuntime:        evidence.AssistantGemma4Runtime,
		AssistantGenerateStatus: evidence.AssistantGemma4GenerateStatus,
	}
}

func validateProductionMTPAttachedDrafterPlan(plan AttachedDrafterPlan) error {
	if plan.Mode != defaultProductionMTPPolicy.Mode {
		return core.E("rocm.ProductionMTPAttachedDrafterPlan", "mode must be mtp_attached_drafter", nil)
	}
	if !isROCmGemma4Architecture(plan.Target.Architecture) {
		return core.E("rocm.ProductionMTPAttachedDrafterPlan", "target model must be a Gemma4 text model", nil)
	}
	if !isROCmGemma4AssistantArchitecture(plan.Draft.Architecture) {
		return core.E("rocm.ProductionMTPAttachedDrafterPlan", "draft model must be a Gemma4 assistant attached MTP drafter", nil)
	}
	if plan.DraftTokens <= 0 {
		return core.E("rocm.ProductionMTPAttachedDrafterPlan", "draft tokens must be positive", nil)
	}
	if plan.HelperStatus != hipKernelStatusLinked {
		return core.E("rocm.ProductionMTPAttachedDrafterPlan", "attached drafter decode helper must be linked", nil)
	}
	if plan.NativeAttachment != hipKernelStatusNotLinked {
		return core.E("rocm.ProductionMTPAttachedDrafterPlan", "native HIP drafter attachment must remain explicitly not_linked", nil)
	}
	if err := checkROCmGemma4AttachedDrafterTargetIdentity("rocm.ProductionMTPAttachedDrafterPlan", productionMTPPlanTargetIdentity(plan)); err != nil {
		return err
	}
	if err := checkROCmGemma4AttachedDrafterAssistantIdentity("rocm.ProductionMTPAttachedDrafterPlan", productionMTPPlanDraftIdentity(plan)); err != nil {
		return err
	}
	if err := checkROCmGemma4AttachedDrafterFamilyPair("rocm.ProductionMTPAttachedDrafterPlan", productionMTPPlanTargetIdentity(plan), productionMTPPlanDraftIdentity(plan)); err != nil {
		return err
	}
	return nil
}
