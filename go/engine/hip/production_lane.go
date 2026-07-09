// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"strconv"
	"strings"

	core "dappco.re/go"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
)

const (
	ProductionLaneName                                = "gemma4-e2b-it-q6"
	ProductionLaneModelID                             = modelgemma4.ProductionLaneModelID
	ProductionLaneArchivedBaselineModelID             = modelgemma4.ProductionLaneArchivedBaselineModelID
	ProductionLaneCurrentQualityModelID               = modelgemma4.ProductionLaneCurrentQualityModelID
	ProductionLaneCurrentModelID                      = modelgemma4.ProductionLaneCurrentModelID
	ProductionLaneCurrentConstrainedModelID           = modelgemma4.ProductionLaneCurrentConstrainedModelID
	ProductionLaneArchitecture                        = "gemma4_text"
	ProductionLaneChatTemplate                        = "gemma4"
	ProductionLaneProductDefaultQuantBits             = modelgemma4.ProductionLaneProductDefaultQuantBits
	ProductionLaneQualityQuantBits                    = modelgemma4.ProductionLaneQualityQuantBits
	ProductionLaneConstrainedQuantBits                = modelgemma4.ProductionLaneConstrainedQuantBits
	ProductionLaneContextLength                       = 0
	ProductionLaneLongContextLength                   = modelgemma4.ProductionLaneLongContextLength
	ProductionLaneHyperLongContextLength              = 131072
	ProductionLaneLongFormMaxTokens                   = 8192
	ProductionLaneMaxTokens                           = 0
	ProductionLaneRuns                                = 3
	ProductionLaneRetainedKVCacheDType                = "fp16"
	ProductionLaneLongContextPrefillChunk             = 512
	ProductionLaneLongContextPromptBytes              = 4096
	ProductionLanePagedKVPageSize                     = 2048
	ProductionLaneBookTurnCount                       = 10
	ProductionLaneBookWallSeconds                     = 110
	productionLaneGemma4E2BLayers                     = 35
	productionLaneGemma4E2BLayersLabel                = "35"
	productionLaneGemma4E2BVocabSize                  = 262144
	productionLaneGemma4E2BVocabSizeLabel             = "262144"
	productionLaneGemma4E2BHiddenSize                 = 1536
	productionLaneGemma4E2BHiddenSizeLabel            = "1536"
	productionLaneBookTurnCountLabel                  = "10"
	productionLaneBookWallSecondsLabel                = "110"
	productionLaneRetainedVisibleTokensSecLabel       = "100"
	productionQuantizationLadderLabel                 = "bf16,q8,q6,q4"
	productionAutoRoundAlgorithmsLabel                = "auto-round,auto-round-best,auto-round-light"
	productionAutoRoundFormatsLabel                   = "native,gguf"
	productionAutoRoundSchemesLabel                   = "W4A16,W2A16,W8A16"
	productionAutoRoundFloatFormatsLabel              = "mxfp4,nvfp4,mxfp8,fp8,int2"
	productionAutoRoundGroupSizesLabel                = "32,64,128"
	productionAutoRoundProfilesLabel                  = "w4a16-mxfp4-g128,w4a16-nvfp4-g128,w8a16-fp8-g64,w8a16-mxfp8-g64,w2a16-int2-g128"
	productionAutoRoundCalibrationLabelsLabel         = "autoround_calibration_profile,autoround_calibration_format,autoround_calibration_weight_scheme,autoround_calibration_float_format,autoround_calibration_bits,autoround_calibration_group_size,autoround_calibration_nsamples,autoround_calibration_seqlen,autoround_calibration_iters,autoround_calibration_runtime,autoround_calibration_hip_kernel,autoround_calibration_requires_bench,autoround_calibration_required"
	productionAutoRoundCalibrationDecisionLabelsLabel = "autoround_calibration_candidate,autoround_calibration_decision_reason,autoround_calibration_decision_profile,autoround_calibration_decision_float_format,autoround_calibration_decision_hip_kernel,autoround_calibration_decision_requires_bench"
	productionQuantizationRequiredMetricsLabel        = "load_duration,peak_memory_bytes,retained_restore_duration,raw_decode_tokens_per_sec,active_weight_read_bytes_per_token,memory_bandwidth_bytes_per_sec,long_output_quality_flags,step_down_working_set_bytes,context_length"
	productionBookGateMetricsLabel                    = "production_book_gate_candidate,production_book_gate_reason_code,production_book_gate_q6,production_book_gate_turns,production_book_gate_wall,production_book_gate_decode,production_book_gate_quality,production_book_gate_raw_decode_tok/s,production_book_gate_wall_s,production_book_gate_quality_flags"
	productionBookGateReasonCodesLabel                = "0=pass,1=quant,2=metrics,3=turns,4=wall,5=decode,6=quality"
	productionBookRetainedRouteMetricsLabel           = "book_retained_state,book_retained_state_required,book_prompt_replay_fallback_forbidden,book_state_source_runtime_kv,book_replay_baseline"
	productionBookRetainedArtifactLabelsLabel         = "production_book_retained_artifact_candidate,production_book_retained_artifact_retained_route,production_book_retained_artifact_reason,production_book_retained_artifact_gate_candidate,production_book_retained_artifact_gate_reason_code,production_book_retained_artifact_gate_q6,production_book_retained_artifact_gate_turns,production_book_retained_artifact_gate_wall,production_book_retained_artifact_gate_decode,production_book_retained_artifact_gate_quality,production_book_retained_artifact_raw_decode_tok/s,production_book_retained_artifact_wall_s,production_book_retained_artifact_quality_flags"
	productionLaneActiveParameterEstimate             = modelgemma4.ProductionActiveParameterEstimate
	productionLaneRetainedVisibleTokensSec            = 100
)

var productionQuantizationRequiredMetrics = []string{
	"load_duration",
	"peak_memory_bytes",
	"retained_restore_duration",
	"raw_decode_tokens_per_sec",
	"active_weight_read_bytes_per_token",
	"memory_bandwidth_bytes_per_sec",
	"long_output_quality_flags",
	"step_down_working_set_bytes",
	"context_length",
}

var productionBookGateMetrics = []string{
	"production_book_gate_candidate",
	"production_book_gate_reason_code",
	"production_book_gate_q6",
	"production_book_gate_turns",
	"production_book_gate_wall",
	"production_book_gate_decode",
	"production_book_gate_quality",
	"production_book_gate_raw_decode_tok/s",
	"production_book_gate_wall_s",
	"production_book_gate_quality_flags",
}

var productionBookRetainedRouteMetrics = []string{
	"book_retained_state",
	"book_retained_state_required",
	"book_prompt_replay_fallback_forbidden",
	"book_state_source_runtime_kv",
	"book_replay_baseline",
}

var productionBookRetainedArtifactLabels = []string{
	"production_book_retained_artifact_candidate",
	"production_book_retained_artifact_retained_route",
	"production_book_retained_artifact_reason",
	"production_book_retained_artifact_gate_candidate",
	"production_book_retained_artifact_gate_reason_code",
	"production_book_retained_artifact_gate_q6",
	"production_book_retained_artifact_gate_turns",
	"production_book_retained_artifact_gate_wall",
	"production_book_retained_artifact_gate_decode",
	"production_book_retained_artifact_gate_quality",
	"production_book_retained_artifact_raw_decode_tok/s",
	"production_book_retained_artifact_wall_s",
	"production_book_retained_artifact_quality_flags",
}

var productionAutoRoundAlgorithms = []string{"auto-round", "auto-round-best", "auto-round-light"}
var productionAutoRoundFormats = []string{"native", "gguf"}
var productionAutoRoundSchemes = []string{"W4A16", "W2A16", "W8A16"}
var productionAutoRoundFloatFormats = []string{"mxfp4", "nvfp4", "mxfp8", "fp8", "int2"}
var productionAutoRoundGroupSizes = []int{32, 64, 128}
var productionAutoRoundCalibrationLabels = []string{
	"autoround_calibration_profile",
	"autoround_calibration_format",
	"autoround_calibration_weight_scheme",
	"autoround_calibration_float_format",
	"autoround_calibration_bits",
	"autoround_calibration_group_size",
	"autoround_calibration_nsamples",
	"autoround_calibration_seqlen",
	"autoround_calibration_iters",
	"autoround_calibration_runtime",
	"autoround_calibration_hip_kernel",
	"autoround_calibration_requires_bench",
	"autoround_calibration_required",
}
var productionAutoRoundCalibrationDecisionLabels = []string{
	"autoround_calibration_candidate",
	"autoround_calibration_decision_reason",
	"autoround_calibration_decision_profile",
	"autoround_calibration_decision_float_format",
	"autoround_calibration_decision_hip_kernel",
	"autoround_calibration_decision_requires_bench",
}
var productionAutoRoundProfiles = []ProductionAutoRoundQuantizationProfile{
	{
		Name:                "w4a16-mxfp4-g128",
		Algorithm:           "auto-round-light",
		Format:              "native",
		WeightScheme:        "W4A16",
		FloatFormat:         "mxfp4",
		Bits:                4,
		GroupSize:           128,
		NSamples:            512,
		SeqLen:              2048,
		Iters:               200,
		ProductRole:         "rocm-fp4-planning",
		Runtime:             "planned_hip",
		HIPKernel:           hipKernelStatusNotLinked,
		RequiresCalibration: true,
		RequiresBench:       true,
	},
	{
		Name:                "w4a16-nvfp4-g128",
		Algorithm:           "auto-round-light",
		Format:              "native",
		WeightScheme:        "W4A16",
		FloatFormat:         "nvfp4",
		Bits:                4,
		GroupSize:           128,
		NSamples:            512,
		SeqLen:              2048,
		Iters:               200,
		ProductRole:         "rocm-fp4-planning",
		Runtime:             "planned_hip",
		HIPKernel:           hipKernelStatusNotLinked,
		RequiresCalibration: true,
		RequiresBench:       true,
	},
	{
		Name:                "w8a16-fp8-g64",
		Algorithm:           "auto-round",
		Format:              "native",
		WeightScheme:        "W8A16",
		FloatFormat:         "fp8",
		Bits:                8,
		GroupSize:           64,
		NSamples:            512,
		SeqLen:              2048,
		Iters:               200,
		ProductRole:         "rocm-fp8-planning",
		Runtime:             "planned_hip",
		HIPKernel:           hipKernelStatusNotLinked,
		RequiresCalibration: true,
		RequiresBench:       true,
	},
	{
		Name:                "w8a16-mxfp8-g64",
		Algorithm:           "auto-round",
		Format:              "native",
		WeightScheme:        "W8A16",
		FloatFormat:         "mxfp8",
		Bits:                8,
		GroupSize:           64,
		NSamples:            512,
		SeqLen:              2048,
		Iters:               200,
		ProductRole:         "rocm-fp8-planning",
		Runtime:             "planned_hip",
		HIPKernel:           hipKernelStatusNotLinked,
		RequiresCalibration: true,
		RequiresBench:       true,
	},
	{
		Name:                "w2a16-int2-g128",
		Algorithm:           "auto-round",
		Format:              "native",
		WeightScheme:        "W2A16",
		FloatFormat:         "int2",
		Bits:                2,
		GroupSize:           128,
		NSamples:            512,
		SeqLen:              2048,
		Iters:               200,
		ProductRole:         "rocm-int2-planning",
		Runtime:             "planned_hip",
		HIPKernel:           hipKernelStatusNotLinked,
		RequiresCalibration: true,
		RequiresBench:       true,
	},
}

const (
	productionAutoRoundProfileMXFP4Alias       = "w4a16-mxfp4"
	productionAutoRoundProfileMXFP4GroupAlias  = "w4a16-mxfp4-g128"
	productionAutoRoundProfileNVFP4Alias       = "w4a16-nvfp4"
	productionAutoRoundProfileNVFP4GroupAlias  = "w4a16-nvfp4-g128"
	productionAutoRoundProfileFP8Alias         = "w8a16-fp8"
	productionAutoRoundProfileFP8GroupAlias    = "w8a16-fp8-g64"
	productionAutoRoundProfileMXFP8Alias       = "w8a16-mxfp8"
	productionAutoRoundProfileMXFP8GroupAlias  = "w8a16-mxfp8-g64"
	productionAutoRoundProfileINT2Alias        = "w2a16-int2"
	productionAutoRoundProfileINT2GroupAlias   = "w2a16-int2-g128"
	productionAutoRoundProfileW2A16Alias       = "w2a16"
	productionAutoRoundProfileMXFP4FormatAlias = "mxfp4"
	productionAutoRoundProfileNVFP4FormatAlias = "nvfp4"
	productionAutoRoundProfileFP8FormatAlias   = "fp8"
	productionAutoRoundProfileMXFP8FormatAlias = "mxfp8"
	productionAutoRoundProfileINT2FormatAlias  = "int2"
	productionAutoRoundProfileQ2FormatAlias    = "q2"
)

type ProductionLane struct {
	Name             string
	ModelID          string
	Architecture     string
	ChatTemplate     string
	QuantBits        int
	ContextLength    int
	MaxTokens        int
	Runs             int
	TraceTokenPhases bool
	IncludeOutput    bool
}

type ProductionQuantizationPolicy struct {
	TargetModelID              string
	ArchivedBaseline           string
	DefaultBits                int
	QualityBits                int
	ConstrainedBits            int
	ActiveParameterEstimate    int
	MinimumVisibleTokensPerSec int
	RequiredBenchmarkMetrics   []string
	Tiers                      []ProductionQuantizationTier
	SupportedPacks             []ProductionQuantizationPackSupport
}

type ProductionAutoRoundQuantizationSupport struct {
	Algorithms       []string
	Formats          []string
	WeightSchemes    []string
	FloatFormats     []string
	GroupSizes       []int
	Profiles         []ProductionAutoRoundQuantizationProfile
	CalibrationKnobs []string
	Runtime          string
	HIPKernel        string
}

type ProductionAutoRoundQuantizationProfile struct {
	Name                string
	Algorithm           string
	Format              string
	WeightScheme        string
	FloatFormat         string
	Bits                int
	GroupSize           int
	NSamples            int
	SeqLen              int
	Iters               int
	ProductRole         string
	Runtime             string
	HIPKernel           string
	RequiresCalibration bool
	RequiresBench       bool
}

type ProductionAutoRoundCalibrationPlan struct {
	ProfileName         string
	Algorithm           string
	Format              string
	WeightScheme        string
	FloatFormat         string
	Bits                int
	GroupSize           int
	NSamples            int
	SeqLen              int
	Iters               int
	Runtime             string
	HIPKernel           string
	RequiresCalibration bool
	RequiresBench       bool
	BitsLabel           string
	GroupSizeLabel      string
	NSamplesLabel       string
	SeqLenLabel         string
	ItersLabel          string
	RequiresBenchLabel  string
	CalibrationLabel    string
}

type ProductionAutoRoundCalibrationEvidence struct {
	ProfileName         string
	Format              string
	WeightScheme        string
	FloatFormat         string
	Bits                int
	GroupSize           int
	NSamples            int
	SeqLen              int
	Iters               int
	Runtime             string
	HIPKernel           string
	RequiresCalibration bool
	RequiresBench       bool
}

type ProductionAutoRoundCalibrationDecision struct {
	CalibrationCandidate bool
	RequiresBench        bool
	Reason               string
	ProfileName          string
	FloatFormat          string
	HIPKernel            string
}

type ProductionBookGatePolicy struct {
	QuantBits                 int
	MinimumTurns              int
	MaximumWallSeconds        int
	MinimumRawDecodeTokensSec float64
	MaximumQualityFlags       int
	RequiredMetrics           []string
	ReasonCodes               string
}

func DefaultProductionAutoRoundQuantizationSupport() ProductionAutoRoundQuantizationSupport {
	return ProductionAutoRoundQuantizationSupport{
		Algorithms:       append([]string(nil), productionAutoRoundAlgorithms...),
		Formats:          append([]string(nil), productionAutoRoundFormats...),
		WeightSchemes:    append([]string(nil), productionAutoRoundSchemes...),
		FloatFormats:     append([]string(nil), productionAutoRoundFloatFormats...),
		GroupSizes:       append([]int(nil), productionAutoRoundGroupSizes...),
		Profiles:         DefaultProductionAutoRoundQuantizationProfiles(),
		CalibrationKnobs: []string{"nsamples", "seqlen", "iters"},
		Runtime:          "planned_hip",
		HIPKernel:        hipKernelStatusNotLinked,
	}
}

func DefaultProductionAutoRoundQuantizationProfiles() []ProductionAutoRoundQuantizationProfile {
	return append([]ProductionAutoRoundQuantizationProfile(nil), productionAutoRoundProfiles...)
}

func DefaultProductionAutoRoundCalibrationPlan(profile ProductionAutoRoundQuantizationProfile) ProductionAutoRoundCalibrationPlan {
	return productionAutoRoundCalibrationPlan(profile, 0, 0, 0)
}

func productionAutoRoundCalibrationPlan(profile ProductionAutoRoundQuantizationProfile, nsamplesOverride, seqLenOverride, itersOverride int) ProductionAutoRoundCalibrationPlan {
	nsamples := profile.NSamples
	if nsamplesOverride > 0 {
		nsamples = nsamplesOverride
	}
	seqLen := profile.SeqLen
	if seqLenOverride > 0 {
		seqLen = seqLenOverride
	}
	iters := profile.Iters
	if itersOverride > 0 {
		iters = itersOverride
	}
	plan := ProductionAutoRoundCalibrationPlan{
		ProfileName:         profile.Name,
		Algorithm:           profile.Algorithm,
		Format:              profile.Format,
		WeightScheme:        profile.WeightScheme,
		FloatFormat:         profile.FloatFormat,
		Bits:                profile.Bits,
		GroupSize:           profile.GroupSize,
		NSamples:            nsamples,
		SeqLen:              seqLen,
		Iters:               iters,
		Runtime:             profile.Runtime,
		HIPKernel:           profile.HIPKernel,
		RequiresCalibration: profile.RequiresCalibration,
		RequiresBench:       profile.RequiresBench,
	}
	productionAutoRoundRefreshCalibrationPlanLabels(&plan)
	return plan
}

func DefaultProductionAutoRoundCalibrationLabels() []string {
	return append([]string(nil), productionAutoRoundCalibrationLabels...)
}

func DefaultProductionAutoRoundCalibrationDecisionLabels() []string {
	return append([]string(nil), productionAutoRoundCalibrationDecisionLabels...)
}

func ApplyProductionAutoRoundCalibrationPlanLabels(labels map[string]string, plan ProductionAutoRoundCalibrationPlan) {
	if labels == nil || plan.ProfileName == "" {
		return
	}
	if plan.BitsLabel == "" || plan.GroupSizeLabel == "" || plan.NSamplesLabel == "" || plan.SeqLenLabel == "" || plan.ItersLabel == "" || plan.RequiresBenchLabel == "" || plan.CalibrationLabel == "" {
		productionAutoRoundRefreshCalibrationPlanLabels(&plan)
	}
	labels["autoround_calibration_profile"] = plan.ProfileName
	labels["autoround_calibration_format"] = plan.Format
	labels["autoround_calibration_weight_scheme"] = plan.WeightScheme
	labels["autoround_calibration_float_format"] = plan.FloatFormat
	labels["autoround_calibration_bits"] = plan.BitsLabel
	labels["autoround_calibration_group_size"] = plan.GroupSizeLabel
	labels["autoround_calibration_nsamples"] = plan.NSamplesLabel
	labels["autoround_calibration_seqlen"] = plan.SeqLenLabel
	labels["autoround_calibration_iters"] = plan.ItersLabel
	labels["autoround_calibration_runtime"] = plan.Runtime
	labels["autoround_calibration_hip_kernel"] = plan.HIPKernel
	labels["autoround_calibration_requires_bench"] = plan.RequiresBenchLabel
	labels["autoround_calibration_required"] = plan.CalibrationLabel
}

func ApplyProductionAutoRoundCalibrationLabelEvidence(evidence *ProductionAutoRoundCalibrationEvidence, labels map[string]string) error {
	if evidence == nil {
		return core.E("rocm.ApplyProductionAutoRoundCalibrationLabelEvidence", "evidence is required", nil)
	}
	if labels == nil {
		return core.E("rocm.ApplyProductionAutoRoundCalibrationLabelEvidence", "labels are required", nil)
	}
	evidence.ProfileName = labels["autoround_calibration_profile"]
	evidence.Format = labels["autoround_calibration_format"]
	evidence.WeightScheme = labels["autoround_calibration_weight_scheme"]
	evidence.FloatFormat = labels["autoround_calibration_float_format"]
	evidence.Runtime = labels["autoround_calibration_runtime"]
	evidence.HIPKernel = labels["autoround_calibration_hip_kernel"]
	if err := productionAutoRoundApplyIntLabel(labels, "autoround_calibration_bits", &evidence.Bits); err != nil {
		return err
	}
	if err := productionAutoRoundApplyIntLabel(labels, "autoround_calibration_group_size", &evidence.GroupSize); err != nil {
		return err
	}
	if err := productionAutoRoundApplyIntLabel(labels, "autoround_calibration_nsamples", &evidence.NSamples); err != nil {
		return err
	}
	if err := productionAutoRoundApplyIntLabel(labels, "autoround_calibration_seqlen", &evidence.SeqLen); err != nil {
		return err
	}
	if err := productionAutoRoundApplyIntLabel(labels, "autoround_calibration_iters", &evidence.Iters); err != nil {
		return err
	}
	if err := productionAutoRoundApplyBoolLabel(labels, "autoround_calibration_required", &evidence.RequiresCalibration); err != nil {
		return err
	}
	if err := productionAutoRoundApplyBoolLabel(labels, "autoround_calibration_requires_bench", &evidence.RequiresBench); err != nil {
		return err
	}
	return nil
}

func EvaluateProductionAutoRoundCalibrationEvidence(evidence ProductionAutoRoundCalibrationEvidence) ProductionAutoRoundCalibrationDecision {
	decision := ProductionAutoRoundCalibrationDecision{
		RequiresBench: evidence.RequiresBench,
		ProfileName:   evidence.ProfileName,
		FloatFormat:   evidence.FloatFormat,
		HIPKernel:     evidence.HIPKernel,
	}
	switch {
	case evidence.ProfileName == "":
		decision.Reason = "missing AutoRound calibration profile"
	case evidence.FloatFormat == "":
		decision.Reason = "missing AutoRound FP format"
	case evidence.Bits <= 0 || evidence.GroupSize <= 0:
		decision.Reason = "missing AutoRound calibration shape"
	case evidence.NSamples <= 0 || evidence.SeqLen <= 0 || evidence.Iters <= 0:
		decision.Reason = "missing AutoRound calibration knobs"
	case !evidence.RequiresCalibration:
		decision.Reason = "AutoRound calibration not required"
	case evidence.Runtime != "planned_hip":
		decision.Reason = "AutoRound calibration runtime is not planned HIP"
	case evidence.HIPKernel != hipKernelStatusNotLinked:
		decision.Reason = "AutoRound calibration HIP kernel status is not not_linked"
	default:
		decision.CalibrationCandidate = true
		decision.Reason = "AutoRound calibration target ready for ROCm bench planning"
	}
	return decision
}

func ApplyProductionAutoRoundCalibrationEvidenceDecisionLabels(out map[string]string, evidenceLabels map[string]string) (ProductionAutoRoundCalibrationDecision, error) {
	var evidence ProductionAutoRoundCalibrationEvidence
	if err := ApplyProductionAutoRoundCalibrationLabelEvidence(&evidence, evidenceLabels); err != nil {
		return ProductionAutoRoundCalibrationDecision{}, err
	}
	decision := EvaluateProductionAutoRoundCalibrationEvidence(evidence)
	ApplyProductionAutoRoundCalibrationDecisionLabels(out, decision)
	return decision, nil
}

func ApplyProductionAutoRoundCalibrationDecisionLabels(labels map[string]string, decision ProductionAutoRoundCalibrationDecision) {
	if labels == nil {
		return
	}
	labels["autoround_calibration_candidate"] = boolLabel(decision.CalibrationCandidate)
	labels["autoround_calibration_decision_reason"] = decision.Reason
	labels["autoround_calibration_decision_profile"] = decision.ProfileName
	labels["autoround_calibration_decision_float_format"] = decision.FloatFormat
	labels["autoround_calibration_decision_hip_kernel"] = decision.HIPKernel
	labels["autoround_calibration_decision_requires_bench"] = boolLabel(decision.RequiresBench)
}

func ApplyProductionAutoRoundCalibrationDecisionLabelEvidence(decision *ProductionAutoRoundCalibrationDecision, labels map[string]string) error {
	if decision == nil {
		return core.E("rocm.ApplyProductionAutoRoundCalibrationDecisionLabelEvidence", "decision is required", nil)
	}
	if labels == nil {
		return core.E("rocm.ApplyProductionAutoRoundCalibrationDecisionLabelEvidence", "labels are required", nil)
	}
	decision.Reason = labels["autoround_calibration_decision_reason"]
	decision.ProfileName = labels["autoround_calibration_decision_profile"]
	decision.FloatFormat = labels["autoround_calibration_decision_float_format"]
	decision.HIPKernel = labels["autoround_calibration_decision_hip_kernel"]
	if err := productionAutoRoundApplyBoolLabel(labels, "autoround_calibration_candidate", &decision.CalibrationCandidate); err != nil {
		return err
	}
	if err := productionAutoRoundApplyBoolLabel(labels, "autoround_calibration_decision_requires_bench", &decision.RequiresBench); err != nil {
		return err
	}
	return nil
}

func EvaluateProductionAutoRoundCalibrationDecisionLabels(labels map[string]string) (ProductionAutoRoundCalibrationDecision, error) {
	var decision ProductionAutoRoundCalibrationDecision
	if err := ApplyProductionAutoRoundCalibrationDecisionLabelEvidence(&decision, labels); err != nil {
		return ProductionAutoRoundCalibrationDecision{}, err
	}
	return decision, nil
}

func productionAutoRoundRefreshCalibrationPlanLabels(plan *ProductionAutoRoundCalibrationPlan) {
	if plan == nil {
		return
	}
	plan.BitsLabel = strconv.Itoa(plan.Bits)
	plan.GroupSizeLabel = strconv.Itoa(plan.GroupSize)
	plan.NSamplesLabel = strconv.Itoa(plan.NSamples)
	plan.SeqLenLabel = strconv.Itoa(plan.SeqLen)
	plan.ItersLabel = strconv.Itoa(plan.Iters)
	plan.RequiresBenchLabel = boolLabel(plan.RequiresBench)
	plan.CalibrationLabel = boolLabel(plan.RequiresCalibration)
}

func productionAutoRoundApplyIntLabel(labels map[string]string, key string, out *int) error {
	value := labels[key]
	if value == "" {
		return nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return core.E("rocm.ApplyProductionAutoRoundCalibrationLabelEvidence", "parse "+key, err)
	}
	*out = parsed
	return nil
}

func productionAutoRoundApplyBoolLabel(labels map[string]string, key string, out *bool) error {
	value := labels[key]
	if value == "" {
		return nil
	}
	switch value {
	case "true":
		*out = true
		return nil
	case "false":
		*out = false
		return nil
	}
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "true", "1", "yes":
		*out = true
	case "false", "0", "no":
		*out = false
	default:
		return core.E("rocm.ApplyProductionAutoRoundCalibrationLabelEvidence", "parse "+key, nil)
	}
	return nil
}

func ProductionAutoRoundQuantizationProfileByName(name string) (ProductionAutoRoundQuantizationProfile, bool) {
	needle := normalizeProductionAutoRoundProfileName(name)
	if needle == "" {
		return ProductionAutoRoundQuantizationProfile{}, false
	}
	switch needle {
	case productionAutoRoundProfileMXFP4GroupAlias, productionAutoRoundProfileMXFP4Alias, productionAutoRoundProfileMXFP4FormatAlias:
		return productionAutoRoundProfiles[0], true
	case productionAutoRoundProfileNVFP4GroupAlias, productionAutoRoundProfileNVFP4Alias, productionAutoRoundProfileNVFP4FormatAlias:
		return productionAutoRoundProfiles[1], true
	case productionAutoRoundProfileFP8GroupAlias, productionAutoRoundProfileFP8Alias, productionAutoRoundProfileFP8FormatAlias:
		return productionAutoRoundProfiles[2], true
	case productionAutoRoundProfileMXFP8GroupAlias, productionAutoRoundProfileMXFP8Alias, productionAutoRoundProfileMXFP8FormatAlias:
		return productionAutoRoundProfiles[3], true
	case productionAutoRoundProfileINT2GroupAlias, productionAutoRoundProfileINT2Alias, productionAutoRoundProfileW2A16Alias, productionAutoRoundProfileINT2FormatAlias, productionAutoRoundProfileQ2FormatAlias:
		return productionAutoRoundProfiles[4], true
	default:
		return ProductionAutoRoundQuantizationProfile{}, false
	}
}

func productionAutoRoundQuantizationProfileForFields(scheme, floatFormat string, groupSize int) (ProductionAutoRoundQuantizationProfile, bool) {
	scheme = strings.ToUpper(strings.TrimSpace(scheme))
	floatFormat = normalizeROCmQuantizationAlias(floatFormat)
	if floatFormat == "native" || floatFormat == "gguf" {
		floatFormat = ""
	} else if floatFormat == "q2" || floatFormat == "w2a16" {
		floatFormat = "int2"
	}
	if scheme == "" || floatFormat == "" {
		return ProductionAutoRoundQuantizationProfile{}, false
	}
	switch {
	case scheme == "W4A16" && floatFormat == "mxfp4" && (groupSize == 0 || groupSize == 128):
		return productionAutoRoundProfiles[0], true
	case scheme == "W4A16" && floatFormat == "nvfp4" && (groupSize == 0 || groupSize == 128):
		return productionAutoRoundProfiles[1], true
	case scheme == "W8A16" && floatFormat == "fp8" && (groupSize == 0 || groupSize == 64):
		return productionAutoRoundProfiles[2], true
	case scheme == "W8A16" && floatFormat == "mxfp8" && (groupSize == 0 || groupSize == 64):
		return productionAutoRoundProfiles[3], true
	case scheme == "W2A16" && floatFormat == "int2" && (groupSize == 0 || groupSize == 128):
		return productionAutoRoundProfiles[4], true
	default:
		return ProductionAutoRoundQuantizationProfile{}, false
	}
}

func normalizeProductionAutoRoundProfileName(name string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(name)), "_", "-")
}

func DefaultProductionLane() ProductionLane {
	return ProductionLane{
		Name:             ProductionLaneName,
		ModelID:          ProductionLaneCurrentModelID,
		Architecture:     ProductionLaneArchitecture,
		ChatTemplate:     ProductionLaneChatTemplate,
		QuantBits:        ProductionLaneProductDefaultQuantBits,
		ContextLength:    ProductionLaneContextLength,
		MaxTokens:        ProductionLaneMaxTokens,
		Runs:             ProductionLaneRuns,
		TraceTokenPhases: true,
	}
}

func DefaultProductionBookGatePolicy() ProductionBookGatePolicy {
	policy := defaultProductionBookGatePolicy()
	policy.RequiredMetrics = append([]string(nil), policy.RequiredMetrics...)
	return policy
}

func defaultProductionBookGatePolicy() ProductionBookGatePolicy {
	return ProductionBookGatePolicy{
		QuantBits:                 ProductionLaneProductDefaultQuantBits,
		MinimumTurns:              ProductionLaneBookTurnCount,
		MaximumWallSeconds:        ProductionLaneBookWallSeconds,
		MinimumRawDecodeTokensSec: float64(productionLaneRetainedVisibleTokensSec),
		MaximumQualityFlags:       0,
		RequiredMetrics:           productionBookGateMetrics,
		ReasonCodes:               productionBookGateReasonCodesLabel,
	}
}

func DefaultProductionQuantizationPolicy() ProductionQuantizationPolicy {
	return ProductionQuantizationPolicy{
		TargetModelID:              ProductionLaneCurrentModelID,
		ArchivedBaseline:           ProductionLaneCurrentConstrainedModelID,
		DefaultBits:                ProductionLaneProductDefaultQuantBits,
		QualityBits:                ProductionLaneQualityQuantBits,
		ConstrainedBits:            ProductionLaneConstrainedQuantBits,
		ActiveParameterEstimate:    productionLaneActiveParameterEstimate,
		MinimumVisibleTokensPerSec: productionLaneRetainedVisibleTokensSec,
		RequiredBenchmarkMetrics:   append([]string(nil), productionQuantizationRequiredMetrics...),
		Tiers:                      append([]ProductionQuantizationTier(nil), productionQuantizationTiers...),
		SupportedPacks:             DefaultProductionQuantizationPackSupport(),
	}
}
