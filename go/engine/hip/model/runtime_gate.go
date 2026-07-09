// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"strconv"
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/profile"
)

const RuntimeGatePlanContract = "rocm-runtime-gate-plan-v1"

// RuntimeGateID names a typed runtime fast-path gate. These IDs intentionally
// mirror go-mlx's Gate enum while staying metadata-only in the model package.
type RuntimeGateID string

const (
	GateDirectGreedyToken           RuntimeGateID = "direct_greedy_token"
	GateNativeMLPMatVec             RuntimeGateID = "native_mlp_matvec"
	GateNativeLinearMatVec          RuntimeGateID = "native_linear_matvec"
	GateNativeQ6BitstreamMatVec     RuntimeGateID = "native_q6_bitstream_matvec"
	GateNativeAttentionOMatVec      RuntimeGateID = "native_attention_o_matvec"
	GateGenerationStream            RuntimeGateID = "generation_stream"
	GateAsyncDecodePrefetch         RuntimeGateID = "async_decode_prefetch"
	GateFixedSlidingCache           RuntimeGateID = "fixed_sliding_cache"
	GateFixedSlidingCacheBound      RuntimeGateID = "fixed_sliding_cache_bound"
	GateFixedSharedMask             RuntimeGateID = "fixed_shared_mask"
	GateNativeFixedSlidingAttention RuntimeGateID = "native_fixed_sliding_attention"
	GatePagedDecodeFastConcat       RuntimeGateID = "paged_decode_fast_concat"
	GateNativePagedAttention        RuntimeGateID = "native_paged_attention"
	GateCacheOnlyChunkPrefill       RuntimeGateID = "cache_only_chunk_prefill"
	GateSortedExpertPrefill         RuntimeGateID = "sorted_expert_prefill"
	GateGatherQMMReferenceTests     RuntimeGateID = "gather_qmm_reference_tests"
	GateCompiledMLPDecode           RuntimeGateID = "compiled_mlp_decode"
	GateCompiledLayerDecode         RuntimeGateID = "compiled_layer_decode"
	GatePipelinedDecode             RuntimeGateID = "pipelined_decode"
	GateFixedWideSDPAAttention      RuntimeGateID = "fixed_wide_sdpa_attention"
)

type RuntimeGate struct {
	ID      RuntimeGateID `json:"id,omitempty"`
	Enabled bool          `json:"enabled,omitempty"`
	Source  string        `json:"source,omitempty"`
}

type RuntimeGatePlan struct {
	Contract      string                         `json:"contract,omitempty"`
	Architecture  string                         `json:"architecture,omitempty"`
	Family        string                         `json:"family,omitempty"`
	RuntimeStatus inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	Gates         []RuntimeGate                  `json:"gates,omitempty"`
	GateIDs       []RuntimeGateID                `json:"gate_ids,omitempty"`
	Labels        map[string]string              `json:"labels,omitempty"`
}

func (plan RuntimeGatePlan) Matched() bool {
	return plan.Contract != "" && plan.Architecture != "" && len(plan.GateIDs) > 0
}

func (plan RuntimeGatePlan) Clone() RuntimeGatePlan {
	plan.Gates = append([]RuntimeGate(nil), plan.Gates...)
	plan.GateIDs = append([]RuntimeGateID(nil), plan.GateIDs...)
	plan.Labels = cloneStringMap(plan.Labels)
	return plan
}

func (plan RuntimeGatePlan) GateEnabled(id RuntimeGateID) bool {
	for _, gate := range plan.Gates {
		if gate.ID == id {
			return gate.Enabled
		}
	}
	return false
}

func RuntimeGatePlanForIdentity(path string, identity inference.ModelIdentity) (RuntimeGatePlan, bool) {
	if identity.Path == "" {
		identity.Path = path
	}
	featureRoute, _ := FeatureRouteForIdentity(path, identity)
	runtimeRoute, _ := RuntimeContractRouteForIdentity(path, identity)
	plan := RuntimeGatePlanForRoutes(firstNonEmpty(
		featureRoute.Architecture,
		runtimeRoute.Architecture,
		identity.Labels["engine_architecture_resolved"],
		identity.Labels["architecture_resolved"],
		identity.Architecture,
	), firstNonEmpty(featureRoute.Family, runtimeRoute.Family), featureRoute, runtimeRoute, identity.Labels)
	if !plan.Matched() {
		return RuntimeGatePlan{}, false
	}
	return plan, true
}

func RuntimeGatePlanForRouteSet(set RouteSet) RuntimeGatePlan {
	return RuntimeGatePlanForRoutes(set.Architecture, set.Family, set.FeatureRoute, set.RuntimeContractRoute, set.Model.Labels)
}

func RuntimeGatePlanForRoutes(architecture, family string, featureRoute FeatureRoute, runtimeRoute RuntimeContractRoute, labels map[string]string) RuntimeGatePlan {
	architecture = profile.ArchitectureID(firstNonEmpty(architecture, featureRoute.Architecture, runtimeRoute.Architecture))
	if architecture == "" {
		return RuntimeGatePlan{}
	}
	if family == "" {
		family = firstNonEmpty(featureRoute.Family, runtimeRoute.Family, architecture)
	}
	runtimeStatus := featureRoute.RuntimeStatus
	if runtimeStatus == "" {
		runtimeStatus = runtimeRoute.RuntimeStatus
	}
	builder := runtimeGatePlanBuilder{
		plan: RuntimeGatePlan{
			Contract:      RuntimeGatePlanContract,
			Architecture:  architecture,
			Family:        family,
			RuntimeStatus: runtimeStatus,
		},
		seen: map[RuntimeGateID]bool{},
	}
	builder.add(GateGenerationStream, featureRoute.TextGenerate || runtimeGateLabelBool(labels, "engine_feature_generation_stream"), "feature_route")
	builder.add(GateDirectGreedyToken, runtimeRoute.GreedyToken || runtimeGateLabelBool(labels, "engine_feature_direct_greedy_token"), "runtime_contract")
	builder.add(GateNativeMLPMatVec, runtimeGateLabelBool(labels, "engine_feature_native_mlp_matvec"), "engine_feature_label")
	builder.add(GateNativeLinearMatVec, runtimeGateLabelBool(labels, "engine_feature_native_linear_matvec"), "engine_feature_label")
	builder.add(GateNativeQ6BitstreamMatVec, runtimeGateLabelBool(labels, "engine_feature_native_q6_bitstream_matvec"), "engine_feature_label")
	builder.add(GateNativeAttentionOMatVec, runtimeGateLabelBool(labels, "engine_feature_native_attention_o_matvec"), "engine_feature_label")
	builder.add(GateAsyncDecodePrefetch, runtimeGateLabelBool(labels, "engine_feature_async_decode_prefetch"), "engine_feature_label")
	builder.add(GateFixedSlidingCache, runtimeRoute.FixedSlidingCache ||
		runtimeGateAnyLabelBool(labels, "engine_feature_fixed_sliding_cache", "gemma4_fixed_sliding_cache"), "runtime_contract")
	builder.add(GateFixedSlidingCacheBound, runtimeGateAnyLabelBool(labels, "engine_feature_fixed_sliding_cache_bound", "gemma4_fixed_sliding_cache_bound"), "engine_feature_label")
	builder.add(GateFixedSharedMask, runtimeGateAnyLabelBool(labels, "engine_feature_fixed_shared_mask", "attention_mask_fixed_single_token"), "engine_feature_label")
	builder.add(GateNativeFixedSlidingAttention, runtimeGateLabelBool(labels, "engine_feature_native_fixed_sliding_attention"), "engine_feature_label")
	builder.add(GatePagedDecodeFastConcat, runtimeGateLabelBool(labels, "engine_feature_paged_decode_fast_concat"), "engine_feature_label")
	builder.add(GateNativePagedAttention, runtimeGateLabelBool(labels, "engine_feature_native_paged_attention"), "engine_feature_label")
	builder.add(GateCacheOnlyChunkPrefill, runtimeGateLabelBool(labels, "engine_feature_cache_only_chunk_prefill"), "engine_feature_label")
	builder.add(GateSortedExpertPrefill, featureRoute.MoE || runtimeGateLabelBool(labels, "engine_feature_sorted_expert_prefill"), "feature_route")
	builder.add(GateGatherQMMReferenceTests, runtimeGateLabelBool(labels, "engine_feature_gather_qmm_reference_tests"), "engine_feature_label")
	builder.add(GateCompiledMLPDecode, runtimeGateLabelBool(labels, "engine_feature_compiled_mlp_decode"), "engine_feature_label")
	builder.add(GateCompiledLayerDecode, runtimeGateLabelBool(labels, "engine_feature_compiled_layer_decode"), "engine_feature_label")
	builder.add(GatePipelinedDecode, runtimeGateLabelBool(labels, "engine_feature_pipelined_decode"), "engine_feature_label")
	builder.add(GateFixedWideSDPAAttention, runtimeGateLabelBool(labels, "engine_feature_fixed_wide_sdpa_attention"), "engine_feature_label")
	plan := builder.plan
	plan.Labels = runtimeGatePlanLabels(plan)
	if !plan.Matched() {
		return RuntimeGatePlan{}
	}
	return plan.Clone()
}

type runtimeGatePlanBuilder struct {
	plan RuntimeGatePlan
	seen map[RuntimeGateID]bool
}

func (builder *runtimeGatePlanBuilder) add(id RuntimeGateID, enabled bool, source string) {
	if id == "" || !enabled || builder.seen[id] {
		return
	}
	builder.seen[id] = true
	builder.plan.Gates = append(builder.plan.Gates, RuntimeGate{ID: id, Enabled: true, Source: source})
	builder.plan.GateIDs = append(builder.plan.GateIDs, id)
}

func runtimeGatePlanLabels(plan RuntimeGatePlan) map[string]string {
	if plan.Contract == "" || plan.Architecture == "" {
		return nil
	}
	labels := map[string]string{
		"engine_runtime_gate_plan_contract":    plan.Contract,
		"engine_runtime_gate_plan_reactive":    "true",
		"engine_runtime_gate_architecture":     plan.Architecture,
		"engine_runtime_gate_count":            strconv.Itoa(len(plan.GateIDs)),
		"engine_runtime_gate_ids":              runtimeGateIDsCSV(plan.GateIDs),
		"engine_runtime_gate_ambient_env":      "false",
		"engine_runtime_gate_external_control": "false",
	}
	if plan.Family != "" {
		labels["engine_runtime_gate_family"] = plan.Family
	}
	if plan.RuntimeStatus != "" {
		labels["engine_runtime_gate_runtime_status"] = string(plan.RuntimeStatus)
	}
	for _, gate := range plan.Gates {
		if gate.ID != "" {
			labels["engine_runtime_gate_"+string(gate.ID)] = strconv.FormatBool(gate.Enabled)
		}
	}
	return labels
}

func runtimeGateAnyLabelBool(labels map[string]string, keys ...string) bool {
	for _, key := range keys {
		if runtimeGateLabelBool(labels, key) {
			return true
		}
	}
	return false
}

func runtimeGateLabelBool(labels map[string]string, key string) bool {
	value := strings.TrimSpace(strings.ToLower(labels[key]))
	switch value {
	case "1", "true", "yes", "on", "enabled", "linked", "ready":
		return true
	default:
		return false
	}
}

func runtimeGateIDsCSV(ids []RuntimeGateID) string {
	parts := make([]string, 0, len(ids))
	for _, id := range ids {
		if id != "" {
			parts = append(parts, string(id))
		}
	}
	return strings.Join(parts, ",")
}
