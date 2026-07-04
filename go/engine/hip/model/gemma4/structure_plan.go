// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"strconv"
	"strings"
)

// StructurePlan is the Gemma-4 load-time structure surface. It captures the
// decisions go-mlx makes while wiring a concrete model, but keeps them
// backend-neutral so HIP, CUDA, and CPU runtimes can react to the same metadata.
type StructurePlan struct {
	LayerCount                int      `json:"layer_count,omitempty"`
	LayerTypes                []string `json:"layer_types,omitempty"`
	AttentionKEqV             bool     `json:"attention_k_eq_v,omitempty"`
	AttentionKEqVDeclared     bool     `json:"attention_k_eq_v_declared,omitempty"`
	PerLayerInputs            bool     `json:"per_layer_inputs,omitempty"`
	HiddenSizePerLayerInput   int      `json:"hidden_size_per_layer_input,omitempty"`
	VocabSizePerLayerInput    int      `json:"vocab_size_per_layer_input,omitempty"`
	UseDoubleWideMLP          bool     `json:"use_double_wide_mlp,omitempty"`
	UsesSharedKV              bool     `json:"uses_shared_kv,omitempty"`
	SharedKVLayers            int      `json:"shared_kv_layers,omitempty"`
	MoERouter                 bool     `json:"moe_router,omitempty"`
	NumExperts                int      `json:"num_experts,omitempty"`
	TopKExperts               int      `json:"top_k_experts,omitempty"`
	MoEIntermediateSize       int      `json:"moe_intermediate_size,omitempty"`
	FusedExpertGateUpEligible bool     `json:"fused_expert_gate_up_eligible,omitempty"`
}

func (plan StructurePlan) HasPerLayerInputs() bool {
	return plan.PerLayerInputs || plan.HiddenSizePerLayerInput > 0 || plan.VocabSizePerLayerInput > 0
}

func (plan StructurePlan) HasMoERouter() bool {
	return plan.MoERouter && plan.NumExperts > 0 && plan.TopKExperts > 0
}

func (plan StructurePlan) SharedKVEnabled() bool {
	return plan.UsesSharedKV || plan.SharedKVLayers > 0
}

// StructurePlanOf derives the reactive structure plan from a loaded Gemma-4
// config. Weight presence can still narrow these decisions at load time; this
// is the config-owned plan that later factories and runtimes can inspect.
func StructurePlanOf(cfg TextConfig) StructurePlan {
	layerTypes := LayerTypesOf(cfg)
	plan := StructurePlan{
		LayerCount:              positiveInt(cfg.NumLayers),
		LayerTypes:              layerTypes,
		AttentionKEqV:           cfg.AttentionKEqV,
		AttentionKEqVDeclared:   cfg.AttentionKEqVSet || cfg.AttentionKEqV,
		HiddenSizePerLayerInput: positiveInt(cfg.HiddenSizePerLayer),
		VocabSizePerLayerInput:  positiveInt(cfg.VocabSizePerLayer),
		UseDoubleWideMLP:        cfg.UseDoubleWideMLP,
		SharedKVLayers:          positiveInt(cfg.KVSharedLayers),
		UsesSharedKV:            cfg.KVSharedLayers > 0,
		MoERouter:               cfg.EnableMoEBlock,
		NumExperts:              positiveInt(cfg.NumExperts),
		TopKExperts:             positiveInt(cfg.TopKExperts),
		MoEIntermediateSize:     positiveInt(cfg.MoEIntermediateSize),
	}
	if plan.LayerCount == 0 {
		plan.LayerCount = len(layerTypes)
	}
	plan.PerLayerInputs = plan.HiddenSizePerLayerInput > 0 || plan.VocabSizePerLayerInput > 0
	plan.MoERouter = plan.MoERouter && plan.NumExperts > 0 && plan.TopKExperts > 0
	plan.FusedExpertGateUpEligible = plan.HasMoERouter() && plan.MoEIntermediateSize > 0
	return plan
}

// StructurePlanOfLabels reconstructs the plan from registry/model labels. This
// is what consumers use when only an inspected model identity is available.
func StructurePlanOfLabels(labels map[string]string) StructurePlan {
	plan := StructurePlan{
		LayerCount: firstPositiveIntLabel(labels,
			"gemma4_num_hidden_layers", "num_hidden_layers",
			"gemma4_attention_layer_count", "attention_layer_count"),
		LayerTypes: parseLayerTypeCSV(firstNonEmptyLabel(labels,
			"gemma4_attention_layer_types", "attention_layer_types",
			"gemma4_layer_types", "layer_types")),
		HiddenSizePerLayerInput: firstPositiveIntLabel(labels,
			"gemma4_hidden_size_per_layer_input", "hidden_size_per_layer_input"),
		VocabSizePerLayerInput: firstPositiveIntLabel(labels,
			"gemma4_vocab_size_per_layer_input", "vocab_size_per_layer_input"),
		UseDoubleWideMLP: anyTruthyLabel(labels,
			"gemma4_use_double_wide_mlp", "use_double_wide_mlp"),
		UsesSharedKV: anyTruthyLabel(labels,
			"gemma4_shared_kv", "attention_shared_kv"),
		SharedKVLayers: firstPositiveIntLabel(labels,
			"gemma4_attention_kv_shared_layers", "attention_kv_shared_layers"),
		MoERouter: anyTruthyLabel(labels,
			"gemma4_moe_router", "gemma4_enable_moe_block", "gemma4_mixture"),
		NumExperts: firstPositiveIntLabel(labels,
			"gemma4_num_experts", "num_experts"),
		TopKExperts: firstPositiveIntLabel(labels,
			"gemma4_top_k_experts", "top_k_experts"),
		MoEIntermediateSize: firstPositiveIntLabel(labels,
			"gemma4_moe_intermediate_size", "moe_intermediate_size"),
	}
	if plan.LayerCount == 0 {
		plan.LayerCount = len(plan.LayerTypes)
	}
	plan.PerLayerInputs = anyTruthyLabel(labels, "gemma4_per_layer_inputs", "per_layer_inputs") ||
		plan.HiddenSizePerLayerInput > 0 || plan.VocabSizePerLayerInput > 0
	if value, ok := boolLabel(labels, "gemma4_attention_k_eq_v", "attention_k_eq_v"); ok {
		plan.AttentionKEqV = value
		plan.AttentionKEqVDeclared = true
	}
	plan.UsesSharedKV = plan.UsesSharedKV || plan.SharedKVLayers > 0
	plan.MoERouter = plan.MoERouter && plan.NumExperts > 0 && plan.TopKExperts > 0
	plan.FusedExpertGateUpEligible = anyTruthyLabel(labels, "gemma4_fused_expert_gate_up_eligible") ||
		(plan.HasMoERouter() && plan.MoEIntermediateSize > 0)
	return plan
}

func ApplyStructurePlanLabels(labels map[string]string, plan StructurePlan) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	labels["gemma4_structure_plan_reactive"] = "true"
	if plan.LayerCount > 0 {
		value := strconv.Itoa(plan.LayerCount)
		labels["num_hidden_layers"] = value
		labels["gemma4_num_hidden_layers"] = value
	}
	if len(plan.LayerTypes) > 0 {
		value := strings.Join(normalizeLayerTypes(plan.LayerTypes), ",")
		labels["layer_types"] = value
		labels["gemma4_layer_types"] = value
	}
	if plan.AttentionKEqVDeclared {
		value := strconv.FormatBool(plan.AttentionKEqV)
		labels["attention_k_eq_v"] = value
		labels["gemma4_attention_k_eq_v"] = value
	}
	if plan.HasPerLayerInputs() {
		labels["per_layer_inputs"] = "true"
		labels["gemma4_per_layer_inputs"] = "true"
	}
	if plan.HiddenSizePerLayerInput > 0 {
		value := strconv.Itoa(plan.HiddenSizePerLayerInput)
		labels["hidden_size_per_layer_input"] = value
		labels["gemma4_hidden_size_per_layer_input"] = value
	}
	if plan.VocabSizePerLayerInput > 0 {
		value := strconv.Itoa(plan.VocabSizePerLayerInput)
		labels["vocab_size_per_layer_input"] = value
		labels["gemma4_vocab_size_per_layer_input"] = value
	}
	if plan.UseDoubleWideMLP {
		labels["use_double_wide_mlp"] = "true"
		labels["gemma4_use_double_wide_mlp"] = "true"
	}
	if plan.SharedKVEnabled() {
		labels["attention_shared_kv"] = "true"
		labels["gemma4_shared_kv"] = "true"
	}
	if plan.SharedKVLayers > 0 {
		value := strconv.Itoa(plan.SharedKVLayers)
		labels["attention_kv_shared_layers"] = value
		labels["gemma4_attention_kv_shared_layers"] = value
	}
	if plan.HasMoERouter() {
		labels["gemma4_moe_router"] = "true"
		labels["gemma4_enable_moe_block"] = "true"
	}
	if plan.NumExperts > 0 {
		value := strconv.Itoa(plan.NumExperts)
		labels["num_experts"] = value
		labels["gemma4_num_experts"] = value
	}
	if plan.TopKExperts > 0 {
		value := strconv.Itoa(plan.TopKExperts)
		labels["top_k_experts"] = value
		labels["gemma4_top_k_experts"] = value
	}
	if plan.MoEIntermediateSize > 0 {
		value := strconv.Itoa(plan.MoEIntermediateSize)
		labels["moe_intermediate_size"] = value
		labels["gemma4_moe_intermediate_size"] = value
	}
	if plan.FusedExpertGateUpEligible {
		labels["gemma4_fused_expert_gate_up_eligible"] = "true"
	}
	return labels
}

func parseLayerTypeCSV(value string) []string {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	parts := strings.Split(value, ",")
	return normalizeLayerTypes(parts)
}

func boolLabel(labels map[string]string, keys ...string) (bool, bool) {
	for _, key := range keys {
		switch labelValue(labels, key) {
		case "true", "1", "yes":
			return true, true
		case "false", "0", "no":
			return false, true
		}
	}
	return false, false
}
