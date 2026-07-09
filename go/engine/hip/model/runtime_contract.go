// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"strconv"
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/registry"
	"dappco.re/go/inference/engine/hip/profile"
)

const (
	RuntimeContractRegistryContract = "rocm-model-runtime-contract-registry-v1"
	RuntimeContractRouteName        = "model-runtime-contract-route"
)

type RuntimeContractID string

const (
	RuntimeContractLastTokenLogits          RuntimeContractID = "last_token_logits"
	RuntimeContractGreedyToken              RuntimeContractID = "greedy_token"
	RuntimeContractSuppressedGreedyToken    RuntimeContractID = "suppressed_greedy_token"
	RuntimeContractQueryHeads               RuntimeContractID = "query_heads"
	RuntimeContractLoRALinearResolver       RuntimeContractID = "lora_linear_resolver"
	RuntimeContractDenseSplitParts          RuntimeContractID = "dense_split_parts"
	RuntimeContractCacheTopology            RuntimeContractID = "cache_topology"
	RuntimeContractAttentionCacheLayout     RuntimeContractID = "attention_cache_layout"
	RuntimeContractModelCloser              RuntimeContractID = "model_closer"
	RuntimeContractFixedSlidingPrefillLimit RuntimeContractID = "fixed_sliding_prefill_limit"
	RuntimeContractFixedSlidingCache        RuntimeContractID = "fixed_sliding_cache"
	RuntimeContractThoughtChannelSuppressor RuntimeContractID = "thought_channel_suppressor"
	RuntimeContractModelInfoReporter        RuntimeContractID = "model_info_reporter"
	RuntimeContractMoETextRuntimeReporter   RuntimeContractID = "moe_text_runtime_reporter"
	RuntimeContractDecodeUnavailableReport  RuntimeContractID = "decode_unavailable_reporter"
	RuntimeContractHybridAttentionCachePlan RuntimeContractID = "hybrid_attention_cache_plan"
)

// RuntimeContractRoute is the ROCm analogue of go-mlx's optional model
// capability interfaces. It is metadata first: concrete HIP/CUDA/CPU runners can
// self-register richer routes, while model discovery can already report which
// optional contracts a loaded profile should be expected to expose.
type RuntimeContractRoute struct {
	Contract                    string                         `json:"contract,omitempty"`
	Name                        string                         `json:"name,omitempty"`
	Architecture                string                         `json:"architecture,omitempty"`
	Family                      string                         `json:"family,omitempty"`
	RuntimeStatus               inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	Registered                  bool                           `json:"registered,omitempty"`
	NativeRuntime               bool                           `json:"native_runtime,omitempty"`
	MetadataOnly                bool                           `json:"metadata_only,omitempty"`
	TextGenerate                bool                           `json:"text_generate,omitempty"`
	LastTokenLogits             bool                           `json:"last_token_logits,omitempty"`
	GreedyToken                 bool                           `json:"greedy_token,omitempty"`
	SuppressedGreedyToken       bool                           `json:"suppressed_greedy_token,omitempty"`
	QueryHeads                  bool                           `json:"query_heads,omitempty"`
	LoRALinearResolver          bool                           `json:"lora_linear_resolver,omitempty"`
	DenseSplitParts             bool                           `json:"dense_split_parts,omitempty"`
	CacheTopology               bool                           `json:"cache_topology,omitempty"`
	AttentionCacheLayout        bool                           `json:"attention_cache_layout,omitempty"`
	ModelCloser                 bool                           `json:"model_closer,omitempty"`
	FixedSlidingPrefillLimit    bool                           `json:"fixed_sliding_prefill_limit,omitempty"`
	FixedSlidingCache           bool                           `json:"fixed_sliding_cache,omitempty"`
	ThoughtChannelSuppressor    bool                           `json:"thought_channel_suppressor,omitempty"`
	ModelInfoReporter           bool                           `json:"model_info_reporter,omitempty"`
	MoETextRuntimeReporter      bool                           `json:"moe_text_runtime_reporter,omitempty"`
	DecodeUnavailableReporter   bool                           `json:"decode_unavailable_reporter,omitempty"`
	HybridAttentionCachePlanner bool                           `json:"hybrid_attention_cache_planner,omitempty"`
	ContractIDs                 []RuntimeContractID            `json:"contract_ids,omitempty"`
	Labels                      map[string]string              `json:"labels,omitempty"`
}

func (route RuntimeContractRoute) Matched() bool {
	return route.Contract != "" && route.Architecture != "" && route.Name != ""
}

func (route RuntimeContractRoute) Clone() RuntimeContractRoute {
	route.ContractIDs = append([]RuntimeContractID(nil), route.ContractIDs...)
	route.Labels = cloneStringMap(route.Labels)
	return route
}

var registeredRuntimeContracts = registry.NewOrdered[string, RuntimeContractRoute]()

func RegisterRuntimeContractRoute(route RuntimeContractRoute) {
	route = NormalizeRuntimeContractRoute(route)
	if !route.Matched() {
		return
	}
	registeredRuntimeContracts.Put(route.Architecture, route)
}

func RegisteredRuntimeContractArchitectures() []string {
	return registeredRuntimeContracts.Keys()
}

func RegisteredRuntimeContractRoutes() []RuntimeContractRoute {
	return registeredRuntimeContractSnapshot()
}

func ReplaceRegisteredRuntimeContractRoutes(routes []RuntimeContractRoute) {
	order := make([]string, 0, len(routes))
	values := make(map[string]RuntimeContractRoute, len(routes))
	for _, route := range routes {
		route = NormalizeRuntimeContractRoute(route)
		if !route.Matched() {
			continue
		}
		if _, ok := values[route.Architecture]; !ok {
			order = append(order, route.Architecture)
		}
		values[route.Architecture] = route
	}
	registeredRuntimeContracts.Restore(order, values)
}

func RegisteredRuntimeContractRouteForArchitecture(architecture string) (RuntimeContractRoute, bool) {
	return registeredRuntimeContractForArchitecture(architecture)
}

func RuntimeContractRouteForArchitecture(architecture string) (RuntimeContractRoute, bool) {
	architecture = profile.ArchitectureID(architecture)
	if architecture == "" {
		return RuntimeContractRoute{}, false
	}
	if route, ok := registeredRuntimeContractForArchitecture(architecture); ok {
		return route, true
	}
	architectureProfile, ok := profile.LookupArchitectureProfile(architecture)
	if !ok {
		return RuntimeContractRoute{}, false
	}
	return runtimeContractRouteForProfile(architectureProfile), true
}

func RuntimeContractRouteForIdentity(path string, identity inference.ModelIdentity) (RuntimeContractRoute, bool) {
	if identity.Path == "" {
		identity.Path = path
	}
	architecture := firstNonEmpty(
		identity.Labels["engine_architecture_resolved"],
		identity.Labels["architecture_resolved"],
		identity.Architecture,
	)
	return RuntimeContractRouteForArchitecture(architecture)
}

func RuntimeContractRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (RuntimeContractRoute, bool) {
	return RuntimeContractRouteForIdentity(path, inference.ModelIdentity{
		Path:         path,
		Architecture: info.Architecture,
		VocabSize:    info.VocabSize,
		NumLayers:    info.NumLayers,
		HiddenSize:   info.HiddenSize,
		QuantBits:    info.QuantBits,
		QuantGroup:   info.QuantGroup,
		Labels:       cloneStringMap(labels),
	})
}

func RuntimeContractRouteForInspection(inspection *inference.ModelPackInspection) (RuntimeContractRoute, bool) {
	if inspection == nil {
		return RuntimeContractRoute{}, false
	}
	identity := inspection.Model
	if identity.Path == "" {
		identity.Path = inspection.Path
	}
	labels := cloneStringMap(inspection.Labels)
	if labels == nil {
		labels = map[string]string{}
	}
	for key, value := range identity.Labels {
		if value != "" {
			labels[key] = value
		}
	}
	identity.Labels = labels
	return RuntimeContractRouteForIdentity(identity.Path, identity)
}

func DefaultRuntimeContractRoutes() []RuntimeContractRoute {
	profiles := profile.ArchitectureProfiles()
	routes := make([]RuntimeContractRoute, 0, len(profiles)+len(registeredRuntimeContracts.Keys()))
	seen := map[string]int{}
	for _, architectureProfile := range profiles {
		route := runtimeContractRouteForProfile(architectureProfile)
		if !route.Matched() {
			continue
		}
		seen[route.Architecture] = len(routes)
		routes = append(routes, route)
	}
	for _, route := range registeredRuntimeContractSnapshot() {
		if !route.Matched() {
			continue
		}
		if index, ok := seen[route.Architecture]; ok {
			routes[index] = route.Clone()
			continue
		}
		seen[route.Architecture] = len(routes)
		routes = append(routes, route.Clone())
	}
	return cloneRuntimeContractRoutes(routes)
}

func NormalizeRuntimeContractRoute(route RuntimeContractRoute) RuntimeContractRoute {
	route.Architecture = profile.ArchitectureID(route.Architecture)
	if route.Architecture == "" {
		return RuntimeContractRoute{}
	}
	architectureProfile, hasProfile := profile.LookupArchitectureProfile(route.Architecture)
	if route.Contract == "" {
		route.Contract = RuntimeContractRegistryContract
	}
	if route.Name == "" {
		route.Name = RuntimeContractRouteName
	}
	if route.Family == "" && hasProfile {
		route.Family = firstNonEmpty(architectureProfile.Family, architectureProfile.ID)
	}
	if route.Family == "" {
		route.Family = route.Architecture
	}
	if route.RuntimeStatus == "" && hasProfile {
		route.RuntimeStatus = architectureProfile.RuntimeStatus
	}
	if route.RuntimeStatus == "" && route.NativeRuntime {
		route.RuntimeStatus = inference.FeatureRuntimeNative
	}
	route.Registered = true
	if hasProfile {
		route.NativeRuntime = route.NativeRuntime || architectureProfile.NativeRuntime
		route.TextGenerate = route.TextGenerate || (architectureProfile.Generation && architectureProfile.NativeRuntime && !architectureProfile.AttachedOnly)
		route.MetadataOnly = route.MetadataOnly || !architectureProfile.NativeRuntime
		route.ModelInfoReporter = true
		route.DecodeUnavailableReporter = route.DecodeUnavailableReporter ||
			!architectureProfile.NativeRuntime ||
			!architectureProfile.Generation ||
			architectureProfile.AttachedOnly
		route.MoETextRuntimeReporter = route.MoETextRuntimeReporter || runtimeContractProfileDeclaresMoETextRuntime(architectureProfile)
		route.HybridAttentionCachePlanner = route.HybridAttentionCachePlanner || runtimeContractProfileDeclaresHybridCachePlanner(architectureProfile)
		if runtimeContractProfileDeclaresGemma4Hooks(architectureProfile) {
			route.LastTokenLogits = true
			route.GreedyToken = true
			route.SuppressedGreedyToken = true
			route.QueryHeads = true
			route.LoRALinearResolver = true
			route.DenseSplitParts = true
			route.CacheTopology = true
			route.AttentionCacheLayout = true
			route.ModelCloser = true
			route.FixedSlidingPrefillLimit = true
			route.FixedSlidingCache = true
			route.ThoughtChannelSuppressor = true
		}
	}
	if !route.NativeRuntime {
		route.MetadataOnly = true
	}
	route.ContractIDs = mergeRuntimeContractIDs(runtimeContractIDs(route), route.ContractIDs)
	route.Labels = runtimeContractRouteLabels(route)
	return route.Clone()
}

func runtimeContractRouteForProfile(architectureProfile profile.ArchitectureProfile) RuntimeContractRoute {
	architectureProfile = profile.NormalizeArchitectureProfile(architectureProfile)
	route := RuntimeContractRoute{
		Contract:                    RuntimeContractRegistryContract,
		Name:                        RuntimeContractRouteName,
		Architecture:                architectureProfile.ID,
		Family:                      firstNonEmpty(architectureProfile.Family, architectureProfile.ID),
		RuntimeStatus:               architectureProfile.RuntimeStatus,
		Registered:                  architectureProfile.ID != "",
		NativeRuntime:               architectureProfile.NativeRuntime,
		MetadataOnly:                !architectureProfile.NativeRuntime,
		TextGenerate:                architectureProfile.Generation && architectureProfile.NativeRuntime && !architectureProfile.AttachedOnly,
		ModelInfoReporter:           architectureProfile.ID != "",
		DecodeUnavailableReporter:   !architectureProfile.NativeRuntime || !architectureProfile.Generation || architectureProfile.AttachedOnly,
		MoETextRuntimeReporter:      runtimeContractProfileDeclaresMoETextRuntime(architectureProfile),
		HybridAttentionCachePlanner: runtimeContractProfileDeclaresHybridCachePlanner(architectureProfile),
	}
	if runtimeContractProfileDeclaresGemma4Hooks(architectureProfile) {
		route.LastTokenLogits = true
		route.GreedyToken = true
		route.SuppressedGreedyToken = true
		route.QueryHeads = true
		route.LoRALinearResolver = true
		route.DenseSplitParts = true
		route.CacheTopology = true
		route.AttentionCacheLayout = true
		route.ModelCloser = true
		route.FixedSlidingPrefillLimit = true
		route.FixedSlidingCache = true
		route.ThoughtChannelSuppressor = true
	}
	route.ContractIDs = runtimeContractIDs(route)
	route.Labels = runtimeContractRouteLabels(route)
	return route.Clone()
}

func registeredRuntimeContractForArchitecture(architecture string) (RuntimeContractRoute, bool) {
	route, ok := registeredRuntimeContracts.Get(profile.ArchitectureID(architecture))
	if !ok {
		return RuntimeContractRoute{}, false
	}
	return route.Clone(), true
}

func registeredRuntimeContractSnapshot() []RuntimeContractRoute {
	routes := registeredRuntimeContracts.Values()
	out := make([]RuntimeContractRoute, 0, len(routes))
	for _, route := range routes {
		out = append(out, route.Clone())
	}
	return out
}

func runtimeContractProfileDeclaresGemma4Hooks(architectureProfile profile.ArchitectureProfile) bool {
	id := firstNonEmpty(architectureProfile.ID, architectureProfile.Family)
	return id == "gemma4" ||
		id == "gemma4_text" ||
		id == "gemma4_unified" ||
		id == "gemma4_assistant" ||
		architectureProfile.Family == "gemma4"
}

func runtimeContractProfileDeclaresMoETextRuntime(architectureProfile profile.ArchitectureProfile) bool {
	switch architectureProfile.ID {
	case "qwen3_moe", "qwen3_6_moe", "mixtral", "kimi", "gpt-oss", "minimax_m2":
		return true
	default:
		return architectureProfile.MoE
	}
}

func runtimeContractProfileDeclaresHybridCachePlanner(architectureProfile profile.ArchitectureProfile) bool {
	switch architectureProfile.ID {
	case "qwen3_6", "qwen3_6_moe":
		return true
	default:
		return false
	}
}

func runtimeContractRouteLabels(route RuntimeContractRoute) map[string]string {
	if !route.Matched() {
		return nil
	}
	labels := map[string]string{
		"engine_runtime_contract_route_contract":                       route.Contract,
		"engine_runtime_contract_route":                                route.Name,
		"engine_runtime_contract_registered":                           strconv.FormatBool(route.Registered),
		"engine_runtime_contract_native_runtime":                       strconv.FormatBool(route.NativeRuntime),
		"engine_runtime_contract_metadata_only":                        strconv.FormatBool(route.MetadataOnly),
		"engine_runtime_contract_text_generate":                        strconv.FormatBool(route.TextGenerate),
		"engine_runtime_contract_last_token_logits":                    strconv.FormatBool(route.LastTokenLogits),
		"engine_runtime_contract_greedy_token":                         strconv.FormatBool(route.GreedyToken),
		"engine_runtime_contract_suppressed_greedy_token":              strconv.FormatBool(route.SuppressedGreedyToken),
		"engine_runtime_contract_query_heads":                          strconv.FormatBool(route.QueryHeads),
		"engine_runtime_contract_lora_linear_resolver":                 strconv.FormatBool(route.LoRALinearResolver),
		"engine_runtime_contract_dense_split_parts":                    strconv.FormatBool(route.DenseSplitParts),
		"engine_runtime_contract_cache_topology":                       strconv.FormatBool(route.CacheTopology),
		"engine_runtime_contract_attention_cache_layout":               strconv.FormatBool(route.AttentionCacheLayout),
		"engine_runtime_contract_model_closer":                         strconv.FormatBool(route.ModelCloser),
		"engine_runtime_contract_fixed_sliding_prefill_limit":          strconv.FormatBool(route.FixedSlidingPrefillLimit),
		"engine_runtime_contract_fixed_sliding_cache":                  strconv.FormatBool(route.FixedSlidingCache),
		"engine_runtime_contract_thought_channel_suppressor":           strconv.FormatBool(route.ThoughtChannelSuppressor),
		"engine_runtime_contract_model_info_reporter":                  strconv.FormatBool(route.ModelInfoReporter),
		"engine_runtime_contract_moe_text_runtime_reporter":            strconv.FormatBool(route.MoETextRuntimeReporter),
		"engine_runtime_contract_decode_unavailable_reporter":          strconv.FormatBool(route.DecodeUnavailableReporter),
		"engine_runtime_contract_hybrid_attention_cache_planner":       strconv.FormatBool(route.HybridAttentionCachePlanner),
		"engine_runtime_contract_go_mlx_optional_interface_compatible": strconv.FormatBool(len(route.ContractIDs) > 0),
	}
	if route.Architecture != "" {
		labels["engine_runtime_contract_architecture"] = route.Architecture
	}
	if route.Family != "" {
		labels["engine_runtime_contract_family"] = route.Family
	}
	if route.RuntimeStatus != "" {
		labels["engine_runtime_contract_runtime_status"] = string(route.RuntimeStatus)
	}
	if len(route.ContractIDs) > 0 {
		labels["engine_runtime_contract_ids"] = runtimeContractIDsCSV(route.ContractIDs)
		labels["engine_runtime_contract_count"] = strconv.Itoa(len(route.ContractIDs))
	}
	return labels
}

// RuntimeContractRouteLabels returns the model-owned label contract for a
// runtime-contract route. Existing labels win so probe-enriched metadata is not
// re-normalized away.
func RuntimeContractRouteLabels(route RuntimeContractRoute) map[string]string {
	if len(route.Labels) > 0 {
		return cloneStringMap(route.Labels)
	}
	route = NormalizeRuntimeContractRoute(route)
	return cloneStringMap(route.Labels)
}

func runtimeContractIDs(route RuntimeContractRoute) []RuntimeContractID {
	ids := make([]RuntimeContractID, 0, 16)
	add := func(id RuntimeContractID, enabled bool) {
		if enabled {
			ids = append(ids, id)
		}
	}
	add(RuntimeContractLastTokenLogits, route.LastTokenLogits)
	add(RuntimeContractGreedyToken, route.GreedyToken)
	add(RuntimeContractSuppressedGreedyToken, route.SuppressedGreedyToken)
	add(RuntimeContractQueryHeads, route.QueryHeads)
	add(RuntimeContractLoRALinearResolver, route.LoRALinearResolver)
	add(RuntimeContractDenseSplitParts, route.DenseSplitParts)
	add(RuntimeContractCacheTopology, route.CacheTopology)
	add(RuntimeContractAttentionCacheLayout, route.AttentionCacheLayout)
	add(RuntimeContractModelCloser, route.ModelCloser)
	add(RuntimeContractFixedSlidingPrefillLimit, route.FixedSlidingPrefillLimit)
	add(RuntimeContractFixedSlidingCache, route.FixedSlidingCache)
	add(RuntimeContractThoughtChannelSuppressor, route.ThoughtChannelSuppressor)
	add(RuntimeContractModelInfoReporter, route.ModelInfoReporter)
	add(RuntimeContractMoETextRuntimeReporter, route.MoETextRuntimeReporter)
	add(RuntimeContractDecodeUnavailableReport, route.DecodeUnavailableReporter)
	add(RuntimeContractHybridAttentionCachePlan, route.HybridAttentionCachePlanner)
	return ids
}

func mergeRuntimeContractIDs(primary, secondary []RuntimeContractID) []RuntimeContractID {
	out := make([]RuntimeContractID, 0, len(primary)+len(secondary))
	seen := map[RuntimeContractID]bool{}
	for _, ids := range [][]RuntimeContractID{primary, secondary} {
		for _, id := range ids {
			if id == "" || seen[id] {
				continue
			}
			seen[id] = true
			out = append(out, id)
		}
	}
	return out
}

func runtimeContractIDsCSV(ids []RuntimeContractID) string {
	parts := make([]string, 0, len(ids))
	for _, id := range ids {
		if id != "" {
			parts = append(parts, string(id))
		}
	}
	return strings.Join(parts, ",")
}

// RuntimeContractIDsCSV formats runtime contract IDs using the model-owned
// route label contract.
func RuntimeContractIDsCSV(ids []RuntimeContractID) string {
	return runtimeContractIDsCSV(ids)
}

func cloneRuntimeContractRoutes(routes []RuntimeContractRoute) []RuntimeContractRoute {
	out := append([]RuntimeContractRoute(nil), routes...)
	for index := range out {
		out[index] = out[index].Clone()
	}
	return out
}
