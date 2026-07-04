// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"strconv"
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/profile"
)

const RouteSetContract = "rocm-model-route-set-v1"

// RouteSet is the model-owned registry/factory answer for a concrete model
// identity. It groups the per-feature catalogues so callers can react to the
// loaded model through one stable model package contract.
type RouteSet struct {
	Contract                 string                     `json:"contract,omitempty"`
	Architecture             string                     `json:"architecture,omitempty"`
	Family                   string                     `json:"family,omitempty"`
	Model                    inference.ModelIdentity    `json:"model,omitempty"`
	FeatureRoute             FeatureRoute               `json:"feature_route,omitempty"`
	CacheRoute               CacheRoute                 `json:"cache_route,omitempty"`
	LoaderRoute              LoaderRoute                `json:"loader_route,omitempty"`
	TokenizerRoute           TokenizerRoute             `json:"tokenizer_route,omitempty"`
	LoRAAdapterRoute         LoRAAdapterRoute           `json:"lora_adapter_route,omitempty"`
	MultimodalProcessorRoute MultimodalProcessorRoute   `json:"multimodal_processor_route,omitempty"`
	DiffusionSamplerRoute    DiffusionSamplerRoute      `json:"diffusion_sampler_route,omitempty"`
	StateContextRoute        StateContextRoute          `json:"state_context_route,omitempty"`
	AttachedDrafterRoute     AttachedDrafterRoute       `json:"attached_drafter_route,omitempty"`
	QuantLoaderRoute         QuantLoaderRoute           `json:"quant_loader_route,omitempty"`
	SequenceMixerRoutes      []SequenceMixerLoaderRoute `json:"sequence_mixer_loader_routes,omitempty"`
	RuntimeContractRoute     RuntimeContractRoute       `json:"runtime_contract_route,omitempty"`
	RuntimeGatePlan          RuntimeGatePlan            `json:"runtime_gate_plan,omitempty"`
	RuntimeAuthorPlan        RuntimeAuthorPlan          `json:"runtime_author_plan,omitempty"`
	Labels                   map[string]string          `json:"labels,omitempty"`
}

func (set RouteSet) Matched() bool {
	return set.Contract != "" &&
		set.Architecture != "" &&
		(set.FeatureRoute.Matched() ||
			set.CacheRoute.Matched() ||
			set.LoaderRoute.Matched() ||
			set.TokenizerRoute.Matched() ||
			set.LoRAAdapterRoute.Matched() ||
			set.MultimodalProcessorRoute.Matched() ||
			set.DiffusionSamplerRoute.Matched() ||
			set.StateContextRoute.Matched() ||
			set.AttachedDrafterRoute.Matched() ||
			set.QuantLoaderRoute.Matched() ||
			len(set.SequenceMixerRoutes) > 0 ||
			set.RuntimeContractRoute.Matched() ||
			set.RuntimeGatePlan.Matched() ||
			set.RuntimeAuthorPlan.Matched())
}

func (set RouteSet) Clone() RouteSet {
	set.Model.Labels = cloneStringMap(set.Model.Labels)
	set.FeatureRoute = set.FeatureRoute.Clone()
	set.CacheRoute = set.CacheRoute.Clone()
	set.LoaderRoute = set.LoaderRoute.Clone()
	set.TokenizerRoute = set.TokenizerRoute.Clone()
	set.LoRAAdapterRoute = set.LoRAAdapterRoute.Clone()
	set.MultimodalProcessorRoute = set.MultimodalProcessorRoute.Clone()
	set.DiffusionSamplerRoute = set.DiffusionSamplerRoute.Clone()
	set.StateContextRoute = set.StateContextRoute.Clone()
	set.AttachedDrafterRoute = set.AttachedDrafterRoute.Clone()
	set.QuantLoaderRoute = set.QuantLoaderRoute.Clone()
	set.SequenceMixerRoutes = cloneSequenceMixerLoaderRoutes(set.SequenceMixerRoutes)
	set.RuntimeContractRoute = set.RuntimeContractRoute.Clone()
	set.RuntimeGatePlan = set.RuntimeGatePlan.Clone()
	set.RuntimeAuthorPlan = set.RuntimeAuthorPlan.Clone()
	set.Labels = cloneStringMap(set.Labels)
	return set
}

// RouteSetOptions provides caller-owned catalogues that live outside the model
// package, such as the production quant matrix.
type RouteSetOptions struct {
	QuantLoaderPacks []QuantLoaderPack
}

func RouteSetForIdentity(path string, identity inference.ModelIdentity) (RouteSet, bool) {
	return RouteSetForIdentityWithOptions(path, identity, RouteSetOptions{})
}

func RouteSetForIdentityWithOptions(path string, identity inference.ModelIdentity, opts RouteSetOptions) (RouteSet, bool) {
	identity = routeSetIdentity(path, identity)
	set := RouteSet{
		Contract: RouteSetContract,
		Model:    identity,
	}
	if route, ok := FeatureRouteForIdentity(path, identity); ok {
		set.FeatureRoute = route
	}
	if route, ok := CacheRouteForIdentity(path, identity); ok {
		set.CacheRoute = route
	}
	if route, ok := LoaderRouteForIdentity(path, identity); ok {
		set.LoaderRoute = route
	}
	if route, ok := TokenizerRouteForIdentity(path, identity); ok {
		set.TokenizerRoute = route
	}
	if route, ok := LoRAAdapterRouteForIdentity(path, identity); ok {
		set.LoRAAdapterRoute = route
	}
	if route, ok := MultimodalProcessorRouteForIdentity(path, identity); ok {
		set.MultimodalProcessorRoute = route
	}
	if route, ok := DiffusionSamplerRouteForIdentity(path, identity); ok {
		set.DiffusionSamplerRoute = route
	}
	if route, ok := StateContextRouteForIdentity(path, identity); ok {
		set.StateContextRoute = route
	}
	if route, ok := AttachedDrafterRouteForIdentity(path, identity); ok {
		set.AttachedDrafterRoute = route
	}
	if route, ok := quantLoaderRouteForIdentity(identity, opts.QuantLoaderPacks); ok {
		set.QuantLoaderRoute = route
	}
	set.SequenceMixerRoutes = sequenceMixerLoaderRoutesForIdentity(identity)
	if route, ok := RuntimeContractRouteForIdentity(path, identity); ok {
		set.RuntimeContractRoute = route
	}
	set.Architecture = routeSetArchitecture(set)
	set.Family = routeSetFamily(set)
	set.RuntimeGatePlan = RuntimeGatePlanForRouteSet(set)
	set.RuntimeAuthorPlan = RuntimeAuthorPlanForRouteSet(set)
	set.Labels = routeSetLabels(set)
	if !set.Matched() {
		return RouteSet{}, false
	}
	return set.Clone(), true
}

func RouteSetForInfo(path string, info inference.ModelInfo, labels map[string]string, opts RouteSetOptions) (RouteSet, bool) {
	return RouteSetForIdentityWithOptions(path, inference.ModelIdentity{
		Path:         path,
		Architecture: info.Architecture,
		VocabSize:    info.VocabSize,
		NumLayers:    info.NumLayers,
		HiddenSize:   info.HiddenSize,
		QuantBits:    info.QuantBits,
		QuantGroup:   info.QuantGroup,
		Labels:       cloneStringMap(labels),
	}, opts)
}

func RouteSetForInspection(inspection *inference.ModelPackInspection, opts RouteSetOptions) (RouteSet, bool) {
	if inspection == nil {
		return RouteSet{}, false
	}
	identity := inspection.Model
	path := firstNonEmpty(identity.Path, inspection.Path)
	identity.Path = path
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
	return RouteSetForIdentityWithOptions(path, identity, opts)
}

func routeSetIdentity(path string, identity inference.ModelIdentity) inference.ModelIdentity {
	identity.Labels = cloneStringMap(identity.Labels)
	if identity.Path == "" {
		identity.Path = path
	}
	return identity
}

func quantLoaderRouteForIdentity(identity inference.ModelIdentity, packs []QuantLoaderPack) (QuantLoaderRoute, bool) {
	tokens := QuantLoaderIdentityTokens(identity)
	for _, token := range tokens {
		if route, ok := RegisteredQuantLoaderRouteForToken(token); ok {
			return route, true
		}
	}
	if len(packs) == 0 {
		return QuantLoaderRoute{}, false
	}
	routes := DefaultQuantLoaderRoutesForPacks(packs)
	for _, token := range tokens {
		for _, route := range routes {
			if QuantLoaderRouteMatchesToken(route, token) {
				return route.Clone(), true
			}
		}
	}
	return QuantLoaderRoute{}, false
}

func routeSetArchitecture(set RouteSet) string {
	return firstNonEmpty(
		set.FeatureRoute.Architecture,
		set.CacheRoute.Architecture,
		set.LoaderRoute.Architecture,
		set.TokenizerRoute.Architecture,
		set.LoRAAdapterRoute.Architecture,
		set.MultimodalProcessorRoute.Architecture,
		set.DiffusionSamplerRoute.Architecture,
		set.StateContextRoute.Architecture,
		set.AttachedDrafterRoute.Architecture,
		set.QuantLoaderRoute.Architecture,
		set.RuntimeContractRoute.Architecture,
		set.RuntimeGatePlan.Architecture,
		set.RuntimeAuthorPlan.Architecture,
		set.Model.Labels["engine_architecture_resolved"],
		set.Model.Labels["architecture_resolved"],
		set.Model.Architecture,
	)
}

func routeSetFamily(set RouteSet) string {
	return firstNonEmpty(
		set.FeatureRoute.Family,
		set.CacheRoute.Family,
		set.LoaderRoute.Family,
		set.TokenizerRoute.Family,
		set.LoRAAdapterRoute.Family,
		set.MultimodalProcessorRoute.Family,
		set.DiffusionSamplerRoute.Family,
		set.StateContextRoute.Family,
		set.AttachedDrafterRoute.Family,
		set.QuantLoaderRoute.Family,
		set.RuntimeContractRoute.Family,
		set.RuntimeGatePlan.Family,
		set.RuntimeAuthorPlan.Family,
		set.Architecture,
	)
}

func routeSetLabels(set RouteSet) map[string]string {
	if set.Architecture == "" {
		return nil
	}
	labels := map[string]string{
		"engine_route_set_contract":         set.Contract,
		"engine_route_set_architecture":     set.Architecture,
		"engine_route_set_feature":          strconv.FormatBool(set.FeatureRoute.Matched()),
		"engine_route_set_cache":            strconv.FormatBool(set.CacheRoute.Matched()),
		"engine_route_set_loader":           strconv.FormatBool(set.LoaderRoute.Matched()),
		"engine_route_set_tokenizer":        strconv.FormatBool(set.TokenizerRoute.Matched()),
		"engine_route_set_lora_adapter":     strconv.FormatBool(set.LoRAAdapterRoute.Matched()),
		"engine_route_set_multimodal":       strconv.FormatBool(set.MultimodalProcessorRoute.Matched()),
		"engine_route_set_diffusion":        strconv.FormatBool(set.DiffusionSamplerRoute.Matched()),
		"engine_route_set_state_context":    strconv.FormatBool(set.StateContextRoute.Matched()),
		"engine_route_set_drafter":          strconv.FormatBool(set.AttachedDrafterRoute.Matched()),
		"engine_route_set_quant_loader":     strconv.FormatBool(set.QuantLoaderRoute.Matched()),
		"engine_route_set_sequence_mixer":   strconv.FormatBool(len(set.SequenceMixerRoutes) > 0),
		"engine_route_set_runtime_contract": strconv.FormatBool(set.RuntimeContractRoute.Matched()),
		"engine_route_set_runtime_gate":     strconv.FormatBool(set.RuntimeGatePlan.Matched()),
		"engine_route_set_runtime_author":   strconv.FormatBool(set.RuntimeAuthorPlan.Matched()),
	}
	if set.Family != "" {
		labels["engine_route_set_family"] = set.Family
	}
	if set.CacheRoute.Matched() {
		labels["engine_route_set_cache_modes"] = strings.Join(set.CacheRoute.ModeNames, ",")
		labels["engine_route_set_cache_recommended_mode"] = set.CacheRoute.RecommendedMode
		for key, value := range set.CacheRoute.Labels {
			if value != "" {
				labels[key] = value
			}
		}
	}
	if set.LoaderRoute.Loader != "" {
		labels["engine_route_set_loader_name"] = set.LoaderRoute.Loader
	}
	if set.QuantLoaderRoute.Mode != "" {
		labels["engine_route_set_quant_mode"] = set.QuantLoaderRoute.Mode
	}
	if len(set.SequenceMixerRoutes) > 0 {
		labels["engine_route_set_sequence_mixer_kinds"] = sequenceMixerRouteKindCSV(set.SequenceMixerRoutes)
		labels["engine_route_set_sequence_mixer_cache_modes"] = sequenceMixerRouteCacheModeCSV(set.SequenceMixerRoutes)
	}
	if set.RuntimeContractRoute.Matched() {
		labels["engine_route_set_runtime_contract_ids"] = runtimeContractIDsCSV(set.RuntimeContractRoute.ContractIDs)
		labels["engine_route_set_runtime_contract_count"] = strconv.Itoa(len(set.RuntimeContractRoute.ContractIDs))
	}
	if set.RuntimeGatePlan.Matched() {
		labels["engine_route_set_runtime_gate_ids"] = runtimeGateIDsCSV(set.RuntimeGatePlan.GateIDs)
		labels["engine_route_set_runtime_gate_count"] = strconv.Itoa(len(set.RuntimeGatePlan.GateIDs))
		for key, value := range set.RuntimeGatePlan.Labels {
			if value != "" {
				labels[key] = value
			}
		}
	}
	if set.RuntimeAuthorPlan.Matched() {
		labels["engine_route_set_runtime_author_ids"] = runtimeAuthorCapabilityIDsCSV(set.RuntimeAuthorPlan.CapabilityIDs)
		labels["engine_route_set_runtime_author_count"] = strconv.Itoa(len(set.RuntimeAuthorPlan.CapabilityIDs))
		for key, value := range set.RuntimeAuthorPlan.Labels {
			if value != "" {
				labels[key] = value
			}
		}
	}
	return labels
}

func sequenceMixerLoaderRoutesForIdentity(identity inference.ModelIdentity) []SequenceMixerLoaderRoute {
	kinds := sequenceMixerIdentityKinds(identity)
	routes := make([]SequenceMixerLoaderRoute, 0, len(kinds))
	for _, kind := range kinds {
		route, ok := SequenceMixerLoaderRouteForKind(kind)
		if !ok || !route.Matched() {
			continue
		}
		routes = append(routes, route)
	}
	return cloneSequenceMixerLoaderRoutes(routes)
}

func sequenceMixerIdentityKinds(identity inference.ModelIdentity) []string {
	seen := map[string]bool{}
	kinds := make([]string, 0)
	addKind := func(kind string) {
		kind = NormalizeSequenceMixerKind(kind)
		if kind == "" || seen[kind] {
			return
		}
		if _, ok := SequenceMixerFamilyByKind(kind); !ok {
			return
		}
		seen[kind] = true
		kinds = append(kinds, kind)
	}
	architecture := profile.ArchitectureID(firstNonEmpty(
		identity.Labels["engine_architecture_resolved"],
		identity.Labels["architecture_resolved"],
		identity.Architecture,
	))
	for _, key := range []string{
		"engine_mixer_loader_kind",
		"sequence_mixer_kind",
		"sequence_mixer_model_type",
	} {
		addKind(identity.Labels[key])
	}
	if sequenceMixerArchitectureUsesLayerTypes(architecture) {
		addKind(identity.Labels["model_type"])
		for _, key := range []string{
			"engine_sequence_mixer_layer_types",
			"sequence_mixer_layer_types",
			"layer_types",
		} {
			for _, kind := range splitSequenceMixerKindCSV(identity.Labels[key]) {
				addKind(kind)
			}
		}
	}
	addKind(architecture)
	return kinds
}

func sequenceMixerArchitectureUsesLayerTypes(architecture string) bool {
	if architecture == "composed" || architecture == "hybrid" {
		return true
	}
	_, ok := SequenceMixerFamilyByKind(architecture)
	return ok
}

func splitSequenceMixerKindCSV(value string) []string {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		if kind := NormalizeSequenceMixerKind(part); kind != "" {
			out = append(out, kind)
		}
	}
	return out
}

func cloneSequenceMixerLoaderRoutes(routes []SequenceMixerLoaderRoute) []SequenceMixerLoaderRoute {
	out := make([]SequenceMixerLoaderRoute, 0, len(routes))
	for _, route := range routes {
		if route.Matched() {
			out = append(out, route.Clone())
		}
	}
	return out
}

func sequenceMixerRouteKindCSV(routes []SequenceMixerLoaderRoute) string {
	kinds := make([]string, 0, len(routes))
	for _, route := range routes {
		if route.Kind != "" {
			kinds = append(kinds, route.Kind)
		}
	}
	return strings.Join(kinds, ",")
}

func sequenceMixerRouteCacheModeCSV(routes []SequenceMixerLoaderRoute) string {
	modes := make([]string, 0, len(routes))
	for _, route := range routes {
		if route.CacheMode != "" && !slices.Contains(modes, route.CacheMode) {
			modes = append(modes, route.CacheMode)
		}
	}
	return strings.Join(modes, ",")
}
