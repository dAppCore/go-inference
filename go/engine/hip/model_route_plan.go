// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"context"
	"strconv"
	"strings"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const ROCmModelRoutePlanContract = "rocm-model-route-plan-v1"

// ROCmModelRoutePlan is the compact registry/factory answer for a concrete
// model: which feature, tokenizer, adapter, multimodal, diffusion, state,
// drafter, loader, quant, and sequence-mixer routes should clients use for
// this profile.
type ROCmModelRoutePlan struct {
	Contract                 string                         `json:"contract,omitempty"`
	Architecture             string                         `json:"architecture,omitempty"`
	Family                   string                         `json:"family,omitempty"`
	Model                    inference.ModelIdentity        `json:"model,omitempty"`
	EngineFeatures           ROCmEngineFeatures             `json:"engine_features,omitempty"`
	FeatureRoute             ROCmModelFeatureRoute          `json:"feature_route,omitempty"`
	TokenizerRoute           ROCmModelTokenizerRoute        `json:"tokenizer_route,omitempty"`
	LoRAAdapterRoute         ROCmLoRAAdapterRoute           `json:"lora_adapter_route,omitempty"`
	MultimodalProcessorRoute ROCmMultimodalProcessorRoute   `json:"multimodal_processor_route,omitempty"`
	DiffusionSamplerRoute    ROCmDiffusionSamplerRoute      `json:"diffusion_sampler_route,omitempty"`
	StateContextRoute        ROCmStateContextRoute          `json:"state_context_route,omitempty"`
	AttachedDrafterRoute     ROCmAttachedDrafterRoute       `json:"attached_drafter_route,omitempty"`
	LoadStatus               ROCmModelLoadStatus            `json:"load_status,omitempty"`
	CacheRoute               rocmmodel.CacheRoute           `json:"cache_route,omitempty"`
	CacheProfile             rocmmodel.CacheProfile         `json:"cache_profile,omitempty"`
	LoaderRoute              ROCmModelLoaderRoute           `json:"loader_route,omitempty"`
	QuantLoaderRoute         ROCmQuantLoaderRoute           `json:"quant_loader_route,omitempty"`
	SequenceMixerRoutes      []ROCmSequenceMixerLoaderRoute `json:"sequence_mixer_loader_routes,omitempty"`
	RuntimeContractRoute     ROCmModelRuntimeContractRoute  `json:"runtime_contract_route,omitempty"`
	RuntimeGatePlan          rocmmodel.RuntimeGatePlan      `json:"runtime_gate_plan,omitempty"`
	RuntimeAuthorPlan        rocmmodel.RuntimeAuthorPlan    `json:"runtime_author_plan,omitempty"`
	Labels                   map[string]string              `json:"labels,omitempty"`
}

func (plan ROCmModelRoutePlan) Matched() bool {
	return plan.Contract != "" && plan.Architecture != ""
}

func (plan ROCmModelRoutePlan) clone() ROCmModelRoutePlan {
	plan.Model = rocmCloneModelIdentity(plan.Model)
	plan.EngineFeatures = plan.EngineFeatures.clone()
	plan.FeatureRoute = plan.FeatureRoute.Clone()
	plan.TokenizerRoute = plan.TokenizerRoute.Clone()
	plan.LoRAAdapterRoute = plan.LoRAAdapterRoute.Clone()
	plan.MultimodalProcessorRoute = plan.MultimodalProcessorRoute.Clone()
	plan.DiffusionSamplerRoute = plan.DiffusionSamplerRoute.Clone()
	plan.StateContextRoute = plan.StateContextRoute.Clone()
	plan.AttachedDrafterRoute = plan.AttachedDrafterRoute.Clone()
	plan.LoadStatus = plan.LoadStatus.clone()
	plan.CacheRoute = plan.CacheRoute.Clone()
	plan.CacheProfile = plan.CacheProfile.Clone()
	plan.LoaderRoute = plan.LoaderRoute.Clone()
	plan.QuantLoaderRoute = plan.QuantLoaderRoute.Clone()
	plan.SequenceMixerRoutes = cloneROCmSequenceMixerLoaderRoutes(plan.SequenceMixerRoutes)
	plan.RuntimeContractRoute = plan.RuntimeContractRoute.Clone()
	plan.RuntimeGatePlan = plan.RuntimeGatePlan.Clone()
	plan.RuntimeAuthorPlan = plan.RuntimeAuthorPlan.Clone()
	plan.Labels = cloneStringMap(plan.Labels)
	return plan
}

func ROCmModelRoutePlanForIdentity(path string, model inference.ModelIdentity) (ROCmModelRoutePlan, bool) {
	profile, ok := ResolveROCmModelProfile(path, model)
	if !ok {
		return ROCmModelRoutePlan{}, false
	}
	return ROCmModelRoutePlanForProfile(profile), true
}

func ROCmModelRoutePlanForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmModelRoutePlan, bool) {
	profile, ok := ResolveROCmModelProfileForInfo(path, info, labels)
	if !ok {
		return ROCmModelRoutePlan{}, false
	}
	return ROCmModelRoutePlanForProfile(profile), true
}

func ROCmModelRoutePlanForInspection(inspection *inference.ModelPackInspection) (ROCmModelRoutePlan, bool) {
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok {
		return ROCmModelRoutePlan{}, false
	}
	return ROCmModelRoutePlanForProfile(profile), true
}

func ROCmModelRoutePlanForModel(model inference.TextModel) (ROCmModelRoutePlan, bool) {
	if model == nil {
		return ROCmModelRoutePlan{}, false
	}
	if reporter, ok := model.(ROCmModelRoutePlanReporter); ok {
		plan := reporter.ModelRoutePlan()
		if plan.Matched() {
			return rocmModelRoutePlanWithLiveCacheProfile(plan, model), true
		}
	}
	profile, ok := ResolveROCmModelProfileForModel(model)
	if !ok {
		return ROCmModelRoutePlan{}, false
	}
	plan := ROCmModelRoutePlanForProfile(profile)
	if !plan.Matched() {
		return ROCmModelRoutePlan{}, false
	}
	return rocmModelRoutePlanWithLiveCacheProfile(plan, model), true
}

// ROCmModelRoutePlanForProfileAndModel builds the route plan from the resolved
// registry profile, then overlays live facts exposed by the loaded model. Daemon
// and API paths use this when request labels or model paths refine the static
// profile but the runtime model still owns cache/profile observations.
func ROCmModelRoutePlanForProfileAndModel(profile ROCmModelProfile, model inference.TextModel) ROCmModelRoutePlan {
	plan := ROCmModelRoutePlanForProfile(profile)
	if !plan.Matched() {
		return ROCmModelRoutePlan{}
	}
	return rocmModelRoutePlanWithLiveCacheProfile(plan, model)
}

func rocmModelRoutePlanWithLiveCacheProfile(plan ROCmModelRoutePlan, model inference.TextModel) ROCmModelRoutePlan {
	plan = plan.clone()
	if !plan.Matched() {
		return plan
	}
	reporter, ok := model.(ROCmCacheProfileReporter)
	if !ok || reporter == nil {
		return plan
	}
	cacheProfile, err := reporter.CacheProfile(context.Background())
	if err != nil || !cacheProfile.Matched() {
		return plan
	}
	plan.CacheProfile = cacheProfile.Clone()
	plan.Labels = rocmModelRoutePlanLabels(plan)
	return plan.clone()
}

func ROCmModelRoutePlanForProfile(profile ROCmModelProfile) ROCmModelRoutePlan {
	if !profile.Matched() {
		return ROCmModelRoutePlan{}
	}
	routeProfile := profile.clone()
	if routeProfile.Architecture == "" {
		routeProfile.Architecture = normalizeROCmArchitecture(routeProfile.Model.Architecture)
	}
	if routeProfile.Family == "" {
		routeProfile.Family = firstNonEmptyString(routeProfile.Name, routeProfile.Architecture)
	}
	modelRouteSet, hasModelRouteSet := rocmModelRouteSetForProfile(routeProfile)
	features := routeProfile.EngineFeatures
	if features.empty() {
		features = ROCmEngineFeaturesForProfile(routeProfile)
	}
	routeProfile.EngineFeatures = features

	featureRoute := routeProfile.FeatureRoute
	if !featureRoute.Matched() {
		featureRoute = ROCmModelFeatureRouteForProfile(routeProfile)
	}
	if !featureRoute.Matched() && hasModelRouteSet && modelRouteSet.FeatureRoute.Matched() {
		featureRoute = rocmModelFeatureRouteFromModel(modelRouteSet.FeatureRoute)
	}
	routeProfile.FeatureRoute = featureRoute

	tokenizerRoute := routeProfile.TokenizerRoute
	if !tokenizerRoute.Matched() {
		tokenizerRoute = ROCmModelTokenizerRouteForProfile(routeProfile)
	}
	if !tokenizerRoute.Matched() && hasModelRouteSet && modelRouteSet.TokenizerRoute.Matched() {
		tokenizerRoute = rocmModelTokenizerRouteFromModel(modelRouteSet.TokenizerRoute)
	}
	routeProfile.TokenizerRoute = tokenizerRoute

	loraRoute := routeProfile.LoRAAdapterRoute
	if !loraRoute.Matched() {
		loraRoute = ROCmLoRAAdapterRouteForProfile(routeProfile)
	}
	if !loraRoute.Matched() && hasModelRouteSet && modelRouteSet.LoRAAdapterRoute.Matched() {
		loraRoute = rocmLoRAAdapterRouteFromModel(modelRouteSet.LoRAAdapterRoute)
	}
	routeProfile.LoRAAdapterRoute = loraRoute

	multimodalRoute := routeProfile.MultimodalProcessorRoute
	if !multimodalRoute.Matched() {
		multimodalRoute = ROCmMultimodalProcessorRouteForProfile(routeProfile)
	}
	if !multimodalRoute.Matched() && hasModelRouteSet && modelRouteSet.MultimodalProcessorRoute.Matched() {
		multimodalRoute = rocmMultimodalProcessorRouteFromModel(modelRouteSet.MultimodalProcessorRoute)
	}
	routeProfile.MultimodalProcessorRoute = multimodalRoute

	diffusionRoute := routeProfile.DiffusionSamplerRoute
	if !diffusionRoute.Matched() {
		diffusionRoute = ROCmDiffusionSamplerRouteForProfile(routeProfile)
	}
	if !diffusionRoute.Matched() && hasModelRouteSet && modelRouteSet.DiffusionSamplerRoute.Matched() {
		diffusionRoute = rocmDiffusionSamplerRouteFromModel(modelRouteSet.DiffusionSamplerRoute)
	}
	routeProfile.DiffusionSamplerRoute = diffusionRoute

	stateRoute := routeProfile.StateContextRoute
	if !stateRoute.Matched() {
		stateRoute = ROCmStateContextRouteForProfile(routeProfile)
	}
	if !stateRoute.Matched() && hasModelRouteSet && modelRouteSet.StateContextRoute.Matched() {
		stateRoute = rocmStateContextRouteFromModel(modelRouteSet.StateContextRoute)
	}
	routeProfile.StateContextRoute = stateRoute

	drafterRoute := routeProfile.AttachedDrafterRoute
	if !drafterRoute.Matched() {
		drafterRoute = ROCmAttachedDrafterRouteForProfile(routeProfile)
	}
	if !drafterRoute.Matched() && hasModelRouteSet && modelRouteSet.AttachedDrafterRoute.Matched() {
		drafterRoute = rocmAttachedDrafterRouteFromModel(modelRouteSet.AttachedDrafterRoute)
	}
	routeProfile.AttachedDrafterRoute = drafterRoute

	loadStatus := routeProfile.LoadStatus
	if loadStatus.empty() {
		loadStatus = ROCmModelLoadStatusForProfile(routeProfile)
	}
	routeProfile.LoadStatus = loadStatus

	cacheRoute := routeProfile.CacheRoute
	if !cacheRoute.Matched() && hasModelRouteSet && modelRouteSet.CacheRoute.Matched() {
		cacheRoute = modelRouteSet.CacheRoute
	}

	loaderRoute := ROCmModelLoaderRoute{}
	if !loaderRoute.Matched() {
		loaderRoute = ROCmModelLoaderRouteForProfile(routeProfile)
	}
	if !loaderRoute.Matched() && hasModelRouteSet && modelRouteSet.LoaderRoute.Matched() {
		loaderRoute = rocmModelLoaderRouteFromModel(modelRouteSet.LoaderRoute)
	}

	quantRoute := routeProfile.QuantLoaderRoute
	if !quantRoute.Matched() {
		if route, ok := ROCmQuantLoaderRouteForProfile(routeProfile); ok {
			quantRoute = route
		}
	}
	if !quantRoute.Matched() && hasModelRouteSet && modelRouteSet.QuantLoaderRoute.Matched() {
		quantRoute = rocmQuantLoaderRouteFromModel(modelRouteSet.QuantLoaderRoute)
	}
	sequenceMixerRoutes := cloneROCmSequenceMixerLoaderRoutes(routeProfile.SequenceMixerRoutes)
	if len(sequenceMixerRoutes) == 0 && hasModelRouteSet && len(modelRouteSet.SequenceMixerRoutes) > 0 {
		sequenceMixerRoutes = rocmSequenceMixerLoaderRoutesFromModel(modelRouteSet.SequenceMixerRoutes)
	}
	runtimeContractRoute := routeProfile.RuntimeContractRoute
	if !runtimeContractRoute.Matched() && hasModelRouteSet && modelRouteSet.RuntimeContractRoute.Matched() {
		runtimeContractRoute = modelRouteSet.RuntimeContractRoute.Clone()
	}
	if !runtimeContractRoute.Matched() {
		if route, ok := ROCmModelRuntimeContractRouteForIdentity(routeProfile.Model.Path, routeProfile.Model); ok {
			runtimeContractRoute = route
		}
	}
	runtimeGatePlan := rocmmodel.RuntimeGatePlan{}
	if hasModelRouteSet && modelRouteSet.RuntimeGatePlan.Matched() {
		runtimeGatePlan = modelRouteSet.RuntimeGatePlan
	}
	runtimeAuthorPlan := rocmmodel.RuntimeAuthorPlan{}
	if hasModelRouteSet && modelRouteSet.RuntimeAuthorPlan.Matched() {
		runtimeAuthorPlan = modelRouteSet.RuntimeAuthorPlan
	}

	plan := ROCmModelRoutePlan{
		Contract:                 ROCmModelRoutePlanContract,
		Architecture:             firstNonEmptyString(features.Architecture, routeProfile.Architecture, routeProfile.Model.Architecture, featureRoute.Architecture),
		Family:                   firstNonEmptyString(features.Family, routeProfile.Family, featureRoute.Family, routeProfile.Name),
		Model:                    rocmCloneModelIdentity(routeProfile.Model),
		EngineFeatures:           features,
		FeatureRoute:             featureRoute,
		TokenizerRoute:           tokenizerRoute,
		LoRAAdapterRoute:         loraRoute,
		MultimodalProcessorRoute: multimodalRoute,
		DiffusionSamplerRoute:    diffusionRoute,
		StateContextRoute:        stateRoute,
		AttachedDrafterRoute:     drafterRoute,
		LoadStatus:               loadStatus,
		CacheRoute:               cacheRoute,
		LoaderRoute:              loaderRoute,
		QuantLoaderRoute:         quantRoute,
		SequenceMixerRoutes:      sequenceMixerRoutes,
		RuntimeContractRoute:     runtimeContractRoute,
		RuntimeGatePlan:          runtimeGatePlan,
		RuntimeAuthorPlan:        runtimeAuthorPlan,
	}
	plan.Labels = rocmModelRoutePlanLabels(plan)
	return plan.clone()
}

// ApplyROCmModelRoutePlanLabels returns labels plus the compact route-plan
// labels for plan without mutating the caller's input map.
func ApplyROCmModelRoutePlanLabels(labels map[string]string, plan ROCmModelRoutePlan) map[string]string {
	labels = cloneStringMap(labels)
	if !plan.Matched() {
		return labels
	}
	if labels == nil {
		labels = map[string]string{}
	}
	for key, value := range plan.Labels {
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}

func rocmModelRoutePlanLabels(plan ROCmModelRoutePlan) map[string]string {
	if !plan.Matched() {
		return nil
	}
	labels := cloneStringMap(plan.Labels)
	if labels == nil {
		labels = map[string]string{}
	}
	for key, value := range map[string]string{
		"engine_route_plan_contract":         plan.Contract,
		"engine_route_plan_architecture":     plan.Architecture,
		"engine_route_plan_feature":          strconv.FormatBool(plan.FeatureRoute.Matched()),
		"engine_route_plan_tokenizer":        strconv.FormatBool(plan.TokenizerRoute.Matched()),
		"engine_route_plan_lora_adapter":     strconv.FormatBool(plan.LoRAAdapterRoute.Matched()),
		"engine_route_plan_multimodal":       strconv.FormatBool(plan.MultimodalProcessorRoute.Matched()),
		"engine_route_plan_diffusion":        strconv.FormatBool(plan.DiffusionSamplerRoute.Matched()),
		"engine_route_plan_state_context":    strconv.FormatBool(plan.StateContextRoute.Matched()),
		"engine_route_plan_drafter":          strconv.FormatBool(plan.AttachedDrafterRoute.Matched()),
		"engine_route_plan_cache":            strconv.FormatBool(plan.CacheRoute.Matched()),
		"engine_route_plan_cache_profile":    strconv.FormatBool(plan.CacheProfile.Matched()),
		"engine_route_plan_loader":           strconv.FormatBool(plan.LoaderRoute.Matched()),
		"engine_route_plan_quant_loader":     strconv.FormatBool(plan.QuantLoaderRoute.Matched()),
		"engine_route_plan_sequence_mixer":   strconv.FormatBool(len(plan.SequenceMixerRoutes) > 0),
		"engine_route_plan_runtime_contract": strconv.FormatBool(plan.RuntimeContractRoute.Matched()),
		"engine_route_plan_runtime_gate":     strconv.FormatBool(plan.RuntimeGatePlan.Matched()),
		"engine_route_plan_runtime_author":   strconv.FormatBool(plan.RuntimeAuthorPlan.Matched()),
		"engine_route_plan_text_generate":    strconv.FormatBool(plan.EngineFeatures.TextGenerate),
		"engine_route_plan_native_runtime":   strconv.FormatBool(plan.EngineFeatures.NativeRuntime),
	} {
		if value != "" {
			labels[key] = value
		}
	}
	if plan.Family != "" {
		labels["engine_route_plan_family"] = plan.Family
	}
	if plan.LoadStatus.Status != "" {
		labels["engine_route_plan_load_status"] = string(plan.LoadStatus.Status)
	}
	rocmApplyModelRoutePlanLoadLabels(labels, plan.LoadStatus)
	rocmApplyModelRoutePlanCacheLabels(labels, plan.CacheRoute)
	rocmApplyModelRoutePlanCacheProfileLabels(labels, plan.CacheProfile)
	rocmApplyModelRoutePlanLoaderLabels(labels, plan.LoaderRoute)
	rocmApplyModelRoutePlanQuantLabels(labels, plan.QuantLoaderRoute)
	rocmApplyModelRoutePlanSequenceMixerLabels(labels, plan.SequenceMixerRoutes)
	rocmApplyModelRoutePlanRuntimeContractLabels(labels, plan.RuntimeContractRoute)
	rocmApplyModelRoutePlanRuntimeGateLabels(labels, plan.RuntimeGatePlan)
	rocmApplyModelRoutePlanRuntimeAuthorLabels(labels, plan.RuntimeAuthorPlan)
	rocmApplyModelRoutePlanStateLabels(labels, plan.StateContextRoute)
	rocmApplyModelRoutePlanDrafterLabels(labels, plan.AttachedDrafterRoute)
	return labels
}

func rocmApplyModelRoutePlanCacheProfileLabels(labels map[string]string, profile rocmmodel.CacheProfile) {
	if !profile.Matched() {
		return
	}
	rocmmodel.ApplyCacheProfileLabels(labels, profile)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_cache_profile_contract", profile.Contract)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_cache_profile_architecture", profile.Architecture)
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_cache_profile_total", profile.TotalCaches)
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_cache_profile_local_count", profile.LocalCaches)
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_cache_profile_global_count", profile.GlobalCaches)
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_cache_profile_shared_layers", profile.SharedLayers)
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_cache_profile_cacheless_layers", profile.CachelessLayers)
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_cache_profile_local_window_tokens", profile.LocalWindowTokens)
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_cache_profile_max_cache_tokens", profile.MaxCacheTokens)
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_cache_profile_max_cache_capacity", profile.MaxCacheCapacity)
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_cache_profile_paged_count", profile.PagedCaches)
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_cache_profile_quantized_count", profile.QuantizedCaches)
	labels["engine_route_plan_cache_profile_local_window_leaked"] = strconv.FormatBool(profile.LocalWindowLeaked)
}

func rocmApplyModelRoutePlanSequenceMixerLabels(labels map[string]string, routes []ROCmSequenceMixerLoaderRoute) {
	routes = cloneROCmSequenceMixerLoaderRoutes(routes)
	if len(routes) == 0 {
		return
	}
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_sequence_mixer_count", len(routes))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_sequence_mixer_kinds", rocmSequenceMixerRouteKindsCSV(routes))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_sequence_mixer_cache_modes", rocmSequenceMixerRouteCacheModesCSV(routes))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_sequence_mixer_states", rocmSequenceMixerRouteStatesCSV(routes))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_sequence_mixer_runtimes", rocmSequenceMixerRouteRuntimesCSV(routes))
	labels["engine_route_plan_sequence_mixer_native_runtime"] = strconv.FormatBool(rocmSequenceMixerRoutesAnyNativeRuntime(routes))
	labels["engine_route_plan_sequence_mixer_planned"] = strconv.FormatBool(rocmSequenceMixerRoutesAnyPlanned(routes))
	if len(routes) == 1 {
		for key, value := range routes[0].Labels {
			if value != "" {
				labels[key] = value
			}
		}
	}
}

func rocmApplyModelRoutePlanRuntimeContractLabels(labels map[string]string, route ROCmModelRuntimeContractRoute) {
	if !route.Matched() {
		return
	}
	rocmApplyROCmModelRuntimeContractRouteLabels(labels, route)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_contract_contract", route.Contract)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_contract_route", route.Name)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_contract_architecture", route.Architecture)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_contract_family", route.Family)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_contract_runtime_status", string(route.RuntimeStatus))
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_runtime_contract_count", len(route.ContractIDs))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_contract_ids", rocmModelRuntimeContractIDsCSV(route.ContractIDs))
	labels["engine_route_plan_runtime_contract_registered"] = strconv.FormatBool(route.Registered)
	labels["engine_route_plan_runtime_contract_native_runtime"] = strconv.FormatBool(route.NativeRuntime)
	labels["engine_route_plan_runtime_contract_metadata_only"] = strconv.FormatBool(route.MetadataOnly)
	labels["engine_route_plan_runtime_contract_text_generate"] = strconv.FormatBool(route.TextGenerate)
	labels["engine_route_plan_runtime_contract_cache_topology"] = strconv.FormatBool(route.CacheTopology)
	labels["engine_route_plan_runtime_contract_fixed_sliding_cache"] = strconv.FormatBool(route.FixedSlidingCache)
	labels["engine_route_plan_runtime_contract_go_mlx_optional_interface_compatible"] = strconv.FormatBool(len(route.ContractIDs) > 0)
}

func rocmApplyModelRoutePlanRuntimeAuthorLabels(labels map[string]string, plan rocmmodel.RuntimeAuthorPlan) {
	if !plan.Matched() {
		return
	}
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_author_contract", plan.Contract)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_author_architecture", plan.Architecture)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_author_family", plan.Family)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_author_runtime", plan.Runtime)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_author_runtime_status", string(plan.RuntimeStatus))
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_runtime_author_count", len(plan.CapabilityIDs))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_author_ids", rocmRuntimeAuthorCapabilityIDsCSV(plan.CapabilityIDs))
	labels["engine_route_plan_runtime_author_native_runtime"] = strconv.FormatBool(plan.NativeRuntime)
	labels["engine_route_plan_runtime_author_text_runtime"] = strconv.FormatBool(plan.TextRuntime)
	labels["engine_route_plan_runtime_author_prompt_cache"] = strconv.FormatBool(plan.PromptCache)
	labels["engine_route_plan_runtime_author_cache_profile"] = strconv.FormatBool(plan.CacheProfile)
	for key, value := range plan.Labels {
		if value != "" {
			labels[key] = value
		}
	}
}

func rocmApplyModelRoutePlanRuntimeGateLabels(labels map[string]string, plan rocmmodel.RuntimeGatePlan) {
	if !plan.Matched() {
		return
	}
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_gate_contract", plan.Contract)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_gate_architecture", plan.Architecture)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_gate_family", plan.Family)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_gate_runtime_status", string(plan.RuntimeStatus))
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_runtime_gate_count", len(plan.GateIDs))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_runtime_gate_ids", rocmRuntimeGateIDsCSV(plan.GateIDs))
	for key, value := range plan.Labels {
		if value != "" {
			labels[key] = value
		}
	}
}

func rocmApplyModelRoutePlanCacheLabels(labels map[string]string, route rocmmodel.CacheRoute) {
	if !route.Matched() {
		return
	}
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_cache_contract", route.Contract)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_cache_route", route.Name)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_cache_architecture", route.Architecture)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_cache_family", route.Family)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_cache_runtime_status", string(route.RuntimeStatus))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_cache_default_mode", route.DefaultMode)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_cache_recommended_mode", route.RecommendedMode)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_cache_device_mode", route.DeviceMode)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_cache_modes", strings.Join(route.ModeNames, ","))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_cache_hints", strings.Join(route.CacheHints, ","))
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_cache_mode_count", len(route.ModeNames))
	labels["engine_route_plan_cache_registered"] = strconv.FormatBool(route.Registered)
	labels["engine_route_plan_cache_native_runtime"] = strconv.FormatBool(route.NativeRuntime)
	labels["engine_route_plan_cache_supports_kv"] = strconv.FormatBool(route.SupportsKV)
	labels["engine_route_plan_cache_supports_device"] = strconv.FormatBool(route.SupportsDevice)
	labels["engine_route_plan_cache_supports_recurrent"] = strconv.FormatBool(route.SupportsRecurrent)
	for key, value := range route.Labels {
		if value != "" {
			labels[key] = value
		}
	}
}

func rocmRuntimeGateIDsCSV(ids []rocmmodel.RuntimeGateID) string {
	parts := make([]string, 0, len(ids))
	for _, id := range ids {
		if id != "" {
			parts = append(parts, string(id))
		}
	}
	return strings.Join(parts, ",")
}

func rocmRuntimeAuthorCapabilityIDsCSV(ids []rocmmodel.RuntimeAuthorCapabilityID) string {
	parts := make([]string, 0, len(ids))
	for _, id := range ids {
		if id != "" {
			parts = append(parts, string(id))
		}
	}
	return strings.Join(parts, ",")
}

func rocmSequenceMixerRouteKindsCSV(routes []ROCmSequenceMixerLoaderRoute) string {
	return strings.Join(rocmSequenceMixerRouteStrings(routes, func(route ROCmSequenceMixerLoaderRoute) string {
		return route.Kind
	}), ",")
}

func rocmSequenceMixerRouteCacheModesCSV(routes []ROCmSequenceMixerLoaderRoute) string {
	return strings.Join(rocmSequenceMixerRouteStrings(routes, func(route ROCmSequenceMixerLoaderRoute) string {
		return route.CacheMode
	}), ",")
}

func rocmSequenceMixerRouteStatesCSV(routes []ROCmSequenceMixerLoaderRoute) string {
	return strings.Join(rocmSequenceMixerRouteStrings(routes, func(route ROCmSequenceMixerLoaderRoute) string {
		return route.State
	}), ",")
}

func rocmSequenceMixerRouteRuntimesCSV(routes []ROCmSequenceMixerLoaderRoute) string {
	return strings.Join(rocmSequenceMixerRouteStrings(routes, func(route ROCmSequenceMixerLoaderRoute) string {
		return route.Runtime
	}), ",")
}

func rocmSequenceMixerRouteStrings(routes []ROCmSequenceMixerLoaderRoute, value func(ROCmSequenceMixerLoaderRoute) string) []string {
	parts := make([]string, 0, len(routes))
	seen := map[string]bool{}
	for _, route := range routes {
		if !route.Matched() {
			continue
		}
		part := strings.TrimSpace(value(route))
		if part == "" || seen[part] {
			continue
		}
		seen[part] = true
		parts = append(parts, part)
	}
	return parts
}

func rocmSequenceMixerRoutesAnyNativeRuntime(routes []ROCmSequenceMixerLoaderRoute) bool {
	for _, route := range routes {
		if route.NativeRuntime {
			return true
		}
	}
	return false
}

func rocmSequenceMixerRoutesAnyPlanned(routes []ROCmSequenceMixerLoaderRoute) bool {
	for _, route := range routes {
		if route.Planned {
			return true
		}
	}
	return false
}

func rocmApplyModelRoutePlanLoadLabels(labels map[string]string, status ROCmModelLoadStatus) {
	if status.empty() {
		return
	}
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_load_contract", status.Contract)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_load_target", status.Target)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_load_runtime_status", string(status.RuntimeStatus))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_load_reason", status.Reason)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_loader_name", status.Loader)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_loader_runtime", status.LoaderRuntime)
	labels["engine_route_plan_load_native_runtime"] = strconv.FormatBool(status.NativeRuntime)
	labels["engine_route_plan_load_standalone"] = strconv.FormatBool(status.Standalone)
	labels["engine_route_plan_load_attached_only"] = strconv.FormatBool(status.AttachedOnly)
	labels["engine_route_plan_load_staged"] = strconv.FormatBool(status.Staged)
	labels["engine_route_plan_load_metadata_only"] = strconv.FormatBool(status.MetadataOnly)
	labels["engine_route_plan_load_text_generate"] = strconv.FormatBool(status.TextGenerate)
}

func rocmApplyModelRoutePlanLoaderLabels(labels map[string]string, route ROCmModelLoaderRoute) {
	if !route.Matched() {
		return
	}
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_loader_contract", route.Contract)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_loader_route", route.Name)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_loader_architecture", route.Architecture)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_loader_family", route.Family)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_loader_name", route.Loader)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_loader_runtime", route.Runtime)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_loader_status", string(route.Status))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_loader_target", route.Target)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_loader_runtime_status", string(route.RuntimeStatus))
	labels["engine_route_plan_loader_registered"] = strconv.FormatBool(route.Registered)
	labels["engine_route_plan_loader_native_runtime"] = strconv.FormatBool(route.NativeRuntime)
	labels["engine_route_plan_loader_standalone"] = strconv.FormatBool(route.Standalone)
	labels["engine_route_plan_loader_attached_only"] = strconv.FormatBool(route.AttachedOnly)
	labels["engine_route_plan_loader_staged"] = strconv.FormatBool(route.Staged)
	labels["engine_route_plan_loader_metadata_only"] = strconv.FormatBool(route.MetadataOnly)
	labels["engine_route_plan_loader_text_generate"] = strconv.FormatBool(route.TextGenerate)
}

func rocmApplyModelRoutePlanQuantLabels(labels map[string]string, route ROCmQuantLoaderRoute) {
	if !route.Matched() {
		return
	}
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_quant_contract", route.Contract)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_quant_route", route.Name)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_quant_family", route.Family)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_quant_architecture", route.Architecture)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_quant_size", route.Size)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_quant_pack", route.Pack)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_quant_pack_name", route.PackName)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_quant_model_id", route.ModelID)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_quant_locked_model_id", route.LockedModelID)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_quant_mode", route.Mode)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_quant_product_role", route.ProductRole)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_quant_loader_name", route.Loader)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_quant_runtime", route.Runtime)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_quant_generate_status", route.GenerateStatus)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_quant_target", route.Target)
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_quant_bits", route.Bits)
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_quant_group", route.Group)
	labels["engine_route_plan_quant_registered"] = strconv.FormatBool(route.Registered)
	labels["engine_route_plan_quant_native_runtime"] = strconv.FormatBool(route.NativeRuntime)
	labels["engine_route_plan_quant_runnable_on_card"] = strconv.FormatBool(route.RunnableOnCard)
	labels["engine_route_plan_quant_staged"] = strconv.FormatBool(route.Staged)
	labels["engine_route_plan_quant_load_only"] = strconv.FormatBool(route.LoadOnly)
	labels["engine_route_plan_quant_planned"] = strconv.FormatBool(route.Planned)
	labels["engine_route_plan_quant_requires_bench"] = strconv.FormatBool(route.RequiresBench)
	labels["engine_route_plan_quant_requires_native"] = strconv.FormatBool(route.RequiresNative)
}

func rocmApplyModelRoutePlanStateLabels(labels map[string]string, route ROCmStateContextRoute) {
	if !route.Matched() {
		return
	}
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_state_context_contract", route.Contract)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_state_context_route", route.Name)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_state_context_reference", route.Reference)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_state_context_runtime", route.Runtime)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_state_context_runtime_status", string(route.RuntimeStatus))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_state_context_status", string(route.Status))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_state_context_device_kv_mode", route.DefaultDeviceKVMode)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_state_context_cache_modes", joinNonEmptyStrings(route.CacheModes, ","))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_state_context_backends", joinNonEmptyStrings(route.StateBackends, ","))
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_state_context_window", route.ContextWindow)
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_state_context_default_window", route.DefaultContextWindow)
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_state_context_block_size", route.DefaultStateBlockSize)
	labels["engine_route_plan_state_context_registered"] = strconv.FormatBool(route.Registered)
	labels["engine_route_plan_state_context_native_runtime"] = strconv.FormatBool(route.NativeRuntime)
	labels["engine_route_plan_state_context_attached_only"] = strconv.FormatBool(route.AttachedOnly)
	labels["engine_route_plan_state_context_runtime_owned_kv"] = strconv.FormatBool(route.RuntimeOwnedKV)
	labels["engine_route_plan_state_context_prompt_replay_refused"] = strconv.FormatBool(route.PromptReplayRefused)
	labels["engine_route_plan_state_context_remaining_default"] = strconv.FormatBool(route.RemainingContextDefault)
	labels["engine_route_plan_state_context_retained_state_required"] = strconv.FormatBool(route.RetainedStateRequired)
	labels["engine_route_plan_state_context_attached_drafter_state"] = strconv.FormatBool(route.AttachedDrafterState)
}

func rocmApplyModelRoutePlanDrafterLabels(labels map[string]string, route ROCmAttachedDrafterRoute) {
	if !route.Matched() {
		return
	}
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_contract", route.Contract)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_route", route.Name)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_reference", route.Reference)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_mode", route.Mode)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_role", route.Role)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_runtime", route.Runtime)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_runtime_status", string(route.RuntimeStatus))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_status", string(route.Status))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_target_architecture", route.TargetArchitecture)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_assistant_architecture", route.AssistantArchitecture)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_target_runtime", route.TargetRuntime)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_assistant_runtime", route.AssistantRuntime)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_target_generate_status", route.TargetGenerateStatus)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_assistant_generate_status", route.AssistantGenerateStatus)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_native_attachment", route.NativeAttachment)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_execution_status", route.ExecutionStatus)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_fallback", route.Fallback)
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_assistant_models", joinNonEmptyStrings(route.AssistantModelIDs, ","))
	rocmSetModelRoutePlanLabel(labels, "engine_route_plan_drafter_detection_sources", joinNonEmptyStrings(route.DetectionSources, ","))
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_drafter_default_tokens", route.DefaultDraftTokens)
	rocmSetModelRoutePlanIntLabel(labels, "engine_route_plan_drafter_default_block", route.DefaultDraftBlock)
	labels["engine_route_plan_drafter_registered"] = strconv.FormatBool(route.Registered)
	labels["engine_route_plan_drafter_native_runtime"] = strconv.FormatBool(route.NativeRuntime)
	labels["engine_route_plan_drafter_target"] = strconv.FormatBool(route.Target)
	labels["engine_route_plan_drafter_assistant"] = strconv.FormatBool(route.Assistant)
	labels["engine_route_plan_drafter_attached_only"] = strconv.FormatBool(route.AttachedOnly)
	labels["engine_route_plan_drafter_retained_state_required"] = strconv.FormatBool(route.RetainedStateRequired)
	labels["engine_route_plan_drafter_runtime_owned_kv"] = strconv.FormatBool(route.RuntimeOwnedKV)
	labels["engine_route_plan_drafter_prompt_replay_refused"] = strconv.FormatBool(route.PromptReplayRefused)
	labels["engine_route_plan_drafter_fallback_refused"] = strconv.FormatBool(route.FallbackRefused)
	labels["engine_route_plan_drafter_staged"] = strconv.FormatBool(route.Staged)
	labels["engine_route_plan_drafter_planned"] = strconv.FormatBool(route.Planned)
}

func rocmSetModelRoutePlanLabel(labels map[string]string, key, value string) {
	if value != "" {
		labels[key] = value
	}
}

func rocmSetModelRoutePlanIntLabel(labels map[string]string, key string, value int) {
	if value > 0 {
		labels[key] = strconv.Itoa(value)
	}
}
