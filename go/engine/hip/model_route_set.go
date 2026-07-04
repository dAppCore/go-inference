// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const ROCmModelRouteSetContract = rocmmodel.RouteSetContract

type ROCmModelRouteSet = rocmmodel.RouteSet
type ROCmModelRouteSetOptions = rocmmodel.RouteSetOptions

// ROCmModelRouteSetForIdentity returns the folder-owned route-set contract for
// identity using ROCm's production quant-loader matrix.
func ROCmModelRouteSetForIdentity(path string, identity inference.ModelIdentity) (ROCmModelRouteSet, bool) {
	return ROCmModelRouteSetForIdentityWithOptions(path, identity, defaultROCmModelRouteSetOptions())
}

// ROCmModelRouteSetForIdentityWithOptions returns the folder-owned route-set
// contract for identity using caller-provided route-set options.
func ROCmModelRouteSetForIdentityWithOptions(path string, identity inference.ModelIdentity, opts ROCmModelRouteSetOptions) (ROCmModelRouteSet, bool) {
	return rocmmodel.RouteSetForIdentityWithOptions(path, identity, opts)
}

// ROCmModelRouteSetForInfo adapts the small go-inference ModelInfo shape into
// the route-set resolver using ROCm's production quant-loader matrix.
func ROCmModelRouteSetForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmModelRouteSet, bool) {
	return rocmmodel.RouteSetForInfo(path, info, cloneStringMap(labels), defaultROCmModelRouteSetOptions())
}

// ROCmModelRouteSetForInspection resolves a route set from an inspected model
// pack, preserving inspection labels and production quant-loader defaults.
func ROCmModelRouteSetForInspection(inspection *inference.ModelPackInspection) (ROCmModelRouteSet, bool) {
	return rocmmodel.RouteSetForInspection(inspection, defaultROCmModelRouteSetOptions())
}

// ROCmModelRouteSetForProfile resolves a route set from an already-resolved
// ROCm model profile.
func ROCmModelRouteSetForProfile(profile ROCmModelProfile) (ROCmModelRouteSet, bool) {
	return rocmModelRouteSetForProfile(profile)
}

// ApplyROCmModelRouteSetLabels returns labels plus route-set labels without
// mutating the caller's input map.
func ApplyROCmModelRouteSetLabels(labels map[string]string, set ROCmModelRouteSet) map[string]string {
	labels = cloneStringMap(labels)
	if !set.Matched() {
		return labels
	}
	if labels == nil {
		labels = map[string]string{}
	}
	for key, value := range set.Labels {
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}

func defaultROCmModelRouteSetOptions() ROCmModelRouteSetOptions {
	return ROCmModelRouteSetOptions{
		QuantLoaderPacks: rocmQuantLoaderPacksToModel(DefaultProductionQuantizationPackSupport()),
	}
}

func rocmModelRouteSetForProfile(profile ROCmModelProfile) (rocmmodel.RouteSet, bool) {
	model := rocmCloneModelIdentity(profile.Model)
	if model.Path == "" {
		model.Path = profile.Model.Path
	}
	if model.Architecture == "" {
		model.Architecture = firstNonEmptyString(profile.Architecture, profile.ArchitectureProfile.ID, profile.Gemma4Settings.ID)
	}
	labels := cloneStringMap(model.Labels)
	if labels == nil {
		labels = map[string]string{}
	}
	for key, value := range profile.Labels {
		if labels[key] == "" && value != "" {
			labels[key] = value
		}
	}
	model.Labels = labels
	return rocmmodel.RouteSetForIdentityWithOptions(model.Path, model, defaultROCmModelRouteSetOptions())
}

func rocmApplyModelRouteSetDefaults(profile ROCmModelProfile) ROCmModelProfile {
	routeSet, ok := rocmModelRouteSetForProfile(profile)
	if !ok {
		return profile
	}
	if !profile.FeatureRoute.Matched() && routeSet.FeatureRoute.Matched() {
		profile.FeatureRoute = rocmModelFeatureRouteFromModel(routeSet.FeatureRoute)
	}
	if !profile.TokenizerRoute.Matched() && routeSet.TokenizerRoute.Matched() {
		profile.TokenizerRoute = rocmModelTokenizerRouteFromModel(routeSet.TokenizerRoute)
	}
	if !profile.LoRAAdapterRoute.Matched() && routeSet.LoRAAdapterRoute.Matched() {
		profile.LoRAAdapterRoute = rocmLoRAAdapterRouteFromModel(routeSet.LoRAAdapterRoute)
	}
	if !profile.MultimodalProcessorRoute.Matched() && routeSet.MultimodalProcessorRoute.Matched() {
		profile.MultimodalProcessorRoute = rocmMultimodalProcessorRouteFromModel(routeSet.MultimodalProcessorRoute)
	}
	if !profile.DiffusionSamplerRoute.Matched() && routeSet.DiffusionSamplerRoute.Matched() {
		profile.DiffusionSamplerRoute = rocmDiffusionSamplerRouteFromModel(routeSet.DiffusionSamplerRoute)
	}
	if !profile.StateContextRoute.Matched() && routeSet.StateContextRoute.Matched() {
		profile.StateContextRoute = rocmStateContextRouteFromModel(routeSet.StateContextRoute)
	}
	if !profile.AttachedDrafterRoute.Matched() && routeSet.AttachedDrafterRoute.Matched() {
		profile.AttachedDrafterRoute = rocmAttachedDrafterRouteFromModel(routeSet.AttachedDrafterRoute)
	}
	if !profile.CacheRoute.Matched() && routeSet.CacheRoute.Matched() {
		profile.CacheRoute = routeSet.CacheRoute.Clone()
	}
	if !profile.QuantLoaderRoute.Matched() && routeSet.QuantLoaderRoute.Matched() {
		profile.QuantLoaderRoute = rocmQuantLoaderRouteFromModel(routeSet.QuantLoaderRoute)
	}
	if len(profile.SequenceMixerRoutes) == 0 && len(routeSet.SequenceMixerRoutes) > 0 {
		profile.SequenceMixerRoutes = rocmSequenceMixerLoaderRoutesFromModel(routeSet.SequenceMixerRoutes)
	}
	if !profile.RuntimeContractRoute.Matched() && routeSet.RuntimeContractRoute.Matched() {
		profile.RuntimeContractRoute = routeSet.RuntimeContractRoute.Clone()
	}
	profile.Labels = mergeStringMaps(profile.Labels, routeSet.Labels)
	return profile
}
