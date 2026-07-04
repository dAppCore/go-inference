// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const (
	ROCmDiffusionSamplerRegistryContract = rocmmodel.DiffusionSamplerRegistryContract

	rocmDiffusionSamplerRegistryRouteName = rocmmodel.DiffusionSamplerRouteName
	rocmDiffusionSamplerRuntimeHIP        = rocmmodel.DiffusionSamplerRuntimeHIP
	rocmDiffusionSamplerRuntimeMetadata   = rocmmodel.DiffusionSamplerRuntimeMetadata
)

type ROCmDiffusionSamplerRouteStatus = rocmmodel.DiffusionSamplerRouteStatus

const (
	ROCmDiffusionSamplerExperimentalNative = rocmmodel.DiffusionSamplerExperimentalNative
	ROCmDiffusionSamplerPlannedMetadata    = rocmmodel.DiffusionSamplerPlannedMetadata
)

// ROCmDiffusionSamplerRoute is the model-owned block-diffusion route exposed
// through the registry. It mirrors go-mlx's DiffusionGemma sampler contract
// while keeping ROCm execution explicitly not-linked until the denoising
// sampler/runtime is implemented.
type ROCmDiffusionSamplerRoute = rocmmodel.DiffusionSamplerRoute

// RegisterROCmDiffusionSamplerRoute registers or replaces an
// architecture-keyed block-diffusion sampler route. It mirrors go-mlx's
// capability-owned diffusion contract at the ROCm API layer, so new families
// can advertise sampler metadata without another central switch.
func RegisterROCmDiffusionSamplerRoute(route ROCmDiffusionSamplerRoute) {
	route = normalizeRegisteredROCmDiffusionSamplerRoute(route)
	if !route.Matched() {
		return
	}
	rocmmodel.RegisterDiffusionSamplerRoute(route)
}

// RegisteredROCmDiffusionSamplerRouteArchitectures returns extension
// diffusion-sampler architectures in registration order. Built-in routes are
// intentionally not included.
func RegisteredROCmDiffusionSamplerRouteArchitectures() []string {
	return rocmmodel.RegisteredDiffusionSamplerArchitectures()
}

func normalizeRegisteredROCmDiffusionSamplerRoute(route ROCmDiffusionSamplerRoute) ROCmDiffusionSamplerRoute {
	return rocmmodel.NormalizeDiffusionSamplerRoute(route).Clone()
}

func DefaultROCmDiffusionSamplerRoutes() []ROCmDiffusionSamplerRoute {
	modelRoutes := rocmmodel.DefaultDiffusionSamplerRoutes()
	routes := make([]ROCmDiffusionSamplerRoute, 0, len(modelRoutes))
	for _, modelRoute := range modelRoutes {
		route := rocmDiffusionSamplerRouteFromModel(modelRoute)
		if route.Matched() {
			routes = append(routes, route)
		}
	}
	return routes
}

func ROCmDiffusionSamplerRouteForArchitecture(architecture string) (ROCmDiffusionSamplerRoute, bool) {
	modelRoute, ok := rocmmodel.DiffusionSamplerRouteForArchitecture(architecture)
	if !ok {
		return ROCmDiffusionSamplerRoute{}, false
	}
	route := rocmDiffusionSamplerRouteFromModel(modelRoute)
	if !route.Matched() {
		return ROCmDiffusionSamplerRoute{}, false
	}
	return route, true
}

func ROCmDiffusionSamplerRouteForProfile(profile ROCmModelProfile) ROCmDiffusionSamplerRoute {
	labels := cloneStringMap(profile.Model.Labels)
	model := rocmCloneModelIdentity(profile.Model)
	model.Labels = labels
	if model.Architecture == "" {
		model.Architecture = firstNonEmptyString(profile.Architecture, profile.ArchitectureProfile.ID, profile.Gemma4Settings.ID)
	}
	modelRoute, ok := rocmmodel.DiffusionSamplerRouteForIdentity(model.Path, model)
	if !ok {
		return ROCmDiffusionSamplerRoute{}
	}
	route := rocmDiffusionSamplerRouteFromModel(modelRoute)
	if !route.Matched() {
		return ROCmDiffusionSamplerRoute{}
	}
	return route.Clone()
}

func rocmDiffusionSamplerRouteFromModel(route rocmmodel.DiffusionSamplerRoute) ROCmDiffusionSamplerRoute {
	if route.Labels == nil {
		route.Labels = rocmmodel.DiffusionSamplerRouteLabels(route)
	}
	return route.Clone()
}

func ROCmDiffusionSamplerRouteForIdentity(path string, model inference.ModelIdentity) (ROCmDiffusionSamplerRoute, bool) {
	profile, ok := ResolveROCmModelProfile(path, model)
	if !ok {
		return ROCmDiffusionSamplerRoute{}, false
	}
	route := profile.DiffusionSamplerRoute
	if !route.Matched() {
		route = ROCmDiffusionSamplerRouteForProfile(profile)
	}
	if !route.Matched() {
		return ROCmDiffusionSamplerRoute{}, false
	}
	return route.Clone(), true
}

func ROCmDiffusionSamplerRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmDiffusionSamplerRoute, bool) {
	profile, ok := ResolveROCmModelProfileForInfo(path, info, labels)
	if !ok {
		return ROCmDiffusionSamplerRoute{}, false
	}
	route := profile.DiffusionSamplerRoute
	if !route.Matched() {
		route = ROCmDiffusionSamplerRouteForProfile(profile)
	}
	if !route.Matched() {
		return ROCmDiffusionSamplerRoute{}, false
	}
	return route.Clone(), true
}

func ROCmDiffusionSamplerRouteForInspection(inspection *inference.ModelPackInspection) (ROCmDiffusionSamplerRoute, bool) {
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok {
		return ROCmDiffusionSamplerRoute{}, false
	}
	route := profile.DiffusionSamplerRoute
	if !route.Matched() {
		route = ROCmDiffusionSamplerRouteForProfile(profile)
	}
	if inspection != nil {
		route = route.WithLabels(inspection.Labels)
	}
	if !route.Matched() {
		return ROCmDiffusionSamplerRoute{}, false
	}
	return route.Clone(), true
}

func rocmApplyROCmDiffusionSamplerRouteLabels(labels map[string]string, route ROCmDiffusionSamplerRoute) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if !route.Matched() {
		return labels
	}
	for key, value := range rocmmodel.DiffusionSamplerRouteLabels(route) {
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}
