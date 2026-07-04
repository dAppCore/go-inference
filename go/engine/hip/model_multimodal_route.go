// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const (
	ROCmMultimodalProcessorRegistryContract = rocmmodel.MultimodalProcessorRegistryContract

	rocmMultimodalProcessorRegistryRouteName = rocmmodel.MultimodalProcessorRouteName
	rocmMultimodalProcessorRuntimeHIP        = rocmmodel.MultimodalProcessorRuntimeHIP
	rocmMultimodalProcessorRuntimeMetadata   = rocmmodel.MultimodalProcessorRuntimeMetadata
)

type ROCmMultimodalProcessorRouteStatus = rocmmodel.MultimodalProcessorRouteStatus

const (
	ROCmMultimodalProcessorExperimentalNative = rocmmodel.MultimodalProcessorExperimentalNative
	ROCmMultimodalProcessorPlannedMetadata    = rocmmodel.MultimodalProcessorPlannedMetadata
)

// ROCmMultimodalProcessorRoute is the model-owned image/audio processor route
// exposed through the registry. It keeps Gemma multimodal config metadata
// discoverable while ROCm vision/audio towers and projectors remain explicitly
// not-linked.
type ROCmMultimodalProcessorRoute = rocmmodel.MultimodalProcessorRoute

// RegisterROCmMultimodalProcessorRoute registers or replaces an
// architecture-keyed multimodal processor route. It gives ROCm the same
// model-owned registration shape as go-mlx while keeping the concrete processor
// runtime described through ROCm metadata.
func RegisterROCmMultimodalProcessorRoute(route ROCmMultimodalProcessorRoute) {
	route = normalizeRegisteredROCmMultimodalProcessorRoute(route)
	if !route.Matched() {
		return
	}
	rocmmodel.RegisterMultimodalProcessorRoute(route)
}

// RegisteredROCmMultimodalProcessorRouteArchitectures returns extension
// multimodal processor architectures in registration order. Built-in Gemma
// routes are intentionally not included.
func RegisteredROCmMultimodalProcessorRouteArchitectures() []string {
	return rocmmodel.RegisteredMultimodalProcessorArchitectures()
}

func normalizeRegisteredROCmMultimodalProcessorRoute(route ROCmMultimodalProcessorRoute) ROCmMultimodalProcessorRoute {
	return rocmmodel.NormalizeMultimodalProcessorRoute(route).Clone()
}

func DefaultROCmMultimodalProcessorRoutes() []ROCmMultimodalProcessorRoute {
	modelRoutes := rocmmodel.DefaultMultimodalProcessorRoutes()
	routes := make([]ROCmMultimodalProcessorRoute, 0, len(modelRoutes))
	for _, modelRoute := range modelRoutes {
		route := rocmMultimodalProcessorRouteFromModel(modelRoute)
		if route.Matched() {
			routes = append(routes, route)
		}
	}
	return routes
}

func ROCmMultimodalProcessorRouteForArchitecture(architecture string) (ROCmMultimodalProcessorRoute, bool) {
	modelRoute, ok := rocmmodel.MultimodalProcessorRouteForArchitecture(architecture)
	if !ok {
		return ROCmMultimodalProcessorRoute{}, false
	}
	route := rocmMultimodalProcessorRouteFromModel(modelRoute)
	if !route.Matched() {
		return ROCmMultimodalProcessorRoute{}, false
	}
	return route, true
}

func ROCmMultimodalProcessorRouteForProfile(profile ROCmModelProfile) ROCmMultimodalProcessorRoute {
	labels := cloneStringMap(profile.Model.Labels)
	model := rocmCloneModelIdentity(profile.Model)
	model.Labels = labels
	if model.Architecture == "" {
		model.Architecture = firstNonEmptyString(profile.Architecture, profile.ArchitectureProfile.ID, profile.Gemma4Settings.ID)
	}
	modelRoute, ok := rocmmodel.MultimodalProcessorRouteForIdentity(model.Path, model)
	if !ok {
		return ROCmMultimodalProcessorRoute{}
	}
	route := rocmMultimodalProcessorRouteFromModel(modelRoute)
	return route.Clone()
}

func rocmMultimodalProcessorRouteFromModel(route rocmmodel.MultimodalProcessorRoute) ROCmMultimodalProcessorRoute {
	if route.Labels == nil {
		route.Labels = rocmmodel.MultimodalProcessorRouteLabels(route)
	}
	return route.Clone()
}

func ROCmMultimodalProcessorRouteForIdentity(path string, model inference.ModelIdentity) (ROCmMultimodalProcessorRoute, bool) {
	profile, ok := ResolveROCmModelProfile(path, model)
	if !ok {
		return ROCmMultimodalProcessorRoute{}, false
	}
	route := profile.MultimodalProcessorRoute
	if !route.Matched() {
		route = ROCmMultimodalProcessorRouteForProfile(profile)
	}
	if !route.Matched() {
		return ROCmMultimodalProcessorRoute{}, false
	}
	return route.Clone(), true
}

func ROCmMultimodalProcessorRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmMultimodalProcessorRoute, bool) {
	profile, ok := ResolveROCmModelProfileForInfo(path, info, labels)
	if !ok {
		return ROCmMultimodalProcessorRoute{}, false
	}
	route := profile.MultimodalProcessorRoute
	if !route.Matched() {
		route = ROCmMultimodalProcessorRouteForProfile(profile)
	}
	if !route.Matched() {
		return ROCmMultimodalProcessorRoute{}, false
	}
	return route.Clone(), true
}

func ROCmMultimodalProcessorRouteForInspection(inspection *inference.ModelPackInspection) (ROCmMultimodalProcessorRoute, bool) {
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok {
		return ROCmMultimodalProcessorRoute{}, false
	}
	route := profile.MultimodalProcessorRoute
	if !route.Matched() {
		route = ROCmMultimodalProcessorRouteForProfile(profile)
	}
	if inspection != nil {
		route = route.WithLabels(inspection.Labels)
	}
	if !route.Matched() {
		return ROCmMultimodalProcessorRoute{}, false
	}
	return route.Clone(), true
}

func rocmApplyROCmMultimodalProcessorRouteLabels(labels map[string]string, route ROCmMultimodalProcessorRoute) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if !route.Matched() {
		return labels
	}
	for key, value := range rocmmodel.MultimodalProcessorRouteLabels(route) {
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}

func rocmMultimodalMergeLabels(left, right map[string]string) map[string]string {
	out := cloneStringMap(left)
	if out == nil {
		out = map[string]string{}
	}
	for key, value := range right {
		if value != "" {
			out[key] = value
		}
	}
	return out
}
