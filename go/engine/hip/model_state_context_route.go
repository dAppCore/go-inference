// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const (
	ROCmStateContextRegistryContract = rocmmodel.StateContextRegistryContract

	rocmStateContextRegistryRouteName = rocmmodel.StateContextRouteName
	rocmStateContextRuntimeAPI        = rocmmodel.StateContextRuntimeAPI
	rocmStateContextRuntimeMetadata   = rocmmodel.StateContextRuntimeMetadata
)

type ROCmStateContextRouteStatus = rocmmodel.StateContextRouteStatus

const (
	ROCmStateContextRouteExperimentalRuntime = rocmmodel.StateContextRouteExperimentalRuntime
	ROCmStateContextRouteAttachedRuntime     = rocmmodel.StateContextRouteAttachedRuntime
	ROCmStateContextRoutePlannedMetadata     = rocmmodel.StateContextRoutePlannedMetadata
)

// ROCmStateContextRoute is the model-owned context and retained-state route
// exposed through the registry. It makes Gemma-4's remaining-context default
// and runtime-owned KV lifecycle discoverable without requiring callers to
// scrape generate/state-session labels.
type ROCmStateContextRoute = rocmmodel.StateContextRoute

// RegisterROCmStateContextRoute registers or replaces an architecture-keyed
// retained-state/context route. It gives model packages a reactive way to
// describe runtime-owned KV, sleep/wake, and state-bundle behavior without
// expanding central route switches.
func RegisterROCmStateContextRoute(route ROCmStateContextRoute) {
	route = normalizeRegisteredROCmStateContextRoute(route)
	if !route.Matched() {
		return
	}
	rocmmodel.RegisterStateContextRoute(route)
}

// RegisteredROCmStateContextRouteArchitectures returns extension state-context
// architectures in registration order. Built-in retained-state routes are
// intentionally not included.
func RegisteredROCmStateContextRouteArchitectures() []string {
	return rocmmodel.RegisteredStateContextArchitectures()
}

func normalizeRegisteredROCmStateContextRoute(route ROCmStateContextRoute) ROCmStateContextRoute {
	return rocmmodel.NormalizeStateContextRoute(route).Clone()
}

func DefaultROCmStateContextRoutes() []ROCmStateContextRoute {
	modelRoutes := rocmmodel.DefaultStateContextRoutes()
	routes := make([]ROCmStateContextRoute, 0, len(modelRoutes))
	for _, modelRoute := range modelRoutes {
		route := rocmStateContextRouteFromModel(modelRoute)
		if route.Matched() {
			routes = append(routes, route)
		}
	}
	return routes
}

func ROCmStateContextRouteForArchitecture(architecture string) (ROCmStateContextRoute, bool) {
	modelRoute, ok := rocmmodel.StateContextRouteForArchitecture(architecture)
	if !ok {
		return ROCmStateContextRoute{}, false
	}
	route := rocmStateContextRouteFromModel(modelRoute)
	if !route.Matched() {
		return ROCmStateContextRoute{}, false
	}
	return route, true
}

func ROCmStateContextRouteForProfile(profile ROCmModelProfile) ROCmStateContextRoute {
	labels := cloneStringMap(profile.Model.Labels)
	model := rocmCloneModelIdentity(profile.Model)
	model.Labels = labels
	if model.Architecture == "" {
		model.Architecture = firstNonEmptyString(profile.Architecture, profile.ArchitectureProfile.ID, profile.Gemma4Settings.ID)
	}
	modelRoute, ok := rocmmodel.StateContextRouteForIdentity(model.Path, model)
	if !ok {
		return ROCmStateContextRoute{}
	}
	route := rocmStateContextRouteFromModel(modelRoute)
	if !route.Matched() {
		return ROCmStateContextRoute{}
	}
	return route.Clone()
}

func rocmStateContextRouteFromModel(route rocmmodel.StateContextRoute) ROCmStateContextRoute {
	if route.Labels == nil {
		route.Labels = rocmmodel.StateContextRouteLabels(route)
	}
	if len(route.Capabilities) == 0 {
		route.Capabilities = rocmmodel.StateContextRouteCapabilities(route)
	}
	return route.Clone()
}

func ROCmStateContextRouteForIdentity(path string, model inference.ModelIdentity) (ROCmStateContextRoute, bool) {
	profile, ok := ResolveROCmModelProfile(path, model)
	if !ok {
		return ROCmStateContextRoute{}, false
	}
	route := profile.StateContextRoute
	if !route.Matched() {
		route = ROCmStateContextRouteForProfile(profile)
	}
	if !route.Matched() {
		return ROCmStateContextRoute{}, false
	}
	return route.Clone(), true
}

func ROCmStateContextRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmStateContextRoute, bool) {
	profile, ok := ResolveROCmModelProfileForInfo(path, info, labels)
	if !ok {
		return ROCmStateContextRoute{}, false
	}
	route := profile.StateContextRoute
	if !route.Matched() {
		route = ROCmStateContextRouteForProfile(profile)
	}
	if !route.Matched() {
		return ROCmStateContextRoute{}, false
	}
	return route.Clone(), true
}

func ROCmStateContextRouteForInspection(inspection *inference.ModelPackInspection) (ROCmStateContextRoute, bool) {
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok {
		return ROCmStateContextRoute{}, false
	}
	route := profile.StateContextRoute
	if !route.Matched() {
		route = ROCmStateContextRouteForProfile(profile)
	}
	if inspection != nil {
		if inspection.Model.ContextLength > 0 {
			route.ContextWindow = inspection.Model.ContextLength
		}
		route = route.WithLabels(inspection.Labels)
	}
	if !route.Matched() {
		return ROCmStateContextRoute{}, false
	}
	return route.Clone(), true
}

func rocmApplyROCmStateContextRouteLabels(labels map[string]string, route ROCmStateContextRoute) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if !route.Matched() {
		return labels
	}
	for key, value := range rocmmodel.StateContextRouteLabels(route) {
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}
