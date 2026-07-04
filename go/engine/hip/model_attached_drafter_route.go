// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const (
	ROCmAttachedDrafterRegistryContract = rocmmodel.AttachedDrafterRegistryContract

	rocmAttachedDrafterRegistryRouteName = rocmmodel.AttachedDrafterRouteName
	rocmAttachedDrafterRuntimeMetadata   = rocmmodel.AttachedDrafterRuntimeMetadata
	rocmAttachedDrafterRuntimeHIP        = rocmmodel.AttachedDrafterRuntimeHIP
)

type ROCmAttachedDrafterRouteStatus = rocmmodel.AttachedDrafterRouteStatus

const (
	ROCmAttachedDrafterRouteNativePending   = rocmmodel.AttachedDrafterRouteNativePending
	ROCmAttachedDrafterRouteAttachedOnly    = rocmmodel.AttachedDrafterRouteAttachedOnly
	ROCmAttachedDrafterRoutePlannedMetadata = rocmmodel.AttachedDrafterRoutePlannedMetadata
)

// ROCmAttachedDrafterRoute is the model-registry view of Gemma-4 target plus
// assistant MTP pairing. It makes the go-mlx attached-drafter contract
// discoverable while keeping ROCm native HIP attachment explicitly not-linked.
type ROCmAttachedDrafterRoute = rocmmodel.AttachedDrafterRoute

// RegisterROCmAttachedDrafterRoute registers or replaces an architecture-keyed
// attached-drafter route. It gives model packages a reactive way to advertise
// target/assistant pairing, retained-state, and native HIP attachment without
// expanding central model switches.
func RegisterROCmAttachedDrafterRoute(route ROCmAttachedDrafterRoute) {
	route = normalizeRegisteredROCmAttachedDrafterRoute(route)
	if !route.Matched() {
		return
	}
	rocmmodel.RegisterAttachedDrafterRoute(route)
}

// RegisteredROCmAttachedDrafterRouteArchitectures returns extension
// attached-drafter architectures in registration order. Built-in Gemma routes
// are intentionally not included.
func RegisteredROCmAttachedDrafterRouteArchitectures() []string {
	return rocmmodel.RegisteredAttachedDrafterArchitectures()
}

func normalizeRegisteredROCmAttachedDrafterRoute(route ROCmAttachedDrafterRoute) ROCmAttachedDrafterRoute {
	return rocmmodel.NormalizeAttachedDrafterRoute(route).Clone()
}

func DefaultROCmAttachedDrafterRoutes() []ROCmAttachedDrafterRoute {
	modelRoutes := rocmmodel.DefaultAttachedDrafterRoutes()
	routes := make([]ROCmAttachedDrafterRoute, 0, len(modelRoutes))
	for _, modelRoute := range modelRoutes {
		route := rocmAttachedDrafterRouteFromModel(modelRoute)
		if route.Matched() {
			routes = append(routes, route)
		}
	}
	return routes
}

func ROCmAttachedDrafterRouteForArchitecture(architecture string) (ROCmAttachedDrafterRoute, bool) {
	modelRoute, ok := rocmmodel.AttachedDrafterRouteForArchitecture(architecture)
	if !ok {
		return ROCmAttachedDrafterRoute{}, false
	}
	route := rocmAttachedDrafterRouteFromModel(modelRoute)
	if !route.Matched() {
		return ROCmAttachedDrafterRoute{}, false
	}
	return route, true
}

func ROCmAttachedDrafterRouteForProfile(profile ROCmModelProfile) ROCmAttachedDrafterRoute {
	labels := cloneStringMap(profile.Model.Labels)
	model := rocmCloneModelIdentity(profile.Model)
	model.Labels = labels
	if model.Architecture == "" {
		model.Architecture = firstNonEmptyString(profile.Architecture, profile.ArchitectureProfile.ID, profile.Gemma4Settings.ID)
	}
	modelRoute, ok := rocmmodel.AttachedDrafterRouteForIdentity(model.Path, model)
	if !ok {
		return ROCmAttachedDrafterRoute{}
	}
	route := rocmAttachedDrafterRouteFromModel(modelRoute)
	if !route.Matched() {
		return ROCmAttachedDrafterRoute{}
	}
	return route.Clone()
}

func rocmAttachedDrafterRouteFromModel(route rocmmodel.AttachedDrafterRoute) ROCmAttachedDrafterRoute {
	if route.Labels == nil {
		route.Labels = rocmmodel.AttachedDrafterRouteLabels(route)
	}
	if len(route.Capabilities) == 0 {
		route.Capabilities = rocmmodel.AttachedDrafterRouteCapabilities(route)
	}
	return route.Clone()
}

func ROCmAttachedDrafterRouteForIdentity(path string, model inference.ModelIdentity) (ROCmAttachedDrafterRoute, bool) {
	profile, ok := ResolveROCmModelProfile(path, model)
	if !ok {
		return ROCmAttachedDrafterRoute{}, false
	}
	route := profile.AttachedDrafterRoute
	if !route.Matched() {
		route = ROCmAttachedDrafterRouteForProfile(profile)
	}
	if !route.Matched() {
		return ROCmAttachedDrafterRoute{}, false
	}
	return route.Clone(), true
}

func ROCmAttachedDrafterRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmAttachedDrafterRoute, bool) {
	profile, ok := ResolveROCmModelProfileForInfo(path, info, labels)
	if !ok {
		return ROCmAttachedDrafterRoute{}, false
	}
	route := profile.AttachedDrafterRoute
	if !route.Matched() {
		route = ROCmAttachedDrafterRouteForProfile(profile)
	}
	if !route.Matched() {
		return ROCmAttachedDrafterRoute{}, false
	}
	return route.Clone(), true
}

func ROCmAttachedDrafterRouteForInspection(inspection *inference.ModelPackInspection) (ROCmAttachedDrafterRoute, bool) {
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok {
		return ROCmAttachedDrafterRoute{}, false
	}
	route := profile.AttachedDrafterRoute
	if !route.Matched() {
		route = ROCmAttachedDrafterRouteForProfile(profile)
	}
	if inspection != nil {
		route = route.WithLabels(inspection.Labels)
	}
	if !route.Matched() {
		return ROCmAttachedDrafterRoute{}, false
	}
	return route.Clone(), true
}

func rocmApplyROCmAttachedDrafterRouteLabels(labels map[string]string, route ROCmAttachedDrafterRoute) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if !route.Matched() {
		return labels
	}
	for key, value := range rocmmodel.AttachedDrafterRouteLabels(route) {
		if value != "" {
			labels[key] = value
		}
	}
	targetRetainedDecode := hipKernelStatusNotLinked
	if route.RetainedStateRequired && route.RuntimeOwnedKV {
		targetRetainedDecode = hipKernelStatusLinked
	}
	assistantVerify := hipKernelStatusNotLinked
	nativeHandoff := attachedDrafterNativeHandoffTargetDecodeOnly
	if route.NativeAttachment == hipKernelStatusLinked && route.NativeStateGeneration && route.VerifyForward {
		assistantVerify = hipKernelStatusLinked
		nativeHandoff = attachedDrafterNativeHandoffRetainedStateVerifier
	}
	setDefaultLabel := func(key, value string) {
		if labels[key] == "" && value != "" {
			labels[key] = value
		}
	}
	setDefaultLabel("engine_attached_drafter_native_handoff", nativeHandoff)
	setDefaultLabel("engine_attached_drafter_target_retained_decode", targetRetainedDecode)
	setDefaultLabel("engine_attached_drafter_target_retained_state_decode", targetRetainedDecode)
	setDefaultLabel("engine_attached_drafter_assistant_verify", assistantVerify)
	setDefaultLabel("engine_attached_drafter_assistant_state_verify", assistantVerify)
	return labels
}
