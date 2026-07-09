// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"strings"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
	rocmprofile "dappco.re/go/inference/engine/hip/profile"
)

const (
	ROCmLoRAAdapterRegistryContract = rocmmodel.LoRAAdapterRegistryContract

	rocmLoRAAdapterRegistryRouteName = rocmmodel.LoRAAdapterRouteName
	rocmLoRAAdapterLoaderLinear      = rocmmodel.LoRAAdapterLoaderLinear
	rocmLoRAAdapterRuntimeHIP        = rocmmodel.LoRAAdapterRuntimeHIP
	rocmLoRAAdapterRuntimeMetadata   = rocmmodel.LoRAAdapterRuntimeMetadata
)

type ROCmLoRAAdapterRouteStatus = rocmmodel.LoRAAdapterRouteStatus

const (
	ROCmLoRAAdapterRouteExperimentalNative = rocmmodel.LoRAAdapterRouteExperimentalNative
	ROCmLoRAAdapterRouteStagedNative       = rocmmodel.LoRAAdapterRouteStagedNative
	ROCmLoRAAdapterRoutePlannedMetadata    = rocmmodel.LoRAAdapterRoutePlannedMetadata
	ROCmLoRAAdapterRouteAttachedOnly       = rocmmodel.LoRAAdapterRouteAttachedOnly
)

// ROCmLoRAAdapterRoute is the architecture-keyed adapter route consumers can
// enumerate before model load and refresh from a loaded profile. It mirrors
// go-mlx's model-owned ApplyLoRA target policy while preserving ROCm's current
// staged/runtime status.
type ROCmLoRAAdapterRoute = rocmmodel.LoRAAdapterRoute

// RegisterROCmLoRAAdapterRoute registers or replaces an architecture-keyed
// adapter route. It mirrors go-mlx's model-owned ApplyLoRA target-policy
// contract at the ROCm API layer so families can self-register target paths
// without adding another central switch.
func RegisterROCmLoRAAdapterRoute(route ROCmLoRAAdapterRoute) {
	route = normalizeRegisteredROCmLoRAAdapterRoute(route)
	if !route.Matched() {
		return
	}
	rocmmodel.RegisterLoRAAdapterRoute(route)
}

// RegisteredROCmLoRAAdapterRouteArchitectures returns extension LoRA route
// architectures in resolution order. Built-in target policies are intentionally
// not included.
func RegisteredROCmLoRAAdapterRouteArchitectures() []string {
	return rocmmodel.RegisteredLoRAAdapterArchitectures()
}

func normalizeRegisteredROCmLoRAAdapterRoute(route ROCmLoRAAdapterRoute) ROCmLoRAAdapterRoute {
	return rocmmodel.NormalizeLoRAAdapterRoute(route).Clone()
}

func DefaultROCmLoRAAdapterRoutes() []ROCmLoRAAdapterRoute {
	modelRoutes := rocmmodel.DefaultLoRAAdapterRoutes()
	routes := make([]ROCmLoRAAdapterRoute, 0, len(modelRoutes))
	for _, modelRoute := range modelRoutes {
		route := rocmLoRAAdapterRouteFromModel(modelRoute)
		route = rocmLoRAAdapterRouteWithProfile(route, rocmLoRAAdapterProfileForRoute(route))
		if route.Matched() {
			routes = append(routes, route)
		}
	}
	return routes
}

func ROCmLoRAAdapterRouteForArchitecture(architecture string) (ROCmLoRAAdapterRoute, bool) {
	modelRoute, ok := rocmmodel.LoRAAdapterRouteForArchitecture(architecture)
	if !ok {
		return ROCmLoRAAdapterRoute{}, false
	}
	route := rocmLoRAAdapterRouteFromModel(modelRoute)
	route = rocmLoRAAdapterRouteWithProfile(route, rocmLoRAAdapterProfileForRoute(route))
	if !route.Matched() {
		return ROCmLoRAAdapterRoute{}, false
	}
	return route, true
}

func ROCmLoRAAdapterRouteForProfile(profile ROCmModelProfile) ROCmLoRAAdapterRoute {
	model := rocmCloneModelIdentity(profile.Model)
	model.Labels = cloneStringMap(profile.Model.Labels)
	if model.Architecture == "" {
		model.Architecture = firstNonEmptyString(profile.Architecture, profile.ArchitectureProfile.ID, profile.Gemma4Settings.ID, profile.FeatureRoute.Architecture)
	}
	modelRoute, ok := rocmmodel.LoRAAdapterRouteForIdentity(model.Path, model)
	var route ROCmLoRAAdapterRoute
	if ok {
		route = rocmLoRAAdapterRouteFromModel(modelRoute)
	}
	route = rocmLoRAAdapterRouteWithProfile(route, profile)
	if !route.Matched() {
		return ROCmLoRAAdapterRoute{}
	}
	return route.Clone()
}

func rocmLoRAAdapterRouteWithProfile(route ROCmLoRAAdapterRoute, profile ROCmModelProfile) ROCmLoRAAdapterRoute {
	architectureProfile := profile.ArchitectureProfile
	if architectureProfile.ID == "" {
		architectureProfile = profile.Gemma4Settings
	}
	if architectureProfile.ID == "" {
		if resolved, ok := ROCmArchitectureProfileForArchitecture(firstNonEmptyString(route.Architecture, profile.Architecture)); ok {
			architectureProfile = resolved
		}
	}
	hasArchitectureProfile := architectureProfile.ID != ""
	featureRoute := profile.FeatureRoute
	if !featureRoute.Matched() {
		featureRoute = ROCmModelFeatureRouteForProfile(profile)
	}
	loadStatus := profile.LoadStatus
	if loadStatus.empty() {
		loadStatus = ROCmModelLoadStatusForProfile(profile)
	}
	if len(route.TargetPaths) == 0 {
		targetPolicy, policy, ok := rocmLoRAAdapterPolicyForProfile(architectureProfile)
		if !ok {
			return ROCmLoRAAdapterRoute{}
		}
		route.TargetPolicy = firstNonEmptyString(route.TargetPolicy, targetPolicy)
		route.DefaultTargets = append([]string(nil), policy.DefaultTargets...)
		route.SafeTargets = append([]string(nil), policy.SafeTargets...)
		route.ExtendedTargets = append([]string(nil), policy.ExtendedTargets...)
		route.TargetPaths = cloneStringMap(policy.TargetPaths)
	}
	route.Contract = firstNonEmptyString(route.Contract, ROCmLoRAAdapterRegistryContract)
	route.Name = firstNonEmptyString(route.Name, rocmLoRAAdapterRegistryRouteName)
	route.Architecture = firstNonEmptyString(route.Architecture, profile.Architecture, architectureProfile.ID)
	if hasArchitectureProfile {
		route.Architecture = architectureProfile.ID
	}
	route.Family = firstNonEmptyString(route.Family, profile.Family, architectureProfile.Family, route.Architecture)
	route.Loader = firstNonEmptyString(route.Loader, rocmLoRAAdapterLoaderLinear)
	route.RuntimeStatus = firstNonEmptyRuntimeStatus(route.RuntimeStatus, featureRoute.RuntimeStatus, architectureProfile.RuntimeStatus)
	route.TargetPolicy = firstNonEmptyString(route.TargetPolicy, "registered")
	route.DefaultTargets = cleanROCmLoRATargets(route.DefaultTargets)
	route.SafeTargets = cleanROCmLoRATargets(route.SafeTargets)
	route.ExtendedTargets = cleanROCmLoRATargets(route.ExtendedTargets)
	route.TargetPaths = cleanROCmLoRATargetPaths(route.TargetPaths)
	if len(route.SafeTargets) == 0 {
		route.SafeTargets = cleanROCmLoRATargets(append([]string(nil), route.DefaultTargets...))
	}
	if len(route.DefaultTargets) == 0 {
		route.DefaultTargets = cleanROCmLoRATargets(route.SafeTargets)
	}
	route.NativeRuntime = route.NativeRuntime || architectureProfile.NativeRuntime || featureRoute.NativeRuntime || loadStatus.NativeRuntime
	route.AttachedOnly = route.AttachedOnly || architectureProfile.AttachedOnly || featureRoute.AttachedOnly || loadStatus.AttachedOnly
	route.Registered = !route.AttachedOnly && len(route.TargetPaths) > 0
	route.ApplySupported = route.ApplySupported || route.Registered
	route.LoadSupported = route.LoadSupported || route.Registered
	route.FuseSupported = route.FuseSupported || route.Registered && len(route.TargetPaths) > 0
	route.TrainingSupported = route.TrainingSupported || route.Registered
	if hasArchitectureProfile {
		route.Staged = route.Registered && route.NativeRuntime && (route.Staged || loadStatus.Staged || !featureRoute.TextGenerate)
		route.Planned = route.Registered && !route.NativeRuntime
	}
	route.RequiresExtendedOptIn = route.RequiresExtendedOptIn || len(route.ExtendedTargets) > 0
	route = rocmLoRAAdapterRouteWithStatusDefaults(route)
	route.Capabilities = mergeROCmCapabilityIDs(rocmLoRAAdapterRouteCapabilities(route), route.Capabilities)
	route.Labels = rocmLoRAAdapterRouteLabels(route)
	return route.Clone()
}

func rocmLoRAAdapterProfileForRoute(route ROCmLoRAAdapterRoute) ROCmModelProfile {
	profile := ROCmModelProfile{
		Name:         firstNonEmptyString(route.Family, route.Architecture),
		Family:       route.Family,
		Architecture: route.Architecture,
		Registry:     rocmModelRegistryName,
		Model: inference.ModelIdentity{
			Architecture: route.Architecture,
			Labels:       cloneStringMap(route.Labels),
		},
		LoRAAdapterRoute: route.Clone(),
	}
	if architectureProfile, ok := ROCmArchitectureProfileForArchitecture(route.Architecture); ok {
		profile.ArchitectureProfile = architectureProfile
		profile.Gemma4Settings = architectureProfile
	}
	return profile
}

func rocmLoRAAdapterRouteFromModel(route rocmmodel.LoRAAdapterRoute) ROCmLoRAAdapterRoute {
	if route.Labels == nil {
		route.Labels = rocmmodel.LoRAAdapterRouteLabels(route)
	}
	if len(route.Capabilities) == 0 {
		route.Capabilities = rocmmodel.LoRAAdapterRouteCapabilities(route)
	}
	return route.Clone()
}

func ROCmLoRAAdapterRouteForIdentity(path string, model inference.ModelIdentity) (ROCmLoRAAdapterRoute, bool) {
	profile, ok := ResolveROCmModelProfile(path, model)
	if !ok {
		return ROCmLoRAAdapterRoute{}, false
	}
	route := profile.LoRAAdapterRoute
	if !route.Matched() {
		route = ROCmLoRAAdapterRouteForProfile(profile)
	}
	if !route.Matched() {
		return ROCmLoRAAdapterRoute{}, false
	}
	return route.Clone(), true
}

func ROCmLoRAAdapterRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmLoRAAdapterRoute, bool) {
	profile, ok := ResolveROCmModelProfileForInfo(path, info, labels)
	if !ok {
		return ROCmLoRAAdapterRoute{}, false
	}
	route := profile.LoRAAdapterRoute
	if !route.Matched() {
		route = ROCmLoRAAdapterRouteForProfile(profile)
	}
	if !route.Matched() {
		return ROCmLoRAAdapterRoute{}, false
	}
	return route.Clone(), true
}

func ROCmLoRAAdapterRouteForInspection(inspection *inference.ModelPackInspection) (ROCmLoRAAdapterRoute, bool) {
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok {
		return ROCmLoRAAdapterRoute{}, false
	}
	route := profile.LoRAAdapterRoute
	if !route.Matched() {
		route = ROCmLoRAAdapterRouteForProfile(profile)
	}
	if !route.Matched() {
		return ROCmLoRAAdapterRoute{}, false
	}
	return route.Clone(), true
}

func rocmApplyROCmLoRAAdapterRouteLabels(labels map[string]string, route ROCmLoRAAdapterRoute) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if !route.Matched() {
		return labels
	}
	for key, value := range rocmLoRAAdapterRouteLabels(route) {
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}

func rocmLoRAAdapterRouteStatus(route ROCmLoRAAdapterRoute) ROCmLoRAAdapterRouteStatus {
	switch {
	case route.AttachedOnly:
		return ROCmLoRAAdapterRouteAttachedOnly
	case route.Planned:
		return ROCmLoRAAdapterRoutePlannedMetadata
	case route.Staged:
		return ROCmLoRAAdapterRouteStagedNative
	default:
		return ROCmLoRAAdapterRouteExperimentalNative
	}
}

func rocmLoRAAdapterRouteWithStatusDefaults(route ROCmLoRAAdapterRoute) ROCmLoRAAdapterRoute {
	if route.Runtime == "" {
		route.Runtime = rocmLoRAAdapterRuntimeHIP
		if route.Planned || !route.NativeRuntime {
			route.Runtime = rocmLoRAAdapterRuntimeMetadata
		}
	}
	if route.AttachedOnly {
		route.Registered = false
		route.ApplySupported = false
		route.LoadSupported = false
		route.FuseSupported = false
		route.TrainingSupported = false
		route.Staged = false
		route.Planned = false
	}
	if route.Registered && !route.NativeRuntime {
		route.Planned = true
	}
	if route.Planned {
		route.Runtime = rocmLoRAAdapterRuntimeMetadata
	}
	if route.Status == "" {
		route.Status = rocmLoRAAdapterRouteStatus(route)
	}
	return route
}

func rocmLoRAAdapterRouteCapabilities(route ROCmLoRAAdapterRoute) []inference.CapabilityID {
	return rocmmodel.LoRAAdapterRouteCapabilities(route)
}

func rocmLoRAAdapterRouteLabels(route ROCmLoRAAdapterRoute) map[string]string {
	return rocmmodel.LoRAAdapterRouteLabels(route)
}

func rocmLoRATargetPolicyForArchitecture(architecture string) (Gemma4LoRATargetPolicy, bool) {
	if route, ok := rocmmodel.RegisteredLoRAAdapterRouteForArchitecture(architecture); ok && route.Registered && len(route.TargetPaths) > 0 {
		return cloneGemma4LoRATargetPolicy(Gemma4LoRATargetPolicy{
			DefaultTargets:  append([]string(nil), route.DefaultTargets...),
			SafeTargets:     append([]string(nil), route.SafeTargets...),
			ExtendedTargets: append([]string(nil), route.ExtendedTargets...),
			TargetPaths:     cloneStringMap(route.TargetPaths),
		}), true
	}
	if policy, ok := rocmprofile.LoRATargetPolicyForArchitecture(architecture); ok {
		return policy, true
	}
	return Gemma4LoRATargetPolicy{}, false
}

func rocmLoRAAdapterPolicyForProfile(architectureProfile ROCmArchitectureProfile) (string, Gemma4LoRATargetPolicy, bool) {
	if policy, ok := rocmprofile.LoRATargetPolicyForProfile(architectureProfile); ok {
		return rocmLoRATargetPolicyName(architectureProfile), policy, true
	}
	return "", Gemma4LoRATargetPolicy{}, false
}

func rocmLoRATargetPolicyName(architectureProfile ROCmArchitectureProfile) string {
	if name := rocmprofile.ArchitectureProfileLoRATargetPolicyName(architectureProfile.ID); name != "" {
		return name
	}
	if architectureProfile.Family != "" {
		return architectureProfile.Family
	}
	if architectureProfile.ID != "" {
		return architectureProfile.ID
	}
	return "profile"
}

func cleanROCmLoRATargets(targets []string) []string {
	out := make([]string, 0, len(targets))
	seen := map[string]bool{}
	for _, target := range targets {
		target = strings.TrimSpace(target)
		if target == "" || seen[target] {
			continue
		}
		seen[target] = true
		out = append(out, target)
	}
	return out
}

func cleanROCmLoRATargetPaths(paths map[string]string) map[string]string {
	if len(paths) == 0 {
		return nil
	}
	out := make(map[string]string, len(paths))
	for target, path := range paths {
		target = strings.TrimSpace(target)
		path = strings.TrimSpace(path)
		if target == "" || path == "" {
			continue
		}
		out[target] = path
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
