// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const (
	ROCmModelFeatureRegistryContract = rocmmodel.FeatureRegistryContract

	rocmModelFeatureRegistryRouteName = rocmmodel.FeatureRouteName
)

// ROCmModelFeatureRoute is the architecture-keyed parser/template/capability
// route consumers can enumerate before model load, then refresh from the loaded
// profile once quant and runtime details are known.
type ROCmModelFeatureRoute = rocmmodel.FeatureRoute

// RegisterROCmModelFeatureRoute registers or replaces an architecture-keyed
// feature route. It mirrors go-mlx's model-family self-registration at the
// ROCm API layer so a family can enable parser/template/runtime features
// without adding another central switch.
func RegisterROCmModelFeatureRoute(route ROCmModelFeatureRoute) {
	route = normalizeRegisteredROCmModelFeatureRoute(route)
	if !route.Matched() {
		return
	}
	rocmmodel.RegisterFeatureRoute(route)
}

// RegisteredROCmModelFeatureRouteArchitectures returns extension feature-route
// architectures in resolution order. Built-in profile routes are intentionally
// not included.
func RegisteredROCmModelFeatureRouteArchitectures() []string {
	return rocmmodel.RegisteredFeatureArchitectures()
}

func normalizeRegisteredROCmModelFeatureRoute(route ROCmModelFeatureRoute) ROCmModelFeatureRoute {
	return rocmmodel.NormalizeFeatureRoute(route).Clone()
}

func DefaultROCmModelFeatureRoutes() []ROCmModelFeatureRoute {
	modelRoutes := rocmmodel.DefaultFeatureRoutes()
	routes := make([]ROCmModelFeatureRoute, 0, len(modelRoutes))
	for _, modelRoute := range modelRoutes {
		route := rocmModelFeatureRouteFromModel(modelRoute)
		route = rocmModelFeatureRouteWithEngineFeatures(route, rocmModelFeatureProfileForRoute(route), ROCmEngineFeatures{})
		if route.Matched() {
			routes = append(routes, route)
		}
	}
	return routes
}

func ROCmModelFeatureRouteForArchitecture(architecture string) (ROCmModelFeatureRoute, bool) {
	modelRoute, ok := rocmmodel.FeatureRouteForArchitecture(architecture)
	if !ok {
		return ROCmModelFeatureRoute{}, false
	}
	route := rocmModelFeatureRouteFromModel(modelRoute)
	route = rocmModelFeatureRouteWithEngineFeatures(route, rocmModelFeatureProfileForRoute(route), ROCmEngineFeatures{})
	if !route.Matched() {
		return ROCmModelFeatureRoute{}, false
	}
	return route, true
}

func ROCmModelFeatureRouteForProfile(profile ROCmModelProfile) ROCmModelFeatureRoute {
	features := profile.EngineFeatures
	if features.empty() {
		features = ROCmEngineFeaturesForProfile(profile)
	}
	model := rocmCloneModelIdentity(profile.Model)
	model.Labels = cloneStringMap(profile.Model.Labels)
	if model.Architecture == "" {
		model.Architecture = firstNonEmptyString(profile.Architecture, profile.ArchitectureProfile.ID, profile.Gemma4Settings.ID, features.Architecture)
	}
	modelRoute, ok := rocmmodel.FeatureRouteForIdentity(model.Path, model)
	var route ROCmModelFeatureRoute
	if ok {
		route = rocmModelFeatureRouteFromModel(modelRoute)
	}
	route = rocmModelFeatureRouteWithEngineFeatures(route, profile, features)
	if !route.Matched() {
		return ROCmModelFeatureRoute{}
	}
	return route.Clone()
}

func rocmModelFeatureRouteWithEngineFeatures(route ROCmModelFeatureRoute, profile ROCmModelProfile, features ROCmEngineFeatures) ROCmModelFeatureRoute {
	architectureProfile := profile.ArchitectureProfile
	if architectureProfile.ID == "" {
		architectureProfile = profile.Gemma4Settings
	}
	if architectureProfile.ID == "" {
		if resolved, ok := ROCmArchitectureProfileForArchitecture(firstNonEmptyString(profile.Architecture, features.Architecture)); ok {
			architectureProfile = resolved
		}
	}
	hasArchitectureProfile := architectureProfile.ID != ""
	if features.empty() && (profile.Architecture != "" || hasArchitectureProfile) {
		features = ROCmEngineFeaturesForProfile(profile)
	}
	route.Contract = firstNonEmptyString(route.Contract, ROCmModelFeatureRegistryContract)
	route.Name = firstNonEmptyString(route.Name, rocmModelFeatureRegistryRouteName)
	route.Architecture = firstNonEmptyString(route.Architecture, features.Architecture, profile.Architecture, architectureProfile.ID)
	route.Family = firstNonEmptyString(route.Family, features.Family, profile.Family, architectureProfile.Family, route.Architecture)
	route.RuntimeStatus = firstNonEmptyRuntimeStatus(route.RuntimeStatus, features.RuntimeStatus, architectureProfile.RuntimeStatus)
	route.ReasoningParserID = firstNonEmptyString(route.ReasoningParserID, features.ReasoningParserID, architectureProfile.ParserID)
	route.ToolParserID = firstNonEmptyString(route.ToolParserID, features.ToolParserID, architectureProfile.ToolParserID)
	route.ChatTemplateID = firstNonEmptyString(route.ChatTemplateID, features.ChatTemplateID, architectureProfile.ChatTemplate)
	route.GenerationRole = firstNonEmptyString(route.GenerationRole, architectureProfile.GenerationRole)
	route.Registered = route.Registered || route.Architecture != ""
	route.NativeRuntime = route.NativeRuntime || features.NativeRuntime || architectureProfile.NativeRuntime
	route.Generation = route.Generation || architectureProfile.Generation
	if hasArchitectureProfile {
		route.TextGenerate = features.TextGenerate
	} else {
		route.TextGenerate = route.TextGenerate || features.TextGenerate
	}
	route.Chat = route.Chat || architectureProfile.Chat
	if hasArchitectureProfile {
		route.ModelContextWindow = features.ModelContextWindow
	} else {
		route.ModelContextWindow = route.ModelContextWindow || features.ModelContextWindow
	}
	route.ReasoningParse = route.ReasoningParse || features.ReasoningParse || route.ReasoningParserID != ""
	route.ToolParse = route.ToolParse || features.ToolParse || route.ToolParserID != ""
	route.ChatTemplate = route.ChatTemplate || features.ChatTemplate || route.ChatTemplateID != ""
	route.DefaultThinking = route.DefaultThinking || features.DefaultThinking || architectureProfile.DefaultThinking
	route.RequiresChatTemplate = route.RequiresChatTemplate || architectureProfile.RequiresChatTemplate
	route.Embeddings = route.Embeddings || features.Embeddings || architectureProfile.Embeddings
	route.Rerank = route.Rerank || features.Rerank || architectureProfile.Rerank
	route.MoE = route.MoE || features.MoE || architectureProfile.MoE
	route.SequenceMixer = route.SequenceMixer || features.SequenceMixer
	route.AttachedOnly = route.AttachedOnly || features.AttachedOnly || architectureProfile.AttachedOnly
	route.Capabilities = mergeROCmCapabilityIDs(rocmModelFeatureRouteCapabilities(route), mergeROCmCapabilityIDs(features.EnabledCapabilities(), route.Capabilities))
	route.Labels = rocmModelFeatureRouteLabels(route)
	return route.Clone()
}

func rocmModelFeatureProfileForRoute(route ROCmModelFeatureRoute) ROCmModelProfile {
	profile := ROCmModelProfile{
		Name:         firstNonEmptyString(route.Family, route.Architecture),
		Family:       route.Family,
		Architecture: route.Architecture,
		Registry:     rocmModelRegistryName,
		Model: inference.ModelIdentity{
			Architecture: route.Architecture,
			Labels:       cloneStringMap(route.Labels),
		},
	}
	if architectureProfile, ok := ROCmArchitectureProfileForArchitecture(route.Architecture); ok {
		profile.ArchitectureProfile = architectureProfile
		profile.Gemma4Settings = architectureProfile
	}
	return profile
}

func rocmModelFeatureRouteFromModel(route rocmmodel.FeatureRoute) ROCmModelFeatureRoute {
	if route.Labels == nil || len(route.Capabilities) == 0 {
		normalized := rocmmodel.NormalizeFeatureRoute(route)
		if route.Labels == nil {
			route.Labels = normalized.Labels
		}
		if len(route.Capabilities) == 0 {
			route.Capabilities = normalized.Capabilities
		}
	}
	return route.Clone()
}

func ROCmModelFeatureRouteForIdentity(path string, model inference.ModelIdentity) (ROCmModelFeatureRoute, bool) {
	profile, ok := ResolveROCmModelProfile(path, model)
	if !ok {
		return ROCmModelFeatureRoute{}, false
	}
	return profile.FeatureRoute.Clone(), true
}

func ROCmModelFeatureRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmModelFeatureRoute, bool) {
	profile, ok := ResolveROCmModelProfileForInfo(path, info, labels)
	if !ok {
		return ROCmModelFeatureRoute{}, false
	}
	return profile.FeatureRoute.Clone(), true
}

func ROCmModelFeatureRouteForInspection(inspection *inference.ModelPackInspection) (ROCmModelFeatureRoute, bool) {
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok {
		return ROCmModelFeatureRoute{}, false
	}
	return profile.FeatureRoute.Clone(), true
}

func rocmApplyROCmModelFeatureRouteLabels(labels map[string]string, route ROCmModelFeatureRoute) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if !route.Matched() {
		return labels
	}
	for key, value := range rocmModelFeatureRouteLabels(route) {
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}

func rocmModelFeatureRouteLabels(route ROCmModelFeatureRoute) map[string]string {
	return rocmmodel.FeatureRouteLabels(route)
}

func rocmModelFeatureRouteCapabilities(route ROCmModelFeatureRoute) []inference.CapabilityID {
	return rocmmodel.FeatureRouteCapabilities(route)
}

func mergeROCmCapabilityIDs(primary, secondary []inference.CapabilityID) []inference.CapabilityID {
	out := make([]inference.CapabilityID, 0, len(primary)+len(secondary))
	seen := map[inference.CapabilityID]bool{}
	for _, ids := range [][]inference.CapabilityID{primary, secondary} {
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

func firstNonEmptyRuntimeStatus(values ...inference.FeatureRuntimeStatus) inference.FeatureRuntimeStatus {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
