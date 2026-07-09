// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const (
	ROCmModelTokenizerRegistryContract = rocmmodel.TokenizerRegistryContract

	rocmModelTokenizerRegistryRouteName = rocmmodel.TokenizerRouteName
	rocmModelTokenizerLoaderHFJSON      = rocmmodel.TokenizerLoaderHFJSON
	rocmModelTokenizerRuntimeHost       = rocmmodel.TokenizerRuntimeHost
)

// ROCmModelTokenizerRoute is the architecture-keyed tokenizer and
// chat-template route. It mirrors go-mlx's model-owned tokenizer surface while
// keeping concrete tokenizer implementations behind go-inference identities.
type ROCmModelTokenizerRoute = rocmmodel.TokenizerRoute

// RegisterROCmModelTokenizerRoute registers or replaces an architecture-keyed
// tokenizer route. It gives ROCm the same self-registration shape as go-mlx
// model packages without requiring central switch edits for every family.
func RegisterROCmModelTokenizerRoute(route ROCmModelTokenizerRoute) {
	route = normalizeRegisteredROCmModelTokenizerRoute(route)
	if !route.Matched() {
		return
	}
	rocmmodel.RegisterTokenizerRoute(route)
}

// RegisteredROCmModelTokenizerRouteArchitectures returns extension tokenizer
// architectures in resolution order. Built-in profile routes are intentionally
// not included.
func RegisteredROCmModelTokenizerRouteArchitectures() []string {
	return rocmmodel.RegisteredTokenizerArchitectures()
}

func normalizeRegisteredROCmModelTokenizerRoute(route ROCmModelTokenizerRoute) ROCmModelTokenizerRoute {
	return rocmmodel.NormalizeTokenizerRoute(route).Clone()
}

func DefaultROCmModelTokenizerRoutes() []ROCmModelTokenizerRoute {
	modelRoutes := rocmmodel.DefaultTokenizerRoutes()
	routes := make([]ROCmModelTokenizerRoute, 0, len(modelRoutes))
	for _, modelRoute := range modelRoutes {
		route := rocmModelTokenizerRouteFromModel(modelRoute)
		route = rocmModelTokenizerRouteWithProfile(route, rocmModelTokenizerProfileForRoute(route))
		if route.Matched() {
			routes = append(routes, route)
		}
	}
	return routes
}

func ROCmModelTokenizerRouteForArchitecture(architecture string) (ROCmModelTokenizerRoute, bool) {
	modelRoute, ok := rocmmodel.TokenizerRouteForArchitecture(architecture)
	if !ok {
		return ROCmModelTokenizerRoute{}, false
	}
	route := rocmModelTokenizerRouteFromModel(modelRoute)
	route = rocmModelTokenizerRouteWithProfile(route, rocmModelTokenizerProfileForRoute(route))
	if !route.Matched() {
		return ROCmModelTokenizerRoute{}, false
	}
	return route, true
}

func ROCmModelTokenizerRouteForProfile(profile ROCmModelProfile) ROCmModelTokenizerRoute {
	model := rocmCloneModelIdentity(profile.Model)
	model.Labels = cloneStringMap(profile.Model.Labels)
	if model.Architecture == "" {
		model.Architecture = firstNonEmptyString(profile.Architecture, profile.ArchitectureProfile.ID, profile.Gemma4Settings.ID, profile.FeatureRoute.Architecture)
	}
	modelRoute, ok := rocmmodel.TokenizerRouteForIdentity(model.Path, model)
	var route ROCmModelTokenizerRoute
	if ok {
		route = rocmModelTokenizerRouteFromModel(modelRoute)
	}
	route = rocmModelTokenizerRouteWithProfile(route, profile)
	if !route.Matched() {
		return ROCmModelTokenizerRoute{}
	}
	return route.Clone()
}

func rocmModelTokenizerRouteWithProfile(route ROCmModelTokenizerRoute, profile ROCmModelProfile) ROCmModelTokenizerRoute {
	featureRoute := profile.FeatureRoute
	if !featureRoute.Matched() {
		featureRoute = ROCmModelFeatureRouteForProfile(profile)
	}
	architectureProfile := profile.ArchitectureProfile
	if architectureProfile.ID == "" {
		architectureProfile = profile.Gemma4Settings
	}
	if architectureProfile.ID == "" {
		if resolved, ok := ROCmArchitectureProfileForArchitecture(firstNonEmptyString(route.Architecture, profile.Architecture, featureRoute.Architecture)); ok {
			architectureProfile = resolved
		}
	}
	route.Contract = firstNonEmptyString(route.Contract, ROCmModelTokenizerRegistryContract)
	route.Name = firstNonEmptyString(route.Name, rocmModelTokenizerRegistryRouteName)
	route.Architecture = firstNonEmptyString(route.Architecture, featureRoute.Architecture, profile.Architecture, architectureProfile.ID)
	route.Family = firstNonEmptyString(route.Family, featureRoute.Family, profile.Family, architectureProfile.Family, route.Architecture)
	route.Loader = firstNonEmptyString(route.Loader, rocmModelTokenizerLoaderHFJSON)
	route.Runtime = firstNonEmptyString(route.Runtime, rocmModelTokenizerRuntimeHost)
	route.TokenizerKind = firstNonEmptyString(route.TokenizerKind, route.Tokenizer.Kind, rocmTokenizerKindForArchitectureProfile(architectureProfile))
	route.TokenizerPath = firstNonEmptyString(route.TokenizerPath, route.Tokenizer.Path)
	route.ChatTemplateID = firstNonEmptyString(route.ChatTemplateID, route.Tokenizer.ChatTemplate, featureRoute.ChatTemplateID, architectureProfile.ChatTemplate)
	route.ChatTemplateSource = firstNonEmptyString(route.ChatTemplateSource, rocmTokenizerChatTemplateSource(route.ChatTemplateID, ""))
	route.ReasoningParserID = firstNonEmptyString(route.ReasoningParserID, featureRoute.ReasoningParserID, architectureProfile.ParserID)
	route.ToolParserID = firstNonEmptyString(route.ToolParserID, featureRoute.ToolParserID, architectureProfile.ToolParserID)
	route.GenerationRole = firstNonEmptyString(route.GenerationRole, featureRoute.GenerationRole, architectureProfile.GenerationRole)
	if route.Tokenizer.Kind == "" {
		route.Tokenizer.Kind = route.TokenizerKind
	}
	if route.Tokenizer.Path == "" {
		route.Tokenizer.Path = route.TokenizerPath
	}
	if route.Tokenizer.ChatTemplate == "" {
		route.Tokenizer.ChatTemplate = route.ChatTemplateID
	}
	route.ThinkingChannel = route.ThinkingChannel ||
		(route.ThinkingChannelOpen != "" && route.ThinkingChannelClose != "") ||
		(route.ThinkingChannelOpenID != 0 && route.ThinkingChannelCloseID != 0)
	route.Registered = route.Registered || route.Architecture != ""
	route.NativeRuntime = route.NativeRuntime || featureRoute.NativeRuntime || architectureProfile.NativeRuntime
	route.ChatTemplate = route.ChatTemplate || route.ChatTemplateID != ""
	route.RequiresChatTemplate = route.RequiresChatTemplate || featureRoute.RequiresChatTemplate || architectureProfile.RequiresChatTemplate
	route.ModelOwnedTemplate = route.ModelOwnedTemplate || (route.ChatTemplateID != "" && !route.SidecarTemplate)
	route.Generation = route.Generation || featureRoute.Generation || architectureProfile.Generation
	route.Chat = route.Chat || featureRoute.Chat || architectureProfile.Chat
	if len(route.RequiredFiles) == 0 {
		route.RequiredFiles = []string{rocmmodel.TokenizerRequiredSidecar}
	}
	if len(route.OptionalFiles) == 0 {
		route.OptionalFiles = []string{"tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json", "generation_config.json"}
	}
	route.Capabilities = mergeROCmCapabilityIDs(rocmTokenizerRouteCapabilities(route.ChatTemplate), route.Capabilities)
	route.Labels = rocmModelTokenizerRouteLabels(route)
	return route.Clone()
}

func rocmModelTokenizerProfileForRoute(route ROCmModelTokenizerRoute) ROCmModelProfile {
	profile := ROCmModelProfile{
		Name:         firstNonEmptyString(route.Family, route.Architecture),
		Family:       route.Family,
		Architecture: route.Architecture,
		Registry:     rocmModelRegistryName,
		Model: inference.ModelIdentity{
			Architecture: route.Architecture,
			Labels:       cloneStringMap(route.Labels),
		},
		TokenizerRoute: route.Clone(),
	}
	if architectureProfile, ok := ROCmArchitectureProfileForArchitecture(route.Architecture); ok {
		profile.ArchitectureProfile = architectureProfile
		profile.Gemma4Settings = architectureProfile
	}
	return profile
}

func rocmModelTokenizerRouteFromModel(route rocmmodel.TokenizerRoute) ROCmModelTokenizerRoute {
	if route.Labels == nil {
		route.Labels = rocmmodel.TokenizerRouteLabels(route)
	}
	if len(route.Capabilities) == 0 {
		route.Capabilities = rocmmodel.TokenizerRouteCapabilities(route.ChatTemplate)
	}
	return route.Clone()
}

func ROCmModelTokenizerRouteForIdentity(path string, model inference.ModelIdentity) (ROCmModelTokenizerRoute, bool) {
	profile, ok := ResolveROCmModelProfile(path, model)
	if !ok {
		return ROCmModelTokenizerRoute{}, false
	}
	return profile.TokenizerRoute.Clone(), true
}

func ROCmModelTokenizerRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmModelTokenizerRoute, bool) {
	profile, ok := ResolveROCmModelProfileForInfo(path, info, labels)
	if !ok {
		return ROCmModelTokenizerRoute{}, false
	}
	return profile.TokenizerRoute.Clone(), true
}

func ROCmModelTokenizerRouteForInspection(inspection *inference.ModelPackInspection) (ROCmModelTokenizerRoute, bool) {
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok {
		return ROCmModelTokenizerRoute{}, false
	}
	route := profile.TokenizerRoute
	if !route.Matched() {
		route = ROCmModelTokenizerRouteForProfile(profile)
	}
	if inspection != nil {
		route = route.WithTokenizerIdentity(inspection.Tokenizer, inspection.Labels)
	}
	return route.Clone(), true
}

func rocmApplyROCmModelTokenizerRouteLabels(labels map[string]string, route ROCmModelTokenizerRoute) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if !route.Matched() {
		return labels
	}
	for key, value := range rocmModelTokenizerRouteLabels(route) {
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}

func rocmApplyROCmModelTokenizerCapabilityLabels(labels map[string]string, model inference.ModelIdentity) map[string]string {
	if route, ok := ROCmModelTokenizerRouteForIdentity(model.Path, model); ok {
		return rocmApplyROCmModelTokenizerRouteLabels(labels, route)
	}
	if route, ok := ROCmModelTokenizerRouteForArchitecture(model.Architecture); ok {
		return rocmApplyROCmModelTokenizerRouteLabels(labels, route)
	}
	return labels
}

func rocmModelTokenizerRouteLabels(route ROCmModelTokenizerRoute) map[string]string {
	return rocmmodel.TokenizerRouteLabels(route)
}

func rocmTokenizerRouteCapabilities(chatTemplate bool) []inference.CapabilityID {
	return rocmmodel.TokenizerRouteCapabilities(chatTemplate)
}

func rocmTokenizerChatTemplateSource(chatTemplateID, fallback string) string {
	if fallback != "" {
		return fallback
	}
	if chatTemplateID != "" {
		return "registry"
	}
	return ""
}
