// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"strconv"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/registry"
	"dappco.re/go/inference/engine/hip/profile"
)

const (
	FeatureRegistryContract = "rocm-model-feature-registry-v1"

	FeatureRouteName = "model-feature-route"
)

// FeatureRoute is the folder-owned parser/template/capability route catalogue.
// It lets model-family packages advertise engine features without importing the
// root rocm package or extending central switches.
type FeatureRoute struct {
	Contract             string                         `json:"contract,omitempty"`
	Name                 string                         `json:"name,omitempty"`
	Architecture         string                         `json:"architecture,omitempty"`
	Family               string                         `json:"family,omitempty"`
	RuntimeStatus        inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	ReasoningParserID    string                         `json:"reasoning_parser_id,omitempty"`
	ToolParserID         string                         `json:"tool_parser_id,omitempty"`
	ChatTemplateID       string                         `json:"chat_template_id,omitempty"`
	GenerationRole       string                         `json:"generation_role,omitempty"`
	Registered           bool                           `json:"registered,omitempty"`
	NativeRuntime        bool                           `json:"native_runtime,omitempty"`
	Generation           bool                           `json:"generation,omitempty"`
	TextGenerate         bool                           `json:"text_generate,omitempty"`
	Chat                 bool                           `json:"chat,omitempty"`
	ModelContextWindow   bool                           `json:"model_context_window,omitempty"`
	ReasoningParse       bool                           `json:"reasoning_parse,omitempty"`
	ToolParse            bool                           `json:"tool_parse,omitempty"`
	ChatTemplate         bool                           `json:"chat_template,omitempty"`
	DefaultThinking      bool                           `json:"default_thinking,omitempty"`
	RequiresChatTemplate bool                           `json:"requires_chat_template,omitempty"`
	Embeddings           bool                           `json:"embeddings,omitempty"`
	Rerank               bool                           `json:"rerank,omitempty"`
	MoE                  bool                           `json:"moe,omitempty"`
	SequenceMixer        bool                           `json:"sequence_mixer,omitempty"`
	AttachedOnly         bool                           `json:"attached_only,omitempty"`
	Capabilities         []inference.CapabilityID       `json:"capabilities,omitempty"`
	Labels               map[string]string              `json:"labels,omitempty"`
}

func (route FeatureRoute) Matched() bool {
	return route.Contract != "" && route.Architecture != "" && route.Name != ""
}

func (route FeatureRoute) Clone() FeatureRoute {
	route.Capabilities = append([]inference.CapabilityID(nil), route.Capabilities...)
	route.Labels = cloneStringMap(route.Labels)
	return route
}

var registeredFeatures = registry.NewOrdered[string, FeatureRoute]()

// RegisterFeatureRoute registers or replaces feature metadata by architecture.
func RegisterFeatureRoute(route FeatureRoute) {
	route = NormalizeFeatureRoute(route)
	if !route.Matched() {
		return
	}
	registeredFeatures.Put(route.Architecture, route)
}

func RegisteredFeatureArchitectures() []string {
	return registeredFeatures.Keys()
}

func RegisteredFeatureRoutes() []FeatureRoute {
	return registeredFeatureSnapshot()
}

func ReplaceRegisteredFeatureRoutes(routes []FeatureRoute) {
	order := make([]string, 0, len(routes))
	values := make(map[string]FeatureRoute, len(routes))
	for _, route := range routes {
		route = NormalizeFeatureRoute(route)
		if !route.Matched() {
			continue
		}
		if _, ok := values[route.Architecture]; !ok {
			order = append(order, route.Architecture)
		}
		values[route.Architecture] = route
	}
	registeredFeatures.Restore(order, values)
}

func RegisteredFeatureRouteForArchitecture(architecture string) (FeatureRoute, bool) {
	return registeredFeatureForArchitecture(architecture)
}

func FeatureRouteForArchitecture(architecture string) (FeatureRoute, bool) {
	architecture = profile.ArchitectureID(architecture)
	if architecture == "" {
		return FeatureRoute{}, false
	}
	if route, ok := registeredFeatureForArchitecture(architecture); ok {
		return route, true
	}
	architectureProfile, ok := profile.LookupArchitectureProfile(architecture)
	if !ok {
		return FeatureRoute{}, false
	}
	return featureRouteForProfile(architectureProfile), true
}

func FeatureRouteForIdentity(path string, identity inference.ModelIdentity) (FeatureRoute, bool) {
	if identity.Path == "" {
		identity.Path = path
	}
	architecture := firstNonEmpty(
		identity.Labels["engine_architecture_resolved"],
		identity.Labels["architecture_resolved"],
		identity.Architecture,
	)
	return FeatureRouteForArchitecture(architecture)
}

func FeatureRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (FeatureRoute, bool) {
	return FeatureRouteForIdentity(path, inference.ModelIdentity{
		Path:         path,
		Architecture: info.Architecture,
		VocabSize:    info.VocabSize,
		NumLayers:    info.NumLayers,
		HiddenSize:   info.HiddenSize,
		QuantBits:    info.QuantBits,
		QuantGroup:   info.QuantGroup,
		Labels:       cloneStringMap(labels),
	})
}

func FeatureRouteForInspection(inspection *inference.ModelPackInspection) (FeatureRoute, bool) {
	if inspection == nil {
		return FeatureRoute{}, false
	}
	identity := inspection.Model
	if identity.Path == "" {
		identity.Path = inspection.Path
	}
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
	return FeatureRouteForIdentity(identity.Path, identity)
}

func DefaultFeatureRoutes() []FeatureRoute {
	profiles := profile.ArchitectureProfiles()
	routes := make([]FeatureRoute, 0, len(profiles)+len(registeredFeatures.Keys()))
	seen := map[string]int{}
	for _, architectureProfile := range profiles {
		route := featureRouteForProfile(architectureProfile)
		if !route.Matched() {
			continue
		}
		seen[route.Architecture] = len(routes)
		routes = append(routes, route)
	}
	for _, route := range registeredFeatureSnapshot() {
		if !route.Matched() {
			continue
		}
		if index, ok := seen[route.Architecture]; ok {
			routes[index] = route.Clone()
			continue
		}
		seen[route.Architecture] = len(routes)
		routes = append(routes, route.Clone())
	}
	return cloneFeatureRoutes(routes)
}

func NormalizeFeatureRoute(route FeatureRoute) FeatureRoute {
	route.Architecture = profile.ArchitectureID(route.Architecture)
	if route.Architecture == "" {
		return FeatureRoute{}
	}
	architectureProfile, hasProfile := profile.LookupArchitectureProfile(route.Architecture)
	if route.Contract == "" {
		route.Contract = FeatureRegistryContract
	}
	if route.Name == "" {
		route.Name = FeatureRouteName
	}
	if route.Family == "" && hasProfile {
		route.Family = firstNonEmpty(architectureProfile.Family, architectureProfile.ID)
	}
	if route.Family == "" {
		route.Family = route.Architecture
	}
	if route.RuntimeStatus == "" && hasProfile {
		route.RuntimeStatus = architectureProfile.RuntimeStatus
	}
	if route.RuntimeStatus == "" && route.NativeRuntime {
		route.RuntimeStatus = inference.FeatureRuntimeNative
	}
	if route.ReasoningParserID == "" && hasProfile {
		route.ReasoningParserID = architectureProfile.ParserID
	}
	if route.ToolParserID == "" && hasProfile {
		route.ToolParserID = architectureProfile.ToolParserID
	}
	if route.ChatTemplateID == "" && hasProfile {
		route.ChatTemplateID = architectureProfile.ChatTemplate
	}
	if route.GenerationRole == "" && hasProfile {
		route.GenerationRole = architectureProfile.GenerationRole
	}
	route.Registered = true
	if hasProfile {
		route.NativeRuntime = route.NativeRuntime || architectureProfile.NativeRuntime
		route.Generation = route.Generation || architectureProfile.Generation
		route.Chat = route.Chat || architectureProfile.Chat
		route.DefaultThinking = route.DefaultThinking || architectureProfile.DefaultThinking
		route.RequiresChatTemplate = route.RequiresChatTemplate || architectureProfile.RequiresChatTemplate
		route.Embeddings = route.Embeddings || architectureProfile.Embeddings
		route.Rerank = route.Rerank || architectureProfile.Rerank
		route.MoE = route.MoE || architectureProfile.MoE
		route.SequenceMixer = route.SequenceMixer || featureProfileDeclaresSequenceMixer(architectureProfile)
		route.AttachedOnly = route.AttachedOnly || architectureProfile.AttachedOnly
	}
	route.ReasoningParse = route.ReasoningParse || route.ReasoningParserID != ""
	route.ToolParse = route.ToolParse || route.ToolParserID != ""
	route.ChatTemplate = route.ChatTemplate || route.ChatTemplateID != ""
	if route.Generation && route.NativeRuntime && !route.AttachedOnly {
		route.TextGenerate = true
	}
	if route.Generation && !route.AttachedOnly {
		route.ModelContextWindow = true
	}
	route.Capabilities = mergeFeatureCapabilityIDs(featureRouteCapabilities(route), route.Capabilities)
	route.Labels = featureRouteLabels(route)
	return route.Clone()
}

func featureRouteForProfile(architectureProfile profile.ArchitectureProfile) FeatureRoute {
	architectureProfile = profile.NormalizeArchitectureProfile(architectureProfile)
	route := FeatureRoute{
		Contract:             FeatureRegistryContract,
		Name:                 FeatureRouteName,
		Architecture:         architectureProfile.ID,
		Family:               firstNonEmpty(architectureProfile.Family, architectureProfile.ID),
		RuntimeStatus:        architectureProfile.RuntimeStatus,
		ReasoningParserID:    architectureProfile.ParserID,
		ToolParserID:         architectureProfile.ToolParserID,
		ChatTemplateID:       architectureProfile.ChatTemplate,
		GenerationRole:       architectureProfile.GenerationRole,
		Registered:           architectureProfile.ID != "",
		NativeRuntime:        architectureProfile.NativeRuntime,
		Generation:           architectureProfile.Generation,
		TextGenerate:         architectureProfile.NativeRuntime && architectureProfile.Generation && !architectureProfile.AttachedOnly,
		Chat:                 architectureProfile.Chat,
		ModelContextWindow:   architectureProfile.Generation && !architectureProfile.AttachedOnly,
		ReasoningParse:       architectureProfile.ParserID != "",
		ToolParse:            architectureProfile.ToolParserID != "",
		ChatTemplate:         architectureProfile.ChatTemplate != "",
		DefaultThinking:      architectureProfile.DefaultThinking,
		RequiresChatTemplate: architectureProfile.RequiresChatTemplate,
		Embeddings:           architectureProfile.Embeddings,
		Rerank:               architectureProfile.Rerank,
		MoE:                  architectureProfile.MoE,
		SequenceMixer:        featureProfileDeclaresSequenceMixer(architectureProfile),
		AttachedOnly:         architectureProfile.AttachedOnly,
	}
	route.Capabilities = featureRouteCapabilities(route)
	route.Labels = featureRouteLabels(route)
	return route.Clone()
}

func registeredFeatureForArchitecture(architecture string) (FeatureRoute, bool) {
	route, ok := registeredFeatures.Get(profile.ArchitectureID(architecture))
	if !ok {
		return FeatureRoute{}, false
	}
	return route.Clone(), true
}

func registeredFeatureSnapshot() []FeatureRoute {
	routes := registeredFeatures.Values()
	out := make([]FeatureRoute, 0, len(routes))
	for _, route := range routes {
		out = append(out, route.Clone())
	}
	return out
}

func featureRouteLabels(route FeatureRoute) map[string]string {
	if !route.Matched() {
		return nil
	}
	labels := map[string]string{
		"engine_feature_route_contract":               route.Contract,
		"engine_feature_route":                        route.Name,
		"engine_feature_route_registered":             strconv.FormatBool(route.Registered),
		"engine_feature_route_native_runtime":         strconv.FormatBool(route.NativeRuntime),
		"engine_feature_route_generation":             strconv.FormatBool(route.Generation),
		"engine_feature_route_text_generate":          strconv.FormatBool(route.TextGenerate),
		"engine_feature_route_chat":                   strconv.FormatBool(route.Chat),
		"engine_feature_route_model_context_window":   strconv.FormatBool(route.ModelContextWindow),
		"engine_feature_route_reasoning_parse":        strconv.FormatBool(route.ReasoningParse),
		"engine_feature_route_tool_parse":             strconv.FormatBool(route.ToolParse),
		"engine_feature_route_chat_template":          strconv.FormatBool(route.ChatTemplate),
		"engine_feature_route_default_thinking":       strconv.FormatBool(route.DefaultThinking),
		"engine_feature_route_requires_chat_template": strconv.FormatBool(route.RequiresChatTemplate),
		"engine_feature_route_embeddings":             strconv.FormatBool(route.Embeddings),
		"engine_feature_route_rerank":                 strconv.FormatBool(route.Rerank),
		"engine_feature_route_moe":                    strconv.FormatBool(route.MoE),
		"engine_feature_route_sequence_mixer":         strconv.FormatBool(route.SequenceMixer),
		"engine_feature_route_attached_only":          strconv.FormatBool(route.AttachedOnly),
	}
	if route.Architecture != "" {
		labels["engine_feature_route_architecture"] = route.Architecture
	}
	if route.Family != "" {
		labels["engine_feature_route_family"] = route.Family
	}
	if route.RuntimeStatus != "" {
		labels["engine_feature_route_runtime_status"] = string(route.RuntimeStatus)
	}
	if route.ReasoningParserID != "" {
		labels["engine_feature_route_reasoning_parser"] = route.ReasoningParserID
	}
	if route.ToolParserID != "" {
		labels["engine_feature_route_tool_parser"] = route.ToolParserID
	}
	if route.ChatTemplateID != "" {
		labels["engine_feature_route_chat_template_id"] = route.ChatTemplateID
	}
	if route.GenerationRole != "" {
		labels["engine_feature_route_generation_role"] = route.GenerationRole
	}
	if len(route.Capabilities) > 0 {
		labels["engine_feature_route_capabilities"] = capabilityIDsCSV(route.Capabilities)
	}
	return labels
}

// FeatureRouteLabels returns the labels for a feature route using the
// model-owned registry contract.
func FeatureRouteLabels(route FeatureRoute) map[string]string {
	return cloneStringMap(featureRouteLabels(route))
}

func featureRouteCapabilities(route FeatureRoute) []inference.CapabilityID {
	capabilities := make([]inference.CapabilityID, 0, 6)
	add := func(id inference.CapabilityID, enabled bool) {
		if enabled {
			capabilities = append(capabilities, id)
		}
	}
	add(inference.CapabilityGenerate, route.TextGenerate)
	add(inference.CapabilityChatTemplate, route.ChatTemplate)
	add(inference.CapabilityEmbeddings, route.Embeddings)
	add(inference.CapabilityRerank, route.Rerank)
	add(inference.CapabilityReasoningParse, route.ReasoningParse)
	add(inference.CapabilityToolParse, route.ToolParse)
	return capabilities
}

// FeatureRouteCapabilities returns the capability IDs implied by a feature
// route using the model-owned capability contract.
func FeatureRouteCapabilities(route FeatureRoute) []inference.CapabilityID {
	return append([]inference.CapabilityID(nil), featureRouteCapabilities(route)...)
}

func featureProfileDeclaresSequenceMixer(architectureProfile profile.ArchitectureProfile) bool {
	architecture := firstNonEmpty(architectureProfile.ID, architectureProfile.Family)
	if architecture == "composed" || architecture == "hybrid" {
		return true
	}
	return architectureProfile.Family == "composed" || architectureProfile.Family == "hybrid"
}

func mergeFeatureCapabilityIDs(primary, secondary []inference.CapabilityID) []inference.CapabilityID {
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

func capabilityIDsCSV(ids []inference.CapabilityID) string {
	out := ""
	for _, id := range ids {
		if id == "" {
			continue
		}
		if out != "" {
			out += ","
		}
		out += string(id)
	}
	return out
}

func cloneFeatureRoutes(routes []FeatureRoute) []FeatureRoute {
	out := append([]FeatureRoute(nil), routes...)
	for i := range out {
		out[i] = out[i].Clone()
	}
	return out
}
