// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"strconv"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/registry"
	"dappco.re/go/inference/engine/hip/profile"
)

const (
	TokenizerRegistryContract = "rocm-model-tokenizer-registry-v1"

	TokenizerRouteName       = "model-tokenizer-route"
	TokenizerLoaderHFJSON    = "hf-tokenizer-json"
	TokenizerRuntimeHost     = "host"
	TokenizerRequiredSidecar = "tokenizer.json"
)

// TokenizerRoute is the folder-owned tokenizer/chat-template route catalogue.
// It mirrors the root API contract while remaining independent of root package
// types, so model families can self-register tokenizer behavior.
type TokenizerRoute struct {
	Contract               string                      `json:"contract,omitempty"`
	Name                   string                      `json:"name,omitempty"`
	Architecture           string                      `json:"architecture,omitempty"`
	Family                 string                      `json:"family,omitempty"`
	Loader                 string                      `json:"loader,omitempty"`
	Runtime                string                      `json:"runtime,omitempty"`
	TokenizerKind          string                      `json:"tokenizer_kind,omitempty"`
	TokenizerPath          string                      `json:"tokenizer_path,omitempty"`
	ConfigPath             string                      `json:"config_path,omitempty"`
	ChatTemplateID         string                      `json:"chat_template_id,omitempty"`
	ChatTemplateSource     string                      `json:"chat_template_source,omitempty"`
	ReasoningParserID      string                      `json:"reasoning_parser_id,omitempty"`
	ToolParserID           string                      `json:"tool_parser_id,omitempty"`
	GenerationRole         string                      `json:"generation_role,omitempty"`
	BOSID                  int32                       `json:"bos_id,omitempty"`
	EOSID                  int32                       `json:"eos_id,omitempty"`
	PADID                  int32                       `json:"pad_id,omitempty"`
	ThinkingChannel        bool                        `json:"thinking_channel,omitempty"`
	ThinkingChannelOpen    string                      `json:"thinking_channel_open,omitempty"`
	ThinkingChannelClose   string                      `json:"thinking_channel_close,omitempty"`
	ThinkingChannelOpenID  int32                       `json:"thinking_channel_open_id,omitempty"`
	ThinkingChannelCloseID int32                       `json:"thinking_channel_close_id,omitempty"`
	Registered             bool                        `json:"registered,omitempty"`
	NativeRuntime          bool                        `json:"native_runtime,omitempty"`
	SidecarTokenizer       bool                        `json:"sidecar_tokenizer,omitempty"`
	SidecarConfig          bool                        `json:"sidecar_config,omitempty"`
	ChatTemplate           bool                        `json:"chat_template,omitempty"`
	RequiresChatTemplate   bool                        `json:"requires_chat_template,omitempty"`
	ModelOwnedTemplate     bool                        `json:"model_owned_template,omitempty"`
	SidecarTemplate        bool                        `json:"sidecar_template,omitempty"`
	Generation             bool                        `json:"generation,omitempty"`
	Chat                   bool                        `json:"chat,omitempty"`
	RequiredFiles          []string                    `json:"required_files,omitempty"`
	OptionalFiles          []string                    `json:"optional_files,omitempty"`
	Capabilities           []inference.CapabilityID    `json:"capabilities,omitempty"`
	Tokenizer              inference.TokenizerIdentity `json:"tokenizer,omitempty"`
	Labels                 map[string]string           `json:"labels,omitempty"`
}

func (route TokenizerRoute) Matched() bool {
	return route.Contract != "" && route.Architecture != "" && route.Loader != ""
}

func (route TokenizerRoute) Clone() TokenizerRoute {
	route.RequiredFiles = append([]string(nil), route.RequiredFiles...)
	route.OptionalFiles = append([]string(nil), route.OptionalFiles...)
	route.Capabilities = append([]inference.CapabilityID(nil), route.Capabilities...)
	route.Tokenizer = cloneTokenizerIdentity(route.Tokenizer)
	route.Labels = cloneStringMap(route.Labels)
	return route
}

func (route TokenizerRoute) WithTokenizerIdentity(tokenizer inference.TokenizerIdentity, labels map[string]string) TokenizerRoute {
	route.Tokenizer = cloneTokenizerIdentity(tokenizer)
	route.TokenizerKind = firstNonEmpty(tokenizer.Kind, route.TokenizerKind)
	route.TokenizerPath = firstNonEmpty(tokenizer.Path, route.TokenizerPath)
	route.BOSID = firstNonZeroInt32(tokenizer.BOSID, route.BOSID)
	route.EOSID = firstNonZeroInt32(tokenizer.EOSID, route.EOSID)
	route.PADID = firstNonZeroInt32(tokenizer.PADID, route.PADID)
	route.ThinkingChannelOpenID = firstNonZeroInt32(labelInt32(labels["thinking_channel_open_id"]), labelInt32(labels["engine_tokenizer_thinking_channel_open_id"]), labelInt32(labels["gemma4_thinking_channel_open_id"]), route.ThinkingChannelOpenID)
	route.ThinkingChannelCloseID = firstNonZeroInt32(labelInt32(labels["thinking_channel_close_id"]), labelInt32(labels["engine_tokenizer_thinking_channel_close_id"]), labelInt32(labels["gemma4_thinking_channel_close_id"]), route.ThinkingChannelCloseID)
	route.ThinkingChannel = route.ThinkingChannel ||
		(route.ThinkingChannelOpen != "" && route.ThinkingChannelClose != "") ||
		(route.ThinkingChannelOpenID != 0 && route.ThinkingChannelCloseID != 0)
	route.SidecarTokenizer = labels["tokenizer_json"] == "present" || route.TokenizerPath != ""
	route.SidecarConfig = labels["tokenizer_config"] == "present"
	route.SidecarTemplate = tokenizer.ChatTemplate != "" && tokenizer.ChatTemplate != route.ChatTemplateID
	route.ChatTemplate = route.ChatTemplate || tokenizer.ChatTemplate != ""
	if route.SidecarTemplate {
		route.ChatTemplateSource = "sidecar"
	} else {
		route.ChatTemplateSource = tokenizerChatTemplateSource(route.ChatTemplateID, route.ChatTemplateSource)
	}
	if route.Tokenizer.ChatTemplate == "" {
		route.Tokenizer.ChatTemplate = route.ChatTemplateID
	}
	if route.Tokenizer.Kind == "" {
		route.Tokenizer.Kind = route.TokenizerKind
	}
	if route.Tokenizer.Path == "" {
		route.Tokenizer.Path = route.TokenizerPath
	}
	route.ThinkingChannel = route.ThinkingChannel ||
		(route.ThinkingChannelOpen != "" && route.ThinkingChannelClose != "") ||
		(route.ThinkingChannelOpenID != 0 && route.ThinkingChannelCloseID != 0)
	route.Capabilities = tokenizerRouteCapabilities(route.ChatTemplate)
	route.Labels = tokenizerRouteLabels(route)
	return route.Clone()
}

var registeredTokenizers = registry.NewOrdered[string, TokenizerRoute]()

// RegisterTokenizerRoute registers or replaces tokenizer metadata by
// architecture.
func RegisterTokenizerRoute(route TokenizerRoute) {
	route = NormalizeTokenizerRoute(route)
	if !route.Matched() {
		return
	}
	registeredTokenizers.Put(route.Architecture, route)
}

func RegisteredTokenizerArchitectures() []string {
	return registeredTokenizers.Keys()
}

func RegisteredTokenizerRoutes() []TokenizerRoute {
	return registeredTokenizerSnapshot()
}

func ReplaceRegisteredTokenizerRoutes(routes []TokenizerRoute) {
	order := make([]string, 0, len(routes))
	values := make(map[string]TokenizerRoute, len(routes))
	for _, route := range routes {
		route = NormalizeTokenizerRoute(route)
		if !route.Matched() {
			continue
		}
		if _, ok := values[route.Architecture]; !ok {
			order = append(order, route.Architecture)
		}
		values[route.Architecture] = route
	}
	registeredTokenizers.Restore(order, values)
}

func RegisteredTokenizerRouteForArchitecture(architecture string) (TokenizerRoute, bool) {
	return registeredTokenizerForArchitecture(architecture)
}

func TokenizerRouteForArchitecture(architecture string) (TokenizerRoute, bool) {
	architecture = profile.ArchitectureID(architecture)
	if architecture == "" {
		return TokenizerRoute{}, false
	}
	if route, ok := registeredTokenizerForArchitecture(architecture); ok {
		return route, true
	}
	architectureProfile, ok := profile.LookupArchitectureProfile(architecture)
	if !ok {
		return TokenizerRoute{}, false
	}
	return tokenizerRouteForProfile(architectureProfile), true
}

func TokenizerRouteForIdentity(path string, identity inference.ModelIdentity) (TokenizerRoute, bool) {
	if identity.Path == "" {
		identity.Path = path
	}
	architecture := firstNonEmpty(
		identity.Labels["engine_architecture_resolved"],
		identity.Labels["architecture_resolved"],
		identity.Architecture,
	)
	return TokenizerRouteForArchitecture(architecture)
}

func TokenizerRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (TokenizerRoute, bool) {
	return TokenizerRouteForIdentity(path, inference.ModelIdentity{
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

func TokenizerRouteForInspection(inspection *inference.ModelPackInspection) (TokenizerRoute, bool) {
	if inspection == nil {
		return TokenizerRoute{}, false
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
	route, ok := TokenizerRouteForIdentity(identity.Path, identity)
	if !ok {
		return TokenizerRoute{}, false
	}
	return route.WithTokenizerIdentity(inspection.Tokenizer, inspection.Labels), true
}

func DefaultTokenizerRoutes() []TokenizerRoute {
	profiles := profile.ArchitectureProfiles()
	routes := make([]TokenizerRoute, 0, len(profiles)+len(registeredTokenizers.Keys()))
	seen := map[string]int{}
	for _, architectureProfile := range profiles {
		route := tokenizerRouteForProfile(architectureProfile)
		if !route.Matched() {
			continue
		}
		seen[route.Architecture] = len(routes)
		routes = append(routes, route)
	}
	for _, route := range registeredTokenizerSnapshot() {
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
	return cloneTokenizerRoutes(routes)
}

func NormalizeTokenizerRoute(route TokenizerRoute) TokenizerRoute {
	route.Architecture = profile.ArchitectureID(route.Architecture)
	if route.Architecture == "" {
		return TokenizerRoute{}
	}
	architectureProfile, hasProfile := profile.LookupArchitectureProfile(route.Architecture)
	if route.Contract == "" {
		route.Contract = TokenizerRegistryContract
	}
	if route.Name == "" {
		route.Name = TokenizerRouteName
	}
	if route.Loader == "" {
		route.Loader = TokenizerLoaderHFJSON
	}
	if route.Runtime == "" {
		route.Runtime = TokenizerRuntimeHost
	}
	if route.Family == "" && hasProfile {
		route.Family = firstNonEmpty(architectureProfile.Family, architectureProfile.ID)
	}
	if route.Family == "" {
		route.Family = route.Architecture
	}
	if route.TokenizerKind == "" {
		route.TokenizerKind = route.Tokenizer.Kind
	}
	if route.TokenizerKind == "" && hasProfile {
		route.TokenizerKind = profile.ArchitectureProfileTokenizerKindForProfile(architectureProfile)
	}
	if route.ChatTemplateID == "" {
		route.ChatTemplateID = route.Tokenizer.ChatTemplate
	}
	if route.ChatTemplateID == "" && hasProfile {
		route.ChatTemplateID = architectureProfile.ChatTemplate
	}
	if route.ReasoningParserID == "" && hasProfile {
		route.ReasoningParserID = architectureProfile.ParserID
	}
	if route.ToolParserID == "" && hasProfile {
		route.ToolParserID = architectureProfile.ToolParserID
	}
	if route.GenerationRole == "" && hasProfile {
		route.GenerationRole = architectureProfile.GenerationRole
	}
	if route.ChatTemplateSource == "" {
		route.ChatTemplateSource = tokenizerChatTemplateSource(route.ChatTemplateID, "")
	}
	if route.Tokenizer.Kind == "" {
		route.Tokenizer.Kind = route.TokenizerKind
	}
	if route.Tokenizer.ChatTemplate == "" {
		route.Tokenizer.ChatTemplate = route.ChatTemplateID
	}
	if route.Tokenizer.Path == "" {
		route.Tokenizer.Path = route.TokenizerPath
	}
	route.ThinkingChannel = route.ThinkingChannel ||
		(route.ThinkingChannelOpen != "" && route.ThinkingChannelClose != "") ||
		(route.ThinkingChannelOpenID != 0 && route.ThinkingChannelCloseID != 0)
	route.Registered = true
	if hasProfile {
		route.NativeRuntime = route.NativeRuntime || architectureProfile.NativeRuntime
		route.RequiresChatTemplate = route.RequiresChatTemplate || architectureProfile.RequiresChatTemplate
		route.Generation = route.Generation || architectureProfile.Generation
		route.Chat = route.Chat || architectureProfile.Chat
	}
	route.ChatTemplate = route.ChatTemplate || route.ChatTemplateID != ""
	route.ModelOwnedTemplate = route.ModelOwnedTemplate || (route.ChatTemplateID != "" && !route.SidecarTemplate)
	if len(route.RequiredFiles) == 0 {
		route.RequiredFiles = []string{TokenizerRequiredSidecar}
	}
	if len(route.OptionalFiles) == 0 {
		route.OptionalFiles = []string{"tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json", "generation_config.json"}
	}
	if len(route.Capabilities) == 0 {
		route.Capabilities = tokenizerRouteCapabilities(route.ChatTemplate)
	}
	route.Labels = tokenizerRouteLabels(route)
	return route.Clone()
}

func tokenizerRouteForProfile(architectureProfile profile.ArchitectureProfile) TokenizerRoute {
	architectureProfile = profile.NormalizeArchitectureProfile(architectureProfile)
	route := TokenizerRoute{
		Contract:             TokenizerRegistryContract,
		Name:                 TokenizerRouteName,
		Architecture:         architectureProfile.ID,
		Family:               firstNonEmpty(architectureProfile.Family, architectureProfile.ID),
		Loader:               TokenizerLoaderHFJSON,
		Runtime:              TokenizerRuntimeHost,
		TokenizerKind:        profile.ArchitectureProfileTokenizerKindForProfile(architectureProfile),
		ChatTemplateID:       architectureProfile.ChatTemplate,
		ChatTemplateSource:   tokenizerChatTemplateSource(architectureProfile.ChatTemplate, ""),
		ReasoningParserID:    architectureProfile.ParserID,
		ToolParserID:         architectureProfile.ToolParserID,
		GenerationRole:       architectureProfile.GenerationRole,
		Registered:           architectureProfile.ID != "",
		NativeRuntime:        architectureProfile.NativeRuntime,
		ChatTemplate:         architectureProfile.ChatTemplate != "",
		RequiresChatTemplate: architectureProfile.RequiresChatTemplate,
		ModelOwnedTemplate:   architectureProfile.ChatTemplate != "",
		Generation:           architectureProfile.Generation,
		Chat:                 architectureProfile.Chat,
		RequiredFiles:        []string{TokenizerRequiredSidecar},
		OptionalFiles:        []string{"tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json", "generation_config.json"},
		Capabilities:         tokenizerRouteCapabilities(architectureProfile.ChatTemplate != ""),
	}
	route.Tokenizer = inference.TokenizerIdentity{
		Kind:         route.TokenizerKind,
		ChatTemplate: route.ChatTemplateID,
	}
	route.Labels = tokenizerRouteLabels(route)
	return route.Clone()
}

func registeredTokenizerForArchitecture(architecture string) (TokenizerRoute, bool) {
	route, ok := registeredTokenizers.Get(profile.ArchitectureID(architecture))
	if !ok {
		return TokenizerRoute{}, false
	}
	return route.Clone(), true
}

func registeredTokenizerSnapshot() []TokenizerRoute {
	routes := registeredTokenizers.Values()
	out := make([]TokenizerRoute, 0, len(routes))
	for _, route := range routes {
		out = append(out, route.Clone())
	}
	return out
}

func tokenizerRouteLabels(route TokenizerRoute) map[string]string {
	if !route.Matched() {
		return nil
	}
	labels := map[string]string{
		"engine_tokenizer_route_contract":         route.Contract,
		"engine_tokenizer_route":                  route.Name,
		"engine_tokenizer_loader":                 route.Loader,
		"engine_tokenizer_runtime":                route.Runtime,
		"engine_tokenizer_registered":             strconv.FormatBool(route.Registered),
		"engine_tokenizer_native_runtime":         strconv.FormatBool(route.NativeRuntime),
		"engine_tokenizer_sidecar":                strconv.FormatBool(route.SidecarTokenizer),
		"engine_tokenizer_config_sidecar":         strconv.FormatBool(route.SidecarConfig),
		"engine_tokenizer_chat_template":          strconv.FormatBool(route.ChatTemplate),
		"engine_tokenizer_requires_chat_template": strconv.FormatBool(route.RequiresChatTemplate),
		"engine_tokenizer_model_owned_template":   strconv.FormatBool(route.ModelOwnedTemplate),
		"engine_tokenizer_sidecar_template":       strconv.FormatBool(route.SidecarTemplate),
		"engine_tokenizer_generation":             strconv.FormatBool(route.Generation),
		"engine_tokenizer_chat":                   strconv.FormatBool(route.Chat),
		"engine_tokenizer_required_files":         joinNonEmptyStrings(route.RequiredFiles, ","),
		"engine_tokenizer_optional_files":         joinNonEmptyStrings(route.OptionalFiles, ","),
	}
	if route.Architecture != "" {
		labels["engine_tokenizer_architecture"] = route.Architecture
	}
	if route.Family != "" {
		labels["engine_tokenizer_family"] = route.Family
	}
	if route.TokenizerKind != "" {
		labels["engine_tokenizer_kind"] = route.TokenizerKind
	}
	if route.TokenizerPath != "" {
		labels["engine_tokenizer_path"] = route.TokenizerPath
	}
	if route.ConfigPath != "" {
		labels["engine_tokenizer_config_path"] = route.ConfigPath
	}
	if route.ChatTemplateID != "" {
		labels["engine_tokenizer_chat_template_id"] = route.ChatTemplateID
	}
	if route.ChatTemplateSource != "" {
		labels["engine_tokenizer_chat_template_source"] = route.ChatTemplateSource
	}
	if route.ReasoningParserID != "" {
		labels["engine_tokenizer_reasoning_parser"] = route.ReasoningParserID
	}
	if route.ToolParserID != "" {
		labels["engine_tokenizer_tool_parser"] = route.ToolParserID
	}
	if route.GenerationRole != "" {
		labels["engine_tokenizer_generation_role"] = route.GenerationRole
	}
	if route.BOSID != 0 {
		labels["engine_tokenizer_bos_id"] = strconv.FormatInt(int64(route.BOSID), 10)
	}
	if route.EOSID != 0 {
		labels["engine_tokenizer_eos_id"] = strconv.FormatInt(int64(route.EOSID), 10)
	}
	if route.PADID != 0 {
		labels["engine_tokenizer_pad_id"] = strconv.FormatInt(int64(route.PADID), 10)
	}
	if route.ThinkingChannel {
		labels["engine_tokenizer_thinking_channel"] = "true"
	}
	if route.ThinkingChannelOpen != "" {
		labels["engine_tokenizer_thinking_channel_open"] = route.ThinkingChannelOpen
	}
	if route.ThinkingChannelClose != "" {
		labels["engine_tokenizer_thinking_channel_close"] = route.ThinkingChannelClose
	}
	if route.ThinkingChannelOpenID != 0 {
		labels["engine_tokenizer_thinking_channel_open_id"] = strconv.FormatInt(int64(route.ThinkingChannelOpenID), 10)
	}
	if route.ThinkingChannelCloseID != 0 {
		labels["engine_tokenizer_thinking_channel_close_id"] = strconv.FormatInt(int64(route.ThinkingChannelCloseID), 10)
	}
	if len(route.Capabilities) > 0 {
		labels["engine_tokenizer_capabilities"] = capabilityIDsCSV(route.Capabilities)
	}
	return labels
}

// TokenizerRouteLabels returns labels for a tokenizer route using the
// model-owned tokenizer registry contract.
func TokenizerRouteLabels(route TokenizerRoute) map[string]string {
	return cloneStringMap(tokenizerRouteLabels(route))
}

func tokenizerRouteCapabilities(chatTemplate bool) []inference.CapabilityID {
	capabilities := []inference.CapabilityID{inference.CapabilityTokenizer}
	if chatTemplate {
		capabilities = append(capabilities, inference.CapabilityChatTemplate)
	}
	return capabilities
}

// TokenizerRouteCapabilities returns capability IDs implied by tokenizer route
// metadata using the model-owned tokenizer contract.
func TokenizerRouteCapabilities(chatTemplate bool) []inference.CapabilityID {
	return append([]inference.CapabilityID(nil), tokenizerRouteCapabilities(chatTemplate)...)
}

func tokenizerChatTemplateSource(chatTemplateID, fallback string) string {
	if fallback != "" {
		return fallback
	}
	if chatTemplateID != "" {
		return "registry"
	}
	return ""
}

func cloneTokenizerIdentity(identity inference.TokenizerIdentity) inference.TokenizerIdentity {
	identity.Labels = cloneStringMap(identity.Labels)
	return identity
}

func cloneTokenizerRoutes(routes []TokenizerRoute) []TokenizerRoute {
	out := append([]TokenizerRoute(nil), routes...)
	for i := range out {
		out[i] = out[i].Clone()
	}
	return out
}

func firstNonZeroInt32(values ...int32) int32 {
	for _, value := range values {
		if value != 0 {
			return value
		}
	}
	return 0
}

func labelInt32(value string) int32 {
	parsed, _ := strconv.ParseInt(value, 10, 32)
	return int32(parsed)
}

func joinNonEmptyStrings(values []string, sep string) string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value != "" {
			out = append(out, value)
		}
	}
	if len(out) == 0 {
		return ""
	}
	result := out[0]
	for _, value := range out[1:] {
		result += sep + value
	}
	return result
}
