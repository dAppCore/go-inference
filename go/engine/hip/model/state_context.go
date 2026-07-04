// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"strconv"
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/registry"
	"dappco.re/go/inference/engine/hip/profile"
)

const (
	StateContextRegistryContract = "rocm-state-context-registry-v1"

	StateContextRouteName       = "state-context-route"
	StateContextRuntimeAPI      = "runtime-api"
	StateContextRuntimeMetadata = "metadata"
)

type StateContextRouteStatus string

const (
	StateContextRouteExperimentalRuntime StateContextRouteStatus = "experimental_runtime"
	StateContextRouteAttachedRuntime     StateContextRouteStatus = "attached_runtime"
	StateContextRoutePlannedMetadata     StateContextRouteStatus = "planned_metadata"
)

// StateContextRoute is the folder-owned context and retained-state route. It
// exposes model-declared KV/state lifecycle behavior without importing the root
// rocm package or binding callers to a concrete runtime implementation.
type StateContextRoute struct {
	Contract                string                         `json:"contract,omitempty"`
	Name                    string                         `json:"name,omitempty"`
	Architecture            string                         `json:"architecture,omitempty"`
	Family                  string                         `json:"family,omitempty"`
	Runtime                 string                         `json:"runtime,omitempty"`
	RuntimeStatus           inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	Status                  StateContextRouteStatus        `json:"status,omitempty"`
	Reference               string                         `json:"reference,omitempty"`
	Registered              bool                           `json:"registered,omitempty"`
	NativeRuntime           bool                           `json:"native_runtime,omitempty"`
	AttachedOnly            bool                           `json:"attached_only,omitempty"`
	StateSession            bool                           `json:"state_session,omitempty"`
	SleepState              bool                           `json:"sleep_state,omitempty"`
	WakeState               bool                           `json:"wake_state,omitempty"`
	ForkState               bool                           `json:"fork_state,omitempty"`
	CaptureState            bool                           `json:"capture_state,omitempty"`
	RestoreState            bool                           `json:"restore_state,omitempty"`
	ResetState              bool                           `json:"reset_state,omitempty"`
	RuntimeOwnedKV          bool                           `json:"runtime_owned_kv,omitempty"`
	PromptReplayRefused     bool                           `json:"prompt_replay_refused,omitempty"`
	RemainingContextDefault bool                           `json:"remaining_context_default,omitempty"`
	ModelContextWindow      bool                           `json:"model_context_window,omitempty"`
	DeviceKVState           bool                           `json:"device_kv_state,omitempty"`
	HIPDeviceMirror         bool                           `json:"hip_device_mirror,omitempty"`
	PackageLocalKV          bool                           `json:"package_local_kv,omitempty"`
	BlockBundleRefs         bool                           `json:"block_bundle_refs,omitempty"`
	PortableRefs            bool                           `json:"portable_refs,omitempty"`
	RetainedStateRequired   bool                           `json:"retained_state_required,omitempty"`
	AttachedDrafterState    bool                           `json:"attached_drafter_state,omitempty"`
	Staged                  bool                           `json:"staged,omitempty"`
	Planned                 bool                           `json:"planned,omitempty"`
	ContextWindow           int                            `json:"context_window,omitempty"`
	DefaultContextWindow    int                            `json:"default_context_window,omitempty"`
	DefaultStateBlockSize   int                            `json:"default_state_block_size,omitempty"`
	DefaultDeviceKVMode     string                         `json:"default_device_kv_mode,omitempty"`
	Gemma4Size              string                         `json:"gemma4_size,omitempty"`
	Gemma4QuantMode         string                         `json:"gemma4_quant_mode,omitempty"`
	CacheModes              []string                       `json:"cache_modes,omitempty"`
	StateBackends           []string                       `json:"state_backends,omitempty"`
	Capabilities            []inference.CapabilityID       `json:"capabilities,omitempty"`
	Labels                  map[string]string              `json:"labels,omitempty"`
}

func (route StateContextRoute) Matched() bool {
	return route.Contract != "" && route.Name != "" && route.Architecture != "" && route.StateSession
}

func (route StateContextRoute) Clone() StateContextRoute {
	route.CacheModes = append([]string(nil), route.CacheModes...)
	route.StateBackends = append([]string(nil), route.StateBackends...)
	route.Capabilities = append([]inference.CapabilityID(nil), route.Capabilities...)
	route.Labels = cloneStringMap(route.Labels)
	return route
}

func (route StateContextRoute) WithLabels(labels map[string]string) StateContextRoute {
	route = route.withLabels(labels)
	route.finalize()
	return route.Clone()
}

var registeredStateContexts = registry.NewOrdered[string, StateContextRoute]()

// RegisterStateContextRoute registers or replaces state/context metadata by
// architecture.
func RegisterStateContextRoute(route StateContextRoute) {
	route = NormalizeStateContextRoute(route)
	if !route.Matched() {
		return
	}
	registeredStateContexts.Put(route.Architecture, route)
}

func RegisteredStateContextArchitectures() []string {
	return registeredStateContexts.Keys()
}

func RegisteredStateContextRoutes() []StateContextRoute {
	return registeredStateContextSnapshot()
}

func ReplaceRegisteredStateContextRoutes(routes []StateContextRoute) {
	order := make([]string, 0, len(routes))
	values := make(map[string]StateContextRoute, len(routes))
	for _, route := range routes {
		route = NormalizeStateContextRoute(route)
		if !route.Matched() {
			continue
		}
		if _, ok := values[route.Architecture]; !ok {
			order = append(order, route.Architecture)
		}
		values[route.Architecture] = route
	}
	registeredStateContexts.Restore(order, values)
}

func RegisteredStateContextRouteForArchitecture(architecture string) (StateContextRoute, bool) {
	return registeredStateContextForArchitecture(architecture)
}

func StateContextRouteForArchitecture(architecture string) (StateContextRoute, bool) {
	architecture = profile.ArchitectureID(architecture)
	if architecture == "" {
		return StateContextRoute{}, false
	}
	if route, ok := registeredStateContextForArchitecture(architecture); ok {
		return route, true
	}
	architectureProfile, ok := profile.LookupArchitectureProfile(architecture)
	if !ok {
		return StateContextRoute{}, false
	}
	if !stateContextArchitectureProfileSupported(architectureProfile) {
		return StateContextRoute{}, false
	}
	route := staticStateContextRoute(architectureProfile.ID, firstNonEmpty(architectureProfile.Family, architectureProfile.ID), architectureProfile)
	if !route.Matched() {
		return StateContextRoute{}, false
	}
	return route, true
}

func StateContextRouteForIdentity(path string, identity inference.ModelIdentity) (StateContextRoute, bool) {
	if identity.Path == "" {
		identity.Path = path
	}
	architecture := firstNonEmpty(
		identity.Labels["engine_architecture_profile"],
		identity.Labels["architecture_model_type"],
		identity.Labels["engine_architecture_resolved"],
		identity.Labels["architecture_resolved"],
		identity.Architecture,
	)
	route, ok := StateContextRouteForArchitecture(architecture)
	if ok {
		if identity.ContextLength > 0 {
			route.ContextWindow = identity.ContextLength
		}
		return route.WithLabels(identity.Labels), true
	}
	route = staticStateContextRoute(stateContextArchitecture(architecture, identity.Labels), "", profile.ArchitectureProfile{})
	if identity.ContextLength > 0 {
		route.ContextWindow = identity.ContextLength
	}
	route = route.WithLabels(identity.Labels)
	if !route.Matched() {
		return StateContextRoute{}, false
	}
	return route, true
}

func StateContextRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (StateContextRoute, bool) {
	return StateContextRouteForIdentity(path, inference.ModelIdentity{
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

func StateContextRouteForInspection(inspection *inference.ModelPackInspection) (StateContextRoute, bool) {
	if inspection == nil {
		return StateContextRoute{}, false
	}
	identity := inspection.Model
	if identity.Path == "" {
		identity.Path = inspection.Path
	}
	labels := mergeStateContextLabels(identity.Labels, inspection.Labels)
	identity.Labels = labels
	return StateContextRouteForIdentity(identity.Path, identity)
}

func DefaultStateContextRoutes() []StateContextRoute {
	profiles := profile.ArchitectureProfiles()
	routes := make([]StateContextRoute, 0, len(profiles)+len(registeredStateContexts.Keys()))
	seen := map[string]int{}
	for _, architectureProfile := range profiles {
		if !stateContextArchitectureProfileSupported(architectureProfile) {
			continue
		}
		route, ok := StateContextRouteForArchitecture(architectureProfile.ID)
		if !ok {
			continue
		}
		seen[route.Architecture] = len(routes)
		routes = append(routes, route)
	}
	for _, route := range registeredStateContextSnapshot() {
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
	return cloneStateContextRoutes(routes)
}

func NormalizeStateContextRoute(route StateContextRoute) StateContextRoute {
	route.Architecture = profile.ArchitectureID(route.Architecture)
	if route.Architecture == "" {
		return StateContextRoute{}
	}
	architectureProfile, hasProfile := profile.LookupArchitectureProfile(route.Architecture)
	if route.Contract == "" {
		route.Contract = StateContextRegistryContract
	}
	if route.Name == "" {
		route.Name = StateContextRouteName
	}
	if route.Family == "" && hasProfile {
		route.Family = firstNonEmpty(architectureProfile.Family, architectureProfile.ID)
	}
	if route.Family == "" {
		route.Family = route.Architecture
	}
	if route.Runtime == "" {
		route.Runtime = StateContextRuntimeMetadata
	}
	if route.Reference == "" {
		route.Reference = "registered_retained_state"
	}
	if hasProfile {
		route.NativeRuntime = route.NativeRuntime || architectureProfile.NativeRuntime
		route.AttachedOnly = route.AttachedOnly || architectureProfile.AttachedOnly
	}
	route.StateSession = route.StateSession ||
		route.SleepState ||
		route.WakeState ||
		route.ForkState ||
		route.CaptureState ||
		route.RestoreState ||
		route.ResetState ||
		route.RuntimeOwnedKV ||
		route.RetainedStateRequired
	if len(route.CacheModes) == 0 && hasProfile {
		route.CacheModes = append([]string(nil), architectureProfile.CacheHints...)
	}
	if len(route.CacheModes) == 0 {
		route.CacheModes = []string{"retained-state"}
	}
	if len(route.StateBackends) == 0 {
		route.StateBackends = []string{"package-local-kv", "hip-device-mirror", "block-bundle-refs"}
	}
	route.DefaultContextWindow = firstPositiveInt(route.DefaultContextWindow, 4096)
	route.DefaultStateBlockSize = firstPositiveInt(route.DefaultStateBlockSize, 128)
	route.DefaultDeviceKVMode = firstNonEmpty(route.DefaultDeviceKVMode, "k-q8-v-q4")
	route.Registered = route.Architecture != "" && route.StateSession
	route.finalize()
	return route.Clone()
}

func registeredStateContextForArchitecture(architecture string) (StateContextRoute, bool) {
	route, ok := registeredStateContexts.Get(profile.ArchitectureID(architecture))
	if !ok {
		return StateContextRoute{}, false
	}
	return route.Clone(), true
}

func registeredStateContextSnapshot() []StateContextRoute {
	routes := registeredStateContexts.Values()
	out := make([]StateContextRoute, 0, len(routes))
	for _, route := range routes {
		out = append(out, route.Clone())
	}
	return out
}

func staticStateContextRoute(architecture, family string, architectureProfile profile.ArchitectureProfile) StateContextRoute {
	architecture = profile.ArchitectureID(architecture)
	route := StateContextRoute{
		Contract:              StateContextRegistryContract,
		Name:                  StateContextRouteName,
		Architecture:          architecture,
		Family:                family,
		Runtime:               StateContextRuntimeMetadata,
		RuntimeStatus:         inference.FeatureRuntimeMetadataOnly,
		DefaultContextWindow:  4096,
		DefaultStateBlockSize: 128,
		DefaultDeviceKVMode:   "k-q8-v-q4",
		CacheModes:            append([]string(nil), architectureProfile.CacheHints...),
		StateBackends:         []string{"package-local-kv", "hip-device-mirror", "block-bundle-refs"},
	}
	if len(route.CacheModes) == 0 && stateContextGemma4Architecture(architecture) {
		route.CacheModes = []string{"q8", "paged", "k-q8-v-q4", "retained-state"}
	}
	switch {
	case stateContextGemma4Architecture(architecture):
		route.Reference = "go_mlx_gemma4_retained_state"
		route.NativeRuntime = architectureProfile.NativeRuntime
		route.StateSession = true
		route.SleepState = true
		route.WakeState = true
		route.ForkState = true
		route.CaptureState = true
		route.RestoreState = true
		route.ResetState = true
		route.RuntimeOwnedKV = true
		route.PromptReplayRefused = true
		route.RemainingContextDefault = true
		route.ModelContextWindow = true
		route.DeviceKVState = true
		route.HIPDeviceMirror = true
		route.PackageLocalKV = true
		route.BlockBundleRefs = true
		route.PortableRefs = true
		route.RetainedStateRequired = true
	case stateContextGemma4AssistantArchitecture(architecture):
		route.Reference = "go_mlx_gemma4_attached_drafter_retained_state"
		route.NativeRuntime = architectureProfile.NativeRuntime
		route.AttachedOnly = true
		route.StateSession = true
		route.SleepState = true
		route.WakeState = true
		route.ForkState = true
		route.RuntimeOwnedKV = true
		route.PromptReplayRefused = true
		route.RemainingContextDefault = true
		route.ModelContextWindow = true
		route.DeviceKVState = true
		route.HIPDeviceMirror = true
		route.PackageLocalKV = true
		route.BlockBundleRefs = true
		route.PortableRefs = true
		route.RetainedStateRequired = true
		route.AttachedDrafterState = true
	default:
		route.Architecture = firstNonEmpty(architecture, route.Architecture)
	}
	if route.Family == "" {
		if architectureProfile, ok := profile.LookupArchitectureProfile(route.Architecture); ok {
			route.Family = firstNonEmpty(architectureProfile.Family, architectureProfile.ID)
		}
	}
	route.finalize()
	return route.Clone()
}

func (route StateContextRoute) withLabels(labels map[string]string) StateContextRoute {
	if len(labels) == 0 {
		return route
	}
	route.ContextWindow = firstPositiveInt(
		stateContextLabelInt(labels["engine_state_context_window"]),
		stateContextLabelInt(labels["context_length"]),
		route.ContextWindow,
	)
	route.Gemma4Size = firstNonEmpty(labels["gemma4_size"], route.Gemma4Size)
	route.Gemma4QuantMode = firstNonEmpty(labels["gemma4_quant_mode"], labels["production_quant_mode"], route.Gemma4QuantMode)
	if cacheHints := firstNonEmpty(labels["engine_architecture_cache_hints"], labels["engine_state_context_cache_modes"]); cacheHints != "" {
		route.CacheModes = stateContextSplitCSV(cacheHints)
	}
	route.DefaultDeviceKVMode = firstNonEmpty(labels["device_kv_mode"], labels["kv_cache_mode"], labels["attention_kv_mode"], route.DefaultDeviceKVMode)
	if labels["engine_model_context_window"] == "true" || labels["engine_feature_model_context_window"] == "true" || labels["engine_feature_route_model_context_window"] == "true" {
		route.ModelContextWindow = true
	}
	if labels["engine_device_kv_state"] == "true" || labels["gemma4_q4_device_kv_state"] != "" {
		route.DeviceKVState = true
	}
	if labels["attached_drafter_retained_state_required"] == "true" || labels["attached.drafter.retained_state_required"] == "true" {
		route.RetainedStateRequired = true
		route.AttachedDrafterState = true
	}
	if route.Architecture == "" {
		route.Architecture = profile.ArchitectureID(firstNonEmpty(labels["engine_architecture_profile"], labels["architecture_model_type"], labels["engine_architecture_resolved"], labels["architecture_resolved"]))
	}
	return route
}

func (route *StateContextRoute) finalize() {
	if route == nil {
		return
	}
	route.Architecture = profile.ArchitectureID(route.Architecture)
	route.Registered = route.Architecture != "" && route.StateSession
	if route.Registered {
		if route.NativeRuntime {
			route.Runtime = StateContextRuntimeAPI
			if route.RuntimeStatus == "" {
				route.RuntimeStatus = inference.FeatureRuntimeExperimental
			}
			if route.AttachedOnly {
				route.Status = StateContextRouteAttachedRuntime
			} else {
				route.Status = StateContextRouteExperimentalRuntime
			}
			route.Staged = false
			route.Planned = false
		} else {
			route.Runtime = firstNonEmpty(route.Runtime, StateContextRuntimeMetadata)
			if route.RuntimeStatus == "" {
				route.RuntimeStatus = inference.FeatureRuntimeMetadataOnly
			}
			route.Status = StateContextRoutePlannedMetadata
			route.Staged = true
			route.Planned = true
		}
	}
	if route.ContextWindow == 0 {
		route.ContextWindow = route.DefaultContextWindow
	}
	route.Capabilities = stateContextRouteCapabilities(*route)
	route.Labels = stateContextRouteLabels(*route)
}

func stateContextArchitecture(architecture string, labels map[string]string) string {
	if architecture := profile.ArchitectureID(architecture); architecture != "" {
		return architecture
	}
	return profile.ArchitectureID(firstNonEmpty(labels["engine_architecture_profile"], labels["architecture_model_type"], labels["engine_architecture_resolved"], labels["architecture_resolved"]))
}

func stateContextArchitectureProfileSupported(architectureProfile profile.ArchitectureProfile) bool {
	if stateContextGemma4Architecture(architectureProfile.ID) || stateContextGemma4AssistantArchitecture(architectureProfile.ID) {
		return true
	}
	for _, hint := range architectureProfile.CacheHints {
		switch strings.TrimSpace(hint) {
		case "retained-state", "attached-drafter":
			return true
		}
	}
	return false
}

func stateContextGemma4Architecture(architecture string) bool {
	switch profile.Gemma4ArchitectureID(architecture) {
	case "gemma4", "gemma4_text", "gemma4_unified":
		return true
	default:
		return false
	}
}

func stateContextGemma4AssistantArchitecture(architecture string) bool {
	return profile.Gemma4ArchitectureID(architecture) == "gemma4_assistant"
}

func stateContextRouteCapabilities(route StateContextRoute) []inference.CapabilityID {
	if !route.Matched() {
		return nil
	}
	capabilities := []inference.CapabilityID{}
	if route.CaptureState || route.RestoreState {
		capabilities = append(capabilities, inference.CapabilityStateBundle)
	}
	if route.WakeState {
		capabilities = append(capabilities, inference.CapabilityStateWake)
	}
	if route.SleepState {
		capabilities = append(capabilities, inference.CapabilityStateSleep)
	}
	if route.ForkState {
		capabilities = append(capabilities, inference.CapabilityStateFork)
	}
	return capabilities
}

// StateContextRouteCapabilities returns the model-owned capability contract for
// a state-context route.
func StateContextRouteCapabilities(route StateContextRoute) []inference.CapabilityID {
	return append([]inference.CapabilityID(nil), stateContextRouteCapabilities(route)...)
}

func stateContextRouteLabels(route StateContextRoute) map[string]string {
	if !route.Matched() {
		return nil
	}
	labels := map[string]string{
		"engine_state_context_route_contract":            route.Contract,
		"engine_state_context_route":                     route.Name,
		"engine_state_context_runtime":                   route.Runtime,
		"engine_state_context_status":                    string(route.Status),
		"engine_state_context_registered":                strconv.FormatBool(route.Registered),
		"engine_state_context_native_runtime":            strconv.FormatBool(route.NativeRuntime),
		"engine_state_context_attached_only":             strconv.FormatBool(route.AttachedOnly),
		"engine_state_context_state_session":             strconv.FormatBool(route.StateSession),
		"engine_state_context_sleep_state":               strconv.FormatBool(route.SleepState),
		"engine_state_context_wake_state":                strconv.FormatBool(route.WakeState),
		"engine_state_context_fork_state":                strconv.FormatBool(route.ForkState),
		"engine_state_context_capture_state":             strconv.FormatBool(route.CaptureState),
		"engine_state_context_restore_state":             strconv.FormatBool(route.RestoreState),
		"engine_state_context_reset_state":               strconv.FormatBool(route.ResetState),
		"engine_state_context_runtime_owned_kv":          strconv.FormatBool(route.RuntimeOwnedKV),
		"engine_state_context_prompt_replay_refused":     strconv.FormatBool(route.PromptReplayRefused),
		"engine_state_context_remaining_context_default": strconv.FormatBool(route.RemainingContextDefault),
		"engine_state_context_model_context_window":      strconv.FormatBool(route.ModelContextWindow),
		"engine_state_context_device_kv_state":           strconv.FormatBool(route.DeviceKVState),
		"engine_state_context_hip_device_mirror":         strconv.FormatBool(route.HIPDeviceMirror),
		"engine_state_context_package_local_kv":          strconv.FormatBool(route.PackageLocalKV),
		"engine_state_context_block_bundle_refs":         strconv.FormatBool(route.BlockBundleRefs),
		"engine_state_context_portable_refs":             strconv.FormatBool(route.PortableRefs),
		"engine_state_context_retained_state_required":   strconv.FormatBool(route.RetainedStateRequired),
		"engine_state_context_attached_drafter_state":    strconv.FormatBool(route.AttachedDrafterState),
		"engine_state_context_staged":                    strconv.FormatBool(route.Staged),
		"engine_state_context_planned":                   strconv.FormatBool(route.Planned),
		"engine_state_context_cache_modes":               joinNonEmptyStrings(route.CacheModes, ","),
		"engine_state_context_state_backends":            joinNonEmptyStrings(route.StateBackends, ","),
		"engine_state_context_capabilities":              stateContextCapabilityLabels(route.Capabilities),
		"engine_state_context_default_device_kv_mode":    route.DefaultDeviceKVMode,
	}
	if route.Architecture != "" {
		labels["engine_state_context_architecture"] = route.Architecture
	}
	if route.Family != "" {
		labels["engine_state_context_family"] = route.Family
	}
	if route.RuntimeStatus != "" {
		labels["engine_state_context_runtime_status"] = string(route.RuntimeStatus)
	}
	if route.Reference != "" {
		labels["engine_state_context_reference"] = route.Reference
	}
	setIntLabel(labels, "engine_state_context_window", route.ContextWindow)
	setIntLabel(labels, "engine_state_context_default_window", route.DefaultContextWindow)
	setIntLabel(labels, "engine_state_context_default_block_size", route.DefaultStateBlockSize)
	if route.Gemma4Size != "" {
		labels["engine_state_context_gemma4_size"] = route.Gemma4Size
	}
	if route.Gemma4QuantMode != "" {
		labels["engine_state_context_gemma4_quant_mode"] = route.Gemma4QuantMode
	}
	return labels
}

// StateContextRouteLabels returns the normalized model-owned label contract for
// a state-context route.
func StateContextRouteLabels(route StateContextRoute) map[string]string {
	route = NormalizeStateContextRoute(route)
	return cloneStringMap(route.Labels)
}

func stateContextLabelInt(value string) int {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 0 {
		return 0
	}
	return parsed
}

func stateContextSplitCSV(value string) []string {
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func stateContextCapabilityLabels(capabilities []inference.CapabilityID) string {
	if len(capabilities) == 0 {
		return ""
	}
	values := make([]string, 0, len(capabilities))
	for _, capability := range capabilities {
		if capability != "" {
			values = append(values, string(capability))
		}
	}
	return joinNonEmptyStrings(values, ",")
}

func mergeStateContextLabels(left, right map[string]string) map[string]string {
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

func cloneStateContextRoutes(routes []StateContextRoute) []StateContextRoute {
	out := append([]StateContextRoute(nil), routes...)
	for i := range out {
		out[i] = out[i].Clone()
	}
	return out
}
