// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"sort"
	"strconv"
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/registry"
	"dappco.re/go/inference/engine/hip/profile"
)

const (
	LoRAAdapterRegistryContract = "rocm-lora-adapter-registry-v1"

	LoRAAdapterRouteName       = "model-lora-adapter-route"
	LoRAAdapterLoaderLinear    = "lora-linear"
	LoRAAdapterRuntimeHIP      = "hip"
	LoRAAdapterRuntimeMetadata = "metadata"
)

type LoRAAdapterRouteStatus string

const (
	LoRAAdapterRouteExperimentalNative LoRAAdapterRouteStatus = "experimental_native"
	LoRAAdapterRouteStagedNative       LoRAAdapterRouteStatus = "staged_native"
	LoRAAdapterRoutePlannedMetadata    LoRAAdapterRouteStatus = "planned_metadata"
	LoRAAdapterRouteAttachedOnly       LoRAAdapterRouteStatus = "attached_only"
)

type LoRATargetPolicy = profile.LoRATargetPolicy

// LoRAAdapterRoute is the folder-owned adapter target-policy catalogue. It is
// pure metadata, so model-family packages can register ApplyLoRA target paths
// without importing the root rocm package.
type LoRAAdapterRoute struct {
	Contract              string                         `json:"contract,omitempty"`
	Name                  string                         `json:"name,omitempty"`
	Architecture          string                         `json:"architecture,omitempty"`
	Family                string                         `json:"family,omitempty"`
	Loader                string                         `json:"loader,omitempty"`
	Runtime               string                         `json:"runtime,omitempty"`
	RuntimeStatus         inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	Status                LoRAAdapterRouteStatus         `json:"status,omitempty"`
	TargetPolicy          string                         `json:"target_policy,omitempty"`
	DefaultTargets        []string                       `json:"default_targets,omitempty"`
	SafeTargets           []string                       `json:"safe_targets,omitempty"`
	ExtendedTargets       []string                       `json:"extended_targets,omitempty"`
	TargetPaths           map[string]string              `json:"target_paths,omitempty"`
	Registered            bool                           `json:"registered,omitempty"`
	NativeRuntime         bool                           `json:"native_runtime,omitempty"`
	ApplySupported        bool                           `json:"apply_supported,omitempty"`
	LoadSupported         bool                           `json:"load_supported,omitempty"`
	FuseSupported         bool                           `json:"fuse_supported,omitempty"`
	TrainingSupported     bool                           `json:"training_supported,omitempty"`
	Staged                bool                           `json:"staged,omitempty"`
	Planned               bool                           `json:"planned,omitempty"`
	AttachedOnly          bool                           `json:"attached_only,omitempty"`
	RequiresExtendedOptIn bool                           `json:"requires_extended_opt_in,omitempty"`
	Capabilities          []inference.CapabilityID       `json:"capabilities,omitempty"`
	Labels                map[string]string              `json:"labels,omitempty"`
}

func (route LoRAAdapterRoute) Matched() bool {
	return route.Contract != "" && route.Architecture != "" && route.Loader != ""
}

func (route LoRAAdapterRoute) Clone() LoRAAdapterRoute {
	route.DefaultTargets = append([]string(nil), route.DefaultTargets...)
	route.SafeTargets = append([]string(nil), route.SafeTargets...)
	route.ExtendedTargets = append([]string(nil), route.ExtendedTargets...)
	route.TargetPaths = cloneStringMap(route.TargetPaths)
	route.Capabilities = append([]inference.CapabilityID(nil), route.Capabilities...)
	route.Labels = cloneStringMap(route.Labels)
	return route
}

var registeredLoRAAdapters = registry.NewOrdered[string, LoRAAdapterRoute]()

// RegisterLoRAAdapterRoute registers or replaces adapter route metadata by
// architecture.
func RegisterLoRAAdapterRoute(route LoRAAdapterRoute) {
	route = NormalizeLoRAAdapterRoute(route)
	if !route.Matched() {
		return
	}
	registeredLoRAAdapters.Put(route.Architecture, route)
}

func RegisteredLoRAAdapterArchitectures() []string {
	return registeredLoRAAdapters.Keys()
}

func RegisteredLoRAAdapterRoutes() []LoRAAdapterRoute {
	return registeredLoRAAdapterSnapshot()
}

func ReplaceRegisteredLoRAAdapterRoutes(routes []LoRAAdapterRoute) {
	order := make([]string, 0, len(routes))
	values := make(map[string]LoRAAdapterRoute, len(routes))
	for _, route := range routes {
		route = NormalizeLoRAAdapterRoute(route)
		if !route.Matched() {
			continue
		}
		if _, ok := values[route.Architecture]; !ok {
			order = append(order, route.Architecture)
		}
		values[route.Architecture] = route
	}
	registeredLoRAAdapters.Restore(order, values)
}

func RegisteredLoRAAdapterRouteForArchitecture(architecture string) (LoRAAdapterRoute, bool) {
	return registeredLoRAAdapterForArchitecture(architecture)
}

func LoRAAdapterRouteForArchitecture(architecture string) (LoRAAdapterRoute, bool) {
	architecture = profile.ArchitectureID(architecture)
	if architecture == "" {
		return LoRAAdapterRoute{}, false
	}
	if route, ok := registeredLoRAAdapterForArchitecture(architecture); ok {
		return route, true
	}
	architectureProfile, ok := profile.LookupArchitectureProfile(architecture)
	if !ok {
		return LoRAAdapterRoute{}, false
	}
	return loRAAdapterRouteForProfile(architectureProfile)
}

func LoRAAdapterRouteForIdentity(path string, identity inference.ModelIdentity) (LoRAAdapterRoute, bool) {
	if identity.Path == "" {
		identity.Path = path
	}
	architecture := firstNonEmpty(
		identity.Labels["engine_architecture_resolved"],
		identity.Labels["architecture_resolved"],
		identity.Architecture,
	)
	return LoRAAdapterRouteForArchitecture(architecture)
}

func LoRAAdapterRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (LoRAAdapterRoute, bool) {
	return LoRAAdapterRouteForIdentity(path, inference.ModelIdentity{
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

func LoRAAdapterRouteForInspection(inspection *inference.ModelPackInspection) (LoRAAdapterRoute, bool) {
	if inspection == nil {
		return LoRAAdapterRoute{}, false
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
	return LoRAAdapterRouteForIdentity(identity.Path, identity)
}

func DefaultLoRAAdapterRoutes() []LoRAAdapterRoute {
	profiles := profile.ArchitectureProfiles()
	routes := make([]LoRAAdapterRoute, 0, len(profiles)+len(registeredLoRAAdapters.Keys()))
	seen := map[string]int{}
	for _, architectureProfile := range profiles {
		route, ok := loRAAdapterRouteForProfile(architectureProfile)
		if !ok || !route.Matched() {
			continue
		}
		seen[route.Architecture] = len(routes)
		routes = append(routes, route)
	}
	for _, route := range registeredLoRAAdapterSnapshot() {
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
	return cloneLoRAAdapterRoutes(routes)
}

func NormalizeLoRAAdapterRoute(route LoRAAdapterRoute) LoRAAdapterRoute {
	route.Architecture = profile.ArchitectureID(route.Architecture)
	if route.Architecture == "" {
		return LoRAAdapterRoute{}
	}
	architectureProfile, hasProfile := profile.LookupArchitectureProfile(route.Architecture)
	if route.Contract == "" {
		route.Contract = LoRAAdapterRegistryContract
	}
	if route.Name == "" {
		route.Name = LoRAAdapterRouteName
	}
	if route.Loader == "" {
		route.Loader = LoRAAdapterLoaderLinear
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
	if route.TargetPolicy == "" {
		route.TargetPolicy = "registered"
	}
	route.DefaultTargets = cleanLoRATargets(route.DefaultTargets)
	route.SafeTargets = cleanLoRATargets(route.SafeTargets)
	route.ExtendedTargets = cleanLoRATargets(route.ExtendedTargets)
	route.TargetPaths = cleanLoRATargetPaths(route.TargetPaths)
	if len(route.SafeTargets) == 0 {
		route.SafeTargets = cleanLoRATargets(append([]string(nil), route.DefaultTargets...))
	}
	if len(route.DefaultTargets) == 0 {
		route.DefaultTargets = cleanLoRATargets(route.SafeTargets)
	}
	if hasProfile {
		route.NativeRuntime = route.NativeRuntime || architectureProfile.NativeRuntime
		route.AttachedOnly = route.AttachedOnly || architectureProfile.AttachedOnly
	}
	if route.Registered || len(route.TargetPaths) > 0 {
		route.Registered = !route.AttachedOnly
	}
	if route.Registered {
		route.ApplySupported = route.ApplySupported || len(route.TargetPaths) > 0
		route.LoadSupported = route.LoadSupported || len(route.TargetPaths) > 0
		route.FuseSupported = route.FuseSupported || len(route.TargetPaths) > 0
		route.TrainingSupported = route.TrainingSupported || len(route.TargetPaths) > 0
	}
	route.RequiresExtendedOptIn = route.RequiresExtendedOptIn || len(route.ExtendedTargets) > 0
	route = loRAAdapterRouteWithStatusDefaults(route)
	route.Capabilities = mergeFeatureCapabilityIDs(loRAAdapterRouteCapabilities(route), route.Capabilities)
	route.Labels = loRAAdapterRouteLabels(route)
	return route.Clone()
}

func loRAAdapterRouteForProfile(architectureProfile profile.ArchitectureProfile) (LoRAAdapterRoute, bool) {
	architectureProfile = profile.NormalizeArchitectureProfile(architectureProfile)
	targetPolicy, policy, ok := loRAAdapterPolicyForProfile(architectureProfile)
	if !ok {
		return LoRAAdapterRoute{}, false
	}
	attachedOnly := architectureProfile.AttachedOnly
	nativeRuntime := architectureProfile.NativeRuntime
	registered := !attachedOnly && len(policy.TargetPaths) > 0
	staged := registered && nativeRuntime && !architectureProfile.Generation
	planned := registered && !nativeRuntime
	runtime := LoRAAdapterRuntimeHIP
	if planned {
		runtime = LoRAAdapterRuntimeMetadata
	}
	route := LoRAAdapterRoute{
		Contract:              LoRAAdapterRegistryContract,
		Name:                  LoRAAdapterRouteName,
		Architecture:          architectureProfile.ID,
		Family:                firstNonEmpty(architectureProfile.Family, architectureProfile.ID),
		Loader:                LoRAAdapterLoaderLinear,
		Runtime:               runtime,
		RuntimeStatus:         architectureProfile.RuntimeStatus,
		TargetPolicy:          targetPolicy,
		DefaultTargets:        append([]string(nil), policy.DefaultTargets...),
		SafeTargets:           append([]string(nil), policy.SafeTargets...),
		ExtendedTargets:       append([]string(nil), policy.ExtendedTargets...),
		TargetPaths:           cloneStringMap(policy.TargetPaths),
		Registered:            registered,
		NativeRuntime:         nativeRuntime,
		ApplySupported:        registered,
		LoadSupported:         registered,
		FuseSupported:         registered && len(policy.TargetPaths) > 0,
		TrainingSupported:     registered,
		Staged:                staged,
		Planned:               planned,
		AttachedOnly:          attachedOnly,
		RequiresExtendedOptIn: len(policy.ExtendedTargets) > 0,
	}
	route.Status = loRAAdapterRouteStatus(route)
	route.Capabilities = loRAAdapterRouteCapabilities(route)
	route.Labels = loRAAdapterRouteLabels(route)
	return route.Clone(), true
}

func registeredLoRAAdapterForArchitecture(architecture string) (LoRAAdapterRoute, bool) {
	route, ok := registeredLoRAAdapters.Get(profile.ArchitectureID(architecture))
	if !ok {
		return LoRAAdapterRoute{}, false
	}
	return route.Clone(), true
}

func registeredLoRAAdapterSnapshot() []LoRAAdapterRoute {
	routes := registeredLoRAAdapters.Values()
	out := make([]LoRAAdapterRoute, 0, len(routes))
	for _, route := range routes {
		out = append(out, route.Clone())
	}
	return out
}

func LoRATargetPolicyForArchitecture(architecture string) (LoRATargetPolicy, bool) {
	if route, ok := registeredLoRAAdapterForArchitecture(architecture); ok && route.Registered && len(route.TargetPaths) > 0 {
		return profile.CloneLoRATargetPolicy(LoRATargetPolicy{
			DefaultTargets:  append([]string(nil), route.DefaultTargets...),
			SafeTargets:     append([]string(nil), route.SafeTargets...),
			ExtendedTargets: append([]string(nil), route.ExtendedTargets...),
			TargetPaths:     cloneStringMap(route.TargetPaths),
		}), true
	}
	if policy, ok := profile.LoRATargetPolicyForArchitecture(architecture); ok {
		return policy, true
	}
	architectureProfile, ok := profile.LookupArchitectureProfile(architecture)
	if !ok {
		return LoRATargetPolicy{}, false
	}
	_, policy, ok := loRAAdapterPolicyForProfile(architectureProfile)
	return policy, ok
}

func LoRATargetPath(architecture, target string) (string, bool) {
	policy, ok := LoRATargetPolicyForArchitecture(architecture)
	if !ok {
		return "", false
	}
	target = strings.TrimSpace(target)
	if target == "" {
		return "", false
	}
	canonical, ok := policy.TargetPaths[target]
	if !ok || strings.TrimSpace(canonical) == "" {
		return "", false
	}
	return canonical, true
}

func LoRASafeTarget(architecture, target string) bool {
	policy, ok := LoRATargetPolicyForArchitecture(architecture)
	if !ok {
		return false
	}
	target = strings.TrimSpace(target)
	return slices.Contains(policy.SafeTargets, target)
}

func LoRAExtendedTarget(architecture, target string) bool {
	policy, ok := LoRATargetPolicyForArchitecture(architecture)
	if !ok {
		return false
	}
	target = strings.TrimSpace(target)
	return slices.Contains(policy.ExtendedTargets, target)
}

func LoRACanonicalTarget(architecture, target string) (string, bool) {
	target = strings.TrimSpace(target)
	if target == "" {
		return "", false
	}
	if canonical, ok := LoRATargetPath(architecture, target); ok {
		return canonical, true
	}
	parts := strings.Split(target, ".")
	if len(parts) >= 2 {
		short := strings.Join(parts[len(parts)-2:], ".")
		if canonical, ok := LoRATargetPath(architecture, short); ok {
			return joinLoRACanonicalTarget(parts[:len(parts)-2], canonical), true
		}
	}
	if len(parts) >= 1 {
		short := parts[len(parts)-1]
		if canonical, ok := LoRATargetPath(architecture, short); ok {
			return joinLoRACanonicalTarget(parts[:len(parts)-1], canonical), true
		}
	}
	return "", false
}

func loRAAdapterPolicyForProfile(architectureProfile profile.ArchitectureProfile) (string, LoRATargetPolicy, bool) {
	if policy, ok := profile.LoRATargetPolicyForProfile(architectureProfile); ok {
		return loRATargetPolicyName(architectureProfile), policy, true
	}
	return "", LoRATargetPolicy{}, false
}

func loRATargetPolicyName(architectureProfile profile.ArchitectureProfile) string {
	if name := profile.ArchitectureProfileLoRATargetPolicyName(architectureProfile.ID); name != "" {
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

func loRAAdapterRouteStatus(route LoRAAdapterRoute) LoRAAdapterRouteStatus {
	switch {
	case route.AttachedOnly:
		return LoRAAdapterRouteAttachedOnly
	case route.Planned:
		return LoRAAdapterRoutePlannedMetadata
	case route.Staged:
		return LoRAAdapterRouteStagedNative
	default:
		return LoRAAdapterRouteExperimentalNative
	}
}

func loRAAdapterRouteWithStatusDefaults(route LoRAAdapterRoute) LoRAAdapterRoute {
	if route.Runtime == "" {
		route.Runtime = LoRAAdapterRuntimeHIP
		if route.Planned || !route.NativeRuntime {
			route.Runtime = LoRAAdapterRuntimeMetadata
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
		route.Runtime = LoRAAdapterRuntimeMetadata
	}
	if route.Status == "" {
		route.Status = loRAAdapterRouteStatus(route)
	}
	return route
}

func loRAAdapterRouteCapabilities(route LoRAAdapterRoute) []inference.CapabilityID {
	if !route.Registered {
		return nil
	}
	capabilities := []inference.CapabilityID{inference.CapabilityLoRAInference}
	if route.TrainingSupported {
		capabilities = append(capabilities, inference.CapabilityLoRATraining)
	}
	if route.FuseSupported {
		capabilities = append(capabilities, inference.CapabilityModelMerge)
	}
	return capabilities
}

// LoRAAdapterRouteCapabilities returns capability IDs implied by an adapter
// route using the model-owned LoRA registry contract.
func LoRAAdapterRouteCapabilities(route LoRAAdapterRoute) []inference.CapabilityID {
	return append([]inference.CapabilityID(nil), loRAAdapterRouteCapabilities(route)...)
}

func loRAAdapterRouteLabels(route LoRAAdapterRoute) map[string]string {
	if !route.Matched() {
		return nil
	}
	labels := map[string]string{
		"engine_lora_adapter_route_contract":       route.Contract,
		"engine_lora_route_contract":               route.Contract,
		"engine_lora_adapter_route":                route.Name,
		"engine_lora_route":                        route.Name,
		"engine_lora_loader":                       route.Loader,
		"engine_lora_runtime":                      route.Runtime,
		"engine_lora_status":                       string(route.Status),
		"engine_lora_target_policy":                route.TargetPolicy,
		"engine_lora_registered":                   strconv.FormatBool(route.Registered),
		"engine_lora_native_runtime":               strconv.FormatBool(route.NativeRuntime),
		"engine_lora_apply_supported":              strconv.FormatBool(route.ApplySupported),
		"engine_lora_load_supported":               strconv.FormatBool(route.LoadSupported),
		"engine_lora_fuse_supported":               strconv.FormatBool(route.FuseSupported),
		"engine_lora_training_supported":           strconv.FormatBool(route.TrainingSupported),
		"engine_lora_staged":                       strconv.FormatBool(route.Staged),
		"engine_lora_planned":                      strconv.FormatBool(route.Planned),
		"engine_lora_attached_only":                strconv.FormatBool(route.AttachedOnly),
		"engine_lora_extended_targets_require_opt": strconv.FormatBool(route.RequiresExtendedOptIn),
		"engine_lora_default_targets":              strings.Join(route.DefaultTargets, ","),
		"engine_lora_safe_targets":                 strings.Join(route.SafeTargets, ","),
		"engine_lora_extended_targets":             strings.Join(route.ExtendedTargets, ","),
		"engine_lora_target_count":                 strconv.Itoa(len(route.TargetPaths)),
	}
	if route.Architecture != "" {
		labels["engine_lora_architecture"] = route.Architecture
	}
	if route.Family != "" {
		labels["engine_lora_family"] = route.Family
	}
	if route.RuntimeStatus != "" {
		labels["engine_lora_runtime_status"] = string(route.RuntimeStatus)
	}
	if len(route.TargetPaths) > 0 {
		labels["engine_lora_target_paths"] = loRATargetPathPairs(route.TargetPaths)
	}
	if len(route.Capabilities) > 0 {
		labels["engine_lora_capabilities"] = capabilityIDsCSV(route.Capabilities)
	}
	return labels
}

// LoRAAdapterRouteLabels returns labels for an adapter route using the
// model-owned LoRA registry contract.
func LoRAAdapterRouteLabels(route LoRAAdapterRoute) map[string]string {
	return cloneStringMap(loRAAdapterRouteLabels(route))
}

func cleanLoRATargets(targets []string) []string {
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

func cleanLoRATargetPaths(paths map[string]string) map[string]string {
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

func loRATargetPathPairs(paths map[string]string) string {
	if len(paths) == 0 {
		return ""
	}
	keys := make([]string, 0, len(paths))
	for key := range paths {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		value := strings.TrimSpace(paths[key])
		if value == "" {
			continue
		}
		parts = append(parts, key+"="+value)
	}
	return strings.Join(parts, ",")
}

func joinLoRACanonicalTarget(prefix []string, canonical string) string {
	if len(prefix) == 0 {
		return canonical
	}
	parts := append([]string(nil), prefix...)
	parts = append(parts, canonical)
	return strings.Join(parts, ".")
}

func cloneLoRAAdapterRoutes(routes []LoRAAdapterRoute) []LoRAAdapterRoute {
	out := append([]LoRAAdapterRoute(nil), routes...)
	for i := range out {
		out[i] = out[i].Clone()
	}
	return out
}
