// SPDX-Licence-Identifier: EUPL-1.2

// Package model owns ROCm's model-family contract catalogues. It is intentionally
// pure metadata: concrete HIP/CUDA/CPU loaders can self-register here without
// importing the root rocm package or extending central switches.
package model

import (
	"strconv"
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/registry"
	"dappco.re/go/inference/engine/hip/profile"
)

const (
	LoaderRegistryContract = "rocm-model-loader-registry-v1"

	RuntimeHIP      = "hip"
	RuntimeMetadata = "metadata"

	StatusStandaloneNative = "standalone_native"
	StatusStagedNative     = "staged_native"
	StatusAttachedOnly     = "attached_only"
	StatusMetadataOnly     = "metadata_only"
)

// LoaderRoute is the folder-owned model-loader metadata route. It mirrors the
// root ROCm API surface while staying independent of root package types.
type LoaderRoute struct {
	Contract      string                         `json:"contract,omitempty"`
	Name          string                         `json:"name,omitempty"`
	Architecture  string                         `json:"architecture,omitempty"`
	Family        string                         `json:"family,omitempty"`
	Loader        string                         `json:"loader,omitempty"`
	Runtime       string                         `json:"runtime,omitempty"`
	Status        string                         `json:"status,omitempty"`
	Target        string                         `json:"target,omitempty"`
	RuntimeStatus inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	Reason        string                         `json:"reason,omitempty"`
	Registered    bool                           `json:"registered,omitempty"`
	NativeRuntime bool                           `json:"native_runtime,omitempty"`
	Standalone    bool                           `json:"standalone,omitempty"`
	AttachedOnly  bool                           `json:"attached_only,omitempty"`
	Staged        bool                           `json:"staged,omitempty"`
	MetadataOnly  bool                           `json:"metadata_only,omitempty"`
	TextGenerate  bool                           `json:"text_generate,omitempty"`
	Labels        map[string]string              `json:"labels,omitempty"`
}

func (route LoaderRoute) Matched() bool {
	return route.Contract != "" && route.Architecture != "" && route.Loader != ""
}

func (route LoaderRoute) Clone() LoaderRoute {
	route.Labels = cloneStringMap(route.Labels)
	return route
}

var registeredLoaders = registry.NewOrdered[string, LoaderRoute]()

// RegisterLoaderRoute registers or replaces loader metadata by architecture.
func RegisterLoaderRoute(route LoaderRoute) {
	route = NormalizeLoaderRoute(route)
	if !route.Matched() {
		return
	}
	registeredLoaders.Put(route.Architecture, route)
}

func RegisteredLoaderArchitectures() []string {
	return registeredLoaders.Keys()
}

// RegisteredLoaderRoutes returns extension loader routes in registration order.
func RegisteredLoaderRoutes() []LoaderRoute {
	return registeredLoaderSnapshot()
}

// ReplaceRegisteredLoaderRoutes replaces extension loader registrations. It is
// useful for embedding code that needs a scoped registry view and for tests that
// snapshot process-global registrations before exercising self-registration.
func ReplaceRegisteredLoaderRoutes(routes []LoaderRoute) {
	order := make([]string, 0, len(routes))
	values := make(map[string]LoaderRoute, len(routes))
	for _, route := range routes {
		route = NormalizeLoaderRoute(route)
		if !route.Matched() {
			continue
		}
		if _, ok := values[route.Architecture]; !ok {
			order = append(order, route.Architecture)
		}
		values[route.Architecture] = route
	}
	registeredLoaders.Restore(order, values)
}

// RegisteredLoaderRouteForArchitecture resolves only extension registrations.
func RegisteredLoaderRouteForArchitecture(architecture string) (LoaderRoute, bool) {
	return registeredLoaderForArchitecture(architecture)
}

func LoaderRouteForArchitecture(architecture string) (LoaderRoute, bool) {
	architecture = profile.ArchitectureID(architecture)
	if architecture == "" {
		return LoaderRoute{}, false
	}
	if route, ok := registeredLoaderForArchitecture(architecture); ok {
		return route, true
	}
	architectureProfile, ok := profile.LookupArchitectureProfile(architecture)
	if !ok {
		return LoaderRoute{}, false
	}
	return loaderRouteForProfile(architectureProfile), true
}

// LoaderRouteForIdentity resolves a loader route from backend-neutral model
// identity metadata. Resolved-architecture labels win over the raw architecture
// string because config probes may refine wrapper classes into load targets.
func LoaderRouteForIdentity(path string, identity inference.ModelIdentity) (LoaderRoute, bool) {
	if identity.Path == "" {
		identity.Path = path
	}
	architecture := firstNonEmpty(
		identity.Labels["engine_architecture_resolved"],
		identity.Labels["architecture_resolved"],
		identity.Architecture,
	)
	return LoaderRouteForArchitecture(architecture)
}

// LoaderRouteForInfo adapts the small TextModel.Info shape plus caller labels
// into the same loader-route resolver used for inspected model packs.
func LoaderRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (LoaderRoute, bool) {
	return LoaderRouteForIdentity(path, inference.ModelIdentity{
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

// LoaderRouteForInspection resolves from a portable model-pack inspection,
// merging inspection labels with model-owned labels without mutating either.
func LoaderRouteForInspection(inspection *inference.ModelPackInspection) (LoaderRoute, bool) {
	if inspection == nil {
		return LoaderRoute{}, false
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
	return LoaderRouteForIdentity(identity.Path, identity)
}

func DefaultLoaderRoutes() []LoaderRoute {
	profiles := profile.ArchitectureProfiles()
	routes := make([]LoaderRoute, 0, len(profiles)+len(registeredLoaders.Keys()))
	seen := map[string]int{}
	for _, architectureProfile := range profiles {
		route := loaderRouteForProfile(architectureProfile)
		if !route.Matched() {
			continue
		}
		seen[route.Architecture] = len(routes)
		routes = append(routes, route)
	}
	for _, route := range registeredLoaderSnapshot() {
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
	return cloneLoaderRoutes(routes)
}

func LoaderArchitectures() []string {
	routes := DefaultLoaderRoutes()
	out := make([]string, 0, len(routes))
	for _, route := range routes {
		if route.Architecture != "" {
			out = append(out, route.Architecture)
		}
	}
	return out
}

func NormalizeLoaderRoute(route LoaderRoute) LoaderRoute {
	route.Architecture = profile.ArchitectureID(route.Architecture)
	if route.Architecture == "" {
		return LoaderRoute{}
	}
	if route.Contract == "" {
		route.Contract = LoaderRegistryContract
	}
	if route.Name == "" {
		route.Name = "architecture-loader"
	}
	if route.Loader == "" {
		route.Loader = loaderNameForArchitecture(route.Architecture)
	}
	if route.Family == "" {
		if architectureProfile, ok := profile.LookupArchitectureProfile(route.Architecture); ok {
			route.Family = firstNonEmpty(architectureProfile.Family, architectureProfile.ID)
		}
	}
	if route.Family == "" {
		route.Family = route.Architecture
	}
	if route.Status == "" {
		route.Status = statusForRoute(route)
	}
	route = routeWithStatusDefaults(route)
	if route.RuntimeStatus == "" {
		if architectureProfile, ok := profile.LookupArchitectureProfile(route.Architecture); ok {
			route.RuntimeStatus = architectureProfile.RuntimeStatus
		}
	}
	if route.RuntimeStatus == "" && route.NativeRuntime {
		route.RuntimeStatus = inference.FeatureRuntimeNative
	}
	route.Labels = loaderRouteLabels(route)
	return route.Clone()
}

func loaderRouteForProfile(architectureProfile profile.ArchitectureProfile) LoaderRoute {
	architectureProfile = profile.NormalizeArchitectureProfile(architectureProfile)
	route := LoaderRoute{
		Contract:      LoaderRegistryContract,
		Name:          "architecture-loader",
		Architecture:  architectureProfile.ID,
		Family:        firstNonEmpty(architectureProfile.Family, architectureProfile.ID),
		Loader:        loaderNameForArchitecture(architectureProfile.ID),
		RuntimeStatus: architectureProfile.RuntimeStatus,
		NativeRuntime: architectureProfile.NativeRuntime,
		AttachedOnly:  architectureProfile.AttachedOnly,
		TextGenerate:  architectureProfile.NativeRuntime && architectureProfile.Generation && !architectureProfile.AttachedOnly,
	}
	switch {
	case architectureProfile.AttachedOnly:
		route.Status = StatusAttachedOnly
		route.Reason = "architecture is declared as an attached drafter and must load beside a target model"
	case !architectureProfile.NativeRuntime:
		route.Status = StatusMetadataOnly
		route.Reason = "architecture is recognised by the registry but has no native runtime loader yet"
	case route.TextGenerate:
		route.Status = StatusStandaloneNative
		route.Reason = "native standalone text-generation path is advertised by the resolved model profile"
	default:
		route.Status = StatusStagedNative
		route.Reason = "native metadata/config loader is staged while standalone generation remains pending"
	}
	return NormalizeLoaderRoute(route)
}

func registeredLoaderForArchitecture(architecture string) (LoaderRoute, bool) {
	route, ok := registeredLoaders.Get(profile.ArchitectureID(architecture))
	if !ok {
		return LoaderRoute{}, false
	}
	return route.Clone(), true
}

func registeredLoaderSnapshot() []LoaderRoute {
	routes := registeredLoaders.Values()
	out := make([]LoaderRoute, 0, len(routes))
	for _, route := range routes {
		out = append(out, route.Clone())
	}
	return out
}

func statusForRoute(route LoaderRoute) string {
	switch {
	case route.MetadataOnly || route.Runtime == RuntimeMetadata:
		return StatusMetadataOnly
	case route.AttachedOnly:
		return StatusAttachedOnly
	case route.Staged:
		return StatusStagedNative
	case route.NativeRuntime || route.Registered || route.TextGenerate:
		return StatusStandaloneNative
	default:
		return StatusMetadataOnly
	}
}

func routeWithStatusDefaults(route LoaderRoute) LoaderRoute {
	switch route.Status {
	case StatusAttachedOnly:
		route.Target = firstNonEmpty(route.Target, "attached")
		route.AttachedOnly = true
		route.NativeRuntime = true
		route.Registered = true
	case StatusMetadataOnly:
		route.Target = firstNonEmpty(route.Target, "metadata")
		route.MetadataOnly = true
		route.NativeRuntime = false
		route.Registered = false
	case StatusStagedNative:
		route.Target = firstNonEmpty(route.Target, "standalone")
		route.Standalone = true
		route.Staged = true
		route.NativeRuntime = true
		route.Registered = true
	case StatusStandaloneNative:
		route.Target = firstNonEmpty(route.Target, "standalone")
		route.Standalone = true
		route.NativeRuntime = true
		route.Registered = true
		if !route.Staged {
			route.TextGenerate = true
		}
	}
	if route.Runtime == "" {
		route.Runtime = RuntimeHIP
		if route.MetadataOnly || !route.NativeRuntime {
			route.Runtime = RuntimeMetadata
		}
	}
	return route
}

func loaderNameForArchitecture(architecture string) string {
	switch architecture {
	case "glm4":
		return "glm"
	case "gpt-oss":
		return "gpt_oss"
	default:
		return architecture
	}
}

func loaderRouteLabels(route LoaderRoute) map[string]string {
	if !route.Matched() {
		return nil
	}
	labels := map[string]string{
		"engine_loader_contract":      route.Contract,
		"engine_loader":               route.Loader,
		"engine_loader_runtime":       route.Runtime,
		"engine_loader_registered":    strconv.FormatBool(route.Registered),
		"engine_loader_native":        strconv.FormatBool(route.NativeRuntime),
		"engine_loader_standalone":    strconv.FormatBool(route.Standalone),
		"engine_loader_attached_only": strconv.FormatBool(route.AttachedOnly),
		"engine_loader_staged":        strconv.FormatBool(route.Staged),
		"engine_loader_metadata_only": strconv.FormatBool(route.MetadataOnly),
		"engine_loader_text_generate": strconv.FormatBool(route.TextGenerate),
	}
	if route.Architecture != "" {
		labels["engine_loader_architecture"] = route.Architecture
	}
	if route.Family != "" {
		labels["engine_loader_family"] = route.Family
	}
	if route.Status != "" {
		labels["engine_loader_status"] = route.Status
	}
	if route.Target != "" {
		labels["engine_loader_target"] = route.Target
	}
	if route.RuntimeStatus != "" {
		labels["engine_loader_runtime_status"] = string(route.RuntimeStatus)
	}
	if route.Reason != "" {
		labels["engine_loader_reason"] = strings.TrimSpace(route.Reason)
	}
	return labels
}

// LoaderRouteLabels returns the model-owned label contract for a loader route.
func LoaderRouteLabels(route LoaderRoute) map[string]string {
	return cloneStringMap(loaderRouteLabels(route))
}

func cloneLoaderRoutes(routes []LoaderRoute) []LoaderRoute {
	out := append([]LoaderRoute(nil), routes...)
	for i := range out {
		out[i] = out[i].Clone()
	}
	return out
}

func cloneStringMap(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	out := make(map[string]string, len(values))
	for key, value := range values {
		out[key] = value
	}
	return out
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
