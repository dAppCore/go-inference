// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"strconv"
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/registry"
	"dappco.re/go/inference/engine/hip/profile"
	rocmscheme "dappco.re/go/inference/engine/hip/scheme"
)

const (
	CacheFactoryRouteContract = "rocm-cache-factory-route-v1"
	CacheFactoryRouteName     = "model-cache-factory-route"

	CacheRuntimeHIP      = "hip"
	CacheRuntimeMetadata = "metadata"
	CacheRuntimePlanned  = "planned_hip"
	CacheRuntimeRetained = "retained_state"
	CacheRuntimeAttached = "attached_drafter"
	CacheModeBlockPrefix = "block-prefix"
	CacheModeRetained    = "retained-state"
	CacheModeAttached    = "attached-drafter"
	CacheModeFP16        = "fp16"
	CacheModeQ8          = "q8"
	CacheModeKQ8VQ4      = "k-q8-v-q4"
	CacheModePaged       = "paged"
	CacheModeFixed       = "fixed"
	CacheModeTurboQuant  = "turboquant"
)

// CacheModeRoute describes one cache/state holder the ROCm cache factory can
// plan for. It is metadata-only here; HIP/CUDA/CPU runtimes bind it later.
type CacheModeRoute struct {
	Mode          string                         `json:"mode,omitempty"`
	State         string                         `json:"state,omitempty"`
	Runtime       string                         `json:"runtime,omitempty"`
	RuntimeStatus inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	Registered    bool                           `json:"registered,omitempty"`
	Constructible bool                           `json:"constructible,omitempty"`
	NativeKV      bool                           `json:"native_kv,omitempty"`
	DeviceKV      bool                           `json:"device_kv,omitempty"`
	Quantized     bool                           `json:"quantized,omitempty"`
	Paged         bool                           `json:"paged,omitempty"`
	Fixed         bool                           `json:"fixed,omitempty"`
	Recurrent     bool                           `json:"recurrent,omitempty"`
	MetadataOnly  bool                           `json:"metadata_only,omitempty"`
	Labels        map[string]string              `json:"labels,omitempty"`
}

func (route CacheModeRoute) Matched() bool {
	return route.Mode != ""
}

func (route CacheModeRoute) Clone() CacheModeRoute {
	route.Labels = cloneStringMap(route.Labels)
	return route
}

// CacheRoute is the model-owned cache factory answer for a concrete
// architecture/profile. It mirrors go-mlx's cache factory contract while using
// ROCm cache modes and profile hints.
type CacheRoute struct {
	Contract          string                         `json:"contract,omitempty"`
	Name              string                         `json:"name,omitempty"`
	Architecture      string                         `json:"architecture,omitempty"`
	Family            string                         `json:"family,omitempty"`
	RuntimeStatus     inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	DefaultMode       string                         `json:"default_mode,omitempty"`
	RecommendedMode   string                         `json:"recommended_mode,omitempty"`
	DeviceMode        string                         `json:"device_mode,omitempty"`
	CacheHints        []string                       `json:"cache_hints,omitempty"`
	ModeNames         []string                       `json:"mode_names,omitempty"`
	Modes             []CacheModeRoute               `json:"modes,omitempty"`
	Registered        bool                           `json:"registered,omitempty"`
	NativeRuntime     bool                           `json:"native_runtime,omitempty"`
	SupportsKV        bool                           `json:"supports_kv,omitempty"`
	SupportsDevice    bool                           `json:"supports_device,omitempty"`
	SupportsRecurrent bool                           `json:"supports_recurrent,omitempty"`
	Labels            map[string]string              `json:"labels,omitempty"`
}

func (route CacheRoute) Matched() bool {
	return route.Contract != "" && route.Architecture != "" && len(route.Modes) > 0
}

func (route CacheRoute) Clone() CacheRoute {
	route.CacheHints = append([]string(nil), route.CacheHints...)
	route.ModeNames = append([]string(nil), route.ModeNames...)
	route.Modes = cloneCacheModeRoutes(route.Modes)
	route.Labels = cloneStringMap(route.Labels)
	return route
}

var registeredCacheRoutes = registry.NewOrdered[string, CacheRoute]()

func RegisterCacheRoute(route CacheRoute) {
	route = NormalizeCacheRoute(route)
	if !route.Matched() {
		return
	}
	registeredCacheRoutes.Put(route.Architecture, route)
}

func RegisteredCacheRouteArchitectures() []string {
	return registeredCacheRoutes.Keys()
}

func RegisteredCacheRoutes() []CacheRoute {
	routes := registeredCacheRoutes.Values()
	out := make([]CacheRoute, 0, len(routes))
	for _, route := range routes {
		out = append(out, route.Clone())
	}
	return out
}

func ReplaceRegisteredCacheRoutes(routes []CacheRoute) {
	order := make([]string, 0, len(routes))
	values := make(map[string]CacheRoute, len(routes))
	for _, route := range routes {
		route = NormalizeCacheRoute(route)
		if !route.Matched() {
			continue
		}
		if _, ok := values[route.Architecture]; !ok {
			order = append(order, route.Architecture)
		}
		values[route.Architecture] = route
	}
	registeredCacheRoutes.Restore(order, values)
}

func RegisteredCacheRouteForArchitecture(architecture string) (CacheRoute, bool) {
	route, ok := registeredCacheRoutes.Get(profile.ArchitectureID(architecture))
	if !ok {
		return CacheRoute{}, false
	}
	return route.Clone(), true
}

func CacheRouteForArchitecture(architecture string) (CacheRoute, bool) {
	architecture = profile.ArchitectureID(architecture)
	if architecture == "" {
		return CacheRoute{}, false
	}
	if route, ok := RegisteredCacheRouteForArchitecture(architecture); ok {
		return route, true
	}
	architectureProfile, ok := profile.LookupArchitectureProfile(architecture)
	if !ok {
		return CacheRoute{}, false
	}
	return cacheRouteForProfile(architectureProfile, nil), true
}

func CacheRouteForIdentity(path string, identity inference.ModelIdentity) (CacheRoute, bool) {
	if identity.Path == "" {
		identity.Path = path
	}
	architecture := firstNonEmpty(
		identity.Labels["engine_architecture_resolved"],
		identity.Labels["architecture_resolved"],
		identity.Architecture,
	)
	architecture = profile.ArchitectureID(architecture)
	if architecture == "" {
		return CacheRoute{}, false
	}
	if route, ok := RegisteredCacheRouteForArchitecture(architecture); ok {
		return cacheRouteWithIdentityLabels(route, identity.Labels), true
	}
	architectureProfile, ok := profile.LookupArchitectureProfile(architecture)
	if !ok {
		return CacheRoute{}, false
	}
	return cacheRouteForProfile(architectureProfile, identity.Labels), true
}

func CacheRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (CacheRoute, bool) {
	return CacheRouteForIdentity(path, inference.ModelIdentity{
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

func CacheRouteForInspection(inspection *inference.ModelPackInspection) (CacheRoute, bool) {
	if inspection == nil {
		return CacheRoute{}, false
	}
	identity := inspection.Model
	path := firstNonEmpty(identity.Path, inspection.Path)
	identity.Path = path
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
	return CacheRouteForIdentity(path, identity)
}

func DefaultCacheModeRoutes() []CacheModeRoute {
	modes := append([]string(nil), rocmscheme.CacheModes()...)
	for _, mode := range []string{CacheModeBlockPrefix, CacheModeRetained, CacheModeAttached} {
		if !slices.Contains(modes, mode) {
			modes = append(modes, mode)
		}
	}
	out := make([]CacheModeRoute, 0, len(modes))
	for _, mode := range modes {
		if route, ok := CacheModeRouteForMode(mode); ok {
			out = append(out, route)
		}
	}
	return cloneCacheModeRoutes(out)
}

func CacheModeRouteForMode(mode string) (CacheModeRoute, bool) {
	mode = normalizeCacheMode(mode)
	if mode == "" {
		return CacheModeRoute{}, false
	}
	state := ""
	registered := false
	if cache, ok := rocmscheme.CacheFor(mode); ok {
		registered = true
		state = cache.Serves().String()
	}
	switch mode {
	case CacheModeBlockPrefix:
		state = SequenceMixerStateKVCache
	case CacheModeRetained, CacheModeAttached:
		state = "retained-state"
	}
	if state == "" {
		return CacheModeRoute{}, false
	}
	route := CacheModeRoute{
		Mode:          mode,
		State:         state,
		Registered:    registered,
		Constructible: registered || mode == CacheModeBlockPrefix || mode == CacheModeRetained || mode == CacheModeAttached,
	}
	switch mode {
	case CacheModeFP16:
		route.Runtime = CacheRuntimeHIP
		route.RuntimeStatus = inference.FeatureRuntimeNative
		route.NativeKV = true
		route.DeviceKV = true
	case CacheModeQ8, CacheModeKQ8VQ4:
		route.Runtime = CacheRuntimeHIP
		route.RuntimeStatus = inference.FeatureRuntimeNative
		route.NativeKV = true
		route.DeviceKV = true
		route.Quantized = true
	case CacheModePaged:
		route.Runtime = CacheRuntimePlanned
		route.RuntimeStatus = inference.FeatureRuntimePlanned
		route.Paged = true
	case CacheModeFixed:
		route.Runtime = CacheRuntimePlanned
		route.RuntimeStatus = inference.FeatureRuntimePlanned
		route.Fixed = true
	case CacheModeTurboQuant, SequenceMixerCacheModeCompaction, SequenceMixerCacheModeCompactionFull:
		route.Runtime = CacheRuntimePlanned
		route.RuntimeStatus = inference.FeatureRuntimePlanned
		route.Quantized = true
	case SequenceMixerCacheModeRecurrent:
		route.Runtime = CacheRuntimeMetadata
		route.RuntimeStatus = inference.FeatureRuntimeMetadataOnly
		route.Recurrent = true
	case CacheModeRetained:
		route.Runtime = CacheRuntimeRetained
		route.RuntimeStatus = inference.FeatureRuntimeMetadataOnly
		route.Recurrent = true
	case CacheModeAttached:
		route.Runtime = CacheRuntimeAttached
		route.RuntimeStatus = inference.FeatureRuntimeMetadataOnly
		route.Recurrent = true
	default:
		route.Runtime = CacheRuntimeMetadata
		route.RuntimeStatus = inference.FeatureRuntimeMetadataOnly
	}
	route.MetadataOnly = route.RuntimeStatus == inference.FeatureRuntimeMetadataOnly
	route.Labels = cacheModeRouteLabels(route)
	return route.Clone(), true
}

func NormalizeCacheRoute(route CacheRoute) CacheRoute {
	route.Architecture = profile.ArchitectureID(route.Architecture)
	if route.Architecture == "" {
		return CacheRoute{}
	}
	architectureProfile, hasProfile := profile.LookupArchitectureProfile(route.Architecture)
	if route.Contract == "" {
		route.Contract = CacheFactoryRouteContract
	}
	if route.Name == "" {
		route.Name = CacheFactoryRouteName
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
	if len(route.CacheHints) == 0 && hasProfile {
		route.CacheHints = append([]string(nil), architectureProfile.CacheHints...)
	}
	route.CacheHints = normalizeCacheModes(route.CacheHints)
	if len(route.Modes) == 0 {
		route.Modes = DefaultCacheModeRoutes()
	} else {
		route.Modes = normalizeCacheModeRoutes(route.Modes)
	}
	route.ModeNames = cacheRouteModeNames(route.Modes)
	if route.DefaultMode == "" {
		route.DefaultMode = firstNonEmpty(cacheRouteFirstAvailableHint(route.CacheHints, route.Modes), SequenceMixerCacheModeDefault)
	}
	route.DefaultMode = normalizeCacheMode(route.DefaultMode)
	if route.RecommendedMode == "" {
		route.RecommendedMode = route.DefaultMode
	}
	route.RecommendedMode = normalizeCacheMode(route.RecommendedMode)
	route.DeviceMode = normalizeCacheMode(route.DeviceMode)
	route.Registered = true
	if hasProfile {
		route.NativeRuntime = route.NativeRuntime || architectureProfile.NativeRuntime
	}
	route.SupportsKV, route.SupportsDevice, route.SupportsRecurrent = cacheRouteSupport(route.Modes)
	route.Labels = cacheRouteLabels(route)
	return route.Clone()
}

func cacheRouteForProfile(architectureProfile profile.ArchitectureProfile, labels map[string]string) CacheRoute {
	architectureProfile = profile.NormalizeArchitectureProfile(architectureProfile)
	route := CacheRoute{
		Contract:      CacheFactoryRouteContract,
		Name:          CacheFactoryRouteName,
		Architecture:  architectureProfile.ID,
		Family:        firstNonEmpty(architectureProfile.Family, architectureProfile.ID),
		RuntimeStatus: architectureProfile.RuntimeStatus,
		CacheHints:    append([]string(nil), architectureProfile.CacheHints...),
		Modes:         DefaultCacheModeRoutes(),
		Registered:    architectureProfile.ID != "",
		NativeRuntime: architectureProfile.NativeRuntime,
	}
	route = NormalizeCacheRoute(route)
	return cacheRouteWithIdentityLabels(route, labels)
}

func cacheRouteWithIdentityLabels(route CacheRoute, labels map[string]string) CacheRoute {
	route = route.Clone()
	recommended := firstNonEmpty(
		labels["kv_cache_mode"],
		labels["device_kv_mode"],
		labels["memory_plan_cache_mode"],
		labels["recommended_cache_mode"],
		labels["cache_mode"],
	)
	if recommended != "" {
		route.RecommendedMode = normalizeCacheMode(recommended)
	}
	deviceMode := firstNonEmpty(labels["device_kv_mode"], labels["attention_kv_mode"])
	if deviceMode != "" {
		route.DeviceMode = normalizeCacheMode(deviceMode)
	}
	route.Labels = cacheRouteLabels(route)
	return route.Clone()
}

func cacheRouteLabels(route CacheRoute) map[string]string {
	if route.Architecture == "" {
		return nil
	}
	labels := map[string]string{
		"engine_cache_factory_contract":           firstNonEmpty(route.Contract, CacheFactoryRouteContract),
		"engine_cache_factory_route":              firstNonEmpty(route.Name, CacheFactoryRouteName),
		"engine_cache_factory_registered":         strconv.FormatBool(route.Registered),
		"engine_cache_factory_native_runtime":     strconv.FormatBool(route.NativeRuntime),
		"engine_cache_factory_supports_kv":        strconv.FormatBool(route.SupportsKV),
		"engine_cache_factory_supports_device":    strconv.FormatBool(route.SupportsDevice),
		"engine_cache_factory_supports_recurrent": strconv.FormatBool(route.SupportsRecurrent),
		"engine_cache_factory_modes":              strings.Join(route.ModeNames, ","),
		"engine_cache_factory_mode_count":         strconv.Itoa(len(route.ModeNames)),
	}
	if route.Architecture != "" {
		labels["engine_cache_factory_architecture"] = route.Architecture
	}
	if route.Family != "" {
		labels["engine_cache_factory_family"] = route.Family
	}
	if route.RuntimeStatus != "" {
		labels["engine_cache_factory_runtime_status"] = string(route.RuntimeStatus)
	}
	if route.DefaultMode != "" {
		labels["engine_cache_factory_default_mode"] = route.DefaultMode
	}
	if route.RecommendedMode != "" {
		labels["engine_cache_factory_recommended_mode"] = route.RecommendedMode
	}
	if route.DeviceMode != "" {
		labels["engine_cache_factory_device_mode"] = route.DeviceMode
	}
	if len(route.CacheHints) > 0 {
		labels["engine_cache_factory_hints"] = strings.Join(route.CacheHints, ",")
		labels["engine_cache_factory_hint_count"] = strconv.Itoa(len(route.CacheHints))
	}
	return labels
}

func cacheModeRouteLabels(route CacheModeRoute) map[string]string {
	labels := map[string]string{
		"engine_cache_mode":               route.Mode,
		"engine_cache_mode_state":         route.State,
		"engine_cache_mode_runtime":       route.Runtime,
		"engine_cache_mode_registered":    strconv.FormatBool(route.Registered),
		"engine_cache_mode_constructible": strconv.FormatBool(route.Constructible),
		"engine_cache_mode_native_kv":     strconv.FormatBool(route.NativeKV),
		"engine_cache_mode_device_kv":     strconv.FormatBool(route.DeviceKV),
		"engine_cache_mode_quantized":     strconv.FormatBool(route.Quantized),
		"engine_cache_mode_paged":         strconv.FormatBool(route.Paged),
		"engine_cache_mode_fixed":         strconv.FormatBool(route.Fixed),
		"engine_cache_mode_recurrent":     strconv.FormatBool(route.Recurrent),
		"engine_cache_mode_metadata_only": strconv.FormatBool(route.MetadataOnly),
	}
	if route.RuntimeStatus != "" {
		labels["engine_cache_mode_runtime_status"] = string(route.RuntimeStatus)
	}
	return labels
}

func normalizeCacheModeRoutes(routes []CacheModeRoute) []CacheModeRoute {
	out := make([]CacheModeRoute, 0, len(routes))
	seen := map[string]bool{}
	for _, route := range routes {
		if route.Mode == "" {
			continue
		}
		modeRoute, ok := CacheModeRouteForMode(route.Mode)
		if !ok {
			modeRoute = route.Clone()
			modeRoute.Mode = normalizeCacheMode(modeRoute.Mode)
			modeRoute.Labels = cacheModeRouteLabels(modeRoute)
		}
		if modeRoute.Mode == "" || seen[modeRoute.Mode] {
			continue
		}
		seen[modeRoute.Mode] = true
		out = append(out, modeRoute)
	}
	return out
}

func cloneCacheModeRoutes(routes []CacheModeRoute) []CacheModeRoute {
	out := append([]CacheModeRoute(nil), routes...)
	for index := range out {
		out[index] = out[index].Clone()
	}
	return out
}

func cacheRouteModeNames(routes []CacheModeRoute) []string {
	names := make([]string, 0, len(routes))
	for _, route := range routes {
		if route.Mode != "" && !slices.Contains(names, route.Mode) {
			names = append(names, route.Mode)
		}
	}
	return names
}

func cacheRouteFirstAvailableHint(hints []string, modes []CacheModeRoute) string {
	names := cacheRouteModeNames(modes)
	for _, hint := range hints {
		hint = normalizeCacheMode(hint)
		if hint != "" && slices.Contains(names, hint) {
			return hint
		}
	}
	return ""
}

func cacheRouteSupport(routes []CacheModeRoute) (kv, device, recurrent bool) {
	for _, route := range routes {
		if route.State == SequenceMixerStateKVCache {
			kv = true
		}
		if route.DeviceKV {
			device = true
		}
		if route.Recurrent {
			recurrent = true
		}
	}
	return kv, device, recurrent
}

func normalizeCacheModes(modes []string) []string {
	out := make([]string, 0, len(modes))
	for _, mode := range modes {
		mode = normalizeCacheMode(mode)
		if mode != "" && !slices.Contains(out, mode) {
			out = append(out, mode)
		}
	}
	return out
}

func normalizeCacheMode(mode string) string {
	mode = strings.ToLower(strings.TrimSpace(mode))
	mode = strings.ReplaceAll(mode, "_", "-")
	return mode
}
