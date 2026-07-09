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
	DiffusionSamplerRegistryContract = "rocm-diffusion-sampler-registry-v1"

	DiffusionSamplerRouteName       = "block-diffusion-sampler-route"
	DiffusionSamplerRuntimeHIP      = "hip"
	DiffusionSamplerRuntimeMetadata = "metadata"
)

type DiffusionSamplerRouteStatus string

const (
	DiffusionSamplerExperimentalNative DiffusionSamplerRouteStatus = "experimental_native"
	DiffusionSamplerPlannedMetadata    DiffusionSamplerRouteStatus = "planned_metadata"
)

// DiffusionSamplerRoute is the folder-owned block-diffusion sampler route.
// Model packages can register these routes without importing the root rocm
// package, while HIP execution remains explicit through runtime metadata.
type DiffusionSamplerRoute struct {
	Contract               string                         `json:"contract,omitempty"`
	Name                   string                         `json:"name,omitempty"`
	Architecture           string                         `json:"architecture,omitempty"`
	Family                 string                         `json:"family,omitempty"`
	Runtime                string                         `json:"runtime,omitempty"`
	RuntimeStatus          inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	Status                 DiffusionSamplerRouteStatus    `json:"status,omitempty"`
	Reference              string                         `json:"reference,omitempty"`
	DiffusionRuntime       string                         `json:"diffusion_runtime,omitempty"`
	SamplerRuntime         string                         `json:"sampler_runtime,omitempty"`
	TrunkRuntime           string                         `json:"trunk_runtime,omitempty"`
	ExecutionStatus        string                         `json:"execution_status,omitempty"`
	Fallback               string                         `json:"fallback,omitempty"`
	Registered             bool                           `json:"registered,omitempty"`
	NativeRuntime          bool                           `json:"native_runtime,omitempty"`
	BlockDiffusion         bool                           `json:"block_diffusion,omitempty"`
	Sampler                bool                           `json:"sampler,omitempty"`
	Trunk                  bool                           `json:"trunk,omitempty"`
	Generation             bool                           `json:"generation,omitempty"`
	SelfConditioning       bool                           `json:"self_conditioning,omitempty"`
	EncoderLayerScalars    bool                           `json:"encoder_layer_scalars,omitempty"`
	GlobalCanvasMask       bool                           `json:"global_canvas_mask,omitempty"`
	BlockLocalCanvasMask   bool                           `json:"block_local_canvas_mask,omitempty"`
	KVCacheRollback        bool                           `json:"kv_cache_rollback,omitempty"`
	Streaming              bool                           `json:"streaming,omitempty"`
	Staged                 bool                           `json:"staged,omitempty"`
	Planned                bool                           `json:"planned,omitempty"`
	FallbackRefused        bool                           `json:"fallback_refused,omitempty"`
	CanvasLength           int                            `json:"canvas_length,omitempty"`
	DefaultCanvasLength    int                            `json:"default_canvas_length,omitempty"`
	ReferenceCanvasLength  int                            `json:"reference_canvas_length,omitempty"`
	DefaultMaxSteps        int                            `json:"default_max_steps,omitempty"`
	ReferenceMaxSteps      int                            `json:"reference_max_steps,omitempty"`
	StabilityThreshold     int                            `json:"stability_threshold,omitempty"`
	ConfidenceThreshold    float64                        `json:"confidence_threshold,omitempty"`
	EntropyBound           float64                        `json:"entropy_bound,omitempty"`
	MaxTemperature         float64                        `json:"max_temperature,omitempty"`
	MinTemperature         float64                        `json:"min_temperature,omitempty"`
	TemperatureExponent    float64                        `json:"temperature_exponent,omitempty"`
	RequiredFiles          []string                       `json:"required_files,omitempty"`
	OptionalFiles          []string                       `json:"optional_files,omitempty"`
	RequiredWeightLeaves   []string                       `json:"required_weight_leaves,omitempty"`
	OptionalWeightPrefixes []string                       `json:"optional_weight_prefixes,omitempty"`
	Labels                 map[string]string              `json:"labels,omitempty"`
}

func (route DiffusionSamplerRoute) Matched() bool {
	return route.Contract != "" && route.Name != "" && route.Architecture != "" && route.BlockDiffusion
}

func (route DiffusionSamplerRoute) Clone() DiffusionSamplerRoute {
	route.RequiredFiles = append([]string(nil), route.RequiredFiles...)
	route.OptionalFiles = append([]string(nil), route.OptionalFiles...)
	route.RequiredWeightLeaves = append([]string(nil), route.RequiredWeightLeaves...)
	route.OptionalWeightPrefixes = append([]string(nil), route.OptionalWeightPrefixes...)
	route.Labels = cloneStringMap(route.Labels)
	return route
}

func (route DiffusionSamplerRoute) WithLabels(labels map[string]string) DiffusionSamplerRoute {
	route = route.withLabels(labels)
	route.finalize()
	return route.Clone()
}

var registeredDiffusionSamplers = registry.NewOrdered[string, DiffusionSamplerRoute]()

// RegisterDiffusionSamplerRoute registers or replaces sampler metadata by
// architecture.
func RegisterDiffusionSamplerRoute(route DiffusionSamplerRoute) {
	route = NormalizeDiffusionSamplerRoute(route)
	if !route.Matched() {
		return
	}
	registeredDiffusionSamplers.Put(route.Architecture, route)
}

func RegisteredDiffusionSamplerArchitectures() []string {
	return registeredDiffusionSamplers.Keys()
}

func RegisteredDiffusionSamplerRoutes() []DiffusionSamplerRoute {
	return registeredDiffusionSamplerSnapshot()
}

func ReplaceRegisteredDiffusionSamplerRoutes(routes []DiffusionSamplerRoute) {
	order := make([]string, 0, len(routes))
	values := make(map[string]DiffusionSamplerRoute, len(routes))
	for _, route := range routes {
		route = NormalizeDiffusionSamplerRoute(route)
		if !route.Matched() {
			continue
		}
		if _, ok := values[route.Architecture]; !ok {
			order = append(order, route.Architecture)
		}
		values[route.Architecture] = route
	}
	registeredDiffusionSamplers.Restore(order, values)
}

func RegisteredDiffusionSamplerRouteForArchitecture(architecture string) (DiffusionSamplerRoute, bool) {
	return registeredDiffusionSamplerForArchitecture(architecture)
}

func DiffusionSamplerRouteForArchitecture(architecture string) (DiffusionSamplerRoute, bool) {
	architecture = profile.ArchitectureID(architecture)
	if architecture == "" {
		return DiffusionSamplerRoute{}, false
	}
	if route, ok := registeredDiffusionSamplerForArchitecture(architecture); ok {
		return route, true
	}
	architectureProfile, ok := profile.LookupArchitectureProfile(architecture)
	if !ok {
		return DiffusionSamplerRoute{}, false
	}
	route := staticDiffusionSamplerRoute(architectureProfile.ID, firstNonEmpty(architectureProfile.Family, architectureProfile.ID))
	if !route.Matched() {
		return DiffusionSamplerRoute{}, false
	}
	return route, true
}

func DiffusionSamplerRouteForIdentity(path string, identity inference.ModelIdentity) (DiffusionSamplerRoute, bool) {
	if identity.Path == "" {
		identity.Path = path
	}
	architecture := firstNonEmpty(
		identity.Labels["engine_architecture_resolved"],
		identity.Labels["architecture_resolved"],
		identity.Architecture,
	)
	route, ok := DiffusionSamplerRouteForArchitecture(architecture)
	if ok {
		return route.WithLabels(identity.Labels), true
	}
	route = staticDiffusionSamplerRoute(diffusionSamplerArchitecture(architecture, identity.Labels), "")
	route = route.WithLabels(identity.Labels)
	if !route.Matched() {
		return DiffusionSamplerRoute{}, false
	}
	return route, true
}

func DiffusionSamplerRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (DiffusionSamplerRoute, bool) {
	return DiffusionSamplerRouteForIdentity(path, inference.ModelIdentity{
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

func DiffusionSamplerRouteForInspection(inspection *inference.ModelPackInspection) (DiffusionSamplerRoute, bool) {
	if inspection == nil {
		return DiffusionSamplerRoute{}, false
	}
	identity := inspection.Model
	if identity.Path == "" {
		identity.Path = inspection.Path
	}
	labels := mergeDiffusionLabels(identity.Labels, inspection.Labels)
	identity.Labels = labels
	return DiffusionSamplerRouteForIdentity(identity.Path, identity)
}

func DefaultDiffusionSamplerRoutes() []DiffusionSamplerRoute {
	architectures := []string{"diffusion_gemma"}
	routes := make([]DiffusionSamplerRoute, 0, len(architectures)+len(registeredDiffusionSamplers.Keys()))
	seen := map[string]int{}
	for _, architecture := range architectures {
		route, ok := DiffusionSamplerRouteForArchitecture(architecture)
		if !ok {
			continue
		}
		seen[route.Architecture] = len(routes)
		routes = append(routes, route)
	}
	for _, route := range registeredDiffusionSamplerSnapshot() {
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
	return cloneDiffusionSamplerRoutes(routes)
}

func NormalizeDiffusionSamplerRoute(route DiffusionSamplerRoute) DiffusionSamplerRoute {
	route.Architecture = profile.ArchitectureID(route.Architecture)
	if route.Architecture == "" {
		return DiffusionSamplerRoute{}
	}
	architectureProfile, hasProfile := profile.LookupArchitectureProfile(route.Architecture)
	if route.Contract == "" {
		route.Contract = DiffusionSamplerRegistryContract
	}
	if route.Name == "" {
		route.Name = DiffusionSamplerRouteName
	}
	if route.Family == "" && hasProfile {
		route.Family = firstNonEmpty(architectureProfile.Family, architectureProfile.ID)
	}
	if route.Family == "" {
		route.Family = route.Architecture
	}
	if route.Runtime == "" {
		route.Runtime = DiffusionSamplerRuntimeMetadata
	}
	if len(route.RequiredFiles) == 0 {
		route.RequiredFiles = []string{"config.json", "tokenizer.json"}
	}
	if len(route.OptionalFiles) == 0 {
		route.OptionalFiles = []string{"tokenizer_config.json", "model.safetensors.index.json", "model.safetensors"}
	}
	if len(route.RequiredWeightLeaves) == 0 {
		route.RequiredWeightLeaves = []string{"self_conditioning.pre_norm.weight", "self_conditioning.gate_proj.weight", "self_conditioning.up_proj.weight", "self_conditioning.down_proj.weight"}
	}
	if len(route.OptionalWeightPrefixes) == 0 {
		route.OptionalWeightPrefixes = []string{"model.encoder.language_model.layers.", "model.decoder.", "model.language_model."}
	}
	route.BlockDiffusion = route.BlockDiffusion || route.Sampler || route.SelfConditioning || route.KVCacheRollback
	route.Registered = route.Architecture != "" && route.BlockDiffusion
	route = diffusionSamplerWithDefaults(route)
	route = diffusionSamplerWithRuntimeDefaults(route)
	route.finalize()
	return route.Clone()
}

func registeredDiffusionSamplerForArchitecture(architecture string) (DiffusionSamplerRoute, bool) {
	route, ok := registeredDiffusionSamplers.Get(profile.ArchitectureID(architecture))
	if !ok {
		return DiffusionSamplerRoute{}, false
	}
	return route.Clone(), true
}

func registeredDiffusionSamplerSnapshot() []DiffusionSamplerRoute {
	routes := registeredDiffusionSamplers.Values()
	out := make([]DiffusionSamplerRoute, 0, len(routes))
	for _, route := range routes {
		out = append(out, route.Clone())
	}
	return out
}

func staticDiffusionSamplerRoute(architecture, family string) DiffusionSamplerRoute {
	architecture = profile.ArchitectureID(architecture)
	route := DiffusionSamplerRoute{
		Contract:               DiffusionSamplerRegistryContract,
		Name:                   DiffusionSamplerRouteName,
		Architecture:           architecture,
		Family:                 family,
		Runtime:                DiffusionSamplerRuntimeMetadata,
		RuntimeStatus:          inference.FeatureRuntimeMetadataOnly,
		RequiredFiles:          []string{"config.json", "tokenizer.json"},
		OptionalFiles:          []string{"tokenizer_config.json", "model.safetensors.index.json", "model.safetensors"},
		RequiredWeightLeaves:   []string{"self_conditioning.pre_norm.weight", "self_conditioning.gate_proj.weight", "self_conditioning.up_proj.weight", "self_conditioning.down_proj.weight"},
		OptionalWeightPrefixes: []string{"model.encoder.language_model.layers.", "model.decoder.", "model.language_model."},
		DefaultCanvasLength:    64,
		ReferenceCanvasLength:  256,
		DefaultMaxSteps:        16,
		ReferenceMaxSteps:      48,
		StabilityThreshold:     1,
		ConfidenceThreshold:    0.005,
		EntropyBound:           0.3,
		MaxTemperature:         0.8,
		MinTemperature:         0.4,
		TemperatureExponent:    1.0,
	}
	switch architecture {
	case "diffusion_gemma":
		route.Reference = "go_mlx_diffusion_gemma"
		route.DiffusionRuntime = KernelStatusNotLinked
		route.SamplerRuntime = KernelStatusNotLinked
		route.TrunkRuntime = "model_pack_metadata"
		route.ExecutionStatus = KernelStatusNotLinked
		route.Fallback = "refused"
		route.BlockDiffusion = true
		route.Sampler = true
		route.Trunk = true
		route.Generation = true
		route.SelfConditioning = true
		route.EncoderLayerScalars = true
		route.GlobalCanvasMask = true
		route.BlockLocalCanvasMask = true
		route.KVCacheRollback = true
		route.Streaming = true
		route.FallbackRefused = true
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

func (route DiffusionSamplerRoute) withLabels(labels map[string]string) DiffusionSamplerRoute {
	if len(labels) == 0 {
		return route
	}
	if labels["block_diffusion_model"] == "true" {
		route.BlockDiffusion = true
	}
	route.DiffusionRuntime = firstNonEmpty(labels["diffusion_runtime"], route.DiffusionRuntime)
	route.SamplerRuntime = firstNonEmpty(labels["diffusion_sampler_runtime"], route.SamplerRuntime)
	route.TrunkRuntime = firstNonEmpty(labels["diffusion_trunk_runtime"], route.TrunkRuntime)
	route.Reference = firstNonEmpty(labels["diffusion_reference"], route.Reference)
	route.Fallback = firstNonEmpty(labels["diffusion_fallback"], labels["reactive_diffusion_fallback"], route.Fallback)
	route.ExecutionStatus = firstNonEmpty(labels["diffusion_execution_status"], route.ExecutionStatus)
	route.CanvasLength = firstPositiveInt(diffusionLabelInt(labels["diffusion_canvas_length"]), route.CanvasLength)
	route.DefaultCanvasLength = firstPositiveInt(diffusionLabelInt(labels["diffusion_default_canvas_length"]), route.DefaultCanvasLength)
	route.ReferenceCanvasLength = firstPositiveInt(diffusionLabelInt(labels["diffusion_reference_canvas_length"]), route.ReferenceCanvasLength)
	route.DefaultMaxSteps = firstPositiveInt(diffusionLabelInt(labels["diffusion_default_max_steps"]), route.DefaultMaxSteps)
	route.ReferenceMaxSteps = firstPositiveInt(diffusionLabelInt(labels["diffusion_reference_max_steps"]), route.ReferenceMaxSteps)
	route.StabilityThreshold = firstPositiveInt(diffusionLabelInt(labels["diffusion_stability_threshold"]), route.StabilityThreshold)
	route.ConfidenceThreshold = firstPositiveFloat(diffusionLabelFloat(labels["diffusion_confidence_threshold"]), route.ConfidenceThreshold)
	route.EntropyBound = firstPositiveFloat(diffusionLabelFloat(labels["diffusion_entropy_bound"]), route.EntropyBound)
	route.MaxTemperature = firstPositiveFloat(diffusionLabelFloat(labels["diffusion_max_temperature"]), route.MaxTemperature)
	route.MinTemperature = firstPositiveFloat(diffusionLabelFloat(labels["diffusion_min_temperature"]), route.MinTemperature)
	route.TemperatureExponent = firstPositiveFloat(diffusionLabelFloat(labels["diffusion_temperature_exponent"]), route.TemperatureExponent)
	if route.Architecture == "" {
		route.Architecture = profile.ArchitectureID(firstNonEmpty(labels["architecture_model_type"], labels["engine_architecture_resolved"], labels["architecture_resolved"]))
	}
	return route
}

func (route *DiffusionSamplerRoute) finalize() {
	if route == nil {
		return
	}
	route.Architecture = profile.ArchitectureID(route.Architecture)
	route.BlockDiffusion = route.BlockDiffusion || route.Architecture == "diffusion_gemma"
	if route.BlockDiffusion {
		route.Sampler = true
		route.Trunk = true
		route.Generation = true
	}
	route.Registered = route.Architecture != "" && route.BlockDiffusion
	route.NativeRuntime = route.Registered && route.DiffusionRuntime == KernelStatusLinked && route.SamplerRuntime == KernelStatusLinked
	if route.NativeRuntime {
		route.Runtime = DiffusionSamplerRuntimeHIP
		route.RuntimeStatus = inference.FeatureRuntimeExperimental
		route.Status = DiffusionSamplerExperimentalNative
		route.ExecutionStatus = "ready"
		route.Staged = false
		route.Planned = false
	} else if route.Registered {
		route.Runtime = firstNonEmpty(route.Runtime, DiffusionSamplerRuntimeMetadata)
		if route.RuntimeStatus == "" {
			route.RuntimeStatus = inference.FeatureRuntimeMetadataOnly
		}
		route.Status = DiffusionSamplerPlannedMetadata
		route.ExecutionStatus = firstNonEmpty(route.ExecutionStatus, KernelStatusNotLinked)
		route.Staged = true
		route.Planned = true
	}
	route.FallbackRefused = route.Fallback == "refused" || route.FallbackRefused
	if route.Fallback == "" && route.FallbackRefused {
		route.Fallback = "refused"
	}
	if route.CanvasLength == 0 {
		route.CanvasLength = route.ReferenceCanvasLength
	}
	route.Labels = diffusionSamplerRouteLabels(*route)
}

func diffusionSamplerWithDefaults(route DiffusionSamplerRoute) DiffusionSamplerRoute {
	route.DefaultCanvasLength = firstPositiveInt(route.DefaultCanvasLength, 64)
	route.ReferenceCanvasLength = firstPositiveInt(route.ReferenceCanvasLength, 256)
	route.DefaultMaxSteps = firstPositiveInt(route.DefaultMaxSteps, 16)
	route.ReferenceMaxSteps = firstPositiveInt(route.ReferenceMaxSteps, 48)
	route.StabilityThreshold = firstPositiveInt(route.StabilityThreshold, 1)
	route.ConfidenceThreshold = firstPositiveFloat(route.ConfidenceThreshold, 0.005)
	route.EntropyBound = firstPositiveFloat(route.EntropyBound, 0.3)
	route.MaxTemperature = firstPositiveFloat(route.MaxTemperature, 0.8)
	route.MinTemperature = firstPositiveFloat(route.MinTemperature, 0.4)
	route.TemperatureExponent = firstPositiveFloat(route.TemperatureExponent, 1.0)
	if route.BlockDiffusion && route.TrunkRuntime == "" {
		route.TrunkRuntime = "model_pack_metadata"
	}
	if route.BlockDiffusion && !route.NativeRuntime && route.Fallback == "" {
		route.Fallback = "refused"
	}
	return route
}

func diffusionSamplerWithRuntimeDefaults(route DiffusionSamplerRoute) DiffusionSamplerRoute {
	runtime := KernelStatusNotLinked
	if route.NativeRuntime {
		runtime = KernelStatusLinked
	}
	if route.BlockDiffusion || route.Sampler {
		route.DiffusionRuntime = firstNonEmpty(route.DiffusionRuntime, runtime)
		route.SamplerRuntime = firstNonEmpty(route.SamplerRuntime, runtime)
	}
	return route
}

func diffusionSamplerArchitecture(architecture string, labels map[string]string) string {
	if labels["block_diffusion_model"] == "true" {
		if architecture := profile.ArchitectureID(labels["architecture_model_type"]); architecture == "diffusion_gemma" {
			return architecture
		}
	}
	if architecture := profile.ArchitectureID(architecture); architecture != "" {
		return architecture
	}
	return profile.ArchitectureID(firstNonEmpty(labels["engine_architecture_resolved"], labels["architecture_resolved"]))
}

func diffusionSamplerRouteLabels(route DiffusionSamplerRoute) map[string]string {
	if !route.Matched() {
		return nil
	}
	labels := map[string]string{
		"engine_diffusion_sampler_route_contract":       route.Contract,
		"engine_diffusion_sampler_route":                route.Name,
		"engine_diffusion_sampler_runtime":              route.Runtime,
		"engine_diffusion_sampler_status":               string(route.Status),
		"engine_diffusion_sampler_registered":           strconv.FormatBool(route.Registered),
		"engine_diffusion_sampler_native_runtime":       strconv.FormatBool(route.NativeRuntime),
		"engine_diffusion_sampler_block_diffusion":      strconv.FormatBool(route.BlockDiffusion),
		"engine_diffusion_sampler_sampler":              strconv.FormatBool(route.Sampler),
		"engine_diffusion_sampler_trunk":                strconv.FormatBool(route.Trunk),
		"engine_diffusion_sampler_generation":           strconv.FormatBool(route.Generation),
		"engine_diffusion_sampler_self_conditioning":    strconv.FormatBool(route.SelfConditioning),
		"engine_diffusion_sampler_encoder_scalars":      strconv.FormatBool(route.EncoderLayerScalars),
		"engine_diffusion_sampler_global_canvas_mask":   strconv.FormatBool(route.GlobalCanvasMask),
		"engine_diffusion_sampler_block_local_mask":     strconv.FormatBool(route.BlockLocalCanvasMask),
		"engine_diffusion_sampler_kv_cache_rollback":    strconv.FormatBool(route.KVCacheRollback),
		"engine_diffusion_sampler_streaming":            strconv.FormatBool(route.Streaming),
		"engine_diffusion_sampler_staged":               strconv.FormatBool(route.Staged),
		"engine_diffusion_sampler_planned":              strconv.FormatBool(route.Planned),
		"engine_diffusion_sampler_fallback_refused":     strconv.FormatBool(route.FallbackRefused),
		"engine_diffusion_sampler_required_files":       joinNonEmptyStrings(route.RequiredFiles, ","),
		"engine_diffusion_sampler_optional_files":       joinNonEmptyStrings(route.OptionalFiles, ","),
		"engine_diffusion_sampler_required_weight_leaf": joinNonEmptyStrings(route.RequiredWeightLeaves, ","),
		"engine_diffusion_sampler_optional_weight_root": joinNonEmptyStrings(route.OptionalWeightPrefixes, ","),
	}
	if route.Architecture != "" {
		labels["engine_diffusion_sampler_architecture"] = route.Architecture
	}
	if route.Family != "" {
		labels["engine_diffusion_sampler_family"] = route.Family
	}
	if route.RuntimeStatus != "" {
		labels["engine_diffusion_sampler_runtime_status"] = string(route.RuntimeStatus)
	}
	setStringLabel(labels, "engine_diffusion_sampler_reference", route.Reference)
	setStringLabel(labels, "engine_diffusion_sampler_diffusion_runtime", route.DiffusionRuntime)
	setStringLabel(labels, "engine_diffusion_sampler_sampler_runtime", route.SamplerRuntime)
	setStringLabel(labels, "engine_diffusion_sampler_trunk_runtime", route.TrunkRuntime)
	setStringLabel(labels, "engine_diffusion_sampler_execution_status", route.ExecutionStatus)
	setStringLabel(labels, "engine_diffusion_sampler_fallback", route.Fallback)
	setIntLabel(labels, "engine_diffusion_sampler_canvas_length", route.CanvasLength)
	setIntLabel(labels, "engine_diffusion_sampler_default_canvas_length", route.DefaultCanvasLength)
	setIntLabel(labels, "engine_diffusion_sampler_reference_canvas_length", route.ReferenceCanvasLength)
	setIntLabel(labels, "engine_diffusion_sampler_default_max_steps", route.DefaultMaxSteps)
	setIntLabel(labels, "engine_diffusion_sampler_reference_max_steps", route.ReferenceMaxSteps)
	setIntLabel(labels, "engine_diffusion_sampler_stability_threshold", route.StabilityThreshold)
	setFloatLabel(labels, "engine_diffusion_sampler_confidence_threshold", route.ConfidenceThreshold)
	setFloatLabel(labels, "engine_diffusion_sampler_entropy_bound", route.EntropyBound)
	setFloatLabel(labels, "engine_diffusion_sampler_max_temperature", route.MaxTemperature)
	setFloatLabel(labels, "engine_diffusion_sampler_min_temperature", route.MinTemperature)
	setFloatLabel(labels, "engine_diffusion_sampler_temperature_exponent", route.TemperatureExponent)
	return labels
}

// DiffusionSamplerRouteLabels returns the normalized model-owned label contract
// for a diffusion sampler route.
func DiffusionSamplerRouteLabels(route DiffusionSamplerRoute) map[string]string {
	route = NormalizeDiffusionSamplerRoute(route)
	return cloneStringMap(route.Labels)
}

func setFloatLabel(labels map[string]string, key string, value float64) {
	if value > 0 {
		labels[key] = strconv.FormatFloat(value, 'g', -1, 64)
	}
}

func diffusionLabelInt(value string) int {
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

func diffusionLabelFloat(value string) float64 {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil || parsed < 0 {
		return 0
	}
	return parsed
}

func firstPositiveFloat(values ...float64) float64 {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func mergeDiffusionLabels(left, right map[string]string) map[string]string {
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

func cloneDiffusionSamplerRoutes(routes []DiffusionSamplerRoute) []DiffusionSamplerRoute {
	out := append([]DiffusionSamplerRoute(nil), routes...)
	for i := range out {
		out[i] = out[i].Clone()
	}
	return out
}
