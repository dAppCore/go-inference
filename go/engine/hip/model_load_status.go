// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"strconv"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const (
	rocmModelLoadStatusContract      = "rocm-model-load-status-v1"
	ROCmModelLoaderRegistryContract  = "rocm-model-loader-registry-v1"
	rocmModelLoaderRuntimeHIP        = "hip"
	rocmModelLoaderRuntimeMetadata   = "metadata"
	rocmModelLoaderRegistryRouteName = "architecture-loader"
)

type ROCmModelLoadStatusID = string

const (
	ROCmModelLoadStandaloneNative ROCmModelLoadStatusID = "standalone_native"
	ROCmModelLoadStagedNative     ROCmModelLoadStatusID = "staged_native"
	ROCmModelLoadAttachedOnly     ROCmModelLoadStatusID = "attached_only"
	ROCmModelLoadMetadataOnly     ROCmModelLoadStatusID = "metadata_only"
)

type ROCmModelLoadStatus struct {
	Contract         string                         `json:"contract,omitempty"`
	Architecture     string                         `json:"architecture,omitempty"`
	Family           string                         `json:"family,omitempty"`
	Loader           string                         `json:"loader,omitempty"`
	LoaderRuntime    string                         `json:"loader_runtime,omitempty"`
	LoaderContract   string                         `json:"loader_contract,omitempty"`
	Status           ROCmModelLoadStatusID          `json:"status,omitempty"`
	Target           string                         `json:"target,omitempty"`
	RuntimeStatus    inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	Reason           string                         `json:"reason,omitempty"`
	LoaderRegistered bool                           `json:"loader_registered,omitempty"`
	NativeRuntime    bool                           `json:"native_runtime,omitempty"`
	Standalone       bool                           `json:"standalone,omitempty"`
	AttachedOnly     bool                           `json:"attached_only,omitempty"`
	Staged           bool                           `json:"staged,omitempty"`
	MetadataOnly     bool                           `json:"metadata_only,omitempty"`
	TextGenerate     bool                           `json:"text_generate,omitempty"`
	Labels           map[string]string              `json:"labels,omitempty"`
}

func (status ROCmModelLoadStatus) clone() ROCmModelLoadStatus {
	status.Labels = cloneStringMap(status.Labels)
	return status
}

func (status ROCmModelLoadStatus) empty() bool {
	return status.Contract == "" &&
		status.Architecture == "" &&
		status.Family == "" &&
		status.Loader == "" &&
		status.LoaderRuntime == "" &&
		status.LoaderContract == "" &&
		status.Status == "" &&
		status.Target == "" &&
		status.RuntimeStatus == "" &&
		status.Reason == "" &&
		!status.LoaderRegistered &&
		!status.NativeRuntime &&
		!status.Standalone &&
		!status.AttachedOnly &&
		!status.Staged &&
		!status.MetadataOnly &&
		!status.TextGenerate &&
		len(status.Labels) == 0
}

func ROCmModelLoadStatusForProfile(profile ROCmModelProfile) ROCmModelLoadStatus {
	architectureProfile := profile.ArchitectureProfile
	if architectureProfile.ID == "" {
		architectureProfile = profile.Gemma4Settings
	}
	if architectureProfile.ID == "" {
		if resolved, ok := ROCmArchitectureProfileForArchitecture(profile.Architecture); ok {
			architectureProfile = resolved
		}
	}
	features := profile.EngineFeatures
	if features.empty() {
		features = ROCmEngineFeaturesForProfile(profile)
	}
	status := ROCmModelLoadStatus{
		Contract:      rocmModelLoadStatusContract,
		Architecture:  firstNonEmptyString(profile.Architecture, architectureProfile.ID, features.Architecture),
		Family:        firstNonEmptyString(profile.Family, architectureProfile.Family, features.Family),
		RuntimeStatus: architectureProfile.RuntimeStatus,
		NativeRuntime: architectureProfile.NativeRuntime,
		AttachedOnly:  architectureProfile.AttachedOnly,
		TextGenerate:  features.TextGenerate,
	}
	if status.Architecture == "" {
		status.Architecture = features.Architecture
	}
	if status.Family == "" {
		status.Family = status.Architecture
	}
	switch {
	case architectureProfile.AttachedOnly:
		status.Status = ROCmModelLoadAttachedOnly
		status.Target = "attached"
		status.Reason = "architecture is declared as an attached drafter and must load beside a target model"
	case !architectureProfile.NativeRuntime:
		status.Status = ROCmModelLoadMetadataOnly
		status.Target = "metadata"
		status.MetadataOnly = true
		status.Reason = "architecture is recognised by the registry but has no native runtime loader yet"
	case features.TextGenerate:
		status.Status = ROCmModelLoadStandaloneNative
		status.Target = "standalone"
		status.Standalone = true
		status.Reason = "native standalone text-generation path is advertised by the resolved model profile"
	default:
		status.Status = ROCmModelLoadStagedNative
		status.Target = "standalone"
		status.Standalone = true
		status.Staged = true
		status.Reason = "native metadata/config loader is staged while standalone generation remains pending"
	}
	route := ROCmModelLoaderRouteForStatus(status)
	status = rocmModelLoadStatusWithRoute(status, route)
	status.Labels = rocmModelLoadStatusLabels(status)
	return status
}

func rocmModelLoadStatusWithRoute(status ROCmModelLoadStatus, route ROCmModelLoaderRoute) ROCmModelLoadStatus {
	if !route.Matched() {
		return status
	}
	if route.Architecture != "" {
		status.Architecture = route.Architecture
	}
	if route.Family != "" {
		status.Family = route.Family
	}
	if route.Status != "" {
		status.Status = ROCmModelLoadStatusID(route.Status)
	}
	if route.Target != "" {
		status.Target = route.Target
	}
	if route.RuntimeStatus != "" {
		status.RuntimeStatus = route.RuntimeStatus
	}
	if route.Reason != "" {
		status.Reason = route.Reason
	}
	status.NativeRuntime = route.NativeRuntime
	status.Standalone = route.Standalone
	status.AttachedOnly = route.AttachedOnly
	status.Staged = route.Staged
	status.MetadataOnly = route.MetadataOnly
	status.TextGenerate = route.TextGenerate
	status.Loader = route.Loader
	status.LoaderRuntime = route.Runtime
	status.LoaderContract = route.Contract
	status.LoaderRegistered = route.Registered
	return status
}

func rocmModelLoadStatusFromLoaderRoute(route ROCmModelLoaderRoute) ROCmModelLoadStatus {
	if !route.Matched() {
		return ROCmModelLoadStatus{}
	}
	status := rocmModelLoadStatusWithRoute(ROCmModelLoadStatus{
		Contract: rocmModelLoadStatusContract,
	}, route)
	status.Labels = rocmModelLoadStatusLabels(status)
	return status.clone()
}

func ROCmModelLoadStatusForIdentity(path string, model inference.ModelIdentity) (ROCmModelLoadStatus, bool) {
	profile, ok := ResolveROCmModelProfile(path, model)
	if !ok {
		return ROCmModelLoadStatus{}, false
	}
	return profile.LoadStatus.clone(), true
}

func ROCmModelLoadStatusForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmModelLoadStatus, bool) {
	profile, ok := ResolveROCmModelProfileForInfo(path, info, labels)
	if !ok {
		return ROCmModelLoadStatus{}, false
	}
	return profile.LoadStatus.clone(), true
}

func ROCmModelLoadStatusForInspection(inspection *inference.ModelPackInspection) (ROCmModelLoadStatus, bool) {
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok {
		return ROCmModelLoadStatus{}, false
	}
	return profile.LoadStatus.clone(), true
}

func rocmModelLoadStatusLabels(status ROCmModelLoadStatus) map[string]string {
	if status.empty() {
		return nil
	}
	labels := map[string]string{
		"engine_loader_contract":     firstNonEmptyString(status.LoaderContract, ROCmModelLoaderRegistryContract),
		"engine_loader_registered":   strconv.FormatBool(status.LoaderRegistered),
		"engine_load_contract":       firstNonEmptyString(status.Contract, rocmModelLoadStatusContract),
		"engine_load_status":         string(status.Status),
		"engine_load_native_runtime": strconv.FormatBool(status.NativeRuntime),
		"engine_load_standalone":     strconv.FormatBool(status.Standalone),
		"engine_load_attached_only":  strconv.FormatBool(status.AttachedOnly),
		"engine_load_staged":         strconv.FormatBool(status.Staged),
		"engine_load_metadata_only":  strconv.FormatBool(status.MetadataOnly),
		"engine_load_text_generate":  strconv.FormatBool(status.TextGenerate),
	}
	if status.Architecture != "" {
		labels["engine_load_architecture"] = status.Architecture
	}
	if status.Family != "" {
		labels["engine_load_family"] = status.Family
	}
	if status.Loader != "" {
		labels["engine_loader"] = status.Loader
	}
	if status.LoaderRuntime != "" {
		labels["engine_loader_runtime"] = status.LoaderRuntime
	}
	if status.Target != "" {
		labels["engine_load_target"] = status.Target
	}
	if status.RuntimeStatus != "" {
		labels["engine_load_runtime_status"] = string(status.RuntimeStatus)
	}
	if status.Reason != "" {
		labels["engine_load_reason"] = status.Reason
	}
	return labels
}

// ROCmModelLoaderRoute is the architecture-keyed loader route consumers can
// use before calling LoadModel. It mirrors go-mlx's model-loader registry at
// the contract layer while preserving ROCm's single HIP runtime loader.
type ROCmModelLoaderRoute = rocmmodel.LoaderRoute

// RegisterROCmModelLoaderRoute registers or replaces an architecture-keyed
// model loader route. It mirrors go-mlx's RegisterModelLoader contract at the
// ROCm API layer: a model family can self-register the loader metadata that
// go-ai/go-ml need before LoadModel, without adding another central switch.
func RegisterROCmModelLoaderRoute(route ROCmModelLoaderRoute) {
	route = normalizeRegisteredROCmModelLoaderRoute(route)
	if !route.Matched() {
		return
	}
	rocmmodel.RegisterLoaderRoute(route)
}

// RegisteredROCmModelLoaderRouteArchitectures returns extension loader
// architectures in resolution order. Built-in architecture-profile routes are
// intentionally not included.
func RegisteredROCmModelLoaderRouteArchitectures() []string {
	return rocmmodel.RegisteredLoaderArchitectures()
}

func normalizeRegisteredROCmModelLoaderRoute(route ROCmModelLoaderRoute) ROCmModelLoaderRoute {
	route.Architecture = ROCmArchitectureID(route.Architecture)
	if route.Architecture == "" {
		return ROCmModelLoaderRoute{}
	}
	if route.Contract == "" {
		route.Contract = ROCmModelLoaderRegistryContract
	}
	if route.Name == "" {
		route.Name = rocmModelLoaderRegistryRouteName
	}
	if route.Loader == "" {
		route.Loader = route.Architecture
	}
	if route.Family == "" {
		if profile, ok := ROCmArchitectureProfileForArchitecture(route.Architecture); ok {
			route.Family = firstNonEmptyString(profile.Family, profile.ID)
		}
	}
	if route.Family == "" {
		route.Family = route.Architecture
	}
	if route.Status == "" {
		route.Status = rocmModelLoaderRouteStatus(route)
	}
	route = rocmModelLoaderRouteWithStatusDefaults(route)
	if route.RuntimeStatus == "" {
		if profile, ok := ROCmArchitectureProfileForArchitecture(route.Architecture); ok {
			route.RuntimeStatus = profile.RuntimeStatus
		}
	}
	if route.RuntimeStatus == "" && route.NativeRuntime {
		route.RuntimeStatus = inference.FeatureRuntimeNative
	}
	route.Labels = rocmModelLoaderRouteLabels(route)
	return route.Clone()
}

func rocmModelLoaderRouteStatus(route ROCmModelLoaderRoute) string {
	switch {
	case route.MetadataOnly || route.Runtime == rocmModelLoaderRuntimeMetadata:
		return string(ROCmModelLoadMetadataOnly)
	case route.AttachedOnly:
		return string(ROCmModelLoadAttachedOnly)
	case route.Staged:
		return string(ROCmModelLoadStagedNative)
	case route.NativeRuntime || route.Registered || route.TextGenerate:
		return string(ROCmModelLoadStandaloneNative)
	default:
		return string(ROCmModelLoadMetadataOnly)
	}
}

func rocmModelLoaderRouteWithStatusDefaults(route ROCmModelLoaderRoute) ROCmModelLoaderRoute {
	switch route.Status {
	case string(ROCmModelLoadAttachedOnly):
		route.Target = firstNonEmptyString(route.Target, "attached")
		route.AttachedOnly = true
		route.NativeRuntime = true
		route.Registered = true
	case string(ROCmModelLoadMetadataOnly):
		route.Target = firstNonEmptyString(route.Target, "metadata")
		route.MetadataOnly = true
		route.NativeRuntime = false
		route.Registered = false
	case string(ROCmModelLoadStagedNative):
		route.Target = firstNonEmptyString(route.Target, "standalone")
		route.Standalone = true
		route.Staged = true
		route.NativeRuntime = true
		route.Registered = true
	case string(ROCmModelLoadStandaloneNative):
		route.Target = firstNonEmptyString(route.Target, "standalone")
		route.Standalone = true
		route.NativeRuntime = true
		route.Registered = true
		if !route.Staged {
			route.TextGenerate = true
		}
	}
	if route.Runtime == "" {
		route.Runtime = rocmModelLoaderRuntimeHIP
		if route.MetadataOnly || !route.NativeRuntime {
			route.Runtime = rocmModelLoaderRuntimeMetadata
		}
	}
	return route
}

func ROCmModelLoaderRouteForStatus(status ROCmModelLoadStatus) ROCmModelLoaderRoute {
	if status.empty() {
		return ROCmModelLoaderRoute{}
	}
	base := rocmModelLoaderRouteFromStatus(status)
	if registered, ok := rocmmodel.RegisteredLoaderRouteForArchitecture(status.Architecture); ok {
		return rocmMergeRegisteredModelLoaderRoute(base, rocmModelLoaderRouteFromModel(registered))
	}
	if modelRoute, ok := rocmmodel.LoaderRouteForArchitecture(status.Architecture); ok {
		return rocmMergeRegisteredModelLoaderRoute(base, rocmModelLoaderRouteFromModel(modelRoute))
	}
	return base
}

func rocmModelLoaderRouteFromStatus(status ROCmModelLoadStatus) ROCmModelLoaderRoute {
	runtime := rocmModelLoaderRuntimeHIP
	if status.MetadataOnly || !status.NativeRuntime {
		runtime = rocmModelLoaderRuntimeMetadata
	}
	route := ROCmModelLoaderRoute{
		Contract:      ROCmModelLoaderRegistryContract,
		Name:          rocmModelLoaderRegistryRouteName,
		Architecture:  status.Architecture,
		Family:        status.Family,
		Loader:        status.Architecture,
		Runtime:       runtime,
		Status:        string(status.Status),
		Target:        status.Target,
		RuntimeStatus: status.RuntimeStatus,
		Reason:        status.Reason,
		Registered:    status.NativeRuntime && !status.MetadataOnly,
		NativeRuntime: status.NativeRuntime,
		Standalone:    status.Standalone,
		AttachedOnly:  status.AttachedOnly,
		Staged:        status.Staged,
		MetadataOnly:  status.MetadataOnly,
		TextGenerate:  status.TextGenerate,
	}
	route.Labels = rocmModelLoaderRouteLabels(route)
	return route.Clone()
}

func rocmModelLoaderRouteWithStatus(route ROCmModelLoaderRoute, status ROCmModelLoadStatus) ROCmModelLoaderRoute {
	base := rocmModelLoaderRouteFromStatus(status)
	if !route.Matched() {
		return base
	}
	route.Contract = firstNonEmptyString(route.Contract, base.Contract)
	route.Name = firstNonEmptyString(route.Name, base.Name)
	route.Architecture = firstNonEmptyString(route.Architecture, base.Architecture)
	route.Family = firstNonEmptyString(route.Family, base.Family, route.Architecture)
	route.Loader = firstNonEmptyString(route.Loader, base.Loader)
	route.Runtime = firstNonEmptyString(base.Runtime, route.Runtime)
	route.Status = firstNonEmptyString(base.Status, route.Status)
	route.Target = firstNonEmptyString(base.Target, route.Target)
	route.RuntimeStatus = firstNonEmptyRuntimeStatus(base.RuntimeStatus, route.RuntimeStatus)
	route.Reason = firstNonEmptyString(base.Reason, route.Reason)
	route.Registered = base.Registered
	route.NativeRuntime = base.NativeRuntime
	route.Standalone = base.Standalone
	route.AttachedOnly = base.AttachedOnly
	route.Staged = base.Staged
	route.MetadataOnly = base.MetadataOnly
	route.TextGenerate = base.TextGenerate
	route.Labels = rocmModelLoaderRouteLabels(route)
	return route.Clone()
}

func rocmMergeRegisteredModelLoaderRoute(base, registered ROCmModelLoaderRoute) ROCmModelLoaderRoute {
	if !registered.Matched() {
		return base
	}
	if registered.Contract == "" {
		registered.Contract = base.Contract
	}
	if registered.Name == "" {
		registered.Name = base.Name
	}
	if registered.Architecture == "" {
		registered.Architecture = base.Architecture
	}
	if registered.Family == "" {
		registered.Family = base.Family
	}
	if registered.Loader == "" {
		registered.Loader = base.Loader
	}
	if registered.Runtime == "" {
		registered.Runtime = base.Runtime
	}
	if registered.Status == "" {
		registered.Status = base.Status
	}
	if registered.Target == "" {
		registered.Target = base.Target
	}
	if registered.RuntimeStatus == "" {
		registered.RuntimeStatus = base.RuntimeStatus
	}
	if registered.Reason == "" {
		registered.Reason = base.Reason
	}
	return registered.Clone()
}

func ROCmModelLoaderRouteForProfile(profile ROCmModelProfile) ROCmModelLoaderRoute {
	status := profile.LoadStatus
	if status.empty() {
		status = ROCmModelLoadStatusForProfile(profile)
	}
	model := rocmCloneModelIdentity(profile.Model)
	model.Labels = cloneStringMap(profile.Model.Labels)
	if model.Architecture == "" {
		model.Architecture = firstNonEmptyString(profile.Architecture, profile.ArchitectureProfile.ID, profile.Gemma4Settings.ID)
	}
	base := rocmModelLoaderRouteFromStatus(status)
	if registered, ok := rocmmodel.RegisteredLoaderRouteForArchitecture(model.Architecture); ok {
		return rocmMergeRegisteredModelLoaderRoute(base, rocmModelLoaderRouteFromModel(registered))
	}
	if modelRoute, ok := rocmmodel.LoaderRouteForIdentity(model.Path, model); ok {
		route := rocmMergeRegisteredModelLoaderRoute(base, rocmModelLoaderRouteFromModel(modelRoute))
		if route.Matched() {
			return route.Clone()
		}
	}
	return ROCmModelLoaderRouteForStatus(status).Clone()
}

func ROCmModelLoaderRouteForArchitecture(architecture string) (ROCmModelLoaderRoute, bool) {
	if registered, ok := rocmmodel.RegisteredLoaderRouteForArchitecture(architecture); ok {
		return rocmModelLoaderRouteFromModel(registered), true
	}
	modelRoute, ok := rocmmodel.LoaderRouteForArchitecture(architecture)
	if !ok {
		return ROCmModelLoaderRoute{}, false
	}
	route := rocmModelLoaderRouteFromModel(modelRoute)
	if !route.Matched() {
		return ROCmModelLoaderRoute{}, false
	}
	return route, true
}

func ROCmModelLoaderRouteForIdentity(path string, model inference.ModelIdentity) (ROCmModelLoaderRoute, bool) {
	profile, ok := ResolveROCmModelProfile(path, model)
	if !ok {
		return ROCmModelLoaderRoute{}, false
	}
	return ROCmModelLoaderRouteForProfile(profile), true
}

func ROCmModelLoaderRouteForInspection(inspection *inference.ModelPackInspection) (ROCmModelLoaderRoute, bool) {
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok {
		return ROCmModelLoaderRoute{}, false
	}
	return ROCmModelLoaderRouteForProfile(profile), true
}

func DefaultROCmModelLoaderRoutes() []ROCmModelLoaderRoute {
	modelRoutes := rocmmodel.DefaultLoaderRoutes()
	routes := make([]ROCmModelLoaderRoute, 0, len(modelRoutes))
	for _, modelRoute := range modelRoutes {
		route := rocmModelLoaderRouteFromModel(modelRoute)
		if route.Matched() {
			routes = append(routes, route)
		}
	}
	return routes
}

func rocmModelLoaderProfileForRoute(route ROCmModelLoaderRoute) ROCmModelProfile {
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

func rocmModelLoaderRouteFromModel(route rocmmodel.LoaderRoute) ROCmModelLoaderRoute {
	return normalizeRegisteredROCmModelLoaderRoute(route).Clone()
}

func rocmModelLoaderRouteLabels(route ROCmModelLoaderRoute) map[string]string {
	return rocmmodel.LoaderRouteLabels(route)
}

func rocmApplyROCmModelLoadStatusLabels(labels map[string]string, status ROCmModelLoadStatus) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if status.empty() {
		return labels
	}
	for key, value := range rocmModelLoadStatusLabels(status) {
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}
