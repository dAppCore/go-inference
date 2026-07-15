// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"strings"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

// ROCmModelProfileRequest is the public, backend-neutral input for a registered
// model-profile factory. Native load paths carry extra internal config context,
// but external factories should react to the model identity contract shared with
// go-ai/go-ml callers.
type ROCmModelProfileRequest struct {
	Path  string
	Model inference.ModelIdentity
}

// ROCmModelProfileFactory resolves a loaded or inspected model identity into a
// ROCm model profile. Registered factories run before the built-in Gemma-4 and
// architecture-profile factories, so model families can self-register without
// adding another central switch.
type ROCmModelProfileFactory interface {
	Name() string
	BuildROCmModelProfile(ROCmModelProfileRequest) (ROCmModelProfile, bool)
}

// RegisterROCmModelProfileFactory registers factory by name. A later factory
// with the same name replaces the existing factory while preserving resolution
// order, mirroring the override-friendly go-mlx registry style.
func RegisterROCmModelProfileFactory(factory ROCmModelProfileFactory) {
	if factory == nil {
		return
	}
	name := strings.TrimSpace(factory.Name())
	if name == "" {
		return
	}
	rocmmodel.RegisterProfileFactory(rocmModelProfileFactoryAdapter{factory: factory})
}

// RegisteredROCmModelProfileFactoryNames returns active model-owned and
// extension factory names in root resolution order. Generic architecture-profile
// fallback factories are kept last so concrete model-family registrations and
// caller extensions can react before the catch-all profile resolves.
func RegisteredROCmModelProfileFactoryNames() []string {
	factories := registeredROCmModelProfileFactoryAdapters()
	out := make([]string, 0, len(factories))
	for _, factory := range factories {
		if factory == nil {
			continue
		}
		if name := strings.TrimSpace(factory.Name()); name != "" {
			out = append(out, name)
		}
	}
	return out
}

func registeredROCmModelProfileFactoryAdapters() []rocmModelProfileFactory {
	factories := rocmmodel.RegisteredProfileFactories()
	out := make([]rocmModelProfileFactory, 0, len(factories))
	for _, factory := range factories {
		if factory == nil {
			continue
		}
		out = append(out, registeredROCmModelProfileFactory{factory: factory})
	}
	return rocmOrderModelProfileFactories(out)
}

func rocmOrderModelProfileFactories(factories []rocmModelProfileFactory) []rocmModelProfileFactory {
	if len(factories) == 0 {
		return nil
	}
	out := make([]rocmModelProfileFactory, 0, len(factories))
	var fallbacks []rocmModelProfileFactory
	for _, factory := range factories {
		if factory == nil {
			continue
		}
		if strings.TrimSpace(factory.Name()) == (genericROCmArchitectureProfileFactory{}).Name() {
			fallbacks = append(fallbacks, factory)
			continue
		}
		out = append(out, factory)
	}
	out = append(out, fallbacks...)
	return out
}

func appendROCmModelProfileFactoryFallbacks(factories []rocmModelProfileFactory, fallbacks ...rocmModelProfileFactory) []rocmModelProfileFactory {
	seen := map[string]struct{}{}
	for _, factory := range factories {
		if factory == nil {
			continue
		}
		if name := strings.TrimSpace(factory.Name()); name != "" {
			seen[name] = struct{}{}
		}
	}
	for _, fallback := range fallbacks {
		if fallback == nil {
			continue
		}
		name := strings.TrimSpace(fallback.Name())
		if name == "" {
			continue
		}
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		factories = append(factories, fallback)
	}
	return factories
}

type registeredROCmModelProfileFactory struct {
	factory rocmmodel.ProfileFactory
}

func (factory registeredROCmModelProfileFactory) Name() string {
	if factory.factory == nil {
		return ""
	}
	return strings.TrimSpace(factory.factory.Name())
}

func (factory registeredROCmModelProfileFactory) BuildROCmModelProfile(req rocmModelProfileRequest) (ROCmModelProfile, bool) {
	if factory.factory == nil {
		return ROCmModelProfile{}, false
	}
	profile, ok := rocmmodel.ResolveProfileFactory(factory.factory, rocmmodel.ProfileRequest{
		Path:  req.Path,
		Model: rocmCloneModelIdentity(req.Model),
	})
	if !ok || !profile.Matched() {
		return ROCmModelProfile{}, false
	}
	converted := rocmModelProfileFromModel(profile)
	converted.Labels = rocmRegisteredModelProfileFactoryLabels(converted.Labels, factory.Name(), converted)
	return converted, true
}

type rocmModelProfileFactoryAdapter struct {
	factory ROCmModelProfileFactory
}

func (factory rocmModelProfileFactoryAdapter) Name() string {
	if factory.factory == nil {
		return ""
	}
	return strings.TrimSpace(factory.factory.Name())
}

func (factory rocmModelProfileFactoryAdapter) BuildModelProfile(req rocmmodel.ProfileRequest) (rocmmodel.Profile, bool) {
	if factory.factory == nil {
		return rocmmodel.Profile{}, false
	}
	profile, ok := factory.factory.BuildROCmModelProfile(ROCmModelProfileRequest{
		Path:  req.Path,
		Model: rocmCloneModelIdentity(req.Model),
	})
	if !ok || !profile.Matched() {
		return rocmmodel.Profile{}, false
	}
	if profile.Model.Path == "" {
		profile.Model.Path = firstNonEmptyString(req.Model.Path, req.Path)
	}
	if profile.Model.Architecture == "" {
		profile.Model.Architecture = firstNonEmptyString(profile.Architecture, req.Model.Architecture)
	}
	if profile.Architecture == "" {
		profile.Architecture = firstNonEmptyString(profile.ArchitectureProfile.ID, profile.Model.Architecture)
	}
	if profile.Registry == "" {
		profile.Registry = rocmModelRegistryName
	}
	profile.Model.Labels = cloneStringMap(profile.Model.Labels)
	profile.Labels = rocmRegisteredModelProfileFactoryLabels(profile.Labels, factory.Name(), profile)
	return rocmModelProfileToModel(profile), true
}

func rocmRegisteredModelProfileFactoryLabels(labels map[string]string, factoryName string, profile ROCmModelProfile) map[string]string {
	labels = cloneStringMap(labels)
	if labels == nil {
		labels = map[string]string{}
	}
	setDefault := func(key, value string) {
		if labels[key] == "" && value != "" {
			labels[key] = value
		}
	}
	factoryName = strings.TrimSpace(factoryName)
	family := firstNonEmptyString(profile.Family, profile.Name, factoryName)
	architecture := firstNonEmptyString(profile.Architecture, profile.ArchitectureProfile.ID, profile.Model.Architecture)
	setDefault("engine_registry", rocmModelRegistryName)
	setDefault("engine_profile", firstNonEmptyString(profile.Name, factoryName))
	setDefault("engine_profile_family", family)
	setDefault("engine_profile_source", "registered_factory")
	setDefault("engine_profile_factory", factoryName)
	setDefault("engine_profile_matched", "true")
	setDefault("engine_profile_reactive", "true")
	setDefault("engine_profile_architecture", architecture)
	return labels
}

func rocmModelProfileToModel(profile ROCmModelProfile) rocmmodel.Profile {
	model := rocmCloneModelIdentity(profile.Model)
	architecture := firstNonEmptyString(profile.Architecture, profile.ArchitectureProfile.ID, profile.Gemma4Settings.ID, model.Architecture)
	if model.Architecture == "" {
		model.Architecture = architecture
	}
	family := firstNonEmptyString(profile.Family, profile.Name, architecture)
	routeSet := rocmmodel.RouteSet{
		Contract:     rocmmodel.RouteSetContract,
		Architecture: architecture,
		Family:       family,
		Model:        model,
	}
	if profile.FeatureRoute.Matched() {
		routeSet.FeatureRoute = profile.FeatureRoute.Clone()
	}
	if profile.CacheRoute.Matched() {
		routeSet.CacheRoute = profile.CacheRoute.Clone()
	}
	if profile.TokenizerRoute.Matched() {
		routeSet.TokenizerRoute = profile.TokenizerRoute.Clone()
	}
	if profile.LoRAAdapterRoute.Matched() {
		routeSet.LoRAAdapterRoute = profile.LoRAAdapterRoute.Clone()
	}
	if profile.MultimodalProcessorRoute.Matched() {
		routeSet.MultimodalProcessorRoute = profile.MultimodalProcessorRoute.Clone()
	}
	if profile.DiffusionSamplerRoute.Matched() {
		routeSet.DiffusionSamplerRoute = profile.DiffusionSamplerRoute.Clone()
	}
	if profile.StateContextRoute.Matched() {
		routeSet.StateContextRoute = profile.StateContextRoute.Clone()
	}
	if profile.AttachedDrafterRoute.Matched() {
		routeSet.AttachedDrafterRoute = profile.AttachedDrafterRoute.Clone()
	}
	if !profile.LoadStatus.empty() {
		routeSet.LoaderRoute = rocmModelLoaderRouteFromLoadStatus(profile.LoadStatus).Clone()
	}
	if profile.QuantLoaderRoute.Matched() {
		routeSet.QuantLoaderRoute = profile.QuantLoaderRoute.Clone()
	}
	if profile.RuntimeContractRoute.Matched() {
		routeSet.RuntimeContractRoute = profile.RuntimeContractRoute.Clone()
	}
	routeSet.Labels = cloneStringMap(profile.Labels)
	return rocmmodel.Profile{
		Contract:     rocmmodel.ProfileFactoryRegistryContract,
		Name:         profile.Name,
		Family:       family,
		Architecture: architecture,
		Registry:     firstNonEmptyString(profile.Registry, rocmModelRegistryName),
		Model:        model,
		RouteSet:     routeSet,
		Labels:       cloneStringMap(profile.Labels),
	}
}

func rocmModelProfileFromModel(profile rocmmodel.Profile) ROCmModelProfile {
	routeSet := profile.RouteSet
	architecture := firstNonEmptyString(profile.Architecture, routeSet.Architecture, profile.Model.Architecture)
	family := firstNonEmptyString(profile.Family, routeSet.Family, profile.Name, architecture)
	model := rocmCloneModelIdentity(profile.Model)
	if routeSet.Model.Path != "" || routeSet.Model.Architecture != "" || len(routeSet.Model.Labels) > 0 {
		routeSetModel := rocmCloneModelIdentity(routeSet.Model)
		if model.Path == "" {
			model.Path = routeSetModel.Path
		}
		if model.Architecture == "" {
			model.Architecture = routeSetModel.Architecture
		}
		model.Labels = rocmMergeModelProfileLabels(routeSetModel.Labels, model.Labels)
	}
	if model.Architecture == "" {
		model.Architecture = architecture
	}
	root := ROCmModelProfile{
		Name:         profile.Name,
		Family:       family,
		Architecture: architecture,
		Registry:     firstNonEmptyString(profile.Registry, rocmModelRegistryName),
		Model:        model,
		Labels:       rocmModelProfileLabelsFromModel(profile),
	}
	if architectureProfile, ok := ROCmArchitectureProfileForArchitecture(architecture); ok {
		root.ArchitectureProfile = architectureProfile
		root.Gemma4Settings = architectureProfile
	}
	if rocmModelProfilePreservesModelRouteSet(profile) {
		if routeSet.FeatureRoute.Matched() {
			root.FeatureRoute = rocmModelFeatureRouteFromModel(routeSet.FeatureRoute)
		}
		if routeSet.CacheRoute.Matched() {
			root.CacheRoute = routeSet.CacheRoute.Clone()
		}
		if routeSet.TokenizerRoute.Matched() {
			root.TokenizerRoute = rocmModelTokenizerRouteFromModel(routeSet.TokenizerRoute)
		}
		if routeSet.LoRAAdapterRoute.Matched() {
			root.LoRAAdapterRoute = rocmLoRAAdapterRouteFromModel(routeSet.LoRAAdapterRoute)
		}
		if routeSet.MultimodalProcessorRoute.Matched() {
			root.MultimodalProcessorRoute = rocmMultimodalProcessorRouteFromModel(routeSet.MultimodalProcessorRoute)
		}
		if routeSet.DiffusionSamplerRoute.Matched() {
			root.DiffusionSamplerRoute = rocmDiffusionSamplerRouteFromModel(routeSet.DiffusionSamplerRoute)
		}
		if routeSet.StateContextRoute.Matched() {
			root.StateContextRoute = rocmStateContextRouteFromModel(routeSet.StateContextRoute)
		}
		if routeSet.AttachedDrafterRoute.Matched() {
			root.AttachedDrafterRoute = rocmAttachedDrafterRouteFromModel(routeSet.AttachedDrafterRoute)
		}
		if routeSet.LoaderRoute.Matched() {
			root.LoadStatus = rocmModelLoadStatusFromLoaderRoute(rocmModelLoaderRouteFromModel(routeSet.LoaderRoute))
		}
		if routeSet.QuantLoaderRoute.Matched() {
			root.QuantLoaderRoute = rocmQuantLoaderRouteFromModel(routeSet.QuantLoaderRoute)
		}
		if len(routeSet.SequenceMixerRoutes) > 0 {
			root.SequenceMixerRoutes = rocmSequenceMixerLoaderRoutesFromModel(routeSet.SequenceMixerRoutes)
		}
		if routeSet.RuntimeContractRoute.Matched() {
			root.RuntimeContractRoute = routeSet.RuntimeContractRoute.Clone()
		}
	}
	return root.clone()
}

func rocmModelProfilePreservesModelRouteSet(profile rocmmodel.Profile) bool {
	return strings.TrimSpace(profile.Labels["engine_profile_source"]) != "architecture_profile"
}

func rocmModelProfileLabelsFromModel(profile rocmmodel.Profile) map[string]string {
	labels := cloneStringMap(profile.RouteSet.Labels)
	for key, value := range profile.Labels {
		if value == "" {
			continue
		}
		if labels == nil {
			labels = map[string]string{}
		}
		labels[key] = value
	}
	return labels
}

func rocmMergeModelProfileLabels(left, right map[string]string) map[string]string {
	out := cloneStringMap(left)
	for key, value := range right {
		if value == "" {
			continue
		}
		if out == nil {
			out = map[string]string{}
		}
		out[key] = value
	}
	return out
}

func rocmMergeModelProfileIdentityLabels(identity inference.ModelIdentity, labels map[string]string) inference.ModelIdentity {
	identity.Labels = rocmMergeModelProfileLabels(identity.Labels, labels)
	return identity
}

func rocmResolvedModelProfileIsGemma4(profile ROCmModelProfile) bool {
	return profile.Family == "gemma4" ||
		isROCmGemma4Architecture(profile.Architecture) ||
		isROCmGemma4Architecture(profile.Model.Architecture) ||
		isROCmGemma4AssistantArchitecture(profile.Architecture) ||
		isROCmGemma4AssistantArchitecture(profile.Model.Architecture)
}

func rocmMergeHydratedModelProfile(profile, hydrated ROCmModelProfile) ROCmModelProfile {
	if !hydrated.Matched() {
		return profile
	}
	hydrated.Labels = rocmMergeModelProfileLabels(hydrated.Labels, profile.Labels)
	return hydrated
}

func rocmModelLoaderRouteFromLoadStatus(status ROCmModelLoadStatus) ROCmModelLoaderRoute {
	route := ROCmModelLoaderRoute{
		Contract:      firstNonEmptyString(status.LoaderContract, ROCmModelLoaderRegistryContract),
		Name:          rocmModelLoaderRegistryRouteName,
		Architecture:  status.Architecture,
		Family:        status.Family,
		Loader:        status.Loader,
		Runtime:       status.LoaderRuntime,
		Status:        string(status.Status),
		Target:        status.Target,
		RuntimeStatus: status.RuntimeStatus,
		Reason:        status.Reason,
		Registered:    status.LoaderRegistered,
		NativeRuntime: status.NativeRuntime,
		Standalone:    status.Standalone,
		AttachedOnly:  status.AttachedOnly,
		Staged:        status.Staged,
		MetadataOnly:  status.MetadataOnly,
		TextGenerate:  status.TextGenerate,
		Labels:        cloneStringMap(status.Labels),
	}
	return route.Clone()
}
