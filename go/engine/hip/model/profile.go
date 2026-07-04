// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/registry"
)

const (
	ProfileRegistryName            = "rocm-model-registry-v1"
	ProfileFactoryRegistryContract = "rocm-model-profile-factory-registry-v1"
)

// ProfileRequest is the model-package input for a registered profile factory.
// It intentionally carries only backend-neutral identity data so model-family
// packages can self-register without importing the root ROCm package.
type ProfileRequest struct {
	Path  string
	Model inference.ModelIdentity
}

// Profile is the model-owned profile factory result. Root packages can enrich
// it with backend/runtime details, but family packages can describe the loaded
// model, route set, and labels here without central switches.
type Profile struct {
	Contract     string                  `json:"contract,omitempty"`
	Name         string                  `json:"name,omitempty"`
	Family       string                  `json:"family,omitempty"`
	Architecture string                  `json:"architecture,omitempty"`
	Registry     string                  `json:"registry,omitempty"`
	Model        inference.ModelIdentity `json:"model,omitempty"`
	RouteSet     RouteSet                `json:"route_set,omitempty"`
	Labels       map[string]string       `json:"labels,omitempty"`
}

func (profile Profile) Matched() bool {
	return strings.TrimSpace(profile.Name) != ""
}

func (profile Profile) Clone() Profile {
	profile.Model.Labels = cloneStringMap(profile.Model.Labels)
	profile.RouteSet = profile.RouteSet.Clone()
	profile.Labels = cloneStringMap(profile.Labels)
	return profile
}

// ProfileFactory resolves a model identity into a model-owned profile. A model
// family can register one from its package init, mirroring go-mlx's
// self-registering model loaders while keeping this package root-agnostic.
type ProfileFactory interface {
	Name() string
	BuildModelProfile(ProfileRequest) (Profile, bool)
}

var registeredProfileFactories = registry.NewOrdered[string, ProfileFactory]()

func RegisterProfileFactory(factory ProfileFactory) {
	if factory == nil {
		return
	}
	name := strings.TrimSpace(factory.Name())
	if name == "" {
		return
	}
	registeredProfileFactories.Put(name, factory)
}

func RegisteredProfileFactoryNames() []string {
	return registeredProfileFactories.Keys()
}

func RegisteredProfileFactories() []ProfileFactory {
	return registeredProfileFactories.Values()
}

func ReplaceRegisteredProfileFactories(factories []ProfileFactory) {
	order := make([]string, 0, len(factories))
	values := make(map[string]ProfileFactory, len(factories))
	for _, factory := range factories {
		if factory == nil {
			continue
		}
		name := strings.TrimSpace(factory.Name())
		if name == "" {
			continue
		}
		if _, ok := values[name]; !ok {
			order = append(order, name)
		}
		values[name] = factory
	}
	registeredProfileFactories.Restore(order, values)
}

func ResolveRegisteredProfile(path string, identity inference.ModelIdentity) (Profile, bool) {
	req := ProfileRequest{Path: path, Model: cloneModelIdentity(identity)}
	if req.Model.Path == "" {
		req.Model.Path = path
	}
	for _, factory := range registeredProfileFactories.Values() {
		profile, ok := ResolveProfileFactory(factory, req)
		if ok {
			return profile, true
		}
	}
	return Profile{}, false
}

func ResolveProfileFactory(factory ProfileFactory, req ProfileRequest) (Profile, bool) {
	if factory == nil {
		return Profile{}, false
	}
	req.Model = cloneModelIdentity(req.Model)
	if req.Model.Path == "" {
		req.Model.Path = req.Path
	}
	profile, ok := factory.BuildModelProfile(req)
	if !ok || !profile.Matched() {
		return Profile{}, false
	}
	return normalizeRegisteredProfile(profile, strings.TrimSpace(factory.Name()), req), true
}

func normalizeRegisteredProfile(profile Profile, factoryName string, req ProfileRequest) Profile {
	profile.Model = cloneModelIdentity(profile.Model)
	if profile.Model.Path == "" {
		profile.Model.Path = firstNonEmpty(req.Model.Path, req.Path)
	}
	if profile.Model.Architecture == "" {
		profile.Model.Architecture = firstNonEmpty(profile.Architecture, req.Model.Architecture)
	}
	if profile.Architecture == "" {
		profile.Architecture = firstNonEmpty(profile.RouteSet.Architecture, profile.Model.Architecture)
	}
	if profile.Family == "" {
		profile.Family = firstNonEmpty(profile.RouteSet.Family, profile.Name, profile.Architecture)
	}
	if profile.Contract == "" {
		profile.Contract = ProfileFactoryRegistryContract
	}
	if profile.Registry == "" {
		profile.Registry = ProfileRegistryName
	}
	profile.RouteSet = normalizeProfileRouteSet(profile.RouteSet, profile)
	if !profile.RouteSet.Matched() {
		if routeSet, ok := RouteSetForIdentity(profile.Model.Path, profile.Model); ok {
			profile.RouteSet = routeSet
		}
	}
	profile.Labels = registeredProfileLabels(profile.Labels, factoryName, profile)
	return profile.Clone()
}

func registeredProfileLabels(labels map[string]string, factoryName string, profile Profile) map[string]string {
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
	family := firstNonEmpty(profile.Family, profile.Name, factoryName)
	architecture := firstNonEmpty(profile.Architecture, profile.RouteSet.Architecture, profile.Model.Architecture)
	setDefault("engine_registry", ProfileRegistryName)
	setDefault("engine_profile", firstNonEmpty(profile.Name, factoryName))
	setDefault("engine_profile_family", family)
	setDefault("engine_profile_source", "registered_factory")
	setDefault("engine_profile_factory", factoryName)
	setDefault("engine_profile_matched", "true")
	setDefault("engine_profile_reactive", "true")
	setDefault("engine_profile_architecture", architecture)
	return labels
}

func normalizeProfileRouteSet(routeSet RouteSet, profile Profile) RouteSet {
	routeSet = routeSet.Clone()
	routeSet.Model = cloneModelIdentity(routeSet.Model)
	if routeSet.Model.Path == "" {
		routeSet.Model.Path = profile.Model.Path
	}
	if routeSet.Model.Architecture == "" {
		routeSet.Model.Architecture = firstNonEmpty(profile.Model.Architecture, profile.Architecture)
	}
	if routeSet.Model.Labels == nil {
		routeSet.Model.Labels = cloneStringMap(profile.Model.Labels)
	}
	if routeSet.Contract == "" && profileRouteSetHasRoute(routeSet) {
		routeSet.Contract = RouteSetContract
	}
	if routeSet.Architecture == "" {
		routeSet.Architecture = firstNonEmpty(profile.Architecture, routeSet.Model.Architecture)
	}
	if routeSet.Family == "" {
		routeSet.Family = firstNonEmpty(profile.Family, profile.Name, routeSet.Architecture)
	}
	if routeSet.Labels == nil && routeSet.Architecture != "" {
		routeSet.Labels = routeSetLabels(routeSet)
	}
	return routeSet.Clone()
}

func profileRouteSetHasRoute(routeSet RouteSet) bool {
	return routeSet.FeatureRoute.Matched() ||
		routeSet.CacheRoute.Matched() ||
		routeSet.LoaderRoute.Matched() ||
		routeSet.TokenizerRoute.Matched() ||
		routeSet.LoRAAdapterRoute.Matched() ||
		routeSet.MultimodalProcessorRoute.Matched() ||
		routeSet.DiffusionSamplerRoute.Matched() ||
		routeSet.StateContextRoute.Matched() ||
		routeSet.AttachedDrafterRoute.Matched() ||
		routeSet.QuantLoaderRoute.Matched() ||
		len(routeSet.SequenceMixerRoutes) > 0 ||
		routeSet.RuntimeContractRoute.Matched() ||
		routeSet.RuntimeGatePlan.Matched() ||
		routeSet.RuntimeAuthorPlan.Matched()
}

func cloneModelIdentity(identity inference.ModelIdentity) inference.ModelIdentity {
	identity.Labels = cloneStringMap(identity.Labels)
	return identity
}
