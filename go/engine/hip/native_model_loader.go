// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import "dappco.re/go/inference/engine/hip/internal/registry"

type rocmNativeModelLoadFunc func(*hipRuntime, string, nativeLoadConfig) (nativeModel, error)

type rocmNativeModelLoader struct {
	name string
	load rocmNativeModelLoadFunc
}

var registeredROCmNativeModelLoaders = registry.NewOrdered[string, rocmNativeModelLoader]()

func init() {
	registerDefaultROCmNativeModelLoaders()
}

func registerDefaultROCmNativeModelLoaders() {
	for _, route := range DefaultROCmModelLoaderRoutes() {
		if !rocmNativeModelLoaderRouteHasStandaloneLoader(route) {
			continue
		}
		registerROCmNativeModelLoader(route.Architecture, route.Loader, loadHIPDefaultNativeModel)
	}
}

func rocmNativeModelLoaderRouteHasStandaloneLoader(route ROCmModelLoaderRoute) bool {
	return route.Matched() &&
		route.Runtime == rocmModelLoaderRuntimeHIP &&
		route.NativeRuntime &&
		route.Standalone &&
		!route.AttachedOnly &&
		!route.MetadataOnly
}

func registerROCmNativeModelLoader(architecture, name string, load rocmNativeModelLoadFunc) {
	architecture = ROCmArchitectureID(architecture)
	if architecture == "" || load == nil {
		return
	}
	if name == "" {
		name = architecture
	}
	registeredROCmNativeModelLoaders.Put(architecture, rocmNativeModelLoader{name: name, load: load})
}

func registeredROCmNativeModelLoaderArchitectures() []string {
	return registeredROCmNativeModelLoaders.Keys()
}

// RegisteredROCmNativeModelLoaderRegistrations returns live native loader
// registrations in resolution order. It intentionally exposes metadata only:
// concrete loader functions stay inside the ROCm runtime boundary.
func RegisteredROCmNativeModelLoaderRegistrations() []ROCmNativeModelLoaderRegistration {
	architectures := registeredROCmNativeModelLoaderArchitectures()
	registrations := make([]ROCmNativeModelLoaderRegistration, 0, len(architectures))
	for _, architecture := range architectures {
		registration, ok := ROCmNativeModelLoaderRegistrationForArchitecture(architecture)
		if !ok {
			continue
		}
		registrations = append(registrations, registration)
	}
	return registrations
}

// ROCmNativeModelLoaderRegistrationForArchitecture returns the live native
// loader registration for architecture, if one exists.
func ROCmNativeModelLoaderRegistrationForArchitecture(architecture string) (ROCmNativeModelLoaderRegistration, bool) {
	architecture = ROCmArchitectureID(architecture)
	loader, ok := lookupROCmNativeModelLoader(architecture)
	if !ok {
		return ROCmNativeModelLoaderRegistration{}, false
	}
	route, _ := ROCmModelLoaderRouteForArchitecture(architecture)
	registration := ROCmNativeModelLoaderRegistration{
		Architecture: architecture,
		Loader:       loader.name,
		Route:        route,
		Registered:   true,
	}
	if route.Matched() {
		registration.Architecture = route.Architecture
		registration.NativeRuntime = route.NativeRuntime
		registration.Standalone = route.Standalone
		registration.TextGenerate = route.TextGenerate
	}
	return registration.clone(), true
}

func lookupROCmNativeModelLoader(architecture string) (rocmNativeModelLoader, bool) {
	architecture = ROCmArchitectureID(architecture)
	if architecture == "" {
		return rocmNativeModelLoader{}, false
	}
	return registeredROCmNativeModelLoaders.Get(architecture)
}

func rocmNativeModelLoaderForConfig(cfg nativeLoadConfig) (rocmNativeModelLoader, bool) {
	return lookupROCmNativeModelLoader(rocmNativeModelLoaderArchitecture(cfg))
}

func rocmNativeModelLoaderArchitecture(cfg nativeLoadConfig) string {
	return ROCmArchitectureID(firstNonEmptyString(
		cfg.ModelLabels["engine_architecture_resolved"],
		cfg.ModelLabels["architecture_resolved"],
		cfg.EngineProfile.Architecture,
		cfg.EngineProfile.Model.Architecture,
		cfg.ModelInfo.Architecture,
	))
}
