// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"strconv"
	"strings"

	core "dappco.re/go"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
	rocmscheme "dappco.re/go/inference/engine/hip/scheme"
)

// ROCmModelRegistrySnapshot is the public, copy-safe registry view exposed to
// CLI/API consumers that need to react to model-declared engine capabilities.
type ROCmModelRegistrySnapshot struct {
	Name                      string                         `json:"name"`
	Backend                   string                         `json:"backend"`
	DefaultFamily             string                         `json:"default_family,omitempty"`
	Factories                 []string                       `json:"factories,omitempty"`
	ArchitectureProfiles      []Gemma4ArchitectureSettings   `json:"architecture_profiles,omitempty"`
	FeatureRoutes             []ROCmModelFeatureRoute        `json:"feature_routes,omitempty"`
	TokenizerRoutes           []ROCmModelTokenizerRoute      `json:"tokenizer_routes,omitempty"`
	LoRAAdapterRoutes         []ROCmLoRAAdapterRoute         `json:"lora_adapter_routes,omitempty"`
	MultimodalProcessorRoutes []ROCmMultimodalProcessorRoute `json:"multimodal_processor_routes,omitempty"`
	DiffusionSamplerRoutes    []ROCmDiffusionSamplerRoute    `json:"diffusion_sampler_routes,omitempty"`
	StateContextRoutes        []ROCmStateContextRoute        `json:"state_context_routes,omitempty"`
	AttachedDrafterRoutes     []ROCmAttachedDrafterRoute     `json:"attached_drafter_routes,omitempty"`
	LoaderRoutes              []ROCmModelLoaderRoute         `json:"loader_routes,omitempty"`
	CacheModeRoutes           []ROCmCacheModeRoute           `json:"cache_mode_routes,omitempty"`
	CacheRoutes               []ROCmCacheRoute               `json:"cache_routes,omitempty"`
	QuantSchemes              []ROCmQuantScheme              `json:"quant_schemes,omitempty"`
	QuantLoaderRoutes         []ROCmQuantLoaderRoute         `json:"quant_loader_routes,omitempty"`
	MixerLoaderRoutes         []ROCmSequenceMixerLoaderRoute `json:"mixer_loader_routes,omitempty"`
	AlgorithmProfiles         []ROCmAlgorithmProfile         `json:"algorithm_profiles,omitempty"`
	Labels                    map[string]string              `json:"labels,omitempty"`
}

func DefaultROCmModelRegistryName() string {
	return rocmModelRegistryName
}

func DefaultROCmModelRegistrySnapshot(backend string) ROCmModelRegistrySnapshot {
	if strings.TrimSpace(backend) == "" {
		backend = "rocm"
	}
	profiles := DefaultROCmArchitectureProfiles()
	featureRoutes := DefaultROCmModelFeatureRoutes()
	tokenizerRoutes := DefaultROCmModelTokenizerRoutes()
	loraRoutes := DefaultROCmLoRAAdapterRoutes()
	multimodalRoutes := DefaultROCmMultimodalProcessorRoutes()
	diffusionRoutes := DefaultROCmDiffusionSamplerRoutes()
	stateContextRoutes := DefaultROCmStateContextRoutes()
	attachedDrafterRoutes := DefaultROCmAttachedDrafterRoutes()
	routes := DefaultROCmModelLoaderRoutes()
	cacheModeRoutes := DefaultROCmCacheModeRoutes()
	cacheRoutes := defaultROCmModelRegistryCacheRoutes(profiles)
	quantSchemes := DefaultROCmQuantSchemes()
	quantRoutes := DefaultROCmQuantLoaderRoutes()
	mixerRoutes := DefaultROCmSequenceMixerLoaderRoutes()
	algorithmProfiles := DefaultROCmAlgorithmProfiles()
	schemeMixerKinds := rocmscheme.MixerKinds()
	schemeCacheModes := rocmscheme.CacheModes()
	schemeQuantKinds := rocmscheme.QuantKinds()
	modelLoaderRoutes := rocmmodel.DefaultLoaderRoutes()
	modelLoaderArchitectures := rocmmodel.LoaderArchitectures()
	return ROCmModelRegistrySnapshot{
		Name:                      rocmModelRegistryName,
		Backend:                   strings.TrimSpace(backend),
		DefaultFamily:             "gemma4",
		Factories:                 defaultROCmModelProfileFactoryNames(),
		ArchitectureProfiles:      profiles,
		FeatureRoutes:             featureRoutes,
		TokenizerRoutes:           tokenizerRoutes,
		LoRAAdapterRoutes:         loraRoutes,
		MultimodalProcessorRoutes: multimodalRoutes,
		DiffusionSamplerRoutes:    diffusionRoutes,
		StateContextRoutes:        stateContextRoutes,
		AttachedDrafterRoutes:     attachedDrafterRoutes,
		LoaderRoutes:              routes,
		CacheModeRoutes:           cacheModeRoutes,
		CacheRoutes:               cacheRoutes,
		QuantSchemes:              quantSchemes,
		QuantLoaderRoutes:         quantRoutes,
		MixerLoaderRoutes:         mixerRoutes,
		AlgorithmProfiles:         algorithmProfiles,
		Labels: map[string]string{
			"architecture_resolution_contract":           ROCmArchitectureResolutionContract,
			"engine_registry":                            rocmModelRegistryName,
			"engine_algorithm_profile_contract":          ROCmAlgorithmProfileRegistryContract,
			"engine_config_probe_contract":               ROCmModelConfigProbeContract,
			"engine_feature_route_contract":              ROCmModelFeatureRegistryContract,
			"engine_lora_route_contract":                 ROCmLoRAAdapterRegistryContract,
			"engine_tokenizer_route_contract":            ROCmModelTokenizerRegistryContract,
			"engine_model_loader_contract":               rocmmodel.LoaderRegistryContract,
			"engine_model_loader_architectures":          core.Join(",", modelLoaderArchitectures...),
			"engine_loader_contract":                     ROCmModelLoaderRegistryContract,
			"engine_cache_factory_contract":              ROCmCacheFactoryRouteContract,
			"engine_mixer_loader_contract":               ROCmSequenceMixerLoaderRegistryContract,
			"engine_multimodal_processor_route_contract": ROCmMultimodalProcessorRegistryContract,
			"engine_diffusion_sampler_route_contract":    ROCmDiffusionSamplerRegistryContract,
			"engine_state_context_route_contract":        ROCmStateContextRegistryContract,
			"engine_attached_drafter_route_contract":     ROCmAttachedDrafterRegistryContract,
			"engine_scheme_contract":                     rocmscheme.RegistryContract,
			"engine_scheme_mixer_kinds":                  core.Join(",", schemeMixerKinds...),
			"engine_scheme_cache_modes":                  core.Join(",", schemeCacheModes...),
			"engine_scheme_quant_kinds":                  core.Join(",", schemeQuantKinds...),
			"engine_quant_scheme_contract":               ROCmQuantSchemeRegistryContract,
			"engine_quant_scheme_kinds":                  rocmQuantSchemeKindsCSV(quantSchemes),
			"engine_quant_loader_contract":               ROCmQuantLoaderRegistryContract,
			"engine_profile_reactive":                    "true",
			"engine_profile_family":                      "gemma4",
			"engine_registry_scope":                      "architecture_profiles",
			"algorithm_profile_count":                    strconv.Itoa(len(algorithmProfiles)),
			"attached_drafter_route_count":               strconv.Itoa(len(attachedDrafterRoutes)),
			"cache_mode_route_count":                     strconv.Itoa(len(cacheModeRoutes)),
			"cache_route_count":                          strconv.Itoa(len(cacheRoutes)),
			"feature_route_count":                        strconv.Itoa(len(featureRoutes)),
			"loader_route_count":                         strconv.Itoa(len(routes)),
			"diffusion_sampler_route_count":              strconv.Itoa(len(diffusionRoutes)),
			"lora_adapter_route_count":                   strconv.Itoa(len(loraRoutes)),
			"model_loader_count":                         strconv.Itoa(len(modelLoaderRoutes)),
			"mixer_loader_route_count":                   strconv.Itoa(len(mixerRoutes)),
			"multimodal_processor_route_count":           strconv.Itoa(len(multimodalRoutes)),
			"quant_scheme_count":                         strconv.Itoa(len(quantSchemes)),
			"quant_loader_route_count":                   strconv.Itoa(len(quantRoutes)),
			"scheme_cache_count":                         strconv.Itoa(len(schemeCacheModes)),
			"scheme_mixer_count":                         strconv.Itoa(len(schemeMixerKinds)),
			"scheme_quant_count":                         strconv.Itoa(len(schemeQuantKinds)),
			"state_context_route_count":                  strconv.Itoa(len(stateContextRoutes)),
			"tokenizer_route_count":                      strconv.Itoa(len(tokenizerRoutes)),
			"profile_count":                              strconv.Itoa(len(profiles)),
			"production_contract":                        "reactive-inference-v1",
		},
	}
}

func defaultROCmModelRegistryCacheRoutes(profiles []ROCmArchitectureProfile) []ROCmCacheRoute {
	routes := make([]ROCmCacheRoute, 0, len(profiles))
	seen := map[string]bool{}
	for _, profile := range profiles {
		if profile.ID == "" || seen[profile.ID] {
			continue
		}
		route, ok := ROCmCacheRouteForArchitecture(profile.ID)
		if !ok || !route.Matched() {
			continue
		}
		seen[profile.ID] = true
		routes = append(routes, route.Clone())
	}
	return routes
}
