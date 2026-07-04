// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"strings"

	"dappco.re/go/inference"
)

const rocmModelRegistryName = "rocm-model-registry-v1"

// ROCmModelProfile is the runtime-facing model registry result. It is resolved
// from the loaded model metadata/config, then carried with the native model so
// execution paths can react to what the model declares.
type ROCmModelProfile struct {
	Name                     string                         `json:"name,omitempty"`
	Family                   string                         `json:"family,omitempty"`
	Architecture             string                         `json:"architecture,omitempty"`
	Registry                 string                         `json:"registry,omitempty"`
	Model                    inference.ModelIdentity        `json:"model,omitempty"`
	ArchitectureProfile      ROCmArchitectureProfile        `json:"architecture_profile,omitempty"`
	EngineFeatures           ROCmEngineFeatures             `json:"engine_features,omitempty"`
	FeatureRoute             ROCmModelFeatureRoute          `json:"feature_route,omitempty"`
	TokenizerRoute           ROCmModelTokenizerRoute        `json:"tokenizer_route,omitempty"`
	LoRAAdapterRoute         ROCmLoRAAdapterRoute           `json:"lora_adapter_route,omitempty"`
	MultimodalProcessorRoute ROCmMultimodalProcessorRoute   `json:"multimodal_processor_route,omitempty"`
	DiffusionSamplerRoute    ROCmDiffusionSamplerRoute      `json:"diffusion_sampler_route,omitempty"`
	StateContextRoute        ROCmStateContextRoute          `json:"state_context_route,omitempty"`
	AttachedDrafterRoute     ROCmAttachedDrafterRoute       `json:"attached_drafter_route,omitempty"`
	LoadStatus               ROCmModelLoadStatus            `json:"load_status,omitempty"`
	CacheRoute               ROCmCacheRoute                 `json:"cache_route,omitempty"`
	QuantLoaderRoute         ROCmQuantLoaderRoute           `json:"quant_loader_route,omitempty"`
	SequenceMixerRoutes      []ROCmSequenceMixerLoaderRoute `json:"sequence_mixer_loader_routes,omitempty"`
	RuntimeContractRoute     ROCmModelRuntimeContractRoute  `json:"runtime_contract_route,omitempty"`
	Gemma4Settings           Gemma4ArchitectureSettings     `json:"gemma4_settings,omitempty"`
	Gemma4EngineFeatures     Gemma4EngineFeatures           `json:"gemma4_engine_features,omitempty"`
	Gemma4DeclaredFeatures   Gemma4DeclaredFeatures         `json:"gemma4_declared_features,omitempty"`
	Gemma4LoRATargetPolicy   Gemma4LoRATargetPolicy         `json:"gemma4_lora_target_policy,omitempty"`
	Labels                   map[string]string              `json:"labels,omitempty"`
}

func (profile ROCmModelProfile) Matched() bool {
	return strings.TrimSpace(profile.Name) != ""
}

func (profile ROCmModelProfile) clone() ROCmModelProfile {
	profile.Model.Labels = cloneStringMap(profile.Model.Labels)
	profile.ArchitectureProfile = cloneGemma4ArchitectureSettings(profile.ArchitectureProfile)
	profile.EngineFeatures = profile.EngineFeatures.clone()
	profile.FeatureRoute = profile.FeatureRoute.Clone()
	profile.TokenizerRoute = profile.TokenizerRoute.Clone()
	profile.LoRAAdapterRoute = profile.LoRAAdapterRoute.Clone()
	profile.MultimodalProcessorRoute = profile.MultimodalProcessorRoute.Clone()
	profile.DiffusionSamplerRoute = profile.DiffusionSamplerRoute.Clone()
	profile.StateContextRoute = profile.StateContextRoute.Clone()
	profile.AttachedDrafterRoute = profile.AttachedDrafterRoute.Clone()
	profile.LoadStatus = profile.LoadStatus.clone()
	profile.CacheRoute = profile.CacheRoute.Clone()
	profile.QuantLoaderRoute = profile.QuantLoaderRoute.Clone()
	profile.SequenceMixerRoutes = cloneROCmSequenceMixerLoaderRoutes(profile.SequenceMixerRoutes)
	profile.RuntimeContractRoute = profile.RuntimeContractRoute.Clone()
	profile.Gemma4Settings = cloneGemma4ArchitectureSettings(profile.Gemma4Settings)
	profile.Gemma4LoRATargetPolicy = cloneGemma4LoRATargetPolicy(profile.Gemma4LoRATargetPolicy)
	profile.Labels = cloneStringMap(profile.Labels)
	return profile
}

type rocmModelProfileRequest struct {
	Path             string
	Model            inference.ModelIdentity
	Gemma4TextConfig nativeGemma4TextConfig
}

type rocmModelProfileFactory interface {
	Name() string
	BuildROCmModelProfile(rocmModelProfileRequest) (ROCmModelProfile, bool)
}

type rocmModelProfileRegistry struct {
	factories []rocmModelProfileFactory
}

func defaultROCmModelProfileRegistry() rocmModelProfileRegistry {
	return rocmModelProfileRegistry{factories: defaultROCmModelProfileFactories()}
}

func defaultROCmModelProfileFactories() []rocmModelProfileFactory {
	factories := registeredROCmModelProfileFactoryAdapters()
	return appendROCmModelProfileFactoryFallbacks(factories,
		gemma4ROCmModelProfileFactory{},
		genericROCmArchitectureProfileFactory{},
	)
}

func defaultROCmModelProfileFactoryNames() []string {
	return defaultROCmModelProfileRegistry().FactoryNames()
}

func (registry rocmModelProfileRegistry) FactoryNames() []string {
	out := make([]string, 0, len(registry.factories))
	for _, factory := range registry.factories {
		if factory == nil {
			continue
		}
		if name := strings.TrimSpace(factory.Name()); name != "" {
			out = append(out, name)
		}
	}
	return out
}

func (registry rocmModelProfileRegistry) Resolve(req rocmModelProfileRequest) (ROCmModelProfile, bool) {
	for _, factory := range registry.factories {
		if factory == nil {
			continue
		}
		profile, ok := factory.BuildROCmModelProfile(req)
		if !ok || !profile.Matched() {
			continue
		}
		if profile.Registry == "" {
			profile.Registry = rocmModelRegistryName
		}
		profile = rocmHydrateResolvedModelProfile(profile, req)
		profile.EngineFeatures = ROCmEngineFeaturesForProfile(profile)
		if !profile.FeatureRoute.Matched() {
			profile.FeatureRoute = ROCmModelFeatureRouteForProfile(profile)
		}
		if !profile.TokenizerRoute.Matched() {
			profile.TokenizerRoute = ROCmModelTokenizerRouteForProfile(profile)
		}
		if !profile.LoRAAdapterRoute.Matched() {
			profile.LoRAAdapterRoute = ROCmLoRAAdapterRouteForProfile(profile)
		}
		if !profile.MultimodalProcessorRoute.Matched() {
			profile.MultimodalProcessorRoute = ROCmMultimodalProcessorRouteForProfile(profile)
		}
		if !profile.DiffusionSamplerRoute.Matched() {
			profile.DiffusionSamplerRoute = ROCmDiffusionSamplerRouteForProfile(profile)
		}
		if !profile.StateContextRoute.Matched() {
			profile.StateContextRoute = ROCmStateContextRouteForProfile(profile)
		}
		if !profile.AttachedDrafterRoute.Matched() {
			profile.AttachedDrafterRoute = ROCmAttachedDrafterRouteForProfile(profile)
		}
		if profile.LoadStatus.empty() {
			profile.LoadStatus = ROCmModelLoadStatusForProfile(profile)
		}
		if !profile.CacheRoute.Matched() {
			if route, ok := ROCmCacheRouteForIdentity(profile.Model.Path, profile.Model); ok {
				profile.CacheRoute = route
			}
		}
		if !profile.QuantLoaderRoute.Matched() {
			if route, ok := ROCmQuantLoaderRouteForProfile(profile); ok {
				profile.QuantLoaderRoute = route
			}
		}
		if !profile.RuntimeContractRoute.Matched() {
			if route, ok := ROCmModelRuntimeContractRouteForIdentity(profile.Model.Path, profile.Model); ok {
				profile.RuntimeContractRoute = route
			}
		}
		profile = rocmApplyModelRouteSetDefaults(profile)
		return profile.clone(), true
	}
	return ROCmModelProfile{}, false
}

func rocmHydrateResolvedModelProfile(profile ROCmModelProfile, req rocmModelProfileRequest) ROCmModelProfile {
	if profile.Model.Path == "" {
		profile.Model.Path = firstNonEmptyString(req.Model.Path, req.Path)
	}
	if profile.Model.Architecture == "" {
		profile.Model.Architecture = firstNonEmptyString(profile.Architecture, req.Model.Architecture)
	}
	if !rocmResolvedModelProfileIsGemma4(profile) {
		return profile
	}
	gemmaReq := req
	gemmaReq.Model = rocmMergeModelProfileIdentityLabels(profile.Model, profile.Labels)
	hydrated, ok := (gemma4ROCmModelProfileFactory{}).BuildROCmModelProfile(gemmaReq)
	if !ok || !hydrated.Matched() {
		return profile
	}
	return rocmMergeHydratedModelProfile(profile, hydrated)
}

type gemma4ROCmModelProfileFactory struct{}

func (gemma4ROCmModelProfileFactory) Name() string { return "gemma4" }

func (gemma4ROCmModelProfileFactory) BuildROCmModelProfile(req rocmModelProfileRequest) (ROCmModelProfile, bool) {
	model := req.Model
	if model.Path == "" {
		model.Path = req.Path
	}
	model = rocmModelIdentityWithResolvedArchitecture(model)
	if settings, ok := Gemma4ArchitectureSettingsForArchitecture(model.Architecture); ok && settings.AttachedOnly {
		model.Architecture = settings.ID
		if model.QuantBits == 0 {
			model.QuantBits = 16
		}
		if model.QuantType == "" {
			model.QuantType = "bf16"
		}
		labels := cloneStringMap(model.Labels)
		size := firstNonEmptyString(labels["gemma4_size"], rocmGemma4ModelPackSize(model, model.Path))
		labels = rocmGemma4MTPAssistantLabels(size, labels)
		model.Labels = labels
		return ROCmModelProfile{
			Name:                   "gemma4",
			Family:                 "gemma4",
			Architecture:           settings.ID,
			Registry:               rocmModelRegistryName,
			Model:                  model,
			ArchitectureProfile:    settings,
			Gemma4Settings:         settings,
			Gemma4EngineFeatures:   Gemma4EngineFeatures{},
			Gemma4DeclaredFeatures: Gemma4DeclaredFeatures{},
			Labels:                 rocmApplyStaticGemma4ModelProfileLabels(nil, settings.ID),
		}, true
	}
	model = rocmGemma4ModelWithInferredPathQuant(model)
	if !rocmIsGemma4SizeQuantIdentity(model.Architecture) {
		return ROCmModelProfile{}, false
	}
	labels := cloneStringMap(model.Labels)
	labels = rocmApplyGemma4NativeConfigFeatureLabels(labels, req.Gemma4TextConfig)
	model.Labels = labels
	declared := Gemma4DeclaredFeaturesForIdentity(model)
	features := Gemma4EngineFeaturesForIdentity(model)
	settings, _ := Gemma4ArchitectureSettingsForArchitecture(model.Architecture)
	loraPolicy, _ := Gemma4LoRATargetPolicyForArchitecture(model.Architecture)
	profileLabels := map[string]string{
		"engine_registry":         rocmModelRegistryName,
		"engine_profile":          "gemma4",
		"engine_profile_family":   "gemma4",
		"engine_profile_source":   "model_config",
		"engine_profile_matched":  "true",
		"engine_profile_reactive": "true",
	}
	if model.Architecture != "" {
		profileLabels["engine_profile_architecture"] = model.Architecture
	}
	return ROCmModelProfile{
		Name:                   "gemma4",
		Family:                 "gemma4",
		Architecture:           model.Architecture,
		Registry:               rocmModelRegistryName,
		Model:                  model,
		ArchitectureProfile:    settings,
		Gemma4Settings:         settings,
		Gemma4EngineFeatures:   features,
		Gemma4DeclaredFeatures: declared,
		Gemma4LoRATargetPolicy: loraPolicy,
		Labels:                 profileLabels,
	}, true
}

func rocmNativeLoadModelIdentity(path string, cfg nativeLoadConfig) inference.ModelIdentity {
	identity := inference.ModelIdentity{
		Path:          path,
		Architecture:  cfg.ModelInfo.Architecture,
		VocabSize:     cfg.ModelInfo.VocabSize,
		NumLayers:     cfg.ModelInfo.NumLayers,
		HiddenSize:    cfg.ModelInfo.HiddenSize,
		QuantBits:     cfg.ModelInfo.QuantBits,
		QuantGroup:    cfg.ModelInfo.QuantGroup,
		ContextLength: cfg.ContextSize,
		Labels:        cloneStringMap(cfg.ModelLabels),
	}
	if identity.QuantType == "" {
		identity.QuantType = identity.Labels["quant_type"]
	}
	if identity.QuantType == "" && rocmIsGemma4SizeQuantIdentity(identity.Architecture) {
		identity.QuantType = identity.Labels["gemma4_quant_mode"]
	}
	return identity
}

func rocmResolveNativeLoadModelProfile(path string, cfg nativeLoadConfig) ROCmModelProfile {
	profile, ok := defaultROCmModelProfileRegistry().Resolve(rocmModelProfileRequest{
		Path:             path,
		Model:            rocmNativeLoadModelIdentity(path, cfg),
		Gemma4TextConfig: cfg.Gemma4TextConfig,
	})
	if !ok {
		return ROCmModelProfile{}
	}
	return profile
}

func rocmApplyResolvedModelProfileLabels(labels map[string]string, path string, model inference.ModelIdentity) map[string]string {
	if settings, ok := Gemma4ArchitectureSettingsForArchitecture(model.Architecture); ok && settings.AttachedOnly {
		return rocmApplyStaticGemma4ModelProfileLabels(labels, settings.ID)
	}
	profile, ok := defaultROCmModelProfileRegistry().Resolve(rocmModelProfileRequest{
		Path:  path,
		Model: model,
	})
	if !ok {
		return labels
	}
	return rocmApplyModelProfileLabels(labels, profile)
}

func rocmApplyModelProfileLabels(labels map[string]string, profile ROCmModelProfile) map[string]string {
	if !profile.Matched() {
		return labels
	}
	if labels == nil {
		labels = map[string]string{}
	}
	for key, value := range profile.Labels {
		if value != "" {
			labels[key] = value
		}
	}
	architectureProfile := profile.ArchitectureProfile
	if architectureProfile.ID == "" {
		architectureProfile = profile.Gemma4Settings
	}
	rocmApplyGemma4ArchitectureSettingsLabels(labels, architectureProfile)
	engineFeatures := profile.EngineFeatures
	if engineFeatures.empty() {
		engineFeatures = ROCmEngineFeaturesForProfile(profile)
	}
	rocmApplyROCmEngineFeatureLabels(labels, engineFeatures)
	featureProfile := profile
	featureProfile.ArchitectureProfile = architectureProfile
	featureProfile.EngineFeatures = engineFeatures
	featureRoute := profile.FeatureRoute
	if !featureRoute.Matched() {
		featureRoute = ROCmModelFeatureRouteForProfile(featureProfile)
	}
	rocmApplyROCmModelFeatureRouteLabels(labels, featureRoute)
	tokenizerProfile := featureProfile
	tokenizerProfile.FeatureRoute = featureRoute
	tokenizerRoute := profile.TokenizerRoute
	if !tokenizerRoute.Matched() {
		tokenizerRoute = ROCmModelTokenizerRouteForProfile(tokenizerProfile)
	}
	rocmApplyROCmModelTokenizerRouteLabels(labels, tokenizerRoute)
	loraProfile := tokenizerProfile
	loraProfile.TokenizerRoute = tokenizerRoute
	loraRoute := profile.LoRAAdapterRoute
	if !loraRoute.Matched() {
		loraRoute = ROCmLoRAAdapterRouteForProfile(loraProfile)
	}
	rocmApplyROCmLoRAAdapterRouteLabels(labels, loraRoute)
	multimodalProfile := loraProfile
	multimodalProfile.LoRAAdapterRoute = loraRoute
	multimodalProfile.Model.Labels = rocmMultimodalMergeLabels(multimodalProfile.Model.Labels, labels)
	multimodalRoute := profile.MultimodalProcessorRoute
	if !multimodalRoute.Matched() {
		multimodalRoute = ROCmMultimodalProcessorRouteForProfile(multimodalProfile)
	}
	rocmApplyROCmMultimodalProcessorRouteLabels(labels, multimodalRoute)
	diffusionProfile := multimodalProfile
	diffusionProfile.MultimodalProcessorRoute = multimodalRoute
	diffusionProfile.Model.Labels = rocmMultimodalMergeLabels(diffusionProfile.Model.Labels, labels)
	diffusionRoute := profile.DiffusionSamplerRoute
	if !diffusionRoute.Matched() {
		diffusionRoute = ROCmDiffusionSamplerRouteForProfile(diffusionProfile)
	}
	rocmApplyROCmDiffusionSamplerRouteLabels(labels, diffusionRoute)
	stateContextProfile := diffusionProfile
	stateContextProfile.DiffusionSamplerRoute = diffusionRoute
	stateContextProfile.Model.Labels = rocmMultimodalMergeLabels(stateContextProfile.Model.Labels, labels)
	stateContextRoute := profile.StateContextRoute
	if !stateContextRoute.Matched() {
		stateContextRoute = ROCmStateContextRouteForProfile(stateContextProfile)
	}
	rocmApplyROCmStateContextRouteLabels(labels, stateContextRoute)
	attachedDrafterProfile := stateContextProfile
	attachedDrafterProfile.StateContextRoute = stateContextRoute
	attachedDrafterProfile.Model.Labels = rocmMultimodalMergeLabels(attachedDrafterProfile.Model.Labels, labels)
	attachedDrafterRoute := profile.AttachedDrafterRoute
	if !attachedDrafterRoute.Matched() {
		attachedDrafterRoute = ROCmAttachedDrafterRouteForProfile(attachedDrafterProfile)
	}
	rocmApplyROCmAttachedDrafterRouteLabels(labels, attachedDrafterRoute)
	loadStatus := profile.LoadStatus
	if loadStatus.empty() {
		loadStatus = ROCmModelLoadStatusForProfile(profile)
	}
	rocmApplyROCmModelLoadStatusLabels(labels, loadStatus)
	cacheRoute := profile.CacheRoute
	if !cacheRoute.Matched() {
		if route, ok := ROCmCacheRouteForIdentity(profile.Model.Path, profile.Model); ok {
			cacheRoute = route
		}
	}
	rocmApplyROCmCacheRouteLabels(labels, cacheRoute)
	quantRoute := profile.QuantLoaderRoute
	if !quantRoute.Matched() {
		if route, ok := ROCmQuantLoaderRouteForProfile(profile); ok {
			quantRoute = route
		}
	}
	rocmApplyROCmQuantLoaderRouteLabels(labels, quantRoute)
	runtimeContractRoute := profile.RuntimeContractRoute
	if !runtimeContractRoute.Matched() {
		if route, ok := ROCmModelRuntimeContractRouteForIdentity(profile.Model.Path, profile.Model); ok {
			runtimeContractRoute = route
		}
	}
	rocmApplyROCmModelRuntimeContractRouteLabels(labels, runtimeContractRoute)
	if profile.Family == "gemma4" {
		rocmApplyGemma4EngineFeatureLabels(labels, profile.Gemma4EngineFeatures, profile.Gemma4DeclaredFeatures)
		rocmApplyGemma4LoRAPolicyLabels(labels, profile.Architecture, profile.Gemma4LoRATargetPolicy)
	}
	return labels
}

func rocmApplyNativeLoadModelProfile(path string, cfg *nativeLoadConfig) {
	if cfg == nil {
		return
	}
	profile := rocmResolveNativeLoadModelProfile(path, *cfg)
	if !profile.Matched() {
		return
	}
	cfg.EngineProfile = profile
	cfg.ModelLabels = rocmApplyModelProfileLabels(cfg.ModelLabels, profile)
}
