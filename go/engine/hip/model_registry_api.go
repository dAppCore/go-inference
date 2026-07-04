// SPDX-Licence-Identifier: EUPL-1.2

package hip

import "dappco.re/go/inference"

// ROCmModelIdentityReporter is implemented by loaded ROCm models that can
// expose the richer, context-bearing model identity used by state bundles,
// capability reports, and reactive registry routing.
type ROCmModelIdentityReporter interface {
	ModelIdentity() inference.ModelIdentity
}

// ROCmModelProfileReporter is implemented by loaded ROCm models that can expose
// the resolved model registry profile used for reactive runtime routing.
type ROCmModelProfileReporter interface {
	ModelProfile() ROCmModelProfile
}

// ROCmModelRoutePlanReporter is implemented by loaded ROCm models that can
// expose the compact model-route plan used by API clients and daemon bridges.
type ROCmModelRoutePlanReporter interface {
	ModelRoutePlan() ROCmModelRoutePlan
}

// ResolveROCmModelProfile resolves the default model registry for a concrete
// backend-neutral identity. Runtime load paths use an internal config-aware
// resolver; this API is for go-ai/go-ml style consumers that already have
// model metadata and need the same reactive feature/profile contract.
func ResolveROCmModelProfile(path string, model inference.ModelIdentity) (ROCmModelProfile, bool) {
	if model.Path == "" {
		model.Path = path
	}
	profile, ok := defaultROCmModelProfileRegistry().Resolve(rocmModelProfileRequest{
		Path:  path,
		Model: model,
	})
	if !ok {
		return ROCmModelProfile{}, false
	}
	return profile.clone(), true
}

// ResolveROCmModelProfileForInspection resolves the default registry from an
// already-inspected model pack. Inspection labels are included because config
// probes can refine the architecture before any weights are loaded.
func ResolveROCmModelProfileForInspection(inspection *inference.ModelPackInspection) (ROCmModelProfile, bool) {
	if inspection == nil {
		return ROCmModelProfile{}, false
	}
	model := inspection.Model
	path := firstNonEmptyString(model.Path, inspection.Path)
	model.Path = path
	labels := cloneStringMap(inspection.Labels)
	if labels == nil {
		labels = map[string]string{}
	}
	for key, value := range model.Labels {
		if value != "" {
			labels[key] = value
		}
	}
	model.Labels = labels
	return ResolveROCmModelProfile(path, model)
}

// ResolveROCmModelProfileForInfo adapts the small go-inference ModelInfo shape
// into the registry's identity resolver. Labels are cloned before resolution.
func ResolveROCmModelProfileForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmModelProfile, bool) {
	return ResolveROCmModelProfile(path, inference.ModelIdentity{
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

// ResolveROCmModelProfileForModel resolves the registry from a loaded model.
// Model-owned profile/identity reporters win over the small TextModel.Info()
// shape so wrappers can stay reactive without exposing concrete ROCm types.
func ResolveROCmModelProfileForModel(model inference.TextModel) (ROCmModelProfile, bool) {
	if model == nil {
		return ROCmModelProfile{}, false
	}
	if reporter, ok := model.(ROCmModelProfileReporter); ok {
		profile := reporter.ModelProfile()
		if profile.Matched() {
			return profile.clone(), true
		}
	}
	identity := rocmTextModelIdentity(model)
	if rocmModelIdentityIsZero(identity) {
		return ROCmModelProfile{}, false
	}
	return ResolveROCmModelProfile(identity.Path, identity)
}

func ROCmEngineFeaturesForInspection(inspection *inference.ModelPackInspection) (ROCmEngineFeatures, bool) {
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok {
		return ROCmEngineFeatures{}, false
	}
	return profile.EngineFeatures.clone(), true
}

func rocmTextModelIdentity(model inference.TextModel) inference.ModelIdentity {
	if model == nil {
		return inference.ModelIdentity{}
	}
	if reporter, ok := model.(ROCmModelIdentityReporter); ok {
		identity := reporter.ModelIdentity()
		if !rocmModelIdentityIsZero(identity) {
			return rocmCloneModelIdentity(identity)
		}
	}
	info := model.Info()
	if info.Architecture == "" {
		info.Architecture = model.ModelType()
	}
	return inference.ModelIdentity{
		Architecture: normalizeROCmArchitecture(info.Architecture),
		VocabSize:    info.VocabSize,
		NumLayers:    info.NumLayers,
		HiddenSize:   info.HiddenSize,
		QuantBits:    info.QuantBits,
		QuantGroup:   info.QuantGroup,
	}
}

func rocmCloneModelIdentity(identity inference.ModelIdentity) inference.ModelIdentity {
	identity.Labels = cloneStringMap(identity.Labels)
	return identity
}

func rocmModelIdentityIsZero(identity inference.ModelIdentity) bool {
	return identity.ID == "" &&
		identity.Path == "" &&
		identity.Architecture == "" &&
		identity.Revision == "" &&
		identity.Hash == "" &&
		identity.QuantBits == 0 &&
		identity.QuantGroup == 0 &&
		identity.QuantType == "" &&
		identity.ContextLength == 0 &&
		identity.NumLayers == 0 &&
		identity.HiddenSize == 0 &&
		identity.VocabSize == 0 &&
		len(identity.Labels) == 0
}

// ApplyROCmModelProfileLabels returns labels plus the registry-derived feature
// labels for profile without mutating the caller's input map.
func ApplyROCmModelProfileLabels(labels map[string]string, profile ROCmModelProfile) map[string]string {
	return rocmApplyModelProfileLabels(cloneStringMap(labels), profile)
}
