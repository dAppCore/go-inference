// SPDX-Licence-Identifier: EUPL-1.2

package hip

import "dappco.re/go/inference"

type genericROCmArchitectureProfileFactory struct{}

func (genericROCmArchitectureProfileFactory) Name() string { return "architecture-profile" }

func (genericROCmArchitectureProfileFactory) BuildROCmModelProfile(req rocmModelProfileRequest) (ROCmModelProfile, bool) {
	model := req.Model
	if model.Path == "" {
		model.Path = req.Path
	}
	architecture := firstNonEmptyString(
		model.Labels["engine_architecture_resolved"],
		model.Labels["architecture_resolved"],
		model.Architecture,
	)
	profile, ok := ROCmArchitectureProfileForArchitecture(architecture)
	if !ok {
		return ROCmModelProfile{}, false
	}
	model.Architecture = profile.ID
	family := firstNonEmptyString(profile.Family, profile.ID)
	labels := rocmArchitectureProfileModelLabels(profile)
	return ROCmModelProfile{
		Name:                family,
		Family:              family,
		Architecture:        profile.ID,
		Registry:            rocmModelRegistryName,
		Model:               model,
		ArchitectureProfile: cloneGemma4ArchitectureSettings(profile),
		Labels:              labels,
	}, true
}

func rocmArchitectureProfileModelLabels(profile ROCmArchitectureProfile) map[string]string {
	family := firstNonEmptyString(profile.Family, profile.ID)
	labels := map[string]string{
		"engine_registry":         rocmModelRegistryName,
		"engine_profile":          family,
		"engine_profile_family":   family,
		"engine_profile_source":   "architecture_profile",
		"engine_profile_matched":  "true",
		"engine_profile_reactive": "true",
	}
	if profile.ID != "" {
		labels["engine_profile_architecture"] = profile.ID
	}
	return labels
}

func ResolveROCmArchitectureProfileForIdentity(path string, model inference.ModelIdentity) (ROCmArchitectureProfile, bool) {
	if model.Path == "" {
		model.Path = path
	}
	architecture := firstNonEmptyString(
		model.Labels["engine_architecture_resolved"],
		model.Labels["architecture_resolved"],
		model.Architecture,
	)
	profile, ok := ROCmArchitectureProfileForArchitecture(architecture)
	if !ok {
		return ROCmArchitectureProfile{}, false
	}
	return cloneGemma4ArchitectureSettings(profile), true
}
