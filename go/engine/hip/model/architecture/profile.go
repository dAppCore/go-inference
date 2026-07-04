// SPDX-Licence-Identifier: EUPL-1.2

// Package architecture provides the generic model-profile factory backed by the
// architecture catalogue.
package architecture

import (
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/model"
	rocmprofile "dappco.re/go/inference/engine/hip/profile"
)

// ProfileFactory resolves any registered or built-in architecture profile into
// a neutral model profile.
type ProfileFactory struct{}

func (ProfileFactory) Name() string { return "architecture-profile" }

func (ProfileFactory) BuildModelProfile(req model.ProfileRequest) (model.Profile, bool) {
	identity := cloneModelIdentity(req.Model)
	if identity.Path == "" {
		identity.Path = req.Path
	}
	architecture := firstNonEmpty(
		identity.Labels["engine_architecture_resolved"],
		identity.Labels["architecture_resolved"],
		identity.Architecture,
	)
	architectureProfile, ok := rocmprofile.LookupArchitectureProfile(architecture)
	if !ok {
		return model.Profile{}, false
	}
	identity.Architecture = architectureProfile.ID
	family := firstNonEmpty(architectureProfile.Family, architectureProfile.ID)
	routeSet, _ := model.RouteSetForIdentity(identity.Path, identity)
	return model.Profile{
		Contract:     model.ProfileFactoryRegistryContract,
		Name:         family,
		Family:       family,
		Architecture: architectureProfile.ID,
		Registry:     model.ProfileRegistryName,
		Model:        identity,
		RouteSet:     routeSet,
		Labels:       profileLabels(architectureProfile),
	}, true
}

func profileLabels(profile rocmprofile.ArchitectureProfile) map[string]string {
	family := firstNonEmpty(profile.Family, profile.ID)
	labels := map[string]string{
		"engine_registry":         model.ProfileRegistryName,
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

func cloneModelIdentity(identity inference.ModelIdentity) inference.ModelIdentity {
	identity.Labels = cloneStringMap(identity.Labels)
	return identity
}

func cloneStringMap(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	out := make(map[string]string, len(values))
	for key, value := range values {
		out[key] = value
	}
	return out
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
