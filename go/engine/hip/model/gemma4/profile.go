// SPDX-Licence-Identifier: EUPL-1.2

// Package gemma4 registers the Gemma-4 model-family profile with the neutral
// ROCm model registry.
package gemma4

import (
	"maps"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/model"
	rocmprofile "dappco.re/go/inference/engine/hip/profile"
)

func init() {
	model.RegisterProfileFactory(ProfileFactory{})
	for _, settings := range rocmprofile.DefaultGemma4ArchitectureSettings() {
		if settings.AttachedOnly || settings.ChatTemplate == "" {
			continue
		}
		model.RegisterTokenizerRoute(model.TokenizerRoute{
			Architecture:         settings.ID,
			Family:               settings.Family,
			TokenizerKind:        "GemmaTokenizer",
			ChatTemplateID:       settings.ChatTemplate,
			ReasoningParserID:    settings.ParserID,
			ToolParserID:         settings.ToolParserID,
			GenerationRole:       settings.GenerationRole,
			NativeRuntime:        settings.NativeRuntime,
			RequiresChatTemplate: settings.RequiresChatTemplate,
			Generation:           settings.Generation,
			Chat:                 settings.Chat,
			ThinkingChannel:      true,
			ThinkingChannelOpen:  ThinkingChannelOpenMarker,
			ThinkingChannelClose: ThinkingChannelCloseMarker,
		})
	}
}

// ProfileFactory resolves Gemma-4 identities from model-owned metadata without
// importing the root rocm package.
type ProfileFactory struct{}

func (ProfileFactory) Name() string { return "gemma4" }

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
	settings, ok := rocmprofile.Gemma4ArchitectureSettingsForArchitecture(architecture)
	if !ok {
		return model.Profile{}, false
	}
	identity.Architecture = settings.ID
	if settings.AttachedOnly {
		if identity.QuantBits == 0 {
			identity.QuantBits = 16
		}
		if identity.QuantType == "" {
			identity.QuantType = "bf16"
		}
	}
	routeSet, _ := model.RouteSetForIdentity(identity.Path, identity)
	return model.Profile{
		Contract:     model.ProfileFactoryRegistryContract,
		Name:         "gemma4",
		Family:       "gemma4",
		Architecture: settings.ID,
		Registry:     model.ProfileRegistryName,
		Model:        identity,
		RouteSet:     routeSet,
		Labels:       profileLabels(settings),
	}, true
}

func profileLabels(settings rocmprofile.Gemma4ArchitectureSettings) map[string]string {
	labels := map[string]string{
		"engine_registry":         model.ProfileRegistryName,
		"engine_profile":          "gemma4",
		"engine_profile_family":   "gemma4",
		"engine_profile_source":   "model_config",
		"engine_profile_matched":  "true",
		"engine_profile_reactive": "true",
	}
	if settings.ID != "" {
		labels["engine_profile_architecture"] = settings.ID
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
	maps.Copy(out, values)
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
