// SPDX-Licence-Identifier: EUPL-1.2

package profile

import "strings"

// ArchitectureResolution is the profile-owned dispatch result for a model
// config's architecture signals.
type ArchitectureResolution struct {
	Architecture       string              `json:"architecture,omitempty"`
	Source             string              `json:"source,omitempty"`
	ModelType          string              `json:"model_type,omitempty"`
	TextTowerModelType string              `json:"text_tower_model_type,omitempty"`
	Architectures      []string            `json:"architectures,omitempty"`
	Profile            ArchitectureProfile `json:"profile,omitempty"`
}

func (resolution ArchitectureResolution) Matched() bool {
	return strings.TrimSpace(resolution.Architecture) != ""
}

func (resolution ArchitectureResolution) Clone() ArchitectureResolution {
	resolution.Architectures = cloneStringSlice(resolution.Architectures)
	resolution.Profile = CloneGemma4ArchitectureSettings(resolution.Profile)
	return resolution
}

// ResolveArchitecture maps config.json architecture signals to the registered
// profile id the ROCm loader and API surfaces should dispatch on.
func ResolveArchitecture(modelType, textTowerModelType string, architectures []string) ArchitectureResolution {
	modelType = strings.TrimSpace(modelType)
	textTowerModelType = strings.TrimSpace(textTowerModelType)
	architectures = CleanArchitectureSignals(architectures)
	if modelType != "" {
		id := architectureIDForSignal(modelType)
		if tower := textTowerRefinement(id, textTowerModelType); tower != "" {
			return architectureResolution(tower, "model_type_text_tower", modelType, textTowerModelType, architectures)
		}
		if rerank := rerankRefinement(id, architectures); rerank != "" {
			return architectureResolution(rerank, "model_type_architecture_refinement", modelType, textTowerModelType, architectures)
		}
		return architectureResolution(id, "model_type", modelType, textTowerModelType, architectures)
	}
	if textTowerModelType != "" {
		return architectureResolution(architectureIDForSignal(textTowerModelType), "text_config_model_type", modelType, textTowerModelType, architectures)
	}
	for _, architecture := range architectures {
		if id := architectureIDForSignal(architecture); id != "" {
			return architectureResolution(id, "architectures", modelType, textTowerModelType, architectures)
		}
	}
	return ArchitectureResolution{}
}

// ResolveArchitectureID returns only the architecture id selected by
// ResolveArchitecture.
func ResolveArchitectureID(modelType, textTowerModelType string, architectures []string) string {
	return ResolveArchitecture(modelType, textTowerModelType, architectures).Architecture
}

func CleanArchitectureSignals(architectures []string) []string {
	out := make([]string, 0, len(architectures))
	for _, architecture := range architectures {
		architecture = strings.TrimSpace(architecture)
		if architecture != "" {
			out = append(out, architecture)
		}
	}
	return out
}

func architectureResolution(id, source, modelType, textTowerModelType string, architectures []string) ArchitectureResolution {
	resolution := ArchitectureResolution{
		Architecture:       id,
		Source:             source,
		ModelType:          modelType,
		TextTowerModelType: textTowerModelType,
		Architectures:      cloneStringSlice(architectures),
	}
	if profile, ok := LookupArchitectureProfile(id); ok {
		resolution.Profile = profile
	}
	return resolution.Clone()
}

func architectureIDForSignal(value string) string {
	if profile, ok := LookupArchitectureProfile(value); ok {
		return profile.ID
	}
	return ArchitectureID(value)
}

func textTowerRefinement(id, textTowerModelType string) string {
	if strings.TrimSpace(textTowerModelType) == "" {
		return ""
	}
	base, ok := LookupArchitectureProfile(id)
	if !ok || base.TextTowerID == "" {
		return ""
	}
	if architectureIDForSignal(textTowerModelType) == base.TextTowerID {
		return base.TextTowerID
	}
	return ""
}

func rerankRefinement(id string, architectures []string) string {
	base, ok := LookupArchitectureProfile(id)
	if !ok || base.Rerank {
		return ""
	}
	for _, architecture := range architectures {
		candidate, ok := LookupArchitectureProfile(architecture)
		if ok && candidate.Rerank && candidate.Family == base.Family {
			return candidate.ID
		}
	}
	return ""
}
