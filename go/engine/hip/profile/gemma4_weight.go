// SPDX-Licence-Identifier: EUPL-1.2

package profile

import "strings"

var gemma4WeightWrapperPrefixes = []string{
	"model.language_model.model.",
	"model.language_model.",
	"language_model.model.",
	"language_model.",
	"model.model.",
	"model.",
}
var gemma4WeightSkipPrefixes = []string{
	"vision_tower",
	"multi_modal_projector",
	"audio_tower",
	"embed_audio",
	"embed_vision",
}
var gemma4WeightSkipSubstrings = []string{
	"self_attn.rotary_emb",
	"input_max",
	"input_min",
	"output_max",
	"output_min",
}
var gemma4WeightModelPrefixes = []string{
	"layers.",
	"embed_tokens.",
	"embed_tokens_per_layer.",
	"norm.",
	"per_layer_model_projection.",
	"per_layer_projection_norm.",
}

// CanonicalWeightName applies the architecture registry's checkpoint
// weight-name rules. Unknown architectures pass through unchanged.
func CanonicalWeightName(architecture, name string) (string, bool) {
	settings, ok := Gemma4ArchitectureSettingsForArchitecture(architecture)
	if !ok {
		return name, true
	}
	trimmed := unwrapWeightName(strings.TrimSpace(name), settings.WeightWrapperPrefixes)
	if trimmed == "" {
		return "", false
	}
	for _, prefix := range settings.WeightSkipPrefixes {
		if strings.HasPrefix(trimmed, prefix) {
			return "", false
		}
	}
	for _, substr := range settings.WeightSkipSubstrings {
		if strings.Contains(trimmed, substr) {
			return "", false
		}
	}
	for _, prefix := range settings.WeightModelPrefixes {
		if strings.HasPrefix(trimmed, prefix) {
			return "model." + trimmed, true
		}
	}
	return trimmed, true
}

// TrimWeightWrapperPrefix removes one registered checkpoint wrapper prefix from
// name, reporting whether a Gemma-4 wrapper matched.
func TrimWeightWrapperPrefix(architecture, name string) (string, bool) {
	settings, ok := Gemma4ArchitectureSettingsForArchitecture(architecture)
	if !ok {
		return name, false
	}
	return trimOneWeightWrapper(name, settings.WeightWrapperPrefixes)
}

// UnwrapGemma4WeightName strips all Gemma-4 checkpoint wrapper prefixes from
// name.
func UnwrapGemma4WeightName(name string) string {
	return unwrapWeightName(name, gemma4WeightWrapperPrefixes)
}

// TrimOneGemma4WeightWrapper strips one Gemma-4 checkpoint wrapper prefix from
// name.
func TrimOneGemma4WeightWrapper(name string) (string, bool) {
	return trimOneWeightWrapper(name, gemma4WeightWrapperPrefixes)
}

// Gemma4WeightWrapperPrefixes returns the checkpoint wrapper prefixes used by
// Gemma-4 weight canonicalization.
func Gemma4WeightWrapperPrefixes() []string {
	return cloneStringSlice(gemma4WeightWrapperPrefixes)
}

func unwrapWeightName(name string, wrapperPrefixes []string) string {
	trimmed := name
	for {
		next, changed := trimOneWeightWrapper(trimmed, wrapperPrefixes)
		if !changed {
			return trimmed
		}
		trimmed = next
	}
}

func trimOneWeightWrapper(name string, wrapperPrefixes []string) (string, bool) {
	for _, prefix := range wrapperPrefixes {
		if strings.HasPrefix(name, prefix) {
			return strings.TrimPrefix(name, prefix), true
		}
	}
	return name, false
}
