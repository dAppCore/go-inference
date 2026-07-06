// SPDX-Licence-Identifier: EUPL-1.2

package profile

import (
	"maps"
	"strings"

	"dappco.re/go/inference"
)

// Gemma4ArchitectureSettings is the Gemma-4 family profile used by ROCm
// routing, tokenizer, cache, LoRA, and model-pack metadata.
type Gemma4ArchitectureSettings struct {
	ID                    string                         `json:"id,omitempty"`
	Family                string                         `json:"family,omitempty"`
	TextTowerID           string                         `json:"text_tower_id,omitempty"`
	RuntimeStatus         inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	ParserID              string                         `json:"parser_id,omitempty"`
	ToolParserID          string                         `json:"tool_parser_id,omitempty"`
	TokenizerKind         string                         `json:"tokenizer_kind,omitempty"`
	ChatTemplate          string                         `json:"chat_template,omitempty"`
	GenerationRole        string                         `json:"generation_role,omitempty"`
	DefaultThinking       bool                           `json:"default_thinking,omitempty"`
	RequiresChatTemplate  bool                           `json:"requires_chat_template,omitempty"`
	LoRATargets           []string                       `json:"lora_targets,omitempty"`
	LoRADefaultTargets    []string                       `json:"lora_default_targets,omitempty"`
	LoRATargetPaths       map[string]string              `json:"lora_target_paths,omitempty"`
	LoRAExtendedTargets   []string                       `json:"lora_extended_targets,omitempty"`
	NativeRuntime         bool                           `json:"native_runtime,omitempty"`
	Generation            bool                           `json:"generation,omitempty"`
	Chat                  bool                           `json:"chat,omitempty"`
	Embeddings            bool                           `json:"embeddings,omitempty"`
	Rerank                bool                           `json:"rerank,omitempty"`
	MoE                   bool                           `json:"moe,omitempty"`
	AttachedOnly          bool                           `json:"attached_only,omitempty"`
	WeightWrapperPrefixes []string                       `json:"weight_wrapper_prefixes,omitempty"`
	WeightSkipPrefixes    []string                       `json:"weight_skip_prefixes,omitempty"`
	WeightSkipSubstrings  []string                       `json:"weight_skip_substrings,omitempty"`
	WeightModelPrefixes   []string                       `json:"weight_model_prefixes,omitempty"`
	QuantizationHints     []string                       `json:"quantization_hints,omitempty"`
	CacheHints            []string                       `json:"cache_hints,omitempty"`
	Notes                 []string                       `json:"notes,omitempty"`
	Aliases               []string                       `json:"aliases,omitempty"`
}

var defaultGemma4ArchitectureProfileIDs = []string{"gemma4", "gemma4_text", "gemma4_unified", "gemma4_assistant"}

var gemma4QuantizationHints = []string{"bf16", "q8", "q6", "q4", "mxfp8", "mxfp4"}
var gemma4CacheHints = []string{"q8", "paged", "k-q8-v-q4", "retained-state"}

// DefaultGemma4ArchitectureSettings returns the registry-ready Gemma-4 target
// and attached-drafter architecture profiles.
func DefaultGemma4ArchitectureSettings() []Gemma4ArchitectureSettings {
	out := make([]Gemma4ArchitectureSettings, 0, len(defaultGemma4ArchitectureProfileIDs))
	for _, id := range defaultGemma4ArchitectureProfileIDs {
		settings, ok := Gemma4ArchitectureSettingsForArchitecture(id)
		if !ok {
			continue
		}
		out = append(out, CloneGemma4ArchitectureSettings(settings))
	}
	return out
}

// Gemma4ArchitectureSettingsForArchitecture returns Gemma-4 family settings
// for architecture.
func Gemma4ArchitectureSettingsForArchitecture(architecture string) (Gemma4ArchitectureSettings, bool) {
	id := Gemma4ArchitectureID(architecture)
	switch id {
	case "gemma4", "gemma4_text", "gemma4_unified":
		settings := Gemma4ArchitectureSettings{
			ID:                    id,
			Family:                "gemma4",
			RuntimeStatus:         inference.FeatureRuntimeNative,
			ParserID:              "gemma",
			ToolParserID:          "gemma",
			TokenizerKind:         "GemmaTokenizer",
			ChatTemplate:          "gemma4_hf_turn",
			GenerationRole:        "model",
			DefaultThinking:       true,
			RequiresChatTemplate:  true,
			LoRATargets:           cloneStringSlice(gemma4LoRATargets),
			LoRADefaultTargets:    cloneStringSlice(gemma4LoRADefaultTargets),
			LoRATargetPaths:       cloneStringMap(gemma4LoRATargetPaths),
			LoRAExtendedTargets:   cloneStringSlice(gemma4LoRAExtendedTargets),
			NativeRuntime:         true,
			Generation:            true,
			Chat:                  true,
			WeightWrapperPrefixes: cloneStringSlice(gemma4WeightWrapperPrefixes),
			WeightSkipPrefixes:    cloneStringSlice(gemma4WeightSkipPrefixes),
			WeightSkipSubstrings:  cloneStringSlice(gemma4WeightSkipSubstrings),
			WeightModelPrefixes:   cloneStringSlice(gemma4WeightModelPrefixes),
			QuantizationHints:     cloneStringSlice(gemma4QuantizationHints),
			CacheHints:            cloneStringSlice(gemma4CacheHints),
		}
		switch id {
		case "gemma4":
			settings.TextTowerID = "gemma4_text"
			settings.Aliases = []string{"Gemma4ForConditionalGeneration"}
		case "gemma4_unified":
			settings.Aliases = []string{"Gemma4UnifiedForConditionalGeneration"}
		case "gemma4_text":
			settings.Aliases = []string{"Gemma4ForCausalLM", "Gemma4TextForCausalLM"}
		}
		return settings, true
	case "gemma4_assistant":
		return Gemma4ArchitectureSettings{
			ID:                "gemma4_assistant",
			Family:            "gemma4",
			RuntimeStatus:     inference.FeatureRuntimeNative,
			ParserID:          "gemma",
			ToolParserID:      "gemma",
			TokenizerKind:     "GemmaTokenizer",
			NativeRuntime:     true,
			AttachedOnly:      true,
			QuantizationHints: cloneStringSlice(gemma4QuantizationHints),
			CacheHints:        []string{"retained-state", "attached-drafter"},
			Notes:             []string{"attached MTP drafter; standalone generation unsupported; load beside a Gemma 4 target"},
			Aliases:           []string{"Gemma4AssistantForCausalLM"},
		}, true
	default:
		return Gemma4ArchitectureSettings{}, false
	}
}

// Gemma4ArchitectureID returns the canonical Gemma-4 family id for
// architecture, or "" when architecture is outside the Gemma-4 family.
func Gemma4ArchitectureID(architecture string) string {
	normalized := strings.ToLower(strings.TrimSpace(architecture))
	normalized = strings.ReplaceAll(normalized, "-", "_")
	normalized = strings.ReplaceAll(normalized, ".", "_")
	normalized = strings.ReplaceAll(normalized, " ", "_")
	switch {
	case normalized == "":
		return ""
	case strings.Contains(normalized, "gemma4assistant"):
		return "gemma4_assistant"
	case normalized == "gemma4_assistant" || strings.Contains(normalized, "assistant"):
		return "gemma4_assistant"
	case normalized == "gemma4_unified_text":
		return "gemma4_text"
	case normalized == "gemma4_unified" || strings.Contains(normalized, "gemma4unified"):
		return "gemma4_unified"
	case normalized == "gemma4_text" ||
		strings.Contains(normalized, "gemma4text") ||
		(strings.Contains(normalized, "gemma4") && strings.Contains(normalized, "causallm")):
		return "gemma4_text"
	case normalized == "gemma4" || strings.Contains(normalized, "gemma4"):
		return "gemma4"
	default:
		return ""
	}
}

// CloneGemma4ArchitectureSettings returns a deep copy of settings.
func CloneGemma4ArchitectureSettings(settings Gemma4ArchitectureSettings) Gemma4ArchitectureSettings {
	settings.WeightWrapperPrefixes = cloneStringSlice(settings.WeightWrapperPrefixes)
	settings.LoRATargets = cloneStringSlice(settings.LoRATargets)
	settings.LoRADefaultTargets = cloneStringSlice(settings.LoRADefaultTargets)
	settings.LoRATargetPaths = cloneStringMap(settings.LoRATargetPaths)
	settings.LoRAExtendedTargets = cloneStringSlice(settings.LoRAExtendedTargets)
	settings.WeightSkipPrefixes = cloneStringSlice(settings.WeightSkipPrefixes)
	settings.WeightSkipSubstrings = cloneStringSlice(settings.WeightSkipSubstrings)
	settings.WeightModelPrefixes = cloneStringSlice(settings.WeightModelPrefixes)
	settings.QuantizationHints = cloneStringSlice(settings.QuantizationHints)
	settings.CacheHints = cloneStringSlice(settings.CacheHints)
	settings.Notes = cloneStringSlice(settings.Notes)
	settings.Aliases = cloneStringSlice(settings.Aliases)
	return settings
}

func cloneStringSlice(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	return append([]string(nil), values...)
}

func cloneStringMap(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	out := make(map[string]string, len(values))
	maps.Copy(out, values)
	return out
}
