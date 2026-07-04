// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"strconv"
	"strings"

	"dappco.re/go/inference"
	rocmprofile "dappco.re/go/inference/engine/hip/profile"
)

type Gemma4ArchitectureSettings = rocmprofile.Gemma4ArchitectureSettings

type ROCmArchitectureProfile = Gemma4ArchitectureSettings

const ROCmArchitectureResolutionContract = "rocm-architecture-resolution-v1"

// ROCmArchitectureResolution is the shared dispatch-resolution result for a
// model config's architecture signals. It preserves the source signal so API
// consumers can distinguish wrapper identity from the runtime profile ROCm will
// use for parser, cache, and load-feature decisions.
type ROCmArchitectureResolution struct {
	Contract           string                  `json:"contract,omitempty"`
	Architecture       string                  `json:"architecture,omitempty"`
	Source             string                  `json:"source,omitempty"`
	ModelType          string                  `json:"model_type,omitempty"`
	TextTowerModelType string                  `json:"text_tower_model_type,omitempty"`
	Architectures      []string                `json:"architectures,omitempty"`
	Profile            ROCmArchitectureProfile `json:"profile,omitempty"`
}

func (resolution ROCmArchitectureResolution) Matched() bool {
	return strings.TrimSpace(resolution.Architecture) != ""
}

func (resolution ROCmArchitectureResolution) clone() ROCmArchitectureResolution {
	resolution.Architectures = append([]string(nil), resolution.Architectures...)
	resolution.Profile = cloneGemma4ArchitectureSettings(resolution.Profile)
	return resolution
}

func DefaultGemma4ArchitectureSettings() []Gemma4ArchitectureSettings {
	return rocmprofile.DefaultGemma4ArchitectureSettings()
}

func RegisterROCmArchitectureProfile(profile ROCmArchitectureProfile) {
	rocmprofile.RegisterArchitectureProfile(profile)
}

func RegisteredROCmArchitectureProfileIDs() []string {
	return rocmprofile.RegisteredArchitectureProfileIDs()
}

func RegisteredROCmArchitectureProfiles() []ROCmArchitectureProfile {
	return rocmprofile.RegisteredArchitectureProfiles()
}

func ROCmArchitectureProfiles() []ROCmArchitectureProfile {
	return rocmprofile.ArchitectureProfiles()
}

func DefaultROCmArchitectureProfiles() []ROCmArchitectureProfile {
	return rocmprofile.BuiltinArchitectureProfiles()
}

func ROCmArchitectureID(architecture string) string {
	return rocmprofile.ArchitectureID(architecture)
}

// ResolveROCmArchitecture maps config.json architecture signals to a
// structured registry dispatch result. This is the ROCm-side analogue of
// go-mlx/profile.ResolveArchitecture plus source/profile metadata for API
// consumers.
func ResolveROCmArchitecture(modelType, textTowerModelType string, architectures []string) ROCmArchitectureResolution {
	return rocmArchitectureResolutionFromProfile(rocmprofile.ResolveArchitecture(modelType, textTowerModelType, architectures))
}

// ROCmResolveArchitecture maps config.json architecture signals to the
// registry id that API consumers should use for profile lookup. The order
// follows go-mlx's reactive resolver: top-level model_type first, refined by a
// declared text tower or rerank architecture when applicable; then text_config;
// then architectures fallback.
func ROCmResolveArchitecture(modelType, textTowerModelType string, architectures []string) string {
	return ResolveROCmArchitecture(modelType, textTowerModelType, architectures).Architecture
}

func cleanROCmArchitectureSignals(architectures []string) []string {
	return rocmprofile.CleanArchitectureSignals(architectures)
}

func rocmArchitectureResolutionFromProfile(profileResolution rocmprofile.ArchitectureResolution) ROCmArchitectureResolution {
	profile := cloneGemma4ArchitectureSettings(profileResolution.Profile)
	resolution := ROCmArchitectureResolution{
		Contract:           ROCmArchitectureResolutionContract,
		Architecture:       profileResolution.Architecture,
		Source:             profileResolution.Source,
		ModelType:          profileResolution.ModelType,
		TextTowerModelType: profileResolution.TextTowerModelType,
		Architectures:      append([]string(nil), profileResolution.Architectures...),
		Profile:            profile,
	}
	return resolution.clone()
}

func rocmModelIdentityWithResolvedArchitecture(model inference.ModelIdentity) inference.ModelIdentity {
	resolved := firstNonEmptyString(
		model.Labels["engine_architecture_resolved"],
		model.Labels["architecture_resolved"],
	)
	if strings.TrimSpace(resolved) == "" {
		return model
	}
	model.Architecture = ROCmArchitectureID(resolved)
	return model
}

func rocmApplyArchitectureResolutionLabels(labels map[string]string, cfg rocmModelPackConfigProbe) {
	if labels == nil {
		return
	}
	rocmApplyModelConfigProbeLabels(labels, cfg)
	architectures := append([]string(nil), cfg.Architectures...)
	architectures = append(architectures, cfg.TextConfig.Architectures...)
	resolution := ResolveROCmArchitecture(cfg.ModelType, cfg.TextConfig.ModelType, architectures)
	if !resolution.Matched() {
		return
	}
	resolved := resolution.Architecture
	labels["architecture_resolution_contract"] = resolution.Contract
	labels["engine_architecture_resolution_contract"] = resolution.Contract
	labels["architecture_resolved"] = resolved
	labels["engine_architecture_resolved"] = resolved
	labels["architecture_resolution_source"] = resolution.Source
	if resolution.ModelType != "" {
		labels["architecture_model_type"] = resolution.ModelType
	}
	if resolution.TextTowerModelType != "" {
		labels["architecture_text_tower_model_type"] = resolution.TextTowerModelType
	}
	if len(resolution.Architectures) > 0 {
		labels["architecture_class_count"] = strconv.Itoa(len(resolution.Architectures))
	}
	if profile := resolution.Profile; profile.ID != "" {
		labels["engine_architecture_resolved_family"] = profile.Family
		labels["engine_architecture_resolved_parser"] = profile.ParserID
		if profile.TokenizerKind != "" {
			labels["engine_architecture_resolved_tokenizer_kind"] = profile.TokenizerKind
		}
		labels["engine_architecture_resolved_chat_template"] = profile.ChatTemplate
		labels["engine_architecture_resolved_native_runtime"] = strconv.FormatBool(profile.NativeRuntime)
		labels["engine_architecture_resolved_generation"] = strconv.FormatBool(profile.Generation)
		labels["engine_architecture_resolved_chat"] = strconv.FormatBool(profile.Chat)
		labels["engine_architecture_resolved_moe"] = strconv.FormatBool(profile.MoE)
	}
}

func ROCmArchitectureProfileForArchitecture(architecture string) (ROCmArchitectureProfile, bool) {
	return rocmprofile.LookupArchitectureProfile(architecture)
}

func ROCmArchitectureSettingsForArchitecture(architecture string) (Gemma4ArchitectureSettings, bool) {
	return Gemma4ArchitectureSettingsForArchitecture(architecture)
}

func ROCmDefaultThinkingEnabled(architecture string) bool {
	profile, ok := ROCmArchitectureProfileForArchitecture(architecture)
	return ok && profile.DefaultThinking
}

func ROCmAttachedOnlyArchitecture(architecture string) bool {
	profile, ok := ROCmArchitectureProfileForArchitecture(architecture)
	return ok && profile.AttachedOnly
}

func ROCmRequiresChatTemplate(architecture string) bool {
	profile, ok := ROCmArchitectureProfileForArchitecture(architecture)
	return ok && profile.RequiresChatTemplate
}

func ROCmChatTemplateID(architecture string) (string, bool) {
	profile, ok := ROCmArchitectureProfileForArchitecture(architecture)
	if !ok {
		return "", false
	}
	if profile.ChatTemplate != "" {
		return profile.ChatTemplate, true
	}
	if profile.Family == "qwen" {
		return "qwen", true
	}
	return "", false
}

func ROCmGenerationRole(architecture string) (string, bool) {
	profile, ok := ROCmArchitectureProfileForArchitecture(architecture)
	if !ok || profile.GenerationRole == "" {
		return "", false
	}
	return profile.GenerationRole, true
}

func ROCmReasoningParserID(architecture string) (string, bool) {
	profile, ok := ROCmArchitectureProfileForArchitecture(architecture)
	if !ok || profile.ParserID == "" {
		return "", false
	}
	return profile.ParserID, true
}

func ROCmToolParserID(architecture string) (string, bool) {
	profile, ok := ROCmArchitectureProfileForArchitecture(architecture)
	if !ok || profile.ToolParserID == "" {
		return "", false
	}
	return profile.ToolParserID, true
}

func ROCmTokenizerKind(architecture string) (string, bool) {
	kind := rocmprofile.ArchitectureProfileTokenizerKind(architecture)
	return kind, kind != ""
}

func rocmTokenizerKindForArchitectureProfile(profile ROCmArchitectureProfile) string {
	return rocmprofile.ArchitectureProfileTokenizerKindForProfile(profile)
}

// ROCmCanonicalWeightName applies the architecture registry's checkpoint
// weight-name rules. Unknown architectures pass through unchanged.
func ROCmCanonicalWeightName(architecture, name string) (string, bool) {
	return rocmprofile.CanonicalWeightName(architecture, name)
}

func ROCmTrimWeightWrapperPrefix(architecture, name string) (string, bool) {
	return rocmprofile.TrimWeightWrapperPrefix(architecture, name)
}

func Gemma4ArchitectureSettingsForArchitecture(architecture string) (Gemma4ArchitectureSettings, bool) {
	return rocmprofile.Gemma4ArchitectureSettingsForArchitecture(architecture)
}

func cloneGemma4ArchitectureSettings(settings Gemma4ArchitectureSettings) Gemma4ArchitectureSettings {
	return rocmprofile.CloneGemma4ArchitectureSettings(settings)
}

func rocmApplyGemma4ArchitectureSettingsLabels(labels map[string]string, settings Gemma4ArchitectureSettings) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if settings.ID == "" {
		return labels
	}
	labels["engine_architecture_profile"] = settings.ID
	labels["engine_architecture_family"] = settings.Family
	labels["engine_architecture_native_runtime"] = strconv.FormatBool(settings.NativeRuntime)
	labels["engine_architecture_generation"] = strconv.FormatBool(settings.Generation)
	labels["engine_architecture_chat"] = strconv.FormatBool(settings.Chat)
	if settings.RuntimeStatus != "" {
		labels["engine_architecture_runtime_status"] = string(settings.RuntimeStatus)
	}
	if settings.ParserID != "" {
		labels["engine_architecture_reasoning_parser"] = settings.ParserID
		if labels["reasoning_parser"] == "" {
			labels["reasoning_parser"] = settings.ParserID
		}
	}
	if settings.ToolParserID != "" {
		labels["engine_architecture_tool_parser"] = settings.ToolParserID
		if labels["tool_parser"] == "" {
			labels["tool_parser"] = settings.ToolParserID
		}
	}
	if settings.TokenizerKind != "" {
		labels["engine_architecture_tokenizer_kind"] = settings.TokenizerKind
	}
	labels["engine_architecture_embeddings"] = strconv.FormatBool(settings.Embeddings)
	labels["engine_architecture_rerank"] = strconv.FormatBool(settings.Rerank)
	labels["engine_architecture_moe"] = strconv.FormatBool(settings.MoE)
	labels["engine_architecture_attached_only"] = strconv.FormatBool(settings.AttachedOnly)
	if settings.TextTowerID != "" {
		labels["engine_text_tower"] = settings.TextTowerID
	}
	if settings.GenerationRole != "" {
		labels["engine_generation_role"] = settings.GenerationRole
		if labels["generation_role"] == "" {
			labels["generation_role"] = settings.GenerationRole
		}
	}
	labels["engine_default_thinking"] = strconv.FormatBool(settings.DefaultThinking)
	labels["engine_requires_chat_template"] = strconv.FormatBool(settings.RequiresChatTemplate)
	if settings.ChatTemplate != "" {
		labels["engine_chat_template"] = settings.ChatTemplate
		if labels["chat_template"] == "" || labels["chat_template"] == "present" {
			labels["chat_template"] = settings.ChatTemplate
		}
	}
	if len(settings.QuantizationHints) > 0 {
		labels["engine_architecture_quantization_hints"] = strings.Join(settings.QuantizationHints, ",")
	}
	if len(settings.CacheHints) > 0 {
		labels["engine_architecture_cache_hints"] = strings.Join(settings.CacheHints, ",")
	}
	if len(settings.Notes) > 0 {
		labels["engine_architecture_notes"] = strings.Join(settings.Notes, " | ")
	}
	if len(settings.Aliases) > 0 {
		labels["engine_architecture_aliases"] = strings.Join(settings.Aliases, ",")
	}
	if len(settings.WeightWrapperPrefixes) > 0 ||
		len(settings.WeightSkipPrefixes) > 0 ||
		len(settings.WeightSkipSubstrings) > 0 ||
		len(settings.WeightModelPrefixes) > 0 {
		labels["engine_weight_policy"] = "gemma4"
		labels["engine_weight_policy_source"] = "model_registry"
		labels["engine_weight_wrapper_prefixes"] = strings.Join(settings.WeightWrapperPrefixes, ",")
		labels["engine_weight_skip_prefixes"] = strings.Join(settings.WeightSkipPrefixes, ",")
		labels["engine_weight_skip_substrings"] = strings.Join(settings.WeightSkipSubstrings, ",")
		labels["engine_weight_model_prefixes"] = strings.Join(settings.WeightModelPrefixes, ",")
		labels["gemma4_weight_policy"] = "model_registry"
		labels["gemma4_weight_wrapper_prefixes"] = strings.Join(settings.WeightWrapperPrefixes, ",")
		labels["gemma4_weight_skip_prefixes"] = strings.Join(settings.WeightSkipPrefixes, ",")
		labels["gemma4_weight_skip_substrings"] = strings.Join(settings.WeightSkipSubstrings, ",")
		labels["gemma4_weight_model_prefixes"] = strings.Join(settings.WeightModelPrefixes, ",")
	}
	return labels
}

func rocmApplyStaticGemma4ModelProfileLabels(labels map[string]string, architecture string) map[string]string {
	settings, ok := Gemma4ArchitectureSettingsForArchitecture(architecture)
	if !ok {
		return labels
	}
	if labels == nil {
		labels = map[string]string{}
	}
	labels["engine_registry"] = rocmModelRegistryName
	labels["engine_profile"] = "gemma4"
	labels["engine_profile_family"] = settings.Family
	labels["engine_profile_source"] = "model_config"
	labels["engine_profile_matched"] = "true"
	labels["engine_profile_reactive"] = "true"
	labels["engine_profile_architecture"] = settings.ID
	rocmApplyGemma4ArchitectureSettingsLabels(labels, settings)
	rocmApplyGemma4EngineFeatureLabels(labels, Gemma4EngineFeatures{}, Gemma4DeclaredFeatures{})
	if policy, ok := Gemma4LoRATargetPolicyForArchitecture(settings.ID); ok {
		rocmApplyGemma4LoRAPolicyLabels(labels, settings.ID, policy)
	}
	return labels
}
