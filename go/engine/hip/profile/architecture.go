// SPDX-Licence-Identifier: EUPL-1.2

package profile

import (
	"slices"
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/registry"
)

// ArchitectureProfile is the backend-neutral ROCm model-family metadata used
// by registry, route, and discovery surfaces.
type ArchitectureProfile = Gemma4ArchitectureSettings

var builtinArchitectureProfileIDs = []string{
	"bert",
	"bert_rerank",
	"composed",
	"deepseek",
	"deepseek_r1",
	"deltanet",
	"diffusion_gemma",
	"gemma",
	"gemma2",
	"gemma3",
	"gemma3_text",
	"gemma4",
	"gemma4_assistant",
	"gemma4_text",
	"gemma4_unified",
	"glm",
	"glm4",
	"gpt-oss",
	"granite",
	"gla",
	"gsa",
	"hermes",
	"hybrid",
	"kimi",
	"llama",
	"mamba2",
	"minimax",
	"minimax_m2",
	"mistral",
	"mixtral",
	"mla",
	"moba",
	"nsa",
	"phi",
	"qwen2",
	"qwen3",
	"qwen3_6",
	"qwen3_6_moe",
	"qwen3_moe",
	"qwen3_next",
	"retnet",
	"rwkv7",
}

var supportedNativeArchitectures = map[string]struct{}{
	"bert":                {},
	"bert_rerank":         {},
	"composed":            {},
	"deepseek":            {},
	"deepseek_r1":         {},
	"diffusion_gemma":     {},
	"gemma":               {},
	"gemma2":              {},
	"gemma3":              {},
	"gemma3_text":         {},
	"gemma4":              {},
	"gemma4_assistant":    {},
	"gemma4_text":         {},
	"gemma4_unified":      {},
	"gemma4_unified_text": {},
	"glm":                 {},
	"glm4":                {},
	"gpt-oss":             {},
	"granite":             {},
	"hermes":              {},
	"hybrid":              {},
	"kimi":                {},
	"llama":               {},
	"minimax":             {},
	"minimax_m2":          {},
	"mistral":             {},
	"mixtral":             {},
	"phi":                 {},
	"phi3":                {},
	"qwen2":               {},
	"qwen3":               {},
	"qwen3_6":             {},
	"qwen3_6_moe":         {},
	"qwen3_moe":           {},
	"qwen3_next":          {},
}

var (
	registeredArchitectureProfiles       = registry.NewOrdered[string, ArchitectureProfile]()
	registeredArchitectureProfileAliases = registry.NewOrdered[string, string]()
)

// RegisterArchitectureProfile registers or replaces a model-family profile.
// Registered profiles resolve before the built-in catalogue, so a model package
// can extend or override ROCm planning metadata without adding a root switch.
func RegisterArchitectureProfile(profile ArchitectureProfile) {
	profile = NormalizeArchitectureProfile(profile)
	if profile.ID == "" {
		return
	}
	registeredArchitectureProfiles.Put(profile.ID, profile)
	rebuildRegisteredArchitectureProfileAliases()
}

// RegisteredArchitectureProfileIDs returns extension profile IDs in
// registration order.
func RegisteredArchitectureProfileIDs() []string {
	return registeredArchitectureProfiles.Keys()
}

// RegisteredArchitectureProfiles returns extension profiles in registration
// order, with defensive copies of all slice fields.
func RegisteredArchitectureProfiles() []ArchitectureProfile {
	profiles := registeredArchitectureProfiles.Values()
	out := make([]ArchitectureProfile, 0, len(profiles))
	for _, profile := range profiles {
		out = append(out, CloneGemma4ArchitectureSettings(profile))
	}
	return out
}

// NormalizeArchitectureProfile canonicalizes a profile for registration while
// preserving explicit feature booleans such as Generation=false on staged
// loaders and Rerank=true on cross-encoders.
func NormalizeArchitectureProfile(profile ArchitectureProfile) ArchitectureProfile {
	profile = CloneGemma4ArchitectureSettings(profile)
	profile.ID = ArchitectureID(profile.ID)
	if profile.ID == "" {
		return ArchitectureProfile{}
	}
	if profile.Family == "" {
		profile.Family = ArchitectureProfileFamily(profile.ID)
	}
	if profile.ParserID == "" {
		profile.ParserID = "generic"
	}
	if profile.ToolParserID == "" {
		profile.ToolParserID = profile.ParserID
	}
	if profile.TokenizerKind == "" {
		profile.TokenizerKind = ArchitectureProfileTokenizerKindForProfile(profile)
	}
	if profile.RuntimeStatus == "" {
		if profile.NativeRuntime {
			profile.RuntimeStatus = inference.FeatureRuntimeNative
		} else {
			profile.RuntimeStatus = inference.FeatureRuntimeMetadataOnly
		}
	}
	return profile
}

// ArchitectureProfiles returns the active architecture catalogue: built-ins in
// stable order, then extension registrations that do not replace a built-in.
func ArchitectureProfiles() []ArchitectureProfile {
	out := make([]ArchitectureProfile, 0, len(builtinArchitectureProfileIDs)+len(RegisteredArchitectureProfileIDs()))
	seen := map[string]struct{}{}
	for _, id := range builtinArchitectureProfileIDs {
		profile, ok := LookupArchitectureProfile(id)
		if !ok {
			continue
		}
		out = append(out, profile)
		seen[profile.ID] = struct{}{}
	}
	for _, profile := range RegisteredArchitectureProfiles() {
		if _, ok := seen[profile.ID]; ok {
			continue
		}
		out = append(out, profile)
		seen[profile.ID] = struct{}{}
	}
	return out
}

// BuiltinArchitectureProfiles returns the active ROCm architecture profiles.
// It preserves the original API name used by CLI/report surfaces while now
// including extension registrations after the built-in catalogue.
func BuiltinArchitectureProfiles() []ArchitectureProfile {
	return ArchitectureProfiles()
}

// LookupArchitectureProfile resolves architecture to a copy-safe active
// profile.
func LookupArchitectureProfile(architecture string) (ArchitectureProfile, bool) {
	if profile, ok := registeredArchitectureProfileForArchitecture(architecture); ok {
		return profile, true
	}
	if settings, ok := Gemma4ArchitectureSettingsForArchitecture(architecture); ok {
		return settings, true
	}
	id := ArchitectureID(architecture)
	if id == "" || !KnownArchitectureProfileID(id) {
		return ArchitectureProfile{}, false
	}
	nativeRuntime := SupportedNativeArchitecture(id)
	runtimeStatus := inference.FeatureRuntimeNative
	if !nativeRuntime {
		runtimeStatus = inference.FeatureRuntimeMetadataOnly
	}
	profile := ArchitectureProfile{
		ID:                   id,
		Family:               ArchitectureProfileFamily(id),
		RuntimeStatus:        runtimeStatus,
		ParserID:             ArchitectureProfileParser(id),
		ToolParserID:         ArchitectureProfileParser(id),
		TokenizerKind:        ArchitectureProfileTokenizerKind(id),
		ChatTemplate:         ArchitectureProfileChatTemplate(id),
		GenerationRole:       "assistant",
		RequiresChatTemplate: ArchitectureProfileChat(id),
		NativeRuntime:        nativeRuntime,
		Generation:           ArchitectureProfileGeneration(id),
		Chat:                 ArchitectureProfileChat(id),
		Embeddings:           id == "bert",
		Rerank:               id == "bert_rerank",
		MoE:                  IsMoEArchitecture(id),
		LoRATargets:          ArchitectureProfileLoRATargets(id),
		LoRADefaultTargets:   ArchitectureProfileLoRADefaultTargets(id),
		LoRATargetPaths:      ArchitectureProfileLoRATargetPaths(id),
		LoRAExtendedTargets:  ArchitectureProfileLoRAExtendedTargets(id),
		QuantizationHints:    ArchitectureProfileQuantizationHints(id),
		CacheHints:           ArchitectureProfileCacheHints(id),
		Aliases:              ArchitectureProfileAliases(id),
		Notes:                ArchitectureProfileNotes(id),
	}
	if profile.ParserID == "" {
		profile.ParserID = "generic"
		profile.ToolParserID = "generic"
	}
	if !profile.Chat {
		profile.ChatTemplate = ""
		profile.GenerationRole = ""
		profile.RequiresChatTemplate = false
	}
	return CloneGemma4ArchitectureSettings(profile), true
}

// ArchitectureID returns the canonical profile id for architecture.
func ArchitectureID(architecture string) string {
	if id := Gemma4ArchitectureID(architecture); id != "" {
		return id
	}
	return NormalizeArchitecture(architecture)
}

// IsGemma4TargetArchitecture reports whether architecture identifies a Gemma-4
// target model that can own prompts, adapters, tuning runs, and fused packs.
// The attached assistant drafter is intentionally excluded.
func IsGemma4TargetArchitecture(architecture string) bool {
	switch ArchitectureID(architecture) {
	case "gemma4", "gemma4_text", "gemma4_unified":
		return true
	default:
		return false
	}
}

// IsGemma4LargeVariant reports whether Gemma-4 prompt rendering should use the
// large-variant suppressor path.
func IsGemma4LargeVariant(architecture string, numAttentionHeads int) bool {
	return numAttentionHeads >= 16 && IsGemma4TargetArchitecture(architecture)
}

// DefaultThinkingEnabled reports whether an architecture renders chat prompts
// with reasoning enabled by default. Per-request configs may still override it.
func DefaultThinkingEnabled(architecture string) bool {
	architecture = strings.TrimSpace(architecture)
	if architecture == "" {
		return false
	}
	profile, ok := LookupArchitectureProfile(architecture)
	return ok && profile.DefaultThinking
}

// AttachedOnlyArchitecture reports whether an architecture must be loaded
// attached to a target rather than as a standalone model.
func AttachedOnlyArchitecture(architecture string) bool {
	architecture = strings.TrimSpace(architecture)
	if architecture == "" {
		return false
	}
	profile, ok := LookupArchitectureProfile(architecture)
	return ok && profile.AttachedOnly
}

// ChatTemplateName returns the default chat-template id advertised for an
// architecture. It is metadata-only; callers should still ensure they implement
// the returned template before rendering.
func ChatTemplateName(architecture string) string {
	architecture = strings.TrimSpace(architecture)
	if architecture == "" {
		return ""
	}
	if profile, ok := LookupArchitectureProfile(architecture); ok {
		if profile.ChatTemplate != "" {
			return profile.ChatTemplate
		}
		if profile.Family == "qwen" {
			return "qwen"
		}
		return ""
	}
	switch NormalizeArchitecture(architecture) {
	case "gemma":
		return "gemma"
	case "qwen":
		return "qwen"
	case "llama":
		return "llama"
	default:
		return ""
	}
}

// NormalizeArchitecture canonicalizes ROCm-supported architecture identifiers.
func NormalizeArchitecture(architecture string) string {
	normalized := strings.ToLower(architecture)
	normalized = strings.ReplaceAll(normalized, "-", "_")
	normalized = strings.ReplaceAll(normalized, ".", "_")
	normalized = strings.ReplaceAll(normalized, " ", "_")
	switch {
	case normalized == "":
		return ""
	case strings.Contains(normalized, "bertforsequenceclassification") ||
		strings.Contains(normalized, "robertaforsequenceclassification") ||
		strings.Contains(normalized, "xlmrobertaforsequenceclassification") ||
		strings.Contains(normalized, "debertav2forsequenceclassification") ||
		normalized == "bert_rerank" ||
		normalized == "bert_cross_encoder":
		return "bert_rerank"
	case strings.Contains(normalized, "minimax") && strings.Contains(normalized, "m2"):
		return "minimax_m2"
	case strings.Contains(normalized, "minimax"):
		return "minimax"
	case (strings.Contains(normalized, "qwen3_5") || strings.Contains(normalized, "qwen35") ||
		strings.Contains(normalized, "qwen3_6") || strings.Contains(normalized, "qwen36")) &&
		strings.Contains(normalized, "moe"):
		return "qwen3_6_moe"
	case strings.Contains(normalized, "qwen3_5") || strings.Contains(normalized, "qwen35") ||
		strings.Contains(normalized, "qwen3_6") || strings.Contains(normalized, "qwen36"):
		return "qwen3_6"
	case strings.Contains(normalized, "qwen3") && strings.Contains(normalized, "moe"):
		return "qwen3_moe"
	case strings.Contains(normalized, "qwen3") && strings.Contains(normalized, "next"):
		return "qwen3_next"
	case strings.Contains(normalized, "qwen3"):
		return "qwen3"
	case strings.Contains(normalized, "qwen2"):
		return "qwen2"
	case strings.Contains(normalized, "deepseek"):
		if strings.Contains(normalized, "r1") {
			return "deepseek_r1"
		}
		return "deepseek"
	case strings.Contains(normalized, "gpt_oss") || strings.Contains(normalized, "gptoss"):
		return "gpt-oss"
	case strings.Contains(normalized, "deltanet") || strings.Contains(normalized, "delta_net"):
		return "deltanet"
	case normalized == "gla" || strings.Contains(normalized, "gated_linear_attention") || strings.Contains(normalized, "gatedlinearattention"):
		return "gla"
	case normalized == "gsa" || strings.Contains(normalized, "gated_slot_attention") || strings.Contains(normalized, "gatedslotattention"):
		return "gsa"
	case strings.Contains(normalized, "mamba2") || strings.Contains(normalized, "mamba_2"):
		return "mamba2"
	case normalized == "mla" || strings.Contains(normalized, "multi_head_latent_attention") || strings.Contains(normalized, "multiheadlatentattention"):
		return "mla"
	case normalized == "moba" || strings.Contains(normalized, "mixture_of_block_attention") || strings.Contains(normalized, "mixtureofblockattention"):
		return "moba"
	case normalized == "nsa" || strings.Contains(normalized, "native_sparse_attention") || strings.Contains(normalized, "nativesparseattention"):
		return "nsa"
	case strings.Contains(normalized, "retnet") || strings.Contains(normalized, "retention"):
		return "retnet"
	case strings.Contains(normalized, "rwkv7") || strings.Contains(normalized, "rwkv_7"):
		return "rwkv7"
	case strings.Contains(normalized, "diffusion_gemma") ||
		strings.Contains(normalized, "diffusiongemma") ||
		(strings.Contains(normalized, "diffusion") && strings.Contains(normalized, "gemma")):
		return "diffusion_gemma"
	case strings.Contains(normalized, "gemma4"):
		if strings.Contains(normalized, "assistant") {
			return "gemma4_assistant"
		}
		if strings.Contains(normalized, "unified") {
			if strings.Contains(normalized, "text") {
				return "gemma4_unified_text"
			}
			return "gemma4_unified"
		}
		if strings.Contains(normalized, "text") || strings.Contains(normalized, "forcausallm") {
			return "gemma4_text"
		}
		return "gemma4"
	case normalized == "gemma3_text" ||
		strings.Contains(normalized, "gemma3text") ||
		(strings.Contains(normalized, "gemma3") && strings.Contains(normalized, "text")):
		return "gemma3_text"
	case strings.Contains(normalized, "gemma3"):
		return "gemma3"
	case strings.Contains(normalized, "gemma2"):
		return "gemma2"
	case strings.Contains(normalized, "gemma"):
		return "gemma"
	case strings.Contains(normalized, "mixtral"):
		return "mixtral"
	case strings.Contains(normalized, "mistral"):
		return "mistral"
	case strings.Contains(normalized, "phi3"):
		return "phi"
	case strings.Contains(normalized, "phi4"):
		return "phi"
	case strings.Contains(normalized, "phi"):
		return "phi"
	case strings.Contains(normalized, "bert"):
		return "bert"
	case strings.Contains(normalized, "glm4"):
		return "glm4"
	case strings.Contains(normalized, "glm"):
		return "glm"
	case strings.Contains(normalized, "kimi"):
		return "kimi"
	case strings.Contains(normalized, "llama"):
		return "llama"
	case strings.Contains(normalized, "hermes"):
		return "hermes"
	case strings.Contains(normalized, "granite"):
		return "granite"
	default:
		return normalized
	}
}

func KnownArchitectureProfileID(id string) bool {
	if _, ok := registeredArchitectureProfileForArchitecture(id); ok {
		return true
	}
	return slices.Contains(builtinArchitectureProfileIDs, id)
}

func SupportedNativeArchitecture(architecture string) bool {
	if profile, ok := registeredArchitectureProfileForArchitecture(architecture); ok {
		return profile.NativeRuntime
	}
	architecture = NormalizeArchitecture(architecture)
	if architecture == "" {
		return true
	}
	_, ok := supportedNativeArchitectures[architecture]
	return ok
}

func IsMoEArchitecture(architecture string) bool {
	if profile, ok := registeredArchitectureProfileForArchitecture(architecture); ok {
		return profile.MoE
	}
	architecture = NormalizeArchitecture(architecture)
	return strings.Contains(architecture, "moe") || architecture == "mixtral" || architecture == "minimax_m2"
}

func ArchitectureProfileFamily(id string) string {
	if profile, ok := registeredArchitectureProfileForArchitecture(id); ok && profile.Family != "" {
		return profile.Family
	}
	switch id {
	case "bert", "bert_rerank":
		return "bert"
	case "qwen2", "qwen3", "qwen3_6", "qwen3_6_moe", "qwen3_moe", "qwen3_next":
		return "qwen"
	case "gemma", "gemma2", "gemma3", "gemma3_text", "gemma4", "gemma4_text", "gemma4_assistant", "gemma4_unified", "gemma4_unified_text", "diffusion_gemma":
		return "gemma"
	case "deepseek", "deepseek_r1":
		return "deepseek"
	case "gpt-oss":
		return "gpt-oss"
	case "minimax", "minimax_m2":
		return "minimax"
	case "mixtral", "mistral":
		return "mistral"
	case "glm", "glm4":
		return "glm"
	default:
		return id
	}
}

func ArchitectureProfileParser(id string) string {
	if profile, ok := registeredArchitectureProfileForArchitecture(id); ok {
		if profile.ParserID != "" {
			return profile.ParserID
		}
		if profile.ToolParserID != "" {
			return profile.ToolParserID
		}
	}
	switch id {
	case "deepseek", "deepseek_r1":
		return "deepseek-r1"
	case "gpt-oss":
		return "gpt-oss"
	case "qwen2", "qwen3", "qwen3_6", "qwen3_6_moe", "qwen3_moe", "qwen3_next":
		return "qwen"
	case "gemma", "gemma2", "gemma3", "gemma3_text", "diffusion_gemma":
		return "gemma"
	case "mixtral", "mistral":
		return "mistral"
	case "minimax", "minimax_m2":
		return "minimax"
	case "glm", "glm4":
		return "glm"
	case "bert", "bert_rerank", "phi":
		return "generic"
	default:
		return ArchitectureProfileFamily(id)
	}
}

func ArchitectureProfileGeneration(id string) bool {
	if profile, ok := registeredArchitectureProfileForArchitecture(id); ok {
		return profile.Generation
	}
	switch id {
	case "bert", "bert_rerank", "composed", "hybrid",
		"deltanet", "gla", "gsa", "mamba2", "mla", "moba", "nsa", "retnet", "rwkv7",
		"deepseek", "deepseek_r1", "gpt-oss", "kimi", "minimax_m2",
		"mixtral", "qwen3_6", "qwen3_6_moe", "qwen3_moe":
		return false
	default:
		return true
	}
}

func ArchitectureProfileChat(id string) bool {
	if profile, ok := registeredArchitectureProfileForArchitecture(id); ok {
		return profile.Chat
	}
	switch id {
	case "bert", "bert_rerank", "diffusion_gemma":
		return false
	default:
		return ArchitectureProfileGeneration(id)
	}
}

func ArchitectureProfileChatTemplate(id string) string {
	if profile, ok := registeredArchitectureProfileForArchitecture(id); ok {
		return profile.ChatTemplate
	}
	if !ArchitectureProfileChat(id) {
		return ""
	}
	family := ArchitectureProfileFamily(id)
	switch family {
	case "gpt-oss":
		return "gpt-oss"
	case "qwen", "gemma", "mistral", "minimax", "deepseek", "kimi", "glm", "hermes", "granite", "llama":
		return family
	default:
		if id != "" {
			return id
		}
		return "generic"
	}
}

// ArchitectureProfileTokenizerKind returns the tokenizer implementation token
// declared by the active architecture registry.
func ArchitectureProfileTokenizerKind(architecture string) string {
	id := ArchitectureID(architecture)
	if id == "" {
		return ""
	}
	if profile, ok := registeredArchitectureProfileForArchitecture(id); ok {
		return ArchitectureProfileTokenizerKindForProfile(profile)
	}
	if !KnownArchitectureProfileID(id) {
		return ""
	}
	return architectureProfileTokenizerKind(
		id,
		ArchitectureProfileFamily(id),
		ArchitectureProfileChatTemplate(id),
		ArchitectureProfileParser(id),
	)
}

// ArchitectureProfileTokenizerKindForProfile returns the tokenizer
// implementation token for profile, deriving the built-in family default when
// a profile does not set one explicitly.
func ArchitectureProfileTokenizerKindForProfile(profile ArchitectureProfile) string {
	profile = CloneGemma4ArchitectureSettings(profile)
	if profile.TokenizerKind != "" {
		return profile.TokenizerKind
	}
	family := profile.Family
	if family == "" {
		family = ArchitectureProfileFamily(profile.ID)
	}
	return architectureProfileTokenizerKind(profile.ID, family, profile.ChatTemplate, profile.ParserID)
}

func architectureProfileTokenizerKind(id, family, chatTemplate, parserID string) string {
	switch family {
	case "gemma4", "gemma":
		return "GemmaTokenizer"
	case "qwen":
		return "Qwen2Tokenizer"
	case "bert":
		return "BertTokenizer"
	case "mistral":
		return "MistralTokenizer"
	case "llama":
		return "LlamaTokenizer"
	default:
		if chatTemplate != "" || parserID != "" {
			return "tokenizer.json"
		}
		if id != "" {
			return ""
		}
		return ""
	}
}

func ArchitectureProfileQuantizationHints(id string) []string {
	if profile, ok := registeredArchitectureProfileForArchitecture(id); ok {
		return cloneStringSlice(profile.QuantizationHints)
	}
	hints := []string{"fp16", "bf16", "q8_0", "q4_k_m"}
	if IsMoEArchitecture(id) {
		hints = append(hints, "expert-aware")
	}
	switch id {
	case "minimax_m2":
		hints = append(hints, "jang", "jangtq", "mxtq")
	case "gpt-oss":
		hints = append(hints, "mxfp4")
	case "kimi":
		hints = append(hints, "nvfp4")
	}
	return hints
}

func ArchitectureProfileCacheHints(id string) []string {
	if profile, ok := registeredArchitectureProfileForArchitecture(id); ok {
		return cloneStringSlice(profile.CacheHints)
	}
	if id == "bert" || id == "bert_rerank" {
		return nil
	}
	if id == "composed" || id == "hybrid" {
		return []string{"default", "recurrent", "mla-latent"}
	}
	switch id {
	case "deltanet", "gla", "mamba2", "retnet", "rwkv7":
		return []string{"default", "recurrent"}
	case "mla":
		return []string{"default", "mla-latent"}
	case "gsa", "moba", "nsa":
		return []string{"default", "paged"}
	}
	hints := []string{"q8", "paged"}
	if IsMoEArchitecture(id) || id == "minimax_m2" {
		hints = append(hints, "k-q8-v-q4")
	}
	return hints
}

// ArchitectureProfileLoRATargetPolicyName returns the registry-owned adapter
// policy token for architecture.
func ArchitectureProfileLoRATargetPolicyName(architecture string) string {
	id := ArchitectureID(architecture)
	switch {
	case Gemma4LoRATargetArchitecture(id):
		return "gemma4"
	case id == "composed" || id == "hybrid":
		return "composed_mlp"
	case decoderLoRATargetArchitecture(id):
		return "decoder"
	default:
		return ""
	}
}

// ArchitectureProfileLoRATargets returns the full advertised adapter target set
// for architecture.
func ArchitectureProfileLoRATargets(architecture string) []string {
	id := ArchitectureID(architecture)
	switch {
	case id == "composed" || id == "hybrid":
		return []string{"gate_proj", "up_proj", "down_proj"}
	case decoderLoRATargetArchitecture(id):
		return []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
	default:
		return nil
	}
}

// ArchitectureProfileLoRADefaultTargets returns the narrow adapter target set
// applied when a caller requests LoRA without explicit keys.
func ArchitectureProfileLoRADefaultTargets(architecture string) []string {
	id := ArchitectureID(architecture)
	switch {
	case id == "composed" || id == "hybrid":
		return []string{"gate_proj", "up_proj", "down_proj"}
	case decoderLoRATargetArchitecture(id):
		return []string{"q_proj", "v_proj"}
	default:
		return nil
	}
}

// ArchitectureProfileLoRATargetPaths returns target-key canonicalization rules
// for adapter metadata and linear resolution.
func ArchitectureProfileLoRATargetPaths(architecture string) map[string]string {
	id := ArchitectureID(architecture)
	switch {
	case id == "composed" || id == "hybrid":
		return cloneStringMap(map[string]string{
			"gate_proj":     "mlp.gate_proj",
			"mlp.gate_proj": "mlp.gate_proj",
			"up_proj":       "mlp.up_proj",
			"mlp.up_proj":   "mlp.up_proj",
			"down_proj":     "mlp.down_proj",
			"mlp.down_proj": "mlp.down_proj",
		})
	case decoderLoRATargetArchitecture(id):
		return cloneStringMap(map[string]string{
			"q_proj":           "self_attn.q_proj",
			"self_attn.q_proj": "self_attn.q_proj",
			"k_proj":           "self_attn.k_proj",
			"self_attn.k_proj": "self_attn.k_proj",
			"v_proj":           "self_attn.v_proj",
			"self_attn.v_proj": "self_attn.v_proj",
			"o_proj":           "self_attn.o_proj",
			"self_attn.o_proj": "self_attn.o_proj",
			"gate_proj":        "mlp.gate_proj",
			"mlp.gate_proj":    "mlp.gate_proj",
			"up_proj":          "mlp.up_proj",
			"mlp.up_proj":      "mlp.up_proj",
			"down_proj":        "mlp.down_proj",
			"mlp.down_proj":    "mlp.down_proj",
		})
	default:
		return nil
	}
}

// ArchitectureProfileLoRAExtendedTargets returns adapter targets that require
// an explicit opt-in.
func ArchitectureProfileLoRAExtendedTargets(architecture string) []string {
	return nil
}

func decoderLoRATargetArchitecture(id string) bool {
	switch ArchitectureID(id) {
	case "deepseek", "deepseek_r1",
		"gemma", "gemma2", "gemma3", "gemma3_text",
		"glm", "glm4",
		"gpt-oss",
		"granite",
		"hermes",
		"kimi",
		"llama",
		"minimax", "minimax_m2",
		"mistral", "mixtral",
		"phi",
		"qwen2", "qwen3", "qwen3_6", "qwen3_6_moe", "qwen3_moe", "qwen3_next":
		return true
	default:
		return false
	}
}

func ArchitectureProfileAliases(id string) []string {
	if profile, ok := registeredArchitectureProfileForArchitecture(id); ok {
		return cloneStringSlice(profile.Aliases)
	}
	switch id {
	case "bert":
		return []string{"BertModel", "BertForMaskedLM"}
	case "bert_rerank":
		return []string{"BertForSequenceClassification", "RobertaForSequenceClassification", "XLMRobertaForSequenceClassification", "DebertaV2ForSequenceClassification"}
	case "composed":
		return []string{"composed"}
	case "deepseek":
		return []string{"DeepseekV3ForCausalLM", "DeepSeekV3ForCausalLM"}
	case "deepseek_r1":
		return []string{"DeepseekR1ForCausalLM", "DeepSeekR1ForCausalLM"}
	case "deltanet":
		return []string{"DeltaNetForCausalLM", "DeltaNetModel"}
	case "gemma":
		return []string{"GemmaForCausalLM"}
	case "gemma2":
		return []string{"Gemma2ForCausalLM"}
	case "gemma3":
		return []string{"Gemma3ForCausalLM"}
	case "gemma3_text":
		return []string{"Gemma3TextForCausalLM", "Gemma3ForCausalLM"}
	case "glm", "glm4":
		return []string{"GlmForCausalLM", "ChatGLMForConditionalGeneration"}
	case "gpt-oss":
		return []string{"GptOssForCausalLM", "GPTOSSForCausalLM"}
	case "granite":
		return []string{"GraniteForCausalLM"}
	case "gla":
		return []string{"GLAForCausalLM", "GatedLinearAttentionForCausalLM"}
	case "gsa":
		return []string{"GSAForCausalLM", "GatedSlotAttentionForCausalLM"}
	case "hermes":
		return []string{"HermesForCausalLM", "NousHermesForCausalLM"}
	case "hybrid":
		return []string{"hybrid"}
	case "kimi":
		return []string{"KimiForCausalLM", "KimiK2ForCausalLM", "MoonshotForCausalLM"}
	case "llama":
		return []string{"LlamaForCausalLM"}
	case "mamba2":
		return []string{"Mamba2ForCausalLM", "Mamba2Model"}
	case "minimax_m2":
		return []string{"MiniMaxM2ForCausalLM"}
	case "mistral":
		return []string{"MistralForCausalLM"}
	case "mixtral":
		return []string{"MixtralForCausalLM"}
	case "mla":
		return []string{"MLAForCausalLM", "MultiHeadLatentAttentionForCausalLM"}
	case "moba":
		return []string{"MoBAForCausalLM", "MixtureOfBlockAttentionForCausalLM"}
	case "nsa":
		return []string{"NSAForCausalLM", "NativeSparseAttentionForCausalLM"}
	case "phi":
		return []string{"PhiForCausalLM", "Phi3ForCausalLM", "Phi4ForCausalLM"}
	case "qwen2":
		return []string{"Qwen2ForCausalLM", "Qwen2.5ForCausalLM", "Qwen2_5ForCausalLM"}
	case "qwen3":
		return []string{"Qwen3ForCausalLM"}
	case "qwen3_6":
		return []string{"Qwen3_5ForConditionalGeneration", "Qwen3.5ForConditionalGeneration", "Qwen3_6ForConditionalGeneration", "Qwen3.6ForConditionalGeneration"}
	case "qwen3_6_moe":
		return []string{"Qwen3_5MoeForConditionalGeneration", "Qwen3.5MoeForConditionalGeneration", "Qwen3_6MoeForConditionalGeneration", "Qwen3.6MoeForConditionalGeneration"}
	case "qwen3_moe":
		return []string{"Qwen3MoeForCausalLM"}
	case "qwen3_next":
		return []string{"Qwen3NextForCausalLM"}
	case "retnet":
		return []string{"RetNetForCausalLM", "RetNetModel"}
	case "rwkv7":
		return []string{"RWKV7ForCausalLM", "RWKV7Model"}
	default:
		return nil
	}
}

func ArchitectureProfileNotes(id string) []string {
	if profile, ok := registeredArchitectureProfileForArchitecture(id); ok {
		return cloneStringSlice(profile.Notes)
	}
	switch id {
	case "bert":
		return []string{"native staged encoder loader; embedding pooling kernels pending"}
	case "bert_rerank":
		return []string{"native staged cross-encoder loader; scorer kernels pending"}
	case "composed", "hybrid":
		return []string{"config-composed sequence-mixer loader contract is registered; generic HIP composed runner remains pending"}
	case "deltanet", "gla", "gsa", "mamba2", "mla", "moba", "nsa", "retnet", "rwkv7":
		return []string{"go-mlx metal model family recognised for reactive route parity; ROCm runtime loader remains metadata-only"}
	case "diffusion_gemma":
		return []string{"block-diffusion Gemma model; trunk metadata is recognised and diffusion sampler is routed through the diffuse command"}
	case "qwen3_6":
		return []string{"native staged hybrid linear-attention config/tokenizer loader; standalone generation smoke remains pending"}
	case "qwen3_6_moe":
		return []string{"native staged hybrid linear-attention and sparse-expert config/tokenizer loader; standalone generation smoke remains pending"}
	case "qwen3_moe", "mixtral", "deepseek", "deepseek_r1", "gpt-oss", "kimi", "minimax_m2":
		return []string{"native staged sparse/MoE config-tokenizer path; model-integrated expert decode remains pending"}
	default:
		return nil
	}
}

func registeredArchitectureProfileForArchitecture(architecture string) (ArchitectureProfile, bool) {
	for _, key := range architectureProfileLookupKeys(architecture) {
		if profile, ok := registeredArchitectureProfiles.Get(key); ok {
			return CloneGemma4ArchitectureSettings(profile), true
		}
		if id, ok := registeredArchitectureProfileAliases.Get(key); ok {
			if profile, ok := registeredArchitectureProfiles.Get(id); ok {
				return CloneGemma4ArchitectureSettings(profile), true
			}
		}
	}
	return ArchitectureProfile{}, false
}

func registerArchitectureProfileAlias(id, alias string) {
	for _, key := range architectureProfileLookupKeys(alias) {
		registeredArchitectureProfileAliases.Put(key, id)
	}
}

func rebuildRegisteredArchitectureProfileAliases() {
	registeredArchitectureProfileAliases.Restore(nil, nil)
	for _, profile := range registeredArchitectureProfiles.Values() {
		registerArchitectureProfileAlias(profile.ID, profile.ID)
		for _, alias := range profile.Aliases {
			registerArchitectureProfileAlias(profile.ID, alias)
		}
	}
}

func architectureProfileLookupKeys(value string) []string {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}
	keys := make([]string, 0, 3)
	appendKey := func(key string) {
		key = strings.TrimSpace(key)
		if key == "" {
			return
		}
		if slices.Contains(keys, key) {
			return
		}
		keys = append(keys, key)
	}
	appendKey(value)
	appendKey(ArchitectureID(value))
	appendKey(NormalizeArchitecture(value))
	return keys
}
