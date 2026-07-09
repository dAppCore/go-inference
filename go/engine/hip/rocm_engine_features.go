// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"strconv"
	"strings"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const rocmEngineFeaturesContract = "rocm-engine-features-v1"

// ROCmEngineFeatures is the backend-neutral feature declaration derived from a
// resolved model profile. It is the ROCm-side analogue of go-mlx's model-owned
// EngineFeatures: consumers can ask the loaded model/profile what runtime and
// parser paths it enables without hard-coding a family switch.
type ROCmEngineFeatures struct {
	Contract                    string                         `json:"contract,omitempty"`
	Architecture                string                         `json:"architecture,omitempty"`
	Family                      string                         `json:"family,omitempty"`
	RuntimeStatus               inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	ReasoningParserID           string                         `json:"reasoning_parser_id,omitempty"`
	ToolParserID                string                         `json:"tool_parser_id,omitempty"`
	ChatTemplateID              string                         `json:"chat_template_id,omitempty"`
	NativeRuntime               bool                           `json:"native_runtime,omitempty"`
	DirectGreedyToken           bool                           `json:"direct_greedy_token,omitempty"`
	NativeMLPMatVec             bool                           `json:"native_mlp_matvec,omitempty"`
	NativeLinearMatVec          bool                           `json:"native_linear_matvec,omitempty"`
	NativeQ6BitstreamMatVec     bool                           `json:"native_q6_bitstream_matvec,omitempty"`
	NativeAttentionOMatVec      bool                           `json:"native_attention_o_matvec,omitempty"`
	NativeFixedSlidingAttention bool                           `json:"native_fixed_sliding_attention,omitempty"`
	GenerationStream            bool                           `json:"generation_stream,omitempty"`
	AsyncDecodePrefetch         bool                           `json:"async_decode_prefetch,omitempty"`
	ModelContextWindow          bool                           `json:"model_context_window,omitempty"`
	TextGenerate                bool                           `json:"text_generate,omitempty"`
	DeviceKVState               bool                           `json:"device_kv_state,omitempty"`
	FixedSlidingCache           bool                           `json:"fixed_sliding_cache,omitempty"`
	FixedSlidingCacheBound      bool                           `json:"fixed_sliding_cache_bound,omitempty"`
	CompiledLayerDecode         bool                           `json:"compiled_layer_decode,omitempty"`
	PipelinedDecode             bool                           `json:"pipelined_decode,omitempty"`
	ReasoningParse              bool                           `json:"reasoning_parse,omitempty"`
	ToolParse                   bool                           `json:"tool_parse,omitempty"`
	ChatTemplate                bool                           `json:"chat_template,omitempty"`
	DefaultThinking             bool                           `json:"default_thinking,omitempty"`
	Embeddings                  bool                           `json:"embeddings,omitempty"`
	Rerank                      bool                           `json:"rerank,omitempty"`
	MoE                         bool                           `json:"moe,omitempty"`
	SequenceMixer               bool                           `json:"sequence_mixer,omitempty"`
	AttachedOnly                bool                           `json:"attached_only,omitempty"`
	Capabilities                []inference.CapabilityID       `json:"capabilities,omitempty"`
	Labels                      map[string]string              `json:"labels,omitempty"`
}

func (features ROCmEngineFeatures) clone() ROCmEngineFeatures {
	features.Capabilities = append([]inference.CapabilityID(nil), features.Capabilities...)
	features.Labels = cloneStringMap(features.Labels)
	return features
}

func (features ROCmEngineFeatures) empty() bool {
	return features.Contract == "" &&
		features.Architecture == "" &&
		features.Family == "" &&
		features.RuntimeStatus == "" &&
		features.ReasoningParserID == "" &&
		features.ToolParserID == "" &&
		features.ChatTemplateID == "" &&
		!features.NativeRuntime &&
		!features.DirectGreedyToken &&
		!features.NativeMLPMatVec &&
		!features.NativeLinearMatVec &&
		!features.NativeQ6BitstreamMatVec &&
		!features.NativeAttentionOMatVec &&
		!features.NativeFixedSlidingAttention &&
		!features.GenerationStream &&
		!features.AsyncDecodePrefetch &&
		!features.ModelContextWindow &&
		!features.TextGenerate &&
		!features.DeviceKVState &&
		!features.FixedSlidingCache &&
		!features.FixedSlidingCacheBound &&
		!features.CompiledLayerDecode &&
		!features.PipelinedDecode &&
		!features.ReasoningParse &&
		!features.ToolParse &&
		!features.ChatTemplate &&
		!features.DefaultThinking &&
		!features.Embeddings &&
		!features.Rerank &&
		!features.MoE &&
		!features.SequenceMixer &&
		!features.AttachedOnly &&
		len(features.Capabilities) == 0 &&
		len(features.Labels) == 0
}

func (features ROCmEngineFeatures) EnabledCapabilities() []inference.CapabilityID {
	return append([]inference.CapabilityID(nil), features.Capabilities...)
}

// ROCmEngineFeaturesReporter is implemented by loaded ROCm models that declare
// the runtime feature set they want enabled. This mirrors go-mlx's
// EngineFeaturesModel shape while keeping the ROCm feature surface typed here.
type ROCmEngineFeaturesReporter interface {
	ROCmEngineFeatures() ROCmEngineFeatures
}

// ROCmEngineFeaturesFor returns the engine features declared by a loaded model
// or by its resolved model profile. It is the runtime-facing equivalent of the
// registry metadata helpers below: callers can dispatch on this capability
// instead of concrete model families.
func ROCmEngineFeaturesFor(model any) (ROCmEngineFeatures, bool) {
	if model == nil {
		return ROCmEngineFeatures{}, false
	}
	if reporter, ok := model.(ROCmEngineFeaturesReporter); ok {
		features := reporter.ROCmEngineFeatures()
		if !features.empty() {
			return features.clone(), true
		}
	}
	if reporter, ok := model.(ROCmModelProfileReporter); ok {
		profile := reporter.ModelProfile()
		if profile.Matched() {
			features := profile.EngineFeatures
			if features.empty() {
				features = ROCmEngineFeaturesForProfile(profile)
			}
			if !features.empty() {
				return features.clone(), true
			}
		}
	}
	if textModel, ok := model.(inference.TextModel); ok {
		return ROCmEngineFeaturesForModel(textModel)
	}
	return ROCmEngineFeatures{}, false
}

func ROCmEngineFeaturesForIdentity(path string, model inference.ModelIdentity) (ROCmEngineFeatures, bool) {
	profile, ok := ResolveROCmModelProfile(path, model)
	if !ok {
		return ROCmEngineFeatures{}, false
	}
	return profile.EngineFeatures.clone(), true
}

func ROCmEngineFeaturesForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmEngineFeatures, bool) {
	profile, ok := ResolveROCmModelProfileForInfo(path, info, labels)
	if !ok {
		return ROCmEngineFeatures{}, false
	}
	return profile.EngineFeatures.clone(), true
}

func ROCmEngineFeaturesForModel(model inference.TextModel) (ROCmEngineFeatures, bool) {
	profile, ok := ResolveROCmModelProfileForModel(model)
	if !ok {
		return ROCmEngineFeatures{}, false
	}
	features := profile.EngineFeatures
	if features.empty() {
		features = ROCmEngineFeaturesForProfile(profile)
	}
	return features.clone(), true
}

func ROCmEngineFeaturesForProfile(profile ROCmModelProfile) ROCmEngineFeatures {
	architectureProfile := profile.ArchitectureProfile
	if architectureProfile.ID == "" {
		architectureProfile = profile.Gemma4Settings
	}
	if architectureProfile.ID == "" {
		if resolved, ok := ROCmArchitectureProfileForArchitecture(profile.Architecture); ok {
			architectureProfile = resolved
		}
	}
	features := ROCmEngineFeatures{
		Contract:          rocmEngineFeaturesContract,
		Architecture:      firstNonEmptyString(profile.Architecture, architectureProfile.ID),
		Family:            firstNonEmptyString(profile.Family, architectureProfile.Family, architectureProfile.ID),
		RuntimeStatus:     architectureProfile.RuntimeStatus,
		ReasoningParserID: architectureProfile.ParserID,
		ToolParserID:      architectureProfile.ToolParserID,
		NativeRuntime:     architectureProfile.NativeRuntime,
		DefaultThinking:   architectureProfile.DefaultThinking,
		Embeddings:        architectureProfile.Embeddings,
		Rerank:            architectureProfile.Rerank,
		MoE:               architectureProfile.MoE,
		SequenceMixer:     rocmProfileDeclaresSequenceMixer(profile),
		AttachedOnly:      architectureProfile.AttachedOnly,
	}
	if architectureProfile.ID != "" {
		features.Architecture = architectureProfile.ID
	}
	features.ReasoningParse = features.ReasoningParserID != ""
	features.ToolParse = features.ToolParserID != ""
	if templateID, ok := ROCmChatTemplateID(firstNonEmptyString(architectureProfile.ID, profile.Architecture)); ok {
		features.ChatTemplate = true
		features.ChatTemplateID = templateID
	}
	if profile.Family == "gemma4" {
		features.DirectGreedyToken = profile.Gemma4EngineFeatures.DirectGreedyToken
		features.NativeMLPMatVec = profile.Gemma4EngineFeatures.NativeMLPMatVec
		features.NativeLinearMatVec = profile.Gemma4EngineFeatures.NativeLinearMatVec
		features.NativeQ6BitstreamMatVec = profile.Gemma4EngineFeatures.NativeQ6BitstreamMatVec
		features.NativeAttentionOMatVec = profile.Gemma4EngineFeatures.NativeAttentionOMatVec
		features.NativeFixedSlidingAttention = profile.Gemma4EngineFeatures.NativeFixedSlidingAttention
		features.GenerationStream = profile.Gemma4EngineFeatures.GenerationStream
		features.AsyncDecodePrefetch = profile.Gemma4EngineFeatures.AsyncDecodePrefetch
		features.ModelContextWindow = profile.Gemma4EngineFeatures.ModelContextWindow
		features.TextGenerate = profile.Gemma4EngineFeatures.TextGenerate
		features.DeviceKVState = profile.Gemma4EngineFeatures.DeviceKVState
		features.FixedSlidingCache = profile.Gemma4EngineFeatures.FixedSlidingCache
		features.FixedSlidingCacheBound = profile.Gemma4EngineFeatures.FixedSlidingCacheBound
		features.CompiledLayerDecode = profile.Gemma4EngineFeatures.CompiledLayerDecode
		features.PipelinedDecode = profile.Gemma4EngineFeatures.PipelinedDecode
	}
	if profile.Family != "gemma4" && !features.ModelContextWindow {
		features.ModelContextWindow = architectureProfile.Generation && !architectureProfile.AttachedOnly
	}
	if profile.Family != "gemma4" && !features.TextGenerate {
		features.TextGenerate = architectureProfile.Generation && architectureProfile.NativeRuntime && !architectureProfile.AttachedOnly
	}
	features.Capabilities = rocmEngineFeatureCapabilities(features)
	if registered, ok := rocmmodel.RegisteredFeatureRouteForArchitecture(features.Architecture); ok {
		features = rocmEngineFeaturesWithRegisteredFeatureRoute(features, rocmModelFeatureRouteFromModel(registered))
		features.Capabilities = mergeROCmCapabilityIDs(rocmEngineFeatureCapabilities(features), features.Capabilities)
	}
	features.Labels = rocmEngineFeatureLabels(features)
	return features
}

func rocmEngineFeaturesWithRegisteredFeatureRoute(features ROCmEngineFeatures, route ROCmModelFeatureRoute) ROCmEngineFeatures {
	if !route.Matched() {
		return features
	}
	if route.Architecture != "" {
		features.Architecture = route.Architecture
	}
	if route.Family != "" {
		features.Family = route.Family
	}
	if route.RuntimeStatus != "" {
		features.RuntimeStatus = route.RuntimeStatus
	}
	if route.ReasoningParserID != "" {
		features.ReasoningParserID = route.ReasoningParserID
	}
	if route.ToolParserID != "" {
		features.ToolParserID = route.ToolParserID
	}
	if route.ChatTemplateID != "" {
		features.ChatTemplateID = route.ChatTemplateID
	}
	features.NativeRuntime = features.NativeRuntime || route.NativeRuntime
	features.ModelContextWindow = features.ModelContextWindow || route.ModelContextWindow
	features.TextGenerate = features.TextGenerate || route.TextGenerate
	features.ReasoningParse = features.ReasoningParse || route.ReasoningParse || route.ReasoningParserID != ""
	features.ToolParse = features.ToolParse || route.ToolParse || route.ToolParserID != ""
	features.ChatTemplate = features.ChatTemplate || route.ChatTemplate || route.ChatTemplateID != ""
	features.DefaultThinking = features.DefaultThinking || route.DefaultThinking
	features.Embeddings = features.Embeddings || route.Embeddings
	features.Rerank = features.Rerank || route.Rerank
	features.MoE = features.MoE || route.MoE
	features.SequenceMixer = features.SequenceMixer || route.SequenceMixer
	features.AttachedOnly = features.AttachedOnly || route.AttachedOnly
	features.Capabilities = mergeROCmCapabilityIDs(features.Capabilities, route.Capabilities)
	return features
}

func rocmEngineFeatureCapabilities(features ROCmEngineFeatures) []inference.CapabilityID {
	capabilities := make([]inference.CapabilityID, 0, 6)
	add := func(id inference.CapabilityID, enabled bool) {
		if enabled {
			capabilities = append(capabilities, id)
		}
	}
	add(inference.CapabilityGenerate, features.TextGenerate)
	add(inference.CapabilityChatTemplate, features.ChatTemplate)
	add(inference.CapabilityEmbeddings, features.Embeddings)
	add(inference.CapabilityRerank, features.Rerank)
	add(inference.CapabilityReasoningParse, features.ReasoningParse)
	add(inference.CapabilityToolParse, features.ToolParse)
	return capabilities
}

func rocmEngineFeatureLabels(features ROCmEngineFeatures) map[string]string {
	labels := map[string]string{
		"engine_features_contract":                      firstNonEmptyString(features.Contract, rocmEngineFeaturesContract),
		"engine_feature_native_runtime":                 strconv.FormatBool(features.NativeRuntime),
		"engine_feature_direct_greedy_token":            strconv.FormatBool(features.DirectGreedyToken),
		"engine_feature_native_mlp_matvec":              strconv.FormatBool(features.NativeMLPMatVec),
		"engine_feature_native_linear_matvec":           strconv.FormatBool(features.NativeLinearMatVec),
		"engine_feature_native_q6_bitstream_matvec":     strconv.FormatBool(features.NativeQ6BitstreamMatVec),
		"engine_feature_native_attention_o_matvec":      strconv.FormatBool(features.NativeAttentionOMatVec),
		"engine_feature_native_fixed_sliding_attention": strconv.FormatBool(features.NativeFixedSlidingAttention),
		"engine_feature_generation_stream":              strconv.FormatBool(features.GenerationStream),
		"engine_feature_async_decode_prefetch":          strconv.FormatBool(features.AsyncDecodePrefetch),
		"engine_feature_model_context_window":           strconv.FormatBool(features.ModelContextWindow),
		"engine_feature_text_generate":                  strconv.FormatBool(features.TextGenerate),
		"engine_feature_device_kv_state":                strconv.FormatBool(features.DeviceKVState),
		"engine_feature_fixed_sliding_cache":            strconv.FormatBool(features.FixedSlidingCache),
		"engine_feature_fixed_sliding_cache_bound":      strconv.FormatBool(features.FixedSlidingCacheBound),
		"engine_feature_compiled_layer_decode":          strconv.FormatBool(features.CompiledLayerDecode),
		"engine_feature_pipelined_decode":               strconv.FormatBool(features.PipelinedDecode),
		"engine_feature_reasoning_parse":                strconv.FormatBool(features.ReasoningParse),
		"engine_feature_tool_parse":                     strconv.FormatBool(features.ToolParse),
		"engine_feature_chat_template":                  strconv.FormatBool(features.ChatTemplate),
		"engine_feature_default_thinking":               strconv.FormatBool(features.DefaultThinking),
		"engine_feature_embeddings":                     strconv.FormatBool(features.Embeddings),
		"engine_feature_rerank":                         strconv.FormatBool(features.Rerank),
		"engine_feature_moe":                            strconv.FormatBool(features.MoE),
		"engine_feature_sequence_mixer":                 strconv.FormatBool(features.SequenceMixer),
		"engine_feature_attached_only":                  strconv.FormatBool(features.AttachedOnly),
	}
	if features.Architecture != "" {
		labels["engine_feature_architecture"] = features.Architecture
	}
	if features.Family != "" {
		labels["engine_feature_family"] = features.Family
	}
	if features.RuntimeStatus != "" {
		labels["engine_feature_runtime_status"] = string(features.RuntimeStatus)
	}
	if features.ReasoningParserID != "" {
		labels["engine_feature_reasoning_parser"] = features.ReasoningParserID
	}
	if features.ToolParserID != "" {
		labels["engine_feature_tool_parser"] = features.ToolParserID
	}
	if features.ChatTemplateID != "" {
		labels["engine_feature_chat_template_id"] = features.ChatTemplateID
	}
	if len(features.Capabilities) > 0 {
		labels["engine_feature_capabilities"] = rocmCapabilityIDsCSV(features.Capabilities)
	}
	return labels
}

func rocmProfileDeclaresSequenceMixer(profile ROCmModelProfile) bool {
	for _, labels := range []map[string]string{profile.Model.Labels, profile.Labels} {
		if labels == nil {
			continue
		}
		if labels["sequence_mixer_load_plan_status"] == "valid" ||
			labels["sequence_mixer_config_plan_status"] == "valid" ||
			labels["sequence_mixer_load_plan_candidate"] == "true" ||
			strings.TrimSpace(labels["sequence_mixer_declared_kinds"]) != "" ||
			strings.TrimSpace(labels["attention_layer_types"]) != "" {
			return true
		}
	}
	architecture := firstNonEmptyString(profile.Architecture, profile.ArchitectureProfile.ID, profile.Gemma4Settings.ID)
	switch architecture {
	case "composed", "hybrid":
		return true
	}
	family := firstNonEmptyString(profile.Family, profile.ArchitectureProfile.Family, profile.Gemma4Settings.Family)
	return family == "composed" || family == "hybrid"
}

func rocmApplyROCmEngineFeatureLabels(labels map[string]string, features ROCmEngineFeatures) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if features.empty() {
		return labels
	}
	for key, value := range rocmEngineFeatureLabels(features) {
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}

func rocmCapabilityIDsCSV(ids []inference.CapabilityID) string {
	parts := make([]string, 0, len(ids))
	for _, id := range ids {
		if id != "" {
			parts = append(parts, string(id))
		}
	}
	return strings.Join(parts, ",")
}
