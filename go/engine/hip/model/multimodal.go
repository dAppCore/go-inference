// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"strconv"
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/registry"
	"dappco.re/go/inference/engine/hip/profile"
)

const (
	MultimodalProcessorRegistryContract = "rocm-multimodal-processor-registry-v1"

	MultimodalProcessorRouteName       = "multimodal-processor-route"
	MultimodalProcessorRuntimeHIP      = "hip"
	MultimodalProcessorRuntimeMetadata = "metadata"
	KernelStatusLinked                 = "linked"
	KernelStatusNotLinked              = "not_linked"
)

type MultimodalProcessorRouteStatus string

const (
	MultimodalProcessorExperimentalNative MultimodalProcessorRouteStatus = "experimental_native"
	MultimodalProcessorPlannedMetadata    MultimodalProcessorRouteStatus = "planned_metadata"
)

// MultimodalProcessorRoute is the folder-owned image/audio processor route.
// It keeps model-declared vision/audio metadata discoverable without importing
// the root rocm package or binding to concrete HIP implementations.
type MultimodalProcessorRoute struct {
	Contract                 string                         `json:"contract,omitempty"`
	Name                     string                         `json:"name,omitempty"`
	Architecture             string                         `json:"architecture,omitempty"`
	Family                   string                         `json:"family,omitempty"`
	Runtime                  string                         `json:"runtime,omitempty"`
	RuntimeStatus            inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	Status                   MultimodalProcessorRouteStatus `json:"status,omitempty"`
	Reference                string                         `json:"reference,omitempty"`
	VisionReference          string                         `json:"vision_reference,omitempty"`
	AudioReference           string                         `json:"audio_reference,omitempty"`
	VisionRuntime            string                         `json:"vision_runtime,omitempty"`
	VisionProjectorRuntime   string                         `json:"vision_projector_runtime,omitempty"`
	AudioRuntime             string                         `json:"audio_runtime,omitempty"`
	AudioProjectorRuntime    string                         `json:"audio_projector_runtime,omitempty"`
	AudioFrontEndRuntime     string                         `json:"audio_front_end_runtime,omitempty"`
	Registered               bool                           `json:"registered,omitempty"`
	NativeRuntime            bool                           `json:"native_runtime,omitempty"`
	Multimodal               bool                           `json:"multimodal,omitempty"`
	Vision                   bool                           `json:"vision,omitempty"`
	Audio                    bool                           `json:"audio,omitempty"`
	Video                    bool                           `json:"video,omitempty"`
	Projector                bool                           `json:"projector,omitempty"`
	VisionTower              bool                           `json:"vision_tower,omitempty"`
	AudioTower               bool                           `json:"audio_tower,omitempty"`
	ImageProcessor           bool                           `json:"image_processor,omitempty"`
	AudioProcessor           bool                           `json:"audio_processor,omitempty"`
	Staged                   bool                           `json:"staged,omitempty"`
	Planned                  bool                           `json:"planned,omitempty"`
	ImageTokenID             int                            `json:"image_token_id,omitempty"`
	ImageTokenIndex          int                            `json:"image_token_index,omitempty"`
	VideoTokenID             int                            `json:"video_token_id,omitempty"`
	VideoTokenIndex          int                            `json:"video_token_index,omitempty"`
	AudioTokenID             int                            `json:"audio_token_id,omitempty"`
	AudioTokenIndex          int                            `json:"audio_token_index,omitempty"`
	BOITokenID               int                            `json:"boi_token_id,omitempty"`
	BOITokenIndex            int                            `json:"boi_token_index,omitempty"`
	EOITokenID               int                            `json:"eoi_token_id,omitempty"`
	EOITokenIndex            int                            `json:"eoi_token_index,omitempty"`
	BOATokenID               int                            `json:"boa_token_id,omitempty"`
	BOATokenIndex            int                            `json:"boa_token_index,omitempty"`
	EOATokenID               int                            `json:"eoa_token_id,omitempty"`
	EOATokenIndex            int                            `json:"eoa_token_index,omitempty"`
	SoftTokensPerImage       int                            `json:"soft_tokens_per_image,omitempty"`
	MMTokensPerImage         int                            `json:"mm_tokens_per_image,omitempty"`
	AudioSamplesPerToken     int                            `json:"audio_samples_per_token,omitempty"`
	VisionModelType          string                         `json:"vision_model_type,omitempty"`
	VisionDType              string                         `json:"vision_dtype,omitempty"`
	VisionImageSize          int                            `json:"vision_image_size,omitempty"`
	VisionPatchSize          int                            `json:"vision_patch_size,omitempty"`
	VisionHiddenSize         int                            `json:"vision_hidden_size,omitempty"`
	VisionIntermediateSize   int                            `json:"vision_intermediate_size,omitempty"`
	VisionLayers             int                            `json:"vision_layers,omitempty"`
	VisionHeads              int                            `json:"vision_heads,omitempty"`
	VisionKVHeads            int                            `json:"vision_kv_heads,omitempty"`
	VisionHeadDim            int                            `json:"vision_head_dim,omitempty"`
	VisionGlobalHeadDim      int                            `json:"vision_global_head_dim,omitempty"`
	VisionPoolingKernelSize  int                            `json:"vision_pooling_kernel_size,omitempty"`
	VisionPositionEmbeddings int                            `json:"vision_position_embedding_size,omitempty"`
	AudioModelType           string                         `json:"audio_model_type,omitempty"`
	AudioHiddenSize          int                            `json:"audio_hidden_size,omitempty"`
	AudioEmbedDim            int                            `json:"audio_embed_dim,omitempty"`
	AudioLayers              int                            `json:"audio_layers,omitempty"`
	AudioHeads               int                            `json:"audio_heads,omitempty"`
	AudioAttentionChunkSize  int                            `json:"audio_attention_chunk_size,omitempty"`
	AudioContextLeft         int                            `json:"audio_attention_context_left,omitempty"`
	AudioContextRight        int                            `json:"audio_attention_context_right,omitempty"`
	AudioConvKernelSize      int                            `json:"audio_conv_kernel_size,omitempty"`
	AudioOutputProjDims      int                            `json:"audio_output_proj_dims,omitempty"`
	RequiredFiles            []string                       `json:"required_files,omitempty"`
	OptionalFiles            []string                       `json:"optional_files,omitempty"`
	Labels                   map[string]string              `json:"labels,omitempty"`
}

func (route MultimodalProcessorRoute) Matched() bool {
	return route.Contract != "" && route.Name != "" && route.Architecture != "" && route.Multimodal
}

func (route MultimodalProcessorRoute) Clone() MultimodalProcessorRoute {
	route.RequiredFiles = append([]string(nil), route.RequiredFiles...)
	route.OptionalFiles = append([]string(nil), route.OptionalFiles...)
	route.Labels = cloneStringMap(route.Labels)
	return route
}

func (route MultimodalProcessorRoute) WithLabels(labels map[string]string) MultimodalProcessorRoute {
	route = route.withLabels(labels)
	route.finalize()
	return route.Clone()
}

var registeredMultimodalProcessors = registry.NewOrdered[string, MultimodalProcessorRoute]()

// RegisterMultimodalProcessorRoute registers or replaces processor metadata by
// architecture.
func RegisterMultimodalProcessorRoute(route MultimodalProcessorRoute) {
	route = NormalizeMultimodalProcessorRoute(route)
	if !route.Matched() {
		return
	}
	registeredMultimodalProcessors.Put(route.Architecture, route)
}

func RegisteredMultimodalProcessorArchitectures() []string {
	return registeredMultimodalProcessors.Keys()
}

func RegisteredMultimodalProcessorRoutes() []MultimodalProcessorRoute {
	return registeredMultimodalProcessorSnapshot()
}

func ReplaceRegisteredMultimodalProcessorRoutes(routes []MultimodalProcessorRoute) {
	order := make([]string, 0, len(routes))
	values := make(map[string]MultimodalProcessorRoute, len(routes))
	for _, route := range routes {
		route = NormalizeMultimodalProcessorRoute(route)
		if !route.Matched() {
			continue
		}
		if _, ok := values[route.Architecture]; !ok {
			order = append(order, route.Architecture)
		}
		values[route.Architecture] = route
	}
	registeredMultimodalProcessors.Restore(order, values)
}

func RegisteredMultimodalProcessorRouteForArchitecture(architecture string) (MultimodalProcessorRoute, bool) {
	return registeredMultimodalProcessorForArchitecture(architecture)
}

func MultimodalProcessorRouteForArchitecture(architecture string) (MultimodalProcessorRoute, bool) {
	architecture = profile.ArchitectureID(architecture)
	if architecture == "" {
		return MultimodalProcessorRoute{}, false
	}
	if route, ok := registeredMultimodalProcessorForArchitecture(architecture); ok {
		return route, true
	}
	architectureProfile, ok := profile.LookupArchitectureProfile(architecture)
	if !ok {
		return MultimodalProcessorRoute{}, false
	}
	route := staticMultimodalProcessorRoute(architectureProfile.ID, firstNonEmpty(architectureProfile.Family, architectureProfile.ID))
	if !route.Matched() {
		return MultimodalProcessorRoute{}, false
	}
	return route, true
}

func MultimodalProcessorRouteForIdentity(path string, identity inference.ModelIdentity) (MultimodalProcessorRoute, bool) {
	if identity.Path == "" {
		identity.Path = path
	}
	architecture := firstNonEmpty(
		identity.Labels["engine_architecture_resolved"],
		identity.Labels["architecture_resolved"],
		identity.Architecture,
	)
	route, ok := MultimodalProcessorRouteForArchitecture(architecture)
	if ok {
		return route.WithLabels(identity.Labels), true
	}
	route = staticMultimodalProcessorRoute(multimodalProcessorArchitecture(architecture, identity.Labels), "")
	route = route.WithLabels(identity.Labels)
	if !route.Matched() {
		return MultimodalProcessorRoute{}, false
	}
	return route, true
}

func MultimodalProcessorRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (MultimodalProcessorRoute, bool) {
	return MultimodalProcessorRouteForIdentity(path, inference.ModelIdentity{
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

func MultimodalProcessorRouteForInspection(inspection *inference.ModelPackInspection) (MultimodalProcessorRoute, bool) {
	if inspection == nil {
		return MultimodalProcessorRoute{}, false
	}
	identity := inspection.Model
	if identity.Path == "" {
		identity.Path = inspection.Path
	}
	labels := mergeMultimodalLabels(identity.Labels, inspection.Labels)
	identity.Labels = labels
	return MultimodalProcessorRouteForIdentity(identity.Path, identity)
}

func DefaultMultimodalProcessorRoutes() []MultimodalProcessorRoute {
	architectures := []string{"gemma3", "gemma4", "gemma4_unified"}
	routes := make([]MultimodalProcessorRoute, 0, len(architectures)+len(registeredMultimodalProcessors.Keys()))
	seen := map[string]int{}
	for _, architecture := range architectures {
		route, ok := MultimodalProcessorRouteForArchitecture(architecture)
		if !ok {
			continue
		}
		seen[route.Architecture] = len(routes)
		routes = append(routes, route)
	}
	for _, route := range registeredMultimodalProcessorSnapshot() {
		if !route.Matched() {
			continue
		}
		if index, ok := seen[route.Architecture]; ok {
			routes[index] = route.Clone()
			continue
		}
		seen[route.Architecture] = len(routes)
		routes = append(routes, route.Clone())
	}
	return cloneMultimodalProcessorRoutes(routes)
}

func NormalizeMultimodalProcessorRoute(route MultimodalProcessorRoute) MultimodalProcessorRoute {
	route.Architecture = profile.ArchitectureID(route.Architecture)
	if route.Architecture == "" {
		return MultimodalProcessorRoute{}
	}
	architectureProfile, hasProfile := profile.LookupArchitectureProfile(route.Architecture)
	if route.Contract == "" {
		route.Contract = MultimodalProcessorRegistryContract
	}
	if route.Name == "" {
		route.Name = MultimodalProcessorRouteName
	}
	if route.Family == "" && hasProfile {
		route.Family = firstNonEmpty(architectureProfile.Family, architectureProfile.ID)
	}
	if route.Family == "" {
		route.Family = route.Architecture
	}
	if route.Runtime == "" {
		route.Runtime = MultimodalProcessorRuntimeMetadata
	}
	if len(route.RequiredFiles) == 0 {
		route.RequiredFiles = []string{"config.json"}
	}
	if len(route.OptionalFiles) == 0 {
		route.OptionalFiles = []string{"processor_config.json", "preprocessor_config.json", "tokenizer_config.json"}
	}
	route.Multimodal = route.Multimodal || route.Vision || route.Audio || route.Video
	route.Registered = route.Architecture != "" && route.Multimodal
	route = routeWithRuntimeDefaults(route)
	route.finalize()
	return route.Clone()
}

func registeredMultimodalProcessorForArchitecture(architecture string) (MultimodalProcessorRoute, bool) {
	route, ok := registeredMultimodalProcessors.Get(profile.ArchitectureID(architecture))
	if !ok {
		return MultimodalProcessorRoute{}, false
	}
	return route.Clone(), true
}

func registeredMultimodalProcessorSnapshot() []MultimodalProcessorRoute {
	routes := registeredMultimodalProcessors.Values()
	out := make([]MultimodalProcessorRoute, 0, len(routes))
	for _, route := range routes {
		out = append(out, route.Clone())
	}
	return out
}

func staticMultimodalProcessorRoute(architecture, family string) MultimodalProcessorRoute {
	architecture = profile.ArchitectureID(architecture)
	route := MultimodalProcessorRoute{
		Contract:      MultimodalProcessorRegistryContract,
		Name:          MultimodalProcessorRouteName,
		Architecture:  architecture,
		Family:        family,
		Runtime:       MultimodalProcessorRuntimeMetadata,
		RuntimeStatus: inference.FeatureRuntimeMetadataOnly,
		RequiredFiles: []string{"config.json"},
		OptionalFiles: []string{"processor_config.json", "preprocessor_config.json", "tokenizer_config.json"},
	}
	switch architecture {
	case "gemma3":
		route.Multimodal = true
		route.Vision = true
		route.VisionReference = "go_mlx_gemma3_multimodal_wrapper"
		route.Reference = route.VisionReference
	case "gemma4":
		route.Multimodal = true
		route.Vision = true
		route.Video = true
		route.VisionReference = "go_mlx_gemma4_vision"
		route.Reference = route.VisionReference
	case "gemma4_unified":
		route.Multimodal = true
		route.Audio = true
		route.AudioReference = "go_mlx_gemma4_audio"
		route.Reference = route.AudioReference
	default:
		route.Architecture = firstNonEmpty(architecture, route.Architecture)
	}
	if route.Family == "" {
		if architectureProfile, ok := profile.LookupArchitectureProfile(route.Architecture); ok {
			route.Family = firstNonEmpty(architectureProfile.Family, architectureProfile.ID)
		}
	}
	if route.Vision {
		route.VisionRuntime = KernelStatusNotLinked
		route.VisionProjectorRuntime = KernelStatusNotLinked
	}
	if route.Audio {
		route.AudioRuntime = KernelStatusNotLinked
		route.AudioProjectorRuntime = KernelStatusNotLinked
		route.AudioFrontEndRuntime = KernelStatusNotLinked
	}
	route.finalize()
	return route.Clone()
}

func (route MultimodalProcessorRoute) withLabels(labels map[string]string) MultimodalProcessorRoute {
	if len(labels) == 0 {
		return route
	}
	if labels["multimodal_model"] == "true" || labels["gemma4_multimodal"] == "true" || labels["gemma3_multimodal"] == "true" {
		route.Multimodal = true
	}
	if reference := firstNonEmpty(labels["vision_reference"], labels["audio_reference"], route.Reference); reference != "" {
		route.Reference = reference
	}
	route.VisionReference = firstNonEmpty(labels["vision_reference"], route.VisionReference)
	route.AudioReference = firstNonEmpty(labels["audio_reference"], route.AudioReference)
	route.VisionRuntime = firstNonEmpty(labels["vision_runtime"], route.VisionRuntime)
	route.VisionProjectorRuntime = firstNonEmpty(labels["vision_projector_runtime"], route.VisionProjectorRuntime)
	route.AudioRuntime = firstNonEmpty(labels["audio_runtime"], route.AudioRuntime)
	route.AudioProjectorRuntime = firstNonEmpty(labels["audio_projector_runtime"], route.AudioProjectorRuntime)
	route.AudioFrontEndRuntime = firstNonEmpty(labels["audio_frontend_runtime"], labels["audio_front_end_runtime"], route.AudioFrontEndRuntime)

	route.ImageTokenID = firstPositiveInt(labelInt(labels["image_token_id"]), route.ImageTokenID)
	route.ImageTokenIndex = firstPositiveInt(labelInt(labels["image_token_index"]), route.ImageTokenIndex)
	route.VideoTokenID = firstPositiveInt(labelInt(labels["video_token_id"]), route.VideoTokenID)
	route.VideoTokenIndex = firstPositiveInt(labelInt(labels["video_token_index"]), route.VideoTokenIndex)
	route.AudioTokenID = firstPositiveInt(labelInt(labels["audio_token_id"]), route.AudioTokenID)
	route.AudioTokenIndex = firstPositiveInt(labelInt(labels["audio_token_index"]), route.AudioTokenIndex)
	route.BOITokenID = firstPositiveInt(labelInt(labels["boi_token_id"]), route.BOITokenID)
	route.BOITokenIndex = firstPositiveInt(labelInt(labels["boi_token_index"]), route.BOITokenIndex)
	route.EOITokenID = firstPositiveInt(labelInt(labels["eoi_token_id"]), route.EOITokenID)
	route.EOITokenIndex = firstPositiveInt(labelInt(labels["eoi_token_index"]), route.EOITokenIndex)
	route.BOATokenID = firstPositiveInt(labelInt(labels["boa_token_id"]), route.BOATokenID)
	route.BOATokenIndex = firstPositiveInt(labelInt(labels["boa_token_index"]), route.BOATokenIndex)
	route.EOATokenID = firstPositiveInt(labelInt(labels["eoa_token_id"]), route.EOATokenID)
	route.EOATokenIndex = firstPositiveInt(labelInt(labels["eoa_token_index"]), route.EOATokenIndex)
	route.SoftTokensPerImage = firstPositiveInt(labelInt(labels["vision_soft_tokens_per_image"]), route.SoftTokensPerImage)
	route.MMTokensPerImage = firstPositiveInt(labelInt(labels["mm_tokens_per_image"]), route.MMTokensPerImage)
	route.AudioSamplesPerToken = firstPositiveInt(labelInt(labels["audio_samples_per_token"]), route.AudioSamplesPerToken)

	route.VisionModelType = firstNonEmpty(labels["vision_model_type"], route.VisionModelType)
	route.VisionDType = firstNonEmpty(labels["vision_dtype"], route.VisionDType)
	route.VisionImageSize = firstPositiveInt(labelInt(labels["vision_image_size"]), route.VisionImageSize)
	route.VisionPatchSize = firstPositiveInt(labelInt(labels["vision_patch_size"]), route.VisionPatchSize)
	route.VisionHiddenSize = firstPositiveInt(labelInt(labels["vision_hidden_size"]), route.VisionHiddenSize)
	route.VisionIntermediateSize = firstPositiveInt(labelInt(labels["vision_intermediate_size"]), route.VisionIntermediateSize)
	route.VisionLayers = firstPositiveInt(labelInt(labels["vision_num_hidden_layers"]), route.VisionLayers)
	route.VisionHeads = firstPositiveInt(labelInt(labels["vision_attention_heads"]), route.VisionHeads)
	route.VisionKVHeads = firstPositiveInt(labelInt(labels["vision_kv_heads"]), route.VisionKVHeads)
	route.VisionHeadDim = firstPositiveInt(labelInt(labels["vision_head_dim"]), route.VisionHeadDim)
	route.VisionGlobalHeadDim = firstPositiveInt(labelInt(labels["vision_global_head_dim"]), route.VisionGlobalHeadDim)
	route.VisionPoolingKernelSize = firstPositiveInt(labelInt(labels["vision_pooling_kernel_size"]), route.VisionPoolingKernelSize)
	route.VisionPositionEmbeddings = firstPositiveInt(labelInt(labels["vision_position_embedding_size"]), route.VisionPositionEmbeddings)

	route.AudioModelType = firstNonEmpty(labels["audio_model_type"], route.AudioModelType)
	route.AudioHiddenSize = firstPositiveInt(labelInt(labels["audio_hidden_size"]), route.AudioHiddenSize)
	route.AudioEmbedDim = firstPositiveInt(labelInt(labels["audio_embed_dim"]), route.AudioEmbedDim)
	route.AudioLayers = firstPositiveInt(labelInt(labels["audio_num_hidden_layers"]), route.AudioLayers)
	route.AudioHeads = firstPositiveInt(labelInt(labels["audio_attention_heads"]), route.AudioHeads)
	route.AudioAttentionChunkSize = firstPositiveInt(labelInt(labels["audio_attention_chunk_size"]), route.AudioAttentionChunkSize)
	route.AudioContextLeft = firstPositiveInt(labelInt(labels["audio_attention_context_left"]), route.AudioContextLeft)
	route.AudioContextRight = firstPositiveInt(labelInt(labels["audio_attention_context_right"]), route.AudioContextRight)
	route.AudioConvKernelSize = firstPositiveInt(labelInt(labels["audio_conv_kernel_size"]), route.AudioConvKernelSize)
	route.AudioOutputProjDims = firstPositiveInt(labelInt(labels["audio_output_proj_dims"]), route.AudioOutputProjDims)

	if route.VisionRuntime != "" || route.VisionProjectorRuntime != "" || route.VisionModelType != "" || route.ImageTokenID > 0 || route.ImageTokenIndex > 0 || route.SoftTokensPerImage > 0 || route.MMTokensPerImage > 0 {
		route.Vision = true
	}
	if route.VideoTokenID > 0 || route.VideoTokenIndex > 0 {
		route.Video = true
	}
	if route.AudioRuntime != "" || route.AudioProjectorRuntime != "" || route.AudioFrontEndRuntime != "" || route.AudioModelType != "" || route.AudioTokenID > 0 || route.AudioTokenIndex > 0 || route.AudioSamplesPerToken > 0 {
		route.Audio = true
	}
	if route.Architecture == "" {
		route.Architecture = profile.ArchitectureID(firstNonEmpty(labels["architecture_model_type"], labels["engine_architecture_resolved"], labels["architecture_resolved"]))
	}
	return route
}

func (route *MultimodalProcessorRoute) finalize() {
	if route == nil {
		return
	}
	route.Architecture = profile.ArchitectureID(route.Architecture)
	route.Multimodal = route.Multimodal || route.Vision || route.Audio || route.Video
	route.VisionTower = route.Vision
	route.AudioTower = route.Audio
	route.ImageProcessor = route.Vision
	route.AudioProcessor = route.Audio
	route.Projector = route.Vision || route.Audio
	route.Registered = route.Architecture != "" && route.Multimodal
	route.NativeRuntime = route.Registered && multimodalProcessorModalitiesLinked(*route)
	if route.NativeRuntime {
		route.Runtime = MultimodalProcessorRuntimeHIP
		route.RuntimeStatus = inference.FeatureRuntimeExperimental
		route.Status = MultimodalProcessorExperimentalNative
		route.Staged = false
		route.Planned = false
	} else if route.Registered {
		route.Runtime = firstNonEmpty(route.Runtime, MultimodalProcessorRuntimeMetadata)
		if route.RuntimeStatus == "" {
			route.RuntimeStatus = inference.FeatureRuntimeMetadataOnly
		}
		route.Status = MultimodalProcessorPlannedMetadata
		route.Staged = true
		route.Planned = true
	}
	if route.Reference == "" {
		route.Reference = firstNonEmpty(route.VisionReference, route.AudioReference)
	}
	route.Labels = multimodalProcessorRouteLabels(*route)
}

func routeWithRuntimeDefaults(route MultimodalProcessorRoute) MultimodalProcessorRoute {
	runtime := KernelStatusNotLinked
	if route.NativeRuntime {
		runtime = KernelStatusLinked
	}
	if route.Vision {
		route.VisionRuntime = firstNonEmpty(route.VisionRuntime, runtime)
		route.VisionProjectorRuntime = firstNonEmpty(route.VisionProjectorRuntime, runtime)
	}
	if route.Audio {
		route.AudioRuntime = firstNonEmpty(route.AudioRuntime, runtime)
		route.AudioProjectorRuntime = firstNonEmpty(route.AudioProjectorRuntime, runtime)
		route.AudioFrontEndRuntime = firstNonEmpty(route.AudioFrontEndRuntime, runtime)
	}
	return route
}

func multimodalProcessorArchitecture(architecture string, labels map[string]string) string {
	if labels["multimodal_model"] == "true" {
		if architecture := profile.ArchitectureID(labels["architecture_model_type"]); multimodalStaticArchitecture(architecture) {
			return architecture
		}
	}
	if architecture := profile.ArchitectureID(architecture); architecture != "" {
		return architecture
	}
	return profile.ArchitectureID(firstNonEmpty(labels["engine_architecture_resolved"], labels["architecture_resolved"]))
}

func multimodalStaticArchitecture(architecture string) bool {
	switch profile.ArchitectureID(architecture) {
	case "gemma3", "gemma4", "gemma4_unified":
		return true
	default:
		return false
	}
}

func multimodalProcessorModalitiesLinked(route MultimodalProcessorRoute) bool {
	if route.Vision && (route.VisionRuntime != KernelStatusLinked || route.VisionProjectorRuntime != KernelStatusLinked) {
		return false
	}
	if route.Audio && (route.AudioRuntime != KernelStatusLinked || route.AudioProjectorRuntime != KernelStatusLinked || route.AudioFrontEndRuntime != KernelStatusLinked) {
		return false
	}
	return route.Vision || route.Audio || route.Video
}

func multimodalProcessorRouteLabels(route MultimodalProcessorRoute) map[string]string {
	if !route.Matched() {
		return nil
	}
	labels := map[string]string{
		"engine_multimodal_processor_route_contract":  route.Contract,
		"engine_multimodal_processor_route":           route.Name,
		"engine_multimodal_processor_runtime":         route.Runtime,
		"engine_multimodal_processor_status":          string(route.Status),
		"engine_multimodal_processor_registered":      strconv.FormatBool(route.Registered),
		"engine_multimodal_processor_native_runtime":  strconv.FormatBool(route.NativeRuntime),
		"engine_multimodal_processor_multimodal":      strconv.FormatBool(route.Multimodal),
		"engine_multimodal_processor_vision":          strconv.FormatBool(route.Vision),
		"engine_multimodal_processor_audio":           strconv.FormatBool(route.Audio),
		"engine_multimodal_processor_video":           strconv.FormatBool(route.Video),
		"engine_multimodal_processor_projector":       strconv.FormatBool(route.Projector),
		"engine_multimodal_processor_vision_tower":    strconv.FormatBool(route.VisionTower),
		"engine_multimodal_processor_audio_tower":     strconv.FormatBool(route.AudioTower),
		"engine_multimodal_processor_image_processor": strconv.FormatBool(route.ImageProcessor),
		"engine_multimodal_processor_audio_processor": strconv.FormatBool(route.AudioProcessor),
		"engine_multimodal_processor_staged":          strconv.FormatBool(route.Staged),
		"engine_multimodal_processor_planned":         strconv.FormatBool(route.Planned),
		"engine_multimodal_processor_required_files":  joinNonEmptyStrings(route.RequiredFiles, ","),
		"engine_multimodal_processor_optional_files":  joinNonEmptyStrings(route.OptionalFiles, ","),
	}
	if route.Architecture != "" {
		labels["engine_multimodal_processor_architecture"] = route.Architecture
	}
	if route.Family != "" {
		labels["engine_multimodal_processor_family"] = route.Family
	}
	if route.RuntimeStatus != "" {
		labels["engine_multimodal_processor_runtime_status"] = string(route.RuntimeStatus)
	}
	setStringLabel(labels, "engine_multimodal_processor_reference", route.Reference)
	setStringLabel(labels, "engine_multimodal_processor_vision_reference", route.VisionReference)
	setStringLabel(labels, "engine_multimodal_processor_audio_reference", route.AudioReference)
	setStringLabel(labels, "engine_multimodal_processor_vision_runtime", route.VisionRuntime)
	setStringLabel(labels, "engine_multimodal_processor_vision_projector_runtime", route.VisionProjectorRuntime)
	setStringLabel(labels, "engine_multimodal_processor_audio_runtime", route.AudioRuntime)
	setStringLabel(labels, "engine_multimodal_processor_audio_projector_runtime", route.AudioProjectorRuntime)
	setStringLabel(labels, "engine_multimodal_processor_audio_front_end_runtime", route.AudioFrontEndRuntime)
	setIntLabel(labels, "engine_multimodal_processor_image_token_id", route.ImageTokenID)
	setIntLabel(labels, "engine_multimodal_processor_image_token_index", route.ImageTokenIndex)
	setIntLabel(labels, "engine_multimodal_processor_video_token_id", route.VideoTokenID)
	setIntLabel(labels, "engine_multimodal_processor_video_token_index", route.VideoTokenIndex)
	setIntLabel(labels, "engine_multimodal_processor_audio_token_id", route.AudioTokenID)
	setIntLabel(labels, "engine_multimodal_processor_audio_token_index", route.AudioTokenIndex)
	setIntLabel(labels, "engine_multimodal_processor_boi_token_id", route.BOITokenID)
	setIntLabel(labels, "engine_multimodal_processor_boi_token_index", route.BOITokenIndex)
	setIntLabel(labels, "engine_multimodal_processor_eoi_token_id", route.EOITokenID)
	setIntLabel(labels, "engine_multimodal_processor_eoi_token_index", route.EOITokenIndex)
	setIntLabel(labels, "engine_multimodal_processor_boa_token_id", route.BOATokenID)
	setIntLabel(labels, "engine_multimodal_processor_boa_token_index", route.BOATokenIndex)
	setIntLabel(labels, "engine_multimodal_processor_eoa_token_id", route.EOATokenID)
	setIntLabel(labels, "engine_multimodal_processor_eoa_token_index", route.EOATokenIndex)
	setIntLabel(labels, "engine_multimodal_processor_soft_tokens_per_image", route.SoftTokensPerImage)
	setIntLabel(labels, "engine_multimodal_processor_mm_tokens_per_image", route.MMTokensPerImage)
	setIntLabel(labels, "engine_multimodal_processor_audio_samples_per_token", route.AudioSamplesPerToken)
	setStringLabel(labels, "engine_multimodal_processor_vision_model_type", route.VisionModelType)
	setStringLabel(labels, "engine_multimodal_processor_vision_dtype", route.VisionDType)
	setIntLabel(labels, "engine_multimodal_processor_vision_image_size", route.VisionImageSize)
	setIntLabel(labels, "engine_multimodal_processor_vision_patch_size", route.VisionPatchSize)
	setIntLabel(labels, "engine_multimodal_processor_vision_hidden_size", route.VisionHiddenSize)
	setIntLabel(labels, "engine_multimodal_processor_vision_intermediate_size", route.VisionIntermediateSize)
	setIntLabel(labels, "engine_multimodal_processor_vision_layers", route.VisionLayers)
	setIntLabel(labels, "engine_multimodal_processor_vision_heads", route.VisionHeads)
	setIntLabel(labels, "engine_multimodal_processor_vision_kv_heads", route.VisionKVHeads)
	setIntLabel(labels, "engine_multimodal_processor_vision_head_dim", route.VisionHeadDim)
	setIntLabel(labels, "engine_multimodal_processor_vision_global_head_dim", route.VisionGlobalHeadDim)
	setIntLabel(labels, "engine_multimodal_processor_vision_pooling_kernel_size", route.VisionPoolingKernelSize)
	setIntLabel(labels, "engine_multimodal_processor_vision_position_embedding_size", route.VisionPositionEmbeddings)
	setStringLabel(labels, "engine_multimodal_processor_audio_model_type", route.AudioModelType)
	setIntLabel(labels, "engine_multimodal_processor_audio_hidden_size", route.AudioHiddenSize)
	setIntLabel(labels, "engine_multimodal_processor_audio_embed_dim", route.AudioEmbedDim)
	setIntLabel(labels, "engine_multimodal_processor_audio_layers", route.AudioLayers)
	setIntLabel(labels, "engine_multimodal_processor_audio_heads", route.AudioHeads)
	setIntLabel(labels, "engine_multimodal_processor_audio_attention_chunk_size", route.AudioAttentionChunkSize)
	setIntLabel(labels, "engine_multimodal_processor_audio_attention_context_left", route.AudioContextLeft)
	setIntLabel(labels, "engine_multimodal_processor_audio_attention_context_right", route.AudioContextRight)
	setIntLabel(labels, "engine_multimodal_processor_audio_conv_kernel_size", route.AudioConvKernelSize)
	setIntLabel(labels, "engine_multimodal_processor_audio_output_proj_dims", route.AudioOutputProjDims)
	return labels
}

// MultimodalProcessorRouteLabels returns the normalized model-owned label
// contract for a multimodal processor route.
func MultimodalProcessorRouteLabels(route MultimodalProcessorRoute) map[string]string {
	route = NormalizeMultimodalProcessorRoute(route)
	return cloneStringMap(route.Labels)
}

func setStringLabel(labels map[string]string, key, value string) {
	if value != "" {
		labels[key] = value
	}
}

func setIntLabel(labels map[string]string, key string, value int) {
	if value > 0 {
		labels[key] = strconv.Itoa(value)
	}
}

func labelInt(value string) int {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 0 {
		return 0
	}
	return parsed
}

func mergeMultimodalLabels(left, right map[string]string) map[string]string {
	out := cloneStringMap(left)
	if out == nil {
		out = map[string]string{}
	}
	for key, value := range right {
		if value != "" {
			out[key] = value
		}
	}
	return out
}

func firstPositiveInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func cloneMultimodalProcessorRoutes(routes []MultimodalProcessorRoute) []MultimodalProcessorRoute {
	out := append([]MultimodalProcessorRoute(nil), routes...)
	for i := range out {
		out[i] = out[i].Clone()
	}
	return out
}
