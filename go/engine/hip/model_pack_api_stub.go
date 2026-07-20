// SPDX-Licence-Identifier: EUPL-1.2

//go:build !linux || !amd64 || rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"io/fs"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
	rocmprofile "dappco.re/go/inference/engine/hip/profile"
)

const (
	maxSafetensorsHeaderBytes = 64 << 20

	hipKernelStatusLinked    = "linked"
	hipKernelStatusNotLinked = "not_linked"
	hipKernelStatusPlanned   = "planned"

	hipMLXQ4ProjectionBits = 4

	rocmModelRegistryName = "rocm-model-registry-v1"

	Gemma4RuntimeMLXAffine    = modelgemma4.RuntimeMLXAffine
	Gemma4RuntimeBF16         = modelgemma4.RuntimeBF16
	Gemma4RuntimeGGUF         = modelgemma4.RuntimeGGUF
	Gemma4RuntimePlanned      = modelgemma4.RuntimePlanned
	Gemma4GenerateLinked      = modelgemma4.GenerateLinked
	Gemma4GenerateLoadOnly    = modelgemma4.GenerateLoadOnly
	Gemma4GeneratePlannedOnly = modelgemma4.GeneratePlannedOnly

	ProductionLaneModelID                   = modelgemma4.ProductionLaneModelID
	ProductionLaneArchivedBaselineModelID   = modelgemma4.ProductionLaneArchivedBaselineModelID
	ProductionLaneCurrentQualityModelID     = modelgemma4.ProductionLaneCurrentQualityModelID
	ProductionLaneCurrentModelID            = modelgemma4.ProductionLaneCurrentModelID
	ProductionLaneCurrentConstrainedModelID = modelgemma4.ProductionLaneCurrentConstrainedModelID
	ProductionLaneQualityQuantBits          = modelgemma4.ProductionLaneQualityQuantBits
	ProductionLaneProductDefaultQuantBits   = modelgemma4.ProductionLaneProductDefaultQuantBits
	ProductionLaneConstrainedQuantBits      = modelgemma4.ProductionLaneConstrainedQuantBits
	productionQuantizationLadderLabel       = "bf16,q8,q6,q4"
)

type nativeTensorInfo struct {
	Name       string
	Dimensions []uint64
	Type       uint32
	TypeName   string
	SourcePath string
	DataOffset int64
	Offset     uint64
	ByteSize   uint64
}

type rocmSafetensorsTensor struct {
	DType       string   `json:"dtype,omitempty"`
	Shape       []uint64 `json:"shape,omitempty"`
	DataOffsets []uint64 `json:"data_offsets,omitempty"`
}

type rocmModelPackConfigProbe struct {
	ModelType                string                         `json:"model_type"`
	Architectures            []string                       `json:"architectures"`
	DType                    string                         `json:"dtype"`
	HiddenSize               int                            `json:"hidden_size"`
	NumHiddenLayers          int                            `json:"num_hidden_layers"`
	NumLayers                int                            `json:"num_layers"`
	NumAttentionHeads        int                            `json:"num_attention_heads"`
	NumKeyValueHeads         int                            `json:"num_key_value_heads"`
	HeadDim                  int                            `json:"head_dim"`
	GlobalPartialRotary      float64                        `json:"global_partial_rotary_factor"`
	VocabSize                int                            `json:"vocab_size"`
	VocabSizePerLayer        int                            `json:"vocab_size_per_layer_input"`
	MaxPositionEmbeddings    int                            `json:"max_position_embeddings"`
	MaxSequenceLength        int                            `json:"max_sequence_length"`
	SeqLength                int                            `json:"seq_length"`
	CanvasLength             int                            `json:"canvas_length"`
	SlidingWindow            int                            `json:"sliding_window"`
	SlidingWindowPattern     int                            `json:"sliding_window_pattern"`
	NumKVSharedLayers        *int                           `json:"num_kv_shared_layers"`
	HiddenSizePerLayer       int                            `json:"hidden_size_per_layer_input"`
	RoPEParameters           map[string]rocmRoPEProbe       `json:"rope_parameters"`
	NumExperts               int                            `json:"num_experts"`
	NumExpertsPerTok         int                            `json:"num_experts_per_tok"`
	TopKExperts              int                            `json:"top_k_experts"`
	EnableMoEBlock           bool                           `json:"enable_moe_block"`
	UseDoubleWideMLP         bool                           `json:"use_double_wide_mlp"`
	MoEIntermediateSize      int                            `json:"moe_intermediate_size"`
	ExpertIntermediateSize   int                            `json:"expert_intermediate_size"`
	LayerTypes               []string                       `json:"layer_types"`
	ImageTokenID             int                            `json:"image_token_id"`
	ImageTokenIndex          int                            `json:"image_token_index"`
	VideoTokenID             int                            `json:"video_token_id"`
	BOITokenID               int                            `json:"boi_token_id"`
	BOITokenIndex            int                            `json:"boi_token_index"`
	EOITokenID               int                            `json:"eoi_token_id"`
	EOITokenIndex            int                            `json:"eoi_token_index"`
	AudioTokenID             int                            `json:"audio_token_id"`
	AudioTokenIndex          int                            `json:"audio_token_index"`
	BOATokenID               int                            `json:"boa_token_id"`
	BOATokenIndex            int                            `json:"boa_token_index"`
	EOATokenID               int                            `json:"eoa_token_id"`
	EOATokenIndex            int                            `json:"eoa_token_index"`
	VisionSoftTokensPerImage int                            `json:"vision_soft_tokens_per_image"`
	MMTokensPerImage         int                            `json:"mm_tokens_per_image"`
	TieWordEmbeddings        *bool                          `json:"tie_word_embeddings"`
	QuantizationConfig       rocmQuantizationConfigProbe    `json:"quantization_config"`
	Quantization             rocmQuantizationConfigProbe    `json:"quantization"`
	TextConfig               rocmModelPackTextConfigProbe   `json:"text_config"`
	VisionConfig             rocmModelPackVisionConfigProbe `json:"vision_config"`
	AudioConfig              rocmModelPackAudioConfigProbe  `json:"audio_config"`
}

type rocmModelPackTextConfigProbe struct {
	ModelType              string                   `json:"model_type"`
	Architectures          []string                 `json:"architectures"`
	DType                  string                   `json:"dtype"`
	HiddenSize             int                      `json:"hidden_size"`
	NumHiddenLayers        int                      `json:"num_hidden_layers"`
	NumLayers              int                      `json:"num_layers"`
	NumAttentionHeads      int                      `json:"num_attention_heads"`
	NumKeyValueHeads       int                      `json:"num_key_value_heads"`
	HeadDim                int                      `json:"head_dim"`
	GlobalPartialRotary    float64                  `json:"global_partial_rotary_factor"`
	VocabSize              int                      `json:"vocab_size"`
	VocabSizePerLayer      int                      `json:"vocab_size_per_layer_input"`
	MaxPositionEmbeddings  int                      `json:"max_position_embeddings"`
	MaxSequenceLength      int                      `json:"max_sequence_length"`
	SeqLength              int                      `json:"seq_length"`
	CanvasLength           int                      `json:"canvas_length"`
	SlidingWindow          int                      `json:"sliding_window"`
	SlidingWindowPattern   int                      `json:"sliding_window_pattern"`
	NumKVSharedLayers      *int                     `json:"num_kv_shared_layers"`
	HiddenSizePerLayer     int                      `json:"hidden_size_per_layer_input"`
	RoPEParameters         map[string]rocmRoPEProbe `json:"rope_parameters"`
	NumExperts             int                      `json:"num_experts"`
	NumExpertsPerTok       int                      `json:"num_experts_per_tok"`
	TopKExperts            int                      `json:"top_k_experts"`
	EnableMoEBlock         bool                     `json:"enable_moe_block"`
	UseDoubleWideMLP       bool                     `json:"use_double_wide_mlp"`
	MoEIntermediateSize    int                      `json:"moe_intermediate_size"`
	ExpertIntermediateSize int                      `json:"expert_intermediate_size"`
	LayerTypes             []string                 `json:"layer_types"`
	TieWordEmbeddings      *bool                    `json:"tie_word_embeddings"`
}

type rocmRoPEProbe struct {
	PartialRotaryFactor float64 `json:"partial_rotary_factor"`
	RopeTheta           float64 `json:"rope_theta"`
	RopeType            string  `json:"rope_type"`
	Factor              float64 `json:"factor"`
}

type Gemma4SizeQuantSupport = modelgemma4.SizeQuantSupport

type Gemma4QuantModeSupport = modelgemma4.QuantModeSupport

type ProductionQuantizationPackSupport = modelgemma4.ProductionQuantizationPackSupport

type rocmModelPackVisionConfigProbe struct {
	ModelType             string        `json:"model_type"`
	DType                 string        `json:"dtype"`
	ImageSize             int           `json:"image_size"`
	PatchSize             int           `json:"patch_size"`
	NumChannels           int           `json:"num_channels"`
	HiddenSize            int           `json:"hidden_size"`
	IntermediateSize      int           `json:"intermediate_size"`
	NumHiddenLayers       int           `json:"num_hidden_layers"`
	NumAttentionHeads     int           `json:"num_attention_heads"`
	NumKeyValueHeads      int           `json:"num_key_value_heads"`
	HeadDim               int           `json:"head_dim"`
	GlobalHeadDim         int           `json:"global_head_dim"`
	MaxPositionEmbeddings int           `json:"max_position_embeddings"`
	HiddenActivation      string        `json:"hidden_activation"`
	RMSNormEps            float64       `json:"rms_norm_eps"`
	LayerNormEps          float64       `json:"layer_norm_eps"`
	RoPEParameters        rocmRoPEProbe `json:"rope_parameters"`
	PoolingKernelSize     int           `json:"pooling_kernel_size"`
	PositionEmbeddingSize int           `json:"position_embedding_size"`
	DefaultOutputLength   int           `json:"default_output_length"`
	Standardize           bool          `json:"standardize"`
	UseClippedLinears     bool          `json:"use_clipped_linears"`
}

type rocmModelPackAudioConfigProbe struct {
	ModelType                   string  `json:"model_type"`
	HiddenSize                  int     `json:"hidden_size"`
	AudioEmbedDim               int     `json:"audio_embed_dim"`
	AudioSamplesPerToken        int     `json:"audio_samples_per_token"`
	NumHiddenLayers             int     `json:"num_hidden_layers"`
	NumAttentionHeads           int     `json:"num_attention_heads"`
	AttentionChunkSize          int     `json:"attention_chunk_size"`
	AttentionContextLeft        int     `json:"attention_context_left"`
	AttentionContextRight       int     `json:"attention_context_right"`
	AttentionLogitCap           float64 `json:"attention_logit_cap"`
	AttentionInvalidLogitsValue float64 `json:"attention_invalid_logits_value"`
	ConvKernelSize              int     `json:"conv_kernel_size"`
	OutputProjDims              int     `json:"output_proj_dims"`
	RMSNormEps                  float64 `json:"rms_norm_eps"`
	GradientClipping            float64 `json:"gradient_clipping"`
	ResidualWeight              float64 `json:"residual_weight"`
	HiddenAct                   string  `json:"hidden_act"`
	UseClippedLinears           bool    `json:"use_clipped_linears"`
}

type rocmQuantizationConfigProbe struct {
	Bits         int    `json:"bits"`
	GroupSize    int    `json:"group_size"`
	QuantMethod  string `json:"quant_method"`
	Algorithm    string `json:"algorithm"`
	WeightFormat string `json:"weight_format"`
	Format       string `json:"format"`
	Type         string `json:"type"`
}

// InspectModelPack validates a local model pack without loading tensors. The
// portable build keeps this metadata path live so CLI/API contracts work across
// CPU, CUDA, and legacy ROCm compile targets before native kernels are linked.
func InspectModelPack(ctx context.Context, path string) (*inference.ModelPackInspection, error) {
	return (&rocmBackend{}).InspectModelPack(ctx, path)
}

func (b *rocmBackend) Capabilities() inference.CapabilityReport {
	available := false
	if b != nil {
		available = b.Available()
	}
	return inference.CapabilityReport{
		Runtime: inference.RuntimeIdentity{
			Backend:       "rocm",
			NativeRuntime: false,
			Labels: map[string]string{
				"native_runtime":               "portable_metadata",
				"production_requires_env_gate": "false",
				"production_requires_cli_flag": "false",
			},
		},
		Available: available,
		Capabilities: []inference.Capability{
			inference.SupportedCapability(inference.CapabilityModelFit, inference.CapabilityGroupRuntime),
			inference.SupportedCapability(inference.CapabilityMemoryPlanning, inference.CapabilityGroupRuntime),
			inference.SupportedCapability(inference.CapabilityKVCachePlanning, inference.CapabilityGroupRuntime),
			inference.ExperimentalCapability(inference.CapabilityModelMerge, inference.CapabilityGroupRuntime, "dense F32 safetensors LoRA model-pack merge is linked in the portable CLI path; quantized production Gemma4 merge remains pending"),
		},
		Labels: map[string]string{
			"backend":                      "rocm",
			"native_runtime":               "portable_metadata",
			"production_requires_env_gate": "false",
			"production_requires_cli_flag": "false",
		},
	}
}

func (b *rocmBackend) InspectModelPack(ctx context.Context, path string) (*inference.ModelPackInspection, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	fileManifest, err := rocmmodel.InspectModelPackFiles(path)
	if err != nil {
		return nil, core.E("rocm.InspectModelPack", "stat model pack", err)
	}
	resolvedPath := fileManifest.SourcePath
	root := fileManifest.Root

	inspection := &inference.ModelPackInspection{
		Path: resolvedPath,
		Model: inference.ModelIdentity{
			Path: resolvedPath,
		},
		Labels: map[string]string{
			"backend":                      "rocm",
			"native_runtime":               "portable_metadata",
			"production_requires_env_gate": "false",
			"production_requires_cli_flag": "false",
		},
	}
	weights := fileManifest.WeightPaths()
	inspection.Format = fileManifest.Format
	for key, value := range fileManifest.Labels {
		if value != "" {
			inspection.Labels[key] = value
		}
	}
	inspection.Labels["format"] = inspection.Format
	inspection.Labels["weight_files"] = strconv.Itoa(len(weights))
	if len(weights) == 0 {
		inspection.Notes = append(inspection.Notes, "no GGUF or safetensors weight files found")
	}

	var cfg *rocmModelPackConfigProbe
	if readCfg, err := readROCmModelConfig(root); err != nil {
		inspection.Notes = append(inspection.Notes, "config.json could not be parsed: "+err.Error())
	} else if readCfg != nil {
		cfg = readCfg
		applyROCmPortableModelConfig(inspection, *readCfg)
	}
	if processor, err := readROCmGemma4ProcessorConfig(root); err != nil {
		inspection.Notes = append(inspection.Notes, "processor_config.json could not be parsed: "+err.Error())
	} else if processor != nil {
		applyROCmGemma4ProcessorConfigLabels(inspection, *processor)
	}

	allTensors := []nativeTensorInfo{}
	weightMetadataValid := len(weights) > 0
	for _, weight := range weights {
		switch strings.ToLower(filepath.Ext(weight)) {
		case ".safetensors":
			tensors, err := readROCmSafetensorsNativeTensors(weight)
			if err != nil {
				inspection.Notes = append(inspection.Notes, filepath.Base(weight)+" safetensors metadata could not be parsed: "+err.Error())
				weightMetadataValid = false
				continue
			}
			allTensors = append(allTensors, tensors...)
			mergeROCmPortableSafetensorsLabels(inspection.Labels, tensors)
		case ".gguf":
			inspection.Labels["gguf_weight_files"] = strconv.Itoa(rocmLabelInt(inspection.Labels["gguf_weight_files"]) + 1)
		}
	}
	inspection.Labels["weight_metadata_valid"] = strconv.FormatBool(weightMetadataValid)
	if cfg != nil {
		applyROCmPortableSequenceMixerPlan(inspection, *cfg, allTensors)
	}
	applyROCmInspectionModelProfile(inspection)
	applyROCmPortableArchitectureInspection(inspection, weightMetadataValid)
	applyROCmPortableGemma4ModelPackSupportLabels(inspection)
	appendROCmInspectionCapability(inspection, inference.SupportedCapability(inference.CapabilityModelFit, inference.CapabilityGroupRuntime))
	appendROCmInspectionCapability(inspection, inference.SupportedCapability(inference.CapabilityMemoryPlanning, inference.CapabilityGroupRuntime))
	appendROCmInspectionCapability(inspection, inference.SupportedCapability(inference.CapabilityKVCachePlanning, inference.CapabilityGroupRuntime))
	applyROCmPortableGemma4ModelPackInspectionCapabilities(inspection)
	inspection.Model.Labels = cloneStringMap(inspection.Labels)
	inspection.Notes = append(inspection.Notes, "portable ROCm model-pack metadata is available; native runtime execution is not linked in this build")
	return inspection, nil
}

func readROCmModelConfig(root string) (*rocmModelPackConfigProbe, error) {
	data, err := os.ReadFile(filepath.Join(root, "config.json"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var cfg rocmModelPackConfigProbe
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func readROCmGemma4ProcessorConfig(root string) (*modelgemma4.ProcessorConfig, error) {
	data, err := os.ReadFile(filepath.Join(root, "processor_config.json"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	cfg, err := modelgemma4.ParseProcessorConfig(data)
	if err != nil {
		return nil, err
	}
	return &cfg, nil
}

func applyROCmGemma4ProcessorConfigLabels(inspection *inference.ModelPackInspection, cfg modelgemma4.ProcessorConfig) {
	if inspection == nil || !isROCmGemma4Architecture(inspection.Model.Architecture) {
		return
	}
	labels := inspection.Labels
	labels["processor_config"] = "true"
	modelgemma4.ApplyProcessorConfigLabels(labels, cfg)
	if cfg.ImageProcessor != nil || cfg.VideoProcessor != nil {
		labels["multimodal_model"] = "true"
		labels["gemma4_multimodal"] = "true"
		labels["vision_processor_config"] = "true"
		if labels["vision_runtime"] == "" {
			labels["vision_runtime"] = hipKernelStatusNotLinked
		}
		if labels["vision_projector_runtime"] == "" {
			labels["vision_projector_runtime"] = hipKernelStatusNotLinked
		}
		if labels["vision_reference"] == "" {
			labels["vision_reference"] = "go_mlx_gemma4_vision"
		}
	}
	if cfg.FeatureExtractor != nil {
		labels["multimodal_model"] = "true"
		labels["gemma4_multimodal"] = "true"
		labels["audio_processor_config"] = "true"
		if labels["audio_runtime"] == "" {
			labels["audio_runtime"] = hipKernelStatusNotLinked
		}
		if labels["audio_projector_runtime"] == "" {
			labels["audio_projector_runtime"] = hipKernelStatusNotLinked
		}
		labels["audio_frontend_runtime"] = hipKernelStatusNotLinked
		if labels["audio_reference"] == "" {
			labels["audio_reference"] = "go_mlx_gemma4_audio"
		}
	}
}

func applyROCmPortableModelConfig(inspection *inference.ModelPackInspection, cfg rocmModelPackConfigProbe) {
	if inspection == nil {
		return
	}
	model := inspection.Model
	model.Architecture = firstNonEmptyString(model.Architecture, rocmConfigArchitecture(cfg))
	model.ContextLength = firstPositiveInt(model.ContextLength, cfg.MaxPositionEmbeddings, cfg.MaxSequenceLength, cfg.SeqLength, cfg.TextConfig.MaxPositionEmbeddings, cfg.TextConfig.MaxSequenceLength, cfg.TextConfig.SeqLength)
	model.NumLayers = firstPositiveInt(model.NumLayers, cfg.NumHiddenLayers, cfg.NumLayers, cfg.TextConfig.NumHiddenLayers, cfg.TextConfig.NumLayers)
	model.HiddenSize = firstPositiveInt(model.HiddenSize, cfg.HiddenSize, cfg.TextConfig.HiddenSize)
	model.VocabSize = firstPositiveInt(model.VocabSize, cfg.VocabSize, cfg.TextConfig.VocabSize)
	quant := cfg.QuantizationConfig
	if rocmQuantConfigEmpty(quant) {
		quant = cfg.Quantization
	}
	model.QuantBits = firstPositiveInt(model.QuantBits, quant.Bits)
	model.QuantGroup = firstPositiveInt(model.QuantGroup, quant.GroupSize)
	model.QuantType = firstNonEmptyString(model.QuantType, normalizeROCmLabelToken(firstNonEmptyString(quant.Algorithm, quant.QuantMethod, quant.WeightFormat, quant.Format, quant.Type)), rocmConfigDTypeQuantizationType(firstNonEmptyString(cfg.DType, cfg.TextConfig.DType)))
	inspection.Model = model
	rocmApplyArchitectureResolutionLabels(inspection.Labels, cfg)

	if rocmConfigTiedWordEmbeddings(cfg) {
		inspection.Labels["tied_word_embeddings"] = "true"
	}
	applyROCmPortableAttentionConfigLabels(inspection.Labels, cfg)
	applyROCmPortableMultimodalConfigLabels(inspection, cfg)
	applyROCmPortableDiffusionGemmaConfigLabels(inspection, cfg)
}

func applyROCmPortableAttentionConfigLabels(labels map[string]string, cfg rocmModelPackConfigProbe) {
	if labels == nil {
		return
	}
	maps.Copy(labels, rocmPortableAttentionConfigLabels(cfg))
	layerTypes := rocmConfigLayerTypes(cfg)
	if len(layerTypes) > 0 {
		labels["attention_layer_types"] = strings.Join(layerTypes, ",")
		labels["attention_layer_count"] = strconv.Itoa(len(layerTypes))
	}
	if planTypes, source := rocmConfigSequenceMixerPlanLayerTypes(cfg); len(planTypes) > 0 {
		labels["attention_layer_types"] = strings.Join(planTypes, ",")
		rocmApplySequenceMixerConfigLabels(labels, planTypes, source)
		return
	}
	if source, err := rocmConfigSequenceMixerPlanError(cfg); err != nil {
		rocmApplySequenceMixerConfigErrorLabels(labels, source, err)
	}
}

func applyROCmPortableSequenceMixerPlan(inspection *inference.ModelPackInspection, cfg rocmModelPackConfigProbe, tensors []nativeTensorInfo) {
	if inspection == nil {
		return
	}
	layerTypes := sequenceMixerLayerTypesFromLabels(inspection.Labels)
	if len(layerTypes) == 0 {
		return
	}
	names := make([]string, 0, len(tensors))
	for _, tensor := range tensors {
		names = append(names, tensor.Name)
	}
	plan, err := BuildSequenceMixerLoadPlan(layerTypes, names, firstPositiveInt(inspection.Model.NumLayers, cfg.NumHiddenLayers, cfg.NumLayers, cfg.TextConfig.NumHiddenLayers, cfg.TextConfig.NumLayers))
	rocmApplySequenceMixerLoadPlanLabels(inspection.Labels, plan, err)
	if err != nil {
		return
	}
	inspection.Labels["sequence_mixer_subpath_discovery"] = "safetensors"
	if len(plan.Subpaths.Ambiguous) > 0 {
		inspection.Labels["sequence_mixer_subpath_status"] = "ambiguous"
		inspection.Labels["sequence_mixer_subpath_ambiguous_layers"] = sequenceMixerAmbiguousSubpathCSV(plan.Subpaths.Ambiguous)
		return
	}
	inspection.Labels["sequence_mixer_subpath_count"] = strconv.Itoa(len(plan.Subpaths.Subpaths))
	if len(plan.Subpaths.Subpaths) == 0 {
		inspection.Labels["sequence_mixer_subpath_status"] = "bare"
		return
	}
	inspection.Labels["sequence_mixer_subpath_status"] = "ok"
	inspection.Labels["sequence_mixer_subpaths"] = sequenceMixerSubpathCSV(plan.Subpaths.Subpaths)
}

func applyROCmPortableMultimodalConfigLabels(inspection *inference.ModelPackInspection, cfg rocmModelPackConfigProbe) {
	if inspection == nil {
		return
	}
	architecture := rocmConfigArchitecture(cfg)
	labels := inspection.Labels
	imageToken := firstPositiveInt(cfg.ImageTokenID, cfg.ImageTokenIndex)
	audioToken := firstPositiveInt(cfg.AudioTokenID, cfg.AudioTokenIndex)
	softTokens := firstPositiveInt(cfg.VisionSoftTokensPerImage, cfg.MMTokensPerImage, cfg.VisionConfig.DefaultOutputLength)
	gemma4Architecture := isROCmGemma4Architecture(architecture)
	hasVision := cfg.VisionConfig.ModelType != "" ||
		cfg.VisionConfig.HiddenSize > 0 ||
		cfg.VisionConfig.NumHiddenLayers > 0 ||
		imageToken > 0 ||
		softTokens > 0
	hasAudio := cfg.AudioConfig.ModelType != "" ||
		cfg.AudioConfig.HiddenSize > 0 ||
		cfg.AudioConfig.NumHiddenLayers > 0 ||
		cfg.AudioConfig.AudioEmbedDim > 0 ||
		audioToken > 0
	if gemma4Architecture {
		hasVision = rocmGemma4ConfigHasVision(cfg)
		hasAudio = rocmGemma4ConfigHasAudio(cfg)
	}
	if !hasVision && !hasAudio {
		return
	}
	labels["multimodal_model"] = "true"
	if gemma4Architecture {
		labels["gemma4_multimodal"] = "true"
	}
	if hasVision {
		labels["vision_runtime"] = hipKernelStatusNotLinked
		labels["vision_projector_runtime"] = hipKernelStatusNotLinked
		if gemma4Architecture {
			labels["vision_reference"] = "go_mlx_gemma4_vision"
		} else {
			labels["vision_reference"] = "model_pack_multimodal_metadata"
		}
		if gemma4Architecture {
			modelgemma4.ApplyVisionConfigLabels(labels, rocmGemma4VisionConfigFromProbe(cfg))
		} else {
			if imageToken > 0 {
				labels["image_token_id"] = strconv.Itoa(imageToken)
			}
			if cfg.VideoTokenID > 0 {
				labels["video_token_id"] = strconv.Itoa(cfg.VideoTokenID)
			}
			if softTokens > 0 {
				labels["vision_soft_tokens_per_image"] = strconv.Itoa(softTokens)
			}
			if cfg.VisionConfig.ModelType != "" {
				labels["vision_model_type"] = normalizeROCmLabelToken(cfg.VisionConfig.ModelType)
			}
		}
		inspection.Notes = append(inspection.Notes, "multimodal vision metadata is recognised; native ROCm vision tower and projector kernels are pending")
	}
	if hasAudio {
		labels["audio_runtime"] = hipKernelStatusNotLinked
		labels["audio_projector_runtime"] = hipKernelStatusNotLinked
		labels["audio_frontend_runtime"] = hipKernelStatusNotLinked
		if gemma4Architecture {
			labels["audio_reference"] = "go_mlx_gemma4_audio"
		} else {
			labels["audio_reference"] = "model_pack_audio_metadata"
		}
		if gemma4Architecture {
			modelgemma4.ApplyAudioConfigLabels(labels, rocmGemma4AudioConfigFromProbe(cfg))
		} else {
			if audioToken > 0 {
				labels["audio_token_id"] = strconv.Itoa(audioToken)
			}
			if cfg.BOATokenID > 0 {
				labels["boa_token_id"] = strconv.Itoa(cfg.BOATokenID)
			}
			if cfg.BOATokenIndex > 0 {
				labels["boa_token_index"] = strconv.Itoa(cfg.BOATokenIndex)
			}
			if cfg.EOATokenID > 0 {
				labels["eoa_token_id"] = strconv.Itoa(cfg.EOATokenID)
			}
			if cfg.EOATokenIndex > 0 {
				labels["eoa_token_index"] = strconv.Itoa(cfg.EOATokenIndex)
			}
			if cfg.AudioConfig.AudioSamplesPerToken > 0 {
				labels["audio_samples_per_token"] = strconv.Itoa(cfg.AudioConfig.AudioSamplesPerToken)
			}
			if cfg.AudioConfig.ModelType != "" {
				labels["audio_model_type"] = normalizeROCmLabelToken(cfg.AudioConfig.ModelType)
			}
		}
		inspection.Notes = append(inspection.Notes, "multimodal audio metadata is recognised; native ROCm audio front-end, tower, and projector kernels are pending")
	}
}

func applyROCmPortableDiffusionGemmaConfigLabels(inspection *inference.ModelPackInspection, cfg rocmModelPackConfigProbe) {
	if inspection == nil || normalizeROCmArchitecture(rocmConfigArchitecture(cfg)) != "diffusion_gemma" {
		return
	}
	labels := inspection.Labels
	labels["block_diffusion_model"] = "true"
	labels["diffusion_runtime"] = hipKernelStatusNotLinked
	labels["diffusion_sampler_runtime"] = hipKernelStatusNotLinked
	labels["diffusion_trunk_runtime"] = "model_pack_metadata"
	labels["diffusion_reference"] = "go_mlx_diffusion_gemma"
	labels["diffusion_fallback"] = "refused"
	labels["reactive_diffusion_fallback"] = "refused"
	if canvasLength := firstPositiveInt(cfg.CanvasLength, cfg.TextConfig.CanvasLength); canvasLength > 0 {
		labels["diffusion_canvas_length"] = strconv.Itoa(canvasLength)
	}
	modelgemma4.ApplyDiffusionPolicyLabels(labels, rocmGemma4DiffusionPolicyFromProbe(cfg))
	inspection.Notes = append(inspection.Notes, "DiffusionGemma block-diffusion metadata is recognised; native ROCm canvas denoising sampler is not linked yet")
}

func applyROCmPortableArchitectureInspection(inspection *inference.ModelPackInspection, weightMetadataValid bool) {
	if inspection == nil {
		return
	}
	architectureDetected := strings.TrimSpace(inspection.Model.Architecture) != ""
	architectureOK := recognisedArchitecture(inspection.Model.Architecture)
	quantizationOK := supportedNativeQuantization(inspection.Model.QuantBits, inspection.Model.QuantType)
	inspection.Labels["architecture_detected"] = strconv.FormatBool(architectureDetected)
	inspection.Labels["architecture_supported"] = strconv.FormatBool(architectureOK)
	inspection.Labels["quantization_supported"] = strconv.FormatBool(quantizationOK)
	inspection.Supported = inspection.Format != "missing" && weightMetadataValid && architectureDetected && architectureOK && quantizationOK
	if inspection.Supported {
		inspection.Labels["model_pack_supported"] = "true"
	} else {
		inspection.Labels["model_pack_supported"] = "false"
	}
}

func mergeROCmPortableSafetensorsLabels(labels map[string]string, tensors []nativeTensorInfo) {
	if labels == nil {
		return
	}
	dtypes := map[string]bool{}
	if existing := labels["safetensors_dtypes"]; existing != "" {
		for part := range strings.SplitSeq(existing, ",") {
			if part != "" {
				dtypes[part] = true
			}
		}
	}
	var bytes uint64
	for _, tensor := range tensors {
		bytes += tensor.ByteSize
		if tensor.TypeName != "" {
			dtypes[strings.ToUpper(tensor.TypeName)] = true
		}
	}
	labels["safetensors_tensors"] = strconv.Itoa(rocmLabelInt(labels["safetensors_tensors"]) + len(tensors))
	labels["safetensors_payload_bytes"] = strconv.FormatUint(rocmLabelUint(labels["safetensors_payload_bytes"])+bytes, 10)
	labels["weight_bytes"] = strconv.FormatUint(rocmLabelUint(labels["weight_bytes"])+bytes, 10)
	if len(dtypes) > 0 {
		values := make([]string, 0, len(dtypes))
		for dtype := range dtypes {
			values = append(values, dtype)
		}
		slices.Sort(values)
		labels["safetensors_dtypes"] = strings.Join(values, ",")
	}
}

func readROCmSafetensorsNativeTensors(path string) ([]nativeTensorInfo, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var headerLength uint64
	if err := binary.Read(file, binary.LittleEndian, &headerLength); err != nil {
		return nil, err
	}
	if headerLength == 0 || headerLength > maxSafetensorsHeaderBytes {
		return nil, core.NewError(core.Sprintf("safetensors header length %d is outside supported bounds", headerLength))
	}
	dataOffset := int64(8 + headerLength)
	if info.Size() < dataOffset {
		return nil, core.NewError(core.Sprintf("safetensors file size %d is smaller than header span %d", info.Size(), dataOffset))
	}
	header := make([]byte, int(headerLength))
	if _, err := io.ReadFull(file, header); err != nil {
		return nil, err
	}
	tensors := map[string]rocmSafetensorsTensor{}
	if err := json.Unmarshal(header, &tensors); err != nil {
		return nil, err
	}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		if name != "__metadata__" {
			names = append(names, name)
		}
	}
	slices.Sort(names)
	payloadBytes := uint64(info.Size() - dataOffset)
	out := make([]nativeTensorInfo, 0, len(names))
	for _, name := range names {
		tensor := tensors[name]
		if tensor.DType == "" {
			return nil, core.NewError("safetensors tensor " + name + " is missing dtype")
		}
		if len(tensor.DataOffsets) != 2 {
			return nil, core.NewError("safetensors tensor " + name + " has invalid data_offsets")
		}
		if tensor.DataOffsets[1] < tensor.DataOffsets[0] || tensor.DataOffsets[1] > payloadBytes {
			return nil, core.NewError("safetensors tensor " + name + " has invalid payload range")
		}
		tensorType, ok := rocmSafetensorsNativeTensorType(tensor.DType)
		if !ok {
			return nil, core.NewError("safetensors tensor " + name + " has unsupported dtype " + tensor.DType)
		}
		dtypeBytes, ok := rocmSafetensorsDTypeBytes(tensor.DType)
		if !ok {
			return nil, core.NewError("safetensors tensor " + name + " has unsupported dtype " + tensor.DType)
		}
		shapeBytes, err := rocmSafetensorsShapeBytes(tensor.Shape, dtypeBytes)
		if err != nil {
			return nil, core.NewError("safetensors tensor " + name + " " + err.Error())
		}
		span := tensor.DataOffsets[1] - tensor.DataOffsets[0]
		if shapeBytes != span {
			return nil, core.NewError(core.Sprintf("safetensors tensor %s byte span %d does not match shape bytes %d", name, span, shapeBytes))
		}
		out = append(out, nativeTensorInfo{
			Name:       name,
			Dimensions: append([]uint64(nil), tensor.Shape...),
			Type:       tensorType,
			TypeName:   strings.ToUpper(tensor.DType),
			SourcePath: path,
			DataOffset: dataOffset,
			Offset:     tensor.DataOffsets[0],
			ByteSize:   span,
		})
	}
	if len(out) == 0 {
		return nil, core.NewError("safetensors header contains no tensor entries")
	}
	return out, nil
}

func rocmSafetensorsNativeTensorType(dtype string) (uint32, bool) {
	switch strings.ToUpper(strings.TrimSpace(dtype)) {
	case "F32":
		return 0, true
	case "F16":
		return 1, true
	case "BF16":
		return 30, true
	case "BOOL", "I8", "U8":
		return 24, true
	case "I16", "U16":
		return 25, true
	case "I32", "U32":
		return 26, true
	case "I64":
		return 27, true
	case "U64":
		return 28, true
	default:
		return 0, false
	}
}

func rocmSafetensorsDTypeBytes(dtype string) (uint64, bool) {
	switch strings.ToUpper(strings.TrimSpace(dtype)) {
	case "BOOL", "I8", "U8":
		return 1, true
	case "F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ", "F8_E5M2", "F8_E5M2FN", "F8_E5M2FNUZ":
		return 1, true
	case "I16", "U16", "F16", "BF16":
		return 2, true
	case "I32", "U32", "F32":
		return 4, true
	case "I64", "U64", "F64":
		return 8, true
	default:
		return 0, false
	}
}

func rocmSafetensorsShapeBytes(shape []uint64, dtypeBytes uint64) (uint64, error) {
	elements := uint64(1)
	for _, dimension := range shape {
		if dimension != 0 && elements > (^uint64(0))/dimension {
			return 0, core.NewError("shape element count overflows uint64")
		}
		elements *= dimension
	}
	if dtypeBytes != 0 && elements > (^uint64(0))/dtypeBytes {
		return 0, core.NewError("shape byte count overflows uint64")
	}
	return elements * dtypeBytes, nil
}

func rocmModelPackRoot(path string) (string, error) {
	return rocmmodel.ResolveModelPackRoot(path)
}

func rocmSafetensorsWeightFiles(path string) ([]string, error) {
	manifest, err := rocmmodel.InspectModelPackFiles(path)
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(manifest.WeightFiles))
	for _, weight := range manifest.WeightFiles {
		if weight.Format == rocmmodel.ModelPackFormatSafetensors {
			out = append(out, weight.Path)
		}
	}
	if len(out) == 0 {
		return nil, core.NewError("native safetensors load requires at least one safetensors weight file")
	}
	return out, nil
}

func discoverROCmWeightFiles(path string, info fs.FileInfo) []string {
	manifest, err := rocmmodel.InspectModelPackFiles(path)
	if err != nil {
		return nil
	}
	return manifest.WeightPaths()
}

func rocmIsWeightFile(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".gguf", ".safetensors":
		return true
	default:
		return false
	}
}

func rocmModelPackFormat(weights []string) string {
	hasGGUF := false
	hasSafetensors := false
	for _, weight := range weights {
		switch strings.ToLower(filepath.Ext(weight)) {
		case ".gguf":
			hasGGUF = true
		case ".safetensors":
			hasSafetensors = true
		}
	}
	switch {
	case hasGGUF && hasSafetensors:
		return "mixed"
	case hasGGUF:
		return "gguf"
	case hasSafetensors:
		return "safetensors"
	default:
		return "missing"
	}
}

func rocmConfigArchitecture(cfg rocmModelPackConfigProbe) string {
	if cfg.ModelType != "" {
		return normalizeROCmArchitecture(cfg.ModelType)
	}
	for _, architecture := range cfg.Architectures {
		if normalized := normalizeROCmArchitecture(architecture); normalized != "" {
			return normalized
		}
	}
	if cfg.TextConfig.ModelType != "" {
		return normalizeROCmArchitecture(cfg.TextConfig.ModelType)
	}
	for _, architecture := range cfg.TextConfig.Architectures {
		if normalized := normalizeROCmArchitecture(architecture); normalized != "" {
			return normalized
		}
	}
	return ""
}

func rocmConfigLayerTypes(cfg rocmModelPackConfigProbe) []string {
	numLayers := firstPositiveInt(cfg.NumHiddenLayers, cfg.NumLayers, cfg.TextConfig.NumHiddenLayers, cfg.TextConfig.NumLayers)
	switch {
	case len(cfg.LayerTypes) > 0:
		return normalizeSequenceMixerLayerTypes(cfg.LayerTypes)
	case len(cfg.TextConfig.LayerTypes) > 0:
		return normalizeSequenceMixerLayerTypes(cfg.TextConfig.LayerTypes)
	case numLayers > 0 && rocmConfigUniformSequenceMixerKind(cfg) != "":
		layerTypes := make([]string, numLayers)
		for i := range layerTypes {
			layerTypes[i] = rocmConfigUniformSequenceMixerKind(cfg)
		}
		return layerTypes
	default:
		return nil
	}
}

func normalizeSequenceMixerLayerTypes(values []string) []string {
	return rocmmodel.NormalizeSequenceMixerLayerTypes(values)
}

func rocmConfigSequenceMixerPlanLayerTypes(cfg rocmModelPackConfigProbe) ([]string, string) {
	return rocmmodel.SequenceMixerConfigPlanLayerTypes(rocmSequenceMixerConfigInput(cfg))
}

func rocmConfigSequenceMixerPlanError(cfg rocmModelPackConfigProbe) (string, error) {
	return rocmmodel.SequenceMixerConfigPlanError(rocmSequenceMixerConfigInput(cfg))
}

func rocmConfigUniformSequenceMixerKind(cfg rocmModelPackConfigProbe) string {
	return rocmmodel.SequenceMixerConfigUniformKind(rocmSequenceMixerConfigInput(cfg))
}

func rocmConfigComposedSequenceMixerModelType(cfg rocmModelPackConfigProbe) string {
	return rocmmodel.SequenceMixerConfigComposedModelType(rocmSequenceMixerConfigInput(cfg))
}

func rocmConfigTiedWordEmbeddings(cfg rocmModelPackConfigProbe) bool {
	if cfg.TieWordEmbeddings != nil {
		return *cfg.TieWordEmbeddings
	}
	if cfg.TextConfig.TieWordEmbeddings != nil {
		return *cfg.TextConfig.TieWordEmbeddings
	}
	return isROCmGemma4Architecture(rocmConfigArchitecture(cfg))
}

func rocmQuantConfigEmpty(quant rocmQuantizationConfigProbe) bool {
	return quant.Bits == 0 && quant.GroupSize == 0 && quant.QuantMethod == "" && quant.Algorithm == "" && quant.WeightFormat == "" && quant.Format == "" && quant.Type == ""
}

func rocmConfigDTypeQuantizationType(dtype string) string {
	switch strings.ToLower(strings.TrimSpace(dtype)) {
	case "float32", "fp32", "f32":
		return "f32"
	case "float16", "fp16", "f16":
		return "f16"
	case "bfloat16", "bf16":
		return "bf16"
	default:
		return ""
	}
}

func normalizeROCmArchitecture(architecture string) string {
	return rocmprofile.NormalizeArchitecture(architecture)
}

func isROCmGemma4Architecture(architecture string) bool {
	switch normalizeROCmArchitecture(architecture) {
	case "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text":
		return true
	default:
		return false
	}
}

func isROCmGemma4AssistantArchitecture(architecture string) bool {
	switch normalizeROCmArchitecture(architecture) {
	case "gemma4_assistant", "gemma4_unified_assistant":
		return true
	default:
		return false
	}
}

func supportedNativeArchitecture(architecture string) bool {
	return rocmprofile.SupportedNativeArchitecture(architecture)
}

// recognisedArchitecture is the inspection-surface predicate: known to the registry, runnable or
// not (see rocmprofile.RecognisedArchitecture — the #50 recognised-vs-runnable split).
func recognisedArchitecture(architecture string) bool {
	return rocmprofile.RecognisedArchitecture(architecture)
}

func supportedNativeQuantization(bits int, quantType string) bool {
	if bits == 0 && quantType == "" {
		return true
	}
	if bits > 0 && bits <= 8 {
		return true
	}
	switch strings.ToLower(strings.TrimSpace(quantType)) {
	case "", "f16", "f32", "bf16":
		return true
	default:
		return strings.Contains(quantType, "q2") ||
			strings.Contains(quantType, "q3") ||
			strings.Contains(quantType, "q4") ||
			strings.Contains(quantType, "q5") ||
			strings.Contains(quantType, "q6") ||
			strings.Contains(quantType, "q8")
	}
}

func isROCmMoEArchitecture(architecture string) bool {
	return rocmprofile.IsMoEArchitecture(architecture)
}

func NormalizeDenseLayerType(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.ReplaceAll(value, "-", "_")
	value = strings.ReplaceAll(value, ".", "_")
	return strings.ReplaceAll(value, " ", "_")
}

func DenseWeightNameCandidates(name string) []string {
	candidates := []string{name}
	if after, ok := strings.CutPrefix(name, "model."); ok {
		suffix := after
		return append(candidates,
			"language_model."+name,
			"language_model.model."+suffix,
			"model.language_model."+suffix,
			"model.language_model.model."+suffix,
		)
	}
	return append(candidates,
		"model."+name,
		"language_model."+name,
		"language_model.model."+name,
		"model.language_model."+name,
		"model.language_model.model."+name,
	)
}

func HasResolvedDenseWeightName(names map[string]bool, name string) bool {
	for _, candidate := range DenseWeightNameCandidates(name) {
		if names[candidate] {
			return true
		}
	}
	return false
}

func hipQ8ScaleIsPositiveFinite(scale float32) bool {
	return scale > 0 && !math.IsNaN(float64(scale)) && !math.IsInf(float64(scale), 0)
}

func firstPositiveInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func firstPositiveFloat(values ...float64) float64 {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func rocmLabelInt(value string) int {
	if value == "" {
		return 0
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 0 {
		return 0
	}
	return parsed
}

func rocmLabelUint(value string) uint64 {
	if value == "" {
		return 0
	}
	parsed, err := strconv.ParseUint(value, 10, 64)
	if err != nil {
		return 0
	}
	return parsed
}

func normalizeROCmLabelToken(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.ReplaceAll(value, "-", "_")
	value = strings.ReplaceAll(value, ".", "_")
	return strings.ReplaceAll(value, " ", "_")
}

func appendROCmInspectionCapability(inspection *inference.ModelPackInspection, capability inference.Capability) {
	if inspection == nil || capability.ID == "" {
		return
	}
	for index := range inspection.Capabilities {
		if inspection.Capabilities[index].ID == capability.ID && inspection.Capabilities[index].Group == capability.Group {
			inspection.Capabilities[index] = capability
			return
		}
	}
	inspection.Capabilities = append(inspection.Capabilities, capability)
}

func cloneAdapterIdentity(identity inference.AdapterIdentity) inference.AdapterIdentity {
	identity.TargetKeys = append([]string(nil), identity.TargetKeys...)
	identity.Labels = cloneStringMap(identity.Labels)
	return identity
}
