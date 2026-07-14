// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"slices"
	"strconv"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/gguf"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
	"dappco.re/go/inference/model/quant/codebook"
	"dappco.re/go/inference/model/quant/jang"
)

const maxSafetensorsHeaderBytes = 64 << 20

type rocmModelPackConfigProbe struct {
	ModelType                string                         `json:"model_type"`
	Architectures            []string                       `json:"architectures"`
	DType                    string                         `json:"dtype"`
	HiddenSize               int                            `json:"hidden_size"`
	NumHiddenLayers          int                            `json:"num_hidden_layers"`
	NumLayers                int                            `json:"num_layers"`
	NumAttentionHeads        int                            `json:"num_attention_heads"`
	NumKeyValueHeads         int                            `json:"num_key_value_heads"`
	NumGlobalKVHeads         int                            `json:"num_global_key_value_heads"`
	HeadDim                  int                            `json:"head_dim"`
	GlobalHeadDim            int                            `json:"global_head_dim"`
	GlobalPartialRotary      float64                        `json:"global_partial_rotary_factor"`
	VocabSize                int                            `json:"vocab_size"`
	VocabSizePerLayer        int                            `json:"vocab_size_per_layer_input"`
	IntermediateSize         int                            `json:"intermediate_size"`
	MaxPositionEmbeddings    int                            `json:"max_position_embeddings"`
	MaxSequenceLength        int                            `json:"max_sequence_length"`
	SeqLength                int                            `json:"seq_length"`
	CanvasLength             int                            `json:"canvas_length"`
	BackboneHiddenSize       int                            `json:"backbone_hidden_size"`
	NumCentroids             int                            `json:"num_centroids"`
	CentroidIntermediateTopK int                            `json:"centroid_intermediate_top_k"`
	UseOrderedEmbeddings     *bool                          `json:"use_ordered_embeddings"`
	SlidingWindow            int                            `json:"sliding_window"`
	SlidingWindowPattern     int                            `json:"sliding_window_pattern"`
	NumKVSharedLayers        *int                           `json:"num_kv_shared_layers"`
	HiddenSizePerLayer       int                            `json:"hidden_size_per_layer_input"`
	LayerTypes               []string                       `json:"layer_types"`
	AttentionKEqV            bool                           `json:"attention_k_eq_v"`
	RoPEParameters           map[string]rocmRoPEProbe       `json:"rope_parameters"`
	RMSNormEps               float64                        `json:"rms_norm_eps"`
	FinalLogitSoftcap        float64                        `json:"final_logit_softcapping"`
	NumLocalExperts          int                            `json:"num_local_experts"`
	NumExperts               int                            `json:"num_experts"`
	NumExpertsPerTok         int                            `json:"num_experts_per_tok"`
	UseRoutingBias           bool                           `json:"use_routing_bias"`
	TopKExperts              int                            `json:"top_k_experts"`
	DecoderSparseStep        int                            `json:"decoder_sparse_step"`
	MoEIntermediateSize      int                            `json:"moe_intermediate_size"`
	ExpertIntermediateSize   int                            `json:"expert_intermediate_size"`
	ImageTokenID             int                            `json:"image_token_id"`
	ImageTokenIndex          int                            `json:"image_token_index"`
	BOITokenIndex            int                            `json:"boi_token_index"`
	BOITokenID               int                            `json:"boi_token_id"`
	BOATokenID               int                            `json:"boa_token_id"`
	BOATokenIndex            int                            `json:"boa_token_index"`
	EOITokenIndex            int                            `json:"eoi_token_index"`
	EOITokenID               int                            `json:"eoi_token_id"`
	EOATokenID               int                            `json:"eoa_token_id"`
	EOATokenIndex            int                            `json:"eoa_token_index"`
	AudioTokenID             int                            `json:"audio_token_id"`
	AudioTokenIndex          int                            `json:"audio_token_index"`
	VideoTokenID             int                            `json:"video_token_id"`
	VisionSoftTokensPerImage int                            `json:"vision_soft_tokens_per_image"`
	MMTokensPerImage         int                            `json:"mm_tokens_per_image"`
	QLoRARank                int                            `json:"q_lora_rank"`
	KVLoRARank               int                            `json:"kv_lora_rank"`
	QKNoPEHeadDim            int                            `json:"qk_nope_head_dim"`
	QKRoPEHeadDim            int                            `json:"qk_rope_head_dim"`
	QKHeadDim                int                            `json:"qk_head_dim"`
	VHeadDim                 int                            `json:"v_head_dim"`
	UseDoubleWideMLP         bool                           `json:"use_double_wide_mlp"`
	EnableMoEBlock           bool                           `json:"enable_moe_block"`
	QuantizationConfig       rocmQuantizationConfigProbe    `json:"quantization_config"`
	Quantization             rocmQuantizationConfigProbe    `json:"quantization"`
	TaskSpecificParams       map[string]any                 `json:"task_specific_params"`
	TextConfig               rocmModelPackTextConfigProbe   `json:"text_config"`
	VisionConfig             rocmModelPackVisionConfigProbe `json:"vision_config"`
	AudioConfig              rocmModelPackAudioConfigProbe  `json:"audio_config"`
	TieWordEmbeddings        *bool                          `json:"tie_word_embeddings"`
}

type rocmModelPackTextConfigProbe struct {
	ModelType                string                   `json:"model_type"`
	Architectures            []string                 `json:"architectures"`
	DType                    string                   `json:"dtype"`
	HiddenSize               int                      `json:"hidden_size"`
	NumHiddenLayers          int                      `json:"num_hidden_layers"`
	NumLayers                int                      `json:"num_layers"`
	NumAttentionHeads        int                      `json:"num_attention_heads"`
	NumKeyValueHeads         int                      `json:"num_key_value_heads"`
	NumGlobalKVHeads         int                      `json:"num_global_key_value_heads"`
	HeadDim                  int                      `json:"head_dim"`
	GlobalHeadDim            int                      `json:"global_head_dim"`
	GlobalPartialRotary      float64                  `json:"global_partial_rotary_factor"`
	VocabSize                int                      `json:"vocab_size"`
	VocabSizePerLayer        int                      `json:"vocab_size_per_layer_input"`
	IntermediateSize         int                      `json:"intermediate_size"`
	MaxPositionEmbeddings    int                      `json:"max_position_embeddings"`
	MaxSequenceLength        int                      `json:"max_sequence_length"`
	SeqLength                int                      `json:"seq_length"`
	CanvasLength             int                      `json:"canvas_length"`
	BackboneHiddenSize       int                      `json:"backbone_hidden_size"`
	NumCentroids             int                      `json:"num_centroids"`
	CentroidIntermediateTopK int                      `json:"centroid_intermediate_top_k"`
	UseOrderedEmbeddings     *bool                    `json:"use_ordered_embeddings"`
	SlidingWindow            int                      `json:"sliding_window"`
	SlidingWindowPattern     int                      `json:"sliding_window_pattern"`
	NumKVSharedLayers        *int                     `json:"num_kv_shared_layers"`
	HiddenSizePerLayer       int                      `json:"hidden_size_per_layer_input"`
	LayerTypes               []string                 `json:"layer_types"`
	AttentionKEqV            bool                     `json:"attention_k_eq_v"`
	RoPEParameters           map[string]rocmRoPEProbe `json:"rope_parameters"`
	RMSNormEps               float64                  `json:"rms_norm_eps"`
	FinalLogitSoftcap        float64                  `json:"final_logit_softcapping"`
	NumExperts               int                      `json:"num_experts"`
	NumExpertsPerTok         int                      `json:"num_experts_per_tok"`
	UseRoutingBias           bool                     `json:"use_routing_bias"`
	TopKExperts              int                      `json:"top_k_experts"`
	DecoderSparseStep        int                      `json:"decoder_sparse_step"`
	MoEIntermediateSize      int                      `json:"moe_intermediate_size"`
	ExpertIntermediateSize   int                      `json:"expert_intermediate_size"`
	QLoRARank                int                      `json:"q_lora_rank"`
	KVLoRARank               int                      `json:"kv_lora_rank"`
	QKNoPEHeadDim            int                      `json:"qk_nope_head_dim"`
	QKRoPEHeadDim            int                      `json:"qk_rope_head_dim"`
	QKHeadDim                int                      `json:"qk_head_dim"`
	VHeadDim                 int                      `json:"v_head_dim"`
	UseDoubleWideMLP         bool                     `json:"use_double_wide_mlp"`
	EnableMoEBlock           bool                     `json:"enable_moe_block"`
	TieWordEmbeddings        *bool                    `json:"tie_word_embeddings"`
}

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

type rocmRoPEProbe struct {
	PartialRotaryFactor float64 `json:"partial_rotary_factor"`
	RopeTheta           float64 `json:"rope_theta"`
	RopeType            string  `json:"rope_type"`
	Factor              float64 `json:"factor"`
}

type rocmTokenizerJSONProbe struct {
	Model struct {
		Type string `json:"type"`
	} `json:"model"`
}

type rocmTokenizerConfigProbe struct {
	TokenizerClass string                      `json:"tokenizer_class"`
	ChatTemplate   string                      `json:"chat_template"`
	BOSID          rocmTokenizerTokenID        `json:"bos_token_id"`
	EOSID          rocmTokenizerTokenID        `json:"eos_token_id"`
	PADID          rocmTokenizerTokenID        `json:"pad_token_id"`
	ModelMaxLength rocmTokenizerModelMaxLength `json:"model_max_length"`
}

type rocmQuantizationConfigProbe struct {
	QuantMethod  string `json:"quant_method"`
	Algorithm    string `json:"algorithm"`
	Bits         int    `json:"bits"`
	GroupSize    int    `json:"group_size"`
	WeightFormat string `json:"weight_format"`
	Format       string `json:"format"`
	Scheme       string `json:"scheme"`
	Type         string `json:"type"`
	Iters        int    `json:"iters"`
	NSamples     int    `json:"nsamples"`
	SeqLen       int    `json:"seqlen"`
	Sym          *bool  `json:"sym"`
	Asym         *bool  `json:"asym"`
	LoadIn4Bit   bool   `json:"load_in_4bit"`
	LoadIn8Bit   bool   `json:"load_in_8bit"`
}

type rocmJANGQuantizationInfo = jang.Info
type rocmCodebookProfile = codebook.Profile

type rocmSafetensorsTensor struct {
	DType       string   `json:"dtype"`
	Shape       []uint64 `json:"shape"`
	DataOffsets []uint64 `json:"data_offsets"`
}

type rocmSafetensorsSummary struct {
	TensorCount  int
	HeaderBytes  uint64
	PayloadBytes uint64
	DTypes       []string
}

type rocmSafetensorsIndexProbe struct {
	Metadata  rocmSafetensorsIndexMetadata `json:"metadata"`
	WeightMap map[string]string            `json:"weight_map"`
}

type rocmSafetensorsIndexMetadata struct {
	TotalSize       uint64 `json:"total_size"`
	TotalParameters uint64 `json:"total_parameters"`
}

type rocmSafetensorsPayloadRange struct {
	Name  string
	Start uint64
	End   uint64
}

type rocmTokenizerTokenID struct {
	Values []int32
}

type rocmTokenizerModelMaxLength struct {
	Value int
}

func (length *rocmTokenizerModelMaxLength) UnmarshalJSON(data []byte) error {
	if string(data) == "null" {
		return nil
	}
	var raw any
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&raw); err != nil {
		return nil
	}
	var text string
	switch value := raw.(type) {
	case json.Number:
		text = value.String()
	case string:
		text = value
	default:
		return nil
	}
	parsed, err := strconv.ParseUint(text, 10, 64)
	if err != nil || parsed > 1<<30 {
		return nil
	}
	length.Value = int(parsed)
	return nil
}

func (id *rocmTokenizerTokenID) UnmarshalJSON(data []byte) error {
	if string(data) == "null" {
		return nil
	}
	var single int32
	if err := json.Unmarshal(data, &single); err == nil {
		id.Values = []int32{single}
		return nil
	}
	var many []int32
	if err := json.Unmarshal(data, &many); err == nil {
		id.Values = append([]int32(nil), many...)
		return nil
	}
	return nil
}

func (id rocmTokenizerTokenID) First() int32 {
	for _, value := range id.Values {
		if value != 0 {
			return value
		}
	}
	return 0
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
		Path:   resolvedPath,
		Labels: map[string]string{"backend": "rocm", "native_runtime": "hip"},
	}
	weights := fileManifest.WeightPaths()
	inspection.Format = fileManifest.Format
	for key, value := range fileManifest.Labels {
		if value != "" {
			inspection.Labels[key] = value
		}
	}
	inspection.Labels["weight_files"] = core.Sprintf("%d", len(weights))
	inspection.Labels["format"] = inspection.Format
	if len(weights) == 0 {
		inspection.Notes = append(inspection.Notes, "no GGUF or safetensors weight files found")
	}

	if cfg, err := readROCmModelConfig(root); err != nil {
		inspection.Notes = append(inspection.Notes, "config.json could not be parsed: "+err.Error())
	} else if cfg != nil {
		applyROCmModelConfig(inspection, *cfg)
	}
	if processor, err := readROCmGemma4ProcessorConfig(root); err != nil {
		inspection.Notes = append(inspection.Notes, "processor_config.json could not be parsed: "+err.Error())
	} else if processor != nil {
		applyROCmGemma4ProcessorConfigLabels(inspection, *processor)
	}
	weightMetadataValid := len(weights) > 0
	for _, weight := range weights {
		valid := false
		switch core.Lower(core.PathExt(weight)) {
		case ".gguf":
			valid = applyROCmGGUFInspection(inspection, weight)
		case ".safetensors":
			valid = applyROCmSafetensorsInspection(inspection, weight)
		}
		weightMetadataValid = weightMetadataValid && valid
	}
	if inspection.Format == "safetensors" && weightMetadataValid {
		if err := rocmApplySequenceMixerSafetensorsPackPlanLabels(inspection, weights); err != nil {
			inspection.Notes = append(inspection.Notes, "sequence mixer safetensors plan could not be validated: "+err.Error())
			weightMetadataValid = false
		}
	}
	if indexValid, err := applyROCmSafetensorsIndexInspection(inspection, root, weights); err != nil {
		inspection.Notes = append(inspection.Notes, "safetensors index could not be parsed: "+err.Error())
		weightMetadataValid = false
	} else {
		weightMetadataValid = weightMetadataValid && indexValid
	}
	if !weightMetadataValid {
		clearROCmWeightMetadataLabels(inspection.Labels)
	}
	inspection.Labels["weight_metadata_valid"] = core.Sprintf("%t", weightMetadataValid)
	if jang, err := readROCmJANGConfig(root); err != nil {
		inspection.Notes = append(inspection.Notes, "jang_config.json could not be parsed: "+err.Error())
	} else if jang != nil {
		applyROCmJANGInspection(inspection, *jang)
	}
	if codebook, err := readROCmCodebookConfig(root); err != nil {
		inspection.Notes = append(inspection.Notes, "codebook_config.json could not be parsed: "+err.Error())
	} else if codebook != nil {
		applyROCmCodebookInspection(inspection, *codebook)
	}
	if err := applyROCmTokenizerJSONInspection(inspection, root); err != nil {
		inspection.Notes = append(inspection.Notes, "tokenizer.json could not be parsed: "+err.Error())
	}
	if err := applyROCmTokenizerConfigInspection(inspection, root); err != nil {
		inspection.Notes = append(inspection.Notes, "tokenizer_config.json could not be parsed: "+err.Error())
	}
	applyROCmInspectionModelProfile(inspection)
	applyROCmArchitectureInspection(inspection, weightMetadataValid)
	applyROCmGemma4ModelPackSupportLabels(inspection, resolvedPath)
	applyROCmMemoryFitInspection(ctx, b, inspection)
	appendROCmInspectionCapability(inspection, inference.SupportedCapability(inference.CapabilityModelFit, inference.CapabilityGroupRuntime))
	appendROCmInspectionCapability(inspection, inference.SupportedCapability(inference.CapabilityMemoryPlanning, inference.CapabilityGroupRuntime))
	appendROCmInspectionCapability(inspection, inference.SupportedCapability(inference.CapabilityKVCachePlanning, inference.CapabilityGroupRuntime))
	applyROCmGemma4ModelPackInspectionCapabilities(inspection)
	inspection.Notes = append(inspection.Notes, "native ROCm decode kernels are not linked yet")
	return inspection, nil
}

func discoverROCmWeightFiles(path string, info core.FsFileInfo) []string {
	manifest, err := rocmmodel.InspectModelPackFiles(path)
	if err != nil {
		return nil
	}
	return manifest.WeightPaths()
}

func rocmIsWeightFile(path string) bool {
	ext := core.Lower(core.PathExt(path))
	return ext == ".gguf" || ext == ".safetensors"
}

func rocmModelPackFormat(weights []string) string {
	gguf := 0
	safetensors := 0
	for _, weight := range weights {
		switch core.Lower(core.PathExt(weight)) {
		case ".gguf":
			gguf++
		case ".safetensors":
			safetensors++
		}
	}
	switch {
	case gguf > 0 && safetensors > 0:
		return rocmmodel.ModelPackFormatMixed
	case gguf > 0:
		return rocmmodel.ModelPackFormatGGUF
	case safetensors > 0:
		return rocmmodel.ModelPackFormatSafetensors
	default:
		return rocmmodel.ModelPackFormatMissing
	}
}

func readROCmModelConfig(root string) (*rocmModelPackConfigProbe, error) {
	read := core.ReadFile(core.PathJoin(root, "config.json"))
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return nil, nil
		}
		return nil, read.Value.(error)
	}
	var cfg rocmModelPackConfigProbe
	if result := core.JSONUnmarshal(read.Value.([]byte), &cfg); !result.OK {
		return nil, result.Value.(error)
	}
	return &cfg, nil
}

func readROCmGemma4ProcessorConfig(root string) (*modelgemma4.ProcessorConfig, error) {
	read := core.ReadFile(core.PathJoin(root, "processor_config.json"))
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return nil, nil
		}
		return nil, read.Value.(error)
	}
	cfg, err := modelgemma4.ParseProcessorConfig(read.Value.([]byte))
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

func applyROCmModelConfig(inspection *inference.ModelPackInspection, cfg rocmModelPackConfigProbe) {
	model := inspection.Model
	model.Architecture = firstNonEmptyString(model.Architecture, rocmConfigArchitecture(cfg))
	model.ContextLength = firstPositiveInt(model.ContextLength, cfg.MaxPositionEmbeddings, cfg.MaxSequenceLength, cfg.SeqLength, cfg.SlidingWindow, cfg.TextConfig.MaxPositionEmbeddings, cfg.TextConfig.MaxSequenceLength, cfg.TextConfig.SeqLength, cfg.TextConfig.SlidingWindow)
	model.NumLayers = firstPositiveInt(model.NumLayers, cfg.NumHiddenLayers, cfg.NumLayers, cfg.TextConfig.NumHiddenLayers, cfg.TextConfig.NumLayers)
	model.HiddenSize = firstPositiveInt(model.HiddenSize, cfg.HiddenSize, cfg.TextConfig.HiddenSize)
	model.VocabSize = firstPositiveInt(model.VocabSize, cfg.VocabSize, cfg.TextConfig.VocabSize)
	quant := cfg.QuantizationConfig
	if rocmQuantConfigEmpty(quant) {
		quant = cfg.Quantization
	}
	model.QuantBits = firstPositiveInt(model.QuantBits, rocmQuantConfigBits(quant))
	model.QuantGroup = firstPositiveInt(model.QuantGroup, quant.GroupSize)
	quantType := rocmQuantConfigType(quant)
	if quantType == "" && model.QuantBits == 0 {
		quantType = firstNonEmptyString(rocmConfigDTypeQuantizationType(cfg.DType), rocmConfigDTypeQuantizationType(cfg.TextConfig.DType))
	}
	model.QuantType = firstNonEmptyString(model.QuantType, quantType)
	inspection.Model = model
	rocmApplyArchitectureResolutionLabels(inspection.Labels, cfg)
	if experts := firstPositiveInt(cfg.NumLocalExperts, cfg.NumExperts, cfg.TextConfig.NumExperts); experts > 0 {
		inspection.Labels["moe_experts"] = core.Sprintf("%d", experts)
	}
	if topK := firstPositiveInt(cfg.NumExpertsPerTok, cfg.TopKExperts, cfg.TextConfig.NumExpertsPerTok, cfg.TextConfig.TopKExperts); topK > 0 {
		inspection.Labels["moe_top_k"] = core.Sprintf("%d", topK)
	}
	if sparseStep := firstPositiveInt(cfg.DecoderSparseStep, cfg.TextConfig.DecoderSparseStep); sparseStep > 0 {
		inspection.Labels["moe_sparse_step"] = core.Sprintf("%d", sparseStep)
	}
	applyROCmMiniMaxM2ConfigLabels(inspection, cfg)
	applyROCmMixtralConfigLabels(inspection, cfg)
	if rocmConfigTiedWordEmbeddings(cfg) {
		inspection.Labels["tied_word_embeddings"] = "true"
	}
	applyROCmAttentionConfigLabels(inspection, cfg)
	applyROCmGemma4AssistantConfigLabels(inspection, cfg)
	applyROCmMultimodalConfigLabels(inspection, cfg)
	applyROCmDiffusionGemmaConfigLabels(inspection, cfg)
	applyROCMAutoRoundQuantizationLabels(inspection, quant)
	if rocmConfigHasEmbeddingTask(cfg) {
		inspection.Labels["embedding_model"] = "true"
		appendROCmInspectionCapability(inspection, inference.PlannedCapability(inference.CapabilityEmbeddings, inference.CapabilityGroupModel, "embedding model-pack metadata is recognised; native ROCm embedding kernels are pending"))
	}
	if rocmConfigHasRerankTask(cfg) {
		inspection.Labels["rerank_model"] = "true"
		appendROCmInspectionCapability(inspection, inference.PlannedCapability(inference.CapabilityRerank, inference.CapabilityGroupModel, "rerank model-pack metadata is recognised; native ROCm scorer kernels are pending"))
	}
	if rocmConfigHasClassifierTask(cfg) {
		inspection.Labels["classifier_model"] = "true"
		capability := inference.PlannedCapability(inference.CapabilityClassify, inference.CapabilityGroupModel, "BERT sequence-classifier metadata is recognised; loaded ROCm classifier path is experimental when embedding and projection kernels are linked")
		capability.Labels = map[string]string{"classify_path": "bert_sequence_classifier"}
		appendROCmInspectionCapability(inspection, capability)
	}
}

func rocmQuantConfigEmpty(quant rocmQuantizationConfigProbe) bool {
	return quant.QuantMethod == "" && quant.Algorithm == "" && quant.Bits == 0 && quant.GroupSize == 0 && quant.WeightFormat == "" && quant.Format == "" && quant.Scheme == "" && quant.Type == "" && quant.Iters == 0 && quant.NSamples == 0 && quant.SeqLen == 0 && quant.Sym == nil && quant.Asym == nil && !quant.LoadIn4Bit && !quant.LoadIn8Bit
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

func rocmConfigTiedWordEmbeddings(cfg rocmModelPackConfigProbe) bool {
	if cfg.TieWordEmbeddings != nil {
		return *cfg.TieWordEmbeddings
	}
	if cfg.TextConfig.TieWordEmbeddings != nil {
		return *cfg.TextConfig.TieWordEmbeddings
	}
	return isROCmGemma4Architecture(rocmConfigArchitecture(cfg))
}

func rocmConfigLayerTypes(cfg rocmModelPackConfigProbe) []string {
	numLayers := firstPositiveInt(cfg.NumHiddenLayers, cfg.NumLayers, cfg.TextConfig.NumHiddenLayers, cfg.TextConfig.NumLayers)
	architecture := normalizeROCmArchitecture(rocmConfigArchitecture(cfg))
	isQwen36 := architecture == "qwen3_6" || architecture == "qwen3_6_moe"
	var layerTypes []string
	explicitPattern := false
	switch {
	case len(cfg.LayerTypes) > 0:
		layerTypes = append([]string(nil), cfg.LayerTypes...)
		explicitPattern = true
	case len(cfg.TextConfig.LayerTypes) > 0:
		layerTypes = append([]string(nil), cfg.TextConfig.LayerTypes...)
		explicitPattern = true
	default:
		if numLayers <= 0 {
			return nil
		}
		if uniform := rocmConfigUniformSequenceMixerKind(cfg); uniform != "" {
			layerTypes = make([]string, numLayers)
			for index := range layerTypes {
				layerTypes[index] = uniform
			}
			explicitPattern = true
			break
		}
		if rocmConfigComposedSequenceMixerModelType(cfg) != "" {
			return nil
		}
		pattern := firstPositiveInt(cfg.SlidingWindowPattern, cfg.TextConfig.SlidingWindowPattern)
		if pattern <= 0 {
			pattern = 6
		}
		layerTypes = make([]string, numLayers)
		for index := range layerTypes {
			if pattern > 1 && (index+1)%pattern != 0 {
				layerTypes[index] = "sliding_attention"
			} else {
				layerTypes[index] = "full_attention"
			}
		}
		if len(layerTypes) > 0 {
			layerTypes[len(layerTypes)-1] = "full_attention"
		}
	}
	if explicitPattern && isQwen36 && numLayers > 0 && len(layerTypes) > 0 && len(layerTypes) < numLayers {
		pattern := layerTypes
		layerTypes = make([]string, numLayers)
		for index := range layerTypes {
			layerTypes[index] = pattern[index%len(pattern)]
		}
	}
	if numLayers > 0 && len(layerTypes) >= numLayers {
		layerTypes = layerTypes[:numLayers]
		if !explicitPattern || (!isQwen36 && !rocmLayerTypesIncludeFLAMixer(layerTypes)) {
			layerTypes[len(layerTypes)-1] = "full_attention"
		}
	}
	return layerTypes
}

func rocmConfigSequenceMixerPlanLayerTypes(cfg rocmModelPackConfigProbe) ([]string, string) {
	return rocmmodel.SequenceMixerConfigPlanLayerTypes(rocmSequenceMixerConfigInput(cfg))
}

func rocmConfigSequenceMixerPlanError(cfg rocmModelPackConfigProbe) (string, error) {
	return rocmmodel.SequenceMixerConfigPlanError(rocmSequenceMixerConfigInput(cfg))
}

func normalizeSequenceMixerLayerTypes(values []string) []string {
	return rocmmodel.NormalizeSequenceMixerLayerTypes(values)
}

func rocmConfigUniformSequenceMixerKind(cfg rocmModelPackConfigProbe) string {
	return rocmmodel.SequenceMixerConfigUniformKind(rocmSequenceMixerConfigInput(cfg))
}

func rocmConfigComposedSequenceMixerModelType(cfg rocmModelPackConfigProbe) string {
	return rocmmodel.SequenceMixerConfigComposedModelType(rocmSequenceMixerConfigInput(cfg))
}

func rocmLayerTypesIncludeFLAMixer(layerTypes []string) bool {
	for _, layerType := range layerTypes {
		family, ok := SequenceMixerFamilyByKind(layerType)
		if ok && family.Source == "fla" {
			return true
		}
	}
	return false
}

func rocmConfigKVSharedLayers(cfg rocmModelPackConfigProbe) (int, bool) {
	switch {
	case cfg.NumKVSharedLayers != nil:
		return *cfg.NumKVSharedLayers, true
	case cfg.TextConfig.NumKVSharedLayers != nil:
		return *cfg.TextConfig.NumKVSharedLayers, true
	default:
		return 0, false
	}
}

func applyROCmMiniMaxM2ConfigLabels(inspection *inference.ModelPackInspection, cfg rocmModelPackConfigProbe) {
	if inspection == nil || normalizeROCmArchitecture(rocmConfigArchitecture(cfg)) != "minimax_m2" {
		return
	}
	labels := inspection.Labels
	labels["minimax_m2_sparse_plan"] = "staged_metadata"
	if intermediate := firstPositiveInt(cfg.IntermediateSize, cfg.TextConfig.IntermediateSize); intermediate > 0 {
		labels["minimax_m2_intermediate_size"] = core.Sprintf("%d", intermediate)
	}
	if experts := firstPositiveInt(cfg.NumLocalExperts, cfg.NumExperts, cfg.TextConfig.NumExperts); experts > 0 {
		labels["minimax_m2_local_experts"] = core.Sprintf("%d", experts)
	}
	if topK := firstPositiveInt(cfg.NumExpertsPerTok, cfg.TopKExperts, cfg.TextConfig.NumExpertsPerTok, cfg.TextConfig.TopKExperts); topK > 0 {
		labels["minimax_m2_experts_per_token"] = core.Sprintf("%d", topK)
	}
	if cfg.UseRoutingBias || cfg.TextConfig.UseRoutingBias {
		labels["minimax_m2_routing_bias"] = "true"
		labels["minimax_m2_required_router_bias_tensor"] = "model.layers.0.block_sparse_moe.e_score_correction_bias"
	} else {
		labels["minimax_m2_routing_bias"] = "false"
	}
	labels["minimax_m2_required_router_tensor"] = "model.layers.0.block_sparse_moe.gate.weight"
	labels["minimax_m2_required_expert_tensors"] = "gate_proj,up_proj,down_proj"
}

func applyROCmMixtralConfigLabels(inspection *inference.ModelPackInspection, cfg rocmModelPackConfigProbe) {
	if inspection == nil || normalizeROCmArchitecture(rocmConfigArchitecture(cfg)) != "mixtral" {
		return
	}
	labels := inspection.Labels
	labels["mixtral_sparse_plan"] = "metadata"
	experts := firstPositiveInt(cfg.NumLocalExperts, cfg.NumExperts, cfg.TextConfig.NumExperts)
	if experts == 0 {
		experts = 8
	}
	topK := firstPositiveInt(cfg.NumExpertsPerTok, cfg.TopKExperts, cfg.TextConfig.NumExpertsPerTok, cfg.TextConfig.TopKExperts)
	if topK == 0 {
		topK = 2
	}
	if labels["moe_experts"] == "" {
		labels["moe_experts"] = core.Sprintf("%d", experts)
	}
	if labels["moe_top_k"] == "" {
		labels["moe_top_k"] = core.Sprintf("%d", topK)
	}
	labels["mixtral_local_experts"] = core.Sprintf("%d", experts)
	labels["mixtral_experts_per_token"] = core.Sprintf("%d", topK)
	if sparseStep := firstPositiveInt(cfg.DecoderSparseStep, cfg.TextConfig.DecoderSparseStep); sparseStep > 0 {
		labels["mixtral_sparse_step"] = core.Sprintf("%d", sparseStep)
	} else {
		labels["mixtral_sparse_step"] = "all"
	}
	labels["mixtral_required_router_tensor"] = "model.layers.0.block_sparse_moe.gate.weight"
	labels["mixtral_required_expert_tensors"] = "w1,w2,w3"
}

func applyROCmMultimodalConfigLabels(inspection *inference.ModelPackInspection, cfg rocmModelPackConfigProbe) {
	if inspection == nil {
		return
	}
	architecture := rocmConfigArchitecture(cfg)
	labels := inspection.Labels
	imageToken := firstPositiveInt(cfg.ImageTokenID, cfg.ImageTokenIndex)
	audioToken := firstPositiveInt(cfg.AudioTokenID, cfg.AudioTokenIndex)
	softTokens := firstPositiveInt(cfg.VisionSoftTokensPerImage, cfg.MMTokensPerImage, cfg.VisionConfig.DefaultOutputLength)
	hasVision := rocmModelPackConfigHasVision(cfg)
	hasAudio := rocmModelPackConfigHasAudio(cfg)
	if !hasVision && !hasAudio {
		return
	}
	labels["multimodal_model"] = "true"
	if isROCmGemma4Architecture(architecture) {
		labels["gemma4_multimodal"] = "true"
	}
	if hasVision {
		visionRuntime := hipKernelStatusNotLinked
		if isROCmGemma4Architecture(architecture) {
			visionRuntime = hipKernelStatusLinked
		}
		labels["vision_runtime"] = visionRuntime
		labels["vision_projector_runtime"] = visionRuntime
		switch {
		case isROCmGemma4Architecture(architecture):
			labels["vision_reference"] = "go_mlx_gemma4_vision"
		case normalizeROCmArchitecture(architecture) == "gemma3":
			labels["gemma3_multimodal"] = "true"
			labels["vision_reference"] = "go_mlx_gemma3_multimodal_wrapper"
		default:
			labels["vision_reference"] = "model_pack_multimodal_metadata"
		}
		if isROCmGemma4Architecture(architecture) {
			modelgemma4.ApplyVisionConfigLabels(labels, rocmGemma4VisionConfigFromProbe(cfg))
		} else {
			if imageToken > 0 {
				labels["image_token_id"] = core.Sprintf("%d", imageToken)
			}
			if cfg.BOITokenIndex > 0 {
				labels["boi_token_index"] = core.Sprintf("%d", cfg.BOITokenIndex)
			}
			if cfg.BOITokenID > 0 {
				labels["boi_token_id"] = core.Sprintf("%d", cfg.BOITokenID)
			}
			if cfg.EOITokenIndex > 0 {
				labels["eoi_token_index"] = core.Sprintf("%d", cfg.EOITokenIndex)
			}
			if cfg.EOITokenID > 0 {
				labels["eoi_token_id"] = core.Sprintf("%d", cfg.EOITokenID)
			}
			if cfg.VideoTokenID > 0 {
				labels["video_token_id"] = core.Sprintf("%d", cfg.VideoTokenID)
			}
			if softTokens > 0 {
				labels["vision_soft_tokens_per_image"] = core.Sprintf("%d", softTokens)
			}
			applyROCmVisionTowerLabels(labels, cfg.VisionConfig)
		}
		if isROCmGemma4Architecture(architecture) {
			inspection.Notes = append(inspection.Notes, "Gemma4 vision tower and projector are auto-attached from the model pack")
		} else {
			inspection.Notes = append(inspection.Notes, "multimodal vision metadata is recognised; native ROCm vision tower and projector kernels are pending")
		}
	}
	if hasAudio {
		audioRuntime := hipKernelStatusNotLinked
		if isROCmGemma4Architecture(architecture) {
			audioRuntime = hipKernelStatusLinked
		}
		labels["audio_runtime"] = audioRuntime
		labels["audio_projector_runtime"] = audioRuntime
		labels["audio_frontend_runtime"] = audioRuntime
		if isROCmGemma4Architecture(architecture) {
			labels["audio_reference"] = "go_mlx_gemma4_audio"
		} else {
			labels["audio_reference"] = "model_pack_audio_metadata"
		}
		if isROCmGemma4Architecture(architecture) {
			modelgemma4.ApplyAudioConfigLabels(labels, rocmGemma4AudioConfigFromProbe(cfg))
		} else {
			if audioToken > 0 {
				labels["audio_token_id"] = core.Sprintf("%d", audioToken)
			}
			if cfg.BOATokenID > 0 {
				labels["boa_token_id"] = core.Sprintf("%d", cfg.BOATokenID)
			}
			if cfg.BOATokenIndex > 0 {
				labels["boa_token_index"] = core.Sprintf("%d", cfg.BOATokenIndex)
			}
			if cfg.EOATokenID > 0 {
				labels["eoa_token_id"] = core.Sprintf("%d", cfg.EOATokenID)
			}
			if cfg.EOATokenIndex > 0 {
				labels["eoa_token_index"] = core.Sprintf("%d", cfg.EOATokenIndex)
			}
			applyROCMAudioTowerLabels(labels, cfg.AudioConfig)
		}
		if isROCmGemma4Architecture(architecture) {
			inspection.Notes = append(inspection.Notes, "Gemma4 audio frontend, tower, and projector are auto-attached from the model pack")
		} else {
			inspection.Notes = append(inspection.Notes, "multimodal audio metadata is recognised; native ROCm audio front-end, tower, and projector kernels are pending")
		}
	}
}

func applyROCmDiffusionGemmaConfigLabels(inspection *inference.ModelPackInspection, cfg rocmModelPackConfigProbe) {
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
		labels["diffusion_canvas_length"] = core.Sprintf("%d", canvasLength)
	}
	modelgemma4.ApplyDiffusionPolicyLabels(labels, rocmGemma4DiffusionPolicyFromProbe(cfg))
	inspection.Notes = append(inspection.Notes, "DiffusionGemma block-diffusion metadata is recognised; native ROCm canvas denoising sampler is not linked yet")
}

func applyROCmVisionTowerLabels(labels map[string]string, cfg rocmModelPackVisionConfigProbe) {
	if labels == nil {
		return
	}
	if cfg.ModelType != "" {
		labels["vision_model_type"] = normalizeROCmLabelToken(cfg.ModelType)
	}
	if cfg.DType != "" {
		labels["vision_dtype"] = rocmConfigDTypeQuantizationType(cfg.DType)
		if labels["vision_dtype"] == "" {
			labels["vision_dtype"] = core.Lower(cfg.DType)
		}
	}
	if cfg.ImageSize > 0 {
		labels["vision_image_size"] = core.Sprintf("%d", cfg.ImageSize)
	}
	if cfg.PatchSize > 0 {
		labels["vision_patch_size"] = core.Sprintf("%d", cfg.PatchSize)
	}
	if cfg.NumChannels > 0 {
		labels["vision_num_channels"] = core.Sprintf("%d", cfg.NumChannels)
	}
	if cfg.HiddenSize > 0 {
		labels["vision_hidden_size"] = core.Sprintf("%d", cfg.HiddenSize)
	}
	if cfg.IntermediateSize > 0 {
		labels["vision_intermediate_size"] = core.Sprintf("%d", cfg.IntermediateSize)
	}
	if cfg.NumHiddenLayers > 0 {
		labels["vision_num_hidden_layers"] = core.Sprintf("%d", cfg.NumHiddenLayers)
	}
	if cfg.NumAttentionHeads > 0 {
		labels["vision_attention_heads"] = core.Sprintf("%d", cfg.NumAttentionHeads)
	}
	if cfg.NumKeyValueHeads > 0 {
		labels["vision_kv_heads"] = core.Sprintf("%d", cfg.NumKeyValueHeads)
	}
	if cfg.HeadDim > 0 {
		labels["vision_head_dim"] = core.Sprintf("%d", cfg.HeadDim)
	}
	if cfg.GlobalHeadDim > 0 {
		labels["vision_global_head_dim"] = core.Sprintf("%d", cfg.GlobalHeadDim)
	}
	if cfg.PoolingKernelSize > 0 {
		labels["vision_pooling_kernel_size"] = core.Sprintf("%d", cfg.PoolingKernelSize)
	}
	if cfg.PositionEmbeddingSize > 0 {
		labels["vision_position_embedding_size"] = core.Sprintf("%d", cfg.PositionEmbeddingSize)
	}
	if cfg.HiddenActivation != "" {
		labels["vision_hidden_activation"] = cfg.HiddenActivation
	}
	if cfg.RMSNormEps > 0 {
		labels["vision_rms_norm_eps"] = formatROCmFloat(cfg.RMSNormEps)
	}
	if cfg.RoPEParameters.RopeTheta > 0 {
		labels["vision_rope_theta"] = formatROCmFloat(cfg.RoPEParameters.RopeTheta)
	}
	if cfg.RoPEParameters.RopeType != "" {
		labels["vision_rope_type"] = cfg.RoPEParameters.RopeType
	}
	labels["vision_standardize"] = core.Sprintf("%t", cfg.Standardize)
	labels["vision_use_clipped_linears"] = core.Sprintf("%t", cfg.UseClippedLinears)
}

func applyROCMAudioTowerLabels(labels map[string]string, cfg rocmModelPackAudioConfigProbe) {
	if labels == nil {
		return
	}
	if cfg.ModelType != "" {
		labels["audio_model_type"] = normalizeROCmLabelToken(cfg.ModelType)
	}
	if cfg.HiddenSize > 0 {
		labels["audio_hidden_size"] = core.Sprintf("%d", cfg.HiddenSize)
	}
	if cfg.AudioEmbedDim > 0 {
		labels["audio_embed_dim"] = core.Sprintf("%d", cfg.AudioEmbedDim)
	}
	if cfg.AudioSamplesPerToken > 0 {
		labels["audio_samples_per_token"] = core.Sprintf("%d", cfg.AudioSamplesPerToken)
	}
	if cfg.NumHiddenLayers > 0 {
		labels["audio_num_hidden_layers"] = core.Sprintf("%d", cfg.NumHiddenLayers)
	}
	if cfg.NumAttentionHeads > 0 {
		labels["audio_attention_heads"] = core.Sprintf("%d", cfg.NumAttentionHeads)
	}
	if cfg.AttentionChunkSize > 0 {
		labels["audio_attention_chunk_size"] = core.Sprintf("%d", cfg.AttentionChunkSize)
	}
	if cfg.AttentionContextLeft > 0 {
		labels["audio_attention_context_left"] = core.Sprintf("%d", cfg.AttentionContextLeft)
	}
	if cfg.AttentionContextRight > 0 {
		labels["audio_attention_context_right"] = core.Sprintf("%d", cfg.AttentionContextRight)
	}
	if cfg.AttentionLogitCap > 0 {
		labels["audio_attention_logit_cap"] = formatROCmFloat(cfg.AttentionLogitCap)
	}
	if cfg.AttentionInvalidLogitsValue != 0 {
		labels["audio_attention_invalid_logits_value"] = formatROCmFloat(cfg.AttentionInvalidLogitsValue)
	}
	if cfg.ConvKernelSize > 0 {
		labels["audio_conv_kernel_size"] = core.Sprintf("%d", cfg.ConvKernelSize)
	}
	if cfg.OutputProjDims > 0 {
		labels["audio_output_proj_dims"] = core.Sprintf("%d", cfg.OutputProjDims)
	}
	if cfg.RMSNormEps > 0 {
		labels["audio_rms_norm_eps"] = formatROCmFloat(cfg.RMSNormEps)
	}
	if cfg.GradientClipping > 0 {
		labels["audio_gradient_clipping"] = formatROCmFloat(cfg.GradientClipping)
	}
	if cfg.ResidualWeight > 0 {
		labels["audio_residual_weight"] = formatROCmFloat(cfg.ResidualWeight)
	}
	if cfg.HiddenAct != "" {
		labels["audio_hidden_act"] = cfg.HiddenAct
	}
	labels["audio_use_clipped_linears"] = core.Sprintf("%t", cfg.UseClippedLinears)
}

func normalizeROCmLabelToken(value string) string {
	return core.Replace(core.Lower(value), "-", "_")
}

func applyROCmAttentionConfigLabels(inspection *inference.ModelPackInspection, cfg rocmModelPackConfigProbe) {
	labels := rocmAttentionConfigLabels(cfg)
	if len(labels) == 0 {
		return
	}
	model := inspection.Model
	if model.Labels == nil {
		model.Labels = map[string]string{}
	}
	for key, value := range labels {
		inspection.Labels[key] = value
		model.Labels[key] = value
	}
	inspection.Model = model
}

func rocmAttentionConfigLabels(cfg rocmModelPackConfigProbe) map[string]string {
	out := map[string]string{}
	gemma4Architecture := isROCmGemma4Architecture(rocmConfigArchitecture(cfg))
	for key, value := range rocmDeepSeekMLALabels(cfg) {
		out[key] = value
	}
	if slidingWindow := firstPositiveInt(cfg.SlidingWindow, cfg.TextConfig.SlidingWindow); slidingWindow > 0 {
		out["sliding_window"] = core.Sprintf("%d", slidingWindow)
	}
	if pattern := firstPositiveInt(cfg.SlidingWindowPattern, cfg.TextConfig.SlidingWindowPattern); pattern > 0 {
		out["sliding_window_pattern"] = core.Sprintf("%d", pattern)
	}
	if kvSharedLayers, ok := rocmConfigKVSharedLayers(cfg); ok {
		out["attention_kv_shared_layers"] = core.Sprintf("%d", kvSharedLayers)
	}
	if gemma4Architecture {
		rocmApplyGemma4ConfigLabels(out, rocmGemma4TextConfigFromProbe(cfg))
	}
	attentionHeads := firstPositiveInt(cfg.NumAttentionHeads, cfg.TextConfig.NumAttentionHeads)
	kvHeads := firstPositiveInt(cfg.NumKeyValueHeads, cfg.TextConfig.NumKeyValueHeads)
	globalKVHeads := firstPositiveInt(cfg.NumGlobalKVHeads, cfg.TextConfig.NumGlobalKVHeads)
	headDim := firstPositiveInt(cfg.HeadDim, cfg.TextConfig.HeadDim)
	globalHeadDim := firstPositiveInt(cfg.GlobalHeadDim, cfg.TextConfig.GlobalHeadDim)
	if attentionHeads > 0 {
		out["attention_heads"] = core.Sprintf("%d", attentionHeads)
	}
	if kvHeads > 0 {
		out["attention_kv_heads"] = core.Sprintf("%d", kvHeads)
	}
	if globalKVHeads > 0 {
		out["attention_global_kv_heads"] = core.Sprintf("%d", globalKVHeads)
	}
	if cfg.AttentionKEqV || cfg.TextConfig.AttentionKEqV {
		out["attention_k_eq_v"] = "true"
	}
	if headDim > 0 {
		out["attention_head_dim"] = core.Sprintf("%d", headDim)
	}
	if globalHeadDim > 0 {
		out["attention_global_head_dim"] = core.Sprintf("%d", globalHeadDim)
	}
	if attentionHeads > 0 && headDim > 0 {
		out["attention_query_width"] = core.Sprintf("%d", attentionHeads*headDim)
	}
	if kvHeads > 0 && headDim > 0 {
		out["attention_kv_width"] = core.Sprintf("%d", kvHeads*headDim)
	}
	if globalKVHeads > 0 && globalHeadDim > 0 {
		out["attention_global_kv_width"] = core.Sprintf("%d", globalKVHeads*globalHeadDim)
	}
	if attentionHeads > 0 && kvHeads > 0 && attentionHeads != kvHeads {
		out["attention_gqa"] = "true"
	}
	if eps := firstPositiveFloat(cfg.RMSNormEps, cfg.TextConfig.RMSNormEps); eps > 0 {
		out["rms_norm_eps"] = formatROCmFloat(eps)
	}
	if cap := firstPositiveFloat(cfg.FinalLogitSoftcap, cfg.TextConfig.FinalLogitSoftcap); cap > 0 {
		out["final_logit_softcapping"] = formatROCmFloat(cap)
	}
	for layerType, params := range rocmNativeGemma4RoPEParameters(cfg) {
		labelType := core.Replace(layerType, "_attention", "")
		if params.RopeTheta > 0 {
			out["attention_rope_"+labelType+"_theta"] = formatROCmFloat(params.RopeTheta)
		}
		if params.PartialRotaryFactor > 0 {
			out["attention_rope_"+labelType+"_partial_rotary_factor"] = formatROCmFloat(params.PartialRotaryFactor)
		}
		if params.RopeType != "" {
			out["attention_rope_"+labelType+"_type"] = params.RopeType
		}
		if params.Factor > 0 {
			out["attention_rope_"+labelType+"_factor"] = formatROCmFloat(params.Factor)
		}
	}
	fullLayers := 0
	linearLayers := 0
	slidingLayers := 0
	layerTypes := rocmConfigLayerTypes(cfg)
	if len(layerTypes) > 0 {
		out["attention_layer_types"] = core.Join(",", layerTypes...)
	}
	sequenceLayerTypes, sequenceLayerTypesSource := rocmConfigSequenceMixerPlanLayerTypes(cfg)
	if len(sequenceLayerTypes) > 0 {
		rocmApplySequenceMixerConfigLabels(out, sequenceLayerTypes, sequenceLayerTypesSource)
	} else if sequenceLayerTypesSource, err := rocmConfigSequenceMixerPlanError(cfg); err != nil {
		rocmApplySequenceMixerConfigErrorLabels(out, sequenceLayerTypesSource, err)
	}
	for _, layerType := range layerTypes {
		lower := core.Lower(layerType)
		switch {
		case core.Contains(lower, "linear"):
			linearLayers++
		case core.Contains(lower, "sliding"):
			slidingLayers++
		case core.Contains(lower, "full"):
			fullLayers++
		}
	}
	if linearLayers > 0 {
		out["attention_linear_layers"] = core.Sprintf("%d", linearLayers)
	}
	if fullLayers > 0 {
		out["attention_full_layers"] = core.Sprintf("%d", fullLayers)
	}
	if slidingLayers > 0 {
		out["attention_sliding_layers"] = core.Sprintf("%d", slidingLayers)
	}
	architecture := normalizeROCmArchitecture(rocmConfigArchitecture(cfg))
	if architecture == "qwen3_6" || architecture == "qwen3_6_moe" {
		numLayers := firstPositiveInt(cfg.NumHiddenLayers, cfg.NumLayers, cfg.TextConfig.NumHiddenLayers, cfg.TextConfig.NumLayers)
		slidingWindow := firstPositiveInt(cfg.SlidingWindow, cfg.TextConfig.SlidingWindow)
		plan, err := BuildHybridAttentionCachePlan(numLayers, layerTypes, slidingWindow)
		if err != nil {
			return out
		}
		out["qwen36_hybrid_attention"] = "true"
		out["attention_cacheless_layers"] = core.Sprintf("%d", plan.CachelessLayers)
		out["qwen36_cacheless_layers"] = core.Sprintf("%d", plan.CachelessLayers)
		out["qwen36_hybrid_cache_plan"] = "metadata"
		out["qwen36_kv_cache_count"] = core.Sprintf("%d", plan.GlobalLayers)
		out["qwen36_cache_index_by_layer"] = plan.CacheIndexCSV()
		if slidingWindow > 0 {
			out["qwen36_local_window"] = core.Sprintf("%d", slidingWindow)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func rocmDeepSeekMLALabels(cfg rocmModelPackConfigProbe) map[string]string {
	if normalizeROCmArchitecture(rocmConfigArchitecture(cfg)) != "deepseek" {
		return nil
	}
	kvLoRARank := firstPositiveInt(cfg.KVLoRARank, cfg.TextConfig.KVLoRARank)
	qLoRARank := firstPositiveInt(cfg.QLoRARank, cfg.TextConfig.QLoRARank)
	qkNoPEHeadDim := firstPositiveInt(cfg.QKNoPEHeadDim, cfg.TextConfig.QKNoPEHeadDim)
	qkRoPEHeadDim := firstPositiveInt(cfg.QKRoPEHeadDim, cfg.TextConfig.QKRoPEHeadDim)
	qkHeadDim := firstPositiveInt(cfg.QKHeadDim, cfg.TextConfig.QKHeadDim)
	if qkHeadDim == 0 && (qkNoPEHeadDim > 0 || qkRoPEHeadDim > 0) {
		qkHeadDim = qkNoPEHeadDim + qkRoPEHeadDim
	}
	vHeadDim := firstPositiveInt(cfg.VHeadDim, cfg.TextConfig.VHeadDim)
	out := map[string]string{}
	if qLoRARank > 0 {
		out["deepseek_q_lora_rank"] = core.Sprintf("%d", qLoRARank)
	}
	if kvLoRARank > 0 {
		out["deepseek_kv_lora_rank"] = core.Sprintf("%d", kvLoRARank)
	}
	if qkNoPEHeadDim > 0 {
		out["deepseek_qk_nope_head_dim"] = core.Sprintf("%d", qkNoPEHeadDim)
	}
	if qkRoPEHeadDim > 0 {
		out["deepseek_qk_rope_head_dim"] = core.Sprintf("%d", qkRoPEHeadDim)
	}
	if qkHeadDim > 0 {
		out["deepseek_qk_head_dim"] = core.Sprintf("%d", qkHeadDim)
	}
	if vHeadDim > 0 {
		out["deepseek_v_head_dim"] = core.Sprintf("%d", vHeadDim)
	}
	if len(out) == 0 {
		return nil
	}
	out["deepseek_mla"] = "true"
	if kvLoRARank > 0 && qkNoPEHeadDim > 0 && qkRoPEHeadDim > 0 && qkHeadDim == qkNoPEHeadDim+qkRoPEHeadDim && vHeadDim > 0 {
		out["deepseek_mla_valid"] = "true"
	} else {
		out["deepseek_mla_valid"] = "false"
	}
	return out
}

func rocmQuantConfigBits(quant rocmQuantizationConfigProbe) int {
	if quant.Bits > 0 {
		return quant.Bits
	}
	if quant.LoadIn4Bit {
		return 4
	}
	if quant.LoadIn8Bit {
		return 8
	}
	return 0
}

func rocmQuantConfigType(quant rocmQuantizationConfigProbe) string {
	return normalizeROCmQuantizationAlias(firstNonEmptyString(quant.Algorithm, quant.QuantMethod, quant.WeightFormat, quant.Format, quant.Type))
}

func applyROCMAutoRoundQuantizationLabels(inspection *inference.ModelPackInspection, quant rocmQuantizationConfigProbe) {
	if inspection == nil || !rocmQuantConfigIsAutoRound(quant) {
		return
	}
	inspection.Labels["autoround_quantization"] = "true"
	inspection.Labels["autoround_runtime"] = "planned_hip"
	inspection.Labels["autoround_hip_kernel"] = hipKernelStatusNotLinked
	if method := rocmQuantConfigType(quant); method != "" {
		inspection.Labels["autoround_algorithm"] = method
	}
	if quant.Format != "" {
		inspection.Labels["autoround_format"] = normalizeROCmQuantizationAlias(quant.Format)
	}
	if quant.WeightFormat != "" {
		inspection.Labels["autoround_weight_format"] = normalizeROCmQuantizationAlias(quant.WeightFormat)
	}
	if quant.Scheme != "" {
		inspection.Labels["autoround_scheme"] = core.Trim(quant.Scheme)
	}
	if quant.Bits > 0 {
		inspection.Labels["autoround_bits"] = core.Sprintf("%d", quant.Bits)
	}
	if quant.GroupSize > 0 {
		inspection.Labels["autoround_group_size"] = core.Sprintf("%d", quant.GroupSize)
	}
	if quant.Iters > 0 {
		inspection.Labels["autoround_iters"] = core.Sprintf("%d", quant.Iters)
	}
	if quant.NSamples > 0 {
		inspection.Labels["autoround_nsamples"] = core.Sprintf("%d", quant.NSamples)
	}
	if quant.SeqLen > 0 {
		inspection.Labels["autoround_seqlen"] = core.Sprintf("%d", quant.SeqLen)
	}
	if quant.Sym != nil {
		inspection.Labels["autoround_sym"] = boolLabel(*quant.Sym)
	}
	if quant.Asym != nil {
		inspection.Labels["autoround_asym"] = boolLabel(*quant.Asym)
	}
	if profile, ok := rocmAutoRoundProfileForQuantConfig(quant); ok {
		inspection.Labels["autoround_profile"] = profile.Name
		inspection.Labels["autoround_profile_role"] = profile.ProductRole
		inspection.Labels["autoround_profile_matched"] = "true"
		inspection.Labels["autoround_profile_requires_bench"] = boolLabel(profile.RequiresBench)
		inspection.Labels["autoround_profile_requires_calibration"] = boolLabel(profile.RequiresCalibration)
	}
	if plan, ok := rocmAutoRoundCalibrationPlanForQuantConfig(quant); ok {
		ApplyProductionAutoRoundCalibrationPlanLabels(inspection.Labels, plan)
	}
}

func rocmQuantConfigIsAutoRound(quant rocmQuantizationConfigProbe) bool {
	return rocmQuantizationAliasIsAutoRound(quant.Algorithm, quant.QuantMethod, quant.WeightFormat, quant.Format, quant.Type)
}

func rocmAutoRoundProfileForQuantConfig(quant rocmQuantizationConfigProbe) (ProductionAutoRoundQuantizationProfile, bool) {
	return productionAutoRoundQuantizationProfileForFields(quant.Scheme, firstNonEmptyString(quant.WeightFormat, quant.Format), quant.GroupSize)
}

func rocmAutoRoundCalibrationPlanForQuantConfig(quant rocmQuantizationConfigProbe) (ProductionAutoRoundCalibrationPlan, bool) {
	profile, ok := rocmAutoRoundProfileForQuantConfig(quant)
	if !ok {
		return ProductionAutoRoundCalibrationPlan{}, false
	}
	return productionAutoRoundCalibrationPlan(profile, quant.NSamples, quant.SeqLen, quant.Iters), true
}

func rocmConfigDTypeQuantizationType(dtype string) string {
	switch core.Lower(dtype) {
	case "bfloat16", "bf16":
		return "bf16"
	case "float16", "fp16", "f16":
		return "f16"
	case "float32", "fp32", "f32":
		return "f32"
	default:
		return ""
	}
}

func applyROCmGGUFInspection(inspection *inference.ModelPackInspection, path string) bool {
	info, err := gguf.ReadInfo(path)
	if err != nil {
		inspection.Notes = append(inspection.Notes, "GGUF metadata could not be parsed: "+err.Error())
		return false
	}
	metadata := info.Metadata
	model := inspection.Model
	model.Path = path
	model.Architecture = firstNonEmptyString(model.Architecture, normalizeROCmArchitecture(metadata.Architecture))
	model.ContextLength = firstPositiveInt(model.ContextLength, int(metadata.ContextLength))
	model.NumLayers = firstPositiveInt(model.NumLayers, int(metadata.BlockCount))
	bits, group := quantisationFromFileType(metadata.FileType)
	model.QuantBits = firstPositiveInt(model.QuantBits, bits)
	model.QuantGroup = firstPositiveInt(model.QuantGroup, group)
	model.QuantType = firstNonEmptyString(model.QuantType, core.Lower(gguf.FileTypeName(metadata.FileType)))
	inspection.Model = model
	inspection.Labels["gguf_tensors"] = core.Sprintf("%d", len(info.Tensors))
	inspection.Labels["gguf_alignment"] = core.Sprintf("%d", info.Alignment)
	if metadata.FileSize > 0 {
		inspection.Labels["weight_bytes"] = core.Sprintf("%d", metadata.FileSize)
	}
	return true
}

func applyROCmSafetensorsInspection(inspection *inference.ModelPackInspection, path string) bool {
	summary, err := readROCmSafetensorsSummary(path)
	if err != nil {
		inspection.Notes = append(inspection.Notes, "safetensors header could not be parsed: "+err.Error())
		return false
	}
	model := inspection.Model
	model.Path = firstNonEmptyString(model.Path, path)
	inspection.Model = model
	mergeROCmSafetensorsSummaryLabels(inspection.Labels, summary)
	if inspection.Model.Architecture == "" {
		applyROCmDenseSafetensorsArchitectureInference(inspection, path)
	}
	if err := applyROCmMiniMaxM2SafetensorsPlanLabels(inspection, path); err != nil {
		inspection.Notes = append(inspection.Notes, "MiniMax M2 safetensors staged plan could not be validated: "+err.Error())
		return false
	}
	if err := applyROCmQwen3SafetensorsPlanLabels(inspection, path); err != nil {
		inspection.Notes = append(inspection.Notes, "Qwen3 safetensors staged plan could not be validated: "+err.Error())
		return false
	}
	return true
}

func applyROCmDenseSafetensorsArchitectureInference(inspection *inference.ModelPackInspection, path string) {
	tensors, err := readROCmSafetensorsNativeTensors(path)
	if err != nil {
		inspection.Notes = append(inspection.Notes, "dense safetensors architecture inference could not read tensor names: "+err.Error())
		return
	}
	names := make(map[string]bool, len(tensors))
	for _, tensor := range tensors {
		names[tensor.Name] = true
	}
	architecture := DetectDenseModelType(nil, names)
	if architecture == "" || architecture == "qwen2" {
		return
	}
	inspection.Model.Architecture = architecture
	inspection.Labels["architecture_inferred_from_weights"] = "true"
	inspection.Labels["architecture_inference_source"] = "dense_weight_names"
}

func applyROCmMiniMaxM2SafetensorsPlanLabels(inspection *inference.ModelPackInspection, path string) error {
	if inspection == nil || normalizeROCmArchitecture(inspection.Model.Architecture) != "minimax_m2" {
		return nil
	}
	tensors, err := readROCmSafetensorsNativeTensors(path)
	if err != nil {
		return err
	}
	names := make(map[string]bool, len(tensors))
	for _, tensor := range tensors {
		names[tensor.Name] = true
	}
	missing := rocmMiniMaxM2MissingLayer0TensorNames(names, inspection.Labels["minimax_m2_routing_bias"] == "true")
	inspection.Labels["minimax_m2_layer0_required_tensor_count"] = core.Sprintf("%d", len(rocmMiniMaxM2Layer0RequiredTensorCandidates(inspection.Labels["minimax_m2_routing_bias"] == "true")))
	if len(missing) == 0 {
		inspection.Labels["minimax_m2_layer0_skeleton"] = "present"
		return nil
	}
	inspection.Labels["minimax_m2_layer0_skeleton"] = "missing"
	inspection.Labels["minimax_m2_layer0_missing_tensors"] = core.Join(",", missing...)
	return nil
}

func rocmMiniMaxM2MissingLayer0TensorNames(names map[string]bool, routingBias bool) []string {
	var missing []string
	for _, candidates := range rocmMiniMaxM2Layer0RequiredTensorCandidates(routingBias) {
		if rocmAnyTensorNamePresent(names, candidates) {
			continue
		}
		missing = append(missing, candidates[0])
	}
	slices.Sort(missing)
	return missing
}

func rocmMiniMaxM2Layer0RequiredTensorCandidates(routingBias bool) [][]string {
	required := [][]string{
		{"model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.qkv_proj.weight"},
		{"model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.qkv_proj.weight"},
		{"model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.qkv_proj.weight"},
		{"model.layers.0.self_attn.o_proj.weight"},
		{"model.layers.0.block_sparse_moe.gate.weight"},
		{"model.layers.0.block_sparse_moe.experts.0.gate_proj.weight", "model.layers.0.mlp.experts.0.gate_proj.weight"},
		{"model.layers.0.block_sparse_moe.experts.0.up_proj.weight", "model.layers.0.mlp.experts.0.up_proj.weight"},
		{"model.layers.0.block_sparse_moe.experts.0.down_proj.weight", "model.layers.0.mlp.experts.0.down_proj.weight"},
	}
	if routingBias {
		required = append(required, []string{"model.layers.0.block_sparse_moe.e_score_correction_bias"})
	}
	return required
}

func rocmAnyTensorNamePresent(names map[string]bool, candidates []string) bool {
	for _, candidate := range candidates {
		if names[candidate] {
			return true
		}
	}
	return false
}

func applyROCmQwen3SafetensorsPlanLabels(inspection *inference.ModelPackInspection, path string) error {
	if inspection == nil || !rocmQwen3DenseArchitecture(inspection.Model.Architecture) {
		return nil
	}
	tensors, err := readROCmSafetensorsNativeTensors(path)
	if err != nil {
		return err
	}
	names := make(map[string]bool, len(tensors))
	for _, tensor := range tensors {
		names[tensor.Name] = true
	}
	required := []string{
		"model.layers.0.self_attn.q_norm.weight",
		"model.layers.0.self_attn.k_norm.weight",
	}
	var missing []string
	for _, name := range required {
		if !HasResolvedDenseWeightName(names, name) {
			missing = append(missing, name)
		}
	}
	if len(missing) == len(required) {
		return nil
	}
	inspection.Labels["qwen3_attention_qk_norm"] = "true"
	inspection.Labels["qwen3_qk_norm_required_tensor_count"] = core.Sprintf("%d", len(required))
	inspection.Labels["qwen3_q_norm_tensor"] = required[0]
	inspection.Labels["qwen3_k_norm_tensor"] = required[1]
	if len(missing) == 0 {
		inspection.Labels["qwen3_qk_norm_skeleton"] = "present"
		return nil
	}
	inspection.Labels["qwen3_qk_norm_skeleton"] = "missing"
	inspection.Labels["qwen3_qk_norm_missing_tensors"] = core.Join(",", missing...)
	return nil
}

func rocmQwen3DenseArchitecture(architecture string) bool {
	switch normalizeROCmArchitecture(architecture) {
	case "qwen3", "qwen3_next":
		return true
	default:
		return false
	}
}

func applyROCmSafetensorsIndexInspection(inspection *inference.ModelPackInspection, root string, weights []string) (bool, error) {
	path := core.PathJoin(root, "model.safetensors.index.json")
	read := core.ReadFile(path)
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return true, nil
		}
		return false, read.Value.(error)
	}
	var index rocmSafetensorsIndexProbe
	if result := core.JSONUnmarshal(read.Value.([]byte), &index); !result.OK {
		return false, result.Value.(error)
	}
	if len(index.WeightMap) == 0 {
		return false, core.NewError("safetensors index weight_map is empty")
	}
	knownShards := map[string]bool{}
	safetensorsWeightCount := 0
	for _, weight := range weights {
		if core.Lower(core.PathExt(weight)) != ".safetensors" {
			continue
		}
		safetensorsWeightCount++
		knownShards[core.PathBase(weight)] = true
	}
	referencedShards := map[string]bool{}
	for tensorName, shard := range index.WeightMap {
		if core.Trim(tensorName) == "" || core.Trim(shard) == "" {
			return false, core.NewError("safetensors index contains an empty tensor or shard entry")
		}
		shardBase := core.PathBase(shard)
		if !knownShards[shardBase] {
			return false, core.NewError("safetensors index references missing shard " + shard)
		}
		referencedShards[shardBase] = true
	}
	if safetensorsWeightCount != len(referencedShards) {
		return false, core.NewError(core.Sprintf("safetensors index references %d shard files but %d safetensors files were discovered", len(referencedShards), safetensorsWeightCount))
	}
	inspection.Labels["safetensors_index"] = "present"
	inspection.Labels["safetensors_index_tensors"] = core.Sprintf("%d", len(index.WeightMap))
	inspection.Labels["safetensors_index_shards"] = core.Sprintf("%d", len(referencedShards))
	if len(referencedShards) > 1 {
		inspection.Labels["sharded_safetensors"] = "true"
	}
	if index.Metadata.TotalSize > 0 {
		inspection.Labels["safetensors_index_total_size"] = core.FormatUint(index.Metadata.TotalSize, 10)
		inspection.Labels["weight_bytes"] = core.FormatUint(index.Metadata.TotalSize, 10)
	}
	if index.Metadata.TotalParameters > 0 {
		inspection.Labels["safetensors_index_total_parameters"] = core.FormatUint(index.Metadata.TotalParameters, 10)
	}
	return true, nil
}

func (b *rocmBackend) safetensorsNativeLoadConfig(ctx context.Context, path string, loadConfig inference.LoadConfig) (string, nativeLoadConfig, error) {
	inspection, err := b.InspectModelPack(ctx, path)
	if err != nil {
		return "", nativeLoadConfig{}, err
	}
	if inspection.Format != "safetensors" {
		return "", nativeLoadConfig{}, core.NewError("native safetensors load requires a safetensors model pack")
	}
	if !inspection.Supported {
		return "", nativeLoadConfig{}, core.NewError("model pack is not supported for native ROCm load")
	}
	weightPaths, err := rocmSafetensorsWeightFiles(path)
	if err != nil {
		return "", nativeLoadConfig{}, err
	}
	tensors := []nativeTensorInfo{}
	for _, weightPath := range weightPaths {
		weightTensors, err := readROCmSafetensorsNativeTensors(weightPath)
		if err != nil {
			return "", nativeLoadConfig{}, err
		}
		tensors = append(tensors, weightTensors...)
	}
	sequenceMixerPlan, err := sequenceMixerLoadPlanFromInspection(inspection, tensors)
	if err != nil {
		return "", nativeLoadConfig{}, core.E("rocm.safetensorsNativeLoadConfig", "build sequence mixer load plan", err)
	}
	loadPath := path
	if len(weightPaths) == 1 {
		loadPath = weightPaths[0]
	}
	cfg := nativeLoadConfig{
		ContextSize:        resolveModelContextLength(loadConfig.ContextLen, inspection.Model.ContextLength),
		GPULayerCount:      loadConfig.GPULayers,
		ParallelSlotCount:  loadConfig.ParallelSlots,
		AdapterPath:        loadConfig.AdapterPath,
		ModelInfo:          modelInfoFromIdentity(inspection.Model),
		ModelLabels:        cloneStringMap(inspection.Labels),
		SequenceMixerPlan:  sequenceMixerPlan,
		TokenizerPath:      inspection.Tokenizer.Path,
		Gemma4TextConfig:   rocmNativeGemma4TextConfig(path),
		Tensors:            tensors,
		TiedWordEmbeddings: inspection.Labels["tied_word_embeddings"] == "true",
	}
	declaredVision := cfg.Gemma4TextConfig.Vision
	declaredAudio := cfg.Gemma4TextConfig.Audio
	cfg.Gemma4TextConfig.Vision = declaredVision && rocmNativeTensorPlanHasVision(tensors)
	cfg.Gemma4TextConfig.Audio = declaredAudio && rocmNativeTensorPlanHasAudio(tensors)
	if declaredVision && !cfg.Gemma4TextConfig.Vision {
		cfg.ModelLabels["vision_runtime"] = hipKernelStatusNotLinked
		cfg.ModelLabels["vision_projector_runtime"] = hipKernelStatusNotLinked
		cfg.ModelLabels["vision_payload"] = "absent"
	}
	if declaredAudio && !cfg.Gemma4TextConfig.Audio {
		cfg.ModelLabels["audio_runtime"] = hipKernelStatusNotLinked
		cfg.ModelLabels["audio_projector_runtime"] = hipKernelStatusNotLinked
		cfg.ModelLabels["audio_frontend_runtime"] = hipKernelStatusNotLinked
		cfg.ModelLabels["audio_payload"] = "absent"
	}
	// Gemma4 checkpoints may carry their text model and media towers in the
	// same safetensors pack. Keep the pack root as the default tower source;
	// explicit ROCm load paths and environment overrides are applied later.
	if cfg.Gemma4TextConfig.Vision || cfg.Gemma4TextConfig.Audio {
		mediaRoot, rootErr := rocmModelPackRoot(path)
		if rootErr != nil {
			return "", nativeLoadConfig{}, core.E("rocm.safetensorsNativeLoadConfig", "resolve attached media root", rootErr)
		}
		if cfg.Gemma4TextConfig.Vision {
			cfg.VisionModelPath = mediaRoot
		}
		if cfg.Gemma4TextConfig.Audio {
			cfg.AudioModelPath = mediaRoot
		}
	}
	if isROCmGemma4Architecture(inspection.Model.Architecture) {
		cfg.Gemma4Architecture, err = resolveGemma4ModelPackArchitectureDeclaration(path)
		if err != nil {
			return "", nativeLoadConfig{}, core.E("rocm.safetensorsNativeLoadConfig", "resolve shared Gemma4 architecture", err)
		}
	}
	if len(weightPaths) == 1 && len(tensors) > 0 {
		cfg.DataOffset = tensors[0].DataOffset
	}
	return loadPath, cfg, nil
}

func rocmNativeTensorPlanHasVision(tensors []nativeTensorInfo) bool {
	for _, tensor := range tensors {
		name := strings.ToLower(tensor.Name)
		if strings.Contains(name, "vision_embedder.") ||
			strings.Contains(name, "embed_vision.") ||
			strings.Contains(name, "vision_tower.") ||
			strings.Contains(name, "vision_model.") {
			return true
		}
	}
	return false
}

func rocmNativeTensorPlanHasAudio(tensors []nativeTensorInfo) bool {
	for _, tensor := range tensors {
		name := strings.ToLower(tensor.Name)
		if strings.Contains(name, "audio_tower.") || strings.Contains(name, "embed_audio.") {
			return true
		}
	}
	return false
}

func rocmModelPackRoot(path string) (string, error) {
	return rocmmodel.ResolveModelPackRoot(path)
}

func rocmSafetensorsWeightFiles(path string) ([]string, error) {
	manifest, err := rocmmodel.InspectModelPackFiles(path)
	if err != nil {
		return nil, err
	}
	safetensors := []string{}
	for _, weight := range manifest.WeightFiles {
		if weight.Format == rocmmodel.ModelPackFormatSafetensors {
			safetensors = append(safetensors, weight.Path)
		}
	}
	if len(safetensors) == 0 {
		return nil, core.NewError("native safetensors load requires at least one safetensors weight file")
	}
	return safetensors, nil
}

func readROCmSafetensorsNativeTensors(path string) ([]nativeTensorInfo, error) {
	stat := core.Stat(path)
	if !stat.OK {
		return nil, stat.Value.(error)
	}
	fileSize := stat.Value.(core.FsFileInfo).Size()
	open := core.Open(path)
	if !open.OK {
		return nil, open.Value.(error)
	}
	file := open.Value.(*core.OSFile)
	defer file.Close()
	var headerLength uint64
	if err := binary.Read(file, binary.LittleEndian, &headerLength); err != nil {
		return nil, err
	}
	if headerLength == 0 || headerLength > maxSafetensorsHeaderBytes {
		return nil, core.NewError(core.Sprintf("safetensors header length %d is outside supported bounds", headerLength))
	}
	dataOffset := int64(8 + headerLength)
	if fileSize < dataOffset {
		return nil, core.NewError(core.Sprintf("safetensors file size %d is smaller than header span %d", fileSize, dataOffset))
	}
	header := make([]byte, int(headerLength))
	if _, err := io.ReadFull(file, header); err != nil {
		return nil, err
	}
	if err := rejectDuplicateROCmSafetensorsHeaderKeys(header); err != nil {
		return nil, err
	}
	tensors := map[string]rocmSafetensorsTensor{}
	if result := core.JSONUnmarshal(header, &tensors); !result.OK {
		return nil, result.Value.(error)
	}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		if name != "__metadata__" {
			names = append(names, name)
		}
	}
	slices.Sort(names)
	out := make([]nativeTensorInfo, 0, len(names))
	payloadBytes := uint64(fileSize - dataOffset)
	for _, name := range names {
		tensor := tensors[name]
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
		out = append(out, nativeTensorInfo{
			Name:       name,
			Dimensions: append([]uint64(nil), tensor.Shape...),
			Type:       tensorType,
			TypeName:   tensor.DType,
			SourcePath: path,
			DataOffset: dataOffset,
			Offset:     tensor.DataOffsets[0],
			ByteSize:   tensor.DataOffsets[1] - tensor.DataOffsets[0],
		})
	}
	if len(out) == 0 {
		return nil, core.NewError("safetensors header contains no tensor entries")
	}
	return out, nil
}

func rocmSafetensorsNativeTensorType(dtype string) (uint32, bool) {
	switch core.Upper(dtype) {
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

func mergeROCmSafetensorsSummaryLabels(labels map[string]string, summary rocmSafetensorsSummary) {
	labels["safetensors_tensors"] = core.FormatUint(rocmLabelUint(labels["safetensors_tensors"])+uint64(summary.TensorCount), 10)
	labels["safetensors_header_bytes"] = core.FormatUint(rocmLabelUint(labels["safetensors_header_bytes"])+summary.HeaderBytes, 10)
	labels["safetensors_payload_bytes"] = core.FormatUint(rocmLabelUint(labels["safetensors_payload_bytes"])+summary.PayloadBytes, 10)
	labels["weight_bytes"] = core.FormatUint(rocmLabelUint(labels["weight_bytes"])+summary.PayloadBytes, 10)
	dtypes := map[string]bool{}
	if existing := labels["safetensors_dtypes"]; existing != "" {
		for _, dtype := range core.Split(existing, ",") {
			if dtype != "" {
				dtypes[dtype] = true
			}
		}
	}
	for _, dtype := range summary.DTypes {
		if dtype != "" {
			dtypes[dtype] = true
		}
	}
	if len(dtypes) == 0 {
		return
	}
	values := make([]string, 0, len(dtypes))
	for dtype := range dtypes {
		values = append(values, dtype)
	}
	slices.Sort(values)
	labels["safetensors_dtypes"] = core.Join(",", values...)
}

func rocmLabelUint(value string) uint64 {
	if value == "" {
		return 0
	}
	parsed := core.ParseInt(value, 10, 64)
	if !parsed.OK {
		return 0
	}
	if parsed.Value.(int64) < 0 {
		return 0
	}
	return uint64(parsed.Value.(int64))
}

func clearROCmWeightMetadataLabels(labels map[string]string) {
	for _, key := range []string{
		"gguf_tensors",
		"gguf_alignment",
		"safetensors_tensors",
		"safetensors_header_bytes",
		"safetensors_payload_bytes",
		"safetensors_dtypes",
		"safetensors_index",
		"safetensors_index_tensors",
		"safetensors_index_shards",
		"safetensors_index_total_size",
		"safetensors_index_total_parameters",
		"sharded_safetensors",
		"weight_bytes",
	} {
		delete(labels, key)
	}
}

func applyROCmTokenizerJSONInspection(inspection *inference.ModelPackInspection, root string) error {
	path := core.PathJoin(root, "tokenizer.json")
	read := core.ReadFile(path)
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return nil
		}
		return read.Value.(error)
	}
	var probe rocmTokenizerJSONProbe
	if result := core.JSONUnmarshal(read.Value.([]byte), &probe); !result.OK {
		return result.Value.(error)
	}
	tokenizer := inspection.Tokenizer
	tokenizer.Path = firstNonEmptyString(tokenizer.Path, path)
	tokenizer.Kind = firstNonEmptyString(tokenizer.Kind, probe.Model.Type, "tokenizer.json")
	inspection.Tokenizer = tokenizer
	inspection.Labels["tokenizer_json"] = "present"
	if probe.Model.Type != "" {
		inspection.Labels["tokenizer_json_model"] = probe.Model.Type
	}
	appendROCmInspectionCapability(inspection, inference.ExperimentalCapability(inference.CapabilityTokenizer, inference.CapabilityGroupModel, "tokenizer sidecar metadata is present; native tokenizer loading is pending"))
	return nil
}

func applyROCmTokenizerConfigInspection(inspection *inference.ModelPackInspection, root string) error {
	path := core.PathJoin(root, "tokenizer_config.json")
	read := core.ReadFile(path)
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return nil
		}
		return read.Value.(error)
	}
	var probe rocmTokenizerConfigProbe
	if result := core.JSONUnmarshal(read.Value.([]byte), &probe); !result.OK {
		return result.Value.(error)
	}
	tokenizer := inspection.Tokenizer
	tokenizer.Path = firstNonEmptyString(tokenizer.Path, path)
	tokenizer.Kind = firstNonEmptyString(probe.TokenizerClass, tokenizer.Kind, "tokenizer_config.json")
	tokenizer.ChatTemplate = firstNonEmptyString(tokenizer.ChatTemplate, probe.ChatTemplate)
	tokenizer.BOSID = firstNonZeroInt32(tokenizer.BOSID, probe.BOSID.First())
	tokenizer.EOSID = firstNonZeroInt32(tokenizer.EOSID, probe.EOSID.First())
	tokenizer.PADID = firstNonZeroInt32(tokenizer.PADID, probe.PADID.First())
	inspection.Tokenizer = tokenizer
	inspection.Labels["tokenizer_config"] = "present"
	if probe.ModelMaxLength.Value > 0 {
		model := inspection.Model
		model.ContextLength = firstPositiveInt(model.ContextLength, probe.ModelMaxLength.Value)
		inspection.Model = model
		inspection.Labels["tokenizer_model_max_length"] = core.Sprintf("%d", probe.ModelMaxLength.Value)
	}
	appendROCmInspectionCapability(inspection, inference.ExperimentalCapability(inference.CapabilityTokenizer, inference.CapabilityGroupModel, "tokenizer sidecar metadata is present; native tokenizer loading is pending"))
	if probe.ChatTemplate != "" {
		inspection.Labels["chat_template"] = "present"
		appendROCmInspectionCapability(inspection, inference.ExperimentalCapability(inference.CapabilityChatTemplate, inference.CapabilityGroupModel, "chat template metadata is present; native template parser loading is pending"))
	}
	return nil
}

func readROCmSafetensorsSummary(path string) (rocmSafetensorsSummary, error) {
	stat := core.Stat(path)
	if !stat.OK {
		return rocmSafetensorsSummary{}, stat.Value.(error)
	}
	fileSize := stat.Value.(core.FsFileInfo).Size()
	open := core.Open(path)
	if !open.OK {
		return rocmSafetensorsSummary{}, open.Value.(error)
	}
	file := open.Value.(*core.OSFile)
	defer file.Close()
	var headerLength uint64
	if err := binary.Read(file, binary.LittleEndian, &headerLength); err != nil {
		return rocmSafetensorsSummary{}, err
	}
	if headerLength == 0 || headerLength > maxSafetensorsHeaderBytes {
		return rocmSafetensorsSummary{}, core.NewError(core.Sprintf("safetensors header length %d is outside supported bounds", headerLength))
	}
	payloadOffset := int64(8 + headerLength)
	if fileSize < payloadOffset {
		return rocmSafetensorsSummary{}, core.NewError(core.Sprintf("safetensors file size %d is smaller than header span %d", fileSize, payloadOffset))
	}
	payloadBytes := uint64(fileSize - payloadOffset)
	header := make([]byte, int(headerLength))
	if _, err := io.ReadFull(file, header); err != nil {
		return rocmSafetensorsSummary{}, err
	}
	if err := rejectDuplicateROCmSafetensorsHeaderKeys(header); err != nil {
		return rocmSafetensorsSummary{}, err
	}
	tensors := map[string]rocmSafetensorsTensor{}
	if result := core.JSONUnmarshal(header, &tensors); !result.OK {
		return rocmSafetensorsSummary{}, result.Value.(error)
	}
	summary := rocmSafetensorsSummary{HeaderBytes: headerLength}
	dtypeSeen := map[string]bool{}
	payloadRanges := []rocmSafetensorsPayloadRange{}
	for name, tensor := range tensors {
		if name == "__metadata__" {
			continue
		}
		if tensor.DType == "" {
			return rocmSafetensorsSummary{}, core.NewError("safetensors tensor " + name + " is missing dtype")
		}
		if tensor.Shape == nil {
			return rocmSafetensorsSummary{}, core.NewError("safetensors tensor " + name + " is missing shape")
		}
		dtypeBytes, ok := rocmSafetensorsDTypeBytes(tensor.DType)
		if !ok {
			return rocmSafetensorsSummary{}, core.NewError("safetensors tensor " + name + " has unsupported dtype " + tensor.DType)
		}
		summary.TensorCount++
		if tensor.DType != "" && !dtypeSeen[tensor.DType] {
			dtypeSeen[tensor.DType] = true
			summary.DTypes = append(summary.DTypes, tensor.DType)
		}
		if len(tensor.DataOffsets) != 2 {
			return rocmSafetensorsSummary{}, core.NewError("safetensors tensor " + name + " has invalid data_offsets")
		}
		if tensor.DataOffsets[1] < tensor.DataOffsets[0] {
			return rocmSafetensorsSummary{}, core.NewError("safetensors tensor " + name + " has reversed data_offsets")
		}
		if tensor.DataOffsets[1] > payloadBytes {
			return rocmSafetensorsSummary{}, core.NewError(core.Sprintf("safetensors tensor %s data_offsets end %d exceeds payload bytes %d", name, tensor.DataOffsets[1], payloadBytes))
		}
		shapeBytes, err := rocmSafetensorsShapeBytes(tensor.Shape, dtypeBytes)
		if err != nil {
			return rocmSafetensorsSummary{}, core.NewError("safetensors tensor " + name + " " + err.Error())
		}
		span := tensor.DataOffsets[1] - tensor.DataOffsets[0]
		if span != shapeBytes {
			return rocmSafetensorsSummary{}, core.NewError(core.Sprintf("safetensors tensor %s byte span %d does not match shape bytes %d", name, span, shapeBytes))
		}
		for _, existing := range payloadRanges {
			if tensor.DataOffsets[0] < existing.End && existing.Start < tensor.DataOffsets[1] {
				return rocmSafetensorsSummary{}, core.NewError(core.Sprintf("safetensors tensor %s data_offsets overlaps tensor %s", name, existing.Name))
			}
		}
		payloadRanges = append(payloadRanges, rocmSafetensorsPayloadRange{
			Name:  name,
			Start: tensor.DataOffsets[0],
			End:   tensor.DataOffsets[1],
		})
		if tensor.DataOffsets[1] > summary.PayloadBytes {
			summary.PayloadBytes = tensor.DataOffsets[1]
		}
	}
	if summary.TensorCount == 0 {
		return rocmSafetensorsSummary{}, core.NewError("safetensors header contains no tensor entries")
	}
	slices.Sort(summary.DTypes)
	return summary, nil
}

func rejectDuplicateROCmSafetensorsHeaderKeys(header []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(header))
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok || delim != '{' {
		return core.NewError("safetensors header must be a JSON object")
	}
	seen := map[string]bool{}
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		key, ok := token.(string)
		if !ok {
			return core.NewError("safetensors header key must be a string")
		}
		if seen[key] {
			return core.NewError("safetensors header contains duplicate tensor key " + key)
		}
		seen[key] = true
		if err := skipROCmJSONValue(decoder); err != nil {
			return err
		}
	}
	token, err = decoder.Token()
	if err != nil {
		return err
	}
	delim, ok = token.(json.Delim)
	if !ok || delim != '}' {
		return core.NewError("safetensors header object is not closed")
	}
	if _, err := decoder.Token(); err != io.EOF {
		if err != nil {
			return err
		}
		return core.NewError("safetensors header contains trailing JSON data")
	}
	return nil
}

func skipROCmJSONValue(decoder *json.Decoder) error {
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		for decoder.More() {
			if _, err := decoder.Token(); err != nil {
				return err
			}
			if err := skipROCmJSONValue(decoder); err != nil {
				return err
			}
		}
		token, err = decoder.Token()
		if err != nil {
			return err
		}
		delim, ok = token.(json.Delim)
		if !ok || delim != '}' {
			return core.NewError("JSON object is not closed")
		}
	case '[':
		for decoder.More() {
			if err := skipROCmJSONValue(decoder); err != nil {
				return err
			}
		}
		token, err = decoder.Token()
		if err != nil {
			return err
		}
		delim, ok = token.(json.Delim)
		if !ok || delim != ']' {
			return core.NewError("JSON array is not closed")
		}
	default:
		return core.NewError("unexpected JSON delimiter")
	}
	return nil
}

func rocmSafetensorsDTypeBytes(dtype string) (uint64, bool) {
	upper := core.Upper(dtype)
	switch upper {
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

func readROCmJANGConfig(root string) (*rocmJANGQuantizationInfo, error) {
	return jang.ReadConfig(root)
}

func applyROCmJANGInspection(inspection *inference.ModelPackInspection, jang rocmJANGQuantizationInfo) {
	model := inspection.Model
	model.Architecture = firstNonEmptyString(model.Architecture, normalizeROCmArchitecture(jang.SourceArchitecture))
	model.QuantBits = firstPositiveInt(model.QuantBits, jang.BitsDefault)
	model.QuantGroup = firstPositiveInt(model.QuantGroup, jang.GroupSize)
	model.QuantType = firstNonEmptyString(model.QuantType, rocmJANGQuantizationType(jang))
	inspection.Model = model
	inspection.Labels["jang_profile"] = jang.Profile
	inspection.Labels["jang_weight_format"] = jang.WeightFormat
	inspection.Labels["jang_method"] = jang.Method
	if jang.SourceName != "" {
		inspection.Labels["jang_source_name"] = jang.SourceName
	}
	if jang.SourceOrg != "" {
		inspection.Labels["jang_source_org"] = jang.SourceOrg
	}
	if jang.SourceArchitecture != "" {
		inspection.Labels["jang_source_architecture"] = normalizeROCmArchitecture(jang.SourceArchitecture)
	}
	if jang.GroupSize > 0 {
		inspection.Labels["jang_group_size"] = core.Sprintf("%d", jang.GroupSize)
	}
	if jang.BitsDefault > 0 {
		inspection.Labels["jang_bits_default"] = core.Sprintf("%d", jang.BitsDefault)
	}
	if jang.AttentionBits > 0 {
		inspection.Labels["jang_attention_bits"] = core.Sprintf("%d", jang.AttentionBits)
	}
	if jang.SharedExpertBits > 0 {
		inspection.Labels["jang_shared_expert_bits"] = core.Sprintf("%d", jang.SharedExpertBits)
	}
	if jang.RoutedExpertBits > 0 {
		inspection.Labels["jang_routed_expert_bits"] = core.Sprintf("%d", jang.RoutedExpertBits)
	}
	if jang.EmbedTokensBits > 0 {
		inspection.Labels["jang_embed_tokens_bits"] = core.Sprintf("%d", jang.EmbedTokensBits)
	}
	if jang.LMHeadBits > 0 {
		inspection.Labels["jang_lm_head_bits"] = core.Sprintf("%d", jang.LMHeadBits)
	}
	if jang.Capabilities.ReasoningParser != "" || jang.Capabilities.SupportsThinking {
		inspection.Labels["reasoning_parser"] = firstNonEmptyString(jang.Capabilities.ReasoningParser, "native-family")
		inspection.Capabilities = append(inspection.Capabilities, inference.SupportedCapability(inference.CapabilityReasoningParse, inference.CapabilityGroupModel))
	}
	if jang.Capabilities.ToolParser != "" || jang.Capabilities.SupportsTools {
		inspection.Labels["tool_parser"] = firstNonEmptyString(jang.Capabilities.ToolParser, "native-family")
		inspection.Capabilities = append(inspection.Capabilities, inference.SupportedCapability(inference.CapabilityToolParse, inference.CapabilityGroupModel))
	}
	if jang.Capabilities.CacheType != "" {
		inspection.Labels["cache_type"] = jang.Capabilities.CacheType
	}
	inspection.Capabilities = append(inspection.Capabilities, rocmFixtureKernelCapability(inference.CapabilityJANGTQ, inference.CapabilityGroupRuntime, "JANG/JANGTQ model-pack metadata is recognised and the HIP projection fixture kernel is linked; packed-weight model integration is pending"))
	inspection.Notes = append(inspection.Notes, "JANG/JANGTQ metadata is recognised on ROCm; the projection fixture kernel is linked and packed-weight model integration is pending")
}

func readROCmCodebookConfig(root string) (*rocmCodebookProfile, error) {
	return codebook.ReadProfile(root)
}

func applyROCmCodebookInspection(inspection *inference.ModelPackInspection, profile rocmCodebookProfile) {
	model := inspection.Model
	model.QuantBits = firstPositiveInt(model.QuantBits, profile.IndexBits)
	model.QuantType = firstNonEmptyString(model.QuantType, profile.Type+"."+profile.Format)
	inspection.Model = model
	inspection.Labels["codebook_type"] = profile.Type
	inspection.Labels["codebook_format"] = profile.Format
	inspection.Labels["codebook_tensors"] = core.Sprintf("%d", len(profile.Tensors))
	if profile.CodebookSize > 0 {
		inspection.Labels["codebook_size"] = core.Sprintf("%d", profile.CodebookSize)
	}
	if profile.CodeDim > 0 {
		inspection.Labels["codebook_code_dim"] = core.Sprintf("%d", profile.CodeDim)
	}
	if profile.IndexBits > 0 {
		inspection.Labels["codebook_index_bits"] = core.Sprintf("%d", profile.IndexBits)
	}
	inspection.Capabilities = append(inspection.Capabilities, rocmFixtureKernelCapability(inference.CapabilityCodebookVQ, inference.CapabilityGroupRuntime, "codebook/VQ model-pack metadata is recognised and the HIP lookup fixture kernel is linked; codebook-weight model integration is pending"))
	inspection.Notes = append(inspection.Notes, "codebook/VQ metadata is recognised on ROCm; the lookup fixture kernel is linked and codebook-weight model integration is pending")
}

func applyROCmArchitectureInspection(inspection *inference.ModelPackInspection, weightMetadataValid bool) {
	architectureDetected := inspection.Model.Architecture != ""
	architectureOK := supportedNativeArchitecture(inspection.Model.Architecture)
	quantizationOK := supportedNativeQuantization(inspection.Model.QuantBits, inspection.Model.QuantType)
	inspection.Labels["architecture_detected"] = core.Sprintf("%t", architectureDetected)
	inspection.Labels["architecture_supported"] = core.Sprintf("%t", architectureOK)
	inspection.Labels["quantization_supported"] = core.Sprintf("%t", quantizationOK)
	if isROCmDenseQuickWinArchitecture(inspection.Model.Architecture) {
		inspection.Labels["dense_route_candidate"] = "true"
		inspection.Labels["dense_route_status"] = "experimental"
		inspection.Labels["dense_route_family"] = "loader_neutral"
		inspection.Labels["dense_route_backend"] = "hip_small_decode"
		inspection.Labels["dense_route_reference"] = "gemma4_mlx_affine_matvec"
	}
	if isROCmGemma4AssistantArchitecture(inspection.Model.Architecture) {
		inspection.Labels["attached_drafter"] = "experimental_retained_plan"
		inspection.Labels["mtp_role"] = "drafter"
		inspection.Labels["mtp_target_family"] = "gemma4"
		rocmAddGemma4AttachedDrafterCapabilityBaseLabels(inspection.Labels)
		inspection.Labels["attached_drafter_official_pair_verified"] = "false"
		inspection.Labels["attached_drafter_gemma4_family_pair_verified"] = "false"
		inspection.Notes = append(inspection.Notes, "Gemma4 assistant pack is recognised as an attached MTP drafter with retained/no-replay plan evidence; native HIP packed assistant generation is pending")
	}
	inspection.Supported = inspection.Format != "missing" && weightMetadataValid && architectureDetected && architectureOK && quantizationOK
	if isROCmMoEArchitecture(inspection.Model.Architecture) || inspection.Labels["moe_experts"] != "" || inspection.Labels["gemma4_enable_moe_block"] == "true" {
		inspection.Labels["moe_text_runtime"] = hipKernelStatusNotLinked
		inspection.Labels["moe_text_decode_family"] = rocmMoETextDecodeFamily(inspection.Model.Architecture)
		inspection.Labels["moe_selected_expert_dispatch"] = hipKernelStatusNotLinked
		inspection.Capabilities = append(inspection.Capabilities,
			rocmFixtureKernelCapability(inference.CapabilityMoERouting, inference.CapabilityGroupModel, "MoE architecture metadata is recognised and the HIP router fixture kernel is linked; model integration is pending"),
			rocmFixtureKernelCapability(inference.CapabilityMoELazyExperts, inference.CapabilityGroupRuntime, "MoE lazy expert residency is required for 16GB-class ROCm devices and the HIP residency fixture kernel is linked; expert paging integration is pending"),
		)
	}
	if !architectureOK {
		inspection.Notes = append(inspection.Notes, "architecture is not in the native ROCm allow-list yet")
	}
	if !architectureDetected {
		inspection.Notes = append(inspection.Notes, "model architecture could not be detected from model-pack metadata")
	}
	if !quantizationOK {
		inspection.Notes = append(inspection.Notes, "quantisation is not expected to fit the native ROCm path")
	}
}

func rocmMoETextDecodeFamily(architecture string) string {
	switch normalizeROCmArchitecture(architecture) {
	case "gpt-oss":
		return "gpt_oss"
	case "qwen3_6_moe":
		return "qwen3_moe"
	default:
		return normalizeROCmArchitecture(architecture)
	}
}

func appendROCmInspectionCapability(inspection *inference.ModelPackInspection, capability inference.Capability) {
	for _, existing := range inspection.Capabilities {
		if existing.ID == capability.ID {
			return
		}
	}
	inspection.Capabilities = append(inspection.Capabilities, capability)
}

func applyROCmMemoryFitInspection(ctx context.Context, backend *rocmBackend, inspection *inference.ModelPackInspection) {
	if backend == nil || inspection == nil {
		return
	}
	if !inspection.Supported {
		inspection.Notes = append(inspection.Notes, "memory fit planning skipped because model pack is not supported")
		return
	}
	model := inspection.Model
	if weightBytes := rocmInspectionWeightBytes(inspection.Labels); weightBytes > 0 {
		if model.Labels == nil {
			model.Labels = map[string]string{}
		}
		model.Labels["weight_bytes"] = core.FormatUint(weightBytes, 10)
	}
	report, err := backend.PlanModelFit(ctx, model, 0)
	if err != nil || report == nil {
		if err != nil {
			inspection.Notes = append(inspection.Notes, "memory fit planning failed: "+err.Error())
		}
		return
	}
	inspection.Labels["memory_fit"] = core.Sprintf("%t", report.Fits)
	inspection.Labels["memory_plan_machine_class"] = report.MemoryPlan.MachineClass
	inspection.Labels["memory_plan_cache_mode"] = report.MemoryPlan.CacheMode
	inspection.Labels["memory_plan_kv_cache_bytes"] = core.Sprintf("%d", report.MemoryPlan.KVCacheBytes)
	for key, value := range report.MemoryPlan.Labels {
		inspection.Labels["memory_plan_"+key] = value
	}
	inspection.Notes = append(inspection.Notes, report.Notes...)
}

func rocmInspectionWeightBytes(labels map[string]string) uint64 {
	for _, key := range []string{"weight_bytes", "safetensors_index_total_size", "safetensors_payload_bytes"} {
		if value := rocmLabelUint(labels[key]); value > 0 {
			return value
		}
	}
	return 0
}

func rocmJANGQuantizationType(jang rocmJANGQuantizationInfo) string {
	lower := core.Lower(core.Concat(jang.Profile, " ", jang.WeightFormat, " ", jang.Method))
	if core.Contains(lower, "jangtq") || core.Contains(lower, "mxtq") {
		return "jangtq"
	}
	return "jang"
}

func normalizeROCmQuantizationAlias(value string) string {
	lower := core.Lower(core.Trim(value))
	lower = core.Replace(lower, "-", "_")
	switch {
	case core.Contains(lower, "auto_round_best"):
		return "auto_round_best"
	case core.Contains(lower, "auto_round_light"):
		return "auto_round_light"
	case core.Contains(lower, "auto_round"):
		return "auto_round"
	case core.Contains(lower, "jangtq"):
		return "jangtq"
	case core.Contains(lower, "mxtq"):
		return "mxtq"
	default:
		return lower
	}
}

func rocmQuantizationAliasIsAutoRound(values ...string) bool {
	for _, value := range values {
		value = strings.TrimSpace(value)
		switch {
		case strings.EqualFold(value, "auto_round"),
			strings.EqualFold(value, "auto-round"),
			strings.EqualFold(value, "autoround"),
			strings.EqualFold(value, "auto_round_best"),
			strings.EqualFold(value, "auto-round-best"),
			strings.EqualFold(value, "auto_round_light"),
			strings.EqualFold(value, "auto-round-light"):
			return true
		}
	}
	return false
}

func rocmJANGProfileBits(profile string) int {
	lower := core.Lower(profile)
	switch {
	case core.Contains(lower, "jangtq"):
		return 2
	case core.Contains(lower, "jang_1"):
		return 1
	case core.Contains(lower, "jang_2"):
		return 2
	case core.Contains(lower, "jang_3"):
		return 3
	case core.Contains(lower, "jang_4"):
		return 4
	default:
		return 0
	}
}

func rocmConfigHasEmbeddingTask(cfg rocmModelPackConfigProbe) bool {
	if !core.Contains(core.Lower(core.Concat(cfg.ModelType, " ", core.Join(" ", cfg.Architectures...))), "bert") {
		return false
	}
	return !rocmConfigHasRerankTask(cfg) && !rocmConfigHasClassifierTask(cfg)
}

func rocmConfigHasRerankTask(cfg rocmModelPackConfigProbe) bool {
	haystack := core.Lower(core.Concat(cfg.ModelType, " ", core.Join(" ", cfg.Architectures...)))
	if core.Contains(haystack, "rerank") {
		return true
	}
	for key, values := range cfg.TaskSpecificParams {
		if rocmTaskParamContains(key, values, "rerank") {
			return true
		}
	}
	return false
}

func rocmConfigHasClassifierTask(cfg rocmModelPackConfigProbe) bool {
	haystack := core.Lower(core.Concat(cfg.ModelType, " ", core.Join(" ", cfg.Architectures...)))
	if core.Contains(haystack, "sequenceclassification") || core.Contains(haystack, "sequence_classification") {
		return true
	}
	for key, values := range cfg.TaskSpecificParams {
		if rocmTaskParamContains(key, values, "classification", "classify") {
			return true
		}
	}
	return false
}

func rocmTaskParamContains(key string, value any, needles ...string) bool {
	if rocmLowerContainsAny(core.Lower(key), needles...) {
		return true
	}
	return rocmTaskParamValueContains(value, needles...)
}

func rocmTaskParamValueContains(value any, needles ...string) bool {
	switch typed := value.(type) {
	case map[string]any:
		for key, nested := range typed {
			if rocmTaskParamContains(key, nested, needles...) {
				return true
			}
		}
	case []any:
		for _, nested := range typed {
			if rocmTaskParamValueContains(nested, needles...) {
				return true
			}
		}
	default:
		return rocmLowerContainsAny(core.Lower(core.Sprintf("%v", typed)), needles...)
	}
	return false
}

func rocmLowerContainsAny(lower string, needles ...string) bool {
	for _, needle := range needles {
		if core.Contains(lower, needle) {
			return true
		}
	}
	return false
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

func formatROCmFloat(value float64) string {
	return strconv.FormatFloat(value, 'g', -1, 64)
}

func firstNonZeroInt32(values ...int32) int32 {
	for _, value := range values {
		if value != 0 {
			return value
		}
	}
	return 0
}
