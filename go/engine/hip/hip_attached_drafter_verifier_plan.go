// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"encoding/binary"
	"math"
	"strconv"
	"strings"

	core "dappco.re/go"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
)

type hipAttachedDrafterAssistantVerifierPlan struct {
	Status                    string
	Reason                    string
	HiddenSize                int
	VocabSize                 int
	LayerCount                int
	NumCentroids              int
	TokensPerCentroid         int
	QuantMode                 string
	QuantBits                 int
	QuantGroup                int
	ProjectionEncoding        string
	OrderedEmbeddings         bool
	KernelFamilies            []string
	StageCount                int
	Embedding                 hipDeviceEmbeddingLookupConfig
	Norm                      hipRMSNormDeviceWeightConfig
	PreProjection             hipAttachedDrafterAssistantProjectionPlan
	PostProjection            hipAttachedDrafterAssistantProjectionPlan
	MaskedCentroids           hipAttachedDrafterAssistantProjectionPlan
	TokenOrdering             []int32
	TokenOrderingPointer      nativeDevicePointer
	TokenOrderingBytes        uint64
	TokenOrderingElementBytes int
	TokenOrderingDeviceReady  bool
	Layers                    []hipAttachedDrafterAssistantVerifierLayerPlan
}

type hipAttachedDrafterAssistantVerifierLayerPlan struct {
	Layer              int
	LayerType          string
	HiddenSize         int
	HeadDim            int
	QueryHeads         int
	RoPEBase           float32
	RoPERotaryDim      int
	RoPEFrequencyScale float32
	SlidingWindow      int
	LayerScalar        float32
	InputNorm          hipRMSNormDeviceWeightConfig
	PostAttentionNorm  hipRMSNormDeviceWeightConfig
	PreFeedforward     hipRMSNormDeviceWeightConfig
	PostFeedforward    hipRMSNormDeviceWeightConfig
	QueryNorm          hipRMSNormDeviceWeightConfig
	QueryProjection    hipAttachedDrafterAssistantProjectionPlan
	OutputProjection   hipAttachedDrafterAssistantProjectionPlan
	GateProjection     hipAttachedDrafterAssistantProjectionPlan
	UpProjection       hipAttachedDrafterAssistantProjectionPlan
	DownProjection     hipAttachedDrafterAssistantProjectionPlan
}

type hipAttachedDrafterAssistantProjectionPlan struct {
	Encoding  string
	Rows      int
	Cols      int
	BF16      hipBF16DeviceWeightConfig
	MLXAffine hipMLXQ4DeviceWeightConfig
}

func (plan hipAttachedDrafterAssistantVerifierPlan) Labels() map[string]string {
	labels := map[string]string{
		"attached_drafter_assistant_verifier_plan":   plan.Status,
		"attached_drafter_assistant_verifier_kernel": "not_linked",
	}
	if plan.Reason != "" {
		labels["attached_drafter_assistant_verifier_plan_reason"] = plan.Reason
	}
	if plan.HiddenSize > 0 {
		labels["attached_drafter_assistant_verifier_hidden_size"] = strconv.Itoa(plan.HiddenSize)
	}
	if plan.VocabSize > 0 {
		labels["attached_drafter_assistant_verifier_vocab_size"] = strconv.Itoa(plan.VocabSize)
	}
	if plan.LayerCount > 0 {
		labels["attached_drafter_assistant_verifier_layers"] = strconv.Itoa(plan.LayerCount)
	}
	if plan.NumCentroids > 0 {
		labels["attached_drafter_assistant_verifier_centroids"] = strconv.Itoa(plan.NumCentroids)
	}
	if plan.TokensPerCentroid > 0 {
		labels["attached_drafter_assistant_verifier_tokens_per_centroid"] = strconv.Itoa(plan.TokensPerCentroid)
	}
	if plan.Status == attachedDrafterAssistantVerifierPlanTensorBound {
		labels["attached_drafter_assistant_verifier_ordered_embeddings"] = strconv.FormatBool(plan.OrderedEmbeddings)
	}
	if plan.QuantMode != "" {
		labels["attached_drafter_assistant_verifier_quant_mode"] = plan.QuantMode
	}
	if plan.QuantBits > 0 {
		labels["attached_drafter_assistant_verifier_quant_bits"] = strconv.Itoa(plan.QuantBits)
	}
	if plan.QuantGroup > 0 {
		labels["attached_drafter_assistant_verifier_quant_group"] = strconv.Itoa(plan.QuantGroup)
	}
	if plan.ProjectionEncoding != "" {
		labels["attached_drafter_assistant_verifier_projection_encoding"] = plan.ProjectionEncoding
	}
	if plan.StageCount > 0 {
		labels["attached_drafter_assistant_verifier_stage_count"] = strconv.Itoa(plan.StageCount)
	}
	if len(plan.KernelFamilies) > 0 {
		labels["attached_drafter_assistant_verifier_kernel_families"] = strings.Join(plan.KernelFamilies, ",")
	}
	return labels
}

func hipAttachedDrafterAssistantVerifierPlanFor(target, draft *hipLoadedModel, planLabels map[string]string) (hipAttachedDrafterAssistantVerifierPlan, error) {
	preflight := hipAttachedDrafterAssistantVerifierPreflightFor(target, draft, planLabels)
	if preflight.Status != attachedDrafterAssistantVerifierPreflightTensorReady {
		return hipAttachedDrafterAssistantVerifierPlan{
			Status: attachedDrafterAssistantVerifierPlanNotReady,
			Reason: preflight.Status + ": " + preflight.Reason,
		}, nil
	}
	mode := hipAttachedDrafterAssistantVerifierMode(draft, planLabels)
	bits, supported := hipAttachedDrafterAssistantVerifierQuantBits(mode)
	if !supported {
		return hipAttachedDrafterAssistantVerifierPlan{
			Status:    attachedDrafterAssistantVerifierPlanUnsupported,
			Reason:    "assistant verifier launch plan does not support quant mode " + mode,
			QuantMode: mode,
		}, nil
	}
	binding, missing, invalid := hipAttachedDrafterAssistantVerifierBindingFor(draft, mode)
	if len(invalid) > 0 || len(missing) > 0 {
		return hipAttachedDrafterAssistantVerifierPlan{}, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", "assistant verifier binding is not tensor-ready", nil)
	}
	plan := hipAttachedDrafterAssistantVerifierPlan{
		Status:            attachedDrafterAssistantVerifierPlanTensorBound,
		Reason:            "assistant verifier tensors are shaped for existing HIP launch families; execution loop is linked",
		HiddenSize:        binding.HiddenSize,
		VocabSize:         binding.VocabSize,
		LayerCount:        len(binding.Layers),
		NumCentroids:      binding.NumCentroids,
		TokensPerCentroid: binding.TokensPerCentroid,
		QuantMode:         binding.QuantMode,
		QuantBits:         bits,
		QuantGroup:        hipAttachedDrafterAssistantVerifierQuantGroup(draft),
		OrderedEmbeddings: binding.OrderedEmbeddings,
		KernelFamilies: []string{
			hipKernelNameEmbedLookup,
			hipKernelNameRMSNorm,
			hipAttachedDrafterAssistantVerifierProjectionKernel(binding.AffineQuantized),
			hipAttachedDrafterAssistantVerifierGELUKernel(binding.AffineQuantized),
			hipKernelNameAttentionHeads,
			hipKernelNameVectorAddScaled,
		},
	}
	if binding.OrderedEmbeddings {
		plan.KernelFamilies = append(plan.KernelFamilies,
			hipKernelNamePackedTopK,
			hipKernelNameOrderedEmbeddingCandidates,
			hipKernelNameMLXQ4ProjSelectedGreedy,
			hipKernelNameMLXQ4ProjSelectedGreedyQ6Row64,
		)
	} else {
		plan.KernelFamilies = append(plan.KernelFamilies,
			hipKernelNameMLXQ4ProjGreedy,
			hipKernelNameMLXQ4ProjGreedyQ6Row64,
		)
	}
	if !binding.AffineQuantized {
		plan.ProjectionEncoding = "bf16"
		plan.QuantGroup = 0
	} else {
		plan.ProjectionEncoding = "mlx_affine"
	}
	var err error
	if plan.Embedding, err = hipAttachedDrafterAssistantEmbeddingPlan(binding, plan.QuantGroup); err != nil {
		return hipAttachedDrafterAssistantVerifierPlan{}, err
	}
	if plan.Norm, err = hipAttachedDrafterAssistantNormPlan("model.norm.weight", binding.Norm, plan.HiddenSize); err != nil {
		return hipAttachedDrafterAssistantVerifierPlan{}, err
	}
	if plan.PreProjection, err = hipAttachedDrafterAssistantProjectionPlanFor("pre_projection", binding.PreProjection, plan.QuantGroup, plan.QuantBits); err != nil {
		return hipAttachedDrafterAssistantVerifierPlan{}, err
	}
	if plan.PostProjection, err = hipAttachedDrafterAssistantProjectionPlanFor("post_projection", binding.PostProjection, plan.QuantGroup, plan.QuantBits); err != nil {
		return hipAttachedDrafterAssistantVerifierPlan{}, err
	}
	if binding.OrderedEmbeddings {
		if plan.MaskedCentroids, err = hipAttachedDrafterAssistantProjectionPlanFor("masked_embedding.centroids", binding.MaskedCentroids, plan.QuantGroup, plan.QuantBits); err != nil {
			return hipAttachedDrafterAssistantVerifierPlan{}, err
		}
		plan.TokenOrderingPointer = binding.TokenOrdering.pointer
		plan.TokenOrderingBytes = binding.TokenOrdering.info.ByteSize
		plan.TokenOrderingDeviceReady = draft != nil && draft.driver != nil && binding.TokenOrdering.pointer != 0
		if plan.TokenOrdering, plan.TokenOrderingElementBytes, err = hipAttachedDrafterAssistantTokenOrdering(draft, binding.TokenOrdering, binding.VocabSize, binding.NumCentroids); err != nil {
			return hipAttachedDrafterAssistantVerifierPlan{}, err
		}
	}
	plan.Layers = make([]hipAttachedDrafterAssistantVerifierLayerPlan, 0, len(binding.Layers))
	for _, layer := range binding.Layers {
		layerPlan, err := hipAttachedDrafterAssistantLayerPlanFor(draft, layer, plan.HiddenSize, plan.QuantGroup, plan.QuantBits)
		if err != nil {
			return hipAttachedDrafterAssistantVerifierPlan{}, err
		}
		plan.Layers = append(plan.Layers, layerPlan)
	}
	plan.StageCount = hipAttachedDrafterAssistantVerifierStageCount(plan)
	return plan, nil
}

func hipAttachedDrafterAssistantVerifierMode(draft *hipLoadedModel, planLabels map[string]string) string {
	if draft == nil {
		return ""
	}
	identity := rocmGemma4ModelWithInferredPathQuant(draft.modelIdentity())
	mode := hipAttachedDrafterAssistantLabelValue([]map[string]string{identity.Labels, draft.modelLabels, planLabels},
		"attached_drafter_assistant_gemma4_quant_mode",
		"attached.drafter.assistant.gemma4_quant_mode",
		"gemma4_quant_mode",
	)
	if mode == "" {
		mode = hipAttachedDrafterAssistantQuantModeFromBits(draft.modelInfo.QuantBits)
	}
	return strings.ToLower(strings.TrimSpace(mode))
}

func hipAttachedDrafterAssistantVerifierQuantBits(mode string) (int, bool) {
	mode = strings.ToLower(strings.TrimSpace(mode))
	if mode == "" || mode == modelgemma4.AssistantQuantMode {
		return 16, true
	}
	if !strings.HasPrefix(mode, "q") {
		return 0, false
	}
	value, err := strconv.Atoi(strings.TrimPrefix(mode, "q"))
	if err != nil || value <= 0 || value > 8 {
		return 0, false
	}
	if !hipMLXAffineSupportedBits(value) {
		return value, false
	}
	return value, true
}

func hipAttachedDrafterAssistantVerifierQuantGroup(model *hipLoadedModel) int {
	if model == nil || model.modelInfo.QuantGroup <= 0 {
		return 64
	}
	return model.modelInfo.QuantGroup
}

func hipAttachedDrafterAssistantEmbeddingPlan(binding hipAttachedDrafterAssistantVerifierBinding, groupSize int) (hipDeviceEmbeddingLookupConfig, error) {
	if binding.EmbedTokens.Weight.pointer == 0 {
		return hipDeviceEmbeddingLookupConfig{}, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", "embed_tokens weight pointer is required", nil)
	}
	cfg := hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: binding.EmbedTokens.Weight.pointer,
		EmbeddingBytes:   binding.EmbedTokens.Weight.info.ByteSize,
		VocabSize:        binding.VocabSize,
		HiddenSize:       binding.HiddenSize,
	}
	if binding.AffineQuantized {
		cfg.TableEncoding = hipEmbeddingTableEncodingMLXQ4
		cfg.GroupSize = groupSize
		cfg.ScalePointer = binding.EmbedTokens.Scales.pointer
		cfg.BiasPointer = binding.EmbedTokens.Biases.pointer
		cfg.ScaleBytes = binding.EmbedTokens.Scales.info.ByteSize
		cfg.BiasBytes = binding.EmbedTokens.Biases.info.ByteSize
		cfg.QuantBits = binding.QuantBits()
	} else {
		cfg.TableEncoding = hipEmbeddingTableEncodingBF16
	}
	if err := cfg.validateSingleToken(0); err != nil {
		return hipDeviceEmbeddingLookupConfig{}, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", "embed_tokens config", err)
	}
	return cfg, nil
}

func hipAttachedDrafterAssistantNormPlan(label string, tensor hipTensor, count int) (hipRMSNormDeviceWeightConfig, error) {
	if tensor.pointer == 0 {
		return hipRMSNormDeviceWeightConfig{}, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", label+" pointer is required", nil)
	}
	encoding, expectedBytes, err := hipGemma4NormWeightEncodingAndBytes(tensor.info, label, count)
	if err != nil {
		return hipRMSNormDeviceWeightConfig{}, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", label+" config", err)
	}
	if len(tensor.info.Dimensions) != 1 ||
		tensor.info.Dimensions[0] != uint64(count) ||
		tensor.info.ByteSize != expectedBytes {
		return hipRMSNormDeviceWeightConfig{}, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", label+" tensor shape/type mismatch", nil)
	}
	cfg := hipRMSNormDeviceWeightConfig{
		WeightPointer:  tensor.pointer,
		WeightBytes:    tensor.info.ByteSize,
		Count:          count,
		WeightEncoding: encoding,
	}
	if err := hipValidateGemma4Q4NormConfig(label, cfg, count); err != nil {
		return hipRMSNormDeviceWeightConfig{}, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", label+" config", err)
	}
	return cfg, nil
}

func hipAttachedDrafterAssistantProjectionPlanFor(label string, binding hipAttachedDrafterAssistantLinearBinding, groupSize, bits int) (hipAttachedDrafterAssistantProjectionPlan, error) {
	rows, cols, err := hipAttachedDrafterAssistantLinearDims(label, binding.Weight, binding.Quantized, bits)
	if err != nil {
		return hipAttachedDrafterAssistantProjectionPlan{}, err
	}
	plan := hipAttachedDrafterAssistantProjectionPlan{Rows: rows, Cols: cols}
	if !binding.Quantized {
		plan.Encoding = "bf16"
		plan.BF16 = hipBF16DeviceWeightConfig{
			WeightPointer: binding.Weight.pointer,
			WeightBytes:   binding.Weight.info.ByteSize,
			Rows:          rows,
			Cols:          cols,
		}
		if err := plan.BF16.validate(hipProjectionWeightEncodingBF16); err != nil {
			return hipAttachedDrafterAssistantProjectionPlan{}, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", label+" BF16 projection config", err)
		}
		return plan, nil
	}
	packedCols, err := hipMLXAffinePackedCols(cols, bits)
	if err != nil {
		return hipAttachedDrafterAssistantProjectionPlan{}, err
	}
	if len(binding.Weight.info.Dimensions) != 2 ||
		binding.Weight.info.Dimensions[0] != uint64(rows) ||
		binding.Weight.info.Dimensions[1] != uint64(packedCols) {
		return hipAttachedDrafterAssistantProjectionPlan{}, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", label+" MLX affine packed weight shape mismatch", nil)
	}
	plan.Encoding = "mlx_affine"
	plan.MLXAffine = hipMLXQ4DeviceWeightConfig{
		WeightPointer: binding.Weight.pointer,
		ScalePointer:  binding.Scales.pointer,
		BiasPointer:   binding.Biases.pointer,
		WeightBytes:   binding.Weight.info.ByteSize,
		ScaleBytes:    binding.Scales.info.ByteSize,
		BiasBytes:     binding.Biases.info.ByteSize,
		Rows:          rows,
		Cols:          cols,
		GroupSize:     groupSize,
		Bits:          bits,
	}
	if err := plan.MLXAffine.validateInputCount(cols); err != nil {
		return hipAttachedDrafterAssistantProjectionPlan{}, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", label+" MLX affine projection config", err)
	}
	return plan, nil
}

func hipAttachedDrafterAssistantLinearDims(label string, tensor hipTensor, quantized bool, bits int) (int, int, error) {
	if tensor.pointer == 0 {
		return 0, 0, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", label+" weight pointer is required", nil)
	}
	if len(tensor.info.Dimensions) != 2 {
		return 0, 0, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", label+" weight must be rank 2", nil)
	}
	rows := int(tensor.info.Dimensions[0])
	cols := int(tensor.info.Dimensions[1])
	if rows <= 0 || cols <= 0 {
		return 0, 0, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", label+" dimensions must be positive", nil)
	}
	if quantized {
		unpackedCols, err := hipMLXAffineColsFromPackedCols(cols, bits)
		if err != nil {
			return 0, 0, err
		}
		cols = unpackedCols
	}
	return rows, cols, nil
}

func hipAttachedDrafterAssistantLayerPlanFor(draft *hipLoadedModel, layer hipAttachedDrafterAssistantVerifierLayerBinding, hidden, groupSize, bits int) (hipAttachedDrafterAssistantVerifierLayerPlan, error) {
	var err error
	plan := hipAttachedDrafterAssistantVerifierLayerPlan{Layer: layer.Layer, HiddenSize: hidden}
	if plan.InputNorm, err = hipAttachedDrafterAssistantNormPlan("input_layernorm", layer.InputNorm, hidden); err != nil {
		return plan, err
	}
	if plan.PostAttentionNorm, err = hipAttachedDrafterAssistantNormPlan("post_attention_layernorm", layer.PostAttentionNorm, hidden); err != nil {
		return plan, err
	}
	if plan.PreFeedforward, err = hipAttachedDrafterAssistantNormPlan("pre_feedforward_layernorm", layer.PreFeedforward, hidden); err != nil {
		return plan, err
	}
	if plan.PostFeedforward, err = hipAttachedDrafterAssistantNormPlan("post_feedforward_layernorm", layer.PostFeedforward, hidden); err != nil {
		return plan, err
	}
	if plan.QueryProjection, err = hipAttachedDrafterAssistantProjectionPlanFor("q_proj", layer.QueryProjection, groupSize, bits); err != nil {
		return plan, err
	}
	if plan.OutputProjection, err = hipAttachedDrafterAssistantProjectionPlanFor("o_proj", layer.OutputProjection, groupSize, bits); err != nil {
		return plan, err
	}
	if plan.GateProjection, err = hipAttachedDrafterAssistantProjectionPlanFor("mlp.gate_proj", layer.GateProjection, groupSize, bits); err != nil {
		return plan, err
	}
	if plan.UpProjection, err = hipAttachedDrafterAssistantProjectionPlanFor("mlp.up_proj", layer.UpProjection, groupSize, bits); err != nil {
		return plan, err
	}
	if plan.DownProjection, err = hipAttachedDrafterAssistantProjectionPlanFor("mlp.down_proj", layer.DownProjection, groupSize, bits); err != nil {
		return plan, err
	}
	queryNormCount := hipAttachedDrafterAssistantNormTensorCount(layer.QueryNorm)
	if err := hipAttachedDrafterAssistantFillLayerGeometry(draft, &plan, queryNormCount); err != nil {
		return plan, err
	}
	if plan.QueryNorm, err = hipAttachedDrafterAssistantNormPlan("q_norm", layer.QueryNorm, plan.HeadDim); err != nil {
		return plan, err
	}
	if err := hipAttachedDrafterAssistantLayerPlanInvalidReason(plan); err != nil {
		return plan, err
	}
	plan.LayerScalar, err = hipAttachedDrafterAssistantLayerScalar(draft, layer.LayerScalar)
	if err != nil {
		return plan, err
	}
	return plan, nil
}

func hipAttachedDrafterAssistantFillLayerGeometry(draft *hipLoadedModel, plan *hipAttachedDrafterAssistantVerifierLayerPlan, queryNormCount int) error {
	if plan == nil {
		return core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", "assistant layer plan is nil", nil)
	}
	if plan.QueryProjection.Rows <= 0 || plan.QueryProjection.Cols != plan.HiddenSize {
		return core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", "assistant q_proj shape mismatch", nil)
	}
	layerType := ""
	if draft != nil && plan.Layer >= 0 && plan.Layer < len(draft.gemma4TextConfig.LayerTypes) {
		layerType = draft.gemma4TextConfig.LayerTypes[plan.Layer]
	}
	headDim := 0
	if draft != nil {
		switch layerType {
		case "full_attention":
			headDim = draft.gemma4TextConfig.GlobalHeadDim
		}
		if headDim <= 0 {
			headDim = draft.gemma4TextConfig.HeadDim
		}
	}
	if queryNormCount > 0 && plan.QueryProjection.Rows%queryNormCount == 0 {
		headDim = queryNormCount
	}
	if headDim <= 0 || plan.QueryProjection.Rows%headDim != 0 {
		headDim = hipAttachedDrafterAssistantInferHeadDim(plan.QueryProjection.Rows, layerType)
	}
	if headDim <= 0 || plan.QueryProjection.Rows%headDim != 0 {
		return core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", "assistant q_proj rows do not divide into heads", nil)
	}
	if layerType == "" {
		layerType = hipGemma4Q4LayerTypeFromHeadDim(headDim)
	}
	plan.LayerType = layerType
	plan.HeadDim = headDim
	plan.QueryHeads = plan.QueryProjection.Rows / headDim
	plan.RoPEBase, plan.RoPERotaryDim, plan.RoPEFrequencyScale = draft.loadedGemma4Q4LayerRoPE(layerType, headDim)
	plan.SlidingWindow = draft.loadedGemma4Q4EffectiveSlidingWindow(layerType, headDim)
	return nil
}

func hipAttachedDrafterAssistantNormTensorCount(tensor hipTensor) int {
	if len(tensor.info.Dimensions) != 1 || tensor.info.Dimensions[0] == 0 {
		return 0
	}
	return int(tensor.info.Dimensions[0])
}

func hipAttachedDrafterAssistantInferHeadDim(queryRows int, layerType string) int {
	if queryRows <= 0 {
		return 0
	}
	if layerType == "" && queryRows%2 == 0 {
		return queryRows
	}
	candidates := []int{512, 256, 128, 64, 32, 16, 8, 4, 2, 1}
	if layerType == "full_attention" {
		candidates = []int{512, 256, 128, 64, 32, 16, 8, 4, 2, 1}
	}
	for _, candidate := range candidates {
		if queryRows%candidate == 0 {
			return candidate
		}
	}
	return 0
}

func hipAttachedDrafterAssistantLayerScalar(model *hipLoadedModel, tensor hipTensor) (float32, error) {
	if tensor.pointer == 0 {
		return 1, nil
	}
	if model == nil || model.driver == nil {
		return 1, nil
	}
	scalarBytes, err := hipGemma4LayerScalarBytes(tensor.info, "assistant layer scalar tensor")
	if err != nil {
		return 0, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", "assistant layer scalar config", err)
	}
	payload := make([]byte, scalarBytes)
	if err := model.driver.CopyDeviceToHost(tensor.pointer, payload); err != nil {
		return 0, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", "copy assistant layer scalar", err)
	}
	switch core.Upper(tensor.info.TypeName) {
	case "F32":
		return math.Float32frombits(binary.LittleEndian.Uint32(payload)), nil
	case "BF16":
		return hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload)), nil
	default:
		return 0, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", "assistant layer scalar tensor must be F32 or BF16", nil)
	}
}

func hipAttachedDrafterAssistantTokenOrdering(model *hipLoadedModel, tensor hipTensor, vocabSize, centroids int) ([]int32, int, error) {
	if tensor.pointer == 0 || vocabSize <= 0 || centroids <= 0 {
		return nil, 0, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", "assistant token ordering tensor is required", nil)
	}
	dtype := strings.ToUpper(strings.TrimSpace(tensor.info.TypeName))
	elementBytes := 0
	switch dtype {
	case "I32", "INT32":
		elementBytes = 4
	case "I64", "INT64":
		elementBytes = 8
	default:
		return nil, 0, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", "assistant token ordering dtype must be int32 or int64", nil)
	}
	if tensor.info.ByteSize != uint64(vocabSize*elementBytes) {
		return nil, 0, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", "assistant token ordering byte size mismatch", nil)
	}
	if model == nil || model.driver == nil {
		return nil, elementBytes, nil
	}
	payload := make([]byte, tensor.info.ByteSize)
	if err := model.driver.CopyDeviceToHost(tensor.pointer, payload); err != nil {
		return nil, 0, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", "copy assistant token ordering", err)
	}
	ordering := make([]int32, vocabSize)
	for index := range ordering {
		var id int64
		if elementBytes == 4 {
			id = int64(int32(binary.LittleEndian.Uint32(payload[index*4:])))
		} else {
			id = int64(binary.LittleEndian.Uint64(payload[index*8:]))
		}
		if id < 0 || id >= int64(vocabSize) {
			return nil, 0, core.E("rocm.hip.AttachedDrafterAssistantVerifierPlan", "assistant token ordering contains token outside vocabulary", nil)
		}
		ordering[index] = int32(id)
	}
	return ordering, elementBytes, nil
}

func (binding hipAttachedDrafterAssistantVerifierBinding) QuantBits() int {
	bits, ok := hipAttachedDrafterAssistantVerifierQuantBits(binding.QuantMode)
	if !ok {
		return 0
	}
	return bits
}

func hipAttachedDrafterAssistantVerifierProjectionKernel(quantized bool) string {
	if quantized {
		return hipKernelNameMLXQ4Proj
	}
	return hipKernelNameProjection
}

func hipAttachedDrafterAssistantVerifierGELUKernel(quantized bool) string {
	if quantized {
		return hipKernelNameMLXQ4GELUTanhMul
	}
	return hipKernelNameGELUTanhMul
}

func hipAttachedDrafterAssistantVerifierStageCount(plan hipAttachedDrafterAssistantVerifierPlan) int {
	if plan.Status != attachedDrafterAssistantVerifierPlanTensorBound {
		return 0
	}
	if !plan.OrderedEmbeddings {
		return 5 + len(plan.Layers)*11
	}
	// Embedding, model norm, pre projection, post projection, masked centroid
	// projection, token ordering/top-k, plus per-layer norm/projection blocks.
	return 6 + len(plan.Layers)*11
}
