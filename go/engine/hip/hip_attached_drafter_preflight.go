// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"strconv"
	"strings"

	core "dappco.re/go"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
)

type hipAttachedDrafterAssistantVerifierPreflight struct {
	Status  string
	Reason  string
	Layout  string
	Tensors string
	Missing []string
}

type hipAttachedDrafterAssistantVerifierBinding struct {
	HiddenSize        int
	VocabSize         int
	NumCentroids      int
	TokensPerCentroid int
	QuantMode         string
	AffineQuantized   bool
	OrderedEmbeddings bool
	EmbedTokens       hipAttachedDrafterAssistantLinearBinding
	Norm              hipTensor
	PreProjection     hipAttachedDrafterAssistantLinearBinding
	PostProjection    hipAttachedDrafterAssistantLinearBinding
	MaskedCentroids   hipAttachedDrafterAssistantLinearBinding
	TokenOrdering     hipTensor
	Layers            []hipAttachedDrafterAssistantVerifierLayerBinding
}

type hipAttachedDrafterAssistantVerifierLayerBinding struct {
	Layer             int
	InputNorm         hipTensor
	PostAttentionNorm hipTensor
	PreFeedforward    hipTensor
	PostFeedforward   hipTensor
	LayerScalar       hipTensor
	QueryProjection   hipAttachedDrafterAssistantLinearBinding
	OutputProjection  hipAttachedDrafterAssistantLinearBinding
	QueryNorm         hipTensor
	GateProjection    hipAttachedDrafterAssistantLinearBinding
	UpProjection      hipAttachedDrafterAssistantLinearBinding
	DownProjection    hipAttachedDrafterAssistantLinearBinding
}

type hipAttachedDrafterAssistantLinearBinding struct {
	Weight    hipTensor
	Scales    hipTensor
	Biases    hipTensor
	Quantized bool
}

func (preflight hipAttachedDrafterAssistantVerifierPreflight) Labels() map[string]string {
	labels := map[string]string{
		"attached_drafter_assistant_verifier_preflight": preflight.Status,
		"attached_drafter_assistant_verifier_reason":    preflight.Reason,
		"attached_drafter_assistant_verifier_layout":    preflight.Layout,
		"attached_drafter_assistant_verifier_tensors":   preflight.Tensors,
	}
	if len(preflight.Missing) > 0 {
		labels["attached_drafter_assistant_verifier_missing"] = strings.Join(preflight.Missing, ",")
	}
	return labels
}

func hipAttachedDrafterAssistantVerifierPreflightFor(target, draft *hipLoadedModel, planLabels map[string]string) hipAttachedDrafterAssistantVerifierPreflight {
	if target == nil {
		return hipAttachedDrafterAssistantVerifierNotReady("target model is nil", "target_model")
	}
	if draft == nil {
		return hipAttachedDrafterAssistantVerifierNotReady("draft model is nil", "draft_model")
	}
	if !isROCmGemma4Architecture(target.modelInfo.Architecture) {
		return hipAttachedDrafterAssistantVerifierNotReady("target is not a Gemma4 text model", "target_architecture")
	}
	if !isROCmGemma4AssistantArchitecture(draft.modelInfo.Architecture) {
		return hipAttachedDrafterAssistantVerifierNotReady("draft is not a Gemma4 assistant model", "assistant_architecture")
	}
	if target.modelInfo.VocabSize > 0 && draft.modelInfo.VocabSize > 0 && target.modelInfo.VocabSize != draft.modelInfo.VocabSize {
		return hipAttachedDrafterAssistantVerifierNotReady(
			core.Sprintf("draft vocab size %d does not match target vocab size %d", draft.modelInfo.VocabSize, target.modelInfo.VocabSize),
			"assistant_vocab_size",
		)
	}

	targetIdentity := rocmGemma4ModelWithInferredPathQuant(target.modelIdentity())
	draftIdentity := rocmGemma4ModelWithInferredPathQuant(draft.modelIdentity())
	labelMaps := []map[string]string{draftIdentity.Labels, draft.modelLabels, targetIdentity.Labels, planLabels}

	size := hipAttachedDrafterAssistantLabelValue(labelMaps,
		"attached_drafter_assistant_gemma4_size",
		"attached.drafter.assistant.gemma4_size",
		"gemma4_size",
	)
	if size == "" {
		size = hipAttachedDrafterAssistantLabelValue(labelMaps,
			"attached_drafter_target_gemma4_size",
			"attached.drafter.target.gemma4_size",
		)
	}
	size = modelgemma4.CanonicalSize(size)
	mode := hipAttachedDrafterAssistantLabelValue(labelMaps,
		"attached_drafter_assistant_gemma4_quant_mode",
		"attached.drafter.assistant.gemma4_quant_mode",
		"gemma4_quant_mode",
	)
	if mode == "" {
		mode = hipAttachedDrafterAssistantQuantModeFromBits(draft.modelInfo.QuantBits)
	}
	if size == "" {
		return hipAttachedDrafterAssistantVerifierNotReady("assistant Gemma4 size is missing", "assistant_gemma4_size")
	}
	if mode == "" {
		return hipAttachedDrafterAssistantVerifierNotReady("assistant Gemma4 quant mode is missing", "assistant_gemma4_quant_mode")
	}
	if _, ok := rocmGemma4MTPAssistantQuantModeSupport(size, mode); !ok {
		return hipAttachedDrafterAssistantVerifierNotReady("assistant Gemma4 quant mode is unsupported", "assistant_gemma4_quant_mode")
	}
	if target.modelInfo.HiddenSize > 0 {
		backboneHidden, backboneOK := hipAttachedDrafterAssistantIntLabelValue(labelMaps,
			"attached_drafter_assistant_backbone_hidden_size",
			"attached.drafter.assistant.backbone_hidden_size",
			"engine_attached_drafter_assistant_backbone_hidden_size",
		)
		if backboneOK && backboneHidden != target.modelInfo.HiddenSize {
			return hipAttachedDrafterAssistantVerifierNotReady(
				core.Sprintf("assistant backbone hidden size %d does not match target hidden size %d", backboneHidden, target.modelInfo.HiddenSize),
				"assistant_backbone_hidden_size",
			)
		}
	}

	if draft.modelInfo.NumLayers != modelgemma4.AssistantLayerCount {
		return hipAttachedDrafterAssistantVerifierNotReady(
			core.Sprintf("assistant_layer_count=%d want %d", draft.modelInfo.NumLayers, modelgemma4.AssistantLayerCount),
			"assistant_layer_count",
		)
	}
	if draft.modelInfo.VocabSize > 0 && draft.modelInfo.VocabSize != modelgemma4.AssistantTokenOrderingVocabSize {
		return hipAttachedDrafterAssistantVerifierNotReady(
			core.Sprintf("assistant_vocab_size=%d want %d", draft.modelInfo.VocabSize, modelgemma4.AssistantTokenOrderingVocabSize),
			"assistant_vocab_size",
		)
	}

	layoutStatus, layoutMissing, layoutErr := hipAttachedDrafterAssistantVerifierLayoutStatus(labelMaps)
	if layoutErr != "" {
		return hipAttachedDrafterAssistantVerifierNotReady(layoutErr, layoutMissing...)
	}
	if len(draft.tensors) == 0 {
		return hipAttachedDrafterAssistantVerifierPreflight{
			Status:  attachedDrafterAssistantVerifierPreflightMetadataOnly,
			Reason:  "assistant metadata is compatible; tensor inventory is not loaded",
			Layout:  layoutStatus,
			Tensors: attachedDrafterAssistantVerifierTensorsEmpty,
			Missing: layoutMissing,
		}
	}

	_, missingTensors, invalidTensors := hipAttachedDrafterAssistantVerifierBindingFor(draft, mode)
	if len(invalidTensors) > 0 {
		return hipAttachedDrafterAssistantVerifierPreflight{
			Status:  attachedDrafterAssistantVerifierPreflightNotReady,
			Reason:  "assistant tensor inventory has invalid verifier prerequisites",
			Layout:  layoutStatus,
			Tensors: attachedDrafterAssistantVerifierTensorsMissing,
			Missing: append(invalidTensors, missingTensors...),
		}
	}
	if len(missingTensors) > 0 {
		return hipAttachedDrafterAssistantVerifierPreflight{
			Status:  attachedDrafterAssistantVerifierPreflightNotReady,
			Reason:  "assistant tensor inventory is missing verifier prerequisites",
			Layout:  layoutStatus,
			Tensors: attachedDrafterAssistantVerifierTensorsMissing,
			Missing: append(layoutMissing, missingTensors...),
		}
	}
	return hipAttachedDrafterAssistantVerifierPreflight{
		Status:  attachedDrafterAssistantVerifierPreflightTensorReady,
		Reason:  "assistant verifier prerequisites are present; verifier kernel is not linked",
		Layout:  layoutStatus,
		Tensors: attachedDrafterAssistantVerifierTensorsComplete,
		Missing: layoutMissing,
	}
}

func hipAttachedDrafterAssistantVerifierNotReady(reason string, missing ...string) hipAttachedDrafterAssistantVerifierPreflight {
	return hipAttachedDrafterAssistantVerifierPreflight{
		Status:  attachedDrafterAssistantVerifierPreflightNotReady,
		Reason:  reason,
		Layout:  attachedDrafterAssistantVerifierLayoutInvalid,
		Tensors: attachedDrafterAssistantVerifierTensorsMissing,
		Missing: missing,
	}
}

func hipAttachedDrafterAssistantVerifierLayoutStatus(labelMaps []map[string]string) (string, []string, string) {
	missing := []string{}
	bad := []string{}
	check := func(name, want string, keys ...string) {
		value := hipAttachedDrafterAssistantLabelValue(labelMaps, keys...)
		if value == "" {
			missing = append(missing, name)
			return
		}
		if strings.ToLower(value) != strings.ToLower(want) {
			bad = append(bad, name+"="+value)
		}
	}
	check("assistant_centroids", modelgemma4.AssistantOrderedEmbeddingCentroidsLabel,
		"attached_drafter_assistant_centroids",
		"attached.drafter.assistant_centroids",
	)
	check("assistant_centroid_intermediate_top_k", modelgemma4.AssistantCentroidIntermediateTopKLabel,
		"attached_drafter_assistant_centroid_intermediate_top_k",
		"attached.drafter.assistant_centroid_intermediate_top_k",
	)
	orderedValue := hipAttachedDrafterAssistantLabelValue(labelMaps,
		"attached_drafter_assistant_ordered_embeddings",
		"attached.drafter.assistant_ordered_embeddings",
	)
	orderedEmbeddings := true
	if orderedValue == "" {
		missing = append(missing, "assistant_ordered_embeddings")
	} else {
		switch strings.ToLower(orderedValue) {
		case "true":
			orderedEmbeddings = true
		case "false":
			orderedEmbeddings = false
		default:
			bad = append(bad, "assistant_ordered_embeddings="+orderedValue)
		}
	}
	if orderedEmbeddings {
		check("assistant_token_ordering_shape", modelgemma4.AssistantTokenOrderingShape,
			"attached_drafter_assistant_token_ordering_shape",
			"attached.drafter.assistant_token_ordering_shape",
		)
		if value := hipAttachedDrafterAssistantLabelValue(labelMaps,
			"attached_drafter_assistant_token_ordering_dtype",
			"attached.drafter.assistant_token_ordering_dtype",
		); value != "" && !hipAttachedDrafterAssistantTokenOrderingDTypeOK(value) {
			bad = append(bad, "assistant_token_ordering_dtype="+value)
		}
	}
	if value := hipAttachedDrafterAssistantLabelValue(labelMaps,
		"attached_drafter_assistant_four_layer_drafter",
		"attached.drafter.assistant_four_layer_drafter",
	); value != "" && strings.ToLower(value) != "true" {
		bad = append(bad, "assistant_four_layer_drafter="+value)
	}
	if len(bad) > 0 {
		return attachedDrafterAssistantVerifierLayoutInvalid, bad, "assistant layout contradicts official MTP shape: " + strings.Join(bad, ",")
	}
	if len(missing) > 0 {
		return attachedDrafterAssistantVerifierLayoutInferred, missing, ""
	}
	return attachedDrafterAssistantVerifierLayoutOfficial, nil, ""
}

func hipAttachedDrafterAssistantVerifierBindingFor(model *hipLoadedModel, mode string) (hipAttachedDrafterAssistantVerifierBinding, []string, []string) {
	binding := hipAttachedDrafterAssistantVerifierBinding{
		NumCentroids:    modelgemma4.AssistantOrderedEmbeddingCentroids,
		QuantMode:       strings.ToLower(strings.TrimSpace(mode)),
		AffineQuantized: hipAttachedDrafterAssistantQuantModeRequiresAffine(mode),
	}
	if model == nil {
		return binding, []string{"assistant_model"}, nil
	}
	binding.HiddenSize = model.modelInfo.HiddenSize
	binding.VocabSize = model.modelInfo.VocabSize
	binding.OrderedEmbeddings = hipAttachedDrafterAssistantVerifierHasOrderedEmbeddingTensors(model)
	if binding.VocabSize > 0 && binding.NumCentroids > 0 {
		binding.TokensPerCentroid = binding.VocabSize / binding.NumCentroids
	}
	missing := []string{}
	invalid := []string{}
	var ok bool
	binding.EmbedTokens = hipAttachedDrafterAssistantLinearBindingFor(model.tensors, "model.embed_tokens", binding.AffineQuantized, &missing)
	if binding.Norm, ok = hipAttachedDrafterAssistantTensor(model.tensors, "model.norm.weight"); !ok {
		missing = append(missing, "model.norm.weight")
	}
	binding.PreProjection = hipAttachedDrafterAssistantLinearBindingFor(model.tensors, "pre_projection", binding.AffineQuantized, &missing)
	binding.PostProjection = hipAttachedDrafterAssistantLinearBindingFor(model.tensors, "post_projection", binding.AffineQuantized, &missing)
	if binding.OrderedEmbeddings {
		binding.MaskedCentroids = hipAttachedDrafterAssistantLinearBindingFor(model.tensors, "masked_embedding.centroids", binding.AffineQuantized, &missing)
		if binding.TokenOrdering, ok = hipAttachedDrafterAssistantTensor(model.tensors, "masked_embedding.token_ordering"); !ok {
			missing = append(missing, "masked_embedding.token_ordering")
		} else if !hipAttachedDrafterAssistantTokenOrderingTensorOK(binding.TokenOrdering.info, model.modelInfo.VocabSize, modelgemma4.AssistantOrderedEmbeddingCentroids) {
			invalid = append(invalid, "masked_embedding.token_ordering")
		}
	}

	binding.Layers = make([]hipAttachedDrafterAssistantVerifierLayerBinding, 0, modelgemma4.AssistantLayerCount)
	for layer := 0; layer < modelgemma4.AssistantLayerCount; layer++ {
		prefix := core.Sprintf("model.layers.%d", layer)
		layerBinding := hipAttachedDrafterAssistantVerifierLayerBinding{
			Layer:            layer,
			QueryProjection:  hipAttachedDrafterAssistantLinearBindingFor(model.tensors, prefix+".self_attn.q_proj", binding.AffineQuantized, &missing),
			OutputProjection: hipAttachedDrafterAssistantLinearBindingFor(model.tensors, prefix+".self_attn.o_proj", binding.AffineQuantized, &missing),
			GateProjection:   hipAttachedDrafterAssistantLinearBindingFor(model.tensors, prefix+".mlp.gate_proj", binding.AffineQuantized, &missing),
			UpProjection:     hipAttachedDrafterAssistantLinearBindingFor(model.tensors, prefix+".mlp.up_proj", binding.AffineQuantized, &missing),
			DownProjection:   hipAttachedDrafterAssistantLinearBindingFor(model.tensors, prefix+".mlp.down_proj", binding.AffineQuantized, &missing),
		}
		requiredTensor := func(name string, out *hipTensor) {
			tensor, ok := hipAttachedDrafterAssistantTensor(model.tensors, name)
			if !ok {
				missing = append(missing, name)
				return
			}
			*out = tensor
		}
		requiredTensor(prefix+".input_layernorm.weight", &layerBinding.InputNorm)
		requiredTensor(prefix+".post_attention_layernorm.weight", &layerBinding.PostAttentionNorm)
		requiredTensor(prefix+".pre_feedforward_layernorm.weight", &layerBinding.PreFeedforward)
		requiredTensor(prefix+".post_feedforward_layernorm.weight", &layerBinding.PostFeedforward)
		requiredTensor(prefix+".layer_scalar", &layerBinding.LayerScalar)
		requiredTensor(prefix+".self_attn.q_norm.weight", &layerBinding.QueryNorm)
		binding.Layers = append(binding.Layers, layerBinding)
	}
	if model.modelInfo.NumLayers != modelgemma4.AssistantLayerCount {
		invalid = append(invalid, "assistant_layer_count")
	}
	return binding, missing, invalid
}

func hipAttachedDrafterAssistantVerifierHasOrderedEmbeddingTensors(model *hipLoadedModel) bool {
	if model == nil {
		return false
	}
	if _, ok := hipAttachedDrafterAssistantTensor(model.tensors, "masked_embedding.token_ordering"); ok {
		return true
	}
	if _, ok := hipAttachedDrafterAssistantTensor(model.tensors, "masked_embedding.centroids.weight"); ok {
		return true
	}
	return false
}

func hipAttachedDrafterAssistantLinearBindingFor(tensors map[string]hipTensor, baseName string, quantized bool, missing *[]string) hipAttachedDrafterAssistantLinearBinding {
	binding := hipAttachedDrafterAssistantLinearBinding{Quantized: quantized}
	if tensor, ok := hipAttachedDrafterAssistantTensor(tensors, baseName+".weight"); ok {
		binding.Weight = tensor
	} else {
		*missing = append(*missing, baseName+".weight")
	}
	if !quantized {
		return binding
	}
	if tensor, ok := hipAttachedDrafterAssistantTensor(tensors, baseName+".scales"); ok {
		binding.Scales = tensor
	} else {
		*missing = append(*missing, baseName+".scales")
	}
	if tensor, ok := hipAttachedDrafterAssistantTensor(tensors, baseName+".biases"); ok {
		binding.Biases = tensor
	} else {
		*missing = append(*missing, baseName+".biases")
	}
	return binding
}

func hipAttachedDrafterAssistantQuantModeRequiresAffine(mode string) bool {
	mode = strings.TrimSuffix(strings.ToLower(strings.TrimSpace(mode)), "-status")
	return mode != "" && mode != modelgemma4.AssistantQuantMode
}

func hipAttachedDrafterAssistantTensor(tensors map[string]hipTensor, name string) (hipTensor, bool) {
	if len(tensors) == 0 || name == "" {
		return hipTensor{}, false
	}
	candidates := []string{name}
	if strings.HasPrefix(name, "model.") {
		candidates = append(candidates, "language_model."+name)
	}
	if !strings.HasPrefix(name, "model.") {
		candidates = append(candidates, "model."+name, "language_model."+name)
	}
	for _, candidate := range candidates {
		if tensor, ok := tensors[candidate]; ok && tensor.info.ByteSize > 0 {
			return tensor, true
		}
	}
	return hipTensor{}, false
}

func hipAttachedDrafterAssistantTokenOrderingTensorOK(tensor nativeTensorInfo, vocabSize, centroids int) bool {
	switch tensor.Type {
	case 26, 27:
	default:
		if tensor.TypeName != "" && !hipAttachedDrafterAssistantTokenOrderingDTypeOK(tensor.TypeName) {
			return false
		}
	}
	if len(tensor.Dimensions) == 0 || vocabSize <= 0 || centroids <= 0 || vocabSize%centroids != 0 {
		return false
	}
	if len(tensor.Dimensions) == 1 {
		return tensor.Dimensions[0] == uint64(vocabSize)
	}
	if len(tensor.Dimensions) == 2 {
		return tensor.Dimensions[0] == uint64(centroids) && tensor.Dimensions[1] == uint64(vocabSize/centroids)
	}
	return false
}

func hipAttachedDrafterAssistantTokenOrderingDTypeOK(dtype string) bool {
	switch strings.ToLower(strings.TrimSpace(dtype)) {
	case "i32", "int32", "i64", "int64":
		return true
	default:
		return false
	}
}

func hipAttachedDrafterAssistantLabelValue(labelMaps []map[string]string, keys ...string) string {
	for _, labels := range labelMaps {
		if len(labels) == 0 {
			continue
		}
		for _, key := range keys {
			if value := strings.TrimSpace(labels[key]); value != "" {
				return value
			}
		}
	}
	return ""
}

func hipAttachedDrafterAssistantIntLabelValue(labelMaps []map[string]string, keys ...string) (int, bool) {
	value := hipAttachedDrafterAssistantLabelValue(labelMaps, keys...)
	if value == "" {
		return 0, false
	}
	parsed, err := strconv.Atoi(value)
	return parsed, err == nil && parsed > 0
}

func hipAttachedDrafterAssistantQuantModeFromBits(bits int) string {
	if bits == 16 {
		return modelgemma4.AssistantQuantMode
	}
	if bits > 0 {
		return "q" + strconv.Itoa(bits)
	}
	return ""
}
