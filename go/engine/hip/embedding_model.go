// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"sort"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

type nativeEmbeddingModel interface {
	Embed(ctx context.Context, req inference.EmbeddingRequest) (*inference.EmbeddingResult, error)
}

type nativeRerankModel interface {
	Rerank(ctx context.Context, req inference.RerankRequest) (*inference.RerankResult, error)
}

type hipLoadedEmbeddingConfig struct {
	EmbeddingPointer nativeDevicePointer
	EmbeddingBytes   uint64
	VocabSize        int
	HiddenSize       int
	Family           string
}

type hipLoadedSequenceClassifierConfig struct {
	WeightPointer      nativeDevicePointer
	WeightBytes        uint64
	WeightEncoding     uint32
	BiasPointer        nativeDevicePointer
	BiasBytes          uint64
	BiasEncoding       uint32
	NumLabels          int
	HiddenSize         int
	WeightTensor       string
	BiasTensor         string
	PositiveLabelIndex int
}

type hipLoadedSequenceClassifierWeights struct {
	F32  []float32
	FP16 []uint16
}

func (m *rocmModel) Embed(ctx context.Context, req inference.EmbeddingRequest) (*inference.EmbeddingResult, error) {
	m.clearLastError()
	if err := rocmContextErr(ctx); err != nil {
		m.setLastFailure(err)
		return nil, err
	}
	if m == nil || m.native == nil {
		err := core.E("rocm.Embed", "native model is nil", nil)
		if m != nil {
			m.setLastFailure(err)
		}
		return nil, err
	}
	if err := validateEmbeddingRequest("rocm.Embed", req); err != nil {
		m.setLastFailure(err)
		return nil, err
	}
	native, ok := m.native.(nativeEmbeddingModel)
	if !ok {
		err := hipKernelNotLinkedError("rocm.Embed", hipKernelEmbedding, m.kernelStatus())
		m.setLastFailure(err)
		return nil, err
	}
	result, err := native.Embed(ctx, req)
	if err != nil {
		m.setLastFailure(err)
		return nil, err
	}
	if result == nil {
		err := core.E("rocm.Embed", "native embedding result is nil", nil)
		m.setLastFailure(err)
		return nil, err
	}
	result = cloneEmbeddingResult(result)
	result.Model = m.modelIdentity()
	return result, nil
}

func (m *rocmModel) Rerank(ctx context.Context, req inference.RerankRequest) (*inference.RerankResult, error) {
	m.clearLastError()
	if err := rocmContextErr(ctx); err != nil {
		m.setLastFailure(err)
		return nil, err
	}
	if m == nil || m.native == nil {
		err := core.E("rocm.Rerank", "native model is nil", nil)
		if m != nil {
			m.setLastFailure(err)
		}
		return nil, err
	}
	if err := validateRerankRequest("rocm.Rerank", req); err != nil {
		m.setLastFailure(err)
		return nil, err
	}
	native, ok := m.native.(nativeRerankModel)
	if !ok {
		err := hipKernelNotLinkedError("rocm.Rerank", hipKernelRerank, m.kernelStatus())
		m.setLastFailure(err)
		return nil, err
	}
	result, err := native.Rerank(ctx, req)
	if err != nil {
		m.setLastFailure(err)
		return nil, err
	}
	if result == nil {
		err := core.E("rocm.Rerank", "native rerank result is nil", nil)
		m.setLastFailure(err)
		return nil, err
	}
	result = cloneRerankResult(result)
	result.Model = m.modelIdentity()
	return result, nil
}

func (model *hipLoadedModel) Embed(ctx context.Context, req inference.EmbeddingRequest) (*inference.EmbeddingResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if model == nil {
		return nil, core.E("rocm.hip.Embed", "loaded model is required", nil)
	}
	if err := validateEmbeddingRequest("rocm.hip.Embed", req); err != nil {
		return nil, err
	}
	status := normalizeHIPKernelStatus(model.KernelStatus())
	if status.Embedding != hipKernelStatusLinked {
		return nil, hipKernelNotLinkedError("rocm.hip.Embed", hipKernelEmbedding, status)
	}
	cfg, err := model.loadedEmbeddingConfig()
	if err != nil {
		return nil, core.E("rocm.hip.Embed", "load f32 embedding table", err)
	}
	table, err := model.loadedEmbeddingTable(cfg)
	if err != nil {
		return nil, err
	}
	vectors := make([][]float32, 0, len(req.Input))
	promptTokens := 0
	for _, input := range req.Input {
		tokenIDs := model.Encode(input)
		promptTokens += len(tokenIDs)
		tokens, err := hipTokenEmbeddingVectors(table, cfg, tokenIDs)
		if err != nil {
			return nil, err
		}
		vector, err := hipRunEmbeddingMeanPoolKernel(ctx, model.driver, hipEmbeddingMeanPoolRequest{
			Tokens:     tokens,
			TokenCount: len(tokenIDs),
			Dim:        cfg.HiddenSize,
			Normalize:  req.Normalize,
		})
		if err != nil {
			return nil, err
		}
		vectors = append(vectors, vector)
	}
	return &inference.EmbeddingResult{
		Model:   hipLoadedModelIdentity(model),
		Vectors: vectors,
		Usage: inference.EmbeddingUsage{
			PromptTokens: promptTokens,
			TotalTokens:  promptTokens,
		},
		Labels: mergeStringMaps(req.Labels, map[string]string{
			"backend":                "rocm",
			"embedding_kernel":       hipKernelStatusLinked,
			"embedding_kernel_name":  hipKernelNameEmbedMean,
			"embedding_model_family": cfg.Family,
			"embedding_model_status": "experimental_loaded_f32_table",
			"embedding_pooling":      "mean",
			"embedding_source":       "loaded_f32_token_embeddings",
		}),
	}, nil
}

func (model *hipLoadedModel) Rerank(ctx context.Context, req inference.RerankRequest) (*inference.RerankResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if model == nil {
		return nil, core.E("rocm.hip.Rerank", "loaded model is required", nil)
	}
	if err := validateRerankRequest("rocm.hip.Rerank", req); err != nil {
		return nil, err
	}
	status := normalizeHIPKernelStatus(model.KernelStatus())
	if status.Rerank != hipKernelStatusLinked {
		return nil, hipKernelNotLinkedError("rocm.hip.Rerank", hipKernelRerank, status)
	}
	classifier, hasClassifier, err := model.loadedSequenceClassifierConfig()
	if err != nil {
		return nil, err
	}
	if hasClassifier {
		return model.rerankWithSequenceClassifier(ctx, req, classifier)
	}
	inputs := make([]string, 0, len(req.Documents)+1)
	inputs = append(inputs, req.Query)
	inputs = append(inputs, req.Documents...)
	embedded, err := model.Embed(ctx, inference.EmbeddingRequest{
		Model:     req.Model,
		Input:     inputs,
		Normalize: true,
	})
	if err != nil {
		return nil, err
	}
	if len(embedded.Vectors) != len(inputs) {
		return nil, core.E("rocm.hip.Rerank", "embedding result count mismatch", nil)
	}
	query := embedded.Vectors[0]
	documents := embedded.Vectors[1:]
	dim := len(query)
	flat, err := flattenEqualFloat32Vectors(documents, dim)
	if err != nil {
		return nil, err
	}
	scores, err := hipRunRerankCosineKernel(ctx, model.driver, hipRerankCosineRequest{
		Query:         query,
		Documents:     flat,
		DocumentCount: len(documents),
		Dim:           dim,
	})
	if err != nil {
		return nil, err
	}
	results, err := rocmRerankScoresFromCosine(scores, req.Documents, req.TopN)
	if err != nil {
		return nil, err
	}
	return &inference.RerankResult{
		Model:   hipLoadedModelIdentity(model),
		Results: results,
		Labels: mergeStringMaps(req.Labels, map[string]string{
			"backend":               "rocm",
			"embedding_kernel":      hipKernelStatusLinked,
			"embedding_kernel_name": hipKernelNameEmbedMean,
			"rerank_kernel":         hipKernelStatusLinked,
			"rerank_kernel_name":    hipKernelNameRerank,
			"rerank_model_status":   "experimental_embedding_cosine",
		}),
	}, nil
}

func validateEmbeddingRequest(operation string, req inference.EmbeddingRequest) error {
	if len(req.Input) == 0 {
		return core.E(operation, "input text is required", nil)
	}
	for index, input := range req.Input {
		if core.Trim(input) == "" {
			return core.E(operation, core.Sprintf("input %d is empty", index), nil)
		}
	}
	return nil
}

func validateRerankRequest(operation string, req inference.RerankRequest) error {
	if core.Trim(req.Query) == "" {
		return core.E(operation, "query is required", nil)
	}
	if len(req.Documents) == 0 {
		return core.E(operation, "documents are required", nil)
	}
	for index, document := range req.Documents {
		if core.Trim(document) == "" {
			return core.E(operation, core.Sprintf("document %d is empty", index), nil)
		}
	}
	return nil
}

func cloneEmbeddingResult(result *inference.EmbeddingResult) *inference.EmbeddingResult {
	if result == nil {
		return nil
	}
	out := *result
	out.Model.Labels = cloneStringMap(out.Model.Labels)
	out.Labels = cloneStringMap(out.Labels)
	if len(result.Vectors) > 0 {
		out.Vectors = make([][]float32, len(result.Vectors))
		for index := range result.Vectors {
			out.Vectors[index] = append([]float32(nil), result.Vectors[index]...)
		}
	}
	return &out
}

func cloneRerankResult(result *inference.RerankResult) *inference.RerankResult {
	if result == nil {
		return nil
	}
	out := *result
	out.Model.Labels = cloneStringMap(out.Model.Labels)
	out.Labels = cloneStringMap(out.Labels)
	if len(result.Results) > 0 {
		out.Results = append([]inference.RerankScore(nil), result.Results...)
		for index := range out.Results {
			out.Results[index].Labels = cloneStringMap(out.Results[index].Labels)
		}
	}
	return &out
}

func (model *hipLoadedModel) rerankWithSequenceClassifier(ctx context.Context, req inference.RerankRequest, classifier hipLoadedSequenceClassifierConfig) (*inference.RerankResult, error) {
	embeddingCfg, err := model.loadedEmbeddingConfig()
	if err != nil {
		return nil, core.E("rocm.hip.SequenceRerank", "load f32 embedding table", err)
	}
	if embeddingCfg.HiddenSize != classifier.HiddenSize {
		return nil, core.E("rocm.hip.SequenceRerank", "embedding and classifier hidden sizes must match", nil)
	}
	table, err := model.loadedEmbeddingTable(embeddingCfg)
	if err != nil {
		return nil, err
	}
	weights, err := model.loadedClassifierWeights(classifier)
	if err != nil {
		return nil, err
	}
	bias, err := model.loadedClassifierBias(classifier)
	if err != nil {
		return nil, err
	}
	projectionKernelName := hipKernelNameProjection
	scores := make([]float32, 0, len(req.Documents))
	for _, document := range req.Documents {
		tokenIDs := model.Encode(hipSequenceClassifierPairText(req.Query, document))
		tokens, err := hipTokenEmbeddingVectors(table, embeddingCfg, tokenIDs)
		if err != nil {
			return nil, err
		}
		pooled, err := hipRunEmbeddingMeanPoolKernel(ctx, model.driver, hipEmbeddingMeanPoolRequest{
			Tokens:     tokens,
			TokenCount: len(tokenIDs),
			Dim:        embeddingCfg.HiddenSize,
		})
		if err != nil {
			return nil, err
		}
		var logits []float32
		logits, projectionKernelName, err = model.runSequenceClassifierProjection(ctx, classifier, pooled, weights, bias)
		if err != nil {
			return nil, err
		}
		score, err := hipSequenceClassifierRerankScore(logits, classifier.PositiveLabelIndex)
		if err != nil {
			return nil, err
		}
		scores = append(scores, score)
	}
	results, err := rocmRerankScoresFromCosine(scores, req.Documents, req.TopN)
	if err != nil {
		return nil, err
	}
	for index := range results {
		results[index].Labels = mergeStringMaps(results[index].Labels, map[string]string{
			"rerank_score_source":      "classifier_positive_logit",
			"rerank_classifier_index":  core.Sprintf("%d", classifier.PositiveLabelIndex),
			"rerank_classifier_tensor": classifier.WeightTensor,
		})
	}
	labels := map[string]string{
		"backend":                    "rocm",
		"embedding_kernel":           hipKernelStatusLinked,
		"embedding_kernel_name":      hipKernelNameEmbedMean,
		"embedding_model_family":     embeddingCfg.Family,
		"embedding_model_status":     "experimental_loaded_f32_table",
		"projection_kernel":          hipKernelStatusLinked,
		"projection_kernel_name":     projectionKernelName,
		"rerank_classifier_index":    core.Sprintf("%d", classifier.PositiveLabelIndex),
		"rerank_classifier_encoding": hipProjectionWeightEncodingLabel(classifier.WeightEncoding),
		"rerank_classifier_labels":   core.Sprintf("%d", classifier.NumLabels),
		"rerank_classifier_tensor":   classifier.WeightTensor,
		"rerank_kernel":              hipKernelStatusLinked,
		"rerank_model_status":        "experimental_bert_sequence_classifier",
		"rerank_score_source":        "classifier_positive_logit",
	}
	if classifier.BiasTensor != "" {
		labels["rerank_classifier_bias"] = classifier.BiasTensor
		labels["rerank_classifier_bias_encoding"] = hipProjectionWeightEncodingLabel(classifier.BiasEncoding)
	}
	model.addClassifierLoRALabels(labels)
	return &inference.RerankResult{
		Model:   hipLoadedModelIdentity(model),
		Results: results,
		Labels:  mergeStringMaps(req.Labels, labels),
	}, nil
}

func (model *hipLoadedModel) classifyWithSequenceClassifier(ctx context.Context, prompts []string, cfg inference.GenerateConfig, classifier hipLoadedSequenceClassifierConfig) ([]inference.ClassifyResult, error) {
	embeddingCfg, err := model.loadedEmbeddingConfig()
	if err != nil {
		return nil, core.E("rocm.hip.SequenceClassify", "load f32 embedding table", err)
	}
	if embeddingCfg.HiddenSize != classifier.HiddenSize {
		return nil, core.E("rocm.hip.SequenceClassify", "embedding and classifier hidden sizes must match", nil)
	}
	table, err := model.loadedEmbeddingTable(embeddingCfg)
	if err != nil {
		return nil, err
	}
	weights, err := model.loadedClassifierWeights(classifier)
	if err != nil {
		return nil, err
	}
	bias, err := model.loadedClassifierBias(classifier)
	if err != nil {
		return nil, err
	}
	results := make([]inference.ClassifyResult, len(prompts))
	for index, prompt := range prompts {
		if core.Trim(prompt) == "" {
			return nil, core.E("rocm.hip.SequenceClassify", core.Sprintf("prompt %d is empty", index), nil)
		}
		tokenIDs := model.Encode(prompt)
		tokens, err := hipTokenEmbeddingVectors(table, embeddingCfg, tokenIDs)
		if err != nil {
			return nil, err
		}
		pooled, err := hipRunEmbeddingMeanPoolKernel(ctx, model.driver, hipEmbeddingMeanPoolRequest{
			Tokens:     tokens,
			TokenCount: len(tokenIDs),
			Dim:        embeddingCfg.HiddenSize,
		})
		if err != nil {
			return nil, err
		}
		logits, _, err := model.runSequenceClassifierProjection(ctx, classifier, pooled, weights, bias)
		if err != nil {
			return nil, err
		}
		tokenID, _, err := hipReferenceGreedySample(logits)
		if err != nil {
			return nil, err
		}
		results[index] = inference.ClassifyResult{
			Token: inference.Token{ID: int32(tokenID), Text: core.Sprintf("label_%d", tokenID)},
		}
		if cfg.ReturnLogits {
			results[index].Logits = logits
		}
	}
	return results, nil
}

func (model *hipLoadedModel) runSequenceClassifierProjection(ctx context.Context, classifier hipLoadedSequenceClassifierConfig, pooled []float32, weights hipLoadedSequenceClassifierWeights, bias []float32) ([]float32, string, error) {
	if model.classLoRA != nil {
		logits, err := model.runSequenceClassifierLoRAProjection(ctx, classifier, pooled, bias)
		return logits, hipKernelNameLoRA, err
	}
	logits, err := model.Project(ctx, hipProjectionRequest{
		Input:     pooled,
		F32:       weights.F32,
		FP16:      weights.FP16,
		Rows:      classifier.NumLabels,
		Cols:      classifier.HiddenSize,
		Bias:      bias,
		TensorKey: classifier.WeightTensor,
	})
	return logits, hipKernelNameProjection, err
}

func (model *hipLoadedModel) loadedEmbeddingConfig() (hipLoadedEmbeddingConfig, error) {
	if model == nil {
		return hipLoadedEmbeddingConfig{}, core.E("rocm.hip.EmbeddingTable", "loaded model is required", nil)
	}
	if model.driver == nil || !model.driver.Available() {
		return hipLoadedEmbeddingConfig{}, core.E("rocm.hip.EmbeddingTable", "HIP driver is not available", nil)
	}
	embedding, ok := model.findHIPTensor(isHIPEmbeddingTensor)
	if !ok {
		return hipLoadedEmbeddingConfig{}, core.E("rocm.hip.EmbeddingTable", "embedding tensor is required", nil)
	}
	if !hipTinyTensorIsFP32(embedding.info) {
		return hipLoadedEmbeddingConfig{}, core.E("rocm.hip.EmbeddingTable", "embedding tensor must be f32", nil)
	}
	vocabSize, hiddenSize, err := hipTinyTensorVocabHiddenShape(model.modelInfo, embedding.info)
	if err != nil {
		return hipLoadedEmbeddingConfig{}, core.E("rocm.hip.EmbeddingTable", "embedding shape", err)
	}
	tableCount := uint64(vocabSize) * uint64(hiddenSize)
	if _, err := hipExactUint32Bytes("embedding", embedding.info.ByteSize, tableCount*4); err != nil {
		return hipLoadedEmbeddingConfig{}, core.E("rocm.hip.EmbeddingTable", "embedding byte count", err)
	}
	if embedding.pointer == 0 {
		return hipLoadedEmbeddingConfig{}, core.E("rocm.hip.EmbeddingTable", "embedding tensor pointer is required", nil)
	}
	return hipLoadedEmbeddingConfig{
		EmbeddingPointer: embedding.pointer,
		EmbeddingBytes:   embedding.info.ByteSize,
		VocabSize:        vocabSize,
		HiddenSize:       hiddenSize,
		Family:           firstNonEmptyString(normalizeROCmArchitecture(model.modelInfo.Architecture), "unknown"),
	}, nil
}

func (model *hipLoadedModel) loadedEmbeddingTable(cfg hipLoadedEmbeddingConfig) ([]float32, error) {
	if model == nil || model.driver == nil {
		return nil, core.E("rocm.hip.EmbeddingTable", "HIP driver is nil", nil)
	}
	payload := make([]byte, cfg.EmbeddingBytes)
	if err := model.driver.CopyDeviceToHost(cfg.EmbeddingPointer, payload); err != nil {
		return nil, core.E("rocm.hip.EmbeddingTable", "copy embedding table", err)
	}
	table, err := hipFloat32PayloadValues(payload)
	if err != nil {
		return nil, core.E("rocm.hip.EmbeddingTable", "decode embedding table", err)
	}
	if len(table) != cfg.VocabSize*cfg.HiddenSize {
		return nil, core.E("rocm.hip.EmbeddingTable", "embedding table length must match vocab*hidden", nil)
	}
	if !rocmFloat32SliceFinite(table) {
		return nil, core.E("rocm.hip.EmbeddingTable", "embedding table values must be finite", nil)
	}
	return table, nil
}

func (model *hipLoadedModel) loadedSequenceClassifierConfig() (hipLoadedSequenceClassifierConfig, bool, error) {
	if model == nil {
		return hipLoadedSequenceClassifierConfig{}, false, core.E("rocm.hip.SequenceClassifier", "loaded model is required", nil)
	}
	if normalizeROCmArchitecture(model.modelInfo.Architecture) != "bert" {
		return hipLoadedSequenceClassifierConfig{}, false, nil
	}
	weight, bias, ok, hasBias := model.findHIPSequenceClassifierHead()
	if !ok {
		return hipLoadedSequenceClassifierConfig{}, false, nil
	}
	encoding, err := hipSequenceClassifierWeightEncoding(weight.info)
	if err != nil {
		return hipLoadedSequenceClassifierConfig{}, true, err
	}
	numLabels, hiddenSize, err := hipSequenceClassifierWeightShape(model.modelInfo, weight.info)
	if err != nil {
		return hipLoadedSequenceClassifierConfig{}, true, err
	}
	tableCount := uint64(numLabels) * uint64(hiddenSize)
	expectedWeightBytes := tableCount * 4
	if encoding == hipProjectionWeightEncodingFP16 {
		expectedWeightBytes = tableCount * 2
	}
	if _, err := hipExactUint32Bytes("classifier weight", weight.info.ByteSize, expectedWeightBytes); err != nil {
		return hipLoadedSequenceClassifierConfig{}, true, core.E("rocm.hip.SequenceClassifier", "classifier weight byte count", err)
	}
	if weight.pointer == 0 {
		return hipLoadedSequenceClassifierConfig{}, true, core.E("rocm.hip.SequenceClassifier", "classifier weight tensor pointer is required", nil)
	}
	cfg := hipLoadedSequenceClassifierConfig{
		WeightPointer:      weight.pointer,
		WeightBytes:        weight.info.ByteSize,
		WeightEncoding:     encoding,
		NumLabels:          numLabels,
		HiddenSize:         hiddenSize,
		WeightTensor:       weight.info.Name,
		PositiveLabelIndex: hipSequenceClassifierPositiveLabelIndex(numLabels),
	}
	if hasBias {
		biasEncoding, err := hipSequenceClassifierBiasEncoding(bias.info)
		if err != nil {
			return hipLoadedSequenceClassifierConfig{}, true, err
		}
		if err := hipSequenceClassifierBiasShape(numLabels, bias.info); err != nil {
			return hipLoadedSequenceClassifierConfig{}, true, err
		}
		expectedBiasBytes := uint64(numLabels) * 4
		if biasEncoding == hipProjectionWeightEncodingFP16 {
			expectedBiasBytes = uint64(numLabels) * 2
		}
		if _, err := hipExactUint32Bytes("classifier bias", bias.info.ByteSize, expectedBiasBytes); err != nil {
			return hipLoadedSequenceClassifierConfig{}, true, core.E("rocm.hip.SequenceClassifier", "classifier bias byte count", err)
		}
		if bias.pointer == 0 {
			return hipLoadedSequenceClassifierConfig{}, true, core.E("rocm.hip.SequenceClassifier", "classifier bias tensor pointer is required", nil)
		}
		cfg.BiasPointer = bias.pointer
		cfg.BiasBytes = bias.info.ByteSize
		cfg.BiasEncoding = biasEncoding
		cfg.BiasTensor = bias.info.Name
	}
	return cfg, true, nil
}

func (model *hipLoadedModel) loadedClassifierWeights(cfg hipLoadedSequenceClassifierConfig) (hipLoadedSequenceClassifierWeights, error) {
	payload, err := model.loadedTensorBytes("rocm.hip.SequenceClassifier", "classifier weight", cfg.WeightPointer, cfg.WeightBytes)
	if err != nil {
		return hipLoadedSequenceClassifierWeights{}, err
	}
	wantCount := cfg.NumLabels * cfg.HiddenSize
	switch cfg.WeightEncoding {
	case hipProjectionWeightEncodingF32:
		values, err := hipFloat32PayloadValues(payload)
		if err != nil {
			return hipLoadedSequenceClassifierWeights{}, core.E("rocm.hip.SequenceClassifier", "decode classifier weight", err)
		}
		if len(values) != wantCount {
			return hipLoadedSequenceClassifierWeights{}, core.E("rocm.hip.SequenceClassifier", "classifier weight length must match expected shape", nil)
		}
		return hipLoadedSequenceClassifierWeights{F32: values}, nil
	case hipProjectionWeightEncodingFP16:
		if len(payload) == 0 || len(payload)%2 != 0 {
			return hipLoadedSequenceClassifierWeights{}, core.E("rocm.hip.SequenceClassifier", "classifier fp16 payload byte length must be positive and aligned", nil)
		}
		values := make([]uint16, len(payload)/2)
		for index := range values {
			values[index] = binary.LittleEndian.Uint16(payload[index*2:])
		}
		if len(values) != wantCount {
			return hipLoadedSequenceClassifierWeights{}, core.E("rocm.hip.SequenceClassifier", "classifier weight length must match expected shape", nil)
		}
		return hipLoadedSequenceClassifierWeights{FP16: values}, nil
	default:
		return hipLoadedSequenceClassifierWeights{}, core.E("rocm.hip.SequenceClassifier", "unsupported classifier weight encoding", nil)
	}
}

func (model *hipLoadedModel) loadedClassifierBias(cfg hipLoadedSequenceClassifierConfig) ([]float32, error) {
	if cfg.BiasPointer == 0 || cfg.BiasBytes == 0 {
		return nil, nil
	}
	switch cfg.BiasEncoding {
	case hipProjectionWeightEncodingF32:
		return model.loadedF32TensorPayload("rocm.hip.SequenceClassifier", "classifier bias", cfg.BiasPointer, cfg.BiasBytes, cfg.NumLabels)
	case hipProjectionWeightEncodingFP16:
		payload, err := model.loadedTensorBytes("rocm.hip.SequenceClassifier", "classifier bias", cfg.BiasPointer, cfg.BiasBytes)
		if err != nil {
			return nil, err
		}
		if len(payload) == 0 || len(payload)%2 != 0 {
			return nil, core.E("rocm.hip.SequenceClassifier", "classifier fp16 bias byte length must be positive and aligned", nil)
		}
		values := make([]float32, len(payload)/2)
		for index := range values {
			values[index] = hipFloat16ToFloat32(binary.LittleEndian.Uint16(payload[index*2:]))
		}
		if len(values) != cfg.NumLabels {
			return nil, core.E("rocm.hip.SequenceClassifier", "classifier bias length must match expected shape", nil)
		}
		if !rocmFloat32SliceFinite(values) {
			return nil, core.E("rocm.hip.SequenceClassifier", "classifier bias values must be finite", nil)
		}
		return values, nil
	default:
		return nil, core.E("rocm.hip.SequenceClassifier", "unsupported classifier bias encoding", nil)
	}
}

func (model *hipLoadedModel) loadedF32TensorPayload(operation, label string, pointer nativeDevicePointer, sizeBytes uint64, wantCount int) ([]float32, error) {
	payload, err := model.loadedTensorBytes(operation, label, pointer, sizeBytes)
	if err != nil {
		return nil, err
	}
	values, err := hipFloat32PayloadValues(payload)
	if err != nil {
		return nil, core.E(operation, "decode "+label, err)
	}
	if len(values) != wantCount {
		return nil, core.E(operation, label+" length must match expected shape", nil)
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E(operation, label+" values must be finite", nil)
	}
	return values, nil
}

func (model *hipLoadedModel) loadedTensorBytes(operation, label string, pointer nativeDevicePointer, sizeBytes uint64) ([]byte, error) {
	if model == nil || model.driver == nil {
		return nil, core.E(operation, "HIP driver is nil", nil)
	}
	if pointer == 0 || sizeBytes == 0 {
		return nil, core.E(operation, label+" tensor is required", nil)
	}
	payload := make([]byte, sizeBytes)
	if err := model.driver.CopyDeviceToHost(pointer, payload); err != nil {
		return nil, core.E(operation, "copy "+label, err)
	}
	return payload, nil
}

func isHIPSequenceClassifierWeightTensor(name string) bool {
	_, _, ok := hipSequenceClassifierWeightCandidate(name)
	return ok
}

func isHIPSequenceClassifierBiasTensor(name string) bool {
	name = core.Lower(name)
	return name == "classifier.bias" ||
		name == "score.bias" ||
		core.HasSuffix(name, ".classifier.bias") ||
		core.HasSuffix(name, ".score.bias")
}

type hipSequenceClassifierHeadCandidate struct {
	weight   hipTensor
	priority int
	name     string
	biasName string
}

func (model *hipLoadedModel) findHIPSequenceClassifierHead() (hipTensor, hipTensor, bool, bool) {
	if model == nil {
		return hipTensor{}, hipTensor{}, false, false
	}
	tensors := make(map[string]hipTensor, len(model.tensors))
	for _, tensor := range model.tensors {
		tensors[core.Lower(tensor.info.Name)] = tensor
	}
	candidates := make([]hipSequenceClassifierHeadCandidate, 0, len(tensors))
	for name, tensor := range tensors {
		priority, biasName, ok := hipSequenceClassifierWeightCandidate(name)
		if !ok {
			continue
		}
		candidates = append(candidates, hipSequenceClassifierHeadCandidate{
			weight:   tensor,
			priority: priority,
			name:     name,
			biasName: biasName,
		})
	}
	if len(candidates) == 0 {
		return hipTensor{}, hipTensor{}, false, false
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].priority != candidates[j].priority {
			return candidates[i].priority < candidates[j].priority
		}
		return candidates[i].name < candidates[j].name
	})
	selected := candidates[0]
	bias, hasBias := tensors[selected.biasName]
	return selected.weight, bias, true, hasBias
}

func hipSequenceClassifierWeightCandidate(name string) (int, string, bool) {
	name = core.Lower(name)
	switch {
	case name == "classifier.weight":
		return 0, "classifier.bias", true
	case name == "score.weight":
		return 1, "score.bias", true
	case core.HasSuffix(name, ".classifier.weight"):
		return 2, name[:len(name)-len(".classifier.weight")] + ".classifier.bias", true
	case core.HasSuffix(name, ".score.weight"):
		return 3, name[:len(name)-len(".score.weight")] + ".score.bias", true
	default:
		return 0, "", false
	}
}

func hipSequenceClassifierWeightShape(info inference.ModelInfo, tensor nativeTensorInfo) (int, int, error) {
	if len(tensor.Dimensions) != 2 {
		return 0, 0, core.E("rocm.hip.SequenceClassifier", "classifier weight tensor must be rank 2", nil)
	}
	numLabels, err := hipTinyUint64ToInt("classifier labels", tensor.Dimensions[0])
	if err != nil {
		return 0, 0, err
	}
	hiddenSize, err := hipTinyUint64ToInt("classifier hidden size", tensor.Dimensions[1])
	if err != nil {
		return 0, 0, err
	}
	if info.HiddenSize > 0 && hiddenSize != info.HiddenSize {
		return 0, 0, core.E("rocm.hip.SequenceClassifier", core.Sprintf("classifier hidden size %d does not match model hidden size %d", hiddenSize, info.HiddenSize), nil)
	}
	return numLabels, hiddenSize, nil
}

func hipSequenceClassifierWeightEncoding(tensor nativeTensorInfo) (uint32, error) {
	switch {
	case hipTinyTensorIsFP32(tensor):
		return hipProjectionWeightEncodingF32, nil
	case hipTinyTensorIsFP16(tensor):
		return hipProjectionWeightEncodingFP16, nil
	default:
		return 0, core.E("rocm.hip.SequenceClassifier", "classifier weight tensor must be f32 or f16", nil)
	}
}

func hipSequenceClassifierBiasEncoding(tensor nativeTensorInfo) (uint32, error) {
	switch {
	case hipTinyTensorIsFP32(tensor):
		return hipProjectionWeightEncodingF32, nil
	case hipTinyTensorIsFP16(tensor):
		return hipProjectionWeightEncodingFP16, nil
	default:
		return 0, core.E("rocm.hip.SequenceClassifier", "classifier bias tensor must be f32 or f16", nil)
	}
}

func hipProjectionWeightEncodingLabel(encoding uint32) string {
	switch encoding {
	case hipProjectionWeightEncodingF32:
		return "f32"
	case hipProjectionWeightEncodingFP16:
		return "fp16"
	case hipProjectionWeightEncodingQ8:
		return "q8"
	default:
		return core.Sprintf("%d", encoding)
	}
}

func hipSequenceClassifierBiasShape(numLabels int, tensor nativeTensorInfo) error {
	if len(tensor.Dimensions) != 1 {
		return core.E("rocm.hip.SequenceClassifier", "classifier bias tensor must be rank 1", nil)
	}
	biasLabels, err := hipTinyUint64ToInt("classifier bias labels", tensor.Dimensions[0])
	if err != nil {
		return err
	}
	if biasLabels != numLabels {
		return core.E("rocm.hip.SequenceClassifier", "classifier bias length must match label count", nil)
	}
	return nil
}

func hipSequenceClassifierPositiveLabelIndex(numLabels int) int {
	if numLabels > 1 {
		return 1
	}
	return 0
}

func hipSequenceClassifierPairText(query, document string) string {
	return core.Trim(query) + " [SEP] " + core.Trim(document)
}

func hipSequenceClassifierRerankScore(logits []float32, positiveIndex int) (float32, error) {
	if len(logits) == 0 {
		return 0, core.E("rocm.hip.SequenceClassifier", "classifier logits are required", nil)
	}
	if positiveIndex < 0 || positiveIndex >= len(logits) {
		return 0, core.E("rocm.hip.SequenceClassifier", "positive label index is outside logits", nil)
	}
	return logits[positiveIndex], nil
}

func hipTokenEmbeddingVectors(table []float32, cfg hipLoadedEmbeddingConfig, tokenIDs []int32) ([]float32, error) {
	if err := hipValidateTinyTokenIDs(tokenIDs, cfg.VocabSize); err != nil {
		return nil, err
	}
	out := make([]float32, 0, len(tokenIDs)*cfg.HiddenSize)
	for _, id := range tokenIDs {
		start := int(id) * cfg.HiddenSize
		end := start + cfg.HiddenSize
		if start < 0 || end > len(table) {
			return nil, core.E("rocm.hip.EmbeddingTable", "token embedding row is outside table", nil)
		}
		out = append(out, table[start:end]...)
	}
	return out, nil
}

func hipTinyTokenEmbeddingVectors(table []float32, cfg hipLoadedTinyLMConfig, tokenIDs []int32) ([]float32, error) {
	return hipTokenEmbeddingVectors(table, hipLoadedEmbeddingConfig{
		VocabSize:  cfg.VocabSize,
		HiddenSize: cfg.HiddenSize,
	}, tokenIDs)
}

func flattenEqualFloat32Vectors(vectors [][]float32, dim int) ([]float32, error) {
	if len(vectors) == 0 {
		return nil, core.E("rocm.hip.Rerank", "document vectors are required", nil)
	}
	if dim <= 0 {
		return nil, core.E("rocm.hip.Rerank", "embedding dimension must be positive", nil)
	}
	flat := make([]float32, 0, len(vectors)*dim)
	for index, vector := range vectors {
		if len(vector) != dim {
			return nil, core.E("rocm.hip.Rerank", core.Sprintf("document vector %d dimension mismatch", index), nil)
		}
		flat = append(flat, vector...)
	}
	return flat, nil
}

func rocmRerankScoresFromCosine(scores []float32, texts []string, topN int) ([]inference.RerankScore, error) {
	if len(scores) == 0 {
		return nil, core.E("rocm.hip.Rerank", "scores are required", nil)
	}
	if len(texts) != 0 && len(texts) != len(scores) {
		return nil, core.E("rocm.hip.Rerank", "document text count must match scores", nil)
	}
	results := make([]inference.RerankScore, len(scores))
	for index, score := range scores {
		results[index] = inference.RerankScore{Index: index, Score: float64(score)}
		if len(texts) > 0 {
			results[index].Text = texts[index]
		}
	}
	sort.SliceStable(results, func(i, j int) bool {
		if results[i].Score == results[j].Score {
			return results[i].Index < results[j].Index
		}
		return results[i].Score > results[j].Score
	})
	if topN > 0 && topN < len(results) {
		results = results[:topN]
	}
	return results, nil
}

func hipLoadedModelIdentity(model *hipLoadedModel) inference.ModelIdentity {
	if model == nil {
		return inference.ModelIdentity{}
	}
	info := model.modelInfo
	identity := model.engineProfile.Model
	if rocmModelIdentityIsZero(identity) {
		identity = inference.ModelIdentity{}
	}
	if identity.Architecture == "" {
		identity.Architecture = info.Architecture
	}
	if identity.VocabSize == 0 {
		identity.VocabSize = info.VocabSize
	}
	if identity.NumLayers == 0 {
		identity.NumLayers = info.NumLayers
	}
	if identity.HiddenSize == 0 {
		identity.HiddenSize = info.HiddenSize
	}
	if identity.QuantBits == 0 {
		identity.QuantBits = info.QuantBits
	}
	if identity.QuantGroup == 0 {
		identity.QuantGroup = info.QuantGroup
	}
	if identity.ContextLength == 0 {
		identity.ContextLength = model.contextSize
	}
	identity.Labels = mergeStringMaps(identity.Labels, model.modelLabels)
	if identity.QuantType == "" {
		identity.QuantType = identity.Labels["quant_type"]
	}
	if identity.QuantType == "" && rocmIsGemma4SizeQuantIdentity(identity.Architecture) {
		identity.QuantType = identity.Labels["gemma4_quant_mode"]
	}
	return rocmGemma4ModelWithInferredPathQuant(identity)
}
