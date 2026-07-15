// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"
	"math/bits"
	"os"
	"sync"

	core "dappco.re/go"
)

type hipSmallDecodeRequest struct {
	Architecture string
	Input        []float32
	RMSWeight    []float32
	Epsilon      float32
	QueryFP16    []uint16
	KeyFP16      []uint16
	ValueFP16    []uint16
	OutputFP16   []uint16
	LMHeadFP16   []uint16
	PriorKeys    []float32
	PriorValues  []float32
	Position     int
	RoPEBase     float32
	VocabSize    int
	HiddenSize   int
}

type hipSmallDecodeResult struct {
	Logits        []float32
	Attention     []float32
	UpdatedKeys   []float32
	UpdatedValues []float32
	Projected     []float32
	TokenID       int
	Score         float32
	Labels        map[string]string
}

type hipLoadedSmallDecodeConfig struct {
	Architecture        string
	EmbeddingPointer    nativeDevicePointer
	EmbeddingBytes      uint64
	RMSWeightPointer    nativeDevicePointer
	RMSWeightBytes      uint64
	QueryWeightPointer  nativeDevicePointer
	QueryWeightBytes    uint64
	KeyWeightPointer    nativeDevicePointer
	KeyWeightBytes      uint64
	ValueWeightPointer  nativeDevicePointer
	ValueWeightBytes    uint64
	OutputWeightPointer nativeDevicePointer
	OutputWeightBytes   uint64
	LMHeadPointer       nativeDevicePointer
	LMHeadBytes         uint64
	VocabSize           int
	HiddenSize          int
}

type hipLoadedSmallDecodeRequest struct {
	Input       []float32
	PriorKeys   []float32
	PriorValues []float32
	Position    int
	RoPEBase    float32
	Epsilon     float32
}

type hipRMSNormDeviceWeightConfig struct {
	WeightPointer  nativeDevicePointer
	WeightBytes    uint64
	Count          int
	Epsilon        float32
	WeightEncoding uint32
	Flags          uint32
}

func (req hipSmallDecodeRequest) validate() error {
	if !isROCmSmallDecodeArchitecture(req.Architecture) {
		return core.E("rocm.hip.SmallDecode", "small decode smoke supports only Qwen, Gemma, or dense route architectures", nil)
	}
	if req.HiddenSize <= 0 || req.HiddenSize%2 != 0 || req.VocabSize <= 0 {
		return core.E("rocm.hip.SmallDecode", "hidden size must be positive and even and vocab size must be positive", nil)
	}
	if len(req.Input) != req.HiddenSize {
		return core.E("rocm.hip.SmallDecode", "input length must match hidden size", nil)
	}
	if len(req.RMSWeight) != req.HiddenSize {
		return core.E("rocm.hip.SmallDecode", "RMS weight length must match hidden size", nil)
	}
	if req.Epsilon < 0 || math.IsNaN(float64(req.Epsilon)) || math.IsInf(float64(req.Epsilon), 0) {
		return core.E("rocm.hip.SmallDecode", "epsilon must be non-negative and finite", nil)
	}
	if req.Position < 0 {
		return core.E("rocm.hip.SmallDecode", "position must be non-negative", nil)
	}
	if req.RoPEBase <= 0 || math.IsNaN(float64(req.RoPEBase)) || math.IsInf(float64(req.RoPEBase), 0) {
		return core.E("rocm.hip.SmallDecode", "RoPE base must be positive and finite", nil)
	}
	projectionWeights := req.HiddenSize * req.HiddenSize
	for name, weights := range map[string][]uint16{
		"query":  req.QueryFP16,
		"key":    req.KeyFP16,
		"value":  req.ValueFP16,
		"output": req.OutputFP16,
	} {
		if len(weights) != projectionWeights {
			return core.E("rocm.hip.SmallDecode", name+" projection weight length must match hidden*hidden", nil)
		}
	}
	if len(req.LMHeadFP16) != req.VocabSize*req.HiddenSize {
		return core.E("rocm.hip.SmallDecode", "LM head weight length must match vocab*hidden", nil)
	}
	if len(req.PriorKeys) == 0 || len(req.PriorValues) == 0 {
		return core.E("rocm.hip.SmallDecode", "prior key/value tensors are required", nil)
	}
	if len(req.PriorKeys) != len(req.PriorValues) || len(req.PriorKeys)%req.HiddenSize != 0 {
		return core.E("rocm.hip.SmallDecode", "prior key/value tensors must align with hidden size", nil)
	}
	if req.Position != len(req.PriorKeys)/req.HiddenSize {
		return core.E("rocm.hip.SmallDecode", "decode position must equal prior KV token count", nil)
	}
	return nil
}

func (model *hipLoadedModel) loadedSmallDecodeConfig() (hipLoadedSmallDecodeConfig, error) {
	if model == nil {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "loaded model is required", nil)
	}
	if model.driver == nil || !model.driver.Available() {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "HIP driver is not available", nil)
	}
	architecture := normalizeROCmArchitecture(model.modelInfo.Architecture)
	if !isROCmSmallDecodeArchitecture(architecture) {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "small decode smoke supports only Qwen, Gemma, or dense route architectures", nil)
	}
	hiddenSize := model.modelInfo.HiddenSize
	vocabSize := model.modelInfo.VocabSize
	if hiddenSize <= 0 || hiddenSize%2 != 0 || vocabSize <= 0 {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "model hidden size must be positive and even and vocab size must be positive", nil)
	}
	embedding, ok := model.findHIPTensor(isHIPEmbeddingTensor)
	if !ok {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "embedding tensor is required", nil)
	}
	rms, ok := model.findHIPTensor(hipSmallDecodeRMSWeightTensor)
	if !ok {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "input RMSNorm weight tensor is required", nil)
	}
	query, ok := model.findHIPTensor(hipSmallDecodeProjectionTensor("q_proj"))
	if !ok {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "query projection tensor is required", nil)
	}
	key, ok := model.findHIPTensor(hipSmallDecodeProjectionTensor("k_proj"))
	if !ok {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "key projection tensor is required", nil)
	}
	value, ok := model.findHIPTensor(hipSmallDecodeProjectionTensor("v_proj"))
	if !ok {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "value projection tensor is required", nil)
	}
	output, ok := model.findHIPTensor(hipSmallDecodeProjectionTensor("o_proj"))
	if !ok {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "output projection tensor is required", nil)
	}
	lmHead, ok := model.findHIPTensor(isHIPOutputTensor)
	if !ok {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "LM head tensor is required", nil)
	}
	if err := hipLoadedSmallDecodeRMSWeight(rms, hiddenSize); err != nil {
		return hipLoadedSmallDecodeConfig{}, err
	}
	if err := hipLoadedSmallDecodeEmbedding(embedding, vocabSize, hiddenSize); err != nil {
		return hipLoadedSmallDecodeConfig{}, err
	}
	for label, tensor := range map[string]hipTensor{
		"query":  query,
		"key":    key,
		"value":  value,
		"output": output,
	} {
		if err := hipLoadedSmallDecodeFP16Matrix(label, tensor, hiddenSize, hiddenSize); err != nil {
			return hipLoadedSmallDecodeConfig{}, err
		}
	}
	if err := hipLoadedSmallDecodeFP16Matrix("LM head", lmHead, vocabSize, hiddenSize); err != nil {
		return hipLoadedSmallDecodeConfig{}, err
	}
	return hipLoadedSmallDecodeConfig{
		Architecture:        architecture,
		EmbeddingPointer:    embedding.pointer,
		EmbeddingBytes:      embedding.info.ByteSize,
		RMSWeightPointer:    rms.pointer,
		RMSWeightBytes:      rms.info.ByteSize,
		QueryWeightPointer:  query.pointer,
		QueryWeightBytes:    query.info.ByteSize,
		KeyWeightPointer:    key.pointer,
		KeyWeightBytes:      key.info.ByteSize,
		ValueWeightPointer:  value.pointer,
		ValueWeightBytes:    value.info.ByteSize,
		OutputWeightPointer: output.pointer,
		OutputWeightBytes:   output.info.ByteSize,
		LMHeadPointer:       lmHead.pointer,
		LMHeadBytes:         lmHead.info.ByteSize,
		VocabSize:           vocabSize,
		HiddenSize:          hiddenSize,
	}, nil
}

func hipLoadedSmallDecodeEmbedding(tensor hipTensor, vocabSize, hiddenSize int) error {
	if tensor.pointer == 0 {
		return core.E("rocm.hip.SmallDecode", "embedding pointer is required", nil)
	}
	if !hipTinyTensorIsFP32(tensor.info) {
		return core.E("rocm.hip.SmallDecode", "embedding tensor must be f32", nil)
	}
	if len(tensor.info.Dimensions) != 2 || tensor.info.Dimensions[0] != uint64(vocabSize) || tensor.info.Dimensions[1] != uint64(hiddenSize) {
		return core.E("rocm.hip.SmallDecode", "embedding tensor shape must be vocab-major vocab*hidden", nil)
	}
	if _, err := hipExactUint32Bytes("embedding", tensor.info.ByteSize, uint64(vocabSize)*uint64(hiddenSize)*4); err != nil {
		return core.E("rocm.hip.SmallDecode", "embedding byte count", err)
	}
	return nil
}

func hipSmallDecodeRMSWeightTensor(name string) bool {
	return core.Contains(name, "layers.0") &&
		core.Contains(name, "weight") &&
		(core.Contains(name, "input_layernorm") || core.Contains(name, "attention_norm"))
}

func hipSmallDecodeProjectionTensor(kind string) func(string) bool {
	return func(name string) bool {
		return core.Contains(name, "layers.0") &&
			core.Contains(name, kind) &&
			core.Contains(name, "weight")
	}
}

func hipLoadedSmallDecodeRMSWeight(tensor hipTensor, hiddenSize int) error {
	if tensor.pointer == 0 {
		return core.E("rocm.hip.SmallDecode", "RMSNorm weight pointer is required", nil)
	}
	if !hipTinyTensorIsFP32(tensor.info) {
		return core.E("rocm.hip.SmallDecode", "RMSNorm weight must be f32", nil)
	}
	if len(tensor.info.Dimensions) != 1 || tensor.info.Dimensions[0] != uint64(hiddenSize) {
		return core.E("rocm.hip.SmallDecode", "RMSNorm weight shape must match hidden size", nil)
	}
	if _, err := hipExactUint32Bytes("RMSNorm weight", tensor.info.ByteSize, uint64(hiddenSize)*4); err != nil {
		return core.E("rocm.hip.SmallDecode", "RMSNorm weight byte count", err)
	}
	return nil
}

func hipLoadedSmallDecodeFP16Matrix(label string, tensor hipTensor, rows, cols int) error {
	if tensor.pointer == 0 {
		return core.E("rocm.hip.SmallDecode", label+" weight pointer is required", nil)
	}
	if !hipTinyTensorIsFP16(tensor.info) {
		return core.E("rocm.hip.SmallDecode", label+" weight must be f16", nil)
	}
	if len(tensor.info.Dimensions) != 2 || tensor.info.Dimensions[0] != uint64(rows) || tensor.info.Dimensions[1] != uint64(cols) {
		return core.E("rocm.hip.SmallDecode", label+" weight shape must be row-major rows*cols", nil)
	}
	if _, err := hipExactUint32Bytes(label+" weight", tensor.info.ByteSize, uint64(rows)*uint64(cols)*2); err != nil {
		return core.E("rocm.hip.SmallDecode", label+" weight byte count", err)
	}
	return nil
}

func hipReferenceSmallDecode(req hipSmallDecodeRequest) (hipSmallDecodeResult, error) {
	if err := req.validate(); err != nil {
		return hipSmallDecodeResult{}, err
	}
	normalized, err := hipReferenceRMSNorm(req.Input, req.RMSWeight, req.Epsilon)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	query, err := hipReferenceFP16Projection(normalized, req.QueryFP16, req.HiddenSize, req.HiddenSize, nil)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	key, err := hipReferenceFP16Projection(normalized, req.KeyFP16, req.HiddenSize, req.HiddenSize, nil)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	value, err := hipReferenceFP16Projection(normalized, req.ValueFP16, req.HiddenSize, req.HiddenSize, nil)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	ropeQuery, err := hipReferenceRoPE(query, req.Position, float64(req.RoPEBase))
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	ropeKey, err := hipReferenceRoPE(key, req.Position, float64(req.RoPEBase))
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	priorKeys, err := splitHIPReferenceVectors(req.PriorKeys, req.HiddenSize)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	priorValues, err := splitHIPReferenceVectors(req.PriorValues, req.HiddenSize)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	attentionOutput, attention, updatedKeys, updatedValues, err := hipReferenceDecodeWithKV(ropeQuery, ropeKey, value, priorKeys, priorValues)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	projected, err := hipReferenceFP16Projection(attentionOutput, req.OutputFP16, req.HiddenSize, req.HiddenSize, nil)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	logits, err := hipReferenceFP16Projection(projected, req.LMHeadFP16, req.VocabSize, req.HiddenSize, nil)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	tokenID, score, err := hipReferenceGreedySample(logits)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	return hipSmallDecodeResult{
		Logits:        logits,
		Attention:     attention,
		UpdatedKeys:   flattenHIPReferenceMatrix(updatedKeys),
		UpdatedValues: flattenHIPReferenceMatrix(updatedValues),
		Projected:     projected,
		TokenID:       tokenID,
		Score:         score,
		Labels:        hipSmallDecodeLabels(req),
	}, nil
}

func hipRunLoadedSmallDecode(ctx context.Context, driver nativeHIPDriver, cfg hipLoadedSmallDecodeConfig, req hipLoadedSmallDecodeRequest) (hipSmallDecodeResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipSmallDecodeResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipSmallDecodeResult{}, core.E("rocm.hip.SmallDecode", "HIP driver is not available", nil)
	}
	if err := cfg.validate(); err != nil {
		return hipSmallDecodeResult{}, err
	}
	if err := req.validate(cfg); err != nil {
		return hipSmallDecodeResult{}, err
	}
	normalized, err := hipRunRMSNormKernelWithDeviceWeight(ctx, driver, req.Input, cfg.RMSWeightPointer, cfg.RMSWeightBytes, cfg.HiddenSize, req.Epsilon)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	query, err := hipRunProjectionKernelWithDeviceWeight(ctx, driver, normalized, cfg.QueryWeightPointer, cfg.QueryWeightBytes, cfg.HiddenSize, cfg.HiddenSize)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	key, err := hipRunProjectionKernelWithDeviceWeight(ctx, driver, normalized, cfg.KeyWeightPointer, cfg.KeyWeightBytes, cfg.HiddenSize, cfg.HiddenSize)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	value, err := hipRunProjectionKernelWithDeviceWeight(ctx, driver, normalized, cfg.ValueWeightPointer, cfg.ValueWeightBytes, cfg.HiddenSize, cfg.HiddenSize)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	ropeQuery, err := hipRunRoPEKernel(ctx, driver, hipRoPERequest{Input: query, Position: req.Position, Base: req.RoPEBase})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	ropeKey, err := hipRunRoPEKernel(ctx, driver, hipRoPERequest{Input: key, Position: req.Position, Base: req.RoPEBase})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	updatedKeys := append(append([]float32(nil), req.PriorKeys...), ropeKey...)
	updatedValues := append(append([]float32(nil), req.PriorValues...), value...)
	attention, err := hipRunAttentionKernel(ctx, driver, hipAttentionRequest{Query: ropeQuery, Keys: updatedKeys, Values: updatedValues})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	projected, err := hipRunProjectionKernelWithDeviceWeight(ctx, driver, attention.Output, cfg.OutputWeightPointer, cfg.OutputWeightBytes, cfg.HiddenSize, cfg.HiddenSize)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	logits, err := hipRunProjectionKernelWithDeviceWeight(ctx, driver, projected, cfg.LMHeadPointer, cfg.LMHeadBytes, cfg.VocabSize, cfg.HiddenSize)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	greedy, err := hipRunGreedyKernel(ctx, driver, hipGreedySampleRequest{Logits: logits})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	return hipSmallDecodeResult{
		Logits:        logits,
		Attention:     attention.Weights,
		UpdatedKeys:   updatedKeys,
		UpdatedValues: updatedValues,
		Projected:     projected,
		TokenID:       greedy.TokenID,
		Score:         greedy.Score,
		Labels:        hipLoadedSmallDecodeLabels(cfg, req),
	}, nil
}

func hipRunLoadedSmallDecodeToken(ctx context.Context, model *hipLoadedModel, cfg hipLoadedSmallDecodeConfig, req hipDecodeRequest) (hipDecodeResult, error) {
	if model == nil {
		return hipDecodeResult{}, core.E("rocm.hip.SmallDecode", "loaded model is required", nil)
	}
	if err := req.validate(); err != nil {
		return hipDecodeResult{}, err
	}
	if int(req.TokenID) >= cfg.VocabSize {
		return hipDecodeResult{}, core.E("rocm.hip.SmallDecode", "token ID is outside vocabulary", nil)
	}
	keyWidth, valueWidth, err := req.kvVectorWidths()
	if err != nil {
		return hipDecodeResult{}, err
	}
	if keyWidth != cfg.HiddenSize || valueWidth != cfg.HiddenSize {
		return hipDecodeResult{}, core.E("rocm.hip.SmallDecode", "KV widths must match hidden size", nil)
	}
	priorKeys, priorValues, err := model.restoreLoadedSmallDecodePriorKV(req, keyWidth, valueWidth)
	if err != nil {
		return hipDecodeResult{}, err
	}
	input, err := hipReadLoadedSmallEmbedding(ctx, model.driver, cfg, req.TokenID)
	if err != nil {
		return hipDecodeResult{}, err
	}
	output, err := hipRunLoadedSmallDecode(ctx, model.driver, cfg, hipLoadedSmallDecodeRequest{
		Input:       input,
		PriorKeys:   priorKeys,
		PriorValues: priorValues,
		Position:    req.KV.TokenCount(),
		RoPEBase:    10000,
		Epsilon:     0,
	})
	if err != nil {
		return hipDecodeResult{}, err
	}
	if model.smallLoRA != nil {
		logits, tokenID, score, err := model.runSmallLoRAProjection(ctx, cfg, output.Projected)
		if err != nil {
			return hipDecodeResult{}, err
		}
		output.Logits = logits
		output.TokenID = tokenID
		output.Score = score
		model.addSmallLoRALabels(output.Labels)
	}
	targetKV := req.KV
	if req.DeviceKV != nil {
		cloned, err := req.KV.Clone()
		if err != nil {
			return hipDecodeResult{}, err
		}
		targetKV = cloned
	}
	keyStart := len(output.UpdatedKeys) - cfg.HiddenSize
	valueStart := len(output.UpdatedValues) - cfg.HiddenSize
	if err := targetKV.AppendToken(targetKV.TokenCount(), output.UpdatedKeys[keyStart:], output.UpdatedValues[valueStart:]); err != nil {
		return hipDecodeResult{}, err
	}
	labels := output.Labels
	labels["decode_launch_token"] = core.Sprintf("%d", req.TokenID)
	var deviceKV *rocmDeviceKVCache
	var descriptorTable *rocmDeviceKVDescriptorTable
	if req.DeviceKV != nil {
		device, table, err := hipAppendDecodeDeviceKV(ctx, req, output.UpdatedKeys[keyStart:], output.UpdatedValues[valueStart:], labels)
		if err != nil {
			return hipDecodeResult{}, err
		}
		deviceKV = device
		descriptorTable = table
	}
	return hipDecodeResult{
		Token:           hipTinyToken(model, int32(output.TokenID)),
		Logits:          output.Logits,
		KV:              targetKV,
		DeviceKV:        deviceKV,
		DescriptorTable: descriptorTable,
		Labels:          labels,
	}, nil
}

func (model *hipLoadedModel) restoreLoadedSmallDecodePriorKV(req hipDecodeRequest, keyWidth, valueWidth int) ([]float32, []float32, error) {
	if model == nil {
		return nil, nil, core.E("rocm.hip.SmallDecode", "loaded model is required", nil)
	}
	if req.KV == nil {
		return nil, nil, core.E("rocm.hip.SmallDecode", "KV cache is required", nil)
	}
	tokenCount := req.KV.TokenCount()
	if tokenCount <= 0 {
		return nil, nil, core.E("rocm.hip.SmallDecode", "KV cache must contain prior tokens", nil)
	}
	keyCount := tokenCount * keyWidth
	valueCount := tokenCount * valueWidth
	if cap(model.smallPriorKeys) < keyCount {
		model.smallPriorKeys = make([]float32, keyCount)
	}
	if cap(model.smallPriorValues) < valueCount {
		model.smallPriorValues = make([]float32, valueCount)
	}
	model.smallPriorKeys = model.smallPriorKeys[:keyCount]
	model.smallPriorValues = model.smallPriorValues[:valueCount]
	return req.KV.RestoreInto(0, tokenCount, model.smallPriorKeys, model.smallPriorValues)
}

func hipReadLoadedSmallEmbedding(ctx context.Context, driver nativeHIPDriver, cfg hipLoadedSmallDecodeConfig, tokenID int32) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.SmallDecode", "HIP driver is not available", nil)
	}
	if tokenID < 0 || int(tokenID) >= cfg.VocabSize {
		return nil, core.E("rocm.hip.SmallDecode", "token ID is outside vocabulary", nil)
	}
	rowBytes := uint64(cfg.HiddenSize * 4)
	offset := uint64(tokenID) * rowBytes
	if offset+rowBytes > cfg.EmbeddingBytes {
		return nil, core.E("rocm.hip.SmallDecode", "embedding row exceeds tensor byte size", nil)
	}
	payload := make([]byte, rowBytes)
	pointer := nativeDevicePointer(uintptr(cfg.EmbeddingPointer) + uintptr(offset))
	if err := driver.CopyDeviceToHost(pointer, payload); err != nil {
		return nil, core.E("rocm.hip.SmallDecode", "copy embedding row", err)
	}
	values, err := hipFloat32PayloadValues(payload)
	if err != nil {
		return nil, err
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E("rocm.hip.SmallDecode", "embedding row values must be finite", nil)
	}
	return values, nil
}

func (cfg hipLoadedSmallDecodeConfig) validate() error {
	if !isROCmSmallDecodeArchitecture(cfg.Architecture) {
		return core.E("rocm.hip.SmallDecode", "small decode smoke supports only Qwen, Gemma, or dense route architectures", nil)
	}
	if cfg.HiddenSize <= 0 || cfg.HiddenSize%2 != 0 || cfg.VocabSize <= 0 {
		return core.E("rocm.hip.SmallDecode", "hidden size must be positive and even and vocab size must be positive", nil)
	}
	if cfg.EmbeddingPointer == 0 || cfg.RMSWeightPointer == 0 || cfg.QueryWeightPointer == 0 || cfg.KeyWeightPointer == 0 ||
		cfg.ValueWeightPointer == 0 || cfg.OutputWeightPointer == 0 || cfg.LMHeadPointer == 0 {
		return core.E("rocm.hip.SmallDecode", "loaded weight pointers are required", nil)
	}
	if _, err := hipExactUint32Bytes("embedding", cfg.EmbeddingBytes, uint64(cfg.VocabSize)*uint64(cfg.HiddenSize)*4); err != nil {
		return core.E("rocm.hip.SmallDecode", "embedding byte count", err)
	}
	if _, err := hipExactUint32Bytes("RMSNorm weight", cfg.RMSWeightBytes, uint64(cfg.HiddenSize)*4); err != nil {
		return core.E("rocm.hip.SmallDecode", "RMSNorm weight byte count", err)
	}
	for label, bytes := range map[string]uint64{
		"query":  cfg.QueryWeightBytes,
		"key":    cfg.KeyWeightBytes,
		"value":  cfg.ValueWeightBytes,
		"output": cfg.OutputWeightBytes,
	} {
		if _, err := hipExactUint32Bytes(label+" weight", bytes, uint64(cfg.HiddenSize)*uint64(cfg.HiddenSize)*2); err != nil {
			return core.E("rocm.hip.SmallDecode", label+" weight byte count", err)
		}
	}
	if _, err := hipExactUint32Bytes("LM head weight", cfg.LMHeadBytes, uint64(cfg.VocabSize)*uint64(cfg.HiddenSize)*2); err != nil {
		return core.E("rocm.hip.SmallDecode", "LM head weight byte count", err)
	}
	return nil
}

func (req hipLoadedSmallDecodeRequest) validate(cfg hipLoadedSmallDecodeConfig) error {
	if len(req.Input) != cfg.HiddenSize {
		return core.E("rocm.hip.SmallDecode", "input length must match hidden size", nil)
	}
	if req.Epsilon < 0 || math.IsNaN(float64(req.Epsilon)) || math.IsInf(float64(req.Epsilon), 0) {
		return core.E("rocm.hip.SmallDecode", "epsilon must be non-negative and finite", nil)
	}
	if req.Position < 0 {
		return core.E("rocm.hip.SmallDecode", "position must be non-negative", nil)
	}
	if req.RoPEBase <= 0 || math.IsNaN(float64(req.RoPEBase)) || math.IsInf(float64(req.RoPEBase), 0) {
		return core.E("rocm.hip.SmallDecode", "RoPE base must be positive and finite", nil)
	}
	if len(req.PriorKeys) == 0 || len(req.PriorValues) == 0 {
		return core.E("rocm.hip.SmallDecode", "prior key/value tensors are required", nil)
	}
	if len(req.PriorKeys) != len(req.PriorValues) || len(req.PriorKeys)%cfg.HiddenSize != 0 {
		return core.E("rocm.hip.SmallDecode", "prior key/value tensors must align with hidden size", nil)
	}
	if req.Position != len(req.PriorKeys)/cfg.HiddenSize {
		return core.E("rocm.hip.SmallDecode", "decode position must equal prior KV token count", nil)
	}
	return nil
}

func hipRunSmallDecode(ctx context.Context, driver nativeHIPDriver, req hipSmallDecodeRequest) (hipSmallDecodeResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipSmallDecodeResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipSmallDecodeResult{}, core.E("rocm.hip.SmallDecode", "HIP driver is not available", nil)
	}
	if err := req.validate(); err != nil {
		return hipSmallDecodeResult{}, err
	}
	normalized, err := hipRunRMSNormKernel(ctx, driver, hipRMSNormRequest{Input: req.Input, Weight: req.RMSWeight, Epsilon: req.Epsilon})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	query, err := hipRunProjectionKernel(ctx, driver, hipProjectionRequest{Input: normalized, FP16: req.QueryFP16, Rows: req.HiddenSize, Cols: req.HiddenSize})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	key, err := hipRunProjectionKernel(ctx, driver, hipProjectionRequest{Input: normalized, FP16: req.KeyFP16, Rows: req.HiddenSize, Cols: req.HiddenSize})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	value, err := hipRunProjectionKernel(ctx, driver, hipProjectionRequest{Input: normalized, FP16: req.ValueFP16, Rows: req.HiddenSize, Cols: req.HiddenSize})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	ropeQuery, err := hipRunRoPEKernel(ctx, driver, hipRoPERequest{Input: query, Position: req.Position, Base: req.RoPEBase})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	ropeKey, err := hipRunRoPEKernel(ctx, driver, hipRoPERequest{Input: key, Position: req.Position, Base: req.RoPEBase})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	updatedKeys := append(append([]float32(nil), req.PriorKeys...), ropeKey...)
	updatedValues := append(append([]float32(nil), req.PriorValues...), value...)
	attention, err := hipRunAttentionKernel(ctx, driver, hipAttentionRequest{Query: ropeQuery, Keys: updatedKeys, Values: updatedValues})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	projected, err := hipRunProjectionKernel(ctx, driver, hipProjectionRequest{Input: attention.Output, FP16: req.OutputFP16, Rows: req.HiddenSize, Cols: req.HiddenSize})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	logits, err := hipRunProjectionKernel(ctx, driver, hipProjectionRequest{Input: projected, FP16: req.LMHeadFP16, Rows: req.VocabSize, Cols: req.HiddenSize})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	greedy, err := hipRunGreedyKernel(ctx, driver, hipGreedySampleRequest{Logits: logits})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	return hipSmallDecodeResult{
		Logits:        logits,
		Attention:     attention.Weights,
		UpdatedKeys:   updatedKeys,
		UpdatedValues: updatedValues,
		Projected:     projected,
		TokenID:       greedy.TokenID,
		Score:         greedy.Score,
		Labels:        hipSmallDecodeLabels(req),
	}, nil
}

func hipSmallDecodeLabels(req hipSmallDecodeRequest) map[string]string {
	return map[string]string{
		"decode_kernel":       hipKernelStatusLinked,
		"decode_kernel_name":  "rocm_small_decode_smoke",
		"decode_architecture": normalizeROCmArchitecture(req.Architecture),
		"decode_family":       hipSmallDecodeFamily(req.Architecture),
		"decode_position":     core.Sprintf("%d", req.Position),
		"decode_vocab_size":   core.Sprintf("%d", req.VocabSize),
		"decode_hidden_size":  core.Sprintf("%d", req.HiddenSize),
		"decode_primitives":   "rms_norm,projection,rope,attention,greedy",
	}
}

func isROCmSmallDecodeArchitecture(architecture string) bool {
	switch normalizeROCmArchitecture(architecture) {
	case "qwen2", "qwen3", "qwen3_6", "qwen3_next",
		"gemma", "gemma2", "gemma3", "gemma3_text", "gemma4", "gemma4_text":
		return true
	default:
		return isROCmDenseQuickWinArchitecture(architecture)
	}
}

func hipSmallDecodeFamily(architecture string) string {
	if isROCmDenseQuickWinArchitecture(architecture) {
		return "dense_route"
	}
	switch normalizeROCmArchitecture(architecture) {
	case "qwen2", "qwen3", "qwen3_6", "qwen3_next":
		return "qwen"
	case "gemma", "gemma2", "gemma3", "gemma3_text", "gemma4", "gemma4_text":
		return "gemma"
	default:
		return "unknown"
	}
}

func hipSmallDecodeLoRAModelStatus(architecture string) string {
	switch normalizeROCmArchitecture(architecture) {
	case "qwen2", "qwen3", "qwen3_6", "qwen3_next",
		"gemma", "gemma2", "gemma3", "gemma3_text", "gemma4", "gemma4_text":
		return "experimental_qwen_gemma_small_decode"
	}
	if isROCmDenseQuickWinArchitecture(architecture) {
		return "experimental_dense_small_decode"
	}
	return "experimental_qwen_gemma_small_decode"
}

func hipLoadedSmallDecodeLabels(cfg hipLoadedSmallDecodeConfig, req hipLoadedSmallDecodeRequest) map[string]string {
	labels := map[string]string{
		"decode_tensor_backing": "loaded_device",
		"decode_position":       core.Sprintf("%d", req.Position),
		"decode_vocab_size":     core.Sprintf("%d", cfg.VocabSize),
		"decode_hidden_size":    core.Sprintf("%d", cfg.HiddenSize),
	}
	for key, value := range hipSmallDecodeLabels(hipSmallDecodeRequest{Architecture: cfg.Architecture, Position: req.Position, VocabSize: cfg.VocabSize, HiddenSize: cfg.HiddenSize}) {
		labels[key] = value
	}
	return labels
}

func hipRunProjectionKernel(ctx context.Context, driver nativeHIPDriver, req hipProjectionRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.projectionDeviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.projectionLaunchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipProjectionLaunchConfig(launchBytes, req.Rows)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func hipRunProjectionKernelWithDeviceWeight(ctx context.Context, driver nativeHIPDriver, input []float32, weightPointer nativeDevicePointer, weightBytes uint64, rows, cols int) ([]float32, error) {
	return hipRunProjectionKernelWithDeviceWeightEncoding(ctx, driver, input, weightPointer, weightBytes, rows, cols, hipProjectionWeightEncodingFP16)
}

func hipRunProjectionKernelWithDeviceWeightEncoding(ctx context.Context, driver nativeHIPDriver, input []float32, weightPointer nativeDevicePointer, weightBytes uint64, rows, cols int, encoding uint32) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	weightElements, err := hipProjectionDeviceWeightElementCount(weightBytes, encoding)
	if err != nil {
		return nil, err
	}
	if err := validateHIPProjectionShape(len(input), weightElements, 0, rows, cols); err != nil {
		return nil, err
	}
	inputPayload, err := hipFloat32Payload(input)
	if err != nil {
		return nil, core.E("rocm.hip.ProjectionLaunch", "encode input", err)
	}
	inputBuffer, err := hipUploadByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection input", inputPayload, len(input))
	if err != nil {
		return nil, err
	}
	defer inputBuffer.Close()
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection output", uint64(rows*4), rows)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	launchBytes, err := (hipProjectionLaunchArgs{
		InputPointer:   inputBuffer.Pointer(),
		InputCount:     len(input),
		InputBytes:     inputBuffer.SizeBytes(),
		WeightPointer:  weightPointer,
		WeightBytes:    weightBytes,
		OutputPointer:  output.Pointer(),
		OutputBytes:    output.SizeBytes(),
		Rows:           rows,
		Cols:           cols,
		WeightEncoding: encoding,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipProjectionLaunchConfig(launchBytes, rows)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	return hipReadFloat32DeviceOutput(output, "rocm.hip.ProjectionLaunch", "projection output", rows)
}

func hipRunProjectionKernelWithDeviceInputWeightEncoding(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, weightPointer nativeDevicePointer, weightBytes uint64, rows, cols int, encoding uint32) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "projection device input is required", nil)
	}
	weightElements, err := hipProjectionDeviceWeightElementCount(weightBytes, encoding)
	if err != nil {
		return nil, err
	}
	if err := validateHIPProjectionShape(input.Count(), weightElements, 0, rows, cols); err != nil {
		return nil, err
	}
	if input.SizeBytes() != uint64(cols*4) {
		return nil, core.E("rocm.hip.ProjectionLaunch", "projection device input byte count mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection output", uint64(rows*4), rows)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunProjectionKernelWithDeviceInputWeightEncodingOutput(ctx, driver, input, weightPointer, weightBytes, rows, cols, encoding, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunProjectionKernelWithDeviceInputWeightEncodingOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, weightPointer nativeDevicePointer, weightBytes uint64, rows, cols int, encoding uint32, output *hipDeviceByteBuffer) error {
	return hipRunProjectionKernelWithDeviceInputWeightEncodingOutputWithWorkspace(ctx, driver, input, weightPointer, weightBytes, rows, cols, encoding, output, nil)
}

func hipRunProjectionKernelWithDeviceInputWeightEncodingOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, weightPointer nativeDevicePointer, weightBytes uint64, rows, cols int, encoding uint32, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.ProjectionLaunch", "projection device input is required", nil)
	}
	weightElements, err := hipProjectionDeviceWeightElementCount(weightBytes, encoding)
	if err != nil {
		return err
	}
	if err := validateHIPProjectionShape(input.Count(), weightElements, 0, rows, cols); err != nil {
		return err
	}
	if input.SizeBytes() != uint64(cols*4) {
		return core.E("rocm.hip.ProjectionLaunch", "projection device input byte count mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != rows || output.SizeBytes() != uint64(rows*4) {
		return core.E("rocm.hip.ProjectionLaunch", "projection output shape mismatch", nil)
	}
	launchArgs := hipProjectionLaunchArgs{
		InputPointer:   input.Pointer(),
		InputCount:     input.Count(),
		InputBytes:     input.SizeBytes(),
		WeightPointer:  weightPointer,
		WeightBytes:    weightBytes,
		OutputPointer:  output.Pointer(),
		OutputBytes:    output.SizeBytes(),
		Rows:           rows,
		Cols:           cols,
		WeightEncoding: encoding,
	}
	var launchBytes []byte
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.MoE.RouterProjectionArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config, err := hipProjectionLaunchConfig(launchBytes, rows)
	if err != nil {
		return err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return err
	}
	return nil
}

func hipRunProjectionBatchKernelWithDeviceInputWeightEncoding(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, weightPointer nativeDevicePointer, weightBytes uint64, rows, cols int, encoding uint32, batch int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", "projection batch device input is required", nil)
	}
	if batch <= 0 {
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", "projection batch size must be positive", nil)
	}
	weightElements, err := hipProjectionDeviceWeightElementCount(weightBytes, encoding)
	if err != nil {
		return nil, err
	}
	if err := validateHIPProjectionShape(cols, weightElements, 0, rows, cols); err != nil {
		return nil, err
	}
	if input.Count() != cols*batch || input.SizeBytes() != uint64(cols*batch*4) {
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", "projection batch device input shape mismatch", nil)
	}
	outputCount := rows * batch
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.ProjectionBatchLaunch", "projection batch output", uint64(outputCount*4), outputCount)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipProjectionBatchLaunchArgs{
		InputPointer:   input.Pointer(),
		WeightPointer:  weightPointer,
		WeightBytes:    weightBytes,
		OutputPointer:  output.Pointer(),
		InputBytes:     input.SizeBytes(),
		OutputBytes:    output.SizeBytes(),
		Rows:           rows,
		Cols:           cols,
		Batch:          batch,
		WeightEncoding: encoding,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipProjectionBatchLaunchConfig(launchBytes, rows, batch)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunProjectionBatchKernelWithDeviceInputWeightEncodingOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, weightPointer nativeDevicePointer, weightBytes uint64, rows, cols int, encoding uint32, batch int, output *hipDeviceByteBuffer) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.ProjectionBatchLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.ProjectionBatchLaunch", "projection batch device input is required", nil)
	}
	if batch <= 0 {
		return core.E("rocm.hip.ProjectionBatchLaunch", "projection batch size must be positive", nil)
	}
	weightElements, err := hipProjectionDeviceWeightElementCount(weightBytes, encoding)
	if err != nil {
		return err
	}
	if err := validateHIPProjectionShape(cols, weightElements, 0, rows, cols); err != nil {
		return err
	}
	if input.Count() != cols*batch || input.SizeBytes() != uint64(cols*batch*4) {
		return core.E("rocm.hip.ProjectionBatchLaunch", "projection batch device input shape mismatch", nil)
	}
	outputCount := rows * batch
	if output == nil || output.Pointer() == 0 || output.Count() != outputCount || output.SizeBytes() != uint64(outputCount*4) {
		return core.E("rocm.hip.ProjectionBatchLaunch", "projection batch output shape mismatch", nil)
	}
	launchBytes, err := (hipProjectionBatchLaunchArgs{
		InputPointer:   input.Pointer(),
		WeightPointer:  weightPointer,
		WeightBytes:    weightBytes,
		OutputPointer:  output.Pointer(),
		InputBytes:     input.SizeBytes(),
		OutputBytes:    output.SizeBytes(),
		Rows:           rows,
		Cols:           cols,
		Batch:          batch,
		WeightEncoding: encoding,
	}).Binary()
	if err != nil {
		return err
	}
	config, err := hipProjectionBatchLaunchConfig(launchBytes, rows, batch)
	if err != nil {
		return err
	}
	return hipLaunchKernelContext(ctx, driver, config)
}

func hipProjectionDeviceWeightElementCount(weightBytes uint64, encoding uint32) (int, error) {
	if weightBytes == 0 {
		return 0, core.E("rocm.hip.ProjectionLaunch", "projection weight bytes are required", nil)
	}
	var bytesPerElement uint64
	switch encoding {
	case hipProjectionWeightEncodingFP16, hipProjectionWeightEncodingBF16:
		bytesPerElement = 2
	case hipProjectionWeightEncodingQ8:
		bytesPerElement = 1
	case hipProjectionWeightEncodingF32:
		bytesPerElement = 4
	default:
		return 0, core.E("rocm.hip.ProjectionLaunch", core.Sprintf("unsupported projection weight encoding %d", encoding), nil)
	}
	if weightBytes%bytesPerElement != 0 {
		return 0, core.E("rocm.hip.ProjectionLaunch", "projection weight byte count must be element-aligned", nil)
	}
	elements := weightBytes / bytesPerElement
	if elements > uint64(int(^uint(0)>>1)) {
		return 0, core.E("rocm.hip.ProjectionLaunch", "projection weight element count is out of int range", nil)
	}
	return int(elements), nil
}

func hipRunRMSNormKernel(ctx context.Context, driver nativeHIPDriver, req hipRMSNormRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipSingleBlockLaunchConfig(hipKernelNameRMSNorm, launchBytes, 256)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func hipRunRMSNormKernelWithDeviceWeight(ctx context.Context, driver nativeHIPDriver, input []float32, weightPointer nativeDevicePointer, weightBytes uint64, count int, epsilon float32) ([]float32, error) {
	return hipRunRMSNormKernelWithDeviceWeightConfig(ctx, driver, input, hipRMSNormDeviceWeightConfig{
		WeightPointer:  weightPointer,
		WeightBytes:    weightBytes,
		Count:          count,
		Epsilon:        epsilon,
		WeightEncoding: hipRMSNormWeightEncodingF32,
	})
}

func hipRunRMSNormKernelWithDeviceWeightConfig(ctx context.Context, driver nativeHIPDriver, input []float32, cfg hipRMSNormDeviceWeightConfig) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.RMSNormLaunch", "HIP driver is not available", nil)
	}
	if cfg.WeightEncoding == hipRMSNormWeightEncodingNone {
		if cfg.Flags != 0 {
			return nil, core.E("rocm.hip.RMSNormLaunch", "unit RMSNorm weight does not support flags", nil)
		}
		if cfg.WeightPointer != 0 || cfg.WeightBytes != 0 {
			return nil, core.E("rocm.hip.RMSNormLaunch", "unit RMSNorm weight must not provide a weight pointer", nil)
		}
	} else if cfg.WeightPointer == 0 {
		return nil, core.E("rocm.hip.RMSNormLaunch", "RMSNorm weight pointer is required", nil)
	}
	if cfg.Count <= 0 {
		return nil, core.E("rocm.hip.RMSNormLaunch", "count must be positive", nil)
	}
	if len(input) != cfg.Count {
		return nil, core.E("rocm.hip.RMSNormLaunch", "input length must match count", nil)
	}
	inputPayload, err := hipFloat32Payload(input)
	if err != nil {
		return nil, core.E("rocm.hip.RMSNormLaunch", "encode input", err)
	}
	inputBuffer, err := hipUploadByteBuffer(driver, "rocm.hip.RMSNormLaunch", "rms norm input", inputPayload, len(input))
	if err != nil {
		return nil, err
	}
	defer inputBuffer.Close()
	output, err := hipRunRMSNormKernelWithDeviceInputWeightConfig(ctx, driver, inputBuffer, cfg)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	return hipReadFloat32DeviceOutput(output, "rocm.hip.RMSNormLaunch", "rms norm output", cfg.Count)
}

func hipRunRMSNormKernelWithDeviceInputWeightConfig(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.RMSNormLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.RMSNormLaunch", "RMSNorm input device buffer is required", nil)
	}
	if cfg.WeightEncoding == hipRMSNormWeightEncodingNone {
		if cfg.Flags != 0 {
			return nil, core.E("rocm.hip.RMSNormLaunch", "unit RMSNorm weight does not support flags", nil)
		}
		if cfg.WeightPointer != 0 || cfg.WeightBytes != 0 {
			return nil, core.E("rocm.hip.RMSNormLaunch", "unit RMSNorm weight must not provide a weight pointer", nil)
		}
	} else if cfg.WeightPointer == 0 {
		return nil, core.E("rocm.hip.RMSNormLaunch", "RMSNorm weight pointer is required", nil)
	}
	if cfg.Count <= 0 {
		return nil, core.E("rocm.hip.RMSNormLaunch", "count must be positive", nil)
	}
	if input.Count() != cfg.Count || input.SizeBytes() != uint64(cfg.Count*4) {
		return nil, core.E("rocm.hip.RMSNormLaunch", "RMSNorm input device buffer shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormLaunch", "rms norm output", uint64(cfg.Count*4), cfg.Count)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipRMSNormLaunchArgs{
		InputPointer:   input.Pointer(),
		WeightPointer:  cfg.WeightPointer,
		OutputPointer:  output.Pointer(),
		Count:          cfg.Count,
		InputBytes:     input.SizeBytes(),
		WeightBytes:    cfg.WeightBytes,
		OutputBytes:    output.SizeBytes(),
		Epsilon:        cfg.Epsilon,
		WeightEncoding: cfg.WeightEncoding,
		Flags:          cfg.Flags,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipSingleBlockLaunchConfig(hipKernelNameRMSNorm, launchBytes, 256)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunRMSNormResidualAddKernelWithDeviceInputWeightConfig(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig) (*hipDeviceByteBuffer, error) {
	return hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfig(ctx, driver, input, residual, cfg, 1)
}

func hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfig(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, outputScale float32) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "RMSNorm input device buffer is required", nil)
	}
	if residual == nil || residual.Pointer() == 0 {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "residual device buffer is required", nil)
	}
	if cfg.WeightEncoding == hipRMSNormWeightEncodingNone {
		if cfg.Flags != 0 {
			return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "unit RMSNorm weight does not support flags", nil)
		}
		if cfg.WeightPointer != 0 || cfg.WeightBytes != 0 {
			return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "unit RMSNorm weight must not provide a weight pointer", nil)
		}
	} else if cfg.WeightPointer == 0 {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "RMSNorm weight pointer is required", nil)
	}
	if cfg.Count <= 0 {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "count must be positive", nil)
	}
	if math.IsNaN(float64(outputScale)) || math.IsInf(float64(outputScale), 0) {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "output scale must be finite", nil)
	}
	if input.Count() != cfg.Count || residual.Count() != cfg.Count || input.SizeBytes() != uint64(cfg.Count*4) || residual.SizeBytes() != uint64(cfg.Count*4) {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "RMSNorm residual-add device buffer shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormResidualAddLaunch", "rms norm residual-add output", uint64(cfg.Count*4), cfg.Count)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfigOutput(ctx, driver, input, residual, cfg, output, outputScale); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfigOutput(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, output *hipDeviceByteBuffer, outputScale float32) error {
	return hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfigOutputWithWorkspace(ctx, driver, input, residual, cfg, output, outputScale, nil)
}

func hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfigOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, output *hipDeviceByteBuffer, outputScale float32, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.RMSNormResidualAddLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.RMSNormResidualAddLaunch", "RMSNorm input device buffer is required", nil)
	}
	if residual == nil || residual.Pointer() == 0 {
		return core.E("rocm.hip.RMSNormResidualAddLaunch", "residual device buffer is required", nil)
	}
	if cfg.WeightEncoding == hipRMSNormWeightEncodingNone {
		if cfg.Flags != 0 {
			return core.E("rocm.hip.RMSNormResidualAddLaunch", "unit RMSNorm weight does not support flags", nil)
		}
		if cfg.WeightPointer != 0 || cfg.WeightBytes != 0 {
			return core.E("rocm.hip.RMSNormResidualAddLaunch", "unit RMSNorm weight must not provide a weight pointer", nil)
		}
	} else if cfg.WeightPointer == 0 {
		return core.E("rocm.hip.RMSNormResidualAddLaunch", "RMSNorm weight pointer is required", nil)
	}
	if cfg.Count <= 0 {
		return core.E("rocm.hip.RMSNormResidualAddLaunch", "count must be positive", nil)
	}
	if math.IsNaN(float64(outputScale)) || math.IsInf(float64(outputScale), 0) {
		return core.E("rocm.hip.RMSNormResidualAddLaunch", "output scale must be finite", nil)
	}
	if input.Count() != cfg.Count || residual.Count() != cfg.Count || input.SizeBytes() != uint64(cfg.Count*4) || residual.SizeBytes() != uint64(cfg.Count*4) {
		return core.E("rocm.hip.RMSNormResidualAddLaunch", "RMSNorm residual-add device buffer shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != cfg.Count || output.SizeBytes() != uint64(cfg.Count*4) {
		return core.E("rocm.hip.RMSNormResidualAddLaunch", "RMSNorm residual-add output device buffer shape mismatch", nil)
	}
	launchArgs := hipRMSNormResidualAddLaunchArgs{
		InputPointer:    input.Pointer(),
		WeightPointer:   cfg.WeightPointer,
		ResidualPointer: residual.Pointer(),
		OutputPointer:   output.Pointer(),
		Count:           cfg.Count,
		InputBytes:      input.SizeBytes(),
		WeightBytes:     cfg.WeightBytes,
		ResidualBytes:   residual.SizeBytes(),
		OutputBytes:     output.SizeBytes(),
		Epsilon:         cfg.Epsilon,
		WeightEncoding:  cfg.WeightEncoding,
		Flags:           cfg.Flags,
		OutputScale:     outputScale,
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.RMSResidualAddArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config, err := hipSingleBlockLaunchConfig(hipKernelNameRMSNormResidualAdd, launchBytes, hipRMSNormResidualAddBlockSize(cfg.Count))
	if err != nil {
		return err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return err
	}
	return nil
}

func hipRMSNormResidualAddBlockSize(count int) uint32 {
	if count == 2560 {
		return 512
	}
	return 256
}

func hipRunRMSNormResidualAddNormKernelWithDeviceInputWeightConfig(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, residualCfg, normCfg hipRMSNormDeviceWeightConfig) (*hipDeviceByteBuffer, *hipDeviceByteBuffer, error) {
	return hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfig(ctx, driver, input, residual, residualCfg, normCfg, 1)
}

func hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfig(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, residualCfg, normCfg hipRMSNormDeviceWeightConfig, outputScale float32) (*hipDeviceByteBuffer, *hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "RMSNorm input device buffer is required", nil)
	}
	if residual == nil || residual.Pointer() == 0 {
		return nil, nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "residual device buffer is required", nil)
	}
	if err := hipValidateRMSNormDeviceWeightConfig("RMSNormResidualAddNormLaunch", residualCfg); err != nil {
		return nil, nil, err
	}
	if err := hipValidateRMSNormDeviceWeightConfig("RMSNormResidualAddNormLaunch", normCfg); err != nil {
		return nil, nil, err
	}
	if residualCfg.Count <= 0 || residualCfg.Count != normCfg.Count {
		return nil, nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "RMSNorm counts must be positive and equal", nil)
	}
	if math.IsNaN(float64(outputScale)) || math.IsInf(float64(outputScale), 0) {
		return nil, nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "output scale must be finite", nil)
	}
	if input.Count() != residualCfg.Count || residual.Count() != residualCfg.Count ||
		input.SizeBytes() != uint64(residualCfg.Count*4) ||
		residual.SizeBytes() != uint64(residualCfg.Count*4) {
		return nil, nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "RMSNorm residual-add-norm device buffer shape mismatch", nil)
	}
	residualOutput, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormResidualAddNormLaunch", "rms norm residual-add output", uint64(residualCfg.Count*4), residualCfg.Count)
	if err != nil {
		return nil, nil, err
	}
	normOutput, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormResidualAddNormLaunch", "rms norm residual-add norm output", uint64(normCfg.Count*4), normCfg.Count)
	if err != nil {
		_ = residualOutput.Close()
		return nil, nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = normOutput.Close()
			_ = residualOutput.Close()
		}
	}()
	if err := hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfigOutput(ctx, driver, input, residual, residualCfg, normCfg, residualOutput, normOutput, outputScale); err != nil {
		return nil, nil, err
	}
	success = true
	return residualOutput, normOutput, nil
}

func hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfigOutput(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, residualCfg, normCfg hipRMSNormDeviceWeightConfig, residualOutput, normOutput *hipDeviceByteBuffer, outputScale float32) error {
	return hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfigOutputWithWorkspace(ctx, driver, input, residual, residualCfg, normCfg, residualOutput, normOutput, outputScale, nil)
}

func hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfigOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, residualCfg, normCfg hipRMSNormDeviceWeightConfig, residualOutput, normOutput *hipDeviceByteBuffer, outputScale float32, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.RMSNormResidualAddNormLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.RMSNormResidualAddNormLaunch", "RMSNorm input device buffer is required", nil)
	}
	if residual == nil || residual.Pointer() == 0 {
		return core.E("rocm.hip.RMSNormResidualAddNormLaunch", "residual device buffer is required", nil)
	}
	if err := hipValidateRMSNormDeviceWeightConfig("RMSNormResidualAddNormLaunch", residualCfg); err != nil {
		return err
	}
	if err := hipValidateRMSNormDeviceWeightConfig("RMSNormResidualAddNormLaunch", normCfg); err != nil {
		return err
	}
	if residualCfg.Count <= 0 || residualCfg.Count != normCfg.Count {
		return core.E("rocm.hip.RMSNormResidualAddNormLaunch", "RMSNorm counts must be positive and equal", nil)
	}
	if math.IsNaN(float64(outputScale)) || math.IsInf(float64(outputScale), 0) {
		return core.E("rocm.hip.RMSNormResidualAddNormLaunch", "output scale must be finite", nil)
	}
	if input.Count() != residualCfg.Count || residual.Count() != residualCfg.Count ||
		input.SizeBytes() != uint64(residualCfg.Count*4) ||
		residual.SizeBytes() != uint64(residualCfg.Count*4) {
		return core.E("rocm.hip.RMSNormResidualAddNormLaunch", "RMSNorm residual-add-norm device buffer shape mismatch", nil)
	}
	if residualOutput == nil || residualOutput.Pointer() == 0 || residualOutput.Count() != residualCfg.Count || residualOutput.SizeBytes() != uint64(residualCfg.Count*4) {
		return core.E("rocm.hip.RMSNormResidualAddNormLaunch", "residual output device buffer shape mismatch", nil)
	}
	if normOutput == nil || normOutput.Pointer() == 0 || normOutput.Count() != normCfg.Count || normOutput.SizeBytes() != uint64(normCfg.Count*4) {
		return core.E("rocm.hip.RMSNormResidualAddNormLaunch", "norm output device buffer shape mismatch", nil)
	}
	launchArgs := hipRMSNormResidualAddNormLaunchArgs{
		InputPointer:          input.Pointer(),
		WeightPointer:         residualCfg.WeightPointer,
		ResidualPointer:       residual.Pointer(),
		ResidualOutputPointer: residualOutput.Pointer(),
		NormWeightPointer:     normCfg.WeightPointer,
		NormOutputPointer:     normOutput.Pointer(),
		Count:                 residualCfg.Count,
		InputBytes:            input.SizeBytes(),
		WeightBytes:           residualCfg.WeightBytes,
		ResidualBytes:         residual.SizeBytes(),
		ResidualOutputBytes:   residualOutput.SizeBytes(),
		NormWeightBytes:       normCfg.WeightBytes,
		NormOutputBytes:       normOutput.SizeBytes(),
		Epsilon:               residualCfg.Epsilon,
		WeightEncoding:        residualCfg.WeightEncoding,
		Flags:                 residualCfg.Flags,
		NormEpsilon:           normCfg.Epsilon,
		NormWeightEncoding:    normCfg.WeightEncoding,
		NormFlags:             normCfg.Flags,
		OutputScale:           outputScale,
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.RMSResidualAddNormArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config, err := hipSingleBlockLaunchConfig(hipKernelNameRMSNormResAddNorm, launchBytes, hipRMSNormResidualAddNormBlockSize(residualCfg.Count))
	if err != nil {
		return err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return err
	}
	return nil
}

func hipRMSNormResidualAddNormBlockSize(count int) uint32 {
	if count == 2560 {
		return 512
	}
	return 256
}

func hipValidateRMSNormDeviceWeightConfig(operation string, cfg hipRMSNormDeviceWeightConfig) error {
	if cfg.WeightEncoding == hipRMSNormWeightEncodingNone {
		if cfg.Flags != 0 {
			return core.E("rocm.hip."+operation, "unit RMSNorm weight does not support flags", nil)
		}
		if cfg.WeightPointer != 0 || cfg.WeightBytes != 0 {
			return core.E("rocm.hip."+operation, "unit RMSNorm weight must not provide a weight pointer", nil)
		}
		return nil
	}
	if cfg.WeightPointer == 0 {
		return core.E("rocm.hip."+operation, "RMSNorm weight pointer is required", nil)
	}
	return nil
}

func hipRunRMSNormDeviceToDeviceKernel(ctx context.Context, driver nativeHIPDriver, inputPointer nativeDevicePointer, inputBytes uint64, outputPointer nativeDevicePointer, outputBytes uint64, cfg hipRMSNormDeviceWeightConfig) error {
	return hipRunRMSNormDeviceToDeviceKernelWithWorkspace(ctx, driver, inputPointer, inputBytes, outputPointer, outputBytes, cfg, nil)
}

func hipRunRMSNormDeviceToDeviceKernelWithWorkspace(ctx context.Context, driver nativeHIPDriver, inputPointer nativeDevicePointer, inputBytes uint64, outputPointer nativeDevicePointer, outputBytes uint64, cfg hipRMSNormDeviceWeightConfig, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if cfg.WeightEncoding == hipRMSNormWeightEncodingNone {
		if cfg.Flags != 0 {
			return core.E("rocm.hip.RMSNormLaunch", "unit RMSNorm weight does not support flags", nil)
		}
		if cfg.WeightPointer != 0 || cfg.WeightBytes != 0 {
			return core.E("rocm.hip.RMSNormLaunch", "unit RMSNorm weight must not provide a weight pointer", nil)
		}
	} else if cfg.WeightPointer == 0 {
		return core.E("rocm.hip.RMSNormLaunch", "RMSNorm weight pointer is required", nil)
	}
	launchArgs := hipRMSNormLaunchArgs{
		InputPointer:   inputPointer,
		WeightPointer:  cfg.WeightPointer,
		OutputPointer:  outputPointer,
		Count:          cfg.Count,
		InputBytes:     inputBytes,
		WeightBytes:    cfg.WeightBytes,
		OutputBytes:    outputBytes,
		Epsilon:        cfg.Epsilon,
		WeightEncoding: cfg.WeightEncoding,
		Flags:          cfg.Flags,
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.RMSNormArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config, err := hipSingleBlockLaunchConfig(hipKernelNameRMSNorm, launchBytes, 256)
	if err != nil {
		return err
	}
	return hipLaunchKernelContext(ctx, driver, config)
}

func hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, headCount int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "RMSNorm heads input device buffer is required", nil)
	}
	if cfg.Count <= 0 || headCount <= 0 {
		return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "head dim and head count must be positive", nil)
	}
	if input.Count() != cfg.Count*headCount || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "RMSNorm heads input device buffer shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormHeadsLaunch", "rms norm heads output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunRMSNormHeadsKernelWithDeviceInputWeightConfigOutput(ctx, driver, input, cfg, headCount, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunRMSNormHeadsKernelWithDeviceInputWeightConfigOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, headCount int, output *hipDeviceByteBuffer) error {
	return hipRunRMSNormHeadsKernelWithDeviceInputWeightConfigOutputWithWorkspace(ctx, driver, input, cfg, headCount, output, nil)
}

func hipRunRMSNormHeadsKernelWithDeviceInputWeightConfigOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, headCount int, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.RMSNormHeadsLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.RMSNormHeadsLaunch", "RMSNorm heads input device buffer is required", nil)
	}
	if cfg.Count <= 0 || headCount <= 0 {
		return core.E("rocm.hip.RMSNormHeadsLaunch", "head dim and head count must be positive", nil)
	}
	if input.Count() != cfg.Count*headCount || input.SizeBytes() != uint64(input.Count()*4) {
		return core.E("rocm.hip.RMSNormHeadsLaunch", "RMSNorm heads input device buffer shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != input.Count() || output.SizeBytes() != input.SizeBytes() {
		return core.E("rocm.hip.RMSNormHeadsLaunch", "RMSNorm heads output device buffer shape mismatch", nil)
	}
	launchArgs := hipRMSNormHeadsLaunchArgs{
		InputPointer:   input.Pointer(),
		WeightPointer:  cfg.WeightPointer,
		OutputPointer:  output.Pointer(),
		HeadDim:        cfg.Count,
		HeadCount:      headCount,
		InputBytes:     input.SizeBytes(),
		WeightBytes:    cfg.WeightBytes,
		OutputBytes:    output.SizeBytes(),
		Epsilon:        cfg.Epsilon,
		WeightEncoding: cfg.WeightEncoding,
		Flags:          cfg.Flags,
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.RMSNormHeadsArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameRMSNormHeads,
		Args:   launchBytes,
		GridX:  uint32(headCount),
		GridY:  1,
		GridZ:  1,
		BlockX: 256,
		BlockY: 1,
		BlockZ: 1,
	}
	if err := config.Validate(); err != nil {
		return err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return err
	}
	return nil
}

func hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfig(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, headCount int, position int, base float32, frequencyDim int, rotaryCount int) (*hipDeviceByteBuffer, error) {
	return hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigFrequencyScale(ctx, driver, input, cfg, headCount, position, base, frequencyDim, rotaryCount, 1)
}

func hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigFrequencyScale(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, headCount int, position int, base float32, frequencyDim int, rotaryCount int, frequencyScale float32) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.RMSNormRoPEHeadsLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.RMSNormRoPEHeadsLaunch", "RMSNorm RoPE heads input device buffer is required", nil)
	}
	if cfg.Count <= 0 || cfg.Count%2 != 0 || headCount <= 0 {
		return nil, core.E("rocm.hip.RMSNormRoPEHeadsLaunch", "head dim must be positive/even and head count must be positive", nil)
	}
	if input.Count() != cfg.Count*headCount || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E("rocm.hip.RMSNormRoPEHeadsLaunch", "RMSNorm RoPE heads input device buffer shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormRoPEHeadsLaunch", "rms norm rope heads output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutputFrequencyScale(ctx, driver, input, cfg, headCount, position, base, frequencyDim, rotaryCount, frequencyScale, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, headCount int, position int, base float32, frequencyDim int, rotaryCount int, output *hipDeviceByteBuffer) error {
	return hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutputFrequencyScale(ctx, driver, input, cfg, headCount, position, base, frequencyDim, rotaryCount, 1, output)
}

func hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutputFrequencyScale(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, headCount int, position int, base float32, frequencyDim int, rotaryCount int, frequencyScale float32, output *hipDeviceByteBuffer) error {
	return hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutputFrequencyScaleWithWorkspace(ctx, driver, input, cfg, headCount, position, base, frequencyDim, rotaryCount, frequencyScale, output, nil)
}

func hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutputFrequencyScaleWithWorkspace(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, headCount int, position int, base float32, frequencyDim int, rotaryCount int, frequencyScale float32, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.RMSNormRoPEHeadsLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.RMSNormRoPEHeadsLaunch", "RMSNorm RoPE heads input device buffer is required", nil)
	}
	if cfg.Count <= 0 || cfg.Count%2 != 0 || headCount <= 0 {
		return core.E("rocm.hip.RMSNormRoPEHeadsLaunch", "head dim must be positive/even and head count must be positive", nil)
	}
	if input.Count() != cfg.Count*headCount || input.SizeBytes() != uint64(input.Count()*4) {
		return core.E("rocm.hip.RMSNormRoPEHeadsLaunch", "RMSNorm RoPE heads input device buffer shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != input.Count() || output.SizeBytes() != input.SizeBytes() {
		return core.E("rocm.hip.RMSNormRoPEHeadsLaunch", "RMSNorm RoPE heads output device buffer shape mismatch", nil)
	}
	launchArgs := hipRMSNormRoPEHeadsLaunchArgs{
		InputPointer:   input.Pointer(),
		WeightPointer:  cfg.WeightPointer,
		OutputPointer:  output.Pointer(),
		HeadDim:        cfg.Count,
		HeadCount:      headCount,
		InputBytes:     input.SizeBytes(),
		WeightBytes:    cfg.WeightBytes,
		OutputBytes:    output.SizeBytes(),
		Epsilon:        cfg.Epsilon,
		WeightEncoding: cfg.WeightEncoding,
		Flags:          cfg.Flags,
		Position:       position,
		Base:           base,
		FrequencyDim:   frequencyDim,
		RotaryCount:    rotaryCount,
		FrequencyScale: frequencyScale,
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.RMSNormRoPEHeadsArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameRMSNormRoPEHeads,
		Args:   launchBytes,
		GridX:  uint32(headCount),
		GridY:  1,
		GridZ:  1,
		BlockX: 256,
		BlockY: 1,
		BlockZ: 1,
	}
	if err := config.Validate(); err != nil {
		return err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return err
	}
	return nil
}

func hipRunRMSNormRoPEHeadsPairKernelWithDeviceInputWeightConfigOutputFrequencyScaleWithWorkspace(ctx context.Context, driver nativeHIPDriver, queryInput, keyInput *hipDeviceByteBuffer, queryCfg, keyCfg hipRMSNormDeviceWeightConfig, queryHeadCount, keyHeadCount int, position int, base float32, frequencyDim int, rotaryCount int, frequencyScale float32, queryOutput, keyOutput *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.RMSNormRoPEHeadsPairLaunch", "HIP driver is not available", nil)
	}
	if queryInput == nil || queryInput.Pointer() == 0 || keyInput == nil || keyInput.Pointer() == 0 {
		return core.E("rocm.hip.RMSNormRoPEHeadsPairLaunch", "RMSNorm RoPE pair input device buffers are required", nil)
	}
	if queryCfg.Count <= 0 || queryCfg.Count%2 != 0 || queryCfg.Count != keyCfg.Count || queryHeadCount <= 0 || keyHeadCount <= 0 {
		return core.E("rocm.hip.RMSNormRoPEHeadsPairLaunch", "head dim must be positive/even/equal and head counts must be positive", nil)
	}
	if queryInput.Count() != queryCfg.Count*queryHeadCount || queryInput.SizeBytes() != uint64(queryInput.Count()*4) {
		return core.E("rocm.hip.RMSNormRoPEHeadsPairLaunch", "query input device buffer shape mismatch", nil)
	}
	if keyInput.Count() != keyCfg.Count*keyHeadCount || keyInput.SizeBytes() != uint64(keyInput.Count()*4) {
		return core.E("rocm.hip.RMSNormRoPEHeadsPairLaunch", "key input device buffer shape mismatch", nil)
	}
	if queryOutput == nil || queryOutput.Pointer() == 0 || queryOutput.Count() != queryInput.Count() || queryOutput.SizeBytes() != queryInput.SizeBytes() {
		return core.E("rocm.hip.RMSNormRoPEHeadsPairLaunch", "query output device buffer shape mismatch", nil)
	}
	if keyOutput == nil || keyOutput.Pointer() == 0 || keyOutput.Count() != keyInput.Count() || keyOutput.SizeBytes() != keyInput.SizeBytes() {
		return core.E("rocm.hip.RMSNormRoPEHeadsPairLaunch", "key output device buffer shape mismatch", nil)
	}
	launchArgs := hipRMSNormRoPEHeadsPairLaunchArgs{
		QueryInputPointer:   queryInput.Pointer(),
		QueryWeightPointer:  queryCfg.WeightPointer,
		QueryOutputPointer:  queryOutput.Pointer(),
		KeyInputPointer:     keyInput.Pointer(),
		KeyWeightPointer:    keyCfg.WeightPointer,
		KeyOutputPointer:    keyOutput.Pointer(),
		HeadDim:             queryCfg.Count,
		QueryHeadCount:      queryHeadCount,
		KeyHeadCount:        keyHeadCount,
		QueryInputBytes:     queryInput.SizeBytes(),
		QueryWeightBytes:    queryCfg.WeightBytes,
		QueryOutputBytes:    queryOutput.SizeBytes(),
		KeyInputBytes:       keyInput.SizeBytes(),
		KeyWeightBytes:      keyCfg.WeightBytes,
		KeyOutputBytes:      keyOutput.SizeBytes(),
		QueryEpsilon:        queryCfg.Epsilon,
		QueryWeightEncoding: queryCfg.WeightEncoding,
		QueryFlags:          queryCfg.Flags,
		KeyEpsilon:          keyCfg.Epsilon,
		KeyWeightEncoding:   keyCfg.WeightEncoding,
		KeyFlags:            keyCfg.Flags,
		Position:            position,
		Base:                base,
		FrequencyDim:        frequencyDim,
		RotaryCount:         rotaryCount,
		FrequencyScale:      frequencyScale,
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.RMSNormRoPEHeadsPairArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameRMSNormRoPEHeadsPair,
		Args:   launchBytes,
		GridX:  uint32(queryHeadCount + keyHeadCount),
		GridY:  1,
		GridZ:  1,
		BlockX: 256,
		BlockY: 1,
		BlockZ: 1,
	}
	if err := config.Validate(); err != nil {
		return err
	}
	return hipLaunchKernelContext(ctx, driver, config)
}

func hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfig(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, headCount int, batch int, startPosition int, base float32, frequencyDim int, rotaryCount int) (*hipDeviceByteBuffer, error) {
	return hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfigFrequencyScale(ctx, driver, input, cfg, headCount, batch, startPosition, base, frequencyDim, rotaryCount, 1)
}

func hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfigFrequencyScale(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, headCount int, batch int, startPosition int, base float32, frequencyDim int, rotaryCount int, frequencyScale float32) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.RMSNormRoPEHeadsBatchLaunch", "RMSNorm RoPE heads batch input device buffer is required", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormRoPEHeadsBatchLaunch", "rms norm rope heads batch output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfigFrequencyScaleOutput(ctx, driver, input, cfg, headCount, batch, startPosition, base, frequencyDim, rotaryCount, frequencyScale, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfigFrequencyScaleOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, headCount int, batch int, startPosition int, base float32, frequencyDim int, rotaryCount int, frequencyScale float32, output *hipDeviceByteBuffer) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.RMSNormRoPEHeadsBatchLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.RMSNormRoPEHeadsBatchLaunch", "RMSNorm RoPE heads batch input device buffer is required", nil)
	}
	if cfg.Count <= 0 || cfg.Count%2 != 0 || headCount <= 0 || batch <= 0 {
		return core.E("rocm.hip.RMSNormRoPEHeadsBatchLaunch", "head dim must be positive/even and head count/batch must be positive", nil)
	}
	if input.Count() != cfg.Count*headCount*batch || input.SizeBytes() != uint64(input.Count()*4) {
		return core.E("rocm.hip.RMSNormRoPEHeadsBatchLaunch", "RMSNorm RoPE heads batch input device buffer shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != input.Count() || output.SizeBytes() != input.SizeBytes() {
		return core.E("rocm.hip.RMSNormRoPEHeadsBatchLaunch", "RMSNorm RoPE heads batch output buffer shape mismatch", nil)
	}
	launchBytes, err := (hipRMSNormRoPEHeadsBatchLaunchArgs{
		InputPointer:   input.Pointer(),
		WeightPointer:  cfg.WeightPointer,
		OutputPointer:  output.Pointer(),
		HeadDim:        cfg.Count,
		HeadCount:      headCount,
		Batch:          batch,
		InputBytes:     input.SizeBytes(),
		WeightBytes:    cfg.WeightBytes,
		OutputBytes:    output.SizeBytes(),
		Epsilon:        cfg.Epsilon,
		WeightEncoding: cfg.WeightEncoding,
		Flags:          cfg.Flags,
		StartPosition:  startPosition,
		Base:           base,
		FrequencyDim:   frequencyDim,
		RotaryCount:    rotaryCount,
		FrequencyScale: frequencyScale,
	}).Binary()
	if err != nil {
		return err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameRMSNormRoPEHeadsBatch,
		Args:   launchBytes,
		GridX:  uint32(headCount),
		GridY:  uint32(batch),
		GridZ:  1,
		BlockX: 256,
		BlockY: 1,
		BlockZ: 1,
	}
	if err := config.Validate(); err != nil {
		return err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return err
	}
	return nil
}

func hipRunRMSNormRoPEHeadsPairLaneBatchKernelWithDeviceInputWeightConfigFrequencyScaleOutput(ctx context.Context, driver nativeHIPDriver, queryInput, keyInput *hipDeviceByteBuffer, queryCfg, keyCfg hipRMSNormDeviceWeightConfig, queryHeadCount, keyHeadCount int, positions *hipDeviceTokenBuffer, base float32, frequencyDim int, rotaryCount int, frequencyScale float32, queryOutput, keyOutput *hipDeviceByteBuffer) error {
	const operation = "rocm.hip.RMSNormRoPEHeadsPairLaneBatchLaunch"
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E(operation, "HIP driver is not available", nil)
	}
	if queryInput == nil || queryInput.Pointer() == 0 || keyInput == nil || keyInput.Pointer() == 0 {
		return core.E(operation, "query and key input device buffers are required", nil)
	}
	if queryCfg.Count <= 0 || queryCfg.Count%2 != 0 || queryCfg.Count != keyCfg.Count || queryHeadCount <= 0 || keyHeadCount <= 0 {
		return core.E(operation, "head dim must be positive/even/equal and head counts must be positive", nil)
	}
	if positions == nil || positions.Pointer() == 0 || positions.Count() <= 0 || positions.SizeBytes() != uint64(positions.Count()*4) {
		return core.E(operation, "positions device buffer is required and must contain int32 rows", nil)
	}
	batch := positions.Count()
	if queryInput.Count() != queryCfg.Count*queryHeadCount*batch || queryInput.SizeBytes() != uint64(queryInput.Count()*4) {
		return core.E(operation, "query input device buffer shape mismatch", nil)
	}
	if keyInput.Count() != keyCfg.Count*keyHeadCount*batch || keyInput.SizeBytes() != uint64(keyInput.Count()*4) {
		return core.E(operation, "key input device buffer shape mismatch", nil)
	}
	if queryOutput == nil || queryOutput.Pointer() == 0 || queryOutput.Count() != queryInput.Count() || queryOutput.SizeBytes() != queryInput.SizeBytes() {
		return core.E(operation, "query output device buffer shape mismatch", nil)
	}
	if keyOutput == nil || keyOutput.Pointer() == 0 || keyOutput.Count() != keyInput.Count() || keyOutput.SizeBytes() != keyInput.SizeBytes() {
		return core.E(operation, "key output device buffer shape mismatch", nil)
	}
	launchBytes, err := (hipRMSNormRoPEHeadsPairLaneBatchLaunchArgs{
		QueryInputPointer:   queryInput.Pointer(),
		QueryWeightPointer:  queryCfg.WeightPointer,
		QueryOutputPointer:  queryOutput.Pointer(),
		KeyInputPointer:     keyInput.Pointer(),
		KeyWeightPointer:    keyCfg.WeightPointer,
		KeyOutputPointer:    keyOutput.Pointer(),
		PositionsPointer:    positions.Pointer(),
		HeadDim:             queryCfg.Count,
		QueryHeadCount:      queryHeadCount,
		KeyHeadCount:        keyHeadCount,
		Batch:               batch,
		QueryInputBytes:     queryInput.SizeBytes(),
		QueryWeightBytes:    queryCfg.WeightBytes,
		QueryOutputBytes:    queryOutput.SizeBytes(),
		KeyInputBytes:       keyInput.SizeBytes(),
		KeyWeightBytes:      keyCfg.WeightBytes,
		KeyOutputBytes:      keyOutput.SizeBytes(),
		QueryEpsilon:        queryCfg.Epsilon,
		QueryWeightEncoding: queryCfg.WeightEncoding,
		QueryFlags:          queryCfg.Flags,
		KeyEpsilon:          keyCfg.Epsilon,
		KeyWeightEncoding:   keyCfg.WeightEncoding,
		KeyFlags:            keyCfg.Flags,
		PositionsBytes:      positions.SizeBytes(),
		Base:                base,
		FrequencyDim:        frequencyDim,
		RotaryCount:         rotaryCount,
		FrequencyScale:      frequencyScale,
	}).Binary()
	if err != nil {
		return err
	}
	gridX := queryHeadCount
	if keyHeadCount > gridX {
		gridX = keyHeadCount
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameRMSNormRoPEHeadsPairLaneBatch,
		Args:   launchBytes,
		GridX:  uint32(gridX),
		GridY:  uint32(batch),
		GridZ:  1,
		BlockX: 256,
		BlockY: 1,
		BlockZ: 1,
	}
	if err := config.Validate(); err != nil {
		return err
	}
	return hipLaunchKernelContext(ctx, driver, config)
}

func hipRunRoPEKernel(ctx context.Context, driver nativeHIPDriver, req hipRoPERequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameRoPE, launchBytes, buffers.Count/2)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func hipRunRoPEDeviceKernel(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, position int, base float32, frequencyDim int) (*hipDeviceByteBuffer, error) {
	return hipRunRoPEDeviceKernelWithRotaryCount(ctx, driver, input, position, base, frequencyDim, 0)
}

func hipRunRoPEDeviceKernelWithRotaryCount(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, position int, base float32, frequencyDim int, rotaryCount int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.RoPELaunch", "rope device input is required", nil)
	}
	if input.Count() <= 0 || input.Count()%2 != 0 || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E("rocm.hip.RoPELaunch", "rope device input shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RoPELaunch", "rope output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunRoPEDeviceToDeviceKernelWithRotaryCount(ctx, driver, input.Pointer(), input.SizeBytes(), output.Pointer(), output.SizeBytes(), input.Count(), position, base, frequencyDim, rotaryCount); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunRoPEDeviceToDeviceKernel(ctx context.Context, driver nativeHIPDriver, inputPointer nativeDevicePointer, inputBytes uint64, outputPointer nativeDevicePointer, outputBytes uint64, count int, position int, base float32, frequencyDim int) error {
	return hipRunRoPEDeviceToDeviceKernelWithRotaryCount(ctx, driver, inputPointer, inputBytes, outputPointer, outputBytes, count, position, base, frequencyDim, 0)
}

func hipRunRoPEDeviceToDeviceKernelWithRotaryCount(ctx context.Context, driver nativeHIPDriver, inputPointer nativeDevicePointer, inputBytes uint64, outputPointer nativeDevicePointer, outputBytes uint64, count int, position int, base float32, frequencyDim int, rotaryCount int) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	launchBytes, err := (hipRoPELaunchArgs{
		InputPointer:  inputPointer,
		OutputPointer: outputPointer,
		Count:         count,
		InputBytes:    inputBytes,
		OutputBytes:   outputBytes,
		Position:      position,
		Base:          base,
		FrequencyDim:  frequencyDim,
		RotaryCount:   rotaryCount,
	}).Binary()
	if err != nil {
		return err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameRoPE, launchBytes, count/2)
	if err != nil {
		return err
	}
	return hipLaunchKernelContext(ctx, driver, config)
}

func hipRunRoPEHeadsDeviceKernelWithRotaryCount(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, headDim, headCount int, position int, base float32, frequencyDim int, rotaryCount int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.RoPEHeadsLaunch", "rope heads device input is required", nil)
	}
	if headDim <= 0 || headDim%2 != 0 || headCount <= 0 {
		return nil, core.E("rocm.hip.RoPEHeadsLaunch", "head dim must be positive/even and head count must be positive", nil)
	}
	if input.Count() != headDim*headCount || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E("rocm.hip.RoPEHeadsLaunch", "rope heads device input shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RoPEHeadsLaunch", "rope heads output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipRoPEHeadsLaunchArgs{
		InputPointer:  input.Pointer(),
		OutputPointer: output.Pointer(),
		HeadDim:       headDim,
		HeadCount:     headCount,
		InputBytes:    input.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
		Position:      position,
		Base:          base,
		FrequencyDim:  frequencyDim,
		RotaryCount:   rotaryCount,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameRoPEHeads, launchBytes, headCount*(headDim/2))
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunAttentionKernel(ctx context.Context, driver nativeHIPDriver, req hipAttentionRequest) (hipAttentionResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipAttentionResult{}, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return hipAttentionResult{}, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return hipAttentionResult{}, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return hipAttentionResult{}, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameAttention, launchBytes, buffers.TokenCount)
	if err != nil {
		return hipAttentionResult{}, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return hipAttentionResult{}, err
	}
	return buffers.ReadOutput()
}

func hipRunAttentionOutputKernel(ctx context.Context, driver nativeHIPDriver, req hipAttentionRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameAttention, launchBytes, buffers.TokenCount)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutputOnly()
}

func hipRunAttentionOutputToDeviceKernel(ctx context.Context, driver nativeHIPDriver, req hipAttentionRequest, output *hipDeviceByteBuffer, outputElementOffset int) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	queryPayload, err := hipFloat32Payload(req.Query)
	if err != nil {
		return core.E("rocm.hip.AttentionLaunch", "encode query", err)
	}
	query, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionLaunch", "attention query", queryPayload, len(req.Query))
	if err != nil {
		return err
	}
	defer query.Close()
	return hipRunAttentionOutputFromDeviceQueryToDeviceKernel(ctx, driver, req, query, 0, output, outputElementOffset)
}

func hipRunAttentionOutputFromDeviceQueryToDeviceKernel(ctx context.Context, driver nativeHIPDriver, req hipAttentionRequest, query *hipDeviceByteBuffer, queryElementOffset int, output *hipDeviceByteBuffer, outputElementOffset int) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	dim, tokenCount, err := req.shape()
	if err != nil {
		return err
	}
	if query == nil || query.Pointer() == 0 {
		return core.E("rocm.hip.AttentionLaunch", "attention query device buffer is required", nil)
	}
	if queryElementOffset < 0 || query.Count() < queryElementOffset+dim || query.SizeBytes() < uint64(queryElementOffset+dim)*4 {
		return core.E("rocm.hip.AttentionLaunch", "attention query device buffer shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 {
		return core.E("rocm.hip.AttentionLaunch", "attention destination output buffer is required", nil)
	}
	if outputElementOffset < 0 || output.Count() < outputElementOffset+dim || output.SizeBytes() < uint64(outputElementOffset+dim)*4 {
		return core.E("rocm.hip.AttentionLaunch", "attention destination output buffer shape mismatch", nil)
	}
	weights, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionLaunch", "attention weights", uint64(tokenCount*4), tokenCount)
	if err != nil {
		return err
	}
	defer weights.Close()

	launch := hipAttentionLaunchArgs{
		QueryPointer:  nativeDevicePointer(uintptr(query.Pointer()) + uintptr(queryElementOffset*4)),
		OutputPointer: nativeDevicePointer(uintptr(output.Pointer()) + uintptr(outputElementOffset*4)),
		WeightPointer: weights.Pointer(),
		Dim:           dim,
		TokenCount:    tokenCount,
		QueryBytes:    uint64(dim * 4),
		OutputBytes:   uint64(dim * 4),
		WeightBytes:   weights.SizeBytes(),
		Scale:         req.Scale,
	}
	if req.DeviceKV == nil {
		keyPayload, err := hipFloat32Payload(req.Keys)
		if err != nil {
			return core.E("rocm.hip.AttentionLaunch", "encode keys", err)
		}
		keys, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionLaunch", "attention keys", keyPayload, len(req.Keys))
		if err != nil {
			return err
		}
		defer keys.Close()
		valuePayload, err := hipFloat32Payload(req.Values)
		if err != nil {
			return core.E("rocm.hip.AttentionLaunch", "encode values", err)
		}
		values, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionLaunch", "attention values", valuePayload, len(req.Values))
		if err != nil {
			return err
		}
		defer values.Close()
		launch.KVSource = hipAttentionKVSourceContiguous
		launch.KeyPointer = keys.Pointer()
		launch.ValuePointer = values.Pointer()
		launch.KeyBytes = keys.SizeBytes()
		launch.ValueBytes = values.SizeBytes()
	} else {
		launch.KVSource = hipAttentionKVSourceDevice
		launch.DescriptorPointer = req.DescriptorTable.Pointer()
		launch.DescriptorBytes = req.DescriptorTable.SizeBytes()
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameAttention, launchBytes, tokenCount)
	if err != nil {
		return err
	}
	return hipLaunchKernelContext(ctx, driver, config)
}

func hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernel(ctx context.Context, driver nativeHIPDriver, req hipAttentionRequest, query *hipDeviceByteBuffer, headCount int, output *hipDeviceByteBuffer) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	dim, tokenCount, err := req.shape()
	if err != nil {
		return err
	}
	if headCount <= 0 {
		return core.E("rocm.hip.AttentionHeadsLaunch", "head count must be positive", nil)
	}
	if query == nil || query.Pointer() == 0 || query.Count() != headCount*dim || query.SizeBytes() != uint64(headCount*dim*4) {
		return core.E("rocm.hip.AttentionHeadsLaunch", "attention query device buffer shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != headCount*dim || output.SizeBytes() != uint64(headCount*dim*4) {
		return core.E("rocm.hip.AttentionHeadsLaunch", "attention output device buffer shape mismatch", nil)
	}
	useSharedWeights := tokenCount <= hipAttentionHeadsSharedMaxTokens
	var weights *hipDeviceByteBuffer
	if !useSharedWeights {
		weights, err = hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsLaunch", "attention head weights", uint64(headCount*tokenCount*4), headCount*tokenCount)
		if err != nil {
			return err
		}
		defer weights.Close()
	}
	launch := hipAttentionHeadsLaunchArgs{
		QueryPointer:  query.Pointer(),
		OutputPointer: output.Pointer(),
		Dim:           dim,
		TokenCount:    tokenCount,
		HeadCount:     headCount,
		KeyHeads:      req.keyHeadsOrDefault(),
		QueryBytes:    query.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
		Scale:         req.Scale,
		WindowSize:    req.WindowSize,
	}
	var sharedMemBytes uint32
	if useSharedWeights {
		sharedMemBytes, err = hipAttentionHeadsSharedMemBytes(tokenCount, req.DeviceKV != nil)
		if err != nil {
			return err
		}
		launch.SharedMemBytes = uint64(sharedMemBytes)
	} else {
		launch.WeightPointer = weights.Pointer()
		launch.WeightBytes = weights.SizeBytes()
	}
	if req.DeviceKV == nil {
		launch.KVSource = hipAttentionKVSourceContiguous
		keyPayload, err := hipFloat32Payload(req.Keys)
		if err != nil {
			return core.E("rocm.hip.AttentionHeadsLaunch", "encode keys", err)
		}
		keys, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionHeadsLaunch", "attention keys", keyPayload, len(req.Keys))
		if err != nil {
			return err
		}
		defer keys.Close()
		valuePayload, err := hipFloat32Payload(req.Values)
		if err != nil {
			return core.E("rocm.hip.AttentionHeadsLaunch", "encode values", err)
		}
		values, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionHeadsLaunch", "attention values", valuePayload, len(req.Values))
		if err != nil {
			return err
		}
		defer values.Close()
		launch.KeyPointer = keys.Pointer()
		launch.ValuePointer = values.Pointer()
		launch.KeyBytes = keys.SizeBytes()
		launch.ValueBytes = values.SizeBytes()
	} else {
		launch.KVSource = hipAttentionKVSourceDevice
		launch.DescriptorPointer = req.DescriptorTable.Pointer()
		launch.DescriptorBytes = req.DescriptorTable.SizeBytes()
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return err
	}
	config := hipKernelLaunchConfig{
		Name:           hipKernelNameAttentionHeads,
		Args:           launchBytes,
		GridX:          uint32(headCount),
		GridY:          1,
		GridZ:          1,
		BlockX:         hipAttentionHeadsBlockSize(tokenCount),
		BlockY:         1,
		BlockZ:         1,
		SharedMemBytes: sharedMemBytes,
	}
	if err := config.Validate(); err != nil {
		return err
	}
	return hipLaunchKernelContext(ctx, driver, config)
}

type hipAttentionHeadsBatchCausalDeviceRequest struct {
	Key              *hipDeviceByteBuffer
	Value            *hipDeviceByteBuffer
	DeviceKV         *rocmDeviceKVCache
	DescriptorTable  *rocmDeviceKVDescriptorTable
	VisibleTokenCaps *hipDeviceTokenBuffer
	Dim              int
	TokenCount       int
	HeadCount        int
	KeyHeads         int
	QueryCount       int
	QueryStartToken  int
	WindowSize       int
	Scale            float32
}

type hipAttentionHeadsLaneBatchLane struct {
	DeviceKV        *rocmDeviceKVCache
	DescriptorTable *rocmDeviceKVDescriptorTable
	WindowSize      int
}

type hipAttentionHeadsLaneBatchDeviceRequest struct {
	Lanes     []hipAttentionHeadsLaneBatchLane
	Dim       int
	HeadCount int
	KeyHeads  int
	Scale     float32
}

const hipAttentionHeadsBatchWorkspaceMaxWeights = 64 * 1024

func hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernel(ctx context.Context, driver nativeHIPDriver, req hipAttentionHeadsBatchCausalDeviceRequest, query *hipDeviceByteBuffer, output *hipDeviceByteBuffer) error {
	return hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernelWorkspace(ctx, driver, req, query, output, nil)
}

func hipRunAttentionHeadsLaneBatchOutputFromDeviceQueryToDeviceKernel(ctx context.Context, driver nativeHIPDriver, req hipAttentionHeadsLaneBatchDeviceRequest, query *hipDeviceByteBuffer, output *hipDeviceByteBuffer) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	const operation = "AttentionHeadsLaneBatchLaunch"
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip."+operation, "HIP driver is not available", nil)
	}
	if req.Dim <= 0 || req.HeadCount <= 0 || len(req.Lanes) == 0 {
		return core.E("rocm.hip."+operation, "attention lane dimensions must be positive", nil)
	}
	keyHeads := firstPositiveInt(req.KeyHeads, 1)
	if keyHeads <= 0 || keyHeads > req.HeadCount || req.HeadCount%keyHeads != 0 {
		return core.E("rocm.hip."+operation, "key head count must divide query head count", nil)
	}
	if req.Scale < 0 || math.IsNaN(float64(req.Scale)) || math.IsInf(float64(req.Scale), 0) {
		return core.E("rocm.hip."+operation, "scale must be non-negative and finite", nil)
	}
	queryCount := len(req.Lanes) * req.HeadCount * req.Dim
	if query == nil || query.Pointer() == 0 || query.Count() != queryCount || query.SizeBytes() != uint64(queryCount*4) {
		return core.E("rocm.hip."+operation, "attention query device buffer shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != queryCount || output.SizeBytes() != uint64(queryCount*4) {
		return core.E("rocm.hip."+operation, "attention output device buffer shape mismatch", nil)
	}
	lanePayload := make([]byte, len(req.Lanes)*hipAttentionHeadsLaneDescriptorBytes)
	maxTokenCount := 0
	for index, lane := range req.Lanes {
		if lane.WindowSize < 0 {
			return core.E("rocm.hip."+operation, "lane window size must be non-negative", nil)
		}
		if lane.DeviceKV == nil || lane.DescriptorTable == nil {
			return core.E("rocm.hip."+operation, "lane device KV and descriptor table are required", nil)
		}
		if err := lane.DescriptorTable.CompatibleWith(lane.DeviceKV); err != nil {
			return core.E("rocm.hip."+operation, "lane descriptor table does not match device KV cache", err)
		}
		tokenCount := lane.DeviceKV.TokenCount()
		if tokenCount <= 0 {
			return core.E("rocm.hip."+operation, "lane device KV cache has no tokens", nil)
		}
		keyWidth, valueWidth, ok := lane.DeviceKV.LastVectorWidths()
		if !ok || keyWidth != req.Dim*keyHeads || valueWidth != req.Dim*keyHeads {
			return core.E("rocm.hip."+operation, "lane device KV widths must match attention dimension", nil)
		}
		if tokenCount > maxTokenCount {
			maxTokenCount = tokenCount
		}
		offset := index * hipAttentionHeadsLaneDescriptorBytes
		binary.LittleEndian.PutUint64(lanePayload[offset:], uint64(lane.DescriptorTable.Pointer()))
		binary.LittleEndian.PutUint64(lanePayload[offset+8:], lane.DescriptorTable.SizeBytes())
		binary.LittleEndian.PutUint32(lanePayload[offset+16:], uint32(tokenCount))
		binary.LittleEndian.PutUint32(lanePayload[offset+20:], uint32(lane.WindowSize))
	}
	laneBuffer, err := hipUploadByteBuffer(driver, "rocm.hip."+operation, "attention lane descriptors", lanePayload, len(lanePayload))
	if err != nil {
		return err
	}
	defer laneBuffer.Close()

	launch := hipAttentionHeadsLaneBatchLaunchArgs{
		QueryPointer:  query.Pointer(),
		OutputPointer: output.Pointer(),
		LanePointer:   laneBuffer.Pointer(),
		Dim:           req.Dim,
		LaneCount:     len(req.Lanes),
		HeadCount:     req.HeadCount,
		KeyHeads:      keyHeads,
		MaxTokenCount: maxTokenCount,
		QueryBytes:    query.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
		LaneBytes:     laneBuffer.SizeBytes(),
		Scale:         req.Scale,
	}
	var sharedMemBytes uint32
	var weights *hipDeviceByteBuffer
	if maxTokenCount <= hipAttentionHeadsSharedMaxTokens {
		sharedMemBytes, err = hipAttentionHeadsSharedMemBytes(maxTokenCount, true)
		if err != nil {
			return err
		}
		launch.SharedMemBytes = uint64(sharedMemBytes)
	} else {
		weightCount64 := uint64(len(req.Lanes)) * uint64(req.HeadCount) * uint64(maxTokenCount)
		if weightCount64 > uint64(int(^uint(0)>>1))/4 {
			return core.E("rocm.hip."+operation, "attention lane weight buffer is too large", nil)
		}
		weights, err = hipAllocateByteBuffer(driver, "rocm.hip."+operation, "attention lane weights", weightCount64*4, int(weightCount64))
		if err != nil {
			return err
		}
		defer weights.Close()
		launch.WeightPointer = weights.Pointer()
		launch.WeightBytes = weights.SizeBytes()
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return err
	}
	config := hipKernelLaunchConfig{
		Name:           hipKernelNameAttentionHeadsLaneBatch,
		Args:           launchBytes,
		GridX:          uint32(req.HeadCount),
		GridY:          uint32(len(req.Lanes)),
		GridZ:          1,
		BlockX:         hipAttentionHeadsBlockSize(maxTokenCount),
		BlockY:         1,
		BlockZ:         1,
		SharedMemBytes: sharedMemBytes,
	}
	if err := config.Validate(); err != nil {
		return err
	}
	return hipLaunchKernelContext(ctx, driver, config)
}

func hipRunAttentionHeadsBatchCausalQueryRMSRoPEOutputFromDeviceQueryToDeviceKernel(ctx context.Context, driver nativeHIPDriver, req hipAttentionHeadsBatchCausalDeviceRequest, query *hipDeviceByteBuffer, queryNormCfg hipRMSNormDeviceWeightConfig, ropeStartPosition int, ropeBase float32, ropeFrequencyDim, ropeRotaryCount int, ropeFrequencyScale float32, output *hipDeviceByteBuffer) error {
	return hipRunAttentionHeadsBatchCausalQueryRMSRoPEOutputFromDeviceQueryToDeviceKernelWorkspace(ctx, driver, req, query, queryNormCfg, ropeStartPosition, ropeBase, ropeFrequencyDim, ropeRotaryCount, ropeFrequencyScale, output, nil)
}

func hipRunAttentionHeadsBatchCausalQueryRMSRoPEOutputFromDeviceQueryToDeviceKernelWorkspace(ctx context.Context, driver nativeHIPDriver, req hipAttentionHeadsBatchCausalDeviceRequest, query *hipDeviceByteBuffer, queryNormCfg hipRMSNormDeviceWeightConfig, ropeStartPosition int, ropeBase float32, ropeFrequencyDim, ropeRotaryCount int, ropeFrequencyScale float32, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	const operation = "AttentionHeadsBatchCausalQueryRMSRoPELaunch"
	if req.Dim <= 0 || req.Dim%2 != 0 || req.TokenCount <= 0 || req.HeadCount <= 0 || req.QueryCount <= 0 {
		return core.E("rocm.hip."+operation, "attention batch dimensions must be positive and dimension must be even", nil)
	}
	keyHeads := firstPositiveInt(req.KeyHeads, 1)
	if keyHeads <= 0 || keyHeads > req.HeadCount || req.HeadCount%keyHeads != 0 {
		return core.E("rocm.hip."+operation, "key head count must divide query head count", nil)
	}
	if req.QueryStartToken < 0 || uint64(req.QueryStartToken)+uint64(req.QueryCount) > uint64(req.TokenCount) {
		return core.E("rocm.hip."+operation, "causal query window exceeds token count", nil)
	}
	if req.WindowSize < 0 {
		return core.E("rocm.hip."+operation, "window size must be non-negative", nil)
	}
	if req.Scale < 0 || math.IsNaN(float64(req.Scale)) || math.IsInf(float64(req.Scale), 0) {
		return core.E("rocm.hip."+operation, "scale must be non-negative and finite", nil)
	}
	if queryNormCfg.Count != req.Dim {
		return core.E("rocm.hip."+operation, "query RMSNorm weight count must match dimension", nil)
	}
	if err := hipValidateRMSNormDeviceWeightConfig(operation, queryNormCfg); err != nil {
		return err
	}
	if queryNormCfg.Epsilon < 0 || math.IsNaN(float64(queryNormCfg.Epsilon)) || math.IsInf(float64(queryNormCfg.Epsilon), 0) {
		return core.E("rocm.hip."+operation, "query epsilon must be non-negative and finite", nil)
	}
	if ropeStartPosition < 0 || ropeBase <= 0 || math.IsNaN(float64(ropeBase)) || math.IsInf(float64(ropeBase), 0) {
		return core.E("rocm.hip."+operation, "RoPE position and base are invalid", nil)
	}
	if ropeFrequencyDim < 0 || (ropeFrequencyDim > 0 && ropeFrequencyDim < req.Dim) {
		return core.E("rocm.hip."+operation, "RoPE frequency dimension must be zero or at least dimension", nil)
	}
	if ropeRotaryCount < 0 || ropeRotaryCount > req.Dim || ropeRotaryCount%2 != 0 {
		return core.E("rocm.hip."+operation, "RoPE rotary count must be zero or an even count no larger than dimension", nil)
	}
	if ropeFrequencyScale == 0 {
		ropeFrequencyScale = 1
	}
	if ropeFrequencyScale <= 0 || math.IsNaN(float64(ropeFrequencyScale)) || math.IsInf(float64(ropeFrequencyScale), 0) {
		return core.E("rocm.hip."+operation, "RoPE frequency scale must be positive and finite", nil)
	}
	queryCount := req.QueryCount * req.HeadCount * req.Dim
	if query == nil || query.Pointer() == 0 || query.Count() != queryCount || query.SizeBytes() != uint64(queryCount*4) {
		return core.E("rocm.hip."+operation, "attention query device buffer shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != queryCount || output.SizeBytes() != uint64(queryCount*4) {
		return core.E("rocm.hip."+operation, "attention output device buffer shape mismatch", nil)
	}
	launch := hipAttentionHeadsBatchCausalQueryRMSRoPELaunchArgs{
		QueryInputPointer:   query.Pointer(),
		QueryWeightPointer:  queryNormCfg.WeightPointer,
		OutputPointer:       output.Pointer(),
		Dim:                 req.Dim,
		TokenCount:          req.TokenCount,
		HeadCount:           req.HeadCount,
		KeyHeads:            keyHeads,
		QueryCount:          req.QueryCount,
		QueryStartToken:     req.QueryStartToken,
		WindowSize:          req.WindowSize,
		QueryInputBytes:     query.SizeBytes(),
		QueryWeightBytes:    queryNormCfg.WeightBytes,
		OutputBytes:         output.SizeBytes(),
		Scale:               req.Scale,
		QueryEpsilon:        queryNormCfg.Epsilon,
		QueryWeightEncoding: queryNormCfg.WeightEncoding,
		QueryFlags:          queryNormCfg.Flags,
		RoPEStartPosition:   ropeStartPosition,
		RoPEBase:            ropeBase,
		RoPEFrequencyDim:    ropeFrequencyDim,
		RoPERotaryCount:     ropeRotaryCount,
		RoPEFrequencyScale:  ropeFrequencyScale,
	}
	if req.DeviceKV == nil {
		if req.DescriptorTable != nil {
			return core.E("rocm.hip."+operation, "descriptor table requires device KV cache", nil)
		}
		if req.Key == nil || req.Key.Pointer() == 0 || req.Value == nil || req.Value.Pointer() == 0 {
			return core.E("rocm.hip."+operation, "attention key and value device buffers are required", nil)
		}
		kvCount := req.TokenCount * keyHeads * req.Dim
		if req.Key.Count() != kvCount || req.Value.Count() != kvCount ||
			req.Key.SizeBytes() != uint64(kvCount*4) || req.Value.SizeBytes() != uint64(kvCount*4) {
			return core.E("rocm.hip."+operation, "attention key/value device buffer shape mismatch", nil)
		}
		launch.KVSource = hipAttentionKVSourceContiguous
		launch.KeyPointer = req.Key.Pointer()
		launch.ValuePointer = req.Value.Pointer()
		launch.KeyBytes = req.Key.SizeBytes()
		launch.ValueBytes = req.Value.SizeBytes()
	} else {
		if req.Key != nil || req.Value != nil {
			return core.E("rocm.hip."+operation, "device KV attention must not set contiguous KV buffers", nil)
		}
		if req.DescriptorTable == nil {
			return core.E("rocm.hip."+operation, "device KV attention requires descriptor table", nil)
		}
		if err := req.DescriptorTable.CompatibleWith(req.DeviceKV); err != nil {
			return core.E("rocm.hip."+operation, "descriptor table does not match device KV cache", err)
		}
		keyWidth, valueWidth, ok := req.DeviceKV.LastVectorWidths()
		if !ok {
			return core.E("rocm.hip."+operation, "device KV cache has no pages", nil)
		}
		if keyWidth != req.Dim*keyHeads || valueWidth != req.Dim*keyHeads {
			return core.E("rocm.hip."+operation, "device KV widths must match attention dimension", nil)
		}
		if req.DeviceKV.TokenCount() != req.TokenCount {
			return core.E("rocm.hip."+operation, "device KV token count mismatch", nil)
		}
		launch.KVSource = hipAttentionKVSourceDevice
		launch.DescriptorPointer = req.DescriptorTable.Pointer()
		launch.DescriptorBytes = req.DescriptorTable.SizeBytes()
	}
	useSharedWeights := req.TokenCount <= hipAttentionHeadsSharedMaxTokens
	var attentionSharedMemBytes uint32
	var kernelSharedMemBytes uint32
	var weights *hipDeviceByteBuffer
	var err error
	if useSharedWeights {
		attentionSharedMemBytes, err = hipAttentionHeadsSharedMemBytes(req.TokenCount, req.DeviceKV != nil)
		if err != nil {
			return err
		}
		kernelSharedMemBytes, err = hipAttentionHeadsQueryRMSRoPESharedMemBytes(attentionSharedMemBytes, req.Dim)
		if err != nil {
			return err
		}
		launch.SharedMemBytes = uint64(attentionSharedMemBytes)
	} else {
		weightCount := req.QueryCount * req.HeadCount * req.TokenCount
		if workspace != nil && weightCount <= hipAttentionHeadsBatchWorkspaceMaxWeights {
			weights, err = workspace.EnsureBatchAttentionWeights(driver, weightCount)
			if err != nil {
				return err
			}
		} else {
			weights, err = hipAllocateByteBuffer(driver, "rocm.hip."+operation, "attention batch head weights", uint64(weightCount)*4, weightCount)
			if err != nil {
				return err
			}
			defer weights.Close()
		}
		launch.AttentionWeightPointer = weights.Pointer()
		launch.AttentionWeightBytes = uint64(weightCount) * 4
		kernelSharedMemBytes, err = hipAttentionHeadsQueryRMSRoPESharedMemBytes(0, req.Dim)
		if err != nil {
			return err
		}
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return err
	}
	config := hipKernelLaunchConfig{
		Name:           hipKernelNameAttentionHeadsBatchCausalQueryRMSRoPE,
		Args:           launchBytes,
		GridX:          uint32(req.HeadCount),
		GridY:          uint32(req.QueryCount),
		GridZ:          1,
		BlockX:         hipAttentionHeadsBlockSize(req.TokenCount),
		BlockY:         1,
		BlockZ:         1,
		SharedMemBytes: kernelSharedMemBytes,
	}
	if err := config.Validate(); err != nil {
		return err
	}
	return hipLaunchKernelContext(ctx, driver, config)
}

func hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernelWorkspace(ctx context.Context, driver nativeHIPDriver, req hipAttentionHeadsBatchCausalDeviceRequest, query *hipDeviceByteBuffer, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if req.Dim <= 0 || req.TokenCount <= 0 || req.HeadCount <= 0 || req.QueryCount <= 0 {
		return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch dimensions must be positive", nil)
	}
	keyHeads := firstPositiveInt(req.KeyHeads, 1)
	if keyHeads <= 0 || keyHeads > req.HeadCount || req.HeadCount%keyHeads != 0 {
		return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "key head count must divide query head count", nil)
	}
	if req.QueryStartToken < 0 || uint64(req.QueryStartToken)+uint64(req.QueryCount) > uint64(req.TokenCount) {
		return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "causal query window exceeds token count", nil)
	}
	if req.WindowSize < 0 {
		return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "window size must be non-negative", nil)
	}
	if req.Scale < 0 || math.IsNaN(float64(req.Scale)) || math.IsInf(float64(req.Scale), 0) {
		return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "scale must be non-negative and finite", nil)
	}
	queryCount := req.QueryCount * req.HeadCount * req.Dim
	if query == nil || query.Pointer() == 0 || query.Count() != queryCount || query.SizeBytes() != uint64(queryCount*4) {
		return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "attention query device buffer shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != queryCount || output.SizeBytes() != uint64(queryCount*4) {
		return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "attention output device buffer shape mismatch", nil)
	}
	launch := hipAttentionHeadsBatchCausalLaunchArgs{
		QueryPointer:    query.Pointer(),
		OutputPointer:   output.Pointer(),
		Dim:             req.Dim,
		TokenCount:      req.TokenCount,
		HeadCount:       req.HeadCount,
		KeyHeads:        keyHeads,
		QueryCount:      req.QueryCount,
		QueryStartToken: req.QueryStartToken,
		WindowSize:      req.WindowSize,
		QueryBytes:      query.SizeBytes(),
		OutputBytes:     output.SizeBytes(),
		Scale:           req.Scale,
	}
	if req.VisibleTokenCaps != nil {
		if req.VisibleTokenCaps.Pointer() == 0 || req.VisibleTokenCaps.Count() != req.QueryCount || req.VisibleTokenCaps.SizeBytes() != uint64(req.QueryCount*4) {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "visible-token cap device buffer shape mismatch", nil)
		}
		launch.VisibleCapPointer = req.VisibleTokenCaps.Pointer()
		launch.VisibleCapBytes = req.VisibleTokenCaps.SizeBytes()
	}
	if req.DeviceKV == nil {
		if req.DescriptorTable != nil {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "descriptor table requires device KV cache", nil)
		}
		if req.Key == nil || req.Key.Pointer() == 0 || req.Value == nil || req.Value.Pointer() == 0 {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "attention key and value device buffers are required", nil)
		}
		kvCount := req.TokenCount * keyHeads * req.Dim
		if req.Key.Count() != kvCount || req.Value.Count() != kvCount ||
			req.Key.SizeBytes() != uint64(kvCount*4) || req.Value.SizeBytes() != uint64(kvCount*4) {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "attention key/value device buffer shape mismatch", nil)
		}
		launch.KVSource = hipAttentionKVSourceContiguous
		launch.KeyPointer = req.Key.Pointer()
		launch.ValuePointer = req.Value.Pointer()
		launch.KeyBytes = req.Key.SizeBytes()
		launch.ValueBytes = req.Value.SizeBytes()
	} else {
		if req.Key != nil || req.Value != nil {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "device KV attention must not set contiguous KV buffers", nil)
		}
		if req.DescriptorTable == nil {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "device KV attention requires descriptor table", nil)
		}
		if err := req.DescriptorTable.CompatibleWith(req.DeviceKV); err != nil {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "descriptor table does not match device KV cache", err)
		}
		keyWidth, valueWidth, ok := req.DeviceKV.LastVectorWidths()
		if !ok {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "device KV cache has no pages", nil)
		}
		if keyWidth != req.Dim*keyHeads || valueWidth != req.Dim*keyHeads {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "device KV widths must match attention dimension", nil)
		}
		if req.DeviceKV.TokenCount() != req.TokenCount {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "device KV token count mismatch", nil)
		}
		launch.KVSource = hipAttentionKVSourceDevice
		launch.DescriptorPointer = req.DescriptorTable.Pointer()
		launch.DescriptorBytes = req.DescriptorTable.SizeBytes()
	}
	if hipAttentionHeadsBatchChunkedEligible(req, workspace) {
		return hipRunAttentionHeadsBatchChunkedOutputFromDeviceQueryToDeviceKernelWorkspace(ctx, driver, req, query, output, workspace)
	}
	useSharedWeights := req.TokenCount <= hipAttentionHeadsSharedMaxTokens
	var sharedMemBytes uint32
	var weights *hipDeviceByteBuffer
	var err error
	if useSharedWeights {
		sharedMemBytes, err = hipAttentionHeadsSharedMemBytes(req.TokenCount, req.DeviceKV != nil)
		if err != nil {
			return err
		}
		launch.SharedMemBytes = uint64(sharedMemBytes)
	} else {
		weightCount := req.QueryCount * req.HeadCount * req.TokenCount
		if workspace != nil && weightCount <= hipAttentionHeadsBatchWorkspaceMaxWeights {
			weights, err = workspace.EnsureBatchAttentionWeights(driver, weightCount)
			if err != nil {
				return err
			}
		} else {
			weights, err = hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch head weights", uint64(weightCount)*4, weightCount)
			if err != nil {
				return err
			}
			defer weights.Close()
		}
		launch.WeightPointer = weights.Pointer()
		launch.WeightBytes = uint64(weightCount) * 4
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return err
	}
	kernelName := hipKernelNameAttentionHeadsBatchCausal
	if req.VisibleTokenCaps != nil {
		kernelName = hipKernelNameAttentionHeadsBatchCapped
	}
	config := hipKernelLaunchConfig{
		Name:           kernelName,
		Args:           launchBytes,
		GridX:          uint32(req.HeadCount),
		GridY:          uint32(req.QueryCount),
		GridZ:          1,
		BlockX:         hipAttentionHeadsBlockSize(req.TokenCount),
		BlockY:         1,
		BlockZ:         1,
		SharedMemBytes: sharedMemBytes,
	}
	if err := config.Validate(); err != nil {
		return err
	}
	return hipLaunchKernelContext(ctx, driver, config)
}

type hipAttentionHeadsBatchChunkedEligibilityReason uint8

const hipAttentionHeadsBatchChunkedGQA2Env = "GO_ROCM_ENABLE_EXPERIMENTAL_GQA2_CHUNKED_ATTENTION"

var hipAttentionHeadsBatchChunkedGQA2Enabled = os.Getenv(hipAttentionHeadsBatchChunkedGQA2Env) == "1"

func hipAttentionHeadsBatchChunkedGQA2Eligible(headCount, keyHeads int) bool {
	if !hipAttentionHeadsBatchChunkedGQA2Enabled || headCount <= 0 || keyHeads <= 0 || headCount%keyHeads != 0 {
		return false
	}
	queryHeadsPerKV := headCount / keyHeads
	return queryHeadsPerKV >= 2 && queryHeadsPerKV%2 == 0
}

func hipAttentionHeadsBatchChunkedStage1LaunchConfig(args []byte, queryCount, headCount, keyHeads, chunkCount, chunkSize, dim int) (hipKernelLaunchConfig, error) {
	name := hipKernelNameAttentionHeadsBatchChunkedStage1
	stage1HeadRows := queryCount * headCount
	sharedMemBytes, err := hipAttentionHeadsChunkedSharedMemBytes(chunkSize, dim)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	if hipAttentionHeadsBatchChunkedGQA2Eligible(headCount, keyHeads) {
		name = hipKernelNameAttentionHeadsBatchChunkedStage1GQA2
		stage1HeadRows = queryCount * (headCount / 2)
		sharedMemBytes, err = hipAttentionHeadsBatchChunkedGQA2SharedMemBytes(chunkSize, dim)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
	}
	blocks, err := rocmDeviceKVPositiveUint32("attention batch chunked stage1 blocks", stage1HeadRows*chunkCount)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:           name,
		Args:           args,
		GridX:          blocks,
		GridY:          1,
		GridZ:          1,
		BlockX:         hipAttentionHeadsChunkedBlockSize,
		BlockY:         1,
		BlockZ:         1,
		SharedMemBytes: sharedMemBytes,
	}
	return config, config.Validate()
}

func hipAttentionHeadsBatchChunkedGQA2SharedMemBytes(chunkSize, dim int) (uint32, error) {
	chunk, err := rocmDeviceKVPositiveUint32("attention GQA2 chunked chunk size", chunkSize)
	if err != nil {
		return 0, err
	}
	width, err := rocmDeviceKVPositiveUint32("attention GQA2 chunked query dim", dim)
	if err != nil {
		return 0, err
	}
	bytes := uint64(chunk) * 4
	bytes = hipAttentionHeadsAlignSharedBytes(bytes, 4)
	bytes += uint64(chunk) * 4
	bytes = hipAttentionHeadsAlignSharedBytes(bytes, 8)
	bytes += uint64(chunk) * 8
	bytes = hipAttentionHeadsAlignSharedBytes(bytes, 4)
	bytes += uint64(chunk) * 4
	bytes = hipAttentionHeadsAlignSharedBytes(bytes, 4)
	bytes += uint64(width) * 4
	bytes = hipAttentionHeadsAlignSharedBytes(bytes, 4)
	bytes += uint64(width) * 4
	if bytes > math.MaxUint32 {
		return 0, core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "attention GQA2 chunked shared memory byte count is out of uint32 range", nil)
	}
	return uint32(bytes), nil
}

const (
	hipAttentionHeadsBatchChunkedEligibilityEligible hipAttentionHeadsBatchChunkedEligibilityReason = iota
	hipAttentionHeadsBatchChunkedEligibilityInvalidHeads
	hipAttentionHeadsBatchChunkedEligibilityMissingWorkspace
	hipAttentionHeadsBatchChunkedEligibilityMissingDeviceKV
	hipAttentionHeadsBatchChunkedEligibilityMissingDescriptor
	hipAttentionHeadsBatchChunkedEligibilityInvalidDimension
	hipAttentionHeadsBatchChunkedEligibilityBelowTokenThreshold
	hipAttentionHeadsBatchChunkedEligibilityUnsupportedKVMode
	hipAttentionHeadsBatchChunkedEligibilityTokenCountMismatch
	hipAttentionHeadsBatchChunkedEligibilityVisibleCaps
)

func hipAttentionHeadsBatchChunkedEligibilityReasonFor(req hipAttentionHeadsBatchCausalDeviceRequest, workspace *hipAttentionHeadsChunkedWorkspace) hipAttentionHeadsBatchChunkedEligibilityReason {
	if req.VisibleTokenCaps != nil {
		return hipAttentionHeadsBatchChunkedEligibilityVisibleCaps
	}
	keyHeads := firstPositiveInt(req.KeyHeads, 1)
	if keyHeads <= 0 || keyHeads > req.HeadCount || req.HeadCount%keyHeads != 0 {
		return hipAttentionHeadsBatchChunkedEligibilityInvalidHeads
	}
	if workspace == nil {
		return hipAttentionHeadsBatchChunkedEligibilityMissingWorkspace
	}
	if req.DeviceKV == nil {
		return hipAttentionHeadsBatchChunkedEligibilityMissingDeviceKV
	}
	if req.DescriptorTable == nil {
		return hipAttentionHeadsBatchChunkedEligibilityMissingDescriptor
	}
	if req.Dim <= 0 || req.Dim > hipAttentionHeadsChunkedBlockSize || (keyHeads > 1 && req.Dim&1 != 0) {
		return hipAttentionHeadsBatchChunkedEligibilityInvalidDimension
	}
	minTokenCount := hipAttentionHeadsSharedMaxTokens
	if req.Dim == hipAttentionHeadsChunkedBlockSize ||
		(req.Dim == hipAttentionHeadsChunkedBlockSize/2 && req.WindowSize > 0) {
		minTokenCount = hipAttentionHeadsChunkedBlockSize / 4
	}
	if req.TokenCount <= minTokenCount {
		return hipAttentionHeadsBatchChunkedEligibilityBelowTokenThreshold
	}
	if req.DeviceKV.mode != rocmKVCacheModeKQ8VQ4 {
		return hipAttentionHeadsBatchChunkedEligibilityUnsupportedKVMode
	}
	if req.DeviceKV.TokenCount() != req.TokenCount {
		return hipAttentionHeadsBatchChunkedEligibilityTokenCountMismatch
	}
	return hipAttentionHeadsBatchChunkedEligibilityEligible
}

func hipAttentionHeadsBatchChunkedEligible(req hipAttentionHeadsBatchCausalDeviceRequest, workspace *hipAttentionHeadsChunkedWorkspace) bool {
	return hipAttentionHeadsBatchChunkedEligibilityReasonFor(req, workspace) == hipAttentionHeadsBatchChunkedEligibilityEligible
}

func hipRunAttentionHeadsBatchChunkedOutputFromDeviceQueryToDeviceKernelWorkspace(ctx context.Context, driver nativeHIPDriver, req hipAttentionHeadsBatchCausalDeviceRequest, query *hipDeviceByteBuffer, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if workspace == nil {
		return core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "attention workspace is required", nil)
	}
	if req.DeviceKV == nil || req.DescriptorTable == nil {
		return core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "device KV cache and descriptor table are required", nil)
	}
	if req.Dim <= 0 || req.Dim > hipAttentionHeadsChunkedBlockSize || req.TokenCount <= 0 || req.HeadCount <= 0 || req.QueryCount <= 0 {
		return core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "attention batch dimensions are unsupported", nil)
	}
	keyHeads := firstPositiveInt(req.KeyHeads, 1)
	if keyHeads <= 0 || keyHeads > req.HeadCount || req.HeadCount%keyHeads != 0 {
		return core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "key head count must divide query head count", nil)
	}
	if keyHeads > 1 && req.Dim&1 != 0 {
		return core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "multi-KV q4 attention requires an even head dimension", nil)
	}
	if req.QueryStartToken < 0 || uint64(req.QueryStartToken)+uint64(req.QueryCount) > uint64(req.TokenCount) {
		return core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "causal query window exceeds token count", nil)
	}
	if req.WindowSize < 0 {
		return core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "window size must be non-negative", nil)
	}
	if req.Scale < 0 || math.IsNaN(float64(req.Scale)) || math.IsInf(float64(req.Scale), 0) {
		return core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "scale must be non-negative and finite", nil)
	}
	queryCount := req.QueryCount * req.HeadCount * req.Dim
	if query == nil || query.Pointer() == 0 || query.Count() != queryCount || query.SizeBytes() != uint64(queryCount*4) {
		return core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "attention query device buffer shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != queryCount || output.SizeBytes() != uint64(queryCount*4) {
		return core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "attention output device buffer shape mismatch", nil)
	}
	if err := req.DescriptorTable.CompatibleWith(req.DeviceKV); err != nil {
		return core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "descriptor table does not match device KV cache", err)
	}
	keyWidth, valueWidth, ok := req.DeviceKV.LastVectorWidths()
	if !ok {
		return core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "device KV cache has no pages", nil)
	}
	kvWidth := req.Dim * keyHeads
	if req.DeviceKV.mode != rocmKVCacheModeKQ8VQ4 || keyWidth != kvWidth || valueWidth != kvWidth || req.DeviceKV.TokenCount() != req.TokenCount {
		return core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "device KV cache shape is unsupported", nil)
	}

	chunkSize := hipAttentionHeadsChunkSize
	chunkStartToken, chunkCount := hipAttentionHeadsBatchChunkedActiveRange(req.QueryStartToken, req.QueryCount, req.TokenCount, req.WindowSize, chunkSize)
	workspaceHeadRows := req.HeadCount * req.QueryCount
	workspaceTokens := chunkCount * chunkSize
	if err := workspace.Ensure(driver, workspaceHeadRows, req.Dim, workspaceTokens, chunkSize); err != nil {
		return err
	}
	launch := hipAttentionHeadsBatchChunkedLaunchArgs{
		QueryPointer:      query.Pointer(),
		DescriptorPointer: req.DescriptorTable.Pointer(),
		PartialPointer:    workspace.Partial.Pointer(),
		StatsPointer:      workspace.Stats.Pointer(),
		OutputPointer:     output.Pointer(),
		Dim:               req.Dim,
		TokenCount:        req.TokenCount,
		HeadCount:         req.HeadCount,
		KeyHeads:          keyHeads,
		QueryCount:        req.QueryCount,
		QueryStartToken:   req.QueryStartToken,
		WindowSize:        req.WindowSize,
		ChunkStartToken:   chunkStartToken,
		ChunkSize:         chunkSize,
		ChunkCount:        chunkCount,
		QueryBytes:        query.SizeBytes(),
		DescriptorBytes:   req.DescriptorTable.SizeBytes(),
		PartialBytes:      uint64(workspaceHeadRows * chunkCount * req.Dim * 4),
		StatsBytes:        uint64(workspaceHeadRows * chunkCount * 2 * 4),
		OutputBytes:       output.SizeBytes(),
		Scale:             req.Scale,
	}
	launchBytes, err := launch.BinaryInto(workspace.BatchChunkedStage1Args[:])
	if err != nil {
		return err
	}
	stage2LaunchBytes := workspace.BatchChunkedStage2Args[:len(launchBytes)]
	copy(stage2LaunchBytes, launchBytes)
	stage1, err := hipAttentionHeadsBatchChunkedStage1LaunchConfig(launchBytes, req.QueryCount, req.HeadCount, keyHeads, chunkCount, chunkSize, req.Dim)
	if err != nil {
		return err
	}
	if err := hipLaunchKernelContext(ctx, driver, stage1); err != nil {
		return err
	}
	stage2Blocks, err := rocmDeviceKVPositiveUint32("attention batch chunked stage2 blocks", workspaceHeadRows)
	if err != nil {
		return err
	}
	stage2 := hipKernelLaunchConfig{
		Name:           hipKernelNameAttentionHeadsBatchChunkedStage2,
		Args:           stage2LaunchBytes,
		GridX:          stage2Blocks,
		GridY:          1,
		GridZ:          1,
		BlockX:         hipAttentionHeadsChunkedBlockSize,
		BlockY:         1,
		BlockZ:         1,
		SharedMemBytes: 0,
	}
	if err := stage2.Validate(); err != nil {
		return err
	}
	return hipLaunchKernelContext(ctx, driver, stage2)
}

func hipAttentionHeadsBatchChunkedActiveRange(queryStartToken, queryCount, tokenCount, windowSize, chunkSize int) (int, int) {
	if chunkSize <= 0 || tokenCount <= 0 || queryCount <= 0 {
		return 0, 0
	}
	activeEnd := queryStartToken + queryCount
	if activeEnd > tokenCount {
		activeEnd = tokenCount
	}
	if activeEnd < 0 {
		activeEnd = 0
	}
	activeStart := 0
	if windowSize > 0 {
		earliestVisible := queryStartToken + 1
		activeStart = earliestVisible - windowSize
		if activeStart < 0 {
			activeStart = 0
		}
		activeStart = (activeStart / chunkSize) * chunkSize
	}
	if activeStart > activeEnd {
		activeStart = activeEnd
	}
	chunkCount := (activeEnd - activeStart + chunkSize - 1) / chunkSize
	if chunkCount <= 0 {
		chunkCount = 1
	}
	return activeStart, chunkCount
}

type hipAttentionHeadsChunkedWorkspace struct {
	Partial                                    *hipDeviceByteBuffer
	Stats                                      *hipDeviceByteBuffer
	ChunkedStage1Args                          [hipAttentionHeadsChunkedLaunchArgsBytes]byte
	ChunkedStage2Args                          [hipAttentionHeadsChunkedLaunchArgsBytes]byte
	BatchChunkedStage1Args                     [hipAttentionHeadsBatchChunkedLaunchArgsBytes]byte
	BatchChunkedStage2Args                     [hipAttentionHeadsBatchChunkedLaunchArgsBytes]byte
	VectorAddScaledArgs                        [hipVectorAddScaledLaunchArgsBytes]byte
	VectorScaleArgs                            [hipVectorScaleLaunchArgsBytes]byte
	RMSNormArgs                                [hipRMSNormLaunchArgsBytes]byte
	RMSResidualAddArgs                         [hipRMSNormResidualAddArgsBytes]byte
	RMSResidualAddNormArgs                     [hipRMSNormResAddNormArgsBytes]byte
	RMSNormHeadsArgs                           [hipRMSNormHeadsLaunchArgsBytes]byte
	RMSNormRoPEHeadsArgs                       [hipRMSNormRoPEHeadsLaunchArgsBytes]byte
	RMSNormRoPEHeadsPairArgs                   [hipRMSNormRoPEHeadsPairLaunchArgsBytes]byte
	RMSResidualAddGELUTanhProjArgs             [hipRMSResidualAddGELUTanhProjLaunchArgsBytes]byte
	EmbeddingLookupArgs                        [hipEmbeddingLookupLaunchArgsBytes]byte
	KVEncodeTokenArgs                          [hipKVEncodeTokenLaunchArgsBytes]byte
	KVEncodeTokenValueNormArgs                 [hipKVEncodeTokenValueNormLaunchArgsBytes]byte
	KVEncodeTokenValueNormDescriptorAppendArgs [hipKVEncodeTokenValueNormDescriptorAppendLaunchArgsBytes]byte
	KVDescriptorAppendArgs                     [hipKVDescriptorAppendLaunchArgsBytes]byte
	TokenID                                    *hipDeviceByteBuffer
	TokenIDLoaded                              bool
	TokenIDValue                               int32
	EmbeddingOutputs                           map[int]*hipDeviceByteBuffer
	ScaledEmbeddings                           map[int]*hipDeviceByteBuffer
	ScaledEmbeddingFixed                       hipDeviceByteBuffer
	ScaledEmbeddingFixedCap                    int
	ScaledEmbeddingView                        hipDeviceByteBuffer
	PerLayerEmbeddings                         map[int]*hipDeviceByteBuffer
	PerLayerProjected                          map[int]*hipDeviceByteBuffer
	PerLayerProjectedFixed                     hipDeviceByteBuffer
	PerLayerProjectedCap                       int
	PerLayerProjectedView                      hipDeviceByteBuffer
	PerLayerScaled                             map[int]*hipDeviceByteBuffer
	PerLayerScaledFixed                        hipDeviceByteBuffer
	PerLayerScaledFixedCap                     int
	PerLayerScaledView                         hipDeviceByteBuffer
	PerLayerProjScaled                         map[int]*hipDeviceByteBuffer
	PerLayerNorm                               map[int]*hipDeviceByteBuffer
	PerLayerCombined                           map[int]*hipDeviceByteBuffer
	PerLayerOutput                             map[int]*hipDeviceByteBuffer
	PerLayerOutputFixed                        hipDeviceByteBuffer
	PerLayerOutputFixedCap                     int
	PerLayerOutputView                         hipDeviceByteBuffer
	AttentionOutputs                           map[int]*hipDeviceByteBuffer
	AttentionOutputFixed                       hipDeviceByteBuffer
	AttentionOutputFixedCap                    int
	AttentionOutputView                        hipDeviceByteBuffer
	ProjectionOutputs                          map[int]*hipDeviceByteBuffer
	ProjectionOutputFixed                      hipDeviceByteBuffer
	ProjectionOutputCap                        int
	ProjectionArgs                             [hipMLXQ4ProjectionLaunchArgsBytes]byte
	TripleProjectionArgs                       [hipMLXQ4TripleProjLaunchArgsBytes]byte
	GELUTanhMulArgs                            [hipMLXQ4GELUTanhMulLaunchArgsBytes]byte
	GELUTanhMLPPersistentArgs                  [hipMLXQ4GELUTanhMLPPersistentLaunchArgsBytes]byte
	GELUTanhProjArgs                           [hipMLXQ4GELUTanhProjLaunchArgsBytes]byte
	GELUTanhMLPBarrier                         *hipDeviceByteBuffer
	KVProjectionOutputs                        [2]map[int]*hipDeviceByteBuffer
	KVProjectionPairOutputs                    map[int]*hipDeviceByteBuffer
	KVProjectionPairFixed                      hipDeviceByteBuffer
	KVProjectionPairCap                        int
	KVProjectionOutputViews                    [2]hipDeviceByteBuffer
	PrefillInputNormOutput                     map[int]*hipDeviceByteBuffer
	PrefillInputNormFixed                      hipDeviceByteBuffer
	PrefillInputNormCap                        int
	PrefillInputNormView                       hipDeviceByteBuffer
	ActivationOutputs                          map[int]*hipDeviceByteBuffer
	ActivationOutputFixed                      hipDeviceByteBuffer
	ActivationOutputCap                        int
	RMSResidualOutputs                         map[int]*hipDeviceByteBuffer
	RMSNormOutputs                             map[int]*hipDeviceByteBuffer
	RMSResidualNormOutputs                     map[int]*hipDeviceByteBuffer
	RMSResidualNormFixed                       hipDeviceByteBuffer
	RMSResidualNormCap                         int
	RMSRoPEOutputs                             map[int]*hipDeviceByteBuffer
	RMSRoPEFixed                               hipDeviceByteBuffer
	RMSRoPEFixedCap                            int
	RMSRoPEOutputView                          hipDeviceByteBuffer
	KeyRMSRoPEOutputs                          map[int]*hipDeviceByteBuffer
	KeyRMSRoPEOutputView                       hipDeviceByteBuffer
	RMSNoScaleOutputs                          map[int]*hipDeviceByteBuffer
	RMSNoScaleOutputView                       hipDeviceByteBuffer
	KeyValueNormOutputs                        map[int]*hipDeviceByteBuffer
	KeyValueNormFixed                          hipDeviceByteBuffer
	KeyValueNormCap                            int
	KeyValueNormViews                          [2]hipDeviceByteBuffer
	IntermediateOutputs                        map[int]*hipDeviceByteBuffer
	IntermediateFixed                          hipDeviceByteBuffer
	IntermediateFixedCap                       int
	QKVOutputs                                 map[int]*hipDeviceByteBuffer
	QKVOutputFixed                             hipDeviceByteBuffer
	QKVOutputCap                               int
	ProjectionScore                            *hipDeviceByteBuffer
	ProjectionScoresArgs                       [hipMLXQ4ProjectionLaunchArgsBytes]byte
	ProjectionScoreBytes                       []byte
	ProjectionTopK                             *hipDeviceByteBuffer
	ProjectionTopKCap                          int
	ProjectionTopKView                         hipDeviceByteBuffer
	ProjectionTopKWork                         *hipDeviceByteBuffer
	ProjectionTopKWorkCap                      int
	ProjectionTopKWorkView                     hipDeviceByteBuffer
	ProjectionTopKArgs                         [hipPackedTopKLaunchArgsBytes]byte
	ProjectionTopKSampleArgs                   [hipPackedTopKSampleLaunchArgsBytes]byte
	OrderedEmbeddingCandidatesArgs             [hipOrderedEmbeddingCandidatesLaunchArgsBytes]byte
	ProjectionTopKBytes                        []byte
	ProjectionTopPacked                        []uint64
	ProjectionCandidates                       []hipGreedySampleResult
	ProjectionCandidateTokens                  []int32
	ProjectionCandidateTokenOutput             *hipDeviceByteBuffer
	ProjectionCandidateTokenCap                int
	ProjectionCandidateTokenView               hipDeviceTokenBuffer
	ProjectionGreedyBest                       []*hipDeviceByteBuffer
	ProjectionGreedyView                       hipDeviceByteBuffer
	ProjectionGreedyNext                       int
	GreedyFirstSlabSlots                       int
	ProjectionOutputView                       hipDeviceByteBuffer
	ActivationOutputView                       hipDeviceByteBuffer
	QKVOutputView                              hipDeviceByteBuffer
	RMSResidualNormViews                       [2]hipDeviceByteBuffer
	RMSResidualOutputView                      hipDeviceByteBuffer
	RMSNormOutputView                          hipDeviceByteBuffer
	IntermediateOutputView                     hipDeviceByteBuffer
	SampleCandidates                           []hipReferenceCandidate
	SampleWeights                              []float64
	BatchAttentionWeight                       *hipDeviceByteBuffer
	FinalHiddenOutputs                         [2]map[int]*hipDeviceByteBuffer
	FinalHiddenPairOutputs                     map[int]*hipDeviceByteBuffer
	FinalHiddenPairFixed                       hipDeviceByteBuffer
	FinalHiddenPairFixedCap                    int
	FinalHiddenOutputViews                     [2]hipDeviceByteBuffer
	NextInputOutputs                           [2]map[int]*hipDeviceByteBuffer
	NextInputPairOutputs                       map[int]*hipDeviceByteBuffer
	NextInputPairFixed                         hipDeviceByteBuffer
	NextInputPairFixedCap                      int
	NextInputOutputViews                       [2]hipDeviceByteBuffer
	PerLayerInputSet                           hipGemma4Q4PerLayerInputDeviceSet
	PerLayerInputBacking                       [1]*hipDeviceByteBuffer
	AssistantDraftCombinedFixed                hipDeviceByteBuffer
	AssistantDraftCombinedCap                  int
	AssistantDraftCombinedView                 hipDeviceByteBuffer
	AssistantDraftInputHiddenFixed             hipDeviceByteBuffer
	AssistantDraftInputHiddenCap               int
	AssistantDraftInputHiddenView              hipDeviceByteBuffer
	MoE                                        hipGemma4MoEWorkspace
	PrefillTokenBuffer                         *hipDeviceTokenBuffer
	PrefillTokenView                           hipDeviceTokenBuffer
	PrefillTokenPayload                        []byte
	SuppressTokenIDs                           []int32
	SuppressTokenBuffer                        *hipDeviceTokenBuffer
	SuppressTokenView                          hipDeviceTokenBuffer
	SuppressTokenPayload                       []byte
	SuppressTokenInlineIDs                     [hipProjectionGreedySuppressReserveBytes / 4]int32
	SuppressTokenInlineData                    [hipProjectionGreedySuppressReserveBytes]byte
	partialCap                                 int
	statsCap                                   int
	batchWeightCap                             int
}

var hipAttentionHeadsChunkedWorkspacePool = struct {
	sync.Mutex
	workspaces []*hipAttentionHeadsChunkedWorkspace
}{
	workspaces: make([]*hipAttentionHeadsChunkedWorkspace, 0, hipAttentionHeadsChunkedWorkspacePoolMax),
}

const (
	hipAttentionHeadsChunkedWorkspacePoolMax   = 64
	hipAttentionHeadsChunkedWorkspaceWarmDepth = 8
)

func hipNewAttentionHeadsChunkedWorkspace() *hipAttentionHeadsChunkedWorkspace {
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	workspace.initHostMaps()
	return workspace
}

func hipBorrowAttentionHeadsChunkedWorkspace() *hipAttentionHeadsChunkedWorkspace {
	hipAttentionHeadsChunkedWorkspacePool.Lock()
	count := len(hipAttentionHeadsChunkedWorkspacePool.workspaces)
	if count > 0 {
		workspace := hipAttentionHeadsChunkedWorkspacePool.workspaces[count-1]
		hipAttentionHeadsChunkedWorkspacePool.workspaces[count-1] = nil
		hipAttentionHeadsChunkedWorkspacePool.workspaces = hipAttentionHeadsChunkedWorkspacePool.workspaces[:count-1]
		hipAttentionHeadsChunkedWorkspacePool.Unlock()
		workspace.initHostMaps()
		return workspace
	}
	hipAttentionHeadsChunkedWorkspacePool.Unlock()
	workspace := hipNewAttentionHeadsChunkedWorkspace()
	workspace.initHostMaps()
	return workspace
}

func hipReleaseAttentionHeadsChunkedWorkspace(workspace *hipAttentionHeadsChunkedWorkspace) bool {
	if workspace == nil {
		return false
	}
	hipAttentionHeadsChunkedWorkspacePool.Lock()
	released := false
	if len(hipAttentionHeadsChunkedWorkspacePool.workspaces) < hipAttentionHeadsChunkedWorkspacePoolMax {
		hipAttentionHeadsChunkedWorkspacePool.workspaces = append(hipAttentionHeadsChunkedWorkspacePool.workspaces, workspace)
		released = true
	}
	hipAttentionHeadsChunkedWorkspacePool.Unlock()
	return released
}

func hipRecycleAttentionHeadsChunkedWorkspace(workspace *hipAttentionHeadsChunkedWorkspace) error {
	if workspace == nil {
		return nil
	}
	greedyBest := workspace.ProjectionGreedyBest
	workspace.ProjectionGreedyBest = nil
	err := workspace.Close()
	workspace.ProjectionGreedyBest = greedyBest
	workspace.ProjectionGreedyView = hipDeviceByteBuffer{}
	workspace.ProjectionGreedyNext = 0
	workspace.GreedyFirstSlabSlots = 0
	if hipReleaseAttentionHeadsChunkedWorkspace(workspace) {
		return err
	}
	for _, output := range greedyBest {
		if closeErr := output.Close(); closeErr != nil && err == nil {
			err = closeErr
		}
	}
	workspace.ProjectionGreedyBest = nil
	return err
}

func hipPrewarmAttentionHeadsChunkedWorkspacePool() {
	workspaces := make([]*hipAttentionHeadsChunkedWorkspace, 0, hipAttentionHeadsChunkedWorkspaceWarmDepth)
	for range hipAttentionHeadsChunkedWorkspaceWarmDepth {
		workspaces = append(workspaces, hipBorrowAttentionHeadsChunkedWorkspace())
	}
	for _, workspace := range workspaces {
		hipReleaseAttentionHeadsChunkedWorkspace(workspace)
	}
}

func (workspace *hipAttentionHeadsChunkedWorkspace) initHostMaps() {
}

func (workspace *hipAttentionHeadsChunkedWorkspace) resetBorrowedViews() {
	if workspace == nil {
		return
	}
	workspace.ProjectionGreedyView = hipDeviceByteBuffer{}
	workspace.ProjectionGreedyNext = 0
	workspace.ScaledEmbeddingView = hipDeviceByteBuffer{}
	workspace.PerLayerProjectedView = hipDeviceByteBuffer{}
	workspace.PerLayerScaledView = hipDeviceByteBuffer{}
	workspace.PerLayerOutputView = hipDeviceByteBuffer{}
	workspace.AttentionOutputView = hipDeviceByteBuffer{}
	workspace.ProjectionOutputView = hipDeviceByteBuffer{}
	workspace.PrefillInputNormView = hipDeviceByteBuffer{}
	workspace.ProjectionTopKView = hipDeviceByteBuffer{}
	workspace.ProjectionTopKWorkView = hipDeviceByteBuffer{}
	workspace.ProjectionCandidateTokenView = hipDeviceTokenBuffer{}
	workspace.ActivationOutputView = hipDeviceByteBuffer{}
	workspace.RMSRoPEOutputView = hipDeviceByteBuffer{}
	workspace.KeyRMSRoPEOutputView = hipDeviceByteBuffer{}
	workspace.RMSNoScaleOutputView = hipDeviceByteBuffer{}
	workspace.RMSResidualOutputView = hipDeviceByteBuffer{}
	workspace.RMSNormOutputView = hipDeviceByteBuffer{}
	workspace.IntermediateOutputView = hipDeviceByteBuffer{}
	workspace.QKVOutputView = hipDeviceByteBuffer{}
	workspace.PrefillTokenBuffer = nil
	workspace.PrefillTokenView = hipDeviceTokenBuffer{}
	workspace.SuppressTokenBuffer = nil
	workspace.SuppressTokenView = hipDeviceTokenBuffer{}
	for index := range workspace.KVProjectionOutputViews {
		workspace.KVProjectionOutputViews[index] = hipDeviceByteBuffer{}
	}
	for index := range workspace.RMSResidualNormViews {
		workspace.RMSResidualNormViews[index] = hipDeviceByteBuffer{}
	}
	for index := range workspace.KeyValueNormViews {
		workspace.KeyValueNormViews[index] = hipDeviceByteBuffer{}
	}
	for index := range workspace.FinalHiddenOutputViews {
		workspace.FinalHiddenOutputViews[index] = hipDeviceByteBuffer{}
	}
	for index := range workspace.NextInputOutputViews {
		workspace.NextInputOutputViews[index] = hipDeviceByteBuffer{}
	}
	workspace.PerLayerInputSet = hipGemma4Q4PerLayerInputDeviceSet{}
	workspace.PerLayerInputBacking = [1]*hipDeviceByteBuffer{}
	workspace.AssistantDraftCombinedView = hipDeviceByteBuffer{}
	workspace.AssistantDraftInputHiddenView = hipDeviceByteBuffer{}
	workspace.MoE.resetBorrowedViews()
}

const (
	hipProjectionGreedyBestWorkspaceSlots         = 4096
	hipProjectionGreedyPrefillReserveBytes        = 8192
	hipProjectionGreedySuppressReserveBytes       = 96
	hipProjectionGreedyPrefillReserveSlots        = hipProjectionGreedyPrefillReserveBytes / hipMLXQ4ProjectionBestBytes
	hipProjectionGreedySuppressReserveSlots       = hipProjectionGreedySuppressReserveBytes / hipMLXQ4ProjectionBestBytes
	hipProjectionGreedyReserveSlots               = hipProjectionGreedyPrefillReserveSlots + hipProjectionGreedySuppressReserveSlots
	hipProjectionGreedyBestWorkspaceUseSlots      = hipProjectionGreedyBestWorkspaceSlots - hipProjectionGreedyPrefillReserveSlots - hipProjectionGreedySuppressReserveSlots
	hipProjectionGreedyPrefillReserveOffsetBytes  = hipProjectionGreedyBestWorkspaceUseSlots * hipMLXQ4ProjectionBestBytes
	hipProjectionGreedySuppressReserveOffsetBytes = (hipProjectionGreedyBestWorkspaceSlots - hipProjectionGreedySuppressReserveSlots) * hipMLXQ4ProjectionBestBytes
	hipProjectionGreedyReservedWorkspaceSlabIdx   = 0
)

func hipProjectionGreedyRoundFirstSlabSlots(slots int) int {
	minSlots := hipProjectionGreedyReserveSlots + 1
	if slots < minSlots {
		slots = minSlots
	}
	const align = 16
	return (slots + align - 1) / align * align
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureProjectionGreedyBestCapacity(greedySlots int) {
	if workspace == nil || greedySlots <= 0 {
		return
	}
	if len(workspace.ProjectionGreedyBest) > 0 {
		if slots := workspace.projectionGreedyExistingFirstSlabSlots(); slots > 0 {
			workspace.GreedyFirstSlabSlots = slots
		}
		return
	}
	workspace.GreedyFirstSlabSlots = hipProjectionGreedyRoundFirstSlabSlots(hipProjectionGreedyReserveSlots + greedySlots)
}

func (workspace *hipAttentionHeadsChunkedWorkspace) projectionGreedyFirstSlabSlots() int {
	if workspace != nil && workspace.GreedyFirstSlabSlots > 0 {
		return hipProjectionGreedyRoundFirstSlabSlots(workspace.GreedyFirstSlabSlots)
	}
	if slots := workspace.projectionGreedyExistingFirstSlabSlots(); slots > 0 {
		return slots
	}
	return hipProjectionGreedyBestWorkspaceSlots
}

func (workspace *hipAttentionHeadsChunkedWorkspace) projectionGreedyExistingFirstSlabSlots() int {
	if workspace == nil || len(workspace.ProjectionGreedyBest) == 0 || workspace.ProjectionGreedyBest[0] == nil {
		return 0
	}
	sizeBytes := workspace.ProjectionGreedyBest[0].SizeBytes()
	if sizeBytes == 0 || sizeBytes%hipMLXQ4ProjectionBestBytes != 0 {
		return 0
	}
	slots := int(sizeBytes / hipMLXQ4ProjectionBestBytes)
	if slots < hipProjectionGreedyReserveSlots+1 {
		return 0
	}
	return slots
}

func (workspace *hipAttentionHeadsChunkedWorkspace) projectionGreedyFirstSlabUseSlots() int {
	return workspace.projectionGreedyFirstSlabSlots() - hipProjectionGreedyReserveSlots
}

func (workspace *hipAttentionHeadsChunkedWorkspace) projectionGreedyPrefillReserveOffsetBytes() int {
	return workspace.projectionGreedyFirstSlabUseSlots() * hipMLXQ4ProjectionBestBytes
}

func (workspace *hipAttentionHeadsChunkedWorkspace) projectionGreedySuppressReserveOffsetBytes() int {
	return (workspace.projectionGreedyFirstSlabSlots() - hipProjectionGreedySuppressReserveSlots) * hipMLXQ4ProjectionBestBytes
}

func (workspace *hipAttentionHeadsChunkedWorkspace) Ensure(driver nativeHIPDriver, headCount, dim, tokenCount, chunkSize int) error {
	if workspace == nil {
		return core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if headCount <= 0 || dim <= 0 || tokenCount <= 0 || chunkSize <= 0 {
		return core.E("rocm.hip.AttentionHeadsChunkedLaunch", "workspace dimensions must be positive", nil)
	}
	chunkCount := (tokenCount + chunkSize - 1) / chunkSize
	partialCount := headCount * chunkCount * dim
	statsCount := headCount * chunkCount * 2
	partialCap := hipAttentionHeadsChunkedWorkspaceCapacityCount(partialCount)
	statsCap := hipAttentionHeadsChunkedWorkspaceCapacityCount(statsCount)
	if workspace.Partial == nil || workspace.Partial.Pointer() == 0 || workspace.partialCap < partialCount {
		if err := workspace.Partial.Close(); err != nil {
			return err
		}
		partial, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsChunkedLaunch", "attention chunked partials", uint64(partialCap*4), partialCap)
		if err != nil {
			return err
		}
		workspace.Partial = partial
		workspace.partialCap = partialCap
	}
	if workspace.Stats == nil || workspace.Stats.Pointer() == 0 || workspace.statsCap < statsCount {
		if err := workspace.Stats.Close(); err != nil {
			return err
		}
		stats, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsChunkedLaunch", "attention chunked stats", uint64(statsCap*4), statsCap)
		if err != nil {
			return err
		}
		workspace.Stats = stats
		workspace.statsCap = statsCap
	}
	return nil
}

func hipAttentionHeadsChunkedWorkspaceCapacityCount(count int) int {
	if count <= 1 {
		return count
	}
	if count > 1<<30 {
		return count
	}
	return 1 << bits.Len(uint(count-1))
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureTokenIDBuffer(driver nativeHIPDriver) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if workspace.TokenID != nil && workspace.TokenID.Pointer() != 0 && workspace.TokenID.Count() == 1 && workspace.TokenID.SizeBytes() == 4 {
		return workspace.TokenID, nil
	}
	if err := workspace.TokenID.Close(); err != nil {
		return nil, err
	}
	workspace.TokenIDLoaded = false
	tokenID, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsChunkedLaunch", "single token id", 4, 1)
	if err != nil {
		return nil, err
	}
	workspace.TokenID = tokenID
	return tokenID, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureTokenIDValue(driver nativeHIPDriver, tokenID int32, vocabSize int) (*hipDeviceByteBuffer, error) {
	if tokenID < 0 || vocabSize <= 0 || int(tokenID) >= vocabSize {
		return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "token ID is outside vocabulary", nil)
	}
	tokenBuffer, err := workspace.EnsureTokenIDBuffer(driver)
	if err != nil {
		return nil, err
	}
	if workspace.TokenIDLoaded && workspace.TokenIDValue == tokenID {
		return tokenBuffer, nil
	}
	if err := hipWriteSingleTokenID(driver, tokenBuffer.Pointer(), tokenID); err != nil {
		workspace.TokenIDLoaded = false
		return nil, err
	}
	workspace.TokenIDLoaded = true
	workspace.TokenIDValue = tokenID
	return tokenBuffer, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureEmbeddingOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureMappedOutput(driver, &workspace.EmbeddingOutputs, count, "embedding lookup output")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureScaledEmbedding(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureFixedOutputReusableCapacity(driver, &workspace.ScaledEmbeddingFixed, &workspace.ScaledEmbeddingFixedCap, &workspace.ScaledEmbeddingView, count, "scaled embedding output", "scaled embedding output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsurePerLayerEmbedding(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureMappedOutput(driver, &workspace.PerLayerEmbeddings, count, "per-layer embedding output")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsurePerLayerProjected(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureFixedOutputReusableCapacity(driver, &workspace.PerLayerProjectedFixed, &workspace.PerLayerProjectedCap, &workspace.PerLayerProjectedView, count, "per-layer projected output", "per-layer projected output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsurePerLayerScaled(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureFixedOutputReusableCapacity(driver, &workspace.PerLayerScaledFixed, &workspace.PerLayerScaledFixedCap, &workspace.PerLayerScaledView, count, "per-layer scaled output", "per-layer scaled output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsurePerLayerProjectedScaled(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureMappedOutput(driver, &workspace.PerLayerProjScaled, count, "per-layer projected scaled output")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsurePerLayerNorm(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureMappedOutput(driver, &workspace.PerLayerNorm, count, "per-layer norm output")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsurePerLayerCombined(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureMappedOutput(driver, &workspace.PerLayerCombined, count, "per-layer combined output")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsurePerLayerOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureFixedOutputReusableCapacity(driver, &workspace.PerLayerOutputFixed, &workspace.PerLayerOutputFixedCap, &workspace.PerLayerOutputView, count, "per-layer final output", "per-layer final output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureAssistantDraftCombined(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureFixedOutputReusableCapacity(driver, &workspace.AssistantDraftCombinedFixed, &workspace.AssistantDraftCombinedCap, &workspace.AssistantDraftCombinedView, count, "assistant draft-step combined input", "assistant draft-step combined input view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureAssistantDraftInputHidden(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureFixedOutputReusableCapacity(driver, &workspace.AssistantDraftInputHiddenFixed, &workspace.AssistantDraftInputHiddenCap, &workspace.AssistantDraftInputHiddenView, count, "assistant draft-step pre-projection hidden", "assistant draft-step pre-projection hidden view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) BorrowPerLayerInputDeviceSet(driver nativeHIPDriver, layerCount, inputSize int, backing *hipDeviceByteBuffer) (*hipGemma4Q4PerLayerInputDeviceSet, error) {
	return workspace.BorrowPerLayerInputDeviceSetBatch(driver, layerCount, inputSize, backing, "per-layer input slice")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) BorrowPerLayerInputDeviceSetBatch(driver nativeHIPDriver, layerCount, layerValueCount int, backing *hipDeviceByteBuffer, viewLabel string) (*hipGemma4Q4PerLayerInputDeviceSet, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if layerCount <= 0 || layerValueCount <= 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "per-layer input dimensions must be positive", nil)
	}
	if backing == nil || backing.Pointer() == 0 || backing.Count() != layerCount*layerValueCount || backing.SizeBytes() != uint64(layerCount*layerValueCount*4) {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "per-layer input backing shape mismatch", nil)
	}
	if viewLabel == "" {
		viewLabel = "per-layer input slice"
	}
	workspace.PerLayerInputBacking[0] = backing
	workspace.PerLayerInputSet = hipGemma4Q4PerLayerInputDeviceSet{
		driver:           driver,
		layerCount:       layerCount,
		layerStrideBytes: uint64(layerValueCount * 4),
		layerValueCount:  layerValueCount,
		viewLabel:        viewLabel,
		borrowedBacking:  true,
		Backing:          workspace.PerLayerInputBacking[:],
	}
	return &workspace.PerLayerInputSet, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsurePrefillTokenBuffer(driver nativeHIPDriver, tokens []int32) (*hipDeviceTokenBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if len(tokens) == 0 {
		return nil, core.E("rocm.hip.Tokens", "token IDs are required", nil)
	}
	if err := workspace.PrefillTokenBuffer.Close(); err != nil {
		return nil, err
	}
	if len(tokens)*4 <= hipProjectionGreedyPrefillReserveBytes {
		buffer, err := workspace.ensurePrefillTokenBufferInGreedySlab(driver, tokens)
		if err != nil {
			return nil, err
		}
		return buffer, nil
	}
	buffer, err := hipUploadTokenIDs(driver, tokens)
	if err != nil {
		return nil, err
	}
	workspace.PrefillTokenBuffer = buffer
	return buffer, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ensurePrefillTokenBufferInGreedySlab(driver nativeHIPDriver, tokens []int32) (*hipDeviceTokenBuffer, error) {
	slab, err := workspace.ensureProjectionGreedyBestSlab(driver, hipProjectionGreedyReservedWorkspaceSlabIdx)
	if err != nil {
		return nil, err
	}
	payload, err := hipTokenIDsPayloadInto(workspace.PrefillTokenPayload, tokens)
	if err != nil {
		return nil, err
	}
	workspace.PrefillTokenPayload = payload
	pointer := slab.Pointer() + nativeDevicePointer(workspace.projectionGreedyPrefillReserveOffsetBytes())
	if err := hipCopyHostToDeviceLabeled(driver, pointer, payload, "rocm.hip.Tokens", "prefill token buffer"); err != nil {
		return nil, core.E("rocm.hip.Tokens", "copy prefill token buffer", err)
	}
	workspace.PrefillTokenView = hipDeviceTokenBuffer{
		driver:    driver,
		pointer:   pointer,
		count:     len(tokens),
		sizeBytes: uint64(len(payload)),
		borrowed:  true,
	}
	workspace.PrefillTokenBuffer = &workspace.PrefillTokenView
	return workspace.PrefillTokenBuffer, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureSuppressTokenBuffer(driver nativeHIPDriver, tokens []int32) (*hipDeviceTokenBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if len(tokens) == 0 {
		return nil, nil
	}
	if workspace.SuppressTokenBuffer != nil && workspace.SuppressTokenBuffer.Pointer() != 0 &&
		workspace.SuppressTokenBuffer.Count() == len(tokens) && hipInt32SlicesEqual(workspace.SuppressTokenIDs, tokens) {
		return workspace.SuppressTokenBuffer, nil
	}
	if err := workspace.SuppressTokenBuffer.Close(); err != nil {
		return nil, err
	}
	if len(tokens)*4 <= hipProjectionGreedySuppressReserveBytes {
		buffer, err := workspace.ensureSuppressTokenBufferInGreedySlab(driver, tokens)
		if err != nil {
			return nil, err
		}
		return buffer, nil
	}
	buffer, err := hipUploadTokenIDs(driver, tokens)
	if err != nil {
		return nil, err
	}
	workspace.SuppressTokenBuffer = buffer
	workspace.SuppressTokenIDs = append(workspace.SuppressTokenIDs[:0], tokens...)
	return buffer, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ensureSuppressTokenBufferInGreedySlab(driver nativeHIPDriver, tokens []int32) (*hipDeviceTokenBuffer, error) {
	slab, err := workspace.ensureProjectionGreedyBestSlab(driver, hipProjectionGreedyReservedWorkspaceSlabIdx)
	if err != nil {
		return nil, err
	}
	payload, err := workspace.suppressTokenPayload(tokens)
	if err != nil {
		return nil, err
	}
	pointer := slab.Pointer() + nativeDevicePointer(workspace.projectionGreedySuppressReserveOffsetBytes())
	if err := hipCopyHostToDeviceLabeled(driver, pointer, payload, "rocm.hip.Tokens", "suppress token buffer"); err != nil {
		return nil, core.E("rocm.hip.Tokens", "copy suppress token buffer", err)
	}
	workspace.SuppressTokenView = hipDeviceTokenBuffer{
		driver:    driver,
		pointer:   pointer,
		count:     len(tokens),
		sizeBytes: uint64(len(payload)),
		borrowed:  true,
	}
	workspace.SuppressTokenBuffer = &workspace.SuppressTokenView
	return workspace.SuppressTokenBuffer, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) suppressTokenPayload(tokens []int32) ([]byte, error) {
	if len(tokens) == 0 {
		return nil, core.E("rocm.hip.Tokens", "token IDs are required", nil)
	}
	byteCount := len(tokens) * 4
	if byteCount > len(workspace.SuppressTokenInlineData) {
		payload, err := hipTokenIDsPayloadInto(workspace.SuppressTokenPayload, tokens)
		if err != nil {
			return nil, err
		}
		workspace.SuppressTokenPayload = payload
		workspace.SuppressTokenIDs = append(workspace.SuppressTokenIDs[:0], tokens...)
		return payload, nil
	}
	payload := workspace.SuppressTokenInlineData[:byteCount]
	for index, id := range tokens {
		if id < 0 {
			return nil, core.E("rocm.hip.Tokens", "token IDs must be non-negative", nil)
		}
		binary.LittleEndian.PutUint32(payload[index*4:], uint32(id))
		workspace.SuppressTokenInlineIDs[index] = id
	}
	workspace.SuppressTokenPayload = payload
	workspace.SuppressTokenIDs = workspace.SuppressTokenInlineIDs[:len(tokens)]
	return payload, nil
}

func hipInt32SlicesEqual(left, right []int32) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureAttentionOutput(driver nativeHIPDriver, headCount, dim int) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if headCount <= 0 || dim <= 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention output dimensions must be positive", nil)
	}
	count := headCount * dim
	return workspace.ensureFixedOutputReusableCapacity(driver, &workspace.AttentionOutputFixed, &workspace.AttentionOutputFixedCap, &workspace.AttentionOutputView, count, "attention output", "attention output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureBatchAttentionOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureFixedOutputReusableCapacity(driver, &workspace.AttentionOutputFixed, &workspace.AttentionOutputFixedCap, &workspace.AttentionOutputView, count, "attention output", "attention output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureBatchAttentionWeights(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "attention workspace is required", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch weight count must be positive", nil)
	}
	if workspace.BatchAttentionWeight != nil && workspace.BatchAttentionWeight.Pointer() != 0 && workspace.batchWeightCap >= count {
		return workspace.BatchAttentionWeight, nil
	}
	if err := workspace.BatchAttentionWeight.Close(); err != nil {
		return nil, err
	}
	weights, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch head weights", uint64(count)*4, count)
	if err != nil {
		return nil, err
	}
	workspace.BatchAttentionWeight = weights
	workspace.batchWeightCap = count
	return weights, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ensureMappedOutput(driver nativeHIPDriver, outputs *map[int]*hipDeviceByteBuffer, count int, label string) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" count must be positive", nil)
	}
	if outputs == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" storage is required", nil)
	}
	if *outputs == nil {
		*outputs = make(map[int]*hipDeviceByteBuffer, 2)
	}
	if output := (*outputs)[count]; output != nil && output.Pointer() != 0 && output.Count() == count && output.SizeBytes() == uint64(count*4) {
		return output, nil
	}
	if err := (*outputs)[count].Close(); err != nil {
		return nil, err
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsChunkedLaunch", label, uint64(count*4), count)
	if err != nil {
		return nil, err
	}
	(*outputs)[count] = output
	return output, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ensureMappedOutputReusable(driver nativeHIPDriver, outputs *map[int]*hipDeviceByteBuffer, view *hipDeviceByteBuffer, count int, label, viewLabel string) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" count must be positive", nil)
	}
	if outputs == nil || view == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" storage is required", nil)
	}
	if *outputs == nil {
		*outputs = make(map[int]*hipDeviceByteBuffer, 2)
	}
	if output := (*outputs)[count]; output != nil && output.Pointer() != 0 && output.Count() == count && output.SizeBytes() == uint64(count*4) {
		return output, nil
	}
	var best *hipDeviceByteBuffer
	bestCount := 0
	for outputCount, output := range *outputs {
		if output == nil || output.Pointer() == 0 || outputCount < count || output.Count() < count || output.SizeBytes() < uint64(count*4) {
			continue
		}
		if best == nil || outputCount < bestCount {
			best = output
			bestCount = outputCount
		}
	}
	if best != nil {
		*view = hipDeviceByteBuffer{
			driver:    driver,
			pointer:   best.Pointer(),
			count:     count,
			sizeBytes: uint64(count * 4),
			borrowed:  true,
			label:     viewLabel,
		}
		return view, nil
	}
	if err := (*outputs)[count].Close(); err != nil {
		return nil, err
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsChunkedLaunch", label, uint64(count*4), count)
	if err != nil {
		return nil, err
	}
	(*outputs)[count] = output
	return output, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ensureMappedOutputReusableCapacity(driver nativeHIPDriver, outputs *map[int]*hipDeviceByteBuffer, view *hipDeviceByteBuffer, count int, label, viewLabel string) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" count must be positive", nil)
	}
	if outputs == nil || view == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" storage is required", nil)
	}
	if *outputs == nil {
		*outputs = make(map[int]*hipDeviceByteBuffer, 2)
	}
	var best *hipDeviceByteBuffer
	bestCount := 0
	for outputCount, output := range *outputs {
		if output == nil || output.Pointer() == 0 || outputCount < count || output.Count() < count || output.SizeBytes() < uint64(count*4) {
			continue
		}
		if best == nil || outputCount < bestCount {
			best = output
			bestCount = outputCount
		}
	}
	if best == nil {
		capCount := hipAttentionHeadsChunkedWorkspaceCapacityCount(count)
		if err := (*outputs)[capCount].Close(); err != nil {
			return nil, err
		}
		output, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsChunkedLaunch", label, uint64(capCount*4), capCount)
		if err != nil {
			return nil, err
		}
		(*outputs)[capCount] = output
		best = output
	}
	*view = hipDeviceByteBuffer{
		driver:    driver,
		pointer:   best.Pointer(),
		count:     count,
		sizeBytes: uint64(count * 4),
		borrowed:  true,
		label:     viewLabel,
	}
	return view, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ensureFixedOutputReusableCapacity(driver nativeHIPDriver, output *hipDeviceByteBuffer, capCount *int, view *hipDeviceByteBuffer, count int, label, viewLabel string) (*hipDeviceByteBuffer, error) {
	fixed, err := workspace.ensureFixedOutputCapacity(driver, output, capCount, count, label)
	if err != nil {
		return nil, err
	}
	if view == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" view storage is required", nil)
	}
	*view = hipDeviceByteBuffer{
		driver:    driver,
		pointer:   fixed.Pointer(),
		count:     count,
		sizeBytes: uint64(count * 4),
		borrowed:  true,
		label:     viewLabel,
	}
	return view, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ensureFixedOutputCapacity(driver nativeHIPDriver, output *hipDeviceByteBuffer, capCount *int, count int, label string) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" count must be positive", nil)
	}
	if output == nil || capCount == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" storage is required", nil)
	}
	if output.Pointer() == 0 || output.driver != driver || *capCount < count || output.Count() < count || output.SizeBytes() < uint64(count*4) {
		if err := output.Close(); err != nil {
			return nil, err
		}
		capacity := hipAttentionHeadsChunkedWorkspaceCapacityCount(count)
		allocated, err := hipAllocateByteBufferValue(driver, "rocm.hip.AttentionHeadsChunkedLaunch", label, uint64(capacity*4), capacity)
		if err != nil {
			return nil, err
		}
		*output = allocated
		*capCount = capacity
	}
	return output, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ensureFixedPairOutputReusableCapacity(driver nativeHIPDriver, output *hipDeviceByteBuffer, capCount *int, views *[2]hipDeviceByteBuffer, count, slot int, label, viewLabel string) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" count must be positive", nil)
	}
	if output == nil || capCount == nil || views == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" view storage is required", nil)
	}
	slot &= 1
	if output.Pointer() == 0 || output.driver != driver || *capCount < count || output.Count() < *capCount*2 || output.SizeBytes() < uint64(*capCount*2*4) {
		if err := output.Close(); err != nil {
			return nil, err
		}
		capacity := hipAttentionHeadsChunkedWorkspaceCapacityCount(count)
		allocated, err := hipAllocateByteBufferValue(driver, "rocm.hip.AttentionHeadsChunkedLaunch", label, uint64(capacity*2*4), capacity*2)
		if err != nil {
			return nil, err
		}
		*output = allocated
		*capCount = capacity
	}
	(*views)[slot] = hipDeviceByteBuffer{
		driver:    driver,
		pointer:   output.Pointer() + nativeDevicePointer(slot*(*capCount)*4),
		count:     count,
		sizeBytes: uint64(count * 4),
		borrowed:  true,
		label:     viewLabel,
	}
	return &(*views)[slot], nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureProjectionOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureFixedOutputReusableCapacity(driver, &workspace.ProjectionOutputFixed, &workspace.ProjectionOutputCap, &workspace.ProjectionOutputView, count, "attention projection output", "projection output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureKVProjectionOutput(driver nativeHIPDriver, count, slot int) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if slot < 0 || slot >= len(workspace.KVProjectionOutputViews) {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "KV projection output slot is out of range", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "KV projection output count must be positive", nil)
	}
	return workspace.ensureFixedPairOutputReusableCapacity(driver, &workspace.KVProjectionPairFixed, &workspace.KVProjectionPairCap, &workspace.KVProjectionOutputViews, count, slot, "KV projection output pair", "KV projection output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) kvProjectionPairView(driver nativeHIPDriver, output *hipDeviceByteBuffer, capCount, count, slot int) *hipDeviceByteBuffer {
	view := &workspace.KVProjectionOutputViews[slot]
	*view = hipDeviceByteBuffer{
		driver:    driver,
		pointer:   output.Pointer() + nativeDevicePointer(slot*capCount*4),
		count:     count,
		sizeBytes: uint64(count * 4),
		borrowed:  true,
		label:     "KV projection output view",
	}
	return view
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsurePrefillInputNormOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureFixedOutputReusableCapacity(driver, &workspace.PrefillInputNormFixed, &workspace.PrefillInputNormCap, &workspace.PrefillInputNormView, count, "prefill input norm output", "prefill input norm output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureProjectionScoreOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "attention workspace is required", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "projection score count must be positive", nil)
	}
	if workspace.ProjectionScore != nil && workspace.ProjectionScore.Pointer() != 0 && workspace.ProjectionScore.Count() == count && workspace.ProjectionScore.SizeBytes() == uint64(count*hipMLXQ4ProjectionBestBytes) {
		return workspace.ProjectionScore, nil
	}
	if err := workspace.ProjectionScore.Close(); err != nil {
		return nil, err
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4ProjectionScoresLaunch", "MLX q4 projection packed scores", uint64(count*hipMLXQ4ProjectionBestBytes), count)
	if err != nil {
		return nil, err
	}
	workspace.ProjectionScore = output
	return output, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ProjectionScorePayload(count int) ([]byte, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "attention workspace is required", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "projection score count must be positive", nil)
	}
	byteCount := count * hipMLXQ4ProjectionBestBytes
	if cap(workspace.ProjectionScoreBytes) < byteCount {
		workspace.ProjectionScoreBytes = make([]byte, byteCount)
	}
	return workspace.ProjectionScoreBytes[:byteCount], nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureProjectionTopKOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureProjectionTopKOutput(driver, count, false)
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureProjectionTopKWorkOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureProjectionTopKOutput(driver, count, true)
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ensureProjectionTopKOutput(driver nativeHIPDriver, count int, work bool) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.PackedTopKLaunch", "attention workspace is required", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.PackedTopKLaunch", "projection top-k count must be positive", nil)
	}
	buffer := workspace.ProjectionTopK
	capCount := workspace.ProjectionTopKCap
	label := "MLX q4 projection top-k partial scores"
	if work {
		buffer = workspace.ProjectionTopKWork
		capCount = workspace.ProjectionTopKWorkCap
		label = "MLX q4 projection top-k work scores"
	}
	byteCount := uint64(count * hipMLXQ4ProjectionBestBytes)
	if buffer != nil && buffer.Pointer() != 0 && capCount >= count && buffer.SizeBytes() >= byteCount {
		return workspace.projectionTopKView(driver, buffer.Pointer(), byteCount, count, label, work), nil
	}
	if err := buffer.Close(); err != nil {
		return nil, err
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.PackedTopKLaunch", label, byteCount, count)
	if err != nil {
		return nil, err
	}
	if work {
		workspace.ProjectionTopKWork = output
		workspace.ProjectionTopKWorkCap = count
	} else {
		workspace.ProjectionTopK = output
		workspace.ProjectionTopKCap = count
	}
	return workspace.projectionTopKView(driver, output.Pointer(), byteCount, count, label, work), nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) projectionTopKView(driver nativeHIPDriver, pointer nativeDevicePointer, sizeBytes uint64, count int, label string, work bool) *hipDeviceByteBuffer {
	view := &workspace.ProjectionTopKView
	if work {
		view = &workspace.ProjectionTopKWorkView
	}
	*view = hipDeviceByteBuffer{
		driver:    driver,
		pointer:   pointer,
		count:     count,
		sizeBytes: sizeBytes,
		borrowed:  true,
		label:     label,
	}
	return view
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ProjectionTopKPayload(count int) ([]byte, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.PackedTopKLaunch", "attention workspace is required", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.PackedTopKLaunch", "projection top-k count must be positive", nil)
	}
	byteCount := count * hipMLXQ4ProjectionBestBytes
	if cap(workspace.ProjectionTopKBytes) < byteCount {
		workspace.ProjectionTopKBytes = make([]byte, byteCount)
	}
	return workspace.ProjectionTopKBytes[:byteCount], nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureProjectionCandidateTokenOutput(driver nativeHIPDriver, count int) (*hipDeviceTokenBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "attention workspace is required", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "candidate token count must be positive", nil)
	}
	byteCount := uint64(count * 4)
	if workspace.ProjectionCandidateTokenOutput != nil &&
		workspace.ProjectionCandidateTokenOutput.Pointer() != 0 &&
		workspace.ProjectionCandidateTokenCap >= count &&
		workspace.ProjectionCandidateTokenOutput.SizeBytes() >= byteCount {
		return workspace.projectionCandidateTokenView(driver, workspace.ProjectionCandidateTokenOutput.Pointer(), byteCount, count), nil
	}
	if err := workspace.ProjectionCandidateTokenOutput.Close(); err != nil {
		return nil, err
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.OrderedEmbeddingCandidatesLaunch", "ordered embedding candidate tokens", byteCount, count)
	if err != nil {
		return nil, err
	}
	workspace.ProjectionCandidateTokenOutput = output
	workspace.ProjectionCandidateTokenCap = count
	return workspace.projectionCandidateTokenView(driver, output.Pointer(), byteCount, count), nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) projectionCandidateTokenView(driver nativeHIPDriver, pointer nativeDevicePointer, sizeBytes uint64, count int) *hipDeviceTokenBuffer {
	workspace.ProjectionCandidateTokenView = hipDeviceTokenBuffer{
		driver:    driver,
		pointer:   pointer,
		count:     count,
		sizeBytes: sizeBytes,
		borrowed:  true,
	}
	return &workspace.ProjectionCandidateTokenView
}

func (workspace *hipAttentionHeadsChunkedWorkspace) BorrowProjectionGreedyBest(driver nativeHIPDriver) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "attention workspace is required", nil)
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "HIP driver is not available", nil)
	}
	slot := workspace.ProjectionGreedyNext
	slabIndex, slotIndex := workspace.projectionGreedyBestWorkspaceSlot(slot)
	buffer, err := workspace.ensureProjectionGreedyBestSlab(driver, slabIndex)
	if err != nil {
		return nil, err
	}
	view := &workspace.ProjectionGreedyView
	*view = hipDeviceByteBuffer{
		driver:    driver,
		pointer:   buffer.Pointer() + nativeDevicePointer(slotIndex*hipMLXQ4ProjectionBestBytes),
		count:     1,
		sizeBytes: hipMLXQ4ProjectionBestBytes,
		borrowed:  true,
		label:     "MLX q4 projection greedy best slot",
	}
	workspace.ProjectionGreedyNext++
	return view, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) BorrowProjectionGreedyBestBatch(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "attention workspace is required", nil)
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "HIP driver is not available", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "greedy batch slot count must be positive", nil)
	}
	if count > hipProjectionGreedyBestWorkspaceSlots {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "greedy batch slot count exceeds workspace slab capacity", nil)
	}
	slot := workspace.ProjectionGreedyNext
	for {
		slabIndex, slotIndex := workspace.projectionGreedyBestWorkspaceSlot(slot)
		available := hipProjectionGreedyBestWorkspaceSlots - slotIndex
		if slabIndex == hipProjectionGreedyReservedWorkspaceSlabIdx {
			available = workspace.projectionGreedyFirstSlabUseSlots() - slotIndex
		}
		if available >= count {
			buffer, err := workspace.ensureProjectionGreedyBestSlab(driver, slabIndex)
			if err != nil {
				return nil, err
			}
			view := &workspace.ProjectionGreedyView
			*view = hipDeviceByteBuffer{
				driver:    driver,
				pointer:   buffer.Pointer() + nativeDevicePointer(slotIndex*hipMLXQ4ProjectionBestBytes),
				count:     count,
				sizeBytes: uint64(count * hipMLXQ4ProjectionBestBytes),
				borrowed:  true,
				label:     "MLX q4 projection greedy batch best slots",
			}
			workspace.ProjectionGreedyNext = slot + count
			return view, nil
		}
		slot += available
	}
}

func (workspace *hipAttentionHeadsChunkedWorkspace) projectionGreedyBestWorkspaceSlot(slot int) (int, int) {
	firstUseSlots := workspace.projectionGreedyFirstSlabUseSlots()
	if slot < firstUseSlots {
		return 0, slot
	}
	remaining := slot - firstUseSlots
	return 1 + remaining/hipProjectionGreedyBestWorkspaceSlots, remaining % hipProjectionGreedyBestWorkspaceSlots
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ensureProjectionGreedyBestSlab(driver nativeHIPDriver, slabIndex int) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "attention workspace is required", nil)
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "HIP driver is not available", nil)
	}
	if slabIndex < 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "greedy workspace slab index must be non-negative", nil)
	}
	for _, buffer := range workspace.ProjectionGreedyBest {
		if buffer == nil || buffer.driver == driver {
			continue
		}
		for _, output := range workspace.ProjectionGreedyBest {
			if err := output.Close(); err != nil {
				return nil, err
			}
		}
		workspace.ProjectionGreedyBest = workspace.ProjectionGreedyBest[:0]
		workspace.ProjectionGreedyView = hipDeviceByteBuffer{}
		workspace.ProjectionGreedyNext = 0
		break
	}
	for len(workspace.ProjectionGreedyBest) <= slabIndex {
		slots := hipProjectionGreedyBestWorkspaceSlots
		if len(workspace.ProjectionGreedyBest) == hipProjectionGreedyReservedWorkspaceSlabIdx {
			slots = workspace.projectionGreedyFirstSlabSlots()
		}
		sizeBytes := uint64(slots * hipMLXQ4ProjectionBestBytes)
		buffer, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 projection greedy best slots", sizeBytes, slots)
		if err != nil {
			return nil, err
		}
		if err := hipMemsetDevice(driver, buffer.Pointer(), 0, buffer.SizeBytes()); err != nil {
			_ = buffer.Close()
			return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "initialize greedy best slots", err)
		}
		workspace.ProjectionGreedyBest = append(workspace.ProjectionGreedyBest, buffer)
	}
	return workspace.ProjectionGreedyBest[slabIndex], nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureActivationOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureFixedOutputReusableCapacity(driver, &workspace.ActivationOutputFixed, &workspace.ActivationOutputCap, &workspace.ActivationOutputView, count, "activation output", "activation output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureGELUTanhMLPBarrier(driver nativeHIPDriver) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "attention workspace is required", nil)
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "HIP driver is not available", nil)
	}
	if workspace.GELUTanhMLPBarrier != nil &&
		workspace.GELUTanhMLPBarrier.Pointer() != 0 &&
		workspace.GELUTanhMLPBarrier.driver == driver &&
		workspace.GELUTanhMLPBarrier.SizeBytes() >= 8 {
		return workspace.GELUTanhMLPBarrier, nil
	}
	if err := workspace.GELUTanhMLPBarrier.Close(); err != nil {
		return nil, err
	}
	barrier, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "MLX q4 GELU tanh MLP persistent barrier", 8, 2)
	if err != nil {
		return nil, err
	}
	if err := hipMemsetDevice(driver, barrier.Pointer(), 0, barrier.SizeBytes()); err != nil {
		_ = barrier.Close()
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "initialize persistent barrier", err)
	}
	workspace.GELUTanhMLPBarrier = barrier
	return barrier, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureRMSResidualOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureRMSResidualNormOutput(driver, count, 0, "RMS residual/norm output pair", "RMS residual output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureRMSNormOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureRMSResidualNormOutput(driver, count, 1, "RMS residual/norm output pair", "RMS norm output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ensureRMSResidualNormOutput(driver nativeHIPDriver, count, slot int, label, viewLabel string) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", viewLabel+" count must be positive", nil)
	}
	if slot < 0 || slot >= len(workspace.RMSResidualNormViews) {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "RMS residual/norm output slot is out of range", nil)
	}
	view, err := workspace.ensureFixedPairOutputReusableCapacity(driver, &workspace.RMSResidualNormFixed, &workspace.RMSResidualNormCap, &workspace.RMSResidualNormViews, count, slot, label, viewLabel)
	if err != nil {
		return nil, err
	}
	if slot == 0 {
		workspace.RMSResidualOutputView = *view
	} else {
		workspace.RMSNormOutputView = *view
	}
	return view, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) rmsResidualNormView(driver nativeHIPDriver, output *hipDeviceByteBuffer, capCount, count, slot int, label string) *hipDeviceByteBuffer {
	view := &workspace.RMSResidualNormViews[slot]
	*view = hipDeviceByteBuffer{
		driver:    driver,
		pointer:   output.Pointer() + nativeDevicePointer(slot*capCount*4),
		count:     count,
		sizeBytes: uint64(count * 4),
		borrowed:  true,
		label:     label,
	}
	if slot == 0 {
		workspace.RMSResidualOutputView = *view
	} else {
		workspace.RMSNormOutputView = *view
	}
	return view
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureRMSRoPEOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureFixedOutputReusableCapacity(driver, &workspace.RMSRoPEFixed, &workspace.RMSRoPEFixedCap, &workspace.RMSRoPEOutputView, count, "RMS RoPE output", "RMS RoPE output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureKeyRMSRoPEOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureKeyValueNormOutput(driver, count, 0, "key/value norm output pair", "key RMS RoPE output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureRMSNoScaleOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureKeyValueNormOutput(driver, count, 1, "key/value norm output pair", "RMS no-scale output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ensureKeyValueNormOutput(driver nativeHIPDriver, count, slot int, label, viewLabel string) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", viewLabel+" count must be positive", nil)
	}
	if slot < 0 || slot >= len(workspace.KeyValueNormViews) {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "key/value norm output slot is out of range", nil)
	}
	view, err := workspace.ensureFixedPairOutputReusableCapacity(driver, &workspace.KeyValueNormFixed, &workspace.KeyValueNormCap, &workspace.KeyValueNormViews, count, slot, label, viewLabel)
	if err != nil {
		return nil, err
	}
	if slot == 0 {
		workspace.KeyRMSRoPEOutputView = *view
	} else {
		workspace.RMSNoScaleOutputView = *view
	}
	return view, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) keyValueNormView(driver nativeHIPDriver, output *hipDeviceByteBuffer, capCount, count, slot int, label string) *hipDeviceByteBuffer {
	view := &workspace.KeyValueNormViews[slot]
	*view = hipDeviceByteBuffer{
		driver:    driver,
		pointer:   output.Pointer() + nativeDevicePointer(slot*capCount*4),
		count:     count,
		sizeBytes: uint64(count * 4),
		borrowed:  true,
		label:     label,
	}
	if slot == 0 {
		workspace.KeyRMSRoPEOutputView = *view
	} else {
		workspace.RMSNoScaleOutputView = *view
	}
	return view
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureIntermediateOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureFixedOutputReusableCapacity(driver, &workspace.IntermediateFixed, &workspace.IntermediateFixedCap, &workspace.IntermediateOutputView, count, "intermediate output", "intermediate output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureQKVOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureFixedOutputReusableCapacity(driver, &workspace.QKVOutputFixed, &workspace.QKVOutputCap, &workspace.QKVOutputView, count, "QKV output", "QKV output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureFinalHiddenOutput(driver nativeHIPDriver, count, slot int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureFixedPairOutputReusableCapacity(driver, &workspace.FinalHiddenPairFixed, &workspace.FinalHiddenPairFixedCap, &workspace.FinalHiddenOutputViews, count, slot, "final hidden output pair", "final hidden output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureNextInputOutput(driver nativeHIPDriver, count, slot int) (*hipDeviceByteBuffer, error) {
	return workspace.ensureFixedPairOutputReusableCapacity(driver, &workspace.NextInputPairFixed, &workspace.NextInputPairFixedCap, &workspace.NextInputOutputViews, count, slot, "next layer input output pair", "next layer input output view")
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ensureSlottedOutput(driver nativeHIPDriver, outputs *[2]map[int]*hipDeviceByteBuffer, count, slot int, label string) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" count must be positive", nil)
	}
	if outputs == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" storage is required", nil)
	}
	slot &= 1
	if (*outputs)[slot] == nil {
		(*outputs)[slot] = make(map[int]*hipDeviceByteBuffer, 2)
	}
	if output := (*outputs)[slot][count]; output != nil && output.Pointer() != 0 && output.Count() == count && output.SizeBytes() == uint64(count*4) {
		return output, nil
	}
	if err := (*outputs)[slot][count].Close(); err != nil {
		return nil, err
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsChunkedLaunch", label, uint64(count*4), count)
	if err != nil {
		return nil, err
	}
	(*outputs)[slot][count] = output
	return output, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ensureSlottedOutputReusable(driver nativeHIPDriver, outputs *[2]map[int]*hipDeviceByteBuffer, views *[2]hipDeviceByteBuffer, count, slot int, label, viewLabel string) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" count must be positive", nil)
	}
	if outputs == nil || views == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" storage is required", nil)
	}
	slot &= 1
	if (*outputs)[slot] == nil {
		(*outputs)[slot] = make(map[int]*hipDeviceByteBuffer, 2)
	}
	if output := (*outputs)[slot][count]; output != nil && output.Pointer() != 0 && output.Count() == count && output.SizeBytes() == uint64(count*4) {
		return output, nil
	}
	var best *hipDeviceByteBuffer
	bestCount := 0
	for outputCount, output := range (*outputs)[slot] {
		if output == nil || output.Pointer() == 0 || outputCount < count || output.Count() < count || output.SizeBytes() < uint64(count*4) {
			continue
		}
		if best == nil || outputCount < bestCount {
			best = output
			bestCount = outputCount
		}
	}
	if best != nil {
		(*views)[slot] = hipDeviceByteBuffer{
			driver:    driver,
			pointer:   best.Pointer(),
			count:     count,
			sizeBytes: uint64(count * 4),
			borrowed:  true,
			label:     viewLabel,
		}
		return &(*views)[slot], nil
	}
	if err := (*outputs)[slot][count].Close(); err != nil {
		return nil, err
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsChunkedLaunch", label, uint64(count*4), count)
	if err != nil {
		return nil, err
	}
	(*outputs)[slot][count] = output
	return output, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) ensureSlottedPairOutputReusable(driver nativeHIPDriver, outputs *map[int]*hipDeviceByteBuffer, views *[2]hipDeviceByteBuffer, count, slot int, label, viewLabel string) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if count <= 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" count must be positive", nil)
	}
	if outputs == nil || views == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", label+" storage is required", nil)
	}
	slot &= 1
	if *outputs == nil {
		*outputs = make(map[int]*hipDeviceByteBuffer, 2)
	}
	var best *hipDeviceByteBuffer
	bestCount := 0
	for outputCount, output := range *outputs {
		if output == nil || output.Pointer() == 0 || outputCount < count || output.Count() < outputCount*2 || output.SizeBytes() < uint64(outputCount*2*4) {
			continue
		}
		if best == nil || outputCount < bestCount {
			best = output
			bestCount = outputCount
		}
	}
	if best == nil {
		if err := (*outputs)[count].Close(); err != nil {
			return nil, err
		}
		output, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsChunkedLaunch", label, uint64(count*2*4), count*2)
		if err != nil {
			return nil, err
		}
		(*outputs)[count] = output
		best = output
		bestCount = count
	}
	(*views)[slot] = hipDeviceByteBuffer{
		driver:    driver,
		pointer:   best.Pointer() + nativeDevicePointer(slot*bestCount*4),
		count:     count,
		sizeBytes: uint64(count * 4),
		borrowed:  true,
		label:     viewLabel,
	}
	return &(*views)[slot], nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) Close() error {
	if workspace == nil {
		return nil
	}
	var lastErr error
	if err := workspace.Partial.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.Stats.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.TokenID.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.PrefillTokenBuffer.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.SuppressTokenBuffer.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.BatchAttentionWeight.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.MoE.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.ProjectionScore.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.ProjectionTopK.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.ProjectionTopKWork.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.ProjectionCandidateTokenOutput.Close(); err != nil {
		lastErr = err
	}
	for _, output := range workspace.ProjectionGreedyBest {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	for _, output := range workspace.EmbeddingOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	for _, output := range workspace.ScaledEmbeddings {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	if err := workspace.ScaledEmbeddingFixed.Close(); err != nil {
		lastErr = err
	}
	for _, output := range workspace.PerLayerEmbeddings {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	for _, output := range workspace.PerLayerProjected {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	if err := workspace.PerLayerProjectedFixed.Close(); err != nil {
		lastErr = err
	}
	for _, output := range workspace.PerLayerScaled {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	if err := workspace.PerLayerScaledFixed.Close(); err != nil {
		lastErr = err
	}
	for _, output := range workspace.PerLayerProjScaled {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	for _, output := range workspace.PerLayerNorm {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	for _, output := range workspace.PerLayerCombined {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	for _, output := range workspace.PerLayerOutput {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	if err := workspace.PerLayerOutputFixed.Close(); err != nil {
		lastErr = err
	}
	for _, output := range workspace.AttentionOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	if err := workspace.AttentionOutputFixed.Close(); err != nil {
		lastErr = err
	}
	for _, output := range workspace.ProjectionOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	if err := workspace.ProjectionOutputFixed.Close(); err != nil {
		lastErr = err
	}
	for slot := range workspace.KVProjectionOutputs {
		for _, output := range workspace.KVProjectionOutputs[slot] {
			if err := output.Close(); err != nil {
				lastErr = err
			}
		}
	}
	for _, output := range workspace.KVProjectionPairOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	if err := workspace.KVProjectionPairFixed.Close(); err != nil {
		lastErr = err
	}
	for _, output := range workspace.PrefillInputNormOutput {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	if err := workspace.PrefillInputNormFixed.Close(); err != nil {
		lastErr = err
	}
	for _, output := range workspace.ActivationOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	if err := workspace.ActivationOutputFixed.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.GELUTanhMLPBarrier.Close(); err != nil {
		lastErr = err
	}
	for _, output := range workspace.RMSResidualOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	for _, output := range workspace.RMSNormOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	for _, output := range workspace.RMSResidualNormOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	if err := workspace.RMSResidualNormFixed.Close(); err != nil {
		lastErr = err
	}
	for _, output := range workspace.RMSRoPEOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	if err := workspace.RMSRoPEFixed.Close(); err != nil {
		lastErr = err
	}
	for _, output := range workspace.KeyRMSRoPEOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	for _, output := range workspace.RMSNoScaleOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	for _, output := range workspace.KeyValueNormOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	if err := workspace.KeyValueNormFixed.Close(); err != nil {
		lastErr = err
	}
	for _, output := range workspace.IntermediateOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	if err := workspace.IntermediateFixed.Close(); err != nil {
		lastErr = err
	}
	for _, output := range workspace.QKVOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	if err := workspace.QKVOutputFixed.Close(); err != nil {
		lastErr = err
	}
	for slot := range workspace.FinalHiddenOutputs {
		for _, output := range workspace.FinalHiddenOutputs[slot] {
			if err := output.Close(); err != nil {
				lastErr = err
			}
		}
	}
	for _, output := range workspace.FinalHiddenPairOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	if err := workspace.FinalHiddenPairFixed.Close(); err != nil {
		lastErr = err
	}
	for slot := range workspace.NextInputOutputs {
		for _, output := range workspace.NextInputOutputs[slot] {
			if err := output.Close(); err != nil {
				lastErr = err
			}
		}
	}
	for _, output := range workspace.NextInputPairOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	if err := workspace.NextInputPairFixed.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.AssistantDraftCombinedFixed.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.AssistantDraftInputHiddenFixed.Close(); err != nil {
		lastErr = err
	}
	clear(workspace.AttentionOutputs)
	workspace.AttentionOutputView = hipDeviceByteBuffer{}
	clear(workspace.EmbeddingOutputs)
	clear(workspace.ScaledEmbeddings)
	workspace.ScaledEmbeddingFixed = hipDeviceByteBuffer{}
	workspace.ScaledEmbeddingFixedCap = 0
	workspace.ScaledEmbeddingView = hipDeviceByteBuffer{}
	clear(workspace.PerLayerEmbeddings)
	clear(workspace.PerLayerProjected)
	workspace.PerLayerProjectedFixed = hipDeviceByteBuffer{}
	workspace.PerLayerProjectedCap = 0
	workspace.PerLayerProjectedView = hipDeviceByteBuffer{}
	clear(workspace.PerLayerScaled)
	workspace.PerLayerScaledFixed = hipDeviceByteBuffer{}
	workspace.PerLayerScaledFixedCap = 0
	workspace.PerLayerScaledView = hipDeviceByteBuffer{}
	clear(workspace.PerLayerProjScaled)
	clear(workspace.PerLayerNorm)
	clear(workspace.PerLayerCombined)
	clear(workspace.PerLayerOutput)
	workspace.PerLayerOutputFixed = hipDeviceByteBuffer{}
	workspace.PerLayerOutputFixedCap = 0
	workspace.PerLayerOutputView = hipDeviceByteBuffer{}
	workspace.AttentionOutputFixed = hipDeviceByteBuffer{}
	workspace.AttentionOutputFixedCap = 0
	workspace.AttentionOutputView = hipDeviceByteBuffer{}
	clear(workspace.ProjectionOutputs)
	workspace.ProjectionOutputFixed = hipDeviceByteBuffer{}
	workspace.ProjectionOutputCap = 0
	workspace.ProjectionOutputView = hipDeviceByteBuffer{}
	for slot := range workspace.KVProjectionOutputs {
		clear(workspace.KVProjectionOutputs[slot])
	}
	clear(workspace.KVProjectionPairOutputs)
	workspace.KVProjectionPairFixed = hipDeviceByteBuffer{}
	workspace.KVProjectionPairCap = 0
	workspace.KVProjectionOutputViews = [2]hipDeviceByteBuffer{}
	clear(workspace.PrefillInputNormOutput)
	workspace.PrefillInputNormFixed = hipDeviceByteBuffer{}
	workspace.PrefillInputNormCap = 0
	workspace.PrefillInputNormView = hipDeviceByteBuffer{}
	clear(workspace.ActivationOutputs)
	workspace.ActivationOutputFixed = hipDeviceByteBuffer{}
	workspace.ActivationOutputCap = 0
	workspace.ActivationOutputView = hipDeviceByteBuffer{}
	clear(workspace.RMSResidualOutputs)
	workspace.RMSResidualOutputView = hipDeviceByteBuffer{}
	clear(workspace.RMSNormOutputs)
	workspace.RMSNormOutputView = hipDeviceByteBuffer{}
	clear(workspace.RMSResidualNormOutputs)
	workspace.RMSResidualNormFixed = hipDeviceByteBuffer{}
	workspace.RMSResidualNormCap = 0
	workspace.RMSResidualNormViews = [2]hipDeviceByteBuffer{}
	clear(workspace.RMSRoPEOutputs)
	workspace.RMSRoPEFixed = hipDeviceByteBuffer{}
	workspace.RMSRoPEFixedCap = 0
	workspace.RMSRoPEOutputView = hipDeviceByteBuffer{}
	clear(workspace.KeyRMSRoPEOutputs)
	workspace.KeyRMSRoPEOutputView = hipDeviceByteBuffer{}
	clear(workspace.RMSNoScaleOutputs)
	workspace.RMSNoScaleOutputView = hipDeviceByteBuffer{}
	clear(workspace.KeyValueNormOutputs)
	workspace.KeyValueNormFixed = hipDeviceByteBuffer{}
	workspace.KeyValueNormCap = 0
	workspace.KeyValueNormViews = [2]hipDeviceByteBuffer{}
	clear(workspace.IntermediateOutputs)
	workspace.IntermediateFixed = hipDeviceByteBuffer{}
	workspace.IntermediateFixedCap = 0
	workspace.IntermediateOutputView = hipDeviceByteBuffer{}
	clear(workspace.QKVOutputs)
	workspace.QKVOutputFixed = hipDeviceByteBuffer{}
	workspace.QKVOutputCap = 0
	workspace.QKVOutputView = hipDeviceByteBuffer{}
	for slot := range workspace.FinalHiddenOutputs {
		clear(workspace.FinalHiddenOutputs[slot])
	}
	clear(workspace.FinalHiddenPairOutputs)
	workspace.FinalHiddenPairFixed = hipDeviceByteBuffer{}
	workspace.FinalHiddenPairFixedCap = 0
	workspace.FinalHiddenOutputViews = [2]hipDeviceByteBuffer{}
	for slot := range workspace.NextInputOutputs {
		clear(workspace.NextInputOutputs[slot])
	}
	clear(workspace.NextInputPairOutputs)
	workspace.NextInputPairFixed = hipDeviceByteBuffer{}
	workspace.NextInputPairFixedCap = 0
	workspace.NextInputOutputViews = [2]hipDeviceByteBuffer{}
	workspace.PerLayerInputSet = hipGemma4Q4PerLayerInputDeviceSet{}
	workspace.PerLayerInputBacking[0] = nil
	workspace.AssistantDraftCombinedFixed = hipDeviceByteBuffer{}
	workspace.AssistantDraftCombinedCap = 0
	workspace.AssistantDraftCombinedView = hipDeviceByteBuffer{}
	workspace.AssistantDraftInputHiddenFixed = hipDeviceByteBuffer{}
	workspace.AssistantDraftInputHiddenCap = 0
	workspace.AssistantDraftInputHiddenView = hipDeviceByteBuffer{}
	workspace.TokenID = nil
	workspace.TokenIDLoaded = false
	workspace.TokenIDValue = 0
	workspace.ScaledEmbeddingView = hipDeviceByteBuffer{}
	workspace.PrefillTokenBuffer = nil
	workspace.PrefillTokenView = hipDeviceTokenBuffer{}
	workspace.PrefillTokenPayload = nil
	workspace.SuppressTokenBuffer = nil
	workspace.SuppressTokenView = hipDeviceTokenBuffer{}
	workspace.SuppressTokenIDs = nil
	workspace.SuppressTokenPayload = nil
	workspace.BatchAttentionWeight = nil
	workspace.ProjectionScore = nil
	workspace.ProjectionScoreBytes = nil
	workspace.ProjectionTopK = nil
	workspace.ProjectionTopKCap = 0
	workspace.ProjectionTopKView = hipDeviceByteBuffer{}
	workspace.ProjectionTopKWork = nil
	workspace.ProjectionTopKWorkCap = 0
	workspace.ProjectionTopKWorkView = hipDeviceByteBuffer{}
	workspace.ProjectionTopKBytes = nil
	workspace.ProjectionTopPacked = nil
	workspace.ProjectionCandidates = nil
	workspace.ProjectionCandidateTokens = nil
	workspace.ProjectionCandidateTokenOutput = nil
	workspace.ProjectionCandidateTokenCap = 0
	workspace.ProjectionCandidateTokenView = hipDeviceTokenBuffer{}
	workspace.ProjectionGreedyBest = workspace.ProjectionGreedyBest[:0]
	workspace.ProjectionGreedyView = hipDeviceByteBuffer{}
	workspace.ProjectionGreedyNext = 0
	workspace.GreedyFirstSlabSlots = 0
	workspace.SampleCandidates = nil
	workspace.SampleWeights = nil
	workspace.batchWeightCap = 0
	return lastErr
}

func hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernelWithWorkspace(ctx context.Context, driver nativeHIPDriver, req hipAttentionRequest, query *hipDeviceByteBuffer, headCount int, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if workspace == nil {
		return hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernel(ctx, driver, req, query, headCount, output)
	}
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	dim, tokenCount, err := req.shape()
	if err != nil {
		return err
	}
	if !hipAttentionHeadsChunkedEligible(req, headCount, dim, tokenCount) {
		return hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernel(ctx, driver, req, query, headCount, output)
	}
	if headCount <= 0 {
		return core.E("rocm.hip.AttentionHeadsChunkedLaunch", "head count must be positive", nil)
	}
	if query == nil || query.Pointer() == 0 || query.Count() != headCount*dim || query.SizeBytes() != uint64(headCount*dim*4) {
		return core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention query device buffer shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != headCount*dim || output.SizeBytes() != uint64(headCount*dim*4) {
		return core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention output device buffer shape mismatch", nil)
	}
	return hipRunAttentionHeadsChunked(ctx, driver, req, query, headCount, dim, tokenCount, output, workspace)
}

func hipAttentionHeadsChunkedEligible(req hipAttentionRequest, headCount, dim, tokenCount int) bool {
	keyHeads := req.keyHeadsOrDefault()
	if headCount <= 0 || keyHeads <= 0 || keyHeads > headCount || headCount%keyHeads != 0 {
		return false
	}
	if keyHeads > 1 && dim&1 != 0 {
		return false
	}
	if uint64(dim)*uint64(keyHeads) > uint64(^uint32(0)) {
		return false
	}
	if dim <= 0 || dim > hipAttentionHeadsChunkedBlockSize || tokenCount < hipAttentionHeadsChunkSize {
		return false
	}
	if req.WindowSize > 0 && tokenCount <= hipAttentionHeadsSharedMaxTokens {
		return false
	}
	if req.DeviceKV == nil || req.DescriptorTable == nil {
		return false
	}
	if req.DeviceKV.mode != rocmKVCacheModeKQ8VQ4 {
		return false
	}
	return req.DeviceKV.TokenCount() == tokenCount && req.DeviceKV.PageCount() > 0
}

func hipRunAttentionHeadsChunked(ctx context.Context, driver nativeHIPDriver, req hipAttentionRequest, query *hipDeviceByteBuffer, headCount, dim, tokenCount int, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	chunkSize := hipAttentionHeadsChunkSize
	chunkCount := (tokenCount + chunkSize - 1) / chunkSize
	if err := workspace.Ensure(driver, headCount, dim, tokenCount, chunkSize); err != nil {
		return err
	}
	launch := hipAttentionHeadsChunkedLaunchArgs{
		QueryPointer:      query.Pointer(),
		DescriptorPointer: req.DescriptorTable.Pointer(),
		PartialPointer:    workspace.Partial.Pointer(),
		StatsPointer:      workspace.Stats.Pointer(),
		OutputPointer:     output.Pointer(),
		Dim:               dim,
		TokenCount:        tokenCount,
		HeadCount:         headCount,
		KeyHeads:          req.keyHeadsOrDefault(),
		ChunkSize:         chunkSize,
		ChunkCount:        chunkCount,
		QueryBytes:        query.SizeBytes(),
		DescriptorBytes:   req.DescriptorTable.SizeBytes(),
		PartialBytes:      uint64(headCount * chunkCount * dim * 4),
		StatsBytes:        uint64(headCount * chunkCount * 2 * 4),
		OutputBytes:       output.SizeBytes(),
		Scale:             req.Scale,
		WindowSize:        req.WindowSize,
	}
	launchBytes, err := launch.BinaryInto(workspace.ChunkedStage1Args[:])
	if err != nil {
		return err
	}
	stage2LaunchBytes := workspace.ChunkedStage2Args[:len(launchBytes)]
	copy(stage2LaunchBytes, launchBytes)
	sharedMemBytes, err := hipAttentionHeadsChunkedSharedMemBytes(chunkSize, dim)
	if err != nil {
		return err
	}
	gridX, err := rocmDeviceKVPositiveUint32("attention chunked stage1 blocks", headCount*chunkCount)
	if err != nil {
		return err
	}
	stage1 := hipKernelLaunchConfig{
		Name:           hipKernelNameAttentionHeadsChunkedStage1,
		Args:           launchBytes,
		GridX:          gridX,
		GridY:          1,
		GridZ:          1,
		BlockX:         hipAttentionHeadsChunkedBlockSize,
		BlockY:         1,
		BlockZ:         1,
		SharedMemBytes: sharedMemBytes,
	}
	if err := stage1.Validate(); err != nil {
		return err
	}
	if err := hipLaunchKernelContext(ctx, driver, stage1); err != nil {
		return err
	}
	stage2 := hipKernelLaunchConfig{
		Name:           hipKernelNameAttentionHeadsChunkedStage2,
		Args:           stage2LaunchBytes,
		GridX:          uint32(headCount),
		GridY:          1,
		GridZ:          1,
		BlockX:         hipAttentionHeadsChunkedBlockSize,
		BlockY:         1,
		BlockZ:         1,
		SharedMemBytes: 0,
	}
	if err := stage2.Validate(); err != nil {
		return err
	}
	return hipLaunchKernelContext(ctx, driver, stage2)
}

func hipAttentionHeadsChunkedSharedMemBytes(chunkSize, dim int) (uint32, error) {
	chunk, err := rocmDeviceKVPositiveUint32("attention chunked chunk size", chunkSize)
	if err != nil {
		return 0, err
	}
	width, err := rocmDeviceKVPositiveUint32("attention chunked query dim", dim)
	if err != nil {
		return 0, err
	}
	bytes := uint64(chunk) * 4
	bytes = hipAttentionHeadsAlignSharedBytes(bytes, 8)
	bytes += uint64(chunk) * 8
	bytes = hipAttentionHeadsAlignSharedBytes(bytes, 4)
	bytes += uint64(chunk) * 4
	bytes = hipAttentionHeadsAlignSharedBytes(bytes, 4)
	bytes += uint64(width) * 4
	if bytes > math.MaxUint32 {
		return 0, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention chunked shared memory byte count is out of uint32 range", nil)
	}
	return uint32(bytes), nil
}

func hipAttentionHeadsSharedMemBytes(tokenCount int, deviceKV bool) (uint32, error) {
	tokens, err := rocmDeviceKVPositiveUint32("attention token count", tokenCount)
	if err != nil {
		return 0, err
	}
	bytes := uint64(tokens) * 4
	if deviceKV && tokenCount >= 16 {
		bytes = hipAttentionHeadsAlignSharedBytes(bytes, 8)
		bytes += uint64(tokens) * 8
		bytes = hipAttentionHeadsAlignSharedBytes(bytes, 4)
		bytes += uint64(tokens) * 4
	}
	if bytes > math.MaxUint32 {
		return 0, core.E("rocm.hip.AttentionHeadsLaunch", "attention shared memory byte count is out of uint32 range", nil)
	}
	return uint32(bytes), nil
}

func hipAttentionHeadsQueryRMSRoPESharedMemBytes(attentionBytes uint32, dim int) (uint32, error) {
	dimension, err := rocmDeviceKVPositiveUint32("attention query dimension", dim)
	if err != nil {
		return 0, err
	}
	bytes := hipAttentionHeadsAlignSharedBytes(uint64(attentionBytes), 4)
	bytes += uint64(dimension) * 4
	if bytes > math.MaxUint32 {
		return 0, core.E("rocm.hip.AttentionHeadsBatchCausalQueryRMSRoPELaunch", "attention fused shared memory byte count is out of uint32 range", nil)
	}
	return uint32(bytes), nil
}

func hipAttentionHeadsAlignSharedBytes(value, alignment uint64) uint64 {
	if alignment <= 1 {
		return value
	}
	remainder := value % alignment
	if remainder == 0 {
		return value
	}
	return value + alignment - remainder
}

func hipAttentionHeadsBlockSize(tokenCount int) uint32 {
	if tokenCount >= 16 {
		return 512
	}
	return 256
}

func hipRunVectorAddKernel(ctx context.Context, driver nativeHIPDriver, req hipVectorAddRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameVectorAdd, launchBytes, buffers.Count)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func hipRunVectorAddDeviceKernel(ctx context.Context, driver nativeHIPDriver, left, right *hipDeviceByteBuffer) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if left == nil || right == nil || left.Pointer() == 0 || right.Pointer() == 0 {
		return nil, core.E("rocm.hip.VectorAddLaunch", "vector add device inputs are required", nil)
	}
	if left.Count() <= 0 || right.Count() != left.Count() ||
		left.SizeBytes() != uint64(left.Count()*4) ||
		right.SizeBytes() != uint64(right.Count()*4) {
		return nil, core.E("rocm.hip.VectorAddLaunch", "vector add device input shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.VectorAddLaunch", "vector add output", left.SizeBytes(), left.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunVectorAddDeviceKernelOutput(ctx, driver, left, right, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunVectorAddDeviceKernelOutput(ctx context.Context, driver nativeHIPDriver, left, right, output *hipDeviceByteBuffer) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if left == nil || right == nil || left.Pointer() == 0 || right.Pointer() == 0 {
		return core.E("rocm.hip.VectorAddLaunch", "vector add device inputs are required", nil)
	}
	if left.Count() <= 0 || right.Count() != left.Count() ||
		left.SizeBytes() != uint64(left.Count()*4) ||
		right.SizeBytes() != uint64(right.Count()*4) {
		return core.E("rocm.hip.VectorAddLaunch", "vector add device input shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != left.Count() || output.SizeBytes() != left.SizeBytes() {
		return core.E("rocm.hip.VectorAddLaunch", "vector add output shape mismatch", nil)
	}
	launchBytes, err := (hipVectorAddLaunchArgs{
		LeftPointer:   left.Pointer(),
		RightPointer:  right.Pointer(),
		OutputPointer: output.Pointer(),
		Count:         left.Count(),
		LeftBytes:     left.SizeBytes(),
		RightBytes:    right.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
	}).Binary()
	if err != nil {
		return err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameVectorAdd, launchBytes, left.Count())
	if err != nil {
		return err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return err
	}
	return nil
}

func hipRunVectorAddScaledDeviceKernel(ctx context.Context, driver nativeHIPDriver, left, right *hipDeviceByteBuffer, scale float32) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if left == nil || right == nil || left.Pointer() == 0 || right.Pointer() == 0 {
		return nil, core.E("rocm.hip.VectorAddScaledLaunch", "vector add-scaled device inputs are required", nil)
	}
	if left.Count() <= 0 || right.Count() != left.Count() ||
		left.SizeBytes() != uint64(left.Count()*4) ||
		right.SizeBytes() != uint64(right.Count()*4) {
		return nil, core.E("rocm.hip.VectorAddScaledLaunch", "vector add-scaled device input shape mismatch", nil)
	}
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return nil, core.E("rocm.hip.VectorAddScaledLaunch", "scale must be finite", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.VectorAddScaledLaunch", "vector add-scaled output", left.SizeBytes(), left.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunVectorAddScaledDeviceKernelOutput(ctx, driver, left, right, scale, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunVectorAddScaledDeviceKernelOutput(ctx context.Context, driver nativeHIPDriver, left, right *hipDeviceByteBuffer, scale float32, output *hipDeviceByteBuffer) error {
	return hipRunVectorAddScaledDeviceKernelOutputWithWorkspace(ctx, driver, left, right, scale, output, nil)
}

func hipRunVectorAddScaledDeviceKernelOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, left, right *hipDeviceByteBuffer, scale float32, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if left == nil || right == nil || left.Pointer() == 0 || right.Pointer() == 0 {
		return core.E("rocm.hip.VectorAddScaledLaunch", "vector add-scaled device inputs are required", nil)
	}
	if left.Count() <= 0 || right.Count() != left.Count() ||
		left.SizeBytes() != uint64(left.Count()*4) ||
		right.SizeBytes() != uint64(right.Count()*4) {
		return core.E("rocm.hip.VectorAddScaledLaunch", "vector add-scaled device input shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != left.Count() || output.SizeBytes() != left.SizeBytes() {
		return core.E("rocm.hip.VectorAddScaledLaunch", "vector add-scaled output shape mismatch", nil)
	}
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return core.E("rocm.hip.VectorAddScaledLaunch", "scale must be finite", nil)
	}
	launchArgs := hipVectorAddScaledLaunchArgs{
		LeftPointer:   left.Pointer(),
		RightPointer:  right.Pointer(),
		OutputPointer: output.Pointer(),
		Count:         left.Count(),
		LeftBytes:     left.SizeBytes(),
		RightBytes:    right.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
		Scale:         scale,
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.VectorAddScaledArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameVectorAddScaled, launchBytes, left.Count())
	if err != nil {
		return err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return err
	}
	return nil
}

func hipRunVectorScaleKernel(ctx context.Context, driver nativeHIPDriver, req hipVectorScaleRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameVectorScale, launchBytes, buffers.Count)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func hipRunVectorScaleDeviceKernel(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, scale float32) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.VectorScaleLaunch", "vector scale device input is required", nil)
	}
	if input.Count() <= 0 || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E("rocm.hip.VectorScaleLaunch", "vector scale device input shape mismatch", nil)
	}
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return nil, core.E("rocm.hip.VectorScaleLaunch", "scale must be finite", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.VectorScaleLaunch", "vector scale output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunVectorScaleDeviceKernelOutput(ctx, driver, input, scale, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunVectorScaleDeviceKernelOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, scale float32, output *hipDeviceByteBuffer) error {
	return hipRunVectorScaleDeviceKernelOutputWithWorkspace(ctx, driver, input, scale, output, nil)
}

func hipRunVectorScaleDeviceKernelOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, scale float32, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.VectorScaleLaunch", "vector scale device input is required", nil)
	}
	if input.Count() <= 0 || input.SizeBytes() != uint64(input.Count()*4) {
		return core.E("rocm.hip.VectorScaleLaunch", "vector scale device input shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != input.Count() || output.SizeBytes() != input.SizeBytes() {
		return core.E("rocm.hip.VectorScaleLaunch", "vector scale output shape mismatch", nil)
	}
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return core.E("rocm.hip.VectorScaleLaunch", "scale must be finite", nil)
	}
	launchArgs := hipVectorScaleLaunchArgs{
		InputPointer:  input.Pointer(),
		OutputPointer: output.Pointer(),
		Count:         input.Count(),
		InputBytes:    input.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
		Scale:         scale,
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.VectorScaleArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameVectorScale, launchBytes, input.Count())
	if err != nil {
		return err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return err
	}
	return nil
}

func hipRunSwiGLUKernel(ctx context.Context, driver nativeHIPDriver, req hipSwiGLURequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameSwiGLU, launchBytes, buffers.Count)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func hipRunGELUTanhMultiplyKernel(ctx context.Context, driver nativeHIPDriver, req hipGELUTanhMultiplyRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	if err := hipLaunchGELUTanhMultiplyDeviceBuffers(driver, buffers); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func hipRunGELUTanhMultiplyDeviceKernel(ctx context.Context, driver nativeHIPDriver, gate, up *hipDeviceByteBuffer) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if gate == nil || up == nil || gate.Pointer() == 0 || up.Pointer() == 0 {
		return nil, core.E("rocm.hip.GELUTanhMultiplyLaunch", "gate and up device buffers are required", nil)
	}
	if gate.Count() <= 0 || up.Count() != gate.Count() ||
		gate.SizeBytes() != uint64(gate.Count()*4) ||
		up.SizeBytes() != uint64(up.Count()*4) {
		return nil, core.E("rocm.hip.GELUTanhMultiplyLaunch", "gate and up device buffer shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.GELUTanhMultiplyLaunch", "GELU tanh multiply output", gate.SizeBytes(), gate.Count())
	if err != nil {
		return nil, err
	}
	buffers := &hipGELUTanhMultiplyDeviceBuffers{Gate: gate, Up: up, Output: output, Count: gate.Count()}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipLaunchGELUTanhMultiplyDeviceBuffers(driver, buffers); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipLaunchGELUTanhMultiplyDeviceBuffers(driver nativeHIPDriver, buffers *hipGELUTanhMultiplyDeviceBuffers) error {
	launch, err := hipGELUTanhMultiplyLaunchArgsForDeviceBuffers(buffers)
	if err != nil {
		return err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameGELUTanhMul, launchBytes, buffers.Count)
	if err != nil {
		return err
	}
	return hipLaunchKernel(driver, config)
}

func hipRunGreedyKernel(ctx context.Context, driver nativeHIPDriver, req hipGreedySampleRequest) (hipGreedySampleResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipGreedySampleResult{}, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameGreedy, launchBytes, buffers.Count)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return hipGreedySampleResult{}, err
	}
	return buffers.ReadOutput()
}

func hipRunGreedyKernelWithDeviceLogits(ctx context.Context, driver nativeHIPDriver, logits *hipDeviceByteBuffer) (hipGreedySampleResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipGreedySampleResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipGreedySampleResult{}, core.E("rocm.hip.GreedyLaunch", "HIP driver is not available", nil)
	}
	if logits == nil || logits.Pointer() == 0 {
		return hipGreedySampleResult{}, core.E("rocm.hip.GreedyLaunch", "logits device buffer is required", nil)
	}
	if logits.Count() <= 0 || logits.SizeBytes() != uint64(logits.Count()*4) {
		return hipGreedySampleResult{}, core.E("rocm.hip.GreedyLaunch", "logits device buffer shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.GreedyLaunch", "greedy output", hipGreedyResultBytes, 1)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	defer output.Close()
	launchBytes, err := (hipGreedySampleLaunchArgs{
		LogitsPointer: logits.Pointer(),
		OutputPointer: output.Pointer(),
		Count:         logits.Count(),
		LogitsBytes:   logits.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
	}).Binary()
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameGreedy, launchBytes, logits.Count())
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return hipGreedySampleResult{}, err
	}
	return hipReadGreedyResult(output, "rocm.hip.GreedyLaunch", "greedy output", logits.Count())
}

func hipRunSoftcapGreedyKernelWithDeviceLogits(ctx context.Context, driver nativeHIPDriver, logits *hipDeviceByteBuffer, softcap float32) (hipGreedySampleResult, error) {
	return hipRunSoftcapGreedyKernelWithDeviceLogitsSuppressBuffer(ctx, driver, logits, softcap, nil)
}

func hipRunSoftcapGreedyKernelWithDeviceLogitsSuppressBuffer(ctx context.Context, driver nativeHIPDriver, logits *hipDeviceByteBuffer, softcap float32, suppress *hipDeviceTokenBuffer) (hipGreedySampleResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipGreedySampleResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipGreedySampleResult{}, core.E("rocm.hip.SoftcapGreedyLaunch", "HIP driver is not available", nil)
	}
	if logits == nil || logits.Pointer() == 0 {
		return hipGreedySampleResult{}, core.E("rocm.hip.SoftcapGreedyLaunch", "logits device buffer is required", nil)
	}
	if logits.Count() <= 0 || logits.SizeBytes() != uint64(logits.Count()*4) {
		return hipGreedySampleResult{}, core.E("rocm.hip.SoftcapGreedyLaunch", "logits device buffer shape mismatch", nil)
	}
	if softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) {
		return hipGreedySampleResult{}, core.E("rocm.hip.SoftcapGreedyLaunch", "softcap must be non-negative and finite", nil)
	}
	if suppress != nil && (suppress.Pointer() == 0 || suppress.Count() <= 0 || suppress.SizeBytes() != uint64(suppress.Count()*4)) {
		return hipGreedySampleResult{}, core.E("rocm.hip.SoftcapGreedyLaunch", "suppress token buffer shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.SoftcapGreedyLaunch", "softcap greedy output", hipGreedyResultBytes, 1)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	defer output.Close()
	launchArgs := hipSoftcapGreedySampleLaunchArgs{
		LogitsPointer: logits.Pointer(),
		OutputPointer: output.Pointer(),
		Count:         logits.Count(),
		LogitsBytes:   logits.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
		Softcap:       softcap,
	}
	if suppress != nil {
		launchArgs.SuppressPointer = suppress.Pointer()
		launchArgs.SuppressCount = suppress.Count()
	}
	launchBytes, err := launchArgs.Binary()
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameSoftcapGreedy,
		Args:   launchBytes,
		GridX:  1,
		GridY:  1,
		GridZ:  1,
		BlockX: 256,
		BlockY: 1,
		BlockZ: 1,
	}
	if err := config.Validate(); err != nil {
		return hipGreedySampleResult{}, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return hipGreedySampleResult{}, err
	}
	return hipReadGreedyResult(output, "rocm.hip.SoftcapGreedyLaunch", "softcap greedy output", logits.Count())
}
