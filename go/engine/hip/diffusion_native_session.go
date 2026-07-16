// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"
	"sync"

	core "dappco.re/go"
)

const hipDiffusionGemmaOperation = "rocm.hip.DiffusionGemma"

type hipDiffusionSelfConditionConfig struct {
	PreNorm hipRMSNormDeviceWeightConfig
	Gate    hipMLXQ4DeviceWeightConfig
	Up      hipMLXQ4DeviceWeightConfig
	Down    hipMLXQ4DeviceWeightConfig
	Ones    hipRMSNormDeviceWeightConfig
}

type hipROCmNativeDiffusionSession struct {
	mu             sync.Mutex
	loaded         *hipLoadedModel
	driver         nativeHIPDriver
	causalConfig   hipGemma4Q4ForwardConfig
	encoderScalars []float32
	selfCondition  hipDiffusionSelfConditionConfig
	ones           *hipDeviceByteBuffer
	engine         hipGemma4Q4EngineConfig
	mode           string
	device         *hipGemma4Q4DeviceDecodeState
	position       int
	workspace      hipAttentionHeadsChunkedWorkspace
	closed         bool
}

var _ ROCmDiffusionSession = (*hipROCmNativeDiffusionSession)(nil)

func newHIPROCmNativeDiffusionSession(model *hipLoadedModel) (*hipROCmNativeDiffusionSession, error) {
	if model == nil {
		return nil, core.E(hipDiffusionGemmaOperation, "loaded model is required", nil)
	}
	if normalizeROCmArchitecture(model.modelIdentity().Architecture) != "diffusion_gemma" {
		return nil, core.E(hipDiffusionGemmaOperation, "model is not DiffusionGemma", nil)
	}
	if model.driver == nil || !model.driver.Available() {
		return nil, core.E(hipDiffusionGemmaOperation, "HIP driver is not available", nil)
	}
	if model.modelInfo.NumLayers <= 0 || model.modelInfo.HiddenSize <= 0 || model.modelInfo.VocabSize <= 0 {
		return nil, core.E(hipDiffusionGemmaOperation, "model layer, hidden, and vocab sizes are required", nil)
	}
	causal, err := model.cachedGemma4Q4ForwardConfig(model.modelInfo.NumLayers)
	if err != nil {
		return nil, core.E(hipDiffusionGemmaOperation, "load Gemma4 decoder trunk", err)
	}
	encoderScalars := make([]float32, len(causal.Layers))
	for layer := range encoderScalars {
		encoderScalars[layer], err = model.loadedDiffusionGemmaEncoderLayerScalar(layer)
		if err != nil {
			return nil, err
		}
	}
	selfCondition, err := model.loadedDiffusionGemmaSelfConditionConfig()
	if err != nil {
		return nil, err
	}
	ones := make([]float32, model.modelInfo.HiddenSize)
	for index := range ones {
		ones[index] = 1
	}
	onesBuffer, err := hipUploadGemma4Q4Float32Input(model.driver, "diffusion unit RMSNorm weights", ones)
	if err != nil {
		return nil, err
	}
	selfCondition.Ones = hipRMSNormDeviceWeightConfig{
		WeightPointer:  onesBuffer.Pointer(),
		WeightBytes:    onesBuffer.SizeBytes(),
		Count:          model.modelInfo.HiddenSize,
		WeightEncoding: hipRMSNormWeightEncodingF32,
	}
	engine := model.gemma4Q4EngineConfig()
	mode, err := engine.deviceKVMode()
	if err != nil {
		_ = onesBuffer.Close()
		return nil, err
	}
	return &hipROCmNativeDiffusionSession{
		loaded:         model,
		driver:         model.driver,
		causalConfig:   causal,
		encoderScalars: encoderScalars,
		selfCondition:  selfCondition,
		ones:           onesBuffer,
		engine:         engine,
		mode:           mode,
	}, nil
}

func (model *hipLoadedModel) loadedDiffusionGemmaSelfConditionConfig() (hipDiffusionSelfConditionConfig, error) {
	if model == nil || model.modelInfo.HiddenSize <= 0 {
		return hipDiffusionSelfConditionConfig{}, core.E(hipDiffusionGemmaOperation, "loaded model hidden size is required", nil)
	}
	hidden := model.modelInfo.HiddenSize
	groupSize := model.modelInfo.QuantGroup
	if groupSize <= 0 {
		groupSize = 64
	}
	preNorm, err := model.loadedGemma4NormConfig("self_conditioning.pre_norm.weight", "self-conditioning pre-norm", hidden)
	if err != nil {
		return hipDiffusionSelfConditionConfig{}, err
	}
	gate, gateRows, gateCols, err := model.loadedGemma4Q4ProjectionConfig("self_conditioning.gate_proj", "self-conditioning gate projection", groupSize)
	if err != nil {
		return hipDiffusionSelfConditionConfig{}, err
	}
	up, upRows, upCols, err := model.loadedGemma4Q4ProjectionConfig("self_conditioning.up_proj", "self-conditioning up projection", groupSize)
	if err != nil {
		return hipDiffusionSelfConditionConfig{}, err
	}
	down, downRows, downCols, err := model.loadedGemma4Q4ProjectionConfig("self_conditioning.down_proj", "self-conditioning down projection", groupSize)
	if err != nil {
		return hipDiffusionSelfConditionConfig{}, err
	}
	if gateRows <= 0 || gateCols != hidden || upRows != gateRows || upCols != hidden || downRows != hidden || downCols != gateRows {
		return hipDiffusionSelfConditionConfig{}, core.E(hipDiffusionGemmaOperation, "self-conditioning projection geometry mismatch", nil)
	}
	return hipDiffusionSelfConditionConfig{PreNorm: preNorm, Gate: gate, Up: up, Down: down}, nil
}

func (session *hipROCmNativeDiffusionSession) PrefillTokens(ids []int32) (int, error) {
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.closed {
		return 0, core.E(hipDiffusionGemmaOperation, "session is closed", nil)
	}
	if len(ids) == 0 {
		return 0, core.E(hipDiffusionGemmaOperation, "prompt tokens are required", nil)
	}
	if len(ids) > session.contextLimitLocked() {
		return 0, core.E(hipDiffusionGemmaOperation, "prompt exceeds model context window", nil)
	}
	if err := session.closeDeviceLocked(); err != nil {
		return 0, err
	}
	session.position = 0
	if err := session.appendTokensLocked(context.Background(), ids); err != nil {
		_ = session.closeDeviceLocked()
		session.position = 0
		return 0, err
	}
	return session.position, nil
}

func (session *hipROCmNativeDiffusionSession) CacheOffset() int {
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.closed {
		return -1
	}
	return session.position
}

func (session *hipROCmNativeDiffusionSession) Denoise(ctx context.Context, req ROCmDiffusionDenoiseRequest) (ROCmDiffusionStepResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return ROCmDiffusionStepResult{}, err
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.closed {
		return ROCmDiffusionStepResult{}, core.E(hipDiffusionGemmaOperation, "session is closed", nil)
	}
	if err := session.validateDenoiseRequestLocked(req); err != nil {
		return ROCmDiffusionStepResult{}, err
	}
	config, err := hipDiffusionDenoiseForwardConfig(session.causalConfig, session.encoderScalars, len(req.Canvas))
	if err != nil {
		return ROCmDiffusionStepResult{}, err
	}
	embedding, err := hipRunGemma4Q4PrefillEmbeddingBatch(ctx, session.driver, config.Layers[0], req.Canvas)
	if err != nil {
		return ROCmDiffusionStepResult{}, core.E(hipDiffusionGemmaOperation, "embed canvas", err)
	}
	defer embedding.Close()

	perLayerSet, err := hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatch(ctx, session.driver, config, req.Canvas, embedding, 1e-6)
	if err != nil {
		return ROCmDiffusionStepResult{}, core.E(hipDiffusionGemmaOperation, "prepare per-layer canvas inputs", err)
	}
	if perLayerSet != nil {
		defer perLayerSet.Close()
	}
	perLayerInputs, perLayerViews, err := hipDiffusionPerLayerInputViews(perLayerSet)
	if err != nil {
		return ROCmDiffusionStepResult{}, err
	}
	_ = perLayerViews

	initialHidden, err := session.applySelfConditionLocked(ctx, embedding, req.SCEmb, len(req.Canvas))
	if err != nil {
		return ROCmDiffusionStepResult{}, err
	}
	defer initialHidden.Close()
	caps := make([]int32, len(req.Canvas))
	for index := range caps {
		caps[index] = int32(session.position + len(req.Canvas))
	}
	visibleCaps, err := hipUploadTokenIDs(session.driver, caps)
	if err != nil {
		return ROCmDiffusionStepResult{}, err
	}
	defer visibleCaps.Close()
	priorKV := hipGemma4Q4DeviceLayerCaches(session.device, nil, len(config.Layers))
	priorDescriptors, err := hipGemma4Q4DeviceLayerDescriptorTableAliases(session.device, nil, len(config.Layers))
	if err != nil {
		return ROCmDiffusionStepResult{}, core.E(hipDiffusionGemmaOperation, "borrow resident prefix descriptors", err)
	}
	defer hipCloseGemma4Q4DeviceLayerDescriptorTables(priorDescriptors)
	forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowInitialHiddenWithEngineConfig(
		ctx,
		session.driver,
		config,
		req.Canvas,
		session.position,
		1e-6,
		session.mode,
		priorKV,
		priorDescriptors,
		perLayerInputs,
		nil,
		-1,
		nil,
		&session.workspace,
		session.engine,
		initialHidden,
		visibleCaps,
	)
	if err != nil {
		return ROCmDiffusionStepResult{}, core.E(hipDiffusionGemmaOperation, "denoise canvas forward", err)
	}
	last := config.Layers[len(config.Layers)-1]
	finalNorm := last.FinalNorm
	finalNorm.Epsilon = 1e-6
	normalized, err := hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, session.driver, forward.FinalHidden, finalNorm, len(req.Canvas))
	if err != nil {
		_ = forward.Close()
		return ROCmDiffusionStepResult{}, core.E(hipDiffusionGemmaOperation, "normalize denoise output", err)
	}
	logitsDevice, err := hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx, session.driver, normalized, last.LMHeadProjection, len(req.Canvas))
	_ = normalized.Close()
	if err != nil {
		_ = forward.Close()
		return ROCmDiffusionStepResult{}, core.E(hipDiffusionGemmaOperation, "project denoise logits", err)
	}
	defer logitsDevice.Close()
	closeErr := forward.Close()
	if closeErr != nil {
		return ROCmDiffusionStepResult{}, closeErr
	}
	temperature := rocmDiffusionDenoiseTemperature(req.NoiseProportion, req.StepConfig)
	draws := rocmDiffusionCategoricalDraws(req.StepConfig.Seed, req.Step, len(req.Canvas))
	samples, err := hipRunDiffusionSampleKernel(ctx, session.driver, logitsDevice, len(req.Canvas), last.VocabSize, temperature, last.FinalLogitSoftcap, draws)
	if err != nil {
		return ROCmDiffusionStepResult{}, err
	}
	expectedDevice, err := hipRunDiffusionExpectedEmbeddingDeviceKernel(ctx, session.driver, logitsDevice, len(req.Canvas), config.Layers[0].Embedding, float32(math.Sqrt(float64(last.HiddenSize))))
	if err != nil {
		return ROCmDiffusionStepResult{}, err
	}
	defer expectedDevice.Close()
	expected, err := hipReadFloat32DeviceOutput(expectedDevice, hipDiffusionGemmaOperation, "denoise expected embedding", len(req.Canvas)*last.HiddenSize)
	if err != nil {
		return ROCmDiffusionStepResult{}, err
	}
	sampled := make([]int32, len(samples))
	greedy := make([]int32, len(samples))
	entropies := make([]float32, len(samples))
	for index, sample := range samples {
		sampled[index] = sample.Sampled
		greedy[index] = sample.Greedy
		entropies[index] = sample.Entropy
	}
	return rocmDiffusionFinalizeDenoiseStep(req.Canvas, sampled, greedy, entropies, hipDiffusionFloat32ToBF16(expected), last.HiddenSize, req.Step, req.StepConfig)
}

func (session *hipROCmNativeDiffusionSession) TruncateTo(position int) error {
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.closed {
		return core.E(hipDiffusionGemmaOperation, "session is closed", nil)
	}
	if position != session.position {
		return core.E(hipDiffusionGemmaOperation, core.Sprintf("canvas rollback position %d does not match resident prefix %d", position, session.position), nil)
	}
	return nil
}

func (session *hipROCmNativeDiffusionSession) CommitTokens(ids []int32) error {
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.closed {
		return core.E(hipDiffusionGemmaOperation, "session is closed", nil)
	}
	if len(ids) == 0 {
		return nil
	}
	if session.position > session.contextLimitLocked()-len(ids) {
		return core.E(hipDiffusionGemmaOperation, "committed canvas exceeds model context window", nil)
	}
	return session.appendTokensLocked(context.Background(), ids)
}

func (session *hipROCmNativeDiffusionSession) Close() error {
	if session == nil {
		return nil
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.closed {
		return nil
	}
	var lastErr error
	if err := session.closeDeviceLocked(); err != nil {
		lastErr = err
	}
	if err := session.workspace.Close(); err != nil {
		lastErr = err
	}
	if err := session.ones.Close(); err != nil {
		lastErr = err
	}
	session.closed = true
	return lastErr
}

func (session *hipROCmNativeDiffusionSession) appendTokensLocked(ctx context.Context, ids []int32) error {
	ubatchTokens, err := session.engine.prefillUBatchTokens()
	if err != nil {
		return err
	}
	plan, err := hipGemma4Q4PlanPromptPrefill(ids, session.position, ubatchTokens)
	if err != nil {
		return err
	}
	for batchIndex := 0; batchIndex < plan.LenBatches(); batchIndex++ {
		batch := plan.Batch(batchIndex)
		priorKV := hipGemma4Q4DeviceLayerCaches(session.device, nil, len(session.causalConfig.Layers))
		priorDescriptors := hipGemma4Q4DeviceLayerDescriptorTables(session.device, nil, len(session.causalConfig.Layers))
		forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowWithEngineConfig(
			ctx,
			session.driver,
			session.causalConfig,
			batch.Tokens,
			batch.Position,
			1e-6,
			session.mode,
			priorKV,
			priorDescriptors,
			nil,
			nil,
			-1,
			nil,
			&session.workspace,
			session.engine,
		)
		if err != nil {
			return core.E(hipDiffusionGemmaOperation, "materialize causal tokens", err)
		}
		next, stateErr := hipGemma4Q4DeviceDecodeStateFromPrefillForward(forward, session.mode)
		closeErr := forward.Close()
		if stateErr != nil {
			return stateErr
		}
		if closeErr != nil {
			_ = next.Close()
			return closeErr
		}
		previous := session.device
		if err := hipFinalizeGemma4Q4ForwardDeviceState(previous, next); err != nil {
			_ = next.Close()
			return err
		}
		session.device = next
		hipReleaseClosedGemma4Q4DeviceDecodeState(previous)
		session.position = batch.Position + len(batch.Tokens)
	}
	return nil
}

func (session *hipROCmNativeDiffusionSession) applySelfConditionLocked(ctx context.Context, embedding *hipDeviceByteBuffer, scEmb []byte, rows int) (*hipDeviceByteBuffer, error) {
	if embedding == nil || embedding.Pointer() == 0 || rows <= 0 {
		return nil, core.E(hipDiffusionGemmaOperation, "canvas embedding is required", nil)
	}
	current := embedding
	if len(scEmb) > 0 {
		values, err := hipDiffusionBF16ToFloat32(scEmb, rows*session.loaded.modelInfo.HiddenSize)
		if err != nil {
			return nil, err
		}
		scDevice, err := hipUploadGemma4Q4Float32Input(session.driver, "diffusion self-conditioning embedding", values)
		if err != nil {
			return nil, err
		}
		defer scDevice.Close()
		preNorm := session.selfCondition.PreNorm
		preNorm.Epsilon = 1e-6
		normalized, err := hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, session.driver, scDevice, preNorm, rows)
		if err != nil {
			return nil, err
		}
		defer normalized.Close()
		activation, err := hipRunMLXQ4GELUTanhMultiplyBatchKernelWithDeviceInput(ctx, session.driver, normalized, session.selfCondition.Gate, session.selfCondition.Up, rows)
		if err != nil {
			return nil, err
		}
		defer activation.Close()
		conditioned, err := hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx, session.driver, activation, session.selfCondition.Down, rows)
		if err != nil {
			return nil, err
		}
		defer conditioned.Close()
		combined, err := hipRunVectorAddDeviceKernel(ctx, session.driver, embedding, conditioned)
		if err != nil {
			return nil, err
		}
		defer combined.Close()
		current = combined
	}
	ones := session.selfCondition.Ones
	ones.Epsilon = 1e-6
	return hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, session.driver, current, ones, rows)
}

func (session *hipROCmNativeDiffusionSession) validateDenoiseRequestLocked(req ROCmDiffusionDenoiseRequest) error {
	length := len(req.Canvas)
	if session.device == nil || session.position <= 0 {
		return core.E(hipDiffusionGemmaOperation, "prompt prefix is not resident", nil)
	}
	if length <= 0 || req.Prefix != session.position {
		return core.E(hipDiffusionGemmaOperation, "canvas and resident prefix do not align", nil)
	}
	if session.position > session.contextLimitLocked()-length {
		return core.E(hipDiffusionGemmaOperation, "canvas exceeds model context window", nil)
	}
	if len(req.SCEmb) != 0 && len(req.SCEmb) != length*session.loaded.modelInfo.HiddenSize*2 {
		return core.E(hipDiffusionGemmaOperation, "self-conditioning embedding byte count mismatch", nil)
	}
	keyLength := session.position + length
	wantGlobal, wantGlobalShape := rocmDiffusionGlobalCanvasMask(length, keyLength)
	wantLocal, wantLocalShape := rocmDiffusionLocalCanvasMask(length, keyLength, session.position, hipDiffusionSlidingWindow(session.causalConfig))
	if !hipDiffusionIntSlicesEqual(req.GlobalMaskShape, wantGlobalShape) || !hipDiffusionFloatSlicesEqual(req.GlobalMask, wantGlobal) {
		return core.E(hipDiffusionGemmaOperation, "global canvas mask mismatch", nil)
	}
	if !hipDiffusionIntSlicesEqual(req.LocalMaskShape, wantLocalShape) || !hipDiffusionFloatSlicesEqual(req.LocalMask, wantLocal) {
		return core.E(hipDiffusionGemmaOperation, "local canvas mask mismatch", nil)
	}
	return nil
}

func hipDiffusionSlidingWindow(config hipGemma4Q4ForwardConfig) int {
	for _, layer := range config.Layers {
		if layer.SlidingWindow > 0 {
			return layer.SlidingWindow
		}
	}
	return 0
}

func (session *hipROCmNativeDiffusionSession) contextLimitLocked() int {
	if session.loaded != nil && session.loaded.contextSize > 0 {
		return session.loaded.contextSize
	}
	return defaultContextLengthCap
}

func (session *hipROCmNativeDiffusionSession) closeDeviceLocked() error {
	if session.device == nil {
		return nil
	}
	err := session.device.Close()
	hipReleaseClosedGemma4Q4DeviceDecodeState(session.device)
	session.device = nil
	return err
}

func hipDiffusionPerLayerInputViews(set *hipGemma4Q4PerLayerInputDeviceSet) ([]*hipDeviceByteBuffer, []hipDeviceByteBuffer, error) {
	if set == nil {
		return nil, nil, nil
	}
	if set.layerCount <= 0 || set.layerStrideBytes == 0 || set.layerValueCount <= 0 || len(set.Backing) == 0 || set.Backing[0] == nil || set.Backing[0].Pointer() == 0 {
		return nil, nil, core.E(hipDiffusionGemmaOperation, "per-layer input device set is incomplete", nil)
	}
	storage := make([]hipDeviceByteBuffer, set.layerCount)
	views := make([]*hipDeviceByteBuffer, set.layerCount)
	for layer := range storage {
		storage[layer] = hipBorrowDeviceByteBufferValue(
			set.driver,
			"diffusion per-layer input view",
			set.Backing[0].Pointer()+nativeDevicePointer(uint64(layer)*set.layerStrideBytes),
			set.layerStrideBytes,
			set.layerValueCount,
		)
		views[layer] = &storage[layer]
	}
	return views, storage, nil
}

func hipDiffusionBF16ToFloat32(payload []byte, count int) ([]float32, error) {
	if count <= 0 || len(payload) != count*2 {
		return nil, core.E(hipDiffusionGemmaOperation, "BF16 payload byte count mismatch", nil)
	}
	out := make([]float32, count)
	for index := range out {
		out[index] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload[index*2:]))
	}
	return out, nil
}

func hipDiffusionFloat32ToBF16(values []float32) []byte {
	out := make([]byte, len(values)*2)
	for index, value := range values {
		binary.LittleEndian.PutUint16(out[index*2:], hipFloat32ToBFloat16(value))
	}
	return out
}

func hipDiffusionIntSlicesEqual(left, right []int) bool {
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

func hipDiffusionFloatSlicesEqual(left, right []float32) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] && !(math.IsInf(float64(left[index]), -1) && math.IsInf(float64(right[index]), -1)) {
			return false
		}
	}
	return true
}
