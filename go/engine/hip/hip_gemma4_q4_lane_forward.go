// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"math"

	core "dappco.re/go"
)

const hipGemma4Q4LaneForwardOperation = "rocm.hip.Gemma4Q4LaneForward"

type hipGemma4Q4LaneForwardRequest struct {
	Tokens       []int32
	Positions    []int
	DeviceStates []*hipGemma4Q4DeviceDecodeState
	Epsilon      float32
	Mode         string
	Workspace    *hipAttentionHeadsChunkedWorkspace
	EngineConfig hipGemma4Q4EngineConfig
}

type hipGemma4Q4LaneForwardBatch struct {
	Greedy       []hipGreedySampleResult
	DeviceStates []*hipGemma4Q4DeviceDecodeState
	FinalHidden  *hipDeviceByteBuffer
}

func hipRunGemma4Q4LaneForward(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, req hipGemma4Q4LaneForwardRequest) (*hipGemma4Q4LaneForwardBatch, error) {
	if err := hipValidateGemma4Q4LaneForward(driver, cfg, req); err != nil {
		return nil, err
	}
	laneCount := len(req.Tokens)
	mode := firstNonEmptyString(req.Mode, req.DeviceStates[0].mode, rocmKVCacheModeFP16)
	engineConfig := req.EngineConfig
	if engineConfig.DeviceKVMode == "" {
		engineConfig = defaultHIPGemma4Q4EngineConfig()
	}
	engineConfig.DeviceKVMode = mode
	engineConfig.ForceBatchedProjection = true

	workspace := req.Workspace
	if workspace == nil {
		workspace = hipNewAttentionHeadsChunkedWorkspace()
		defer workspace.Close()
	}

	tokenBuffer, err := workspace.EnsurePrefillTokenBuffer(driver, req.Tokens)
	if err != nil {
		return nil, err
	}
	var embeddingView hipDeviceByteBuffer
	hidden, err := hipRunGemma4Q4PrefillEmbeddingBatchTokenBufferWorkspaceView(ctx, driver, cfg.Layers[0], req.Tokens, tokenBuffer, workspace, &embeddingView)
	if err != nil {
		return nil, err
	}

	var perLayerInputs *hipGemma4Q4PerLayerInputDeviceSet
	if cfg.Layers[0].PerLayerInput.hasGlobalPrecompute() {
		perLayerInputs, err = hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatchWorkspaceTokenBuffer(ctx, driver, cfg, req.Tokens, hidden, req.Epsilon, workspace, tokenBuffer)
		if err != nil {
			return nil, err
		}
	}

	positionIDs := make([]int32, laneCount)
	for index, position := range req.Positions {
		positionIDs[index] = int32(position)
	}
	positionBuffer, err := workspace.EnsurePrefillTokenBuffer(driver, positionIDs)
	if err != nil {
		return nil, err
	}

	sharedSources := hipGemma4Q4SharedKVSourceByLayer(cfg)
	nextKV := make([][]*hipGemma4Q4PrefillDeviceKVBatch, laneCount)
	for lane := range nextKV {
		nextKV[lane] = make([]*hipGemma4Q4PrefillDeviceKVBatch, len(cfg.Layers))
	}
	defer func() {
		for lane := range nextKV {
			for _, deviceKV := range nextKV[lane] {
				_ = deviceKV.Close()
			}
		}
	}()

	for layerIndex, layerCfg := range cfg.Layers {
		if metrics := hipActiveDecodeRouteMetrics(); metrics != nil {
			metrics.setLayer(layerIndex, layerCfg.LayerType)
		}
		var inputNormView hipDeviceByteBuffer
		inputNorm, runErr := hipRunGemma4Q4PrefillInputNormBatchWorkspaceView(ctx, driver, layerCfg, hidden, laneCount, workspace, &inputNormView)
		if runErr != nil {
			return nil, runErr
		}

		var query *hipDeviceByteBuffer
		if sharedSources[layerIndex] == layerIndex {
			qkv, runErr := hipRunGemma4Q4PrefillQKVProjectionBatchWorkspaceTransient(ctx, driver, layerCfg, inputNorm, laneCount, workspace, true)
			if runErr != nil {
				return nil, runErr
			}
			qk, runErr := hipRunGemma4Q4LaneQKNormRoPEBatch(ctx, driver, layerCfg, qkv, positionBuffer, laneCount, req.Epsilon, workspace)
			if runErr != nil {
				_ = qkv.Close()
				return nil, runErr
			}
			var valueView hipDeviceByteBuffer
			value, runErr := hipRunGemma4Q4PrefillValueNormBatchWorkspaceView(ctx, driver, layerCfg, qkv, laneCount, req.Epsilon, workspace, true, &valueView)
			if runErr != nil {
				_ = qk.Close()
				_ = qkv.Close()
				return nil, runErr
			}
			for lane := 0; lane < laneCount; lane++ {
				keyView := hipGemma4Q4LaneRowView(driver, qk.Key, lane, layerCfg.keyValueDim(), "lane key")
				valueRowView := hipGemma4Q4LaneRowView(driver, value, lane, layerCfg.keyValueDim(), "lane value")
				laneQK := hipGemma4Q4PrefillRoPEQKBatch{Key: &keyView}
				prior := &req.DeviceStates[lane].layers[layerIndex]
				nextKV[lane][layerIndex], runErr = hipRunGemma4Q4PrefillDeviceKVBatchWithPriorDescriptorIntoWithEngineConfig(
					ctx, driver, layerCfg, prior.cache, prior.descriptorTable, &laneQK, &valueRowView, 1, mode, nil, engineConfig,
				)
				if runErr != nil {
					_ = value.Close()
					_ = qk.Close()
					_ = qkv.Close()
					return nil, runErr
				}
			}
			query = qk.Query
			defer qkv.Close()
			defer qk.Close()
			defer value.Close()
		} else {
			query, runErr = hipRunGemma4Q4LaneSharedQuery(ctx, driver, layerCfg, inputNorm, positionBuffer, req.Positions, laneCount, req.Epsilon, workspace)
			if runErr != nil {
				return nil, runErr
			}
		}

		attentionLanes := make([]hipAttentionHeadsLaneBatchLane, laneCount)
		for lane := 0; lane < laneCount; lane++ {
			source := sharedSources[layerIndex]
			deviceKV := nextKV[lane][source]
			if deviceKV == nil || deviceKV.Cache == nil || deviceKV.DescriptorTable == nil {
				return nil, core.E(hipGemma4Q4LaneForwardOperation, core.Sprintf("layer %d lane %d source KV is unavailable", layerIndex, lane), nil)
			}
			attentionLanes[lane] = hipAttentionHeadsLaneBatchLane{
				DeviceKV:        deviceKV.Cache,
				DescriptorTable: deviceKV.DescriptorTable,
				WindowSize:      layerCfg.SlidingWindow,
			}
		}
		attentionOutput, runErr := workspace.EnsureBatchAttentionOutput(driver, laneCount*layerCfg.QueryHeads*layerCfg.HeadDim)
		if runErr != nil {
			return nil, runErr
		}
		if runErr = hipRunAttentionHeadsLaneBatchOutputFromDeviceQueryToDeviceKernel(ctx, driver, hipAttentionHeadsLaneBatchDeviceRequest{
			Lanes:     attentionLanes,
			Dim:       layerCfg.HeadDim,
			HeadCount: layerCfg.QueryHeads,
			KeyHeads:  firstPositiveInt(layerCfg.KeyHeads, 1),
			Scale:     hipGemma4Q4AttentionScale(layerCfg.HeadDim),
		}, query, attentionOutput); runErr != nil {
			return nil, runErr
		}

		attentionOverride := hipBorrowDeviceByteBufferValue(driver, "lane attention override", attentionOutput.Pointer(), attentionOutput.SizeBytes(), attentionOutput.Count())
		bodyLayer := hipGemma4Q4PrefillLayerKVBatch{AttentionOverride: &attentionOverride}
		var perLayerInput *hipDeviceByteBuffer
		if perLayerInputs != nil {
			perLayerInput = perLayerInputs.Layer(layerIndex)
		}
		body, runErr := hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInputInternal(
			ctx, driver, layerCfg, hidden, &bodyLayer, perLayerInput, laneCount, 0, req.Epsilon,
			workspace, nil, nil, nil, true,
		)
		if runErr != nil {
			return nil, runErr
		}
		nextHidden := hipBorrowDeviceByteBufferValue(driver, "lane final hidden", body.FinalHidden.Pointer(), body.FinalHidden.SizeBytes(), body.FinalHidden.Count())
		if runErr = body.Close(); runErr != nil {
			return nil, runErr
		}
		hidden = &nextHidden
	}

	greedy, err := hipRunGemma4Q4PrefillFinalGreedyBatchSuppressWorkspace(ctx, driver, cfg.Layers[len(cfg.Layers)-1], hidden, laneCount, req.Epsilon, nil, workspace)
	if err != nil {
		return nil, err
	}
	nextStates, err := hipGemma4Q4LaneForwardStates(cfg, req.DeviceStates, nextKV, sharedSources, mode)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			for _, state := range nextStates {
				_ = state.Close()
			}
		}
	}()
	for lane := range nextStates {
		if err := hipFinalizeGemma4Q4ForwardDeviceState(req.DeviceStates[lane], nextStates[lane]); err != nil {
			return nil, core.E(hipGemma4Q4LaneForwardOperation, core.Sprintf("finalize lane %d state", lane), err)
		}
	}
	finalHidden := hipBorrowDeviceByteBufferValue(driver, "lane final hidden result", hidden.Pointer(), hidden.SizeBytes(), hidden.Count())
	success = true
	return &hipGemma4Q4LaneForwardBatch{Greedy: greedy, DeviceStates: nextStates, FinalHidden: &finalHidden}, nil
}

func hipValidateGemma4Q4LaneForward(driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, req hipGemma4Q4LaneForwardRequest) error {
	if driver == nil || !driver.Available() {
		return core.E(hipGemma4Q4LaneForwardOperation, "HIP driver is not available", nil)
	}
	if err := cfg.validate(); err != nil {
		return err
	}
	if len(req.Tokens) == 0 || len(req.Positions) != len(req.Tokens) || len(req.DeviceStates) != len(req.Tokens) {
		return core.E(hipGemma4Q4LaneForwardOperation, "tokens, positions, and device states must have the same positive lane count", nil)
	}
	if req.Epsilon < 0 || math.IsNaN(float64(req.Epsilon)) || math.IsInf(float64(req.Epsilon), 0) {
		return core.E(hipGemma4Q4LaneForwardOperation, "epsilon must be non-negative and finite", nil)
	}
	for _, layer := range cfg.Layers {
		if layer.PerLayerInput.hasLayerApply() && !cfg.Layers[0].PerLayerInput.hasGlobalPrecompute() {
			return core.E(hipGemma4Q4LaneForwardOperation, "lane forwarding requires global per-layer input precompute", nil)
		}
	}
	mode := firstNonEmptyString(req.Mode, req.DeviceStates[0].mode, rocmKVCacheModeFP16)
	if !isROCmKVCacheMode(mode) {
		return core.E(hipGemma4Q4LaneForwardOperation, "device KV mode is unsupported", nil)
	}
	sharedSources := hipGemma4Q4SharedKVSourceByLayer(cfg)
	const maxPosition = int(^uint32(0) >> 1)
	for lane, state := range req.DeviceStates {
		position := req.Positions[lane]
		if position < 0 || position > maxPosition {
			return core.E(hipGemma4Q4LaneForwardOperation, core.Sprintf("lane %d position must fit a non-negative int32", lane), nil)
		}
		if state == nil || state.closed || len(state.layers) != len(cfg.Layers) {
			return core.E(hipGemma4Q4LaneForwardOperation, core.Sprintf("lane %d device state is unavailable or has the wrong layer count", lane), nil)
		}
		if state.mode != "" && state.mode != mode {
			return core.E(hipGemma4Q4LaneForwardOperation, core.Sprintf("lane %d device KV mode mismatch", lane), nil)
		}
		for layerIndex, source := range sharedSources {
			if source != layerIndex {
				continue
			}
			prior := &state.layers[layerIndex]
			if prior.borrowedCache || prior.cache == nil || prior.descriptorTable == nil {
				return core.E(hipGemma4Q4LaneForwardOperation, core.Sprintf("lane %d owner layer %d device KV is unavailable", lane, layerIndex), nil)
			}
			priorTokens := prior.cache.TokenCount()
			if priorTokens != position && (cfg.Layers[layerIndex].SlidingWindow <= 0 || priorTokens <= 0 || priorTokens > position) {
				return core.E(hipGemma4Q4LaneForwardOperation, core.Sprintf("lane %d owner layer %d KV length does not match its position", lane, layerIndex), nil)
			}
		}
	}
	return nil
}

func hipRunGemma4Q4LaneQKNormRoPEBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, qkv *hipGemma4Q4PrefillQKVBatch, positions *hipDeviceTokenBuffer, laneCount int, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipGemma4Q4PrefillRoPEQKBatch, error) {
	if qkv == nil || qkv.Query == nil || qkv.Key == nil {
		return nil, core.E(hipGemma4Q4LaneForwardOperation, "lane Q/K projection buffers are required", nil)
	}
	queryOutput, err := workspace.EnsureRMSRoPEOutput(driver, qkv.Query.Count())
	if err != nil {
		return nil, err
	}
	keyOutput, err := workspace.EnsureKeyRMSRoPEOutput(driver, qkv.Key.Count())
	if err != nil {
		return nil, err
	}
	queryNormCfg := hipGemma4Q4RoPENormConfig(cfg.QueryNorm, epsilon, cfg.HeadDim)
	keyNormCfg := hipGemma4Q4RoPENormConfig(cfg.KeyNorm, epsilon, cfg.HeadDim)
	frequencyDim, rotaryCount := hipGemma4Q4RoPEKernelDims(cfg)
	if err := hipRunRMSNormRoPEHeadsPairLaneBatchKernelWithDeviceInputWeightConfigFrequencyScaleOutput(
		ctx, driver, qkv.Query, qkv.Key, queryNormCfg, keyNormCfg,
		cfg.QueryHeads, firstPositiveInt(cfg.KeyHeads, 1), positions,
		cfg.RoPEBase, frequencyDim, rotaryCount, cfg.effectiveRoPEFrequencyScale(),
		queryOutput, keyOutput,
	); err != nil {
		return nil, err
	}
	out := &hipGemma4Q4PrefillRoPEQKBatch{}
	out.Query = out.borrowQueryView(driver, "lane query RoPE", queryOutput)
	out.Key = out.borrowKeyView(driver, "lane key RoPE", keyOutput)
	return out, nil
}

func hipRunGemma4Q4LaneSharedQuery(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, inputNorm *hipDeviceByteBuffer, positions *hipDeviceTokenBuffer, lanePositions []int, laneCount int, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, error) {
	var projectionView hipDeviceByteBuffer
	query, err := hipRunGemma4Q4PrefillProjectionBatchWorkspaceView(ctx, driver, inputNorm, cfg.QueryProjection, laneCount, workspace, "lane shared query projection", &projectionView, true)
	if err != nil {
		return nil, err
	}
	_ = positions
	queryOutput, err := workspace.EnsureRMSRoPEOutput(driver, query.Count())
	if err != nil {
		return nil, err
	}
	normCfg := hipGemma4Q4RoPENormConfig(cfg.QueryNorm, epsilon, cfg.HeadDim)
	frequencyDim, rotaryCount := hipGemma4Q4RoPEKernelDims(cfg)
	rowCount := cfg.QueryHeads * cfg.HeadDim
	for lane, position := range lanePositions {
		inputRow := hipGemma4Q4LaneRowView(driver, query, lane, rowCount, "lane shared query input")
		outputRow := hipGemma4Q4LaneRowView(driver, queryOutput, lane, rowCount, "lane shared query output")
		if err := hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutputFrequencyScaleWithWorkspace(
			ctx, driver, &inputRow, normCfg, cfg.QueryHeads, position, cfg.RoPEBase,
			frequencyDim, rotaryCount, cfg.effectiveRoPEFrequencyScale(), &outputRow, workspace,
		); err != nil {
			return nil, err
		}
	}
	out := hipBorrowDeviceByteBufferValue(driver, "lane shared query RoPE", queryOutput.Pointer(), queryOutput.SizeBytes(), queryOutput.Count())
	return &out, nil
}

func hipGemma4Q4LaneRowView(driver nativeHIPDriver, buffer *hipDeviceByteBuffer, row, rowCount int, label string) hipDeviceByteBuffer {
	rowBytes := uint64(rowCount * 4)
	return hipBorrowDeviceByteBufferValue(driver, label, buffer.Pointer()+nativeDevicePointer(uint64(row)*rowBytes), rowBytes, rowCount)
}

func hipGemma4Q4LaneForwardStates(cfg hipGemma4Q4ForwardConfig, previous []*hipGemma4Q4DeviceDecodeState, nextKV [][]*hipGemma4Q4PrefillDeviceKVBatch, sharedSources []int, mode string) ([]*hipGemma4Q4DeviceDecodeState, error) {
	nextStates := make([]*hipGemma4Q4DeviceDecodeState, len(previous))
	success := false
	defer func() {
		if !success {
			for _, state := range nextStates {
				_ = state.Close()
			}
		}
	}()
	for lane := range previous {
		state := hipNewGemma4Q4DeviceDecodeState(mode, len(cfg.Layers))
		state.appendLayers = len(cfg.Layers)
		nextStates[lane] = state
		for layerIndex, source := range sharedSources {
			if source != layerIndex {
				shared, err := hipGemma4Q4PrefillSharedDecodeLayerState(state, source)
				if err != nil {
					return nil, err
				}
				state.layers = append(state.layers, shared)
				continue
			}
			deviceKV := nextKV[lane][layerIndex]
			if deviceKV == nil || deviceKV.Cache == nil || deviceKV.DescriptorTable == nil {
				return nil, core.E(hipGemma4Q4LaneForwardOperation, core.Sprintf("lane %d owner layer %d next KV is unavailable", lane, layerIndex), nil)
			}
			if err := hipGemma4Q4PrefillFinalizeRetainWindow(deviceKV); err != nil {
				return nil, err
			}
			state.layers = append(state.layers, hipGemma4Q4DeviceLayerKVState{
				cache:           deviceKV.Cache,
				descriptorTable: deviceKV.DescriptorTable,
				launch:          deviceKV.Launch,
			})
			deviceKV.Cache = nil
			deviceKV.DescriptorTable = nil
			deviceKV.Launch = rocmDeviceKVLaunchDescriptor{}
		}
	}
	success = true
	return nextStates, nil
}
