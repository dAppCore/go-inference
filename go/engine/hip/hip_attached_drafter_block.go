// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"

	core "dappco.re/go"
)

const hipAttachedDrafterTargetVerifyBatchSuffixMinRows = 2

type hipAttachedDrafterAssistantDraftBlockRequest struct {
	LastToken         int32
	TargetHidden      *hipDeviceByteBuffer
	TargetForward     hipGemma4Q4ForwardConfig
	TargetDeviceState *hipGemma4Q4DeviceDecodeState
	Plan              hipAttachedDrafterAssistantVerifierPlan
	InputPlan         hipAttachedDrafterAssistantDraftStepInputPlan
	Position          int
	Epsilon           float32
	Softcap           float32
	SuppressTokens    []int32
	MaxDraftTokens    int
	Workspace         *hipAttentionHeadsChunkedWorkspace
}

type hipAttachedDrafterAssistantDraftBlockResult struct {
	Tokens []int32
	Hidden *hipDeviceByteBuffer
}

type hipAttachedDrafterTargetVerifyBlockRequest struct {
	TargetForward     hipGemma4Q4ForwardConfig
	DeviceKVMode      string
	EngineConfig      hipGemma4Q4EngineConfig
	TargetDeviceState *hipGemma4Q4DeviceDecodeState
	CurrentGreedy     hipGreedySampleResult
	DraftTokens       []int32
	Position          int
	Epsilon           float32
	SuppressTokens    []int32
	GreedyBuffer      *hipDeviceByteBuffer
	Workspace         *hipAttentionHeadsChunkedWorkspace
}

type hipAttachedDrafterTargetVerifyBlockResult struct {
	AcceptedCount             int
	RejectedCount             int
	Replacement               hipGreedySampleResult
	NextGreedy                hipGreedySampleResult
	AllAccepted               bool
	DeviceState               *hipGemma4Q4DeviceDecodeState
	DeviceHidden              *hipDeviceByteBuffer
	PriorDeviceStateFinalized bool
	TargetCalls               int
	VerifiedGreedies          []hipGreedySampleResult
}

func (result *hipAttachedDrafterAssistantDraftBlockResult) Close() error {
	if result == nil {
		return nil
	}
	err := result.Hidden.Close()
	result.Hidden = nil
	result.Tokens = nil
	return err
}

func (result *hipAttachedDrafterTargetVerifyBlockResult) Close() error {
	if result == nil {
		return nil
	}
	var lastErr error
	if err := result.DeviceState.Close(); err != nil {
		lastErr = err
	}
	if err := result.DeviceHidden.Close(); err != nil {
		lastErr = err
	}
	result.DeviceState = nil
	result.DeviceHidden = nil
	result.VerifiedGreedies = nil
	return lastErr
}

func hipAttachedDrafterResolveDraftTokens(requested, remaining int) int {
	if remaining <= 0 {
		return 0
	}
	if requested <= 0 {
		requested = ProductionMTPDefaultDraftTokens
	}
	if requested <= 0 {
		requested = 1
	}
	if requested > remaining {
		return remaining
	}
	return requested
}

func hipAttachedDrafterResolveDraftTokensForTarget(target hipGemma4Q4ForwardConfig, requested, remaining int) int {
	resolved := hipAttachedDrafterResolveDraftTokens(requested, remaining)
	if resolved <= 0 {
		return 0
	}
	if maxProposals := hipAttachedDrafterMaxDraftProposalsForTarget(target); maxProposals > 0 && resolved > maxProposals {
		return maxProposals
	}
	return resolved
}

func hipAttachedDrafterAdaptDraftTokens(current, proposed, accepted int) int {
	if current <= ProductionMTPFallbackDraftTokens || proposed <= 0 {
		return current
	}
	if accepted*2 < proposed {
		return ProductionMTPFallbackDraftTokens
	}
	return current
}

func hipAttachedDrafterMaxDraftProposalsForTarget(target hipGemma4Q4ForwardConfig) int {
	maxProposals := 0
	for _, layer := range target.Layers {
		if layer.SlidingWindow <= 1 {
			continue
		}
		layerProposals := layer.SlidingWindow - 1
		if maxProposals == 0 || layerProposals < maxProposals {
			maxProposals = layerProposals
		}
	}
	return maxProposals
}

func hipRunAttachedDrafterAssistantDraftBlock(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterAssistantDraftBlockRequest) (hipAttachedDrafterAssistantDraftBlockResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipAttachedDrafterAssistantDraftBlockResult{}, err
	}
	if req.MaxDraftTokens <= 0 {
		return hipAttachedDrafterAssistantDraftBlockResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftBlock", "max draft tokens must be positive", nil)
	}
	if len(req.TargetForward.Layers) == 0 {
		return hipAttachedDrafterAssistantDraftBlockResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftBlock", "target forward config has no layers", nil)
	}
	if req.TargetHidden == nil || req.TargetHidden.Pointer() == 0 {
		return hipAttachedDrafterAssistantDraftBlockResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftBlock", "target hidden is required", nil)
	}
	tokens := make([]int32, 0, req.MaxDraftTokens)
	greedyTokenViews := make([]hipDeviceByteBuffer, 0, req.MaxDraftTokens)
	currentToken := req.LastToken
	var currentGreedyToken *hipDeviceByteBuffer
	targetNormCfg := req.TargetForward.Layers[len(req.TargetForward.Layers)-1].FinalNorm
	targetNormCfg.Epsilon = req.Epsilon
	if err := hipValidateRMSNormDeviceWeightConfig("AttachedDrafterAssistantDraftBlock.target_final_norm", targetNormCfg); err != nil {
		return hipAttachedDrafterAssistantDraftBlockResult{}, err
	}
	currentHidden, err := hipAllocateByteBuffer(driver, "rocm.hip.AttachedDrafterAssistantDraftBlock", "target final-norm seed", uint64(targetNormCfg.Count*4), targetNormCfg.Count)
	if err != nil {
		return hipAttachedDrafterAssistantDraftBlockResult{}, err
	}
	ownsCurrentHidden := true
	success := false
	defer func() {
		if success || !ownsCurrentHidden {
			return
		}
		_ = currentHidden.Close()
	}()
	if err := hipRunRMSNormDeviceToDeviceKernelWithWorkspace(ctx, driver, req.TargetHidden.Pointer(), uint64(targetNormCfg.Count)*4, currentHidden.Pointer(), currentHidden.SizeBytes(), targetNormCfg, req.Workspace); err != nil {
		return hipAttachedDrafterAssistantDraftBlockResult{}, err
	}
	useDeviceTokenChain := req.Plan.Embedding.TableEncoding == hipEmbeddingTableEncodingMLXQ4
	for len(tokens) < req.MaxDraftTokens {
		if useDeviceTokenChain {
			proposal, err := hipRunAttachedDrafterAssistantDraftStepDeviceToken(ctx, driver, hipAttachedDrafterAssistantDraftStepProposalRequest{
				LastToken:         currentToken,
				LastGreedyToken:   currentGreedyToken,
				TargetHidden:      currentHidden,
				TargetForward:     req.TargetForward,
				TargetDeviceState: req.TargetDeviceState,
				Plan:              req.Plan,
				InputPlan:         req.InputPlan,
				Position:          req.Position,
				Epsilon:           req.Epsilon,
				Softcap:           req.Softcap,
				SuppressTokens:    req.SuppressTokens,
				Workspace:         req.Workspace,
			})
			if ownsCurrentHidden {
				_ = currentHidden.Close()
				currentHidden = nil
				ownsCurrentHidden = false
			}
			if err != nil {
				return hipAttachedDrafterAssistantDraftBlockResult{}, err
			}
			greedyView, err := hipAttachedDrafterGreedyTokenBorrowedView(proposal.GreedyToken)
			if err != nil {
				_ = proposal.Close()
				return hipAttachedDrafterAssistantDraftBlockResult{}, err
			}
			greedyTokenViews = append(greedyTokenViews, greedyView)
			currentGreedyToken = &greedyTokenViews[len(greedyTokenViews)-1]
			tokens = append(tokens, 0)
			currentHidden = proposal.Hidden
			proposal.Hidden = nil
			ownsCurrentHidden = true
			if err := proposal.Close(); err != nil {
				return hipAttachedDrafterAssistantDraftBlockResult{}, err
			}
			continue
		}
		proposal, err := hipRunAttachedDrafterAssistantDraftStepProposal(ctx, driver, hipAttachedDrafterAssistantDraftStepProposalRequest{
			LastToken:         currentToken,
			TargetHidden:      currentHidden,
			TargetForward:     req.TargetForward,
			TargetDeviceState: req.TargetDeviceState,
			Plan:              req.Plan,
			InputPlan:         req.InputPlan,
			Position:          req.Position,
			Epsilon:           req.Epsilon,
			Softcap:           req.Softcap,
			SuppressTokens:    req.SuppressTokens,
			Workspace:         req.Workspace,
		})
		if ownsCurrentHidden {
			_ = currentHidden.Close()
			currentHidden = nil
			ownsCurrentHidden = false
		}
		if err != nil {
			return hipAttachedDrafterAssistantDraftBlockResult{}, err
		}
		currentToken = int32(proposal.Token.TokenID)
		tokens = append(tokens, currentToken)
		currentHidden = proposal.Hidden
		proposal.Hidden = nil
		ownsCurrentHidden = true
		if err := proposal.Close(); err != nil {
			return hipAttachedDrafterAssistantDraftBlockResult{}, err
		}
	}
	if len(greedyTokenViews) > 0 {
		readTokens, err := hipReadAttachedDrafterGreedyTokenViews(driver, greedyTokenViews, req.Plan.VocabSize)
		if err != nil {
			return hipAttachedDrafterAssistantDraftBlockResult{}, err
		}
		copy(tokens, readTokens)
	}
	success = true
	return hipAttachedDrafterAssistantDraftBlockResult{Tokens: tokens, Hidden: currentHidden}, nil
}

func hipAttachedDrafterGreedyTokenBorrowedView(buffer *hipDeviceByteBuffer) (hipDeviceByteBuffer, error) {
	if err := hipAttachedDrafterValidateGreedyTokenBuffer(buffer); err != nil {
		return hipDeviceByteBuffer{}, err
	}
	return hipBorrowDeviceByteBufferValue(buffer.driver, "attached drafter assistant greedy token view", buffer.Pointer(), buffer.SizeBytes(), buffer.Count()), nil
}

func hipReadAttachedDrafterGreedyTokenViews(driver nativeHIPDriver, views []hipDeviceByteBuffer, vocabSize int) ([]int32, error) {
	if len(views) == 0 {
		return nil, nil
	}
	for index := range views {
		if err := hipAttachedDrafterValidateGreedyTokenBuffer(&views[index]); err != nil {
			return nil, err
		}
	}
	tokens := make([]int32, len(views))
	base := views[0].Pointer()
	contiguous := base != 0
	for index := range views {
		want := base + nativeDevicePointer(index*hipMLXQ4ProjectionBestBytes)
		if views[index].Pointer() != want {
			contiguous = false
			break
		}
	}
	if contiguous {
		payload := make([]byte, len(views)*hipMLXQ4ProjectionBestBytes)
		if err := driver.CopyDeviceToHost(base, payload); err != nil {
			return nil, core.E("rocm.hip.AttachedDrafterAssistantDraftBlock", "copy deferred draft tokens", err)
		}
		for index := range views {
			tokenID, err := hipUnpackGreedyBestTokenID(binary.LittleEndian.Uint32(payload[index*hipMLXQ4ProjectionBestBytes:]), vocabSize)
			if err != nil {
				return nil, err
			}
			tokens[index] = int32(tokenID)
		}
		return tokens, nil
	}
	for index := range views {
		packedLow, err := hipReadDeviceUint32(driver, views[index].Pointer())
		if err != nil {
			return nil, core.E("rocm.hip.AttachedDrafterAssistantDraftBlock", "copy deferred draft token", err)
		}
		tokenID, err := hipUnpackGreedyBestTokenID(packedLow, vocabSize)
		if err != nil {
			return nil, err
		}
		tokens[index] = int32(tokenID)
	}
	return tokens, nil
}

func hipRunAttachedDrafterTargetVerifyBlock(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterTargetVerifyBlockRequest) (hipAttachedDrafterTargetVerifyBlockResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipAttachedDrafterTargetVerifyBlockResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipAttachedDrafterTargetVerifyBlockResult{}, core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "HIP driver is not available", nil)
	}
	if len(req.DraftTokens) == 0 {
		return hipAttachedDrafterTargetVerifyBlockResult{}, core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "draft tokens are required", nil)
	}
	if req.TargetDeviceState == nil || req.TargetDeviceState.closed {
		return hipAttachedDrafterTargetVerifyBlockResult{}, core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "target device KV state is required", nil)
	}
	if int32(req.CurrentGreedy.TokenID) != req.DraftTokens[0] {
		return hipAttachedDrafterTargetVerifyBlockResult{
			AcceptedCount: 0,
			RejectedCount: len(req.DraftTokens),
			Replacement:   req.CurrentGreedy,
			NextGreedy:    req.CurrentGreedy,
		}, nil
	}
	if len(req.DraftTokens) == 1 {
		return hipRunAttachedDrafterTargetVerifyLeadTokenCompact(ctx, driver, req, hipAttachedDrafterTargetVerifyBlockResult{
			AcceptedCount: 1,
			AllAccepted:   true,
		})
	}
	priorLayerKV := hipGemma4Q4DeviceLayerCaches(req.TargetDeviceState, nil, len(req.TargetForward.Layers))
	priorLayerDescriptors, err := hipGemma4Q4DeviceLayerDescriptorTableAliases(req.TargetDeviceState, nil, len(req.TargetForward.Layers))
	if err != nil {
		return hipAttachedDrafterTargetVerifyBlockResult{}, err
	}
	defer hipCloseGemma4Q4DeviceLayerDescriptorTables(priorLayerDescriptors)
	forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowWithEngineConfig(ctx, driver, req.TargetForward, req.DraftTokens, req.Position, req.Epsilon, req.DeviceKVMode, priorLayerKV, priorLayerDescriptors, nil, nil, -1, nil, req.Workspace, req.EngineConfig)
	if err != nil {
		return hipAttachedDrafterTargetVerifyBlockResult{}, err
	}
	result, err := hipResolveAttachedDrafterTargetVerifyBlock(ctx, driver, req, forward)
	if err != nil {
		_ = forward.Close()
		return hipAttachedDrafterTargetVerifyBlockResult{}, err
	}
	result.TargetCalls = 1
	if result.AcceptedCount == 1 {
		if closeErr := forward.Close(); closeErr != nil {
			_ = result.Close()
			return hipAttachedDrafterTargetVerifyBlockResult{}, closeErr
		}
		return hipRunAttachedDrafterTargetVerifyLeadTokenCompact(ctx, driver, req, result)
	}
	if result.AllAccepted {
		nextState, err := hipGemma4Q4DeviceDecodeStateFromPrefillForward(forward, req.DeviceKVMode)
		closeErr := forward.Close()
		if err != nil {
			_ = result.Close()
			return hipAttachedDrafterTargetVerifyBlockResult{}, err
		}
		if closeErr != nil {
			_ = nextState.Close()
			_ = result.Close()
			return hipAttachedDrafterTargetVerifyBlockResult{}, closeErr
		}
		result.DeviceState = nextState
		return result, nil
	}
	if result.AcceptedCount == 0 {
		closeErr := forward.Close()
		if closeErr != nil {
			_ = result.Close()
			return hipAttachedDrafterTargetVerifyBlockResult{}, closeErr
		}
		return result, nil
	}
	if err := hipTruncateAttachedDrafterVerifyForwardToAcceptedPrefix(forward, priorLayerKV, result.AcceptedCount); err == nil {
		nextState, stateErr := hipGemma4Q4DeviceDecodeStateFromPrefillForward(forward, req.DeviceKVMode)
		closeErr := forward.Close()
		if stateErr != nil {
			_ = result.Close()
			return hipAttachedDrafterTargetVerifyBlockResult{}, stateErr
		}
		if closeErr != nil {
			_ = nextState.Close()
			_ = result.Close()
			return hipAttachedDrafterTargetVerifyBlockResult{}, closeErr
		}
		result.DeviceState = nextState
		return result, nil
	}
	closeErr := forward.Close()
	if closeErr != nil {
		_ = result.Close()
		return hipAttachedDrafterTargetVerifyBlockResult{}, closeErr
	}
	prefixForward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowWithEngineConfig(ctx, driver, req.TargetForward, req.DraftTokens[:result.AcceptedCount], req.Position, req.Epsilon, req.DeviceKVMode, priorLayerKV, priorLayerDescriptors, nil, nil, -1, nil, req.Workspace, req.EngineConfig)
	if err != nil {
		_ = result.Close()
		return hipAttachedDrafterTargetVerifyBlockResult{}, err
	}
	nextState, err := hipGemma4Q4DeviceDecodeStateFromPrefillForward(prefixForward, req.DeviceKVMode)
	closeErr = prefixForward.Close()
	if err != nil {
		_ = result.Close()
		return hipAttachedDrafterTargetVerifyBlockResult{}, err
	}
	if closeErr != nil {
		_ = nextState.Close()
		_ = result.Close()
		return hipAttachedDrafterTargetVerifyBlockResult{}, closeErr
	}
	result.DeviceState = nextState
	result.TargetCalls++
	return result, nil
}

func hipRunAttachedDrafterTargetVerifyLeadTokenCompact(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterTargetVerifyBlockRequest, result hipAttachedDrafterTargetVerifyBlockResult) (hipAttachedDrafterTargetVerifyBlockResult, error) {
	if result.AcceptedCount != 1 || len(req.DraftTokens) == 0 {
		_ = result.Close()
		return hipAttachedDrafterTargetVerifyBlockResult{}, core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "compact verify requires one accepted token", nil)
	}
	compact, _, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(ctx, driver, req.TargetForward, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:                 req.DraftTokens[0],
		Position:                req.Position,
		Epsilon:                 req.Epsilon,
		DeviceKVAttention:       true,
		DeviceKVMode:            req.DeviceKVMode,
		EngineConfig:            req.EngineConfig,
		PriorDeviceState:        req.TargetDeviceState,
		ReturnDeviceState:       true,
		DeviceFinalSample:       true,
		FinalGreedyBuffer:       req.GreedyBuffer,
		SuppressTokens:          req.SuppressTokens,
		AttentionWorkspace:      req.Workspace,
		OmitDebugTensors:        true,
		OmitLabels:              true,
		OmitHostState:           true,
		ReturnDeviceFinalHidden: true,
	}, false)
	if err != nil {
		_ = result.Close()
		return hipAttachedDrafterTargetVerifyBlockResult{}, err
	}
	defer hipReleaseForwardDeviceFinalHidden(&compact)
	if compact.DeviceState == nil {
		_ = result.Close()
		return hipAttachedDrafterTargetVerifyBlockResult{}, core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "compact verify forward did not return device KV state", nil)
	}
	if compact.DeviceFinalHidden == nil || compact.DeviceFinalHidden.Pointer() == 0 {
		_ = compact.DeviceState.Close()
		_ = result.Close()
		return hipAttachedDrafterTargetVerifyBlockResult{}, core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "compact verify forward did not return device hidden", nil)
	}
	hidden, err := hipCloneAttachedDrafterTargetHidden(ctx, driver, compact.DeviceFinalHidden, req.TargetForward.Layers[len(req.TargetForward.Layers)-1].HiddenSize, req.Workspace)
	if err != nil {
		_ = compact.DeviceState.Close()
		_ = result.Close()
		return hipAttachedDrafterTargetVerifyBlockResult{}, err
	}
	_ = result.DeviceHidden.Close()
	result.DeviceHidden = hidden
	result.DeviceState = compact.DeviceState
	compact.DeviceState = nil
	if result.AllAccepted {
		result.Replacement = hipGreedySampleResult{}
	} else {
		result.Replacement = compact.Greedy
	}
	result.NextGreedy = compact.Greedy
	if len(result.VerifiedGreedies) == 0 {
		result.VerifiedGreedies = []hipGreedySampleResult{compact.Greedy}
	} else {
		result.VerifiedGreedies[0] = compact.Greedy
	}
	result.PriorDeviceStateFinalized = true
	result.TargetCalls++
	return result, nil
}

func hipCloneAttachedDrafterTargetHidden(ctx context.Context, driver nativeHIPDriver, source *hipDeviceByteBuffer, count int, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, error) {
	if source == nil || source.Pointer() == 0 || count <= 0 || source.Count() != count || source.SizeBytes() != uint64(count*4) {
		return nil, core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "target hidden buffer shape mismatch", nil)
	}
	clone, err := hipAllocateByteBuffer(driver, "rocm.hip.AttachedDrafterTargetVerifyBlock", "compact target hidden clone", source.SizeBytes(), source.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = clone.Close()
		}
	}()
	if err := hipRunVectorScaleDeviceKernelOutputWithWorkspace(ctx, driver, source, 1, clone, workspace); err != nil {
		return nil, err
	}
	success = true
	return clone, nil
}

func hipTruncateAttachedDrafterVerifyForwardToAcceptedPrefix(forward *hipGemma4Q4PrefillForwardBatch, priorLayerKV []*rocmDeviceKVCache, acceptedCount int) error {
	if forward == nil || acceptedCount <= 0 {
		return core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "accepted prefix is required for verify rollback", nil)
	}
	sharedSources := hipGemma4Q4PrefillForwardSharedSourceLayers(forward, nil)
	for index := range forward.Layers {
		layer := &forward.Layers[index]
		if layer.KV == nil || layer.KV.DeviceKV == nil {
			return core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "verify forward layer device KV is required", nil)
		}
		deviceKV := layer.KV.DeviceKV
		cache := deviceKV.Cache
		if cache == nil {
			return core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "verify forward layer device KV cache is required", nil)
		}
		if cache.borrowed {
			continue
		}
		priorTokens := 0
		if len(priorLayerKV) > index && priorLayerKV[index] != nil {
			priorTokens = priorLayerKV[index].TokenCount()
		}
		targetTokens := priorTokens + acceptedCount
		if cache.TokenCount() <= targetTokens {
			continue
		}
		if err := cache.truncateDeviceTokenCount(targetTokens); err != nil {
			return core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", core.Sprintf("truncate verify layer %d", index), err)
		}
		if err := deviceKV.DescriptorTable.Close(); err != nil {
			return core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", core.Sprintf("close verify layer %d descriptor table", index), err)
		}
		table, err := cache.kernelDescriptorTableLabeled("rocm.KVCache.DeviceDescriptor", "attached_drafter_verify_prefix")
		if err != nil {
			return err
		}
		launch, err := cache.KernelLaunchDescriptor(table)
		if err != nil {
			_ = table.Close()
			return err
		}
		deviceKV.DescriptorTable = table
		deviceKV.Launch = launch
	}
	if err := hipRefreshAttachedDrafterVerifySharedAliases(forward, sharedSources); err != nil {
		return err
	}
	return nil
}

func hipRefreshAttachedDrafterVerifySharedAliases(forward *hipGemma4Q4PrefillForwardBatch, sharedSources []int) error {
	for index, sourceIndex := range sharedSources {
		if sourceIndex == index {
			continue
		}
		if sourceIndex < 0 || sourceIndex >= index {
			return core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", core.Sprintf("verify shared layer %d source is unavailable", index), nil)
		}
		layer := &forward.Layers[index]
		source := &forward.Layers[sourceIndex]
		if layer.KV == nil || layer.KV.DeviceKV == nil || source.KV == nil || source.KV.DeviceKV == nil {
			return core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", core.Sprintf("verify shared layer %d device KV is unavailable", index), nil)
		}
		deviceKV := layer.KV.DeviceKV
		sourceKV := source.KV.DeviceKV
		if sourceKV.Cache == nil || sourceKV.DescriptorTable == nil {
			return core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", core.Sprintf("verify shared source layer %d device KV is unavailable", sourceIndex), nil)
		}
		cache, err := sourceKV.Cache.borrowedAlias()
		if err != nil {
			return err
		}
		table, err := sourceKV.DescriptorTable.borrowedAlias()
		if err != nil {
			_ = cache.Close()
			return err
		}
		launch, err := cache.KernelLaunchDescriptor(table)
		if err != nil {
			_ = table.Close()
			_ = cache.Close()
			return err
		}
		_ = deviceKV.DescriptorTable.Close()
		_ = deviceKV.Cache.Close()
		deviceKV.Cache = cache
		deviceKV.DescriptorTable = table
		deviceKV.Launch = launch
		deviceKV.RetainWindow = sourceKV.RetainWindow
	}
	return nil
}

func hipResolveAttachedDrafterTargetVerifyBlock(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterTargetVerifyBlockRequest, forward *hipGemma4Q4PrefillForwardBatch) (hipAttachedDrafterTargetVerifyBlockResult, error) {
	if forward == nil || forward.FinalHidden == nil || forward.FinalHidden.Pointer() == 0 {
		return hipAttachedDrafterTargetVerifyBlockResult{}, core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "target verify forward hidden is required", nil)
	}
	if len(req.TargetForward.Layers) == 0 {
		return hipAttachedDrafterTargetVerifyBlockResult{}, core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "target forward layers are required", nil)
	}
	last := req.TargetForward.Layers[len(req.TargetForward.Layers)-1]
	targetToken := int32(req.CurrentGreedy.TokenID)
	if targetToken != req.DraftTokens[0] {
		return hipAttachedDrafterTargetVerifyBlockResult{
			AcceptedCount: 0,
			RejectedCount: len(req.DraftTokens),
			Replacement:   req.CurrentGreedy,
			NextGreedy:    req.CurrentGreedy,
		}, nil
	}
	greedyBuffer, err := hipAttachedDrafterTargetVerifyGreedyBuffer(driver, req)
	if err != nil {
		return hipAttachedDrafterTargetVerifyBlockResult{}, err
	}
	firstGreedy, err := hipRunGemma4Q4PrefillFinalGreedyForRowSuppressWorkspace(ctx, driver, last, forward.FinalHidden, len(req.DraftTokens), 0, req.Epsilon, greedyBuffer, req.SuppressTokens, req.Workspace)
	if err != nil {
		return hipAttachedDrafterTargetVerifyBlockResult{}, err
	}
	rows := make([]hipGreedySampleResult, 0, len(req.DraftTokens))
	rows = append(rows, firstGreedy)
	accepted := 1
	targetToken = int32(firstGreedy.TokenID)
	if len(req.DraftTokens) > 1 && targetToken == req.DraftTokens[1] {
		remainingRows := len(req.DraftTokens) - 1
		var suffixGreedies []hipGreedySampleResult
		if remainingRows >= hipAttachedDrafterTargetVerifyBatchSuffixMinRows {
			suffixHidden := hipBorrowDeviceByteBufferValue(driver, "attached drafter target verify suffix hidden rows", forward.FinalHidden.Pointer()+nativeDevicePointer(last.HiddenSize*4), uint64(remainingRows*last.HiddenSize*4), remainingRows*last.HiddenSize)
			suffixGreedies, err = hipRunGemma4Q4PrefillFinalGreedyBatchSuppressWorkspace(ctx, driver, last, &suffixHidden, remainingRows, req.Epsilon, req.SuppressTokens, req.Workspace)
			if err != nil {
				return hipAttachedDrafterTargetVerifyBlockResult{}, err
			}
		} else {
			greedy, err := hipRunGemma4Q4PrefillFinalGreedyForRowSuppressWorkspace(ctx, driver, last, forward.FinalHidden, len(req.DraftTokens), 1, req.Epsilon, greedyBuffer, req.SuppressTokens, req.Workspace)
			if err != nil {
				return hipAttachedDrafterTargetVerifyBlockResult{}, err
			}
			suffixGreedies = []hipGreedySampleResult{greedy}
		}
		for index := 1; index < len(req.DraftTokens); index++ {
			if targetToken != req.DraftTokens[index] {
				break
			}
			suffixIndex := index - 1
			if suffixIndex >= len(suffixGreedies) {
				return hipAttachedDrafterTargetVerifyBlockResult{}, core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "target verify greedy suffix result is incomplete", nil)
			}
			rows = append(rows, suffixGreedies[suffixIndex])
			targetToken = int32(suffixGreedies[suffixIndex].TokenID)
			accepted++
		}
	}
	result := hipAttachedDrafterTargetVerifyBlockResult{
		AcceptedCount:    accepted,
		RejectedCount:    len(req.DraftTokens) - accepted,
		VerifiedGreedies: rows,
	}
	if accepted == len(req.DraftTokens) {
		result.AllAccepted = true
		result.NextGreedy = rows[len(rows)-1]
		hidden, err := hipCloneGemma4Q4PrefillFinalHiddenRow(ctx, driver, forward.FinalHidden, len(req.DraftTokens), len(req.DraftTokens)-1, last.HiddenSize, req.Workspace)
		if err != nil {
			return hipAttachedDrafterTargetVerifyBlockResult{}, err
		}
		result.DeviceHidden = hidden
		return result, nil
	}
	if accepted == 0 {
		result.Replacement = req.CurrentGreedy
		result.NextGreedy = req.CurrentGreedy
		return result, nil
	}
	result.Replacement = rows[accepted-1]
	result.NextGreedy = result.Replacement
	hidden, err := hipCloneGemma4Q4PrefillFinalHiddenRow(ctx, driver, forward.FinalHidden, len(req.DraftTokens), accepted-1, last.HiddenSize, req.Workspace)
	if err != nil {
		return hipAttachedDrafterTargetVerifyBlockResult{}, err
	}
	result.DeviceHidden = hidden
	return result, nil
}

func hipAttachedDrafterTargetVerifyGreedyBuffer(driver nativeHIPDriver, req hipAttachedDrafterTargetVerifyBlockRequest) (*hipDeviceByteBuffer, error) {
	if req.GreedyBuffer != nil {
		return req.GreedyBuffer, nil
	}
	if req.Workspace != nil {
		return req.Workspace.BorrowProjectionGreedyBest(driver)
	}
	return nil, nil
}

func hipCloneGemma4Q4PrefillFinalHiddenRow(ctx context.Context, driver nativeHIPDriver, hidden *hipDeviceByteBuffer, tokenCount, row, hiddenSize int, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, error) {
	if hiddenSize <= 0 || tokenCount <= 0 || row < 0 || row >= tokenCount {
		return nil, core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "target hidden row shape is invalid", nil)
	}
	if hidden == nil || hidden.Pointer() == 0 || hidden.Count() != tokenCount*hiddenSize || hidden.SizeBytes() != uint64(hidden.Count()*4) {
		return nil, core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "target hidden batch shape mismatch", nil)
	}
	rowOffset := nativeDevicePointer(row * hiddenSize * 4)
	rowView := hipBorrowDeviceByteBufferValue(driver, "attached drafter target verify hidden row", hidden.Pointer()+rowOffset, uint64(hiddenSize*4), hiddenSize)
	clone, err := hipAllocateByteBuffer(driver, "rocm.hip.AttachedDrafterTargetVerifyBlock", "target verify hidden row clone", rowView.SizeBytes(), rowView.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = clone.Close()
		}
	}()
	if err := hipRunVectorScaleDeviceKernelOutputWithWorkspace(ctx, driver, &rowView, 1, clone, workspace); err != nil {
		return nil, err
	}
	success = true
	return clone, nil
}

func hipGemma4Q4DeviceLayerDescriptorTableAliases(state *hipGemma4Q4DeviceDecodeState, scratch []*rocmDeviceKVDescriptorTable, layerCount int) ([]*rocmDeviceKVDescriptorTable, error) {
	tables := hipGemma4Q4DeviceLayerDescriptorTables(state, scratch, layerCount)
	success := false
	defer func() {
		if !success {
			hipCloseGemma4Q4DeviceLayerDescriptorTables(tables)
		}
	}()
	for index, table := range tables {
		if table == nil {
			continue
		}
		alias, err := table.borrowedAlias()
		if err != nil {
			return nil, err
		}
		tables[index] = alias
	}
	success = true
	return tables, nil
}

func hipCloseGemma4Q4DeviceLayerDescriptorTables(tables []*rocmDeviceKVDescriptorTable) {
	for _, table := range tables {
		_ = table.Close()
	}
}
