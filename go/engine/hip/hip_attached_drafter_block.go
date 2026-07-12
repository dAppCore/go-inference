// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"

	core "dappco.re/go"
)

const hipAttachedDrafterVerifiedMaxDraftTokens = 6

// KQ8VQ4 verifies the whole draft block in one batched target forward and
// truncates the device KV to the accepted prefix (the row-interleaved trim the
// descriptor-append kernel supports), so the chunk size is the max draft block:
// blocks within the cap take the batched path, only an over-cap block chunks.
const hipAttachedDrafterKQ8VQ4VerifiedChunkTokens = hipAttachedDrafterVerifiedMaxDraftTokens

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
		resolved = maxProposals
	}
	if resolved > hipAttachedDrafterVerifiedMaxDraftTokens {
		return hipAttachedDrafterVerifiedMaxDraftTokens
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

func hipAttachedDrafterTargetVerifyChunkTokens(req hipAttachedDrafterTargetVerifyBlockRequest) int {
	mode := firstNonEmptyString(req.DeviceKVMode, req.EngineConfig.DeviceKVMode)
	normalized, ok := normalizeROCmDeviceKVMode(mode)
	if !ok {
		return 0
	}
	if normalized == rocmKVCacheModeKQ8VQ4 {
		return hipAttachedDrafterKQ8VQ4VerifiedChunkTokens
	}
	return 0
}

func hipAttachedDrafterAssistantSeenPosition(nextPosition int) int {
	if nextPosition <= 0 {
		return 0
	}
	return nextPosition - 1
}

func hipAttachedDrafterStableDeviceBufferView(buffer *hipDeviceByteBuffer) *hipDeviceByteBuffer {
	if buffer == nil {
		return nil
	}
	return hipBorrowDeviceByteBuffer(
		buffer.driver,
		firstNonEmptyString(buffer.label, "attached drafter stable device view"),
		buffer.Pointer(),
		buffer.SizeBytes(),
		buffer.Count(),
	)
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
		draftPosition := hipAttachedDrafterAssistantSeenPosition(req.Position + len(tokens))
		if useDeviceTokenChain {
			proposal, err := hipRunAttachedDrafterAssistantDraftStepDeviceToken(ctx, driver, hipAttachedDrafterAssistantDraftStepProposalRequest{
				LastToken:         currentToken,
				LastGreedyToken:   currentGreedyToken,
				TargetHidden:      currentHidden,
				TargetForward:     req.TargetForward,
				TargetDeviceState: req.TargetDeviceState,
				Plan:              req.Plan,
				InputPlan:         req.InputPlan,
				Position:          draftPosition,
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
			Position:          draftPosition,
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
	req.GreedyBuffer = hipAttachedDrafterStableDeviceBufferView(req.GreedyBuffer)
	if int32(req.CurrentGreedy.TokenID) != req.DraftTokens[0] {
		return hipAttachedDrafterTargetVerifyBlockResult{
			AcceptedCount: 0,
			RejectedCount: len(req.DraftTokens),
			Replacement:   req.CurrentGreedy,
			NextGreedy:    req.CurrentGreedy,
		}, nil
	}
	if len(req.DraftTokens) == 1 {
		return hipRunAttachedDrafterTargetVerifyLeadTokenBatch(ctx, driver, req, hipAttachedDrafterTargetVerifyBlockResult{
			AcceptedCount: 1,
			AllAccepted:   true,
		})
	}
	if chunkTokens := hipAttachedDrafterTargetVerifyChunkTokens(req); chunkTokens > 0 && len(req.DraftTokens) > chunkTokens {
		return hipRunAttachedDrafterTargetVerifyBlockBatchedChunks(ctx, driver, req, chunkTokens)
	}
	return hipRunAttachedDrafterTargetVerifyBlockBatched(ctx, driver, req)
}

func hipRunAttachedDrafterTargetVerifyBlockBatchedChunks(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterTargetVerifyBlockRequest, chunkTokens int) (hipAttachedDrafterTargetVerifyBlockResult, error) {
	if chunkTokens <= 0 {
		return hipAttachedDrafterTargetVerifyBlockResult{}, core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "verify chunk size must be positive", nil)
	}
	currentGreedy := req.CurrentGreedy
	currentDeviceState := req.TargetDeviceState
	var latestDeviceState *hipGemma4Q4DeviceDecodeState
	var latestHidden *hipDeviceByteBuffer
	verified := make([]hipGreedySampleResult, 0, len(req.DraftTokens))
	targetCalls := 0
	priorFinalized := false
	success := false
	defer func() {
		if success {
			return
		}
		_ = latestDeviceState.Close()
		_ = latestHidden.Close()
	}()
	for cursor := 0; cursor < len(req.DraftTokens); {
		if int32(currentGreedy.TokenID) != req.DraftTokens[cursor] {
			success = true
			return hipAttachedDrafterTargetVerifyBlockResult{
				AcceptedCount:             cursor,
				RejectedCount:             len(req.DraftTokens) - cursor,
				Replacement:               currentGreedy,
				NextGreedy:                currentGreedy,
				DeviceState:               latestDeviceState,
				DeviceHidden:              latestHidden,
				PriorDeviceStateFinalized: priorFinalized,
				TargetCalls:               targetCalls,
				VerifiedGreedies:          verified,
			}, nil
		}
		end := cursor + chunkTokens
		if end > len(req.DraftTokens) {
			end = len(req.DraftTokens)
		}
		chunkReq := req
		chunkReq.TargetDeviceState = currentDeviceState
		chunkReq.CurrentGreedy = currentGreedy
		chunkReq.DraftTokens = req.DraftTokens[cursor:end]
		chunkReq.Position = req.Position + cursor
		var chunk hipAttachedDrafterTargetVerifyBlockResult
		var err error
		if len(chunkReq.DraftTokens) == 1 {
			chunk, err = hipRunAttachedDrafterTargetVerifyLeadTokenBatch(ctx, driver, chunkReq, hipAttachedDrafterTargetVerifyBlockResult{
				AcceptedCount: 1,
				AllAccepted:   true,
			})
		} else {
			chunk, err = hipRunAttachedDrafterTargetVerifyBlockBatched(ctx, driver, chunkReq)
		}
		if err != nil {
			return hipAttachedDrafterTargetVerifyBlockResult{}, err
		}
		targetCalls += chunk.TargetCalls
		verified = append(verified, chunk.VerifiedGreedies...)
		chunkAccepted := chunk.AcceptedCount
		if chunk.DeviceState != nil {
			previousDeviceState := currentDeviceState
			if !chunk.PriorDeviceStateFinalized {
				if err := hipFinalizeGemma4Q4ForwardDeviceState(previousDeviceState, chunk.DeviceState); err != nil {
					_ = chunk.Close()
					return hipAttachedDrafterTargetVerifyBlockResult{}, err
				}
			}
			if previousDeviceState == req.TargetDeviceState {
				priorFinalized = true
			} else {
				hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
			}
			currentDeviceState = chunk.DeviceState
			latestDeviceState = chunk.DeviceState
			chunk.DeviceState = nil
		}
		if chunk.DeviceHidden != nil {
			_ = latestHidden.Close()
			latestHidden = chunk.DeviceHidden
			chunk.DeviceHidden = nil
		}
		currentGreedy = chunk.NextGreedy
		if chunkAccepted == 0 || !chunk.AllAccepted {
			success = true
			result := hipAttachedDrafterTargetVerifyBlockResult{
				AcceptedCount:             cursor + chunkAccepted,
				RejectedCount:             len(req.DraftTokens) - cursor - chunkAccepted,
				Replacement:               chunk.Replacement,
				NextGreedy:                chunk.NextGreedy,
				DeviceState:               latestDeviceState,
				DeviceHidden:              latestHidden,
				PriorDeviceStateFinalized: priorFinalized,
				TargetCalls:               targetCalls,
				VerifiedGreedies:          verified,
			}
			if err := chunk.Close(); err != nil {
				_ = result.Close()
				return hipAttachedDrafterTargetVerifyBlockResult{}, err
			}
			return result, nil
		}
		if err := chunk.Close(); err != nil {
			return hipAttachedDrafterTargetVerifyBlockResult{}, err
		}
		cursor += chunkAccepted
	}
	success = true
	return hipAttachedDrafterTargetVerifyBlockResult{
		AcceptedCount:             len(req.DraftTokens),
		RejectedCount:             0,
		NextGreedy:                currentGreedy,
		AllAccepted:               true,
		DeviceState:               latestDeviceState,
		DeviceHidden:              latestHidden,
		PriorDeviceStateFinalized: priorFinalized,
		TargetCalls:               targetCalls,
		VerifiedGreedies:          verified,
	}, nil
}

func hipRunAttachedDrafterTargetVerifyBlockCompactSequence(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterTargetVerifyBlockRequest) (hipAttachedDrafterTargetVerifyBlockResult, error) {
	currentGreedy := req.CurrentGreedy
	currentDeviceState := req.TargetDeviceState
	var latestDeviceState *hipGemma4Q4DeviceDecodeState
	var latestHidden *hipDeviceByteBuffer
	verified := make([]hipGreedySampleResult, 0, len(req.DraftTokens))
	targetCalls := 0
	success := false
	defer func() {
		if success {
			return
		}
		_ = latestDeviceState.Close()
		_ = latestHidden.Close()
	}()
	for index, draft := range req.DraftTokens {
		if int32(currentGreedy.TokenID) != draft {
			success = true
			return hipAttachedDrafterTargetVerifyBlockResult{
				AcceptedCount:             index,
				RejectedCount:             len(req.DraftTokens) - index,
				Replacement:               currentGreedy,
				NextGreedy:                currentGreedy,
				DeviceState:               latestDeviceState,
				DeviceHidden:              latestHidden,
				PriorDeviceStateFinalized: targetCalls > 0,
				TargetCalls:               targetCalls,
				VerifiedGreedies:          verified,
			}, nil
		}
		stepReq := req
		stepReq.TargetDeviceState = currentDeviceState
		stepReq.CurrentGreedy = currentGreedy
		stepReq.DraftTokens = req.DraftTokens[index : index+1]
		stepReq.Position = req.Position + index
		step, err := hipRunAttachedDrafterTargetVerifyLeadTokenCompact(ctx, driver, stepReq, hipAttachedDrafterTargetVerifyBlockResult{
			AcceptedCount: 1,
			AllAccepted:   true,
		})
		if err != nil {
			return hipAttachedDrafterTargetVerifyBlockResult{}, err
		}
		if currentDeviceState != req.TargetDeviceState {
			hipReleaseClosedGemma4Q4DeviceDecodeState(currentDeviceState)
		}
		_ = latestHidden.Close()
		latestDeviceState = step.DeviceState
		latestHidden = step.DeviceHidden
		step.DeviceState = nil
		step.DeviceHidden = nil
		currentGreedy = step.NextGreedy
		currentDeviceState = latestDeviceState
		targetCalls += step.TargetCalls
		verified = append(verified, step.NextGreedy)
		if err := step.Close(); err != nil {
			return hipAttachedDrafterTargetVerifyBlockResult{}, err
		}
	}
	success = true
	return hipAttachedDrafterTargetVerifyBlockResult{
		AcceptedCount:             len(req.DraftTokens),
		RejectedCount:             0,
		NextGreedy:                currentGreedy,
		AllAccepted:               true,
		DeviceState:               latestDeviceState,
		DeviceHidden:              latestHidden,
		PriorDeviceStateFinalized: targetCalls > 0,
		TargetCalls:               targetCalls,
		VerifiedGreedies:          verified,
	}, nil
}

func hipRunAttachedDrafterTargetVerifyBlockBatched(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterTargetVerifyBlockRequest) (hipAttachedDrafterTargetVerifyBlockResult, error) {
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
	// Partial acceptance: the batched forward already holds the accepted-prefix
	// hidden and KV, so truncate the device KV to the accepted prefix instead of
	// re-running the accepted tokens through a single-token forward. The prior
	// device state stays live (unfinalized) as the accepted-prefix rollback base.
	if err := hipTruncateAttachedDrafterVerifyForwardToAcceptedPrefix(forward, priorLayerKV, result.AcceptedCount); err != nil {
		_ = forward.Close()
		_ = result.Close()
		return hipAttachedDrafterTargetVerifyBlockResult{}, err
	}
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

func hipRunAttachedDrafterTargetVerifyLeadTokenBatch(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterTargetVerifyBlockRequest, result hipAttachedDrafterTargetVerifyBlockResult) (hipAttachedDrafterTargetVerifyBlockResult, error) {
	if result.AcceptedCount != 1 || len(req.DraftTokens) == 0 {
		_ = result.Close()
		return hipAttachedDrafterTargetVerifyBlockResult{}, core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "batch verify requires one accepted token", nil)
	}
	advanced, err := hipRunAttachedDrafterTargetAdvanceOneBatch(ctx, driver, hipAttachedDrafterTargetAdvanceOneRequest{
		TargetForward:    req.TargetForward,
		DeviceKVMode:     req.DeviceKVMode,
		EngineConfig:     req.EngineConfig,
		PriorDeviceState: req.TargetDeviceState,
		TokenID:          req.DraftTokens[0],
		Position:         req.Position,
		Epsilon:          req.Epsilon,
		SuppressTokens:   req.SuppressTokens,
		GreedyBuffer:     req.GreedyBuffer,
		Workspace:        req.Workspace,
		ReturnHidden:     true,
	})
	if err != nil {
		_ = result.Close()
		return hipAttachedDrafterTargetVerifyBlockResult{}, err
	}
	_ = result.DeviceHidden.Close()
	_ = result.DeviceState.Close()
	result.DeviceHidden = advanced.Current.DeviceFinalHidden
	advanced.Current.DeviceFinalHidden = nil
	advanced.Current.DeviceFinalHiddenBorrowed = false
	result.DeviceState = advanced.DeviceState
	advanced.DeviceState = nil
	if result.AllAccepted {
		result.Replacement = hipGreedySampleResult{}
	} else {
		result.Replacement = advanced.Current.Greedy
	}
	result.NextGreedy = advanced.Current.Greedy
	if len(result.VerifiedGreedies) == 0 {
		result.VerifiedGreedies = []hipGreedySampleResult{advanced.Current.Greedy}
	} else {
		result.VerifiedGreedies[0] = advanced.Current.Greedy
	}
	result.PriorDeviceStateFinalized = true
	result.TargetCalls += advanced.TargetCalls
	if err := advanced.Close(); err != nil {
		_ = result.Close()
		return hipAttachedDrafterTargetVerifyBlockResult{}, err
	}
	return result, nil
}

func hipRunAttachedDrafterTargetVerifyLeadTokenCompact(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterTargetVerifyBlockRequest, result hipAttachedDrafterTargetVerifyBlockResult) (hipAttachedDrafterTargetVerifyBlockResult, error) {
	if result.AcceptedCount != 1 || len(req.DraftTokens) == 0 {
		_ = result.Close()
		return hipAttachedDrafterTargetVerifyBlockResult{}, core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "compact verify requires one accepted token", nil)
	}
	greedyBuffer := hipAttachedDrafterStableDeviceBufferView(req.GreedyBuffer)
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
		FinalGreedyBuffer:       greedyBuffer,
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
	allGreedies, err := hipRunGemma4Q4PrefillFinalGreedyBatchSuppressWorkspace(ctx, driver, last, forward.FinalHidden, len(req.DraftTokens), req.Epsilon, req.SuppressTokens, req.Workspace)
	if err != nil {
		return hipAttachedDrafterTargetVerifyBlockResult{}, err
	}
	rows := make([]hipGreedySampleResult, 0, len(req.DraftTokens))
	accepted := 0
	for index, draftToken := range req.DraftTokens {
		if targetToken != draftToken {
			break
		}
		if index >= len(allGreedies) {
			return hipAttachedDrafterTargetVerifyBlockResult{}, core.E("rocm.hip.AttachedDrafterTargetVerifyBlock", "target verify greedy batch result is incomplete", nil)
		}
		greedy := allGreedies[index]
		rows = append(rows, greedy)
		targetToken = int32(greedy.TokenID)
		accepted++
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
