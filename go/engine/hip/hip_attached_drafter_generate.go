// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	inferdecode "dappco.re/go/inference/decode"
)

type hipAttachedDrafterRuntime struct {
	attachment    AttachedDrafterAttachment
	draft         *hipLoadedModel
	assistantPlan hipAttachedDrafterAssistantVerifierPlan
	inputPlan     hipAttachedDrafterAssistantDraftStepInputPlan
	softcap       float32
}

type hipAttachedDrafterGenerateRequest struct {
	InputTokenIDs       []int32
	InputText           string
	MaxTokens           int
	DraftTokens         int
	AdaptiveDraftTokens bool
	Temperature         float32
	TopK                int
	TopP                float32
	MinP                float32
	StopTokens          []int32
	RepeatPenalty       float32
	InitialDeviceState  *hipGemma4Q4DeviceDecodeState
	RetainDeviceState   func(*hipGemma4Q4DeviceDecodeState) error
	RestoreDeviceState  func(*hipGemma4Q4DeviceDecodeState) error
}

func attachedDrafterAttachError(linked bool, targetRetainedDecode, assistantVerify, assistantPreflightStatus, assistantPlanStatus string) error {
	if linked {
		return nil
	}
	return core.E("rocm.hip.AttachAttachedDrafter", core.Sprintf("native HIP drafter attachment is not linked yet (target retained decode %s; assistant verify %s; assistant preflight %s; assistant plan %s)", targetRetainedDecode, assistantVerify, assistantPreflightStatus, assistantPlanStatus), nil)
}

func (model *hipLoadedModel) storeAttachedDrafterRuntime(runtime *hipAttachedDrafterRuntime) {
	if model == nil {
		return
	}
	model.attachedDrafterMu.Lock()
	defer model.attachedDrafterMu.Unlock()
	model.attachedDrafter = runtime
}

func (model *hipLoadedModel) attachedDrafterRuntimeSnapshot() (*hipAttachedDrafterRuntime, error) {
	if model == nil {
		return nil, core.E("rocm.hip.AttachedDrafterGenerate", "target model is nil", nil)
	}
	model.attachedDrafterMu.Lock()
	defer model.attachedDrafterMu.Unlock()
	if model.attachedDrafter == nil {
		return nil, core.E("rocm.hip.AttachedDrafterGenerate", "native HIP drafter attachment is not linked for this target runtime", nil)
	}
	runtime := *model.attachedDrafter
	runtime.attachment = cloneAttachedDrafterAttachment(runtime.attachment)
	runtime.assistantPlan.Layers = append([]hipAttachedDrafterAssistantVerifierLayerPlan(nil), runtime.assistantPlan.Layers...)
	runtime.assistantPlan.KernelFamilies = append([]string(nil), runtime.assistantPlan.KernelFamilies...)
	runtime.inputPlan.KernelFamilies = append([]string(nil), runtime.inputPlan.KernelFamilies...)
	return &runtime, nil
}

func (model *hipLoadedModel) GenerateAttachedDrafter(ctx context.Context, attachment AttachedDrafterAttachment, prompt string, cfg AttachedDrafterGenerateConfig) (inferdecode.Result, error) {
	runtime, err := model.attachedDrafterRuntimeSnapshot()
	if err != nil {
		return inferdecode.Result{}, err
	}
	if attachment.NativeAttachment != hipKernelStatusLinked {
		return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerate", "linked native attachment is required", nil)
	}
	inputTokens, err := hipGemma4Q4PromptTokenIDsRequired(prompt, model)
	if err != nil {
		return inferdecode.Result{}, err
	}
	return model.runAttachedDrafterGenerate(ctx, runtime, hipAttachedDrafterGenerateRequest{
		InputTokenIDs:       inputTokens,
		InputText:           prompt,
		MaxTokens:           cfg.MaxTokens,
		DraftTokens:         cfg.DraftTokens,
		AdaptiveDraftTokens: cfg.AdaptiveDraftTokens,
		Temperature:         cfg.Temperature,
		TopK:                cfg.TopK,
		TopP:                cfg.TopP,
		MinP:                cfg.MinP,
		StopTokens:          append([]int32(nil), cfg.StopTokens...),
		RepeatPenalty:       cfg.RepeatPenalty,
	})
}

func (model *hipLoadedModel) GenerateAttachedDrafterWithStateRetention(ctx context.Context, attachment AttachedDrafterAttachment, prompt string, cfg AttachedDrafterGenerateConfig, state *StateSession) (inferdecode.Result, error) {
	runtime, err := model.attachedDrafterRuntimeSnapshot()
	if err != nil {
		return inferdecode.Result{}, err
	}
	if attachment.NativeAttachment != hipKernelStatusLinked {
		return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerateWithStateRetention", "linked native attachment is required", nil)
	}
	if state == nil {
		return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerateWithStateRetention", "state session is required", nil)
	}
	inputTokens, err := hipGemma4Q4PromptTokenIDsRequired(prompt, model)
	if err != nil {
		return inferdecode.Result{}, err
	}
	targetCfg, err := model.attachedDrafterTargetForwardConfig()
	if err != nil {
		return inferdecode.Result{}, err
	}
	deviceState, err := state.takeGemma4Q4DeviceDecodeState(model.driver, targetCfg)
	if err != nil {
		return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerateWithStateRetention", "restore retained Gemma4 q4 device state", err)
	}
	result, err := model.runAttachedDrafterGenerate(ctx, runtime, hipAttachedDrafterGenerateRequest{
		InputTokenIDs:       inputTokens,
		InputText:           prompt,
		MaxTokens:           cfg.MaxTokens,
		DraftTokens:         cfg.DraftTokens,
		AdaptiveDraftTokens: cfg.AdaptiveDraftTokens,
		Temperature:         cfg.Temperature,
		TopK:                cfg.TopK,
		TopP:                cfg.TopP,
		MinP:                cfg.MinP,
		StopTokens:          append([]int32(nil), cfg.StopTokens...),
		RepeatPenalty:       cfg.RepeatPenalty,
		InitialDeviceState:  deviceState,
		RetainDeviceState: func(stateKV *hipGemma4Q4DeviceDecodeState) error {
			return state.replaceRuntime(stateKV)
		},
		RestoreDeviceState: func(stateKV *hipGemma4Q4DeviceDecodeState) error {
			return state.replaceRuntime(stateKV)
		},
	})
	if err != nil {
		return inferdecode.Result{}, err
	}
	return result, nil
}

func (model *hipLoadedModel) GenerateAttachedDrafterFromState(ctx context.Context, attachment AttachedDrafterAttachment, req AttachedDrafterStateGenerateRequest) (inferdecode.Result, error) {
	runtime, err := model.attachedDrafterRuntimeSnapshot()
	if err != nil {
		return inferdecode.Result{}, err
	}
	if attachment.NativeAttachment != hipKernelStatusLinked {
		return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerateFromState", "linked native attachment is required", nil)
	}
	if req.State == nil {
		return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerateFromState", "runtime-owned KV state is required", nil)
	}
	targetCfg, err := model.attachedDrafterTargetForwardConfig()
	if err != nil {
		return inferdecode.Result{}, err
	}
	deviceState, err := req.State.takeGemma4Q4DeviceDecodeState(model.driver, targetCfg)
	if err != nil {
		return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerateFromState", "restore retained Gemma4 q4 device state", err)
	}
	if deviceState == nil {
		return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerateFromState", "Gemma4 q4 device KV state is required; refusing prompt replay", nil)
	}
	inputTokens, err := hipGemma4Q4PromptTokenIDsRequired(req.Input, model)
	if err != nil {
		_ = req.State.replaceRuntime(deviceState)
		return inferdecode.Result{}, err
	}
	result, err := model.runAttachedDrafterGenerate(ctx, runtime, hipAttachedDrafterGenerateRequest{
		InputTokenIDs:       inputTokens,
		InputText:           req.Input,
		MaxTokens:           req.MaxTokens,
		DraftTokens:         req.DraftTokens,
		AdaptiveDraftTokens: req.AdaptiveDraftTokens,
		Temperature:         req.Temperature,
		TopK:                req.TopK,
		TopP:                req.TopP,
		MinP:                req.MinP,
		StopTokens:          append([]int32(nil), req.StopTokens...),
		RepeatPenalty:       req.RepeatPenalty,
		InitialDeviceState:  deviceState,
		RetainDeviceState: func(state *hipGemma4Q4DeviceDecodeState) error {
			return req.State.replaceRuntime(state)
		},
		RestoreDeviceState: func(state *hipGemma4Q4DeviceDecodeState) error {
			return req.State.replaceRuntime(state)
		},
	})
	if err != nil {
		return inferdecode.Result{}, err
	}
	return result, nil
}

func (model *hipLoadedModel) attachedDrafterTargetForwardConfig() (hipGemma4Q4ForwardConfig, error) {
	if model == nil {
		return hipGemma4Q4ForwardConfig{}, core.E("rocm.hip.AttachedDrafterGenerate", "target model is nil", nil)
	}
	if model.modelInfo.NumLayers <= 0 {
		return hipGemma4Q4ForwardConfig{}, core.E("rocm.hip.AttachedDrafterGenerate", "loaded Gemma4 q4 layer count is required", nil)
	}
	return model.cachedGemma4Q4ForwardConfig(model.modelInfo.NumLayers)
}

func hipGemma4Q4PromptTokenIDsRequired(prompt string, model *hipLoadedModel) ([]int32, error) {
	tokens, _, err := hipGemma4Q4PromptTokenIDs(prompt, model)
	if err != nil {
		return nil, err
	}
	if len(tokens) == 0 {
		return nil, core.E("rocm.hip.AttachedDrafterGenerate", "input text produced no token IDs", nil)
	}
	return tokens, nil
}

func (model *hipLoadedModel) runAttachedDrafterGenerate(ctx context.Context, runtime *hipAttachedDrafterRuntime, req hipAttachedDrafterGenerateRequest) (inferdecode.Result, error) {
	if err := hipContextErr(ctx); err != nil {
		return inferdecode.Result{}, err
	}
	if model == nil {
		return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerate", "target model is nil", nil)
	}
	if model.driver == nil || !model.driver.Available() {
		return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerate", "HIP driver is not available", nil)
	}
	if runtime == nil {
		return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerate", "attached drafter runtime is required", nil)
	}
	if req.MaxTokens <= 0 {
		return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerate", "max tokens must be positive", nil)
	}
	if len(req.InputTokenIDs) == 0 {
		return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerate", "input token IDs are required", nil)
	}
	if err := hipAttachedDrafterAssistantDraftStepProposalPlanInvalidReason(runtime.assistantPlan, runtime.softcap); err != nil {
		return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerate", "assistant proposal plan", err)
	}
	targetCfg, err := model.attachedDrafterTargetForwardConfig()
	if err != nil {
		return inferdecode.Result{}, err
	}
	generate := inference.GenerateConfig{
		MaxTokens:     req.MaxTokens,
		Temperature:   req.Temperature,
		TopK:          req.TopK,
		TopP:          req.TopP,
		MinP:          req.MinP,
		StopTokens:    append([]int32(nil), req.StopTokens...),
		RepeatPenalty: req.RepeatPenalty,
	}
	if hipGemma4Q4HostSamplingRequested(generate) {
		return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerate", "retained attached drafter currently requires greedy generation", nil)
	}
	engineConfig := model.gemma4Q4EngineConfig()
	deviceKVMode, err := engineConfig.deviceKVMode()
	if err != nil {
		return inferdecode.Result{}, err
	}
	position := 0
	deviceState := req.InitialDeviceState
	deviceStateRetained := false
	if deviceState != nil {
		if deviceState.closed {
			return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerate", "initial Gemma4 q4 device KV state is closed", nil)
		}
		if deviceState.LayerCount() != len(targetCfg.Layers) {
			return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerate", "initial Gemma4 q4 device KV layer count mismatch", nil)
		}
		position = deviceState.maxLayerTokenCount()
	}
	defer func() {
		if deviceStateRetained || deviceState == nil {
			return
		}
		if req.RestoreDeviceState != nil {
			if err := req.RestoreDeviceState(deviceState); err == nil {
				deviceStateRetained = true
				return
			}
		}
		_ = deviceState.Close()
	}()

	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(model.driver, workspace, targetCfg, position+len(req.InputTokenIDs)+req.MaxTokens+1); err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		return inferdecode.Result{}, err
	}
	workspace.EnsureProjectionGreedyBestCapacity(req.MaxTokens + 2)
	finalGreedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
	if err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		return inferdecode.Result{}, err
	}
	defer func() {
		_ = finalGreedyBuffer
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
	}()

	start := time.Now()
	var targetDuration time.Duration
	var draftDuration time.Duration
	suppressTokens := hipGemma4Q4GenerationSuppressTokenIDs(model, generate.StopTokens)
	targetStart := time.Now()
	prefill, err := hipRunAttachedDrafterTargetPrefill(ctx, model.driver, hipAttachedDrafterTargetPrefillRequest{
		TargetForward:     targetCfg,
		DeviceKVMode:      deviceKVMode,
		EngineConfig:      engineConfig,
		InputTokenIDs:     req.InputTokenIDs,
		Position:          position,
		TargetDeviceState: deviceState,
		Epsilon:           1e-6,
		SuppressTokens:    suppressTokens,
		GreedyBuffer:      finalGreedyBuffer,
		Workspace:         workspace,
	})
	targetDuration += nonZeroHIPDuration(time.Since(targetStart))
	if err != nil {
		return inferdecode.Result{}, err
	}
	state := prefill.State
	currentToken := prefill.LastToken
	current := prefill.Current
	deviceState = prefill.DeviceState
	position = prefill.Position
	targetCalls := prefill.TargetCalls

	tokens := make([]inferdecode.Token, 0, req.MaxTokens)
	var accepted, rejected, draftTokens, draftCalls int
	adaptiveDraftTokens := req.DraftTokens
	carryLead := int32(-1)
	stopped := false
	for len(tokens) < req.MaxTokens && !stopped {
		if err := hipContextErr(ctx); err != nil {
			return inferdecode.Result{}, err
		}
		remaining := req.MaxTokens - len(tokens)
		if remaining == 1 && carryLead < 0 {
			tokenID := int32(current.Greedy.TokenID)
			if hipTokenIsStop(tokenID, generate.StopTokens) {
				stopped = true
				break
			}
			tokens = append(tokens, inferdecode.Token{ID: tokenID, Text: hipGeneratedTokenText(model, tokenID)})
			currentToken = tokenID
			carryLead = tokenID
			break
		}
		blockSize := hipAttachedDrafterResolveDraftTokensForTarget(targetCfg, adaptiveDraftTokens, remaining)
		if blockSize <= 0 {
			break
		}
		draftStart := time.Now()
		draftBlock, proposalErr := hipRunAttachedDrafterAssistantDraftBlock(ctx, model.driver, hipAttachedDrafterAssistantDraftBlockRequest{
			LastToken:         currentToken,
			TargetHidden:      current.DeviceFinalHidden,
			TargetForward:     targetCfg,
			TargetDeviceState: deviceState,
			Plan:              runtime.assistantPlan,
			InputPlan:         runtime.inputPlan,
			Position:          position,
			Epsilon:           1e-6,
			Softcap:           runtime.softcap,
			SuppressTokens:    suppressTokens,
			MaxDraftTokens:    blockSize,
			Workspace:         workspace,
		})
		draftDuration += nonZeroHIPDuration(time.Since(draftStart))
		if proposalErr != nil {
			return inferdecode.Result{}, proposalErr
		}
		draftCalls++
		proposedCount := len(draftBlock.Tokens)
		draftTokens += proposedCount
		verifyTokens := draftBlock.Tokens
		carryPresent := carryLead >= 0
		if carryPresent {
			withCarry := make([]int32, 0, len(draftBlock.Tokens)+1)
			withCarry = append(withCarry, carryLead)
			withCarry = append(withCarry, draftBlock.Tokens...)
			verifyTokens = withCarry
		}
		targetStart := time.Now()
		verify, verifyErr := hipRunAttachedDrafterTargetVerifyBlock(ctx, model.driver, hipAttachedDrafterTargetVerifyBlockRequest{
			TargetForward:     targetCfg,
			DeviceKVMode:      deviceKVMode,
			EngineConfig:      engineConfig,
			TargetDeviceState: deviceState,
			CurrentGreedy:     current.Greedy,
			DraftTokens:       verifyTokens,
			Position:          position,
			Epsilon:           1e-6,
			SuppressTokens:    suppressTokens,
			GreedyBuffer:      finalGreedyBuffer,
			Workspace:         workspace,
		})
		targetDuration += nonZeroHIPDuration(time.Since(targetStart))
		if err := draftBlock.Close(); err != nil {
			return inferdecode.Result{}, err
		}
		if verifyErr != nil {
			return inferdecode.Result{}, verifyErr
		}
		targetCalls += verify.TargetCalls
		if carryPresent && verify.AcceptedCount == 0 {
			_ = verify.Close()
			return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerate", "carried target token was not accepted by verifier", nil)
		}
		acceptedFromDraft := verify.AcceptedCount
		emitStart := 0
		if carryPresent {
			acceptedFromDraft--
			emitStart = 1
		}
		if acceptedFromDraft < 0 {
			acceptedFromDraft = 0
		}
		if acceptedFromDraft > proposedCount {
			acceptedFromDraft = proposedCount
		}
		accepted += acceptedFromDraft
		if !verify.AllAccepted {
			rejected += proposedCount - acceptedFromDraft
		}
		if req.AdaptiveDraftTokens {
			adaptiveDraftTokens = hipAttachedDrafterAdaptDraftTokens(adaptiveDraftTokens, proposedCount, acceptedFromDraft)
		}
		for index := emitStart; index < verify.AcceptedCount && len(tokens) < req.MaxTokens; index++ {
			tokenID := verifyTokens[index]
			if hipTokenIsStop(tokenID, generate.StopTokens) {
				stopped = true
				break
			}
			tokens = append(tokens, inferdecode.Token{ID: tokenID, Text: hipGeneratedTokenText(model, tokenID)})
			currentToken = tokenID
		}
		if verify.DeviceState != nil {
			previousDeviceState := deviceState
			if !verify.PriorDeviceStateFinalized {
				if err := hipFinalizeGemma4Q4ForwardDeviceState(previousDeviceState, verify.DeviceState); err != nil {
					_ = verify.Close()
					return inferdecode.Result{}, err
				}
			}
			deviceState = verify.DeviceState
			verify.DeviceState = nil
			hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
			position = deviceState.maxLayerTokenCount()
		}
		if verify.DeviceHidden != nil {
			hipReleaseForwardDeviceFinalHidden(&current)
			current.DeviceFinalHidden = verify.DeviceHidden
			current.DeviceFinalHiddenBorrowed = false
			verify.DeviceHidden = nil
		}
		current.Greedy = verify.NextGreedy
		current.GreedyDevice = nil
		if stopped || len(tokens) == req.MaxTokens {
			carryLead = -1
			_ = verify.Close()
			break
		}
		if verify.AllAccepted {
			carryLead = -1
			_ = verify.Close()
			continue
		}
		replacement := int32(verify.Replacement.TokenID)
		if hipTokenIsStop(replacement, generate.StopTokens) {
			stopped = true
			carryLead = -1
			_ = verify.Close()
			break
		}
		tokens = append(tokens, inferdecode.Token{ID: replacement, Text: hipGeneratedTokenText(model, replacement)})
		currentToken = replacement
		carryLead = replacement
		_ = verify.Close()
	}
	retainCarryState := req.RetainDeviceState != nil || req.RestoreDeviceState != nil
	if carryLead >= 0 && deviceState != nil && retainCarryState {
		stepStart := time.Now()
		flush, nextState, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(ctx, model.driver, targetCfg, state, hipGemma4Q4ForwardRequest{
			TokenID:            carryLead,
			Position:           position,
			Epsilon:            1e-6,
			DeviceKVAttention:  true,
			DeviceKVMode:       deviceKVMode,
			EngineConfig:       engineConfig,
			PriorDeviceState:   deviceState,
			ReturnDeviceState:  true,
			SkipFinalSample:    true,
			AttentionWorkspace: workspace,
			OmitDebugTensors:   true,
			OmitLabels:         true,
			OmitHostState:      true,
		}, false)
		targetDuration += nonZeroHIPDuration(time.Since(stepStart))
		targetCalls++
		if err != nil {
			return inferdecode.Result{}, err
		}
		state = nextState
		if flush.DeviceState == nil {
			return inferdecode.Result{}, core.E("rocm.hip.AttachedDrafterGenerate", "carry flush did not return device KV state", nil)
		}
		previousDeviceState := deviceState
		deviceState = flush.DeviceState
		flush.DeviceState = nil
		hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
		position++
		carryLead = -1
	}
	hipReleaseForwardDeviceFinalHidden(&current)
	if req.RetainDeviceState != nil && deviceState != nil {
		if err := req.RetainDeviceState(deviceState); err != nil {
			return inferdecode.Result{}, err
		}
		deviceStateRetained = true
	}
	duration := nonZeroHIPDuration(time.Since(start))
	metrics := inferdecode.Metrics{
		TargetTokens:   len(tokens),
		DraftTokens:    draftTokens,
		AcceptedTokens: accepted,
		RejectedTokens: rejected,
		EmittedTokens:  len(tokens),
		TargetCalls:    targetCalls,
		DraftCalls:     draftCalls,
		Duration:       duration,
		TargetDuration: targetDuration,
		DraftDuration:  draftDuration,
	}
	if attempted := accepted + rejected; attempted > 0 {
		metrics.AcceptanceRate = float64(accepted) / float64(attempted)
	}
	return inferdecode.Result{
		Mode:    inferdecode.ModeSpeculative,
		Prompt:  req.InputText,
		Text:    inferdecode.TokensText(tokens),
		Tokens:  tokens,
		Metrics: metrics,
	}, nil
}

type hipAttachedDrafterTargetPrefillRequest struct {
	TargetForward     hipGemma4Q4ForwardConfig
	DeviceKVMode      string
	EngineConfig      hipGemma4Q4EngineConfig
	InputTokenIDs     []int32
	Position          int
	TargetDeviceState *hipGemma4Q4DeviceDecodeState
	Epsilon           float32
	SuppressTokens    []int32
	GreedyBuffer      *hipDeviceByteBuffer
	Workspace         *hipAttentionHeadsChunkedWorkspace
}

type hipAttachedDrafterTargetPrefillResult struct {
	Current     hipGemma4Q4ForwardResult
	State       hipGemma4Q4DecodeState
	DeviceState *hipGemma4Q4DeviceDecodeState
	Position    int
	LastToken   int32
	TargetCalls int
}

func hipRunAttachedDrafterTargetPrefill(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterTargetPrefillRequest) (hipAttachedDrafterTargetPrefillResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipAttachedDrafterTargetPrefillResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipAttachedDrafterTargetPrefillResult{}, core.E("rocm.hip.AttachedDrafterTargetPrefill", "HIP driver is not available", nil)
	}
	if len(req.InputTokenIDs) == 0 {
		return hipAttachedDrafterTargetPrefillResult{}, core.E("rocm.hip.AttachedDrafterTargetPrefill", "input token IDs are required", nil)
	}
	if hipGemma4Q4CanUseBatchedGeneratePrefill(req.TargetForward) {
		return hipRunAttachedDrafterTargetPrefillBatched(ctx, driver, req)
	}
	return hipRunAttachedDrafterTargetPrefillStepwise(ctx, driver, req)
}

func hipRunAttachedDrafterTargetPrefillBatched(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterTargetPrefillRequest) (hipAttachedDrafterTargetPrefillResult, error) {
	ubatchTokens, err := req.EngineConfig.prefillUBatchTokens()
	if err != nil {
		return hipAttachedDrafterTargetPrefillResult{}, err
	}
	prefillBatches := hipBorrowGemma4Q4PrefillUBatches(hipGemma4Q4PrefillBatchCount(len(req.InputTokenIDs), ubatchTokens))
	defer hipReleaseGemma4Q4PrefillUBatches(prefillBatches)
	prefillPlan, prefillBatches, err := hipGemma4Q4PlanPromptPrefillInto(req.InputTokenIDs, req.Position, ubatchTokens, prefillBatches)
	if err != nil {
		return hipAttachedDrafterTargetPrefillResult{}, err
	}
	result := hipAttachedDrafterTargetPrefillResult{
		DeviceState: req.TargetDeviceState,
		Position:    req.Position,
		LastToken:   req.InputTokenIDs[len(req.InputTokenIDs)-1],
	}
	success := false
	defer func() {
		if success {
			return
		}
		hipReleaseForwardDeviceFinalHidden(&result.Current)
		if result.DeviceState != nil && result.DeviceState != req.TargetDeviceState {
			_ = result.DeviceState.Close()
		}
	}()
	var priorLayerKVScratch []*rocmDeviceKVCache
	var priorLayerDescriptorScratch []*rocmDeviceKVDescriptorTable
	for batchIndex := 0; batchIndex < prefillPlan.LenBatches(); batchIndex++ {
		if err := hipContextErr(ctx); err != nil {
			return hipAttachedDrafterTargetPrefillResult{}, err
		}
		ubatch := prefillPlan.Batch(batchIndex)
		var priorLayerKV []*rocmDeviceKVCache
		var priorLayerDescriptorTables []*rocmDeviceKVDescriptorTable
		if result.DeviceState != nil {
			priorLayerKVScratch = hipGemma4Q4DeviceLayerCaches(result.DeviceState, priorLayerKVScratch, len(req.TargetForward.Layers))
			priorLayerKV = priorLayerKVScratch
			priorLayerDescriptorScratch = hipGemma4Q4DeviceLayerDescriptorTables(result.DeviceState, priorLayerDescriptorScratch, len(req.TargetForward.Layers))
			priorLayerDescriptorTables = priorLayerDescriptorScratch
		}
		forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowWithEngineConfig(ctx, driver, req.TargetForward, ubatch.Tokens, ubatch.Position, req.Epsilon, req.DeviceKVMode, priorLayerKV, priorLayerDescriptorTables, nil, ubatch.OutputTokens, ubatch.OutputRow, req.GreedyBuffer, req.Workspace, req.EngineConfig)
		if err != nil {
			return hipAttachedDrafterTargetPrefillResult{}, err
		}
		result.TargetCalls++
		if len(forward.Greedy) > 0 {
			greedyOut := forward.Greedy[len(forward.Greedy)-1]
			result.Current.Greedy = greedyOut.Greedy
			result.Current.GreedyDevice = req.GreedyBuffer
			if hipTokenIsSuppressed(int32(result.Current.Greedy.TokenID), req.SuppressTokens) {
				last := req.TargetForward.Layers[len(req.TargetForward.Layers)-1]
				result.Current.Greedy, err = hipRunGemma4Q4PrefillFinalGreedyForRowSuppressWorkspace(ctx, driver, last, forward.FinalHidden, len(ubatch.Tokens), greedyOut.Row, req.Epsilon, req.GreedyBuffer, req.SuppressTokens, req.Workspace)
				if err != nil {
					_ = forward.Close()
					return hipAttachedDrafterTargetPrefillResult{}, err
				}
				result.Current.GreedyDevice = req.GreedyBuffer
			}
			hipReleaseForwardDeviceFinalHidden(&result.Current)
			last := req.TargetForward.Layers[len(req.TargetForward.Layers)-1]
			hidden, err := hipCloneGemma4Q4PrefillFinalHiddenRow(ctx, driver, forward.FinalHidden, len(ubatch.Tokens), greedyOut.Row, last.HiddenSize, req.Workspace)
			if err != nil {
				_ = forward.Close()
				return hipAttachedDrafterTargetPrefillResult{}, err
			}
			result.Current.DeviceFinalHidden = hidden
			result.Current.DeviceFinalHiddenBorrowed = false
		}
		nextDeviceState, stateErr := hipGemma4Q4DeviceDecodeStateFromPrefillForward(forward, req.DeviceKVMode)
		closeErr := forward.Close()
		if stateErr != nil {
			return hipAttachedDrafterTargetPrefillResult{}, stateErr
		}
		if closeErr != nil {
			_ = nextDeviceState.Close()
			return hipAttachedDrafterTargetPrefillResult{}, closeErr
		}
		previousDeviceState := result.DeviceState
		if err := hipFinalizeGemma4Q4ForwardDeviceState(previousDeviceState, nextDeviceState); err != nil {
			_ = nextDeviceState.Close()
			return hipAttachedDrafterTargetPrefillResult{}, err
		}
		result.DeviceState = nextDeviceState
		hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
		result.Position = ubatch.Position + len(ubatch.Tokens)
	}
	if result.Current.DeviceFinalHidden == nil || result.Current.DeviceFinalHidden.Pointer() == 0 {
		return hipAttachedDrafterTargetPrefillResult{}, core.E("rocm.hip.AttachedDrafterTargetPrefill", "prefill did not return target hidden for assistant proposal", nil)
	}
	success = true
	return result, nil
}

func hipRunAttachedDrafterTargetPrefillStepwise(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterTargetPrefillRequest) (hipAttachedDrafterTargetPrefillResult, error) {
	result := hipAttachedDrafterTargetPrefillResult{
		DeviceState: req.TargetDeviceState,
		Position:    req.Position,
		LastToken:   req.InputTokenIDs[len(req.InputTokenIDs)-1],
	}
	success := false
	defer func() {
		if success {
			return
		}
		hipReleaseForwardDeviceFinalHidden(&result.Current)
		if result.DeviceState != nil && result.DeviceState != req.TargetDeviceState {
			_ = result.DeviceState.Close()
		}
	}()
	haveCurrent := false
	for index, tokenID := range req.InputTokenIDs {
		outputToken := index+1 == len(req.InputTokenIDs)
		current, state, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(ctx, driver, req.TargetForward, result.State, hipGemma4Q4ForwardRequest{
			TokenID:                 tokenID,
			Position:                result.Position,
			Epsilon:                 req.Epsilon,
			DeviceKVAttention:       true,
			DeviceKVMode:            req.DeviceKVMode,
			EngineConfig:            req.EngineConfig,
			PriorDeviceState:        result.DeviceState,
			ReturnDeviceState:       true,
			DeviceFinalSample:       outputToken,
			SkipFinalSample:         !outputToken,
			FinalGreedyBuffer:       req.GreedyBuffer,
			SuppressTokens:          req.SuppressTokens,
			AttentionWorkspace:      req.Workspace,
			OmitDebugTensors:        true,
			OmitLabels:              true,
			OmitHostState:           true,
			ReturnDeviceFinalHidden: outputToken,
		}, false)
		result.TargetCalls++
		if err != nil {
			return hipAttachedDrafterTargetPrefillResult{}, err
		}
		if current.DeviceState == nil {
			return hipAttachedDrafterTargetPrefillResult{}, core.E("rocm.hip.AttachedDrafterTargetPrefill", "forward did not return device KV state", nil)
		}
		hipReleaseForwardDeviceFinalHidden(&result.Current)
		result.Current = current
		result.State = state
		previousDeviceState := result.DeviceState
		result.DeviceState = current.DeviceState
		result.Current.DeviceState = nil
		hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
		result.Position++
		haveCurrent = outputToken
	}
	if !haveCurrent || result.Current.DeviceFinalHidden == nil || result.Current.DeviceFinalHidden.Pointer() == 0 {
		return hipAttachedDrafterTargetPrefillResult{}, core.E("rocm.hip.AttachedDrafterTargetPrefill", "prefill did not return target hidden for assistant proposal", nil)
	}
	success = true
	return result, nil
}

func hipReleaseForwardDeviceFinalHidden(result *hipGemma4Q4ForwardResult) {
	if result == nil || result.DeviceFinalHidden == nil {
		return
	}
	if !result.DeviceFinalHiddenBorrowed {
		_ = result.DeviceFinalHidden.Close()
	}
	result.DeviceFinalHidden = nil
	result.DeviceFinalHiddenBorrowed = false
}

func nonZeroHIPDuration(duration time.Duration) time.Duration {
	if duration <= 0 {
		return time.Nanosecond
	}
	return duration
}
