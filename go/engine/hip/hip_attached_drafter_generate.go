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

type hipAttachedDrafterCarryAdvanceRequest struct {
	TargetForward    hipGemma4Q4ForwardConfig
	DeviceKVMode     string
	EngineConfig     hipGemma4Q4EngineConfig
	State            hipGemma4Q4DecodeState
	PriorDeviceState *hipGemma4Q4DeviceDecodeState
	TokenID          int32
	Position         int
	Epsilon          float32
	SuppressTokens   []int32
	GreedyBuffer     *hipDeviceByteBuffer
	Workspace        *hipAttentionHeadsChunkedWorkspace
}

type hipAttachedDrafterCarryAdvanceResult struct {
	Current     hipGemma4Q4ForwardResult
	State       hipGemma4Q4DecodeState
	DeviceState *hipGemma4Q4DeviceDecodeState
	Position    int
}

type hipAttachedDrafterTargetAdvanceOneRequest struct {
	TargetForward    hipGemma4Q4ForwardConfig
	DeviceKVMode     string
	EngineConfig     hipGemma4Q4EngineConfig
	PriorDeviceState *hipGemma4Q4DeviceDecodeState
	TokenID          int32
	Position         int
	Epsilon          float32
	SuppressTokens   []int32
	GreedyBuffer     *hipDeviceByteBuffer
	Workspace        *hipAttentionHeadsChunkedWorkspace
	ReturnHidden     bool
}

type hipAttachedDrafterTargetAdvanceOneResult struct {
	Current     hipGemma4Q4ForwardResult
	DeviceState *hipGemma4Q4DeviceDecodeState
	Position    int
	TargetCalls int
}

func (result *hipAttachedDrafterCarryAdvanceResult) Close() error {
	if result == nil {
		return nil
	}
	var lastErr error
	hipReleaseForwardDeviceFinalHidden(&result.Current)
	if result.DeviceState != nil {
		if err := result.DeviceState.Close(); err != nil {
			lastErr = err
		}
	}
	result.DeviceState = nil
	return lastErr
}

func (result *hipAttachedDrafterTargetAdvanceOneResult) Close() error {
	if result == nil {
		return nil
	}
	var lastErr error
	hipReleaseForwardDeviceFinalHidden(&result.Current)
	if result.DeviceState != nil {
		if err := result.DeviceState.Close(); err != nil {
			lastErr = err
		}
	}
	result.DeviceState = nil
	return lastErr
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

func hipAdvanceAttachedDrafterCarryLead(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterCarryAdvanceRequest) (hipAttachedDrafterCarryAdvanceResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipAttachedDrafterCarryAdvanceResult{}, err
	}
	advanced, err := hipRunAttachedDrafterTargetAdvanceOneBatch(ctx, driver, hipAttachedDrafterTargetAdvanceOneRequest{
		TargetForward:    req.TargetForward,
		DeviceKVMode:     req.DeviceKVMode,
		EngineConfig:     req.EngineConfig,
		PriorDeviceState: req.PriorDeviceState,
		TokenID:          req.TokenID,
		Position:         req.Position,
		Epsilon:          req.Epsilon,
		SuppressTokens:   req.SuppressTokens,
		GreedyBuffer:     req.GreedyBuffer,
		Workspace:        req.Workspace,
		ReturnHidden:     true,
	})
	if err != nil {
		return hipAttachedDrafterCarryAdvanceResult{}, err
	}
	return hipAttachedDrafterCarryAdvanceResult{
		Current:     advanced.Current,
		State:       req.State,
		DeviceState: advanced.DeviceState,
		Position:    advanced.Position,
	}, nil
}

func hipRunAttachedDrafterTargetAdvanceOneBatch(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterTargetAdvanceOneRequest) (hipAttachedDrafterTargetAdvanceOneResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipAttachedDrafterTargetAdvanceOneResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipAttachedDrafterTargetAdvanceOneResult{}, core.E("rocm.hip.AttachedDrafterTargetAdvanceOne", "HIP driver is not available", nil)
	}
	if len(req.TargetForward.Layers) == 0 {
		return hipAttachedDrafterTargetAdvanceOneResult{}, core.E("rocm.hip.AttachedDrafterTargetAdvanceOne", "target forward config has no layers", nil)
	}
	if req.PriorDeviceState == nil || req.PriorDeviceState.closed {
		return hipAttachedDrafterTargetAdvanceOneResult{}, core.E("rocm.hip.AttachedDrafterTargetAdvanceOne", "live target device state is required", nil)
	}
	if req.Position < 0 {
		return hipAttachedDrafterTargetAdvanceOneResult{}, core.E("rocm.hip.AttachedDrafterTargetAdvanceOne", "position must be non-negative", nil)
	}
	priorLayerKV := hipGemma4Q4DeviceLayerCaches(req.PriorDeviceState, nil, len(req.TargetForward.Layers))
	priorLayerDescriptors, err := hipGemma4Q4DeviceLayerDescriptorTableAliases(req.PriorDeviceState, nil, len(req.TargetForward.Layers))
	if err != nil {
		return hipAttachedDrafterTargetAdvanceOneResult{}, err
	}
	defer hipCloseGemma4Q4DeviceLayerDescriptorTables(priorLayerDescriptors)
	greedyBuffer := hipAttachedDrafterStableDeviceBufferView(req.GreedyBuffer)
	forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowWithEngineConfig(ctx, driver, req.TargetForward, []int32{req.TokenID}, req.Position, req.Epsilon, req.DeviceKVMode, priorLayerKV, priorLayerDescriptors, nil, nil, 0, greedyBuffer, req.Workspace, req.EngineConfig)
	if err != nil {
		return hipAttachedDrafterTargetAdvanceOneResult{}, err
	}
	success := false
	defer func() {
		if !success {
			_ = forward.Close()
		}
	}()
	if len(forward.Greedy) != 1 {
		return hipAttachedDrafterTargetAdvanceOneResult{}, core.E("rocm.hip.AttachedDrafterTargetAdvanceOne", "batch-one target advance did not return one greedy row", nil)
	}
	greedy := forward.Greedy[0].Greedy
	if hipTokenIsSuppressed(int32(greedy.TokenID), req.SuppressTokens) {
		last := req.TargetForward.Layers[len(req.TargetForward.Layers)-1]
		greedy, err = hipRunGemma4Q4PrefillFinalGreedyForRowSuppressWorkspace(ctx, driver, last, forward.FinalHidden, 1, 0, req.Epsilon, greedyBuffer, req.SuppressTokens, req.Workspace)
		if err != nil {
			return hipAttachedDrafterTargetAdvanceOneResult{}, err
		}
	}
	var hidden *hipDeviceByteBuffer
	if req.ReturnHidden {
		last := req.TargetForward.Layers[len(req.TargetForward.Layers)-1]
		hidden, err = hipCloneGemma4Q4PrefillFinalHiddenRow(ctx, driver, forward.FinalHidden, 1, 0, last.HiddenSize, req.Workspace)
		if err != nil {
			return hipAttachedDrafterTargetAdvanceOneResult{}, err
		}
	}
	nextDeviceState, stateErr := hipGemma4Q4DeviceDecodeStateFromPrefillForward(forward, req.DeviceKVMode)
	closeErr := forward.Close()
	success = true
	if stateErr != nil {
		_ = hidden.Close()
		return hipAttachedDrafterTargetAdvanceOneResult{}, stateErr
	}
	if closeErr != nil {
		_ = hidden.Close()
		_ = nextDeviceState.Close()
		return hipAttachedDrafterTargetAdvanceOneResult{}, closeErr
	}
	if err := hipFinalizeGemma4Q4ForwardDeviceState(req.PriorDeviceState, nextDeviceState); err != nil {
		_ = hidden.Close()
		_ = nextDeviceState.Close()
		return hipAttachedDrafterTargetAdvanceOneResult{}, err
	}
	return hipAttachedDrafterTargetAdvanceOneResult{
		Current: hipGemma4Q4ForwardResult{
			Greedy:                    greedy,
			GreedyDevice:              greedyBuffer,
			DeviceFinalHidden:         hidden,
			DeviceFinalHiddenBorrowed: false,
		},
		DeviceState: nextDeviceState,
		Position:    req.Position + 1,
		TargetCalls: 1,
	}, nil
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
	prefillGreedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
	if err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		return inferdecode.Result{}, err
	}
	defer func() {
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
		GreedyBuffer:      prefillGreedyBuffer,
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
		if carryLead >= 0 {
			carryGreedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
			if err != nil {
				return inferdecode.Result{}, err
			}
			targetStart := time.Now()
			advanced, err := hipAdvanceAttachedDrafterCarryLead(ctx, model.driver, hipAttachedDrafterCarryAdvanceRequest{
				TargetForward:    targetCfg,
				DeviceKVMode:     deviceKVMode,
				EngineConfig:     engineConfig,
				State:            state,
				PriorDeviceState: deviceState,
				TokenID:          carryLead,
				Position:         position,
				Epsilon:          1e-6,
				SuppressTokens:   suppressTokens,
				GreedyBuffer:     carryGreedyBuffer,
				Workspace:        workspace,
			})
			targetDuration += nonZeroHIPDuration(time.Since(targetStart))
			targetCalls++
			if err != nil {
				return inferdecode.Result{}, err
			}
			hipReleaseForwardDeviceFinalHidden(&current)
			previousDeviceState := deviceState
			current = advanced.Current
			advanced.Current = hipGemma4Q4ForwardResult{}
			state = advanced.State
			deviceState = advanced.DeviceState
			advanced.DeviceState = nil
			position = advanced.Position
			currentToken = carryLead
			carryLead = -1
			hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
			if err := advanced.Close(); err != nil {
				return inferdecode.Result{}, err
			}
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
		verifyGreedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
		if err != nil {
			return inferdecode.Result{}, err
		}
		verifyWorkspace := workspace
		if core.Env("GO_ROCM_ATTACHED_DRAFTER_DISABLE_VERIFY_WORKSPACE") == "1" {
			verifyWorkspace = nil
		}
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
			GreedyBuffer:      verifyGreedyBuffer,
			Workspace:         verifyWorkspace,
		})
		targetDuration += nonZeroHIPDuration(time.Since(targetStart))
		if err := draftBlock.Close(); err != nil {
			return inferdecode.Result{}, err
		}
		if verifyErr != nil {
			return inferdecode.Result{}, verifyErr
		}
		targetCalls += verify.TargetCalls
		if core.Env("GO_ROCM_ATTACHED_DRAFTER_TRACE_BLOCKS") == "1" {
			core.Print(core.Stderr(), "rocm.hip.attached_drafter.block output=%d position=%d carry=%t current=%d draft=%v verify=%v verified=%v accepted=%d all=%t replacement=%d next=%d proposed=%d target_calls=%d",
				len(tokens),
				position,
				carryPresent,
				current.Greedy.TokenID,
				draftBlock.Tokens,
				verifyTokens,
				hipAttachedDrafterGreedyTokenIDs(verify.VerifiedGreedies),
				verify.AcceptedCount,
				verify.AllAccepted,
				verify.Replacement.TokenID,
				verify.NextGreedy.TokenID,
				proposedCount,
				verify.TargetCalls,
			)
		}
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
		flushGreedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
		if err != nil {
			return inferdecode.Result{}, err
		}
		stepStart := time.Now()
		advanced, err := hipAdvanceAttachedDrafterCarryLead(ctx, model.driver, hipAttachedDrafterCarryAdvanceRequest{
			TargetForward:    targetCfg,
			DeviceKVMode:     deviceKVMode,
			EngineConfig:     engineConfig,
			State:            state,
			PriorDeviceState: deviceState,
			TokenID:          carryLead,
			Position:         position,
			Epsilon:          1e-6,
			SuppressTokens:   suppressTokens,
			GreedyBuffer:     flushGreedyBuffer,
			Workspace:        workspace,
		})
		targetDuration += nonZeroHIPDuration(time.Since(stepStart))
		targetCalls++
		if err != nil {
			return inferdecode.Result{}, err
		}
		previousDeviceState := deviceState
		deviceState = advanced.DeviceState
		advanced.DeviceState = nil
		state = advanced.State
		hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
		position = advanced.Position
		carryLead = -1
		if err := advanced.Close(); err != nil {
			return inferdecode.Result{}, err
		}
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

func hipAttachedDrafterGreedyTokenIDs(greedies []hipGreedySampleResult) []int32 {
	tokens := make([]int32, 0, len(greedies))
	for _, greedy := range greedies {
		tokens = append(tokens, int32(greedy.TokenID))
	}
	return tokens
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
	// The target prefill keeps every ubatch — including a trailing single-token
	// ubatch — on the batched projection path so the retained decode state is
	// built from uniform batched kernels rather than the single-row fast path.
	forwardEngineConfig := req.EngineConfig
	forwardEngineConfig.ForceBatchedProjection = true
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
		forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowWithEngineConfig(ctx, driver, req.TargetForward, ubatch.Tokens, ubatch.Position, req.Epsilon, req.DeviceKVMode, priorLayerKV, priorLayerDescriptorTables, nil, ubatch.OutputTokens, ubatch.OutputRow, req.GreedyBuffer, req.Workspace, forwardEngineConfig)
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
