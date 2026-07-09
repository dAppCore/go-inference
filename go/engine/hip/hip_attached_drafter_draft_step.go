// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"math"
	"strconv"
	"strings"

	core "dappco.re/go"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
)

const (
	attachedDrafterAssistantDraftStepInputNotReady = "not_ready"
	attachedDrafterAssistantDraftStepInputLinked   = hipKernelStatusLinked

	attachedDrafterAssistantDraftStepProposalNotReady = "not_ready"
	attachedDrafterAssistantDraftStepProposalLinked   = hipKernelStatusLinked
)

type hipAttachedDrafterAssistantDraftStepInputPlan struct {
	Status               string
	Reason               string
	HiddenSize           int
	VocabSize            int
	TargetHiddenSize     int
	CombinedInputSize    int
	ProjectionEncoding   string
	KernelFamilies       []string
	TargetEmbedding      hipDeviceEmbeddingLookupConfig
	TargetEmbeddingScale float32
	PreProjection        hipAttachedDrafterAssistantProjectionPlan
}

type hipAttachedDrafterAssistantDraftStepInputRequest struct {
	LastToken         int32
	LastGreedyToken   *hipDeviceByteBuffer
	TargetHidden      *hipDeviceByteBuffer
	TargetDeviceState *hipGemma4Q4DeviceDecodeState
	Plan              hipAttachedDrafterAssistantDraftStepInputPlan
	Workspace         *hipAttentionHeadsChunkedWorkspace
}

type hipAttachedDrafterAssistantDraftStepInputResult struct {
	Hidden *hipDeviceByteBuffer
	Labels map[string]string
}

type hipAttachedDrafterAssistantDraftStepHiddenRequest struct {
	LastToken         int32
	LastGreedyToken   *hipDeviceByteBuffer
	TargetHidden      *hipDeviceByteBuffer
	TargetForward     hipGemma4Q4ForwardConfig
	TargetDeviceState *hipGemma4Q4DeviceDecodeState
	Plan              hipAttachedDrafterAssistantVerifierPlan
	InputPlan         hipAttachedDrafterAssistantDraftStepInputPlan
	Position          int
	Epsilon           float32
	Workspace         *hipAttentionHeadsChunkedWorkspace
}

type hipAttachedDrafterAssistantDraftStepHiddenResult struct {
	Normed *hipDeviceByteBuffer
	Hidden *hipDeviceByteBuffer
	Labels map[string]string
}

type hipAttachedDrafterAssistantDraftStepProposalRequest struct {
	LastToken         int32
	LastGreedyToken   *hipDeviceByteBuffer
	TargetHidden      *hipDeviceByteBuffer
	TargetForward     hipGemma4Q4ForwardConfig
	TargetDeviceState *hipGemma4Q4DeviceDecodeState
	Plan              hipAttachedDrafterAssistantVerifierPlan
	InputPlan         hipAttachedDrafterAssistantDraftStepInputPlan
	Position          int
	Epsilon           float32
	Softcap           float32
	SuppressTokens    []int32
	Workspace         *hipAttentionHeadsChunkedWorkspace
}

type hipAttachedDrafterAssistantDraftStepProposalResult struct {
	Token  hipGreedySampleResult
	Logits *hipDeviceByteBuffer
	Hidden *hipDeviceByteBuffer
	Labels map[string]string
}

type hipAttachedDrafterAssistantDraftStepDeviceTokenResult struct {
	GreedyToken *hipDeviceByteBuffer
	Hidden      *hipDeviceByteBuffer
	Labels      map[string]string
}

func (result *hipAttachedDrafterAssistantDraftStepInputResult) Close() error {
	if result == nil {
		return nil
	}
	err := result.Hidden.Close()
	result.Hidden = nil
	result.Labels = nil
	return err
}

func (result *hipAttachedDrafterAssistantDraftStepHiddenResult) Close() error {
	if result == nil {
		return nil
	}
	var lastErr error
	if err := result.Normed.Close(); err != nil {
		lastErr = err
	}
	if err := result.Hidden.Close(); err != nil {
		lastErr = err
	}
	result.Normed = nil
	result.Hidden = nil
	result.Labels = nil
	return lastErr
}

func (result *hipAttachedDrafterAssistantDraftStepProposalResult) Close() error {
	if result == nil {
		return nil
	}
	var lastErr error
	if err := result.Logits.Close(); err != nil {
		lastErr = err
	}
	if err := result.Hidden.Close(); err != nil {
		lastErr = err
	}
	result.Logits = nil
	result.Hidden = nil
	result.Labels = nil
	return lastErr
}

func (result *hipAttachedDrafterAssistantDraftStepDeviceTokenResult) Close() error {
	if result == nil {
		return nil
	}
	err := result.Hidden.Close()
	result.GreedyToken = nil
	result.Hidden = nil
	result.Labels = nil
	return err
}

func hipAttachedDrafterAssistantDraftStepInputPlanForModel(target *hipLoadedModel, assistantPlan hipAttachedDrafterAssistantVerifierPlan) hipAttachedDrafterAssistantDraftStepInputPlan {
	if assistantPlan.Status != attachedDrafterAssistantVerifierPlanTensorBound {
		return hipAttachedDrafterAssistantDraftStepInputPlan{
			Status: attachedDrafterAssistantDraftStepInputNotReady,
			Reason: "assistant verifier plan is " + firstNonEmptyString(assistantPlan.Status, "empty"),
		}
	}
	if target == nil {
		return hipAttachedDrafterAssistantDraftStepInputPlan{
			Status: attachedDrafterAssistantDraftStepInputNotReady,
			Reason: "target model is nil",
		}
	}
	if target.modelInfo.NumLayers <= 0 {
		return hipAttachedDrafterAssistantDraftStepInputPlan{
			Status: attachedDrafterAssistantDraftStepInputNotReady,
			Reason: "target layer count is missing",
		}
	}
	cfg, err := target.cachedGemma4Q4ForwardConfig(target.modelInfo.NumLayers)
	if err != nil {
		return hipAttachedDrafterAssistantDraftStepInputPlan{
			Status: attachedDrafterAssistantDraftStepInputNotReady,
			Reason: "target forward config: " + err.Error(),
		}
	}
	if len(cfg.Layers) == 0 {
		return hipAttachedDrafterAssistantDraftStepInputPlan{
			Status: attachedDrafterAssistantDraftStepInputNotReady,
			Reason: "target forward config has no layers",
		}
	}
	first := cfg.Layers[0]
	plan := hipAttachedDrafterAssistantDraftStepInputPlan{
		Status:               attachedDrafterAssistantDraftStepInputLinked,
		HiddenSize:           assistantPlan.HiddenSize,
		VocabSize:            first.VocabSize,
		TargetHiddenSize:     first.HiddenSize,
		CombinedInputSize:    first.Embedding.HiddenSize + first.HiddenSize,
		ProjectionEncoding:   assistantPlan.PreProjection.Encoding,
		TargetEmbedding:      first.Embedding,
		TargetEmbeddingScale: first.embeddingScale(),
		PreProjection:        assistantPlan.PreProjection,
		KernelFamilies: []string{
			hipKernelNameEmbedLookup,
			hipKernelNameVectorScale,
			hipAttachedDrafterAssistantVerifierProjectionKernel(assistantPlan.PreProjection.Encoding == "mlx_affine"),
		},
	}
	if reason := hipAttachedDrafterAssistantDraftStepInputPlanInvalidReason(plan); reason != "" {
		plan.Status = attachedDrafterAssistantDraftStepInputNotReady
		plan.Reason = reason
	}
	return plan
}

func hipAttachedDrafterAssistantDraftStepInputPlanInvalidReason(plan hipAttachedDrafterAssistantDraftStepInputPlan) string {
	if plan.HiddenSize <= 0 || plan.TargetHiddenSize <= 0 || plan.CombinedInputSize <= 0 {
		return "hidden sizes must be positive"
	}
	if plan.PreProjection.Rows != plan.HiddenSize {
		return "pre_projection rows must match assistant hidden size"
	}
	if plan.TargetEmbedding.HiddenSize <= 0 {
		return "target embedding hidden size is missing"
	}
	if plan.TargetEmbedding.HiddenSize != plan.TargetHiddenSize {
		return "target embedding hidden size must match target hidden size"
	}
	if plan.CombinedInputSize != plan.TargetEmbedding.HiddenSize+plan.TargetHiddenSize {
		return "combined input size must equal target token embedding plus target hidden"
	}
	if plan.PreProjection.Cols != plan.CombinedInputSize {
		return "pre_projection cols must match combined target token and hidden size"
	}
	if plan.TargetEmbedding.VocabSize <= 0 {
		return "target embedding vocab size is missing"
	}
	if plan.TargetEmbeddingScale == 0 {
		return "target embedding scale is missing"
	}
	switch plan.PreProjection.Encoding {
	case "bf16":
		if plan.PreProjection.BF16.WeightPointer == 0 {
			return "pre_projection BF16 weight pointer is required"
		}
	case "mlx_affine":
		if plan.PreProjection.MLXAffine.WeightPointer == 0 {
			return "pre_projection MLX affine weight pointer is required"
		}
	default:
		return "pre_projection encoding is unsupported"
	}
	return ""
}

func (plan hipAttachedDrafterAssistantDraftStepInputPlan) Labels() map[string]string {
	labels := map[string]string{
		"attached_drafter_assistant_draft_step_input_bridge": plan.Status,
	}
	if plan.Reason != "" {
		labels["attached_drafter_assistant_draft_step_input_bridge_reason"] = plan.Reason
	}
	if plan.HiddenSize > 0 {
		labels["attached_drafter_assistant_draft_step_hidden_size"] = strconv.Itoa(plan.HiddenSize)
	}
	if plan.TargetHiddenSize > 0 {
		labels["attached_drafter_assistant_draft_step_target_hidden_size"] = strconv.Itoa(plan.TargetHiddenSize)
	}
	if plan.TargetEmbedding.HiddenSize > 0 {
		labels["attached_drafter_assistant_draft_step_target_embedding_hidden_size"] = strconv.Itoa(plan.TargetEmbedding.HiddenSize)
	}
	if plan.CombinedInputSize > 0 {
		labels["attached_drafter_assistant_draft_step_combined_input_size"] = strconv.Itoa(plan.CombinedInputSize)
	}
	if plan.ProjectionEncoding != "" {
		labels["attached_drafter_assistant_draft_step_pre_projection_encoding"] = plan.ProjectionEncoding
	}
	if len(plan.KernelFamilies) > 0 {
		labels["attached_drafter_assistant_draft_step_kernel_families"] = strings.Join(plan.KernelFamilies, ",")
	}
	return labels
}

func hipAttachedDrafterAssistantDraftStepHiddenRuntimeLabels(plan hipAttachedDrafterAssistantVerifierPlan, input hipAttachedDrafterAssistantDraftStepInputPlan) map[string]string {
	status := attachedDrafterAssistantLayerRuntimeLinked
	reason := ""
	if plan.Status != attachedDrafterAssistantVerifierPlanTensorBound {
		status = attachedDrafterAssistantLayerRuntimeNotReady
		reason = "assistant verifier plan is " + firstNonEmptyString(plan.Status, "empty")
	} else if input.Status != attachedDrafterAssistantDraftStepInputLinked {
		status = attachedDrafterAssistantLayerRuntimeNotReady
		reason = "draft-step input bridge is " + firstNonEmptyString(input.Status, "empty")
	} else if len(plan.Layers) == 0 {
		status = attachedDrafterAssistantLayerRuntimeNotReady
		reason = "assistant layer plan is empty"
	} else if err := hipAttachedDrafterAssistantDraftStepHiddenPlanInvalidReason(plan, input); err != nil {
		status = attachedDrafterAssistantLayerRuntimeNotReady
		reason = err.Error()
	}
	labels := map[string]string{
		"attached_drafter_assistant_draft_step_hidden_runtime": status,
		"attached_drafter_assistant_draft_step_hidden_source":  "assistant_layer_chain",
	}
	if reason != "" {
		labels["attached_drafter_assistant_draft_step_hidden_runtime_reason"] = reason
	}
	if len(plan.Layers) > 0 {
		labels["attached_drafter_assistant_draft_step_hidden_layers"] = strconv.Itoa(len(plan.Layers))
	}
	if plan.PostProjection.Encoding != "" {
		labels["attached_drafter_assistant_draft_step_post_projection_encoding"] = plan.PostProjection.Encoding
	}
	return labels
}

func hipAttachedDrafterAssistantDraftStepProposalRuntimeLabels(plan hipAttachedDrafterAssistantVerifierPlan, input hipAttachedDrafterAssistantDraftStepInputPlan, softcap float32) map[string]string {
	status := attachedDrafterAssistantDraftStepProposalLinked
	reason := ""
	if plan.Status != attachedDrafterAssistantVerifierPlanTensorBound {
		status = attachedDrafterAssistantDraftStepProposalNotReady
		reason = "assistant verifier plan is " + firstNonEmptyString(plan.Status, "empty")
	} else if input.Status != attachedDrafterAssistantDraftStepInputLinked {
		status = attachedDrafterAssistantDraftStepProposalNotReady
		reason = "draft-step input bridge is " + firstNonEmptyString(input.Status, "empty")
	} else if err := hipAttachedDrafterAssistantDraftStepProposalPlanInvalidReason(plan, softcap); err != nil {
		status = attachedDrafterAssistantDraftStepProposalNotReady
		reason = err.Error()
	}
	labels := map[string]string{
		"attached_drafter_assistant_draft_step_proposal_runtime": status,
		"attached_drafter_assistant_draft_step_proposal_source":  "assistant_embedding_lm_head",
	}
	if reason != "" {
		labels["attached_drafter_assistant_draft_step_proposal_runtime_reason"] = reason
	}
	if plan.Embedding.TableEncoding > 0 {
		labels["attached_drafter_assistant_draft_step_proposal_embedding_encoding"] = hipAttachedDrafterAssistantEmbeddingEncodingLabel(plan.Embedding.TableEncoding)
	}
	if hipAttachedDrafterAssistantUsesOrderedEmbeddingCandidates(plan) {
		labels["attached_drafter_assistant_draft_step_proposal_ordered_embeddings"] = "true"
		labels["attached_drafter_assistant_draft_step_proposal_candidate_top_k"] = strconv.Itoa(modelgemma4.AssistantCentroidIntermediateTopK)
	}
	if softcap > 0 {
		labels["attached_drafter_assistant_draft_step_proposal_softcap"] = strconv.FormatFloat(float64(softcap), 'g', -1, 32)
	}
	return labels
}

func hipRunAttachedDrafterAssistantDraftStepInputBridge(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterAssistantDraftStepInputRequest) (hipAttachedDrafterAssistantDraftStepInputResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipAttachedDrafterAssistantDraftStepInputResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipAttachedDrafterAssistantDraftStepInputResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftStep", "HIP driver is not available", nil)
	}
	if req.Plan.Status != attachedDrafterAssistantDraftStepInputLinked {
		return hipAttachedDrafterAssistantDraftStepInputResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftStep", "draft-step input bridge is not linked", nil)
	}
	if reason := hipAttachedDrafterAssistantDraftStepInputPlanInvalidReason(req.Plan); reason != "" {
		return hipAttachedDrafterAssistantDraftStepInputResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftStep", reason, nil)
	}
	if req.LastGreedyToken != nil {
		if err := hipAttachedDrafterValidateGreedyTokenBuffer(req.LastGreedyToken); err != nil {
			return hipAttachedDrafterAssistantDraftStepInputResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftStep", "validate greedy token buffer", err)
		}
	} else if err := req.Plan.TargetEmbedding.validateSingleToken(req.LastToken); err != nil {
		return hipAttachedDrafterAssistantDraftStepInputResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftStep", "validate last token", err)
	}
	if req.TargetHidden == nil || req.TargetHidden.Pointer() == 0 ||
		req.TargetHidden.Count() != req.Plan.TargetHiddenSize ||
		req.TargetHidden.SizeBytes() != uint64(req.Plan.TargetHiddenSize*4) {
		return hipAttachedDrafterAssistantDraftStepInputResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftStep", "target device hidden shape mismatch", nil)
	}
	if req.TargetDeviceState == nil || req.TargetDeviceState.closed || req.TargetDeviceState.maxLayerTokenCount() <= 0 {
		return hipAttachedDrafterAssistantDraftStepInputResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftStep", "target device KV state is required", nil)
	}
	workspaceOwned := false
	if req.Workspace == nil {
		req.Workspace = &hipAttentionHeadsChunkedWorkspace{}
		workspaceOwned = true
		defer req.Workspace.Close()
	}

	var combined *hipDeviceByteBuffer
	var err error
	if workspaceOwned {
		combined, err = hipAllocateByteBuffer(driver, "rocm.hip.AttachedDrafterAssistantDraftStep", "assistant draft-step combined input", uint64(req.Plan.CombinedInputSize*4), req.Plan.CombinedInputSize)
	} else {
		combined, err = req.Workspace.EnsureAssistantDraftCombined(driver, req.Plan.CombinedInputSize)
	}
	if err != nil {
		return hipAttachedDrafterAssistantDraftStepInputResult{}, err
	}
	success := false
	defer func() {
		if !success {
			_ = combined.Close()
		}
	}()

	targetEmbeddingHiddenSize := req.Plan.TargetEmbedding.HiddenSize
	tokenEmbedding := hipBorrowDeviceByteBufferValue(driver, "assistant draft-step token embedding input view", combined.Pointer(), uint64(targetEmbeddingHiddenSize*4), targetEmbeddingHiddenSize)
	targetHidden := hipBorrowDeviceByteBufferValue(driver, "assistant draft-step target hidden input view", combined.Pointer()+nativeDevicePointer(targetEmbeddingHiddenSize*4), uint64(req.Plan.TargetHiddenSize*4), req.Plan.TargetHiddenSize)

	tokenInputSource := "host_token"
	if req.LastGreedyToken != nil {
		if err := hipRunEmbeddingLookupKernelWithDeviceTableGreedyTokenScaledOutputWithWorkspace(ctx, driver, req.Plan.TargetEmbedding, req.LastGreedyToken, &tokenEmbedding, req.Plan.TargetEmbeddingScale, req.Workspace); err != nil {
			return hipAttachedDrafterAssistantDraftStepInputResult{}, err
		}
		tokenInputSource = "device_greedy"
	} else {
		tokenBuffer, err := req.Workspace.EnsureTokenIDValue(driver, req.LastToken, req.Plan.TargetEmbedding.VocabSize)
		if err != nil {
			return hipAttachedDrafterAssistantDraftStepInputResult{}, err
		}
		if err := hipRunEmbeddingLookupKernelWithDeviceTableTokenBufferScaledOutputWithWorkspace(ctx, driver, req.Plan.TargetEmbedding, tokenBuffer, &tokenEmbedding, req.Plan.TargetEmbeddingScale, req.Workspace); err != nil {
			return hipAttachedDrafterAssistantDraftStepInputResult{}, err
		}
	}
	if err := hipRunVectorScaleDeviceKernelOutputWithWorkspace(ctx, driver, req.TargetHidden, 1, &targetHidden, req.Workspace); err != nil {
		return hipAttachedDrafterAssistantDraftStepInputResult{}, err
	}

	var hidden *hipDeviceByteBuffer
	if workspaceOwned {
		hidden, err = hipAllocateByteBuffer(driver, "rocm.hip.AttachedDrafterAssistantDraftStep", "assistant draft-step pre-projection hidden", uint64(req.Plan.HiddenSize*4), req.Plan.HiddenSize)
	} else {
		hidden, err = req.Workspace.EnsureAssistantDraftInputHidden(driver, req.Plan.HiddenSize)
	}
	if err != nil {
		return hipAttachedDrafterAssistantDraftStepInputResult{}, err
	}
	defer func() {
		if !success {
			_ = hidden.Close()
		}
	}()
	if err := hipRunAttachedDrafterAssistantProjectionOutput(ctx, driver, combined, req.Plan.PreProjection, hidden, req.Workspace); err != nil {
		return hipAttachedDrafterAssistantDraftStepInputResult{}, err
	}
	labels := req.Plan.Labels()
	labels["attached_drafter_assistant_draft_step_target_hidden_source"] = "device"
	labels["attached_drafter_assistant_draft_step_device_kv"] = "required"
	labels["attached_drafter_assistant_draft_step_input_buffer"] = "device_combined_token_hidden"
	if !workspaceOwned {
		labels["attached_drafter_assistant_draft_step_input_buffer_reuse"] = "workspace"
	}
	labels["attached_drafter_assistant_draft_step_token_input"] = tokenInputSource
	success = true
	_ = combined.Close()
	return hipAttachedDrafterAssistantDraftStepInputResult{Hidden: hidden, Labels: labels}, nil
}

func hipAttachedDrafterAssistantDraftStepHiddenPlanInvalidReason(plan hipAttachedDrafterAssistantVerifierPlan, input hipAttachedDrafterAssistantDraftStepInputPlan) error {
	if plan.Status != attachedDrafterAssistantVerifierPlanTensorBound {
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStepHidden", "assistant verifier plan is not tensor-bound", nil)
	}
	if input.Status != attachedDrafterAssistantDraftStepInputLinked {
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStepHidden", "draft-step input bridge is not linked", nil)
	}
	if reason := hipAttachedDrafterAssistantDraftStepInputPlanInvalidReason(input); reason != "" {
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStepHidden", reason, nil)
	}
	if plan.HiddenSize <= 0 || input.HiddenSize != plan.HiddenSize {
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStepHidden", "assistant hidden size mismatch", nil)
	}
	if len(plan.Layers) == 0 {
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStepHidden", "assistant layer plan is empty", nil)
	}
	if plan.Norm.Count != plan.HiddenSize {
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStepHidden", "assistant final norm count must match hidden size", nil)
	}
	if err := hipValidateRMSNormDeviceWeightConfig("AttachedDrafterAssistantDraftStepHidden.norm", plan.Norm); err != nil {
		return err
	}
	if plan.PostProjection.Rows != input.TargetHiddenSize || plan.PostProjection.Cols != plan.HiddenSize {
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStepHidden", "assistant post_projection shape mismatch", nil)
	}
	if err := hipAttachedDrafterAssistantProjectionPlanValid(plan.PostProjection, plan.HiddenSize); err != nil {
		return err
	}
	for _, layer := range plan.Layers {
		if err := hipAttachedDrafterAssistantLayerPlanInvalidReason(layer); err != nil {
			return err
		}
	}
	return nil
}

func hipRunAttachedDrafterAssistantDraftStepHidden(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterAssistantDraftStepHiddenRequest) (hipAttachedDrafterAssistantDraftStepHiddenResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipAttachedDrafterAssistantDraftStepHiddenResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipAttachedDrafterAssistantDraftStepHiddenResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftStepHidden", "HIP driver is not available", nil)
	}
	if err := hipAttachedDrafterAssistantDraftStepHiddenPlanInvalidReason(req.Plan, req.InputPlan); err != nil {
		return hipAttachedDrafterAssistantDraftStepHiddenResult{}, err
	}
	if req.TargetDeviceState == nil || req.TargetDeviceState.closed {
		return hipAttachedDrafterAssistantDraftStepHiddenResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftStepHidden", "target device KV state is required", nil)
	}
	if req.Workspace == nil {
		req.Workspace = &hipAttentionHeadsChunkedWorkspace{}
		defer req.Workspace.Close()
	}

	inputResult, err := hipRunAttachedDrafterAssistantDraftStepInputBridge(ctx, driver, hipAttachedDrafterAssistantDraftStepInputRequest{
		LastToken:         req.LastToken,
		LastGreedyToken:   req.LastGreedyToken,
		TargetHidden:      req.TargetHidden,
		TargetDeviceState: req.TargetDeviceState,
		Plan:              req.InputPlan,
		Workspace:         req.Workspace,
	})
	if err != nil {
		return hipAttachedDrafterAssistantDraftStepHiddenResult{}, err
	}
	current := inputResult.Hidden
	inputResult.Hidden = nil
	defer inputResult.Close()

	success := false
	var normed *hipDeviceByteBuffer
	var hidden *hipDeviceByteBuffer
	defer func() {
		if success {
			return
		}
		_ = current.Close()
		_ = normed.Close()
		_ = hidden.Close()
	}()

	targetLayerSources := make([]string, 0, len(req.Plan.Layers))
	for _, layerPlan := range req.Plan.Layers {
		targetLayer, targetLayerConfig, targetLayerIndex, err := hipAttachedDrafterAssistantTargetLayerFor(layerPlan.LayerType, req.TargetForward, req.TargetDeviceState)
		if err != nil {
			return hipAttachedDrafterAssistantDraftStepHiddenResult{}, err
		}
		layerResult, err := hipRunAttachedDrafterAssistantLayer(ctx, driver, hipAttachedDrafterAssistantLayerRequest{
			Hidden:            current,
			TargetLayer:       targetLayer,
			TargetLayerConfig: targetLayerConfig,
			Plan:              layerPlan,
			Position:          req.Position,
			Epsilon:           req.Epsilon,
			Workspace:         req.Workspace,
		})
		if err != nil {
			return hipAttachedDrafterAssistantDraftStepHiddenResult{}, err
		}
		_ = current.Close()
		current = layerResult.Hidden
		layerResult.Hidden = nil
		_ = layerResult.Close()
		targetLayerSources = append(targetLayerSources, strconv.Itoa(targetLayerIndex))
	}

	normCfg := req.Plan.Norm
	normCfg.Epsilon = req.Epsilon
	normed, err = hipAllocateByteBuffer(driver, "rocm.hip.AttachedDrafterAssistantDraftStepHidden", "assistant final norm output", uint64(req.Plan.HiddenSize*4), req.Plan.HiddenSize)
	if err != nil {
		return hipAttachedDrafterAssistantDraftStepHiddenResult{}, err
	}
	if err := hipRunRMSNormDeviceToDeviceKernelWithWorkspace(ctx, driver, current.Pointer(), current.SizeBytes(), normed.Pointer(), normed.SizeBytes(), normCfg, req.Workspace); err != nil {
		return hipAttachedDrafterAssistantDraftStepHiddenResult{}, err
	}
	hidden, err = hipAllocateByteBuffer(driver, "rocm.hip.AttachedDrafterAssistantDraftStepHidden", "assistant post-projection target hidden", uint64(req.Plan.PostProjection.Rows*4), req.Plan.PostProjection.Rows)
	if err != nil {
		return hipAttachedDrafterAssistantDraftStepHiddenResult{}, err
	}
	if err := hipRunAttachedDrafterAssistantProjectionOutput(ctx, driver, normed, req.Plan.PostProjection, hidden, req.Workspace); err != nil {
		return hipAttachedDrafterAssistantDraftStepHiddenResult{}, err
	}

	labels := hipAttachedDrafterAssistantDraftStepHiddenRuntimeLabels(req.Plan, req.InputPlan)
	for key, value := range inputResult.Labels {
		labels[key] = value
	}
	labels["attached_drafter_assistant_draft_step_hidden_runtime"] = attachedDrafterAssistantLayerRuntimeLinked
	labels["attached_drafter_assistant_draft_step_hidden_layers_executed"] = strconv.Itoa(len(req.Plan.Layers))
	labels["attached_drafter_assistant_draft_step_target_layer_sources"] = strings.Join(targetLayerSources, ",")
	labels["attached_drafter_assistant_draft_step_normed"] = "assistant_final_norm"
	labels["attached_drafter_assistant_draft_step_hidden_source"] = "assistant_post_projection"
	success = true
	_ = current.Close()
	return hipAttachedDrafterAssistantDraftStepHiddenResult{Normed: normed, Hidden: hidden, Labels: labels}, nil
}

func hipRunAttachedDrafterAssistantDraftStepProposal(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterAssistantDraftStepProposalRequest) (hipAttachedDrafterAssistantDraftStepProposalResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipAttachedDrafterAssistantDraftStepProposalResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipAttachedDrafterAssistantDraftStepProposalResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "HIP driver is not available", nil)
	}
	if err := hipAttachedDrafterAssistantDraftStepProposalPlanInvalidReason(req.Plan, req.Softcap); err != nil {
		return hipAttachedDrafterAssistantDraftStepProposalResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "proposal plan", err)
	}
	hiddenResult, err := hipRunAttachedDrafterAssistantDraftStepHidden(ctx, driver, hipAttachedDrafterAssistantDraftStepHiddenRequest{
		LastToken:         req.LastToken,
		LastGreedyToken:   req.LastGreedyToken,
		TargetHidden:      req.TargetHidden,
		TargetForward:     req.TargetForward,
		TargetDeviceState: req.TargetDeviceState,
		Plan:              req.Plan,
		InputPlan:         req.InputPlan,
		Position:          req.Position,
		Epsilon:           req.Epsilon,
		Workspace:         req.Workspace,
	})
	if err != nil {
		return hipAttachedDrafterAssistantDraftStepProposalResult{}, err
	}
	labels := make(map[string]string, len(hiddenResult.Labels)+8)
	for key, value := range hiddenResult.Labels {
		labels[key] = value
	}
	for key, value := range hipAttachedDrafterAssistantDraftStepProposalRuntimeLabels(req.Plan, req.InputPlan, req.Softcap) {
		labels[key] = value
	}
	normed := hiddenResult.Normed
	hidden := hiddenResult.Hidden
	hiddenResult.Normed = nil
	hiddenResult.Hidden = nil
	hiddenResult.Labels = nil
	_ = hiddenResult.Close()
	if normed == nil || normed.Pointer() == 0 {
		_ = hidden.Close()
		return hipAttachedDrafterAssistantDraftStepProposalResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "assistant normed hidden is required", nil)
	}
	defer normed.Close()
	var logits *hipDeviceByteBuffer
	success := false
	defer func() {
		if !success {
			_ = logits.Close()
			_ = hidden.Close()
		}
	}()
	token, logits, err := hipRunAttachedDrafterAssistantProposalToken(ctx, driver, normed, req.Plan, req.Softcap, req.SuppressTokens, req.Workspace)
	if err != nil {
		return hipAttachedDrafterAssistantDraftStepProposalResult{}, err
	}
	if logits != nil {
		labels["attached_drafter_assistant_draft_step_logits"] = "dense_retained"
		labels["attached_drafter_assistant_draft_step_token_source"] = "dense_logits_greedy"
	} else if hipAttachedDrafterAssistantUsesOrderedEmbeddingCandidates(req.Plan) {
		labels["attached_drafter_assistant_draft_step_logits"] = "not_retained"
		labels["attached_drafter_assistant_draft_step_token_source"] = "ordered_embedding_selected_greedy"
	} else {
		labels["attached_drafter_assistant_draft_step_logits"] = "not_retained"
		labels["attached_drafter_assistant_draft_step_token_source"] = "projection_greedy"
	}
	labels["attached_drafter_assistant_draft_step_token"] = "greedy"
	labels["attached_drafter_assistant_draft_step_token_id"] = strconv.Itoa(token.TokenID)
	success = true
	return hipAttachedDrafterAssistantDraftStepProposalResult{
		Token:  token,
		Logits: logits,
		Hidden: hidden,
		Labels: labels,
	}, nil
}

func hipRunAttachedDrafterAssistantDraftStepDeviceToken(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterAssistantDraftStepProposalRequest) (hipAttachedDrafterAssistantDraftStepDeviceTokenResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipAttachedDrafterAssistantDraftStepDeviceTokenResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipAttachedDrafterAssistantDraftStepDeviceTokenResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "HIP driver is not available", nil)
	}
	if err := hipAttachedDrafterAssistantDraftStepProposalPlanInvalidReason(req.Plan, req.Softcap); err != nil {
		return hipAttachedDrafterAssistantDraftStepDeviceTokenResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "proposal plan", err)
	}
	hiddenResult, err := hipRunAttachedDrafterAssistantDraftStepHidden(ctx, driver, hipAttachedDrafterAssistantDraftStepHiddenRequest{
		LastToken:         req.LastToken,
		LastGreedyToken:   req.LastGreedyToken,
		TargetHidden:      req.TargetHidden,
		TargetForward:     req.TargetForward,
		TargetDeviceState: req.TargetDeviceState,
		Plan:              req.Plan,
		InputPlan:         req.InputPlan,
		Position:          req.Position,
		Epsilon:           req.Epsilon,
		Workspace:         req.Workspace,
	})
	if err != nil {
		return hipAttachedDrafterAssistantDraftStepDeviceTokenResult{}, err
	}
	labels := make(map[string]string, len(hiddenResult.Labels)+8)
	for key, value := range hiddenResult.Labels {
		labels[key] = value
	}
	for key, value := range hipAttachedDrafterAssistantDraftStepProposalRuntimeLabels(req.Plan, req.InputPlan, req.Softcap) {
		labels[key] = value
	}
	normed := hiddenResult.Normed
	hidden := hiddenResult.Hidden
	hiddenResult.Normed = nil
	hiddenResult.Hidden = nil
	hiddenResult.Labels = nil
	_ = hiddenResult.Close()
	if normed == nil || normed.Pointer() == 0 {
		_ = hidden.Close()
		return hipAttachedDrafterAssistantDraftStepDeviceTokenResult{}, core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "assistant normed hidden is required", nil)
	}
	defer normed.Close()
	success := false
	defer func() {
		if !success {
			_ = hidden.Close()
		}
	}()
	greedyToken, err := hipRunAttachedDrafterAssistantProposalTokenDevice(ctx, driver, normed, req.Plan, req.Softcap, req.SuppressTokens, req.Workspace)
	if err != nil {
		return hipAttachedDrafterAssistantDraftStepDeviceTokenResult{}, err
	}
	labels["attached_drafter_assistant_draft_step_logits"] = "not_retained"
	if hipAttachedDrafterAssistantUsesOrderedEmbeddingCandidates(req.Plan) {
		labels["attached_drafter_assistant_draft_step_token_source"] = "ordered_embedding_selected_greedy_device_deferred"
	} else {
		labels["attached_drafter_assistant_draft_step_token_source"] = "projection_greedy_device_deferred"
	}
	labels["attached_drafter_assistant_draft_step_token"] = "greedy"
	success = true
	return hipAttachedDrafterAssistantDraftStepDeviceTokenResult{
		GreedyToken: greedyToken,
		Hidden:      hidden,
		Labels:      labels,
	}, nil
}

func hipRunAttachedDrafterAssistantProjectionOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, plan hipAttachedDrafterAssistantProjectionPlan, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	switch plan.Encoding {
	case "bf16":
		if err := plan.BF16.validate(hipProjectionWeightEncodingBF16); err != nil {
			return core.E("rocm.hip.AttachedDrafterAssistantProjection", "validate BF16 projection", err)
		}
		return hipRunProjectionKernelWithDeviceInputWeightEncodingOutput(ctx, driver, input, plan.BF16.WeightPointer, plan.BF16.WeightBytes, plan.Rows, plan.Cols, hipProjectionWeightEncodingBF16, output)
	case "mlx_affine":
		if err := plan.MLXAffine.validateInputCount(plan.Cols); err != nil {
			return core.E("rocm.hip.AttachedDrafterAssistantProjection", "validate MLX affine projection", err)
		}
		return hipRunMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace(ctx, driver, input, plan.MLXAffine, output, workspace)
	default:
		return core.E("rocm.hip.AttachedDrafterAssistantProjection", "unsupported projection encoding", nil)
	}
}

func hipRunAttachedDrafterAssistantProposalToken(ctx context.Context, driver nativeHIPDriver, normed *hipDeviceByteBuffer, plan hipAttachedDrafterAssistantVerifierPlan, softcap float32, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (hipGreedySampleResult, *hipDeviceByteBuffer, error) {
	embedding := plan.Embedding
	if err := hipAttachedDrafterAssistantEmbeddingProjectionInvalidReason(embedding, normed, softcap); err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	switch embedding.TableEncoding {
	case hipEmbeddingTableEncodingBF16:
		logits, err := hipAllocateByteBuffer(driver, "rocm.hip.AttachedDrafterAssistantDraftStepProposal", "assistant dense logits", uint64(embedding.VocabSize*4), embedding.VocabSize)
		if err != nil {
			return hipGreedySampleResult{}, nil, err
		}
		success := false
		defer func() {
			if !success {
				_ = logits.Close()
			}
		}()
		if err := hipRunProjectionKernelWithDeviceInputWeightEncodingOutput(ctx, driver, normed, embedding.EmbeddingPointer, embedding.EmbeddingBytes, embedding.VocabSize, embedding.HiddenSize, hipProjectionWeightEncodingBF16, logits); err != nil {
			return hipGreedySampleResult{}, nil, err
		}
		var token hipGreedySampleResult
		if softcap > 0 {
			token, err = hipRunSoftcapGreedyKernelWithDeviceLogits(ctx, driver, logits, softcap)
		} else {
			token, err = hipRunGreedyKernelWithDeviceLogits(ctx, driver, logits)
		}
		if err != nil {
			return hipGreedySampleResult{}, nil, err
		}
		success = true
		return token, logits, nil
	case hipEmbeddingTableEncodingMLXQ4:
		cfg := hipMLXQ4DeviceWeightConfig{
			WeightPointer: embedding.EmbeddingPointer,
			ScalePointer:  embedding.ScalePointer,
			BiasPointer:   embedding.BiasPointer,
			WeightBytes:   embedding.EmbeddingBytes,
			ScaleBytes:    embedding.ScaleBytes,
			BiasBytes:     embedding.BiasBytes,
			Rows:          embedding.VocabSize,
			Cols:          embedding.HiddenSize,
			GroupSize:     embedding.GroupSize,
			Bits:          embedding.QuantBits,
		}
		if hipAttachedDrafterAssistantUsesOrderedEmbeddingCandidates(plan) {
			selected, err := hipAttachedDrafterAssistantOrderedEmbeddingSelectedTokens(ctx, driver, normed, plan, suppressTokens, workspace)
			if err != nil {
				return hipGreedySampleResult{}, nil, err
			}
			token, device, err := hipRunMLXQ4ProjectionSoftcapSelectedGreedyTokenKernelWithDeviceInputBufferResult(ctx, driver, normed, cfg, softcap, selected, nil, workspace)
			if err != nil {
				return hipGreedySampleResult{}, nil, err
			}
			return token, device, nil
		}
		token, _, err := hipRunMLXQ4ProjectionSoftcapGreedyTokenKernelWithDeviceInputBufferSuppressResult(ctx, driver, normed, cfg, softcap, nil, suppressTokens, workspace)
		if err != nil {
			return hipGreedySampleResult{}, nil, err
		}
		return token, nil, nil
	default:
		return hipGreedySampleResult{}, nil, core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "unsupported assistant embedding encoding", nil)
	}
}

func hipRunAttachedDrafterAssistantProposalTokenDevice(ctx context.Context, driver nativeHIPDriver, normed *hipDeviceByteBuffer, plan hipAttachedDrafterAssistantVerifierPlan, softcap float32, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, error) {
	embedding := plan.Embedding
	if err := hipAttachedDrafterAssistantEmbeddingProjectionInvalidReason(embedding, normed, softcap); err != nil {
		return nil, err
	}
	if embedding.TableEncoding != hipEmbeddingTableEncodingMLXQ4 {
		return nil, core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "device-deferred proposal requires MLX affine assistant embedding", nil)
	}
	cfg := hipMLXQ4DeviceWeightConfig{
		WeightPointer: embedding.EmbeddingPointer,
		ScalePointer:  embedding.ScalePointer,
		BiasPointer:   embedding.BiasPointer,
		WeightBytes:   embedding.EmbeddingBytes,
		ScaleBytes:    embedding.ScaleBytes,
		BiasBytes:     embedding.BiasBytes,
		Rows:          embedding.VocabSize,
		Cols:          embedding.HiddenSize,
		GroupSize:     embedding.GroupSize,
		Bits:          embedding.QuantBits,
	}
	if hipAttachedDrafterAssistantUsesOrderedEmbeddingCandidates(plan) {
		selected, err := hipAttachedDrafterAssistantOrderedEmbeddingSelectedTokens(ctx, driver, normed, plan, suppressTokens, workspace)
		if err != nil {
			return nil, err
		}
		return hipRunMLXQ4ProjectionSoftcapSelectedGreedyTokenKernelWithDeviceInputBufferDevice(ctx, driver, normed, cfg, softcap, selected, nil, workspace)
	}
	return hipRunMLXQ4ProjectionSoftcapGreedyTokenKernelWithDeviceInputBufferSuppressDevice(ctx, driver, normed, cfg, softcap, nil, suppressTokens, workspace)
}

func hipAttachedDrafterAssistantUsesOrderedEmbeddingCandidates(plan hipAttachedDrafterAssistantVerifierPlan) bool {
	return hipAttachedDrafterAssistantUsesDeviceOrderedEmbeddingCandidates(plan) ||
		hipAttachedDrafterAssistantUsesHostOrderedEmbeddingCandidates(plan)
}

func hipAttachedDrafterAssistantUsesDeviceOrderedEmbeddingCandidates(plan hipAttachedDrafterAssistantVerifierPlan) bool {
	return plan.Embedding.TableEncoding == hipEmbeddingTableEncodingMLXQ4 &&
		plan.MaskedCentroids.Encoding == "mlx_affine" &&
		plan.NumCentroids > 0 &&
		plan.TokensPerCentroid > 0 &&
		plan.TokenOrderingDeviceReady &&
		plan.TokenOrderingPointer != 0 &&
		plan.TokenOrderingBytes == uint64(plan.NumCentroids*plan.TokensPerCentroid*plan.TokenOrderingElementBytes) &&
		(plan.TokenOrderingElementBytes == 4 || plan.TokenOrderingElementBytes == 8)
}

func hipAttachedDrafterAssistantUsesHostOrderedEmbeddingCandidates(plan hipAttachedDrafterAssistantVerifierPlan) bool {
	return plan.Embedding.TableEncoding == hipEmbeddingTableEncodingMLXQ4 &&
		plan.MaskedCentroids.Encoding == "mlx_affine" &&
		plan.NumCentroids > 0 &&
		plan.TokensPerCentroid > 0 &&
		len(plan.TokenOrdering) == plan.NumCentroids*plan.TokensPerCentroid
}

func hipAttachedDrafterAssistantOrderedEmbeddingSelectedTokens(ctx context.Context, driver nativeHIPDriver, normed *hipDeviceByteBuffer, plan hipAttachedDrafterAssistantVerifierPlan, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceTokenBuffer, error) {
	if hipAttachedDrafterAssistantUsesDeviceOrderedEmbeddingCandidates(plan) {
		return hipAttachedDrafterAssistantOrderedEmbeddingSelectedTokensDevice(ctx, driver, normed, plan, suppressTokens, workspace)
	}
	return hipAttachedDrafterAssistantOrderedEmbeddingSelectedTokensHost(ctx, driver, normed, plan, suppressTokens, workspace)
}

func hipAttachedDrafterAssistantOrderedEmbeddingSelectedTokensDevice(ctx context.Context, driver nativeHIPDriver, normed *hipDeviceByteBuffer, plan hipAttachedDrafterAssistantVerifierPlan, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceTokenBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "ordered embedding candidate selection requires attention workspace", nil)
	}
	if !hipAttachedDrafterAssistantUsesDeviceOrderedEmbeddingCandidates(plan) {
		return nil, core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "device ordered embedding candidate plan is incomplete", nil)
	}
	topK := modelgemma4.AssistantCentroidIntermediateTopK
	if topK > plan.NumCentroids {
		topK = plan.NumCentroids
	}
	centroids, centroidCount, err := hipRunMLXQ4ProjectionSoftcapScoreTopKDeviceWithDeviceInputBufferSuppress(ctx, driver, normed, plan.MaskedCentroids.MLXAffine, 0, topK, nil, workspace)
	if err != nil {
		return nil, err
	}
	var suppress *hipDeviceTokenBuffer
	if len(suppressTokens) > 0 {
		suppress, err = workspace.EnsureSuppressTokenBuffer(driver, suppressTokens)
		if err != nil {
			return nil, err
		}
	}
	return hipRunOrderedEmbeddingCandidatesKernel(ctx, driver, centroids, centroidCount, plan.TokenOrderingPointer, plan.TokenOrderingBytes, plan.TokenOrderingElementBytes, plan.NumCentroids, plan.TokensPerCentroid, suppress, workspace)
}

func hipAttachedDrafterAssistantOrderedEmbeddingSelectedTokensHost(ctx context.Context, driver nativeHIPDriver, normed *hipDeviceByteBuffer, plan hipAttachedDrafterAssistantVerifierPlan, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceTokenBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "ordered embedding candidate selection requires attention workspace", nil)
	}
	if !hipAttachedDrafterAssistantUsesHostOrderedEmbeddingCandidates(plan) {
		return nil, core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "host ordered embedding candidate plan is incomplete", nil)
	}
	topK := modelgemma4.AssistantCentroidIntermediateTopK
	if topK > plan.NumCentroids {
		topK = plan.NumCentroids
	}
	centroids, err := hipRunMLXQ4ProjectionSoftcapScoreKernelWithDeviceInputBufferSuppress(ctx, driver, normed, plan.MaskedCentroids.MLXAffine, 0, topK, nil, workspace)
	if err != nil {
		return nil, err
	}
	suppressed := map[int32]struct{}{}
	for _, token := range suppressTokens {
		if token >= 0 {
			suppressed[token] = struct{}{}
		}
	}
	want := len(centroids) * plan.TokensPerCentroid
	tokens := workspace.ProjectionCandidateTokens[:0]
	if cap(tokens) < want {
		tokens = make([]int32, 0, want)
	}
	for _, centroid := range centroids {
		if centroid.TokenID < 0 || centroid.TokenID >= plan.NumCentroids {
			return nil, core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "ordered embedding centroid is outside range", nil)
		}
		start := centroid.TokenID * plan.TokensPerCentroid
		end := start + plan.TokensPerCentroid
		if start < 0 || end > len(plan.TokenOrdering) {
			return nil, core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "ordered embedding token-ordering range is invalid", nil)
		}
		for _, token := range plan.TokenOrdering[start:end] {
			if _, skip := suppressed[token]; skip {
				continue
			}
			tokens = append(tokens, token)
		}
	}
	if len(tokens) == 0 {
		return nil, core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "ordered embedding selected no candidate tokens", nil)
	}
	workspace.ProjectionCandidateTokens = tokens
	return workspace.EnsureSuppressTokenBuffer(driver, tokens)
}

func hipAttachedDrafterValidateGreedyTokenBuffer(buffer *hipDeviceByteBuffer) error {
	if buffer == nil || buffer.Pointer() == 0 || buffer.Count() != 1 || buffer.SizeBytes() != hipMLXQ4ProjectionBestBytes {
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStep", "packed greedy token buffer is required", nil)
	}
	return nil
}

func hipAttachedDrafterAssistantDraftStepProposalPlanInvalidReason(plan hipAttachedDrafterAssistantVerifierPlan, softcap float32) error {
	if plan.Status != attachedDrafterAssistantVerifierPlanTensorBound {
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "assistant verifier plan is not tensor-bound", nil)
	}
	if plan.HiddenSize <= 0 || plan.VocabSize <= 0 {
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "assistant hidden and vocab sizes must be positive", nil)
	}
	if softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) {
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "softcap must be non-negative and finite", nil)
	}
	if plan.Embedding.VocabSize != plan.VocabSize || plan.Embedding.HiddenSize != plan.HiddenSize {
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "assistant embedding shape must match plan hidden/vocab", nil)
	}
	if err := plan.Embedding.validateShape(); err != nil {
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "assistant embedding config", err)
	}
	switch plan.Embedding.TableEncoding {
	case hipEmbeddingTableEncodingBF16, hipEmbeddingTableEncodingMLXQ4:
		return nil
	default:
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "assistant embedding encoding is unsupported", nil)
	}
}

func hipAttachedDrafterAssistantEmbeddingProjectionInvalidReason(embedding hipDeviceEmbeddingLookupConfig, normed *hipDeviceByteBuffer, softcap float32) error {
	if err := embedding.validateShape(); err != nil {
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "assistant embedding config", err)
	}
	if normed == nil || normed.Pointer() == 0 ||
		normed.Count() != embedding.HiddenSize ||
		normed.SizeBytes() != uint64(embedding.HiddenSize*4) {
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "assistant normed hidden shape mismatch", nil)
	}
	if softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) {
		return core.E("rocm.hip.AttachedDrafterAssistantDraftStepProposal", "softcap must be non-negative and finite", nil)
	}
	return nil
}

func hipAttachedDrafterAssistantEmbeddingEncodingLabel(encoding uint32) string {
	switch encoding {
	case hipEmbeddingTableEncodingF32:
		return "f32"
	case hipEmbeddingTableEncodingBF16:
		return "bf16"
	case hipEmbeddingTableEncodingMLXQ4:
		return "mlx_affine"
	default:
		return strconv.FormatUint(uint64(encoding), 10)
	}
}
