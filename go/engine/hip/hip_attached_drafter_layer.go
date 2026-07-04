// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"strconv"
	"strings"

	core "dappco.re/go"
)

const (
	attachedDrafterAssistantLayerRuntimeNotReady = "not_ready"
	attachedDrafterAssistantLayerRuntimeLinked   = hipKernelStatusLinked
)

type hipAttachedDrafterAssistantLayerRequest struct {
	Hidden            *hipDeviceByteBuffer
	TargetLayer       hipGemma4Q4DeviceLayerKVState
	TargetLayerConfig hipGemma4Q4Layer0Config
	Plan              hipAttachedDrafterAssistantVerifierLayerPlan
	Position          int
	Epsilon           float32
	Workspace         *hipAttentionHeadsChunkedWorkspace
}

type hipAttachedDrafterAssistantLayerResult struct {
	Hidden *hipDeviceByteBuffer
	Labels map[string]string
}

func (result *hipAttachedDrafterAssistantLayerResult) Close() error {
	if result == nil {
		return nil
	}
	err := result.Hidden.Close()
	result.Hidden = nil
	result.Labels = nil
	return err
}

func hipAttachedDrafterAssistantLayerRuntimeLabels(plan hipAttachedDrafterAssistantVerifierPlan) map[string]string {
	status := attachedDrafterAssistantLayerRuntimeLinked
	reason := ""
	if plan.Status != attachedDrafterAssistantVerifierPlanTensorBound {
		status = attachedDrafterAssistantLayerRuntimeNotReady
		reason = "assistant verifier plan is " + firstNonEmptyString(plan.Status, "empty")
	} else if len(plan.Layers) == 0 {
		status = attachedDrafterAssistantLayerRuntimeNotReady
		reason = "assistant layer plan is empty"
	} else {
		for _, layer := range plan.Layers {
			if err := hipAttachedDrafterAssistantLayerPlanInvalidReason(layer); err != nil {
				status = attachedDrafterAssistantLayerRuntimeNotReady
				reason = err.Error()
				break
			}
		}
	}
	labels := map[string]string{
		"attached_drafter_assistant_layer_runtime": status,
		"attached_drafter_assistant_layer_kv":      "target_device",
	}
	if reason != "" {
		labels["attached_drafter_assistant_layer_runtime_reason"] = reason
	}
	if len(plan.Layers) > 0 {
		labels["attached_drafter_assistant_layer_runtime_layers"] = strconv.Itoa(len(plan.Layers))
	}
	labels["attached_drafter_assistant_layer_kernel_families"] = strings.Join([]string{
		hipKernelNameRMSNorm,
		hipAttachedDrafterAssistantVerifierProjectionKernel(plan.ProjectionEncoding == "mlx_affine"),
		hipKernelNameAttentionHeads,
		hipAttachedDrafterAssistantVerifierGELUKernel(plan.ProjectionEncoding == "mlx_affine"),
		hipKernelNameVectorAdd,
		hipKernelNameVectorAddScaled,
	}, ",")
	return labels
}

func hipAttachedDrafterAssistantLayerPlanInvalidReason(plan hipAttachedDrafterAssistantVerifierLayerPlan) error {
	if plan.HiddenSize <= 0 || plan.HeadDim <= 0 || plan.QueryHeads <= 0 {
		return core.E("rocm.hip.AttachedDrafterAssistantLayer", "assistant layer hidden/head geometry is missing", nil)
	}
	if plan.QueryProjection.Rows != plan.QueryHeads*plan.HeadDim || plan.QueryProjection.Cols != plan.HiddenSize {
		return core.E("rocm.hip.AttachedDrafterAssistantLayer", "assistant q_proj shape mismatch", nil)
	}
	if plan.OutputProjection.Rows != plan.HiddenSize || plan.OutputProjection.Cols != plan.QueryHeads*plan.HeadDim {
		return core.E("rocm.hip.AttachedDrafterAssistantLayer", "assistant o_proj shape mismatch", nil)
	}
	if plan.GateProjection.Rows <= 0 ||
		plan.GateProjection.Rows != plan.UpProjection.Rows ||
		plan.GateProjection.Cols != plan.HiddenSize ||
		plan.UpProjection.Cols != plan.HiddenSize ||
		plan.DownProjection.Rows != plan.HiddenSize ||
		plan.DownProjection.Cols != plan.GateProjection.Rows {
		return core.E("rocm.hip.AttachedDrafterAssistantLayer", "assistant MLP projection shape mismatch", nil)
	}
	if plan.RoPEBase <= 0 || plan.RoPERotaryDim <= 0 || plan.RoPERotaryDim > plan.HeadDim || plan.RoPERotaryDim%2 != 0 {
		return core.E("rocm.hip.AttachedDrafterAssistantLayer", "assistant RoPE geometry is invalid", nil)
	}
	if plan.RoPEFrequencyScale < 0 {
		return core.E("rocm.hip.AttachedDrafterAssistantLayer", "assistant RoPE frequency scale is invalid", nil)
	}
	for label, norm := range map[string]hipRMSNormDeviceWeightConfig{
		"input_layernorm":            plan.InputNorm,
		"post_attention_layernorm":   plan.PostAttentionNorm,
		"pre_feedforward_layernorm":  plan.PreFeedforward,
		"post_feedforward_layernorm": plan.PostFeedforward,
	} {
		if norm.Count != plan.HiddenSize {
			return core.E("rocm.hip.AttachedDrafterAssistantLayer", label+" count must match hidden size", nil)
		}
		if err := hipValidateRMSNormDeviceWeightConfig("AttachedDrafterAssistantLayer."+label, norm); err != nil {
			return err
		}
	}
	if plan.QueryNorm.Count != plan.HeadDim {
		return core.E("rocm.hip.AttachedDrafterAssistantLayer", "q_norm count must match head dim", nil)
	}
	if err := hipValidateRMSNormDeviceWeightConfig("AttachedDrafterAssistantLayer.q_norm", plan.QueryNorm); err != nil {
		return err
	}
	if err := hipAttachedDrafterAssistantProjectionPlanValid(plan.QueryProjection, plan.HiddenSize); err != nil {
		return err
	}
	if err := hipAttachedDrafterAssistantProjectionPlanValid(plan.OutputProjection, plan.QueryHeads*plan.HeadDim); err != nil {
		return err
	}
	if err := hipAttachedDrafterAssistantProjectionPlanValid(plan.GateProjection, plan.HiddenSize); err != nil {
		return err
	}
	if err := hipAttachedDrafterAssistantProjectionPlanValid(plan.UpProjection, plan.HiddenSize); err != nil {
		return err
	}
	if err := hipAttachedDrafterAssistantProjectionPlanValid(plan.DownProjection, plan.GateProjection.Rows); err != nil {
		return err
	}
	return nil
}

func hipAttachedDrafterAssistantProjectionPlanValid(plan hipAttachedDrafterAssistantProjectionPlan, inputCount int) error {
	if plan.Rows <= 0 || plan.Cols != inputCount {
		return core.E("rocm.hip.AttachedDrafterAssistantLayer", "assistant projection dimensions are invalid", nil)
	}
	switch plan.Encoding {
	case "bf16":
		return plan.BF16.validate(hipProjectionWeightEncodingBF16)
	case "mlx_affine":
		return plan.MLXAffine.validateInputCount(inputCount)
	default:
		return core.E("rocm.hip.AttachedDrafterAssistantLayer", "assistant projection encoding is unsupported", nil)
	}
}

func hipRunAttachedDrafterAssistantLayer(ctx context.Context, driver nativeHIPDriver, req hipAttachedDrafterAssistantLayerRequest) (hipAttachedDrafterAssistantLayerResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipAttachedDrafterAssistantLayerResult{}, core.E("rocm.hip.AttachedDrafterAssistantLayer", "HIP driver is not available", nil)
	}
	if err := hipAttachedDrafterAssistantLayerPlanInvalidReason(req.Plan); err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	if req.Hidden == nil || req.Hidden.Pointer() == 0 ||
		req.Hidden.Count() != req.Plan.HiddenSize ||
		req.Hidden.SizeBytes() != uint64(req.Plan.HiddenSize*4) {
		return hipAttachedDrafterAssistantLayerResult{}, core.E("rocm.hip.AttachedDrafterAssistantLayer", "assistant hidden device buffer shape mismatch", nil)
	}
	if req.TargetLayer.cache == nil || req.TargetLayer.cache.closed || req.TargetLayer.descriptorTable == nil || req.TargetLayer.descriptorTable.closed {
		return hipAttachedDrafterAssistantLayerResult{}, core.E("rocm.hip.AttachedDrafterAssistantLayer", "target device KV layer is required", nil)
	}
	if err := req.TargetLayer.descriptorTable.CompatibleWith(req.TargetLayer.cache); err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, core.E("rocm.hip.AttachedDrafterAssistantLayer", "target device KV descriptor", err)
	}
	targetKeyHeads, targetKVWidth, err := hipAttachedDrafterAssistantTargetAttentionGeometry(req.TargetLayerConfig, req.Plan)
	if err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	keyWidth, valueWidth, ok := req.TargetLayer.cache.LastVectorWidths()
	if !ok || keyWidth != targetKVWidth || valueWidth != targetKVWidth {
		return hipAttachedDrafterAssistantLayerResult{}, core.E("rocm.hip.AttachedDrafterAssistantLayer", "target device KV width mismatch", nil)
	}
	if req.Workspace == nil {
		req.Workspace = &hipAttentionHeadsChunkedWorkspace{}
		defer req.Workspace.Close()
	}

	inputNormCfg := req.Plan.InputNorm
	inputNormCfg.Epsilon = req.Epsilon
	layerInput, err := req.Workspace.EnsureRMSNormOutput(driver, req.Plan.HiddenSize)
	if err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	if err := hipRunRMSNormDeviceToDeviceKernelWithWorkspace(ctx, driver, req.Hidden.Pointer(), req.Hidden.SizeBytes(), layerInput.Pointer(), layerInput.SizeBytes(), inputNormCfg, req.Workspace); err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}

	query, err := req.Workspace.EnsureProjectionOutput(driver, req.Plan.QueryProjection.Rows)
	if err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	if err := hipRunAttachedDrafterAssistantProjectionOutput(ctx, driver, layerInput, req.Plan.QueryProjection, query, req.Workspace); err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	queryNormCfg := hipGemma4Q4RoPENormConfig(req.Plan.QueryNorm, req.Epsilon, req.Plan.HeadDim)
	ropeFrequencyDim, ropeRotaryCount := hipAttachedDrafterAssistantLayerRoPEKernelDims(req.Plan)
	ropeFrequencyScale := req.Plan.RoPEFrequencyScale
	if ropeFrequencyScale == 0 {
		ropeFrequencyScale = 1
	}
	ropeQuery, err := req.Workspace.EnsureRMSRoPEOutput(driver, req.Plan.QueryProjection.Rows)
	if err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	if err := hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutputFrequencyScaleWithWorkspace(ctx, driver, query, queryNormCfg, req.Plan.QueryHeads, req.Position, req.Plan.RoPEBase, ropeFrequencyDim, ropeRotaryCount, ropeFrequencyScale, ropeQuery, req.Workspace); err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}

	attentionOutput, err := req.Workspace.EnsureAttentionOutput(driver, req.Plan.QueryHeads, req.Plan.HeadDim)
	if err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	attentionReq := hipAttentionRequest{
		QueryDim:        req.Plan.HeadDim,
		KeyHeads:        targetKeyHeads,
		DeviceKV:        req.TargetLayer.cache,
		DescriptorTable: req.TargetLayer.descriptorTable,
		WindowSize:      req.Plan.SlidingWindow,
		Scale:           hipGemma4Q4AttentionScale(req.Plan.HeadDim),
	}
	if err := hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernelWithWorkspace(ctx, driver, attentionReq, ropeQuery, req.Plan.QueryHeads, attentionOutput, req.Workspace); err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}

	attentionProjection, err := req.Workspace.EnsureIntermediateOutput(driver, req.Plan.HiddenSize)
	if err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	if err := hipRunAttachedDrafterAssistantProjectionOutput(ctx, driver, attentionOutput, req.Plan.OutputProjection, attentionProjection, req.Workspace); err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	postAttentionNormCfg := req.Plan.PostAttentionNorm
	postAttentionNormCfg.Epsilon = req.Epsilon
	attentionNorm, err := req.Workspace.EnsureRMSNormOutput(driver, req.Plan.HiddenSize)
	if err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	if err := hipRunRMSNormDeviceToDeviceKernelWithWorkspace(ctx, driver, attentionProjection.Pointer(), attentionProjection.SizeBytes(), attentionNorm.Pointer(), attentionNorm.SizeBytes(), postAttentionNormCfg, req.Workspace); err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	attentionResidual, err := req.Workspace.EnsureRMSResidualOutput(driver, req.Plan.HiddenSize)
	if err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	if err := hipRunVectorAddDeviceKernelOutput(ctx, driver, req.Hidden, attentionNorm, attentionResidual); err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}

	preFeedforwardNormCfg := req.Plan.PreFeedforward
	preFeedforwardNormCfg.Epsilon = req.Epsilon
	ffInput, err := req.Workspace.EnsureRMSNormOutput(driver, req.Plan.HiddenSize)
	if err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	if err := hipRunRMSNormDeviceToDeviceKernelWithWorkspace(ctx, driver, attentionResidual.Pointer(), attentionResidual.SizeBytes(), ffInput.Pointer(), ffInput.SizeBytes(), preFeedforwardNormCfg, req.Workspace); err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	mlpOutput, err := req.Workspace.EnsureProjectionOutput(driver, req.Plan.HiddenSize)
	if err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	if err := hipRunAttachedDrafterAssistantMLPOutput(ctx, driver, ffInput, req.Plan.GateProjection, req.Plan.UpProjection, req.Plan.DownProjection, mlpOutput, req.Workspace); err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	postFeedforwardNormCfg := req.Plan.PostFeedforward
	postFeedforwardNormCfg.Epsilon = req.Epsilon
	ffResidual, err := req.Workspace.EnsureRMSNormOutput(driver, req.Plan.HiddenSize)
	if err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	if err := hipRunRMSNormDeviceToDeviceKernelWithWorkspace(ctx, driver, mlpOutput.Pointer(), mlpOutput.SizeBytes(), ffResidual.Pointer(), ffResidual.SizeBytes(), postFeedforwardNormCfg, req.Workspace); err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}

	output, err := hipAllocateByteBuffer(driver, "rocm.hip.AttachedDrafterAssistantLayer", "assistant layer hidden output", uint64(req.Plan.HiddenSize*4), req.Plan.HiddenSize)
	if err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	layerScalar := req.Plan.LayerScalar
	if layerScalar == 0 {
		layerScalar = 1
	}
	if err := hipRunVectorAddScaledDeviceKernelOutputWithWorkspace(ctx, driver, attentionResidual, ffResidual, layerScalar, output, req.Workspace); err != nil {
		return hipAttachedDrafterAssistantLayerResult{}, err
	}
	labels := map[string]string{
		"attached_drafter_assistant_layer_runtime":          attachedDrafterAssistantLayerRuntimeLinked,
		"attached_drafter_assistant_layer":                  strconv.Itoa(req.Plan.Layer),
		"attached_drafter_assistant_layer_type":             req.Plan.LayerType,
		"attached_drafter_assistant_layer_target_kv":        "device",
		"attached_drafter_assistant_layer_target_tokens":    strconv.Itoa(req.TargetLayer.cache.TokenCount()),
		"attached_drafter_assistant_layer_target_key_heads": strconv.Itoa(targetKeyHeads),
		"attached_drafter_assistant_layer_target_kv_width":  strconv.Itoa(targetKVWidth),
		"attached_drafter_assistant_layer_projection_mode":  req.Plan.QueryProjection.Encoding,
	}
	success = true
	return hipAttachedDrafterAssistantLayerResult{Hidden: output, Labels: labels}, nil
}

func hipAttachedDrafterAssistantLayerRoPEKernelDims(plan hipAttachedDrafterAssistantVerifierLayerPlan) (frequencyDim, rotaryCount int) {
	if plan.RoPERotaryDim != plan.HeadDim {
		return plan.HeadDim, plan.RoPERotaryDim
	}
	return 0, 0
}

func hipRunAttachedDrafterAssistantMLPOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, gate, up, down hipAttachedDrafterAssistantProjectionPlan, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if gate.Encoding == "mlx_affine" && up.Encoding == "mlx_affine" && down.Encoding == "mlx_affine" {
		return hipRunGemma4Q4DeviceGELUTanhMLPWithDeviceInputOutput(ctx, driver, input, gate.MLXAffine, up.MLXAffine, down.MLXAffine, output, workspace)
	}
	if gate.Encoding != "bf16" || up.Encoding != "bf16" || down.Encoding != "bf16" {
		return core.E("rocm.hip.AttachedDrafterAssistantLayer", "assistant MLP projection encodings must match", nil)
	}
	gateOutput, err := hipAllocateByteBuffer(driver, "rocm.hip.AttachedDrafterAssistantLayer", "assistant gate projection output", uint64(gate.Rows*4), gate.Rows)
	if err != nil {
		return err
	}
	defer gateOutput.Close()
	if err := hipRunAttachedDrafterAssistantProjectionOutput(ctx, driver, input, gate, gateOutput, workspace); err != nil {
		return err
	}
	upOutput, err := hipAllocateByteBuffer(driver, "rocm.hip.AttachedDrafterAssistantLayer", "assistant up projection output", uint64(up.Rows*4), up.Rows)
	if err != nil {
		return err
	}
	defer upOutput.Close()
	if err := hipRunAttachedDrafterAssistantProjectionOutput(ctx, driver, input, up, upOutput, workspace); err != nil {
		return err
	}
	activated, err := hipRunGELUTanhMultiplyDeviceKernel(ctx, driver, gateOutput, upOutput)
	if err != nil {
		return err
	}
	defer activated.Close()
	return hipRunAttachedDrafterAssistantProjectionOutput(ctx, driver, activated, down, output, workspace)
}

func hipAttachedDrafterAssistantTargetAttentionGeometry(targetCfg hipGemma4Q4Layer0Config, plan hipAttachedDrafterAssistantVerifierLayerPlan) (int, int, error) {
	if plan.HeadDim <= 0 || plan.QueryHeads <= 0 {
		return 0, 0, core.E("rocm.hip.AttachedDrafterAssistantLayer", "assistant attention geometry is missing", nil)
	}
	if targetCfg.HeadDim <= 0 {
		return 0, 0, core.E("rocm.hip.AttachedDrafterAssistantLayer", "target attention geometry is missing", nil)
	}
	if targetCfg.HeadDim != plan.HeadDim {
		return 0, 0, core.E("rocm.hip.AttachedDrafterAssistantLayer", "target attention head dimension mismatch", nil)
	}
	targetKeyHeads := firstPositiveInt(targetCfg.KeyHeads, 1)
	if targetKeyHeads <= 0 || targetKeyHeads > plan.QueryHeads || plan.QueryHeads%targetKeyHeads != 0 {
		return 0, 0, core.E("rocm.hip.AttachedDrafterAssistantLayer", "target key head count must divide assistant query head count", nil)
	}
	return targetKeyHeads, targetCfg.keyValueDim(), nil
}

func hipAttachedDrafterAssistantTargetLayerFor(layerType string, cfg hipGemma4Q4ForwardConfig, state *hipGemma4Q4DeviceDecodeState) (hipGemma4Q4DeviceLayerKVState, hipGemma4Q4Layer0Config, int, error) {
	if layerType == "" {
		return hipGemma4Q4DeviceLayerKVState{}, hipGemma4Q4Layer0Config{}, -1, core.E("rocm.hip.AttachedDrafterAssistantLayer", "assistant layer type is required", nil)
	}
	if state == nil || state.closed || state.LayerCount() == 0 {
		return hipGemma4Q4DeviceLayerKVState{}, hipGemma4Q4Layer0Config{}, -1, core.E("rocm.hip.AttachedDrafterAssistantLayer", "target device state is required", nil)
	}
	if len(cfg.Layers) == 0 || len(cfg.Layers) != state.LayerCount() {
		return hipGemma4Q4DeviceLayerKVState{}, hipGemma4Q4Layer0Config{}, -1, core.E("rocm.hip.AttachedDrafterAssistantLayer", "target forward config must match device state", nil)
	}
	sources := hipGemma4Q4SharedKVSourceByLayer(cfg)
	selected := -1
	for index, layer := range cfg.Layers {
		if layer.LayerType != layerType {
			continue
		}
		source := index
		if index < len(sources) && sources[index] >= 0 {
			source = sources[index]
		}
		if source >= 0 && source < len(state.layers) && state.layers[source].cache != nil {
			selected = source
		}
	}
	if selected < 0 {
		return hipGemma4Q4DeviceLayerKVState{}, hipGemma4Q4Layer0Config{}, -1, core.E("rocm.hip.AttachedDrafterAssistantLayer", "target device KV stream is missing for "+layerType, nil)
	}
	return state.layers[selected], cfg.Layers[selected], selected, nil
}
