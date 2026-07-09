// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func (model *hipLoadedModel) loadedGemma4Q4PackageForwardConfig() (hipGemma4Q4ForwardConfig, bool, error) {
	if model == nil {
		return hipGemma4Q4ForwardConfig{}, false, nil
	}
	if !hipLoadedGemma4Q4GenerateLinked(model) {
		return hipGemma4Q4ForwardConfig{}, false, nil
	}
	if model.modelInfo.NumLayers <= 0 {
		return hipGemma4Q4ForwardConfig{}, true, core.E(hipGemma4Q4Layer0Operation, "loaded Gemma4 MLX affine layer count is required", nil)
	}
	cfg, err := model.cachedGemma4Q4ForwardConfig(model.modelInfo.NumLayers)
	return cfg, true, err
}

func hipRunGemma4Q4PackagePrefill(ctx context.Context, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, req hipPrefillRequest) (hipPrefillResult, error) {
	if model == nil {
		return hipPrefillResult{}, core.E(hipGemma4Q4Layer0Operation, "loaded model is required", nil)
	}
	tokens, err := req.resolvedTokenIDs(model)
	if err != nil {
		return hipPrefillResult{}, err
	}
	mode, err := hipGemma4Q4PackagePrefillKVMode(model, cfg, req)
	if err != nil {
		return hipPrefillResult{}, err
	}
	state := hipGemma4Q4DecodeState{}
	var deviceState *hipGemma4Q4DeviceDecodeState
	success := false
	defer func() {
		if !success {
			_ = deviceState.Close()
		}
	}()
	position := 0
	var current hipGemma4Q4ForwardResult
	for _, tokenID := range tokens {
		current, state, err = hipRunGemma4Q4SingleTokenForwardWithState(ctx, model.driver, cfg, state, hipGemma4Q4ForwardRequest{
			TokenID:           tokenID,
			Position:          position,
			Epsilon:           1e-6,
			DeviceKVAttention: true,
			DeviceKVMode:      mode,
			PriorDeviceState:  deviceState,
			ReturnDeviceState: true,
		})
		if err != nil {
			return hipPrefillResult{}, err
		}
		if current.DeviceState == nil {
			return hipPrefillResult{}, core.E(hipGemma4Q4Layer0Operation, "forward did not return device KV state", nil)
		}
		deviceState = current.DeviceState
		current.DeviceState = nil
		position++
	}
	labels := hipGemma4Q4PackagePrefillLabels(cfg, mode, len(tokens), current.Labels, deviceState)
	success = true
	return hipPrefillResult{
		Logits:              current.Logits,
		PromptTokens:        len(tokens),
		Gemma4Q4State:       state,
		Gemma4Q4DeviceState: deviceState,
		Labels:              labels,
	}, nil
}

func hipRunGemma4Q4PackageDecode(ctx context.Context, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, req hipDecodeRequest) (hipDecodeResult, error) {
	if model == nil {
		return hipDecodeResult{}, core.E(hipGemma4Q4Layer0Operation, "loaded model is required", nil)
	}
	if req.TokenID < 0 {
		return hipDecodeResult{}, core.E("rocm.hip.Decode", "token ID must be non-negative", nil)
	}
	if len(req.Gemma4Q4State.Layers) == 0 {
		return hipDecodeResult{}, core.E("rocm.hip.Decode", "Gemma4 q4 decode state is required", nil)
	}
	if err := cfg.validate(); err != nil {
		return hipDecodeResult{}, err
	}
	if err := req.Gemma4Q4State.validate(cfg); err != nil {
		return hipDecodeResult{}, err
	}
	mode, err := hipGemma4Q4PackageDecodeKVMode(model, req)
	if err != nil {
		return hipDecodeResult{}, err
	}
	position, err := hipGemma4Q4PackageDecodePosition(cfg, req)
	if err != nil {
		return hipDecodeResult{}, err
	}
	current, state, err := hipRunGemma4Q4SingleTokenForwardWithState(ctx, model.driver, cfg, req.Gemma4Q4State, hipGemma4Q4ForwardRequest{
		TokenID:           req.TokenID,
		Position:          position,
		Epsilon:           1e-6,
		DeviceKVAttention: true,
		DeviceKVMode:      mode,
		PriorDeviceState:  req.Gemma4Q4DeviceState,
		ReturnDeviceState: true,
	})
	if err != nil {
		return hipDecodeResult{}, err
	}
	if current.DeviceState == nil {
		return hipDecodeResult{}, core.E(hipGemma4Q4Layer0Operation, "forward did not return device KV state", nil)
	}
	deviceState := current.DeviceState
	current.DeviceState = nil
	labels := hipGemma4Q4PackageDecodeLabels(cfg, mode, state, current.Labels, deviceState)
	tokenID := int32(current.Greedy.TokenID)
	return hipDecodeResult{
		Token: inference.Token{
			ID:   tokenID,
			Text: hipGeneratedTokenText(model, tokenID),
		},
		Logits:              current.Logits,
		Gemma4Q4State:       state,
		Gemma4Q4DeviceState: deviceState,
		Labels:              labels,
	}, nil
}

func hipGemma4Q4PackagePrefillKVMode(model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, req hipPrefillRequest) (string, error) {
	mode := firstNonEmptyString(req.CacheMode, model.gemma4Q4EngineConfig().DeviceKVMode, rocmKVCacheModeKQ8VQ4)
	if !isROCmKVCacheMode(mode) {
		return "", core.E("rocm.hip.Prefill", core.Sprintf("unsupported cache mode %q", mode), nil)
	}
	if req.KeyWidth > 0 || req.ValueWidth > 0 {
		keyWidth, valueWidth, err := hipKVVectorWidths(req.KeyWidth, req.ValueWidth)
		if err != nil {
			return "", core.E("rocm.hip.Prefill", "invalid KV vector widths", err)
		}
		for index, layer := range cfg.Layers {
			if keyWidth != layer.HeadDim || valueWidth != layer.HeadDim {
				return "", core.E("rocm.hip.Prefill", core.Sprintf("Gemma4 q4 layer %d KV widths must match head dimension", index), nil)
			}
		}
	}
	return mode, nil
}

func hipGemma4Q4PackageDecodeKVMode(model *hipLoadedModel, req hipDecodeRequest) (string, error) {
	mode := req.DeviceKVMode
	if mode == "" && req.Gemma4Q4DeviceState != nil {
		mode = req.Gemma4Q4DeviceState.mode
	}
	mode = firstNonEmptyString(mode, model.gemma4Q4EngineConfig().DeviceKVMode, rocmKVCacheModeKQ8VQ4)
	if !isROCmKVCacheMode(mode) {
		return "", core.E("rocm.hip.Decode", core.Sprintf("unsupported device KV cache mode %q", mode), nil)
	}
	return mode, nil
}

func hipGemma4Q4PackageDecodePosition(cfg hipGemma4Q4ForwardConfig, req hipDecodeRequest) (int, error) {
	if req.Position < 0 {
		return 0, core.E("rocm.hip.Decode", "decode position must be non-negative", nil)
	}
	if req.Position > 0 {
		return req.Position, nil
	}
	position := req.Gemma4Q4State.tokenCountForConfig(cfg)
	if devicePosition := req.Gemma4Q4DeviceState.maxLayerTokenCount(); devicePosition > position {
		position = devicePosition
	}
	return position, nil
}

func hipGemma4Q4PackagePrefillLabels(cfg hipGemma4Q4ForwardConfig, mode string, tokenCount int, forwardLabels map[string]string, deviceState *hipGemma4Q4DeviceDecodeState) map[string]string {
	labels := cloneStringMap(forwardLabels)
	if labels == nil {
		labels = map[string]string{}
	}
	labels["attention_kv_backing"] = "hip_device_descriptor"
	labels["attention_kv_mode"] = mode
	labels["gemma4_q4_device_kv_state"] = "forward_returned_device_state"
	labels["gemma4_q4_prefill_kernel"] = hipKernelStatusLinked
	labels["gemma4_q4_prefill_name"] = "rocm_gemma4_q4_package_prefill_experimental"
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_prefill"
	labels["prefill_kernel"] = hipKernelStatusNotLinked
	labels["prefill_prompt_tokens"] = core.Sprintf("%d", tokenCount)
	labels["production_prefill"] = hipKernelStatusNotLinked
	labels["production_decode"] = hipKernelStatusNotLinked
	labels["production_kv_cache_backing"] = hipKernelStatusNotLinked
	labels["runtime_status"] = string(inference.FeatureRuntimeExperimental)
	if len(cfg.Layers) > 0 {
		labels["prefill_layers"] = core.Sprintf("%d", len(cfg.Layers))
	}
	for key, value := range deviceState.Labels() {
		labels[key] = value
	}
	return labels
}

func hipGemma4Q4PackageDecodeLabels(cfg hipGemma4Q4ForwardConfig, mode string, state hipGemma4Q4DecodeState, forwardLabels map[string]string, deviceState *hipGemma4Q4DeviceDecodeState) map[string]string {
	labels := cloneStringMap(forwardLabels)
	if labels == nil {
		labels = map[string]string{}
	}
	labels["attention_kv_backing"] = "hip_device_descriptor"
	labels["attention_kv_mode"] = mode
	labels["decode_kernel"] = hipKernelStatusNotLinked
	labels["gemma4_q4_decode_kernel"] = hipKernelStatusLinked
	labels["gemma4_q4_decode_name"] = "rocm_gemma4_q4_package_decode_experimental"
	labels["gemma4_q4_device_kv_state"] = "forward_returned_device_state"
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_decode"
	labels["production_decode"] = hipKernelStatusNotLinked
	labels["production_prefill"] = hipKernelStatusNotLinked
	labels["production_kv_cache_backing"] = hipKernelStatusNotLinked
	labels["runtime_status"] = string(inference.FeatureRuntimeExperimental)
	if len(cfg.Layers) > 0 {
		labels["decode_state_tokens"] = core.Sprintf("%d", state.tokenCountForConfig(cfg))
	}
	for key, value := range deviceState.Labels() {
		labels[key] = value
	}
	return labels
}
