// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"strconv"
	"strings"

	"dappco.re/go/inference"
)

const ROCmTokenLoopContract = "rocm-token-loop-v1"

// ROCmTokenLoopStatus is the application-facing decode contract for ROCm text
// models. It mirrors go-mlx's token-loop/session split in ROCm terms: text
// generation is driven through the shared inference.TextModel surface, while
// Gemma4 production routes advertise the retained StateSession as the required
// incremental fast path instead of prompt replay.
type ROCmTokenLoopStatus struct {
	Contract              string                         `json:"contract,omitempty"`
	Architecture          string                         `json:"architecture,omitempty"`
	Family                string                         `json:"family,omitempty"`
	Runtime               string                         `json:"runtime,omitempty"`
	RuntimeStatus         inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	TextModel             bool                           `json:"text_model,omitempty"`
	TokenLoop             bool                           `json:"token_loop,omitempty"`
	EmbedBookend          bool                           `json:"embed_bookend,omitempty"`
	DecodeForward         bool                           `json:"decode_forward,omitempty"`
	LMHeadBookend         bool                           `json:"lm_head_bookend,omitempty"`
	SharedGenerateLoop    bool                           `json:"shared_generate_loop,omitempty"`
	IncrementalSession    bool                           `json:"incremental_session,omitempty"`
	SessionState          string                         `json:"session_state,omitempty"`
	CloseSession          bool                           `json:"close_session,omitempty"`
	StepWithID            bool                           `json:"step_with_id,omitempty"`
	PerLayerInputs        bool                           `json:"per_layer_inputs,omitempty"`
	RuntimeOwnedKV        bool                           `json:"runtime_owned_kv,omitempty"`
	DeviceKVState         bool                           `json:"device_kv_state,omitempty"`
	PromptReplayRefused   bool                           `json:"prompt_replay_refused,omitempty"`
	RetainedStateRequired bool                           `json:"retained_state_required,omitempty"`
	FastPath              string                         `json:"fast_path,omitempty"`
	FallbackPath          string                         `json:"fallback_path,omitempty"`
	Reference             string                         `json:"reference,omitempty"`
	Labels                map[string]string              `json:"labels,omitempty"`
}

func (status ROCmTokenLoopStatus) Clone() ROCmTokenLoopStatus {
	status.Labels = cloneStringMap(status.Labels)
	return status
}

func (status ROCmTokenLoopStatus) Matched() bool {
	return status.Contract != "" && status.Architecture != "" && status.TokenLoop
}

func (status ROCmTokenLoopStatus) IncrementalDecodeReady() bool {
	return status.TokenLoop &&
		status.IncrementalSession &&
		status.RuntimeOwnedKV &&
		status.DeviceKVState &&
		status.RetainedStateRequired &&
		status.PromptReplayRefused
}

func ROCmTokenLoopForIdentity(path string, model inference.ModelIdentity) (ROCmTokenLoopStatus, bool) {
	retained, ok := ROCmRetainedStateForIdentity(path, model)
	if !ok {
		return ROCmTokenLoopStatus{}, false
	}
	return rocmTokenLoopStatusFromRetained(retained), true
}

func ROCmTokenLoopForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmTokenLoopStatus, bool) {
	retained, ok := ROCmRetainedStateForInfo(path, info, labels)
	if !ok {
		return ROCmTokenLoopStatus{}, false
	}
	return rocmTokenLoopStatusFromRetained(retained), true
}

func ROCmTokenLoopForInspection(inspection *inference.ModelPackInspection) (ROCmTokenLoopStatus, bool) {
	retained, ok := ROCmRetainedStateForInspection(inspection)
	if !ok {
		return ROCmTokenLoopStatus{}, false
	}
	return rocmTokenLoopStatusFromRetained(retained), true
}

func ROCmTokenLoopForModel(model inference.TextModel) (ROCmTokenLoopStatus, bool) {
	retained, ok := ROCmRetainedStateForModel(model)
	if !ok {
		return ROCmTokenLoopStatus{}, false
	}
	return rocmTokenLoopStatusFromRetained(retained), true
}

func rocmTokenLoopStatusFromRetained(retained ROCmRetainedStateStatus) ROCmTokenLoopStatus {
	labels := cloneStringMap(retained.Labels)
	stepWithID := rocmTokenLoopNeedsIDStep(retained, labels)
	status := ROCmTokenLoopStatus{
		Contract:              ROCmTokenLoopContract,
		Architecture:          retained.Architecture,
		Family:                retained.Family,
		Runtime:               retained.Runtime,
		RuntimeStatus:         retained.RuntimeStatus,
		TextModel:             retained.StateSession,
		TokenLoop:             retained.StateSession,
		EmbedBookend:          retained.StateSession,
		DecodeForward:         retained.StateSession,
		LMHeadBookend:         retained.StateSession,
		SharedGenerateLoop:    retained.StateSession,
		IncrementalSession:    retained.RuntimeOwnedKV && retained.DeviceKVState,
		CloseSession:          retained.StateSession,
		StepWithID:            stepWithID,
		PerLayerInputs:        stepWithID,
		RuntimeOwnedKV:        retained.RuntimeOwnedKV,
		DeviceKVState:         retained.DeviceKVState,
		PromptReplayRefused:   retained.PromptReplayRefused,
		RetainedStateRequired: retained.RetainedStateRequired,
		FastPath:              "retained-state-session",
		FallbackPath:          "text-model-generate",
		Reference:             "go_mlx_session_model",
		Labels:                labels,
	}
	if retained.StateSession {
		status.SessionState = "StateSession"
	}
	if !status.IncrementalDecodeReady() {
		status.FastPath = "metadata-only"
	}
	status.Labels = rocmTokenLoopLabels(status)
	return status.Clone()
}

func rocmTokenLoopNeedsIDStep(retained ROCmRetainedStateStatus, labels map[string]string) bool {
	if labels["gemma4_per_layer_inputs"] == "true" ||
		labels["per_layer_inputs"] == "true" ||
		labels["gemma4_hidden_size_per_layer_input"] != "" ||
		labels["gemma4_vocab_size_per_layer_input"] != "" {
		return true
	}
	if !strings.Contains(strings.ToLower(retained.Architecture), "gemma4") {
		return false
	}
	size := strings.ToUpper(firstNonEmptyString(labels["gemma4_size"], labels["engine_state_context_gemma4_size"]))
	return size == "E2B" || size == "E4B"
}

func rocmTokenLoopLabels(status ROCmTokenLoopStatus) map[string]string {
	labels := cloneStringMap(status.Labels)
	if labels == nil {
		labels = map[string]string{}
	}
	labels["engine_token_loop_contract"] = status.Contract
	labels["engine_token_loop_reference"] = status.Reference
	labels["engine_token_loop_text_model"] = strconv.FormatBool(status.TextModel)
	labels["engine_token_loop_token_loop"] = strconv.FormatBool(status.TokenLoop)
	labels["engine_token_loop_embed_bookend"] = strconv.FormatBool(status.EmbedBookend)
	labels["engine_token_loop_decode_forward"] = strconv.FormatBool(status.DecodeForward)
	labels["engine_token_loop_lm_head_bookend"] = strconv.FormatBool(status.LMHeadBookend)
	labels["engine_token_loop_shared_generate_loop"] = strconv.FormatBool(status.SharedGenerateLoop)
	labels["engine_token_loop_incremental_session"] = strconv.FormatBool(status.IncrementalSession)
	labels["engine_token_loop_close_session"] = strconv.FormatBool(status.CloseSession)
	labels["engine_token_loop_step_with_id"] = strconv.FormatBool(status.StepWithID)
	labels["engine_token_loop_per_layer_inputs"] = strconv.FormatBool(status.PerLayerInputs)
	labels["engine_token_loop_runtime_owned_kv"] = strconv.FormatBool(status.RuntimeOwnedKV)
	labels["engine_token_loop_device_kv_state"] = strconv.FormatBool(status.DeviceKVState)
	labels["engine_token_loop_prompt_replay_refused"] = strconv.FormatBool(status.PromptReplayRefused)
	labels["engine_token_loop_retained_state_required"] = strconv.FormatBool(status.RetainedStateRequired)
	labels["engine_token_loop_incremental_ready"] = strconv.FormatBool(status.IncrementalDecodeReady())
	labels["engine_token_loop_fast_path"] = status.FastPath
	labels["engine_token_loop_fallback_path"] = status.FallbackPath
	if status.SessionState != "" {
		labels["engine_token_loop_session_state"] = status.SessionState
	}
	if status.Architecture != "" {
		labels["engine_token_loop_architecture"] = status.Architecture
	}
	if status.Family != "" {
		labels["engine_token_loop_family"] = status.Family
	}
	if status.Runtime != "" {
		labels["engine_token_loop_runtime"] = status.Runtime
	}
	if status.RuntimeStatus != "" {
		labels["engine_token_loop_runtime_status"] = string(status.RuntimeStatus)
	}
	return labels
}
