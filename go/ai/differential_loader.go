// SPDX-License-Identifier: EUPL-1.2

package ai

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

// DifferentialLoadAction describes how the inference stack should stage a base/fine-tune
// pair before a research or agentic workflow runs.
type DifferentialLoadAction string

const (
	DifferentialLoadBaseOnly         DifferentialLoadAction = "base_only"
	DifferentialLoadReuseBaseAdapter DifferentialLoadAction = "reuse_base_adapter"
	DifferentialLoadCompareModels    DifferentialLoadAction = "compare_models"
)

// DifferentialLoadRequest captures the model relationship the inference stack needs to
// reason about without importing a concrete backend.
type DifferentialLoadRequest struct {
	Base        inference.ModelIdentity      `json:"base,omitempty"`
	Tuned       inference.ModelIdentity      `json:"tuned,omitempty"`
	Adapter     inference.AdapterIdentity    `json:"adapter,omitempty"`
	PreferSplit bool                         `json:"prefer_split,omitempty"`
	SplitMode   inference.SplitInferenceMode `json:"split_mode,omitempty"`
	Endpoints   []inference.SplitEndpoint    `json:"endpoints,omitempty"`
	Labels      map[string]string            `json:"labels,omitempty"`
}

// DifferentialLoadPlan is the policy result consumed by an agent or UI before
// loading base and fine-tuned models for comparison.
type DifferentialLoadPlan struct {
	Action     DifferentialLoadAction        `json:"action"`
	Base       inference.ModelIdentity       `json:"base,omitempty"`
	Tuned      inference.ModelIdentity       `json:"tuned,omitempty"`
	Adapter    inference.AdapterIdentity     `json:"adapter,omitempty"`
	BaseSlice  inference.ModelSlicePlan      `json:"base_slice,omitempty"`
	TunedSlice inference.ModelSlicePlan      `json:"tuned_slice,omitempty"`
	Split      *inference.SplitInferencePlan `json:"split,omitempty"`
	Compare    bool                          `json:"compare,omitempty"`
	Labels     map[string]string             `json:"labels,omitempty"`
}

// PlanDifferentialLoad chooses a safe base/fine-tune loading strategy. It is
// deliberately metadata-only; backends still own tensor placement and loading.
func PlanDifferentialLoad(req DifferentialLoadRequest) core.Result {
	if modelIdentityEmpty(req.Base) {
		return core.Fail(core.E("ai.PlanDifferentialLoad", "base model is required", nil))
	}
	action := DifferentialLoadBaseOnly
	compare := false
	if !adapterIdentityEmpty(req.Adapter) && (modelIdentityEmpty(req.Tuned) || sameModelIdentity(req.Base, req.Tuned)) {
		action = DifferentialLoadReuseBaseAdapter
	} else if !modelIdentityEmpty(req.Tuned) && !sameModelIdentity(req.Base, req.Tuned) {
		action = DifferentialLoadCompareModels
		compare = true
	}

	preset := inference.ModelSlicePresetFull
	mode := req.SplitMode
	if mode == "" && (req.PreferSplit || len(req.Endpoints) > 0) {
		mode = inference.SplitInferenceModeRemoteFFN
	}
	if mode != "" && mode != inference.SplitInferenceModeLocal {
		preset = inference.ModelSlicePresetClient
	}

	baseSlice, err := inference.PlanModelSlice(inference.ModelSliceRequest{
		Preset:  preset,
		Model:   req.Base,
		Adapter: req.Adapter,
		Labels:  req.Labels,
	})
	if err != nil {
		return core.Fail(core.E("ai.PlanDifferentialLoad", "plan base slice", err))
	}

	tunedSlice := inference.ModelSlicePlan{}
	if !modelIdentityEmpty(req.Tuned) {
		tunedSlice, err = inference.PlanModelSlice(inference.ModelSliceRequest{
			Preset:  preset,
			Model:   req.Tuned,
			Adapter: req.Adapter,
			Labels:  req.Labels,
		})
		if err != nil {
			return core.Fail(core.E("ai.PlanDifferentialLoad", "plan tuned slice", err))
		}
	}

	var split *inference.SplitInferencePlan
	if mode != "" {
		splitPlan := inference.SplitInferencePlan{
			Mode:       mode,
			Model:      req.Base,
			Adapter:    req.Adapter,
			LocalSlice: baseSlice,
			Endpoints:  cloneDifferentialEndpoints(req.Endpoints),
			Labels:     core.MapClone(req.Labels),
		}
		if err := inference.ValidateSplitInferencePlan(splitPlan); err != nil {
			return core.Fail(core.E("ai.PlanDifferentialLoad", "validate split plan", err))
		}
		split = &splitPlan
	}

	return core.Ok(DifferentialLoadPlan{
		Action:     action,
		Base:       req.Base,
		Tuned:      req.Tuned,
		Adapter:    req.Adapter,
		BaseSlice:  baseSlice,
		TunedSlice: tunedSlice,
		Split:      split,
		Compare:    compare,
		Labels:     core.MapClone(req.Labels),
	})
}

func modelIdentityEmpty(model inference.ModelIdentity) bool {
	return core.Trim(model.Path) == "" && core.Trim(model.Hash) == "" && core.Trim(model.Architecture) == ""
}

func adapterIdentityEmpty(adapter inference.AdapterIdentity) bool {
	return core.Trim(adapter.Path) == "" && core.Trim(adapter.Hash) == "" && core.Trim(adapter.Format) == ""
}

func sameModelIdentity(left, right inference.ModelIdentity) bool {
	if modelIdentityEmpty(left) || modelIdentityEmpty(right) {
		return false
	}
	if left.Hash != "" && right.Hash != "" {
		return left.Hash == right.Hash
	}
	if left.Path != "" && right.Path != "" {
		return left.Path == right.Path
	}
	return left.Architecture != "" && left.Architecture == right.Architecture
}

func cloneDifferentialEndpoints(endpoints []inference.SplitEndpoint) []inference.SplitEndpoint {
	if len(endpoints) == 0 {
		return nil
	}
	out := make([]inference.SplitEndpoint, len(endpoints))
	for i, endpoint := range endpoints {
		out[i] = endpoint
		out[i].Labels = core.MapClone(endpoint.Labels)
	}
	return out
}
