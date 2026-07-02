// SPDX-License-Identifier: EUPL-1.2

package ai

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestDifferentialLoader_DifferentialLoadReuseBaseAdapter_Good(t *core.T) {
	result := PlanDifferentialLoad(DifferentialLoadRequest{
		Base:    inference.ModelIdentity{Path: "/models/gemma4", Hash: "base"},
		Adapter: inference.AdapterIdentity{Path: "/adapters/project.safetensors", Format: "lora"},
		Labels:  map[string]string{"project": "lthn"},
	})

	core.AssertTrue(t, result.OK)
	plan := result.Value.(DifferentialLoadPlan)
	core.AssertEqual(t, DifferentialLoadReuseBaseAdapter, plan.Action)
	core.AssertFalse(t, plan.Compare)
	core.AssertEqual(t, inference.ModelSlicePresetFull, plan.BaseSlice.Preset)
	core.AssertEqual(t, "lthn", plan.Labels["project"])
}

func TestDifferentialLoader_DifferentialLoadCompareModels_Good(t *core.T) {
	result := PlanDifferentialLoad(DifferentialLoadRequest{
		Base:        inference.ModelIdentity{Path: "/models/base", Hash: "base"},
		Tuned:       inference.ModelIdentity{Path: "/models/fine", Hash: "fine"},
		PreferSplit: true,
		Endpoints: []inference.SplitEndpoint{{
			ID:   "ffn-0",
			Role: inference.SplitEndpointRoleFFN,
			URL:  "http://127.0.0.1:8765",
		}},
	})

	core.AssertTrue(t, result.OK)
	plan := result.Value.(DifferentialLoadPlan)
	core.AssertEqual(t, DifferentialLoadCompareModels, plan.Action)
	core.AssertTrue(t, plan.Compare)
	core.AssertNotNil(t, plan.Split)
	core.AssertEqual(t, inference.SplitInferenceModeRemoteFFN, plan.Split.Mode)
	core.AssertEqual(t, inference.ModelSlicePresetClient, plan.BaseSlice.Preset)
	core.AssertFalse(t, plan.BaseSlice.HasComponent(inference.ModelComponentFFN))
}

func TestDifferentialLoader_PlanDifferentialLoad_Bad(t *core.T) {
	result := PlanDifferentialLoad(DifferentialLoadRequest{})

	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "base model is required")
}

func TestDifferentialLoader_PlanDifferentialLoad_Ugly(t *core.T) {
	result := PlanDifferentialLoad(DifferentialLoadRequest{
		Base:        inference.ModelIdentity{Path: "/models/base", Hash: "base"},
		PreferSplit: true,
	})

	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "requires an ffn endpoint")
}
