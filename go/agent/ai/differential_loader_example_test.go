// SPDX-License-Identifier: EUPL-1.2

package ai

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExamplePlanDifferentialLoad() {
	result := PlanDifferentialLoad(DifferentialLoadRequest{
		Base:    inference.ModelIdentity{Path: "/models/gemma4", Hash: "base"},
		Adapter: inference.AdapterIdentity{Path: "/adapters/project.safetensors", Format: "lora"},
	})
	if !result.OK {
		core.Println(result.Error())
		return
	}
	plan := result.Value.(DifferentialLoadPlan)
	core.Println(plan.Action)
	core.Println(plan.BaseSlice.Preset)
	// Output:
	// reuse_base_adapter
	// full
}
