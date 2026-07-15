// SPDX-Licence-Identifier: EUPL-1.2

package inference

import core "dappco.re/go"

func ExamplePlanModelSlice() {
	plan, err := PlanModelSlice(ModelSliceRequest{Preset: ModelSlicePresetClient})
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(plan.Preset)
	core.Println(plan.HasComponent(ModelComponentAttention))
	core.Println(plan.HasComponent(ModelComponentFFN))
	// Output:
	// client
	// true
	// false
}
