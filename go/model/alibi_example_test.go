// SPDX-Licence-Identifier: EUPL-1.2

package model_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func ExampleALiBiSlopes() {
	core.Println(model.ALiBiSlopes(2))
	// Output: [0.0625 0.00390625]
}

func ExampleApplyALiBi() {
	scores := []float64{0, 0, 0}
	model.ApplyALiBi(scores, 0.5, 2, 0)
	core.Println(scores)
	// Output: [-1 -0.5 0]
}
