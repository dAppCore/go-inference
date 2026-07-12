// SPDX-Licence-Identifier: EUPL-1.2

package model_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func ExampleRopeParams() {
	dim, _ := (model.RopeParams{HeadDim: 80, PartialRotaryFactor: 0.4}).RotaryDim()
	core.Println(dim)
	// Output: 32
}

func ExampleRopeParams_RotaryDim() {
	dim, _ := (model.RopeParams{HeadDim: 64}).RotaryDim()
	core.Println(dim)
	// Output: 64
}
