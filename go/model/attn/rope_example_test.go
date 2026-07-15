// SPDX-Licence-Identifier: EUPL-1.2

package attn_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/attn"
)

func ExampleRopeParams() {
	dim, _ := (attn.RopeParams{HeadDim: 80, PartialRotaryFactor: 0.4}).RotaryDim()
	core.Println(dim)
	// Output: 32
}

func ExampleRopeParams_RotaryDim() {
	dim, _ := (attn.RopeParams{HeadDim: 64}).RotaryDim()
	core.Println(dim)
	// Output: 64
}
