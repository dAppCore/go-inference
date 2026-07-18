// SPDX-Licence-Identifier: EUPL-1.2

package model_test

import (
	"fmt"

	"dappco.re/go/inference/model"
)

// ExampleConcatQuantRows fuses a SwiGLU expert's separate gate and up projections (each [FF, D]
// packed at the same bits/group) into one [2·FF, D] weight — the gate rows first, then up — so a
// single quant matvec outputs [gate‖up] and the caller splits the halves for silu-mul.
func ExampleConcatQuantRows() {
	const FF, D, bits, groupSize = 24, 64, 4, 32
	rows := func(n int) []byte { return make([]byte, n*D*bits/8) }
	scales := func(n int) []byte { return make([]byte, n*(D/groupSize)*2) }
	mk := func() *model.QuantWeight {
		return &model.QuantWeight{Packed: rows(FF), Scales: scales(FF), Biases: scales(FF), Bits: bits, GroupSize: groupSize, OutDim: FF, InDim: D}
	}

	gateUp := model.ConcatQuantRows(mk(), mk())
	fmt.Println(gateUp.OutDim, gateUp.InDim)
	// Output: 48 64
}
