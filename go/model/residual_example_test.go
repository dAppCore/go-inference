// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

func ExampleParallelResidual() {
	r := ParallelResidual([]float32{1, 2}, []float32{3, 4}, []float32{5, 6})
	core.Println(r.Value)
	// Output:
	// [9 12]
}

func ExampleApplyResidualOrder() {
	id := func(x []float32) []float32 { return x }
	r := ApplyResidualOrder(NormPlacementPost, []float32{1, 2}, id, id, id, id)
	core.Println(r.Value)
	// Output:
	// [4 8]
}
