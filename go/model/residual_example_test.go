// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

func ExampleParallelResidual() {
	r := ParallelResidual([]float32{1, 2}, []float32{3, 4}, []float32{5, 6})
	core.Println(r.Value)
	// Output:
	// [9 12]
}
