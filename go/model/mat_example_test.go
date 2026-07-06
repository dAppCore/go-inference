// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// ExampleMatNT shows the shared reference matmul: out = in · wᵀ for row-major in [M×K] and
// w [N×K] (weight stored transposed), the pure-Go path the arch packages use for CPU-side
// linear projections.
func ExampleMatNT() {
	hidden := []float32{1, 2} // M=1, K=2
	weight := []float32{1, 1, 2, 0}  // N=2, K=2: row0=[1,1], row1=[2,0]
	out := MatNT(hidden, weight, 1, 2, 2)
	core.Println(out[0]) // 1*1 + 2*1 = 3
	core.Println(out[1]) // 1*2 + 2*0 = 2
	// Output:
	// 3
	// 2
}
