// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import core "dappco.re/go"

func ExampleCausalConv1dF32() {
	const L, convDim, K = 4, 2, 3
	in := syn(L*convDim, 1)
	w := syn(convDim*K, 2)
	out, newState, err := CausalConv1dF32(in, w, nil, nil, L, convDim, K)
	core.Println(err == nil, len(out), len(newState))
	// Output: true 8 4
}
