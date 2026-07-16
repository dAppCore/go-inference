// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import core "dappco.re/go"

func ExampleSSDScanF32() {
	const H, P, N = 1, 2, 2
	x := syn(H*P, 1)
	dt := syn(H, 2)
	a := syn(H, 3)
	b := syn(H*N, 4)
	c := syn(H*N, 5)
	y, state, err := SSDScanF32(x, dt, a, b, c, nil, nil, 1, H, P, N)
	core.Println(err == nil, len(y), len(state))
	// Output: true 2 4
}
