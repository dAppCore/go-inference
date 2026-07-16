// SPDX-Licence-Identifier: EUPL-1.2

package deltanet

import core "dappco.re/go"

func ExampleGatedDeltaRuleF32() {
	const H, D = 1, 2
	q := syn(H*D, 1)
	k := syn(H*D, 2)
	v := syn(H*D, 3)
	beta := syn(H, 4)
	alpha := syn(H, 5)
	o, state, err := GatedDeltaRuleF32(q, k, v, beta, alpha, nil, 1, H, D, 0.5, testEps)
	core.Println(err == nil, len(o), len(state))
	// Output: true 2 4
}
