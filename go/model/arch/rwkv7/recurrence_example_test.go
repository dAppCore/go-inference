// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import core "dappco.re/go"

func ExampleWKV7F32() {
	const H, K, V = 1, 2, 2
	r := syn(H*K, 1)
	w := syn(H*K, 2)
	k := syn(H*K, 3)
	v := syn(H*V, 4)
	a := syn(H*K, 5)
	b := syn(H*K, 6)
	o, state, err := WKV7F32(r, w, k, v, a, b, nil, 1, H, K, V)
	core.Println(err == nil, len(o), len(state))
	// Output: true 2 4
}
