// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import core "dappco.re/go"

func ExampleNormalizeQuantType() {
	core.Println(NormalizeQuantType("Q4-K M"))
	// Output: q4_k_m
}
