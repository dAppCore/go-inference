// SPDX-Licence-Identifier: EUPL-1.2

package gptneox

import core "dappco.re/go"

func ExampleWeightNames() {
	w := WeightNames("gptj")
	core.Println(w.LayerPrefix, w.Q)
	// Output: transformer.h.%d .attn.q_proj
}
