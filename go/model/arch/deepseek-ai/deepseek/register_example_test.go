// SPDX-Licence-Identifier: EUPL-1.2

package deepseek

import core "dappco.re/go"

func ExampleWeightNames() {
	w := WeightNames()
	core.Println(w.Q, w.Router)
	// Output: .self_attn.q_proj.weight .mlp.gate.weight
}
