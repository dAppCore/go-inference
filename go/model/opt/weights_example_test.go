// SPDX-Licence-Identifier: EUPL-1.2

package opt

import core "dappco.re/go"

func ExampleWeightNames() {
	weights := WeightNames()
	core.Println(weights.Embed, weights.PositionEmbed)
	// Output: model.decoder.embed_tokens model.decoder.embed_positions
}
