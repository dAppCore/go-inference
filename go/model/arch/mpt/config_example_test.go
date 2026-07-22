// SPDX-Licence-Identifier: EUPL-1.2

package mpt

import core "dappco.re/go"

func ExampleConfig_Arch() {
	a, _ := (&Config{ModelType: "mpt", DModel: 8, NHeads: 2, NLayers: 1, ExpansionRatio: 4, VocabSize: 16, AttnConfig: AttentionConfig{ALiBi: true}}).Arch()
	core.Println(a.HeadDim, a.ALiBi)
	// Output: 4 true
}

func ExampleConfig_InferFromWeights() {
	c := Config{ModelType: "mpt", DModel: 8}
	c.InferFromWeights(nil) // no-op: MPT declares every dim in config.json
	core.Println(c.DModel)
	// Output: 8
}
