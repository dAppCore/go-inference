// SPDX-Licence-Identifier: EUPL-1.2

package stablelm

import core "dappco.re/go"

func ExampleConfig_Arch() {
	a, _ := (&Config{ModelType: "stablelm", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, VocabSize: 16, PartialRotaryFactor: .5}).Arch()
	core.Println(a.HeadDim, a.RotaryDim)
	// Output: 4 2
}

func ExampleConfig_InferFromWeights() {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(nil) // no-op: StableLM declares every dim in config.json
	core.Println(cfg.HiddenSize)
	// Output: 8
}
