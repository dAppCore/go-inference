// SPDX-Licence-Identifier: EUPL-1.2

package phi

import core "dappco.re/go"

func ExampleConfig_Arch() {
	cfg := Config{ModelType: "phi", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 12, LayerNormEps: 1e-5}
	arch, err := cfg.Arch()
	core.Println(err == nil)
	core.Println(arch.HeadDim)
	// Output:
	// true
	// 4
}

func ExampleConfig_InferFromWeights() {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(nil)
	core.Println(cfg.HiddenSize)
	// Output: 8
}
