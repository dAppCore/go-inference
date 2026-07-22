// SPDX-Licence-Identifier: EUPL-1.2

package bloom

import core "dappco.re/go"

func ExampleConfig_Arch() {
	cfg := Config{HiddenSize: 1024, NumHiddenLayers: 1, NumAttentionHeads: 16, VocabSize: 8}
	arch, err := cfg.Arch()
	core.Println(err == nil, arch.FF, arch.ALiBi)
	// Output: true 4096 true
}

func ExampleConfig_InferFromWeights() {
	cfg := Config{HiddenSize: 1024}
	cfg.InferFromWeights(nil)
	core.Println(cfg.HiddenSize)
	// Output: 1024
}
