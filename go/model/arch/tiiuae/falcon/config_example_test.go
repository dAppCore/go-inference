// SPDX-Licence-Identifier: EUPL-1.2

package falcon

import core "dappco.re/go"

func ExampleConfig_Arch() {
	cfg := Config{HiddenSize: 64, NumHiddenLayers: 2, NumAttentionHeads: 8, VocabSize: 100, MultiQuery: true, ALiBi: true}
	arch, err := cfg.Arch()
	core.Println(err == nil, arch.KVHeads, arch.ALiBi)
	// Output: true 1 true
}

func ExampleConfig_InferFromWeights() {
	cfg := Config{HiddenSize: 64}
	cfg.InferFromWeights(nil) // no-op: Falcon declares every dim in config.json
	core.Println(cfg.HiddenSize)
	// Output: 64
}
