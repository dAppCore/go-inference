// SPDX-Licence-Identifier: EUPL-1.2

package olmoe

import core "dappco.re/go"

func ExampleConfig_Arch() {
	cfg := Config{
		HiddenSize: 8, IntermediateSize: 12, NumHiddenLayers: 1,
		NumAttentionHeads: 2, NumKeyValueHeads: 1,
		NumExperts: 4, NumExpertsPerTok: 2, VocabSize: 32,
	}
	arch, err := cfg.Arch()
	core.Println(err == nil)
	core.Println(arch.Experts, arch.TopK, arch.NormaliseMoETopK, arch.SharedExperts)
	// Output:
	// true
	// 4 2 false 0
}

func ExampleConfig_InferFromWeights() {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(nil)
	core.Println(cfg.HiddenSize)
	// Output: 8
}
