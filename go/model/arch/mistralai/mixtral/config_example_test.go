// SPDX-Licence-Identifier: EUPL-1.2

package mixtral

import core "dappco.re/go"

func ExampleConfig_Arch() {
	cfg := Config{
		HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1,
		NumAttentionHeads: 2, NumKeyValueHeads: 1,
		NumLocalExperts: 4, NumExpertsPerTok: 2, VocabSize: 32,
	}
	arch, err := cfg.Arch()
	core.Println(err == nil)
	core.Println(arch.Experts, arch.TopK, arch.NormaliseMoETopK)
	// Output:
	// true
	// 4 2 true
}

func ExampleConfig_InferFromWeights() {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(nil)
	core.Println(cfg.HiddenSize)
	// Output: 8
}
