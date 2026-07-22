// SPDX-Licence-Identifier: EUPL-1.2

package llama4

import core "dappco.re/go"

func ExampleConfig_Arch() {
	cfg := Config{HiddenSize: 8, IntermediateSize: 12, IntermediateSizeMLP: 16, NumHiddenLayers: 2, NumAttentionHeads: 2, HeadDim: 4, NumLocalExperts: 2, NumExpertsPerTok: 1, VocabSize: 32}
	arch, err := cfg.Arch()
	core.Println(err == nil, arch.Experts, arch.TopK, arch.SharedExperts)
	// Output: true 2 1 1
}

func ExampleConfig_InferFromWeights() {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(nil) // no-op: Llama 4 declares its geometry
	core.Println(cfg.HiddenSize)
	// Output: 8
}
