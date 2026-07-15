// SPDX-Licence-Identifier: EUPL-1.2

package qwenmoe_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/qwenmoe"
)

func ExampleConfig_Arch() {
	cfg := qwenmoe.Config{
		HiddenSize: 2048, NumHiddenLayers: 24, NumAttentionHeads: 16,
		NumKeyValueHeads: 16, VocabSize: 151936, NumExperts: 60,
		NumExpertsPerTok: 4, MoEIntermediateSize: 1408,
		SharedExpertIntermediateSize: 5632,
	}
	arch, err := cfg.Arch()
	core.Println(err == nil, arch.Experts, arch.TopK, arch.SharedExperts)
	// Output:
	// true 60 4 1
}
