// SPDX-Licence-Identifier: EUPL-1.2

package jetmoe_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/jetmoe"
)

func ExampleConfig_Arch() {
	cfg := jetmoe.Config{HiddenSize: 8, FFNHiddenSize: 4, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, KVChannels: 4, MoENumExperts: 2, MoETopK: 1, VocabSize: 16}
	arch, err := cfg.Arch()
	core.Println(err == nil, arch.Experts, arch.TopK)
	// Output: true 2 1
}
