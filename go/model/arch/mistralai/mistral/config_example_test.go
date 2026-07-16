// SPDX-Licence-Identifier: EUPL-1.2

package mistral_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/arch/mistralai/mistral"
)

func ExampleConfig_Arch() {
	cfg := mistral.Config{
		HiddenSize: 64, NumHiddenLayers: 2, NumAttentionHeads: 8,
		IntermediateSize: 128, VocabSize: 100,
	}
	arch, err := cfg.Arch()
	core.Println(err == nil)
	core.Println(arch.HeadDim, arch.KVHeads)
	// Output:
	// true
	// 8 8
}

func ExampleConfig_InferFromWeights() {
	cfg := mistral.Config{HiddenSize: 64}
	cfg.InferFromWeights(nil)
	core.Println(cfg.HiddenSize)
	// Output: 64
}
