// SPDX-Licence-Identifier: EUPL-1.2

package cohere_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/cohere"
)

func ExampleConfig() {
	qk := true
	cfg := cohere.Config{ModelType: "cohere", UseQKNorm: &qk, LogitScale: 0.0625}
	core.Println(cfg.ModelType, *cfg.UseQKNorm, cfg.LogitScale)
	// Output: cohere true 0.0625
}

func ExampleConfig_Arch() {
	cfg := cohere.Config{ModelType: "cohere2", HiddenSize: 64, IntermediateSize: 128, NumHiddenLayers: 4, NumAttentionHeads: 8, NumKeyValueHeads: 2, VocabSize: 32, SlidingWindow: 16, SlidingWindowPattern: 4}
	arch, err := cfg.Arch()
	core.Println(err == nil, arch.Layer[0].TypeName(), arch.Layer[3].TypeName(), arch.LogitScale)
	// Output: true sliding_attention full_attention 0.0625
}

func ExampleConfig_InferFromWeights() {
	cfg := cohere.Config{HiddenSize: 64}
	cfg.InferFromWeights(nil)
	core.Println(cfg.HiddenSize)
	// Output: 64
}
