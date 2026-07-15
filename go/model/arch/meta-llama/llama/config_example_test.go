// SPDX-Licence-Identifier: EUPL-1.2

package llama_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/arch/meta-llama/llama"
	"dappco.re/go/inference/model/safetensors"
)

func ExampleConfig() {
	tied := true
	cfg := llama.Config{HiddenSize: 2048, NumHiddenLayers: 16, TieWordEmbeddings: &tied}
	core.Println(cfg.HiddenSize, cfg.NumHiddenLayers, *cfg.TieWordEmbeddings)
	// Output: 2048 16 true
}

func ExampleRopeScaling() {
	r := llama.RopeScaling{RopeType: "llama3", Factor: 8}
	core.Println(r.RopeType, r.Factor)
	// Output: llama3 8
}

func ExampleConfig_Arch() {
	cfg := llama.Config{
		HiddenSize: 64, IntermediateSize: 128, NumHiddenLayers: 2,
		NumAttentionHeads: 8, NumKeyValueHeads: 2, VocabSize: 32,
	}
	arch, err := cfg.Arch()
	core.Println(err == nil, arch.HeadDim, arch.KVHeads, len(arch.Layer))
	// Output: true 8 2 2
}

func ExampleConfig_InferFromWeights() {
	cfg := llama.Config{HeadDim: 128}
	cfg.InferFromWeights(map[string]safetensors.Tensor{})
	core.Println(cfg.HeadDim)
	// Output: 128
}

func ExampleLlama3InvFreqs() {
	freqs := llama.Llama3InvFreqs(500000, llama.RopeScaling{
		RopeType: "llama3", Factor: 8, LowFreqFactor: 1, HighFreqFactor: 4,
		OriginalMaxPositionEmbeddings: 8192,
	}, 128)
	core.Println(len(freqs), freqs[0])
	// Output: 64 1
}
