// SPDX-Licence-Identifier: EUPL-1.2

package qwenmoe_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/arch/Qwen/qwenmoe"
	"dappco.re/go/inference/model/safetensors"
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

func ExampleConfig_InferFromWeights() {
	cfg := qwenmoe.Config{NumHiddenLayers: 1, NumAttentionHeads: 2}
	cfg.InferFromWeights(map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": {Shape: []int{16, 8}},
		"model.embed_tokens.weight":              {Shape: []int{32, 8}},
	})
	core.Println(cfg.HeadDim, cfg.VocabSize)
	// Output: 8 32
}
