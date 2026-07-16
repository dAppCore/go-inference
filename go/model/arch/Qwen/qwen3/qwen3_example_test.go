// SPDX-Licence-Identifier: EUPL-1.2

package qwen3

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

func ExampleConfig_ResolvedQuant() {
	c := &Config{Quantization: &model.QuantConfig{Bits: 4, GroupSize: 64}}
	q := c.ResolvedQuant()
	core.Println(q.Bits, q.GroupSize)
	// Output: 4 64
}

func ExampleConfig_InferFromWeights() {
	c := &Config{HiddenSize: 2560, NumHiddenLayers: 1, NumAttentionHeads: 16}
	c.InferFromWeights(map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": {Shape: []int{16 * 128, 2560}},
	})
	core.Println(c.HeadDim)
	// Output: 128
}

func ExampleConfig_Arch() {
	c := &Config{HiddenSize: 8, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 16, HeadDim: 4}
	arch, err := c.Arch()
	core.Println(err == nil)
	core.Println(arch.HeadDim, arch.RotaryDim)
	// Output:
	// true
	// 4 4
}
