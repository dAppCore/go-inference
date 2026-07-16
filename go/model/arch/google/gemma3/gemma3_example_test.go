// SPDX-Licence-Identifier: EUPL-1.2

package gemma3

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

func ExampleConfig_ResolvedQuant() {
	c := &Config{Quantization: &model.QuantConfig{Bits: 4, Mode: "affine"}}
	q := c.ResolvedQuant()
	core.Println(q.Bits, q.Mode)
	// Output: 4 affine
}

func ExampleConfig_InferFromWeights() {
	c := &Config{NumHiddenLayers: 1, NumAttentionHeads: 8}
	c.InferFromWeights(map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": {Shape: []int{2048, 1024}}, // 8 heads * 256 head_dim
	})
	core.Println(c.HeadDim)
	// Output: 256
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
