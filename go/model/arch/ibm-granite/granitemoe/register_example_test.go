// SPDX-Licence-Identifier: EUPL-1.2

package granitemoe

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

func ExampleNormalizeWeights() {
	cfg := &Config{HiddenSize: 2, IntermediateSize: 3, NumHiddenLayers: 1, NumLocalExperts: 2}
	in := map[string]safetensors.Tensor{
		"model.layers.0.block_sparse_moe.input_linear.weight":  tensor(make([]float32, 24), 2, 6, 2),
		"model.layers.0.block_sparse_moe.output_linear.weight": tensor(make([]float32, 12), 2, 2, 3),
		"model.layers.0.block_sparse_moe.router.layer.weight":  tensor(make([]float32, 4), 2, 2),
	}
	out, err := NormalizeWeights(in, cfg)
	_, ok := out["model.layers.0.mlp.experts.1.up_proj.weight"]
	core.Println(err == nil, ok)
	// Output: true true
}
