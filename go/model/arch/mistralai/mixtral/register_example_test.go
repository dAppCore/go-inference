// SPDX-Licence-Identifier: EUPL-1.2

package mixtral

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

func ExampleWeightNames() {
	w := WeightNames()
	core.Println(w.Router)
	// Output: .block_sparse_moe.gate.weight
}

func ExampleNormalizeWeights() {
	in := map[string]safetensors.Tensor{
		"model.layers.0.block_sparse_moe.gate.weight": {Shape: []int{2, 8}},
	}
	out := NormalizeWeights(in)
	_, ok := out["model.layers.0.mlp.gate.weight"]
	core.Println(ok)
	// Output: true
}
