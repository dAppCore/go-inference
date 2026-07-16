// SPDX-Licence-Identifier: EUPL-1.2

package llama4

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

func ExampleNormalizeWeights() {
	in := map[string]safetensors.Tensor{
		"language_model.model.layers.0.feed_forward.router.weight": {Dtype: "F32", Shape: []int{2, 2}, Data: make([]byte, 16)},
	}
	out, err := NormalizeWeights(in)
	_, ok := out["language_model.model.layers.0.mlp.gate.weight"]
	core.Println(err == nil, ok)
	// Output: true true
}
