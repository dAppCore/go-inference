// SPDX-Licence-Identifier: EUPL-1.2

package gpt2

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

func ExampleNormalizeWeights() {
	in := map[string]safetensors.Tensor{
		"transformer.h.0.attn.c_attn.weight": {Dtype: "U8", Shape: []int{2, 6}, Data: []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
	}
	out := NormalizeWeights(in)
	_, ok := out["transformer.h.0.attn.q_proj.weight"]
	core.Println(ok)
	// Output: true
}
