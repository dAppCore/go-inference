// SPDX-Licence-Identifier: EUPL-1.2

package phi

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

func ExampleNormalizePhi3Weights() {
	qkv := safetensors.Tensor{Dtype: "F32", Shape: []int{6, 2}, Data: []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}}
	out := NormalizePhi3Weights(map[string]safetensors.Tensor{
		"model.layers.0.self_attn.qkv_proj.weight": qkv,
	})
	_, ok := out["model.layers.0.self_attn.q_proj.weight"]
	core.Println(ok)
	// Output: true
}
