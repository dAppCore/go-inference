// SPDX-Licence-Identifier: EUPL-1.2

package dbrx

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

func ExampleNormalizeWeights() {
	cfg := Config{DModel: 8, Heads: 2, Layers: 1, Attention: AttentionConfig{KVHeads: 1}}
	in := map[string]safetensors.Tensor{"transformer.wte.weight": {Shape: []int{16, 8}}}
	out := NormalizeWeights(in, cfg)
	_, ok := out["model.embed_tokens.weight"]
	core.Println(ok)
	// Output: true
}
