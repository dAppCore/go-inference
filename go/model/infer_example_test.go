// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"

	"dappco.re/go/inference/model/safetensors"
)

// ExampleWeightAny shows the fallback-name lookup: the first of several candidate tensor
// names present in the checkpoint wins — the pattern an arch uses when different releases
// spell the same weight differently.
func ExampleWeightAny() {
	weights := map[string]safetensors.Tensor{"model.embed_tokens.weight": {Shape: []int{8, 4}}}
	t, ok := WeightAny(weights, "embed_tokens.weight", "model.embed_tokens.weight")
	core.Println(ok)         // the second candidate name resolved
	core.Println(t.Shape[0]) // the resolved tensor's first dim
	// Output:
	// true
	// 8
}

// ExampleInferHeadDim shows the don't-guess rule: a head dim omitted from config.json is
// read from the q_proj weight's actual shape (rows ÷ numHeads) instead of assumed.
func ExampleInferHeadDim() {
	weights := map[string]safetensors.Tensor{"q_proj.weight": {Shape: []int{8 * 64, 2048}}}
	core.Println(InferHeadDim(weights, "q_proj.weight", 8))
	// Output: 64
}

// ExampleInferOutFeaturesPerN shows the per-layer-stacked-projection rule: a projection
// whose output rows are actually numLayers per-layer widths stacked flat resolves the
// per-layer width by dividing the flattened out-features by n.
func ExampleInferOutFeaturesPerN() {
	weights := map[string]safetensors.Tensor{"per_layer_proj.weight": {Shape: []int{30, 256, 2048}}}
	core.Println(InferOutFeaturesPerN(weights, "per_layer_proj.weight", 30))
	// Output: 256
}
