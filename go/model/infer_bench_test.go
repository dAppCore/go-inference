// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// The infer benches baseline the read-the-dim-FROM-THE-SHAPE rule (AX-11): at load, a
// config that omits a dimension recovers it from the actual weight shape rather than
// guessing. WeightAny is the first-present-name scan; InferHeadDim / InferOutFeaturesPerN
// are pure shape arithmetic over the resolved tensor. All three are map-lookup bound with
// no allocation, so these benches pin the resolve path stays alloc-free. Realistic input:
// a several-hundred-tensor checkpoint set.

func benchInferWeights() map[string]safetensors.Tensor {
	m := make(map[string]safetensors.Tensor, 256)
	for i := 0; i < 40; i++ {
		p := "model.layers." + core.Sprintf("%d", i)
		m[p+".self_attn.q_proj.weight"] = safetensors.Tensor{Shape: []int{8 * 128, 2048}}
		m[p+".self_attn.k_proj.weight"] = safetensors.Tensor{Shape: []int{2 * 128, 2048}}
		m[p+".mlp.gate_proj.weight"] = safetensors.Tensor{Shape: []int{8192, 2048}}
	}
	m["model.per_layer_projection.weight"] = safetensors.Tensor{Shape: []int{40, 256, 2048}}
	return m
}

// BenchmarkWeightAny — the alias scan: a handful of candidate names, the last present. One
// map get per candidate, no allocation. The cost a config alias pays to resolve.
func BenchmarkWeightAny(b *testing.B) {
	w := benchInferWeights()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok := WeightAny(w, "alt.q", "model.layers.0.self_attn.q_proj.weight"); !ok {
			b.Fatal("q_proj not resolved")
		}
	}
}

// BenchmarkInferHeadDim — one q_proj lookup + rows÷heads: the head-dim recovery a config
// that omits head_dim pays. Lookup + integer divide, no allocation.
func BenchmarkInferHeadDim(b *testing.B) {
	w := benchInferWeights()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if InferHeadDim(w, "model.layers.0.self_attn.q_proj.weight", 8) != 128 {
			b.Fatal("head dim mis-resolved")
		}
	}
}

// BenchmarkInferOutFeaturesPerN — a rank-3 PLE-tower projection flattened over its leading
// dims ÷ n: the per-layer-width recovery. A lookup + a short product loop, no allocation.
func BenchmarkInferOutFeaturesPerN(b *testing.B) {
	w := benchInferWeights()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if InferOutFeaturesPerN(w, "model.per_layer_projection.weight", 40) != 256 {
			b.Fatal("per-layer width mis-resolved")
		}
	}
}
