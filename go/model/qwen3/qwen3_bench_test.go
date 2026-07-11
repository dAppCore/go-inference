// SPDX-Licence-Identifier: EUPL-1.2

package qwen3

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// The qwen3 config benches baseline the per-load arch derivation (AX-11): Config.Arch builds
// the neutral model.Arch (allocating the per-layer type slice + the derived LayerSpecs) and
// Config.InferFromWeights recovers head_dim/vocab from the weight shapes when the config
// omits them (a q_proj-rows scan). Both run once per load, not per token. Dims: a 36-layer
// dense qwen3 (hidden 2048, 16 heads, GQA 2 KV heads).

func benchQwen3Config() Config {
	return Config{
		HiddenSize: 2048, NumHiddenLayers: 36, NumAttentionHeads: 16, NumKeyValueHeads: 2,
		HeadDim: 128, IntermediateSize: 11008, VocabSize: 151936, RMSNormEps: 1e-6, RopeTheta: 1e6,
	}
}

// BenchmarkConfig_Arch — the per-load Arch build: the all-global layer-type slice + the
// DeriveLayers derivation + the per-layer head geometry fill. The two slices are the cost.
func BenchmarkConfig_Arch(b *testing.B) {
	c := benchQwen3Config()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := c.Arch(); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkConfig_InferFromWeights — the don't-guess dim recovery: with head_dim + vocab
// unset, scan the q_proj rows for head_dim and the embedding rows for vocab. A few map
// lookups, no allocation.
func BenchmarkConfig_InferFromWeights(b *testing.B) {
	weights := map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": {Shape: []int{16 * 128, 2048}},
		"model.embed_tokens.weight":              {Shape: []int{151936, 2048}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := benchQwen3Config()
		c.HeadDim, c.VocabSize = 0, 0 // force the shape-inference path
		c.InferFromWeights(weights)
	}
}
