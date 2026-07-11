// SPDX-Licence-Identifier: EUPL-1.2

package gemma3

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// The gemma3 config benches baseline the per-load arch derivation (AX-11): Config.Arch builds
// the neutral model.Arch — the sliding/global layer schedule (every 6th global), the per-layer
// head geometry, the dual (global/local) RoPE bases — allocating the layer-type slice + the
// derived LayerSpecs; InferFromWeights recovers head_dim/vocab from the weight shapes when the
// config omits them. Both run once per load. Dims: a 34-layer gemma3 (hidden 2560, head_dim
// 256). Pure Go, synthetic — no file.

func benchGemma3Config() Config {
	return Config{
		HiddenSize: 2560, NumHiddenLayers: 34, IntermediateSize: 10240,
		NumAttentionHeads: 8, NumKeyValueHeads: 4, HeadDim: 256, VocabSize: 262144,
		RMSNormEps: 1e-6, RopeTheta: 1e6, RopeLocalBaseFreq: 1e4, SlidingWindow: 1024, SlidingWindowPattern: 6,
	}
}

// BenchmarkConfig_Arch — the per-load Arch build: the sliding/global layer schedule +
// DeriveLayers + the per-layer head fill. The layer slices are the allocation story.
func BenchmarkConfig_Arch(b *testing.B) {
	c := benchGemma3Config()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := c.Arch(); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkConfig_InferFromWeights — the don't-guess dim recovery: with head_dim + vocab unset,
// scan a q_proj's rows for head_dim and the embedding rows for vocab. Map lookups, no allocation.
func BenchmarkConfig_InferFromWeights(b *testing.B) {
	weights := map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": {Shape: []int{8 * 256, 2560}},
		"model.embed_tokens.weight":              {Shape: []int{262144, 2560}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := benchGemma3Config()
		c.HeadDim, c.VocabSize = 0, 0 // force the shape-inference path
		c.InferFromWeights(weights)
	}
}
