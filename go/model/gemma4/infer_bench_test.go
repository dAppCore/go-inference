// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// The gemma4 infer benches baseline the weight-shape inference (AX-11) — the don't-guess dim
// recovery gemma4 runs at load between Parse and Arch: inferGemma4HeadDim reads a
// target-attention layer's head dim from its q_proj rows (gemma4 carries distinct sliding vs
// global head dims), inferGemma4PerLayerInputSize the E2B/E4B PLE-tower width from the
// per-layer projection. Both are map-lookup + shape-arithmetic bound, run once per load.
// Config.Arch + Assemble are benched separately; this covers infer.go. Synthetic tensor set —
// no file.

func benchGemma4Weights() map[string]safetensors.Tensor {
	w := make(map[string]safetensors.Tensor, 64)
	for i := 0; i < 34; i++ {
		p := "model.layers." + itoa(i)
		w[p+".self_attn.q_proj.weight"] = safetensors.Tensor{Shape: []int{8 * 256, 2560}}
	}
	w["model.embed_tokens.weight"] = safetensors.Tensor{Shape: []int{262144, 2560}}
	w["model.per_layer_model_projection.weight"] = safetensors.Tensor{Shape: []int{34, 256, 2560}}
	return w
}

func benchGemma4LayerTypes() []string {
	lt := make([]string, 34)
	for i := range lt {
		if (i+1)%6 == 0 {
			lt[i] = "full_attention"
		} else {
			lt[i] = "sliding_attention"
		}
	}
	return lt
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [4]byte
	n := len(b)
	for i > 0 {
		n--
		b[n] = byte('0' + i%10)
		i /= 10
	}
	return string(b[n:])
}

// BenchmarkInferGemma4HeadDim — resolving the first sliding-attention layer's head dim from
// its q_proj rows: a scan to the matching layer + a rows÷heads divide, no allocation beyond
// the Sprintf'd name.
func BenchmarkInferGemma4HeadDim(b *testing.B) {
	w := benchGemma4Weights()
	lt := benchGemma4LayerTypes()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if inferGemma4HeadDim(w, lt, 8, "sliding_attention") != 256 {
			b.Fatal("head dim mis-resolved")
		}
	}
}

// BenchmarkInferGemma4PerLayerInputSize — the PLE-tower width from the per-layer projection's
// flattened out-features ÷ layer count: one lookup + a short product, no allocation.
func BenchmarkInferGemma4PerLayerInputSize(b *testing.B) {
	w := benchGemma4Weights()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if inferGemma4PerLayerInputSize(w, 34) != 256 {
			b.Fatal("PLE width mis-resolved")
		}
	}
}
