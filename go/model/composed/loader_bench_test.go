// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// The loader benches baseline the per-load widening + layer-schedule work (AX-11):
// tensorF32 widens a bf16 checkpoint tensor to a flat f32 slice — run once per weight at
// load, its allocation is the whole f32 copy (the biggest single load allocation, sized to
// the weight). resolveKinds maps each layer to full/linear attention from the config, a
// small []string per load. Neither is per-token; these pin the load-time cost. Synthetic
// tensors — no checkpoint read.

// BenchmarkTensorF32_BF16 — widening a projection-sized bf16 weight to f32: the per-element
// left-shift unpack into a fresh [len/2] f32 buffer. The dominant per-weight load allocation.
func BenchmarkTensorF32_BF16(b *testing.B) {
	n := benchFF * benchD
	t := safetensors.Tensor{Shape: []int{benchFF, benchD}, Dtype: "BF16", Data: make([]byte, n*2)}
	b.SetBytes(int64(len(t.Data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := tensorF32(t); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkResolveKinds_Interval — deriving a 48-layer schedule from full_attention_interval
// (every 6th layer full): one []string allocation + a modulo pass. The per-load layer-typing.
func BenchmarkResolveKinds_Interval(b *testing.B) {
	cfg := &loaderConfig{NumHiddenLayers: 48, FullAttentionInterval: 6}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := resolveKinds(cfg); err != nil {
			b.Fatal(err)
		}
	}
}
