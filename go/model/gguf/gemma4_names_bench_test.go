// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "testing"

// BenchmarkGemma4Names_gemma4CanonicalTensorName measures the per-tensor name
// mapping cost — it runs once per source tensor (~600×) per conversion.
func BenchmarkGemma4Names_gemma4CanonicalTensorName(b *testing.B) {
	const src = "language_model.model.layers.31.self_attn.q_proj.weight"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := gemma4CanonicalTensorName(src); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGemma4Names_gemma4GGUFShape measures the shape-reversal cost paid per
// tensor alongside the name mapping.
func BenchmarkGemma4Names_gemma4GGUFShape(b *testing.B) {
	shape := []uint64{6144, 1536}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = gemma4GGUFShape(shape)
	}
}
