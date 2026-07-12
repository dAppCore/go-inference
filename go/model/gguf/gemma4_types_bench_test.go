// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "testing"

// BenchmarkGemma4Types_gemma4TensorType measures the per-tensor type-policy
// lookup — it runs once per source tensor during a conversion.
func BenchmarkGemma4Types_gemma4TensorType(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = gemma4TensorType("blk.6.ffn_down.weight", 6, 35)
	}
}
