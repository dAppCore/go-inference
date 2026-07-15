// SPDX-Licence-Identifier: EUPL-1.2

package bert

import "testing"

// BenchmarkLinear measures the per-vector projection at bge-small's hidden size
// — the innermost hot loop of the encoder forward.
func BenchmarkLinear(b *testing.B) {
	const hidden = 384
	x := make([]float32, hidden)
	weight := make([]float32, hidden*hidden)
	bias := make([]float32, hidden)
	for i := range x {
		x[i] = float32(i%7) * 0.1
	}
	for i := range weight {
		weight[i] = float32(i%13) * 0.01
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = linear(x, weight, bias, hidden, hidden)
	}
}

// BenchmarkLinearBatch measures a sequence-wide bge-small projection.
func BenchmarkLinearBatch(b *testing.B) {
	const (
		hidden = 384
		rows   = 14
	)
	x := makeRows(rows, hidden)
	weight := make([]float32, hidden*hidden)
	bias := make([]float32, hidden)
	for i := range x {
		for j := range x[i] {
			x[i][j] = float32((i+j)%7) * 0.1
		}
	}
	for i := range weight {
		weight[i] = float32(i%13) * 0.01
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = linearBatch(x, weight, bias, hidden, hidden)
	}
}

// BenchmarkLayerNorm measures a single-vector LayerNorm at bge-small's width.
func BenchmarkLayerNorm(b *testing.B) {
	const hidden = 384
	x := make([]float32, hidden)
	weight := make([]float32, hidden)
	bias := make([]float32, hidden)
	for i := range x {
		x[i] = float32(i%11) * 0.3
		weight[i] = 1
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = layerNorm(x, weight, bias, 1e-12)
	}
}
