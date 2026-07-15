// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

var sinkParallelResidual any
var sinkApplyResidualOrder any

func BenchmarkParallelResidual(b *testing.B) {
	residual := make([]float32, 4096)
	attention := make([]float32, 4096)
	mlp := make([]float32, 4096)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkParallelResidual = ParallelResidual(residual, attention, mlp).Value
	}
}

func BenchmarkApplyResidualOrder(b *testing.B) {
	hidden := make([]float32, 4096)
	id := func(x []float32) []float32 { return x }
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkApplyResidualOrder = ApplyResidualOrder(NormPlacementPost, hidden, id, id, id, id).Value
	}
}
