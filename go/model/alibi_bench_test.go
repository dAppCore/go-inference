// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

var alibiSink []float32

func BenchmarkALiBiSlopes(b *testing.B) {
	for i := 0; i < b.N; i++ {
		alibiSink = ALiBiSlopes(32)
	}
}

func BenchmarkApplyALiBi(b *testing.B) {
	scores := make([]float64, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ApplyALiBi(scores, 0.5, 1023, 0)
	}
}
