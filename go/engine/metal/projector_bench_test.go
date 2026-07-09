// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkProjectorHasV(b *testing.B) {
	requireNativeRuntime(b)

	p := qmvProjector{v: qmvWeight{wq: copyView(toBF16Bytes([]float32{1}))}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = p.hasV()
	}
}
