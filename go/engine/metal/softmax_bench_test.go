// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkSoftmaxF32Rows8Axis512(b *testing.B) {
	requireNativeRuntime(b)

	const rows, axisSize = 8, 512
	x := syntheticFloat32(rows*axisSize, 5)
	b.SetBytes(int64(len(x) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := SoftmaxF32(x, axisSize); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSoftmaxF32IntoRows8Axis512(b *testing.B) {
	requireNativeRuntime(b)

	const rows, axisSize = 8, 512
	x := syntheticFloat32(rows*axisSize, 5)
	out := make([]float32, len(x))
	b.SetBytes(int64(len(x) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = SoftmaxF32Into(out, x, axisSize)
		if err != nil {
			b.Fatal(err)
		}
	}
}
