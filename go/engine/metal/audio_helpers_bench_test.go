// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

var (
	benchClampF32  []float32
	benchClampBF16 []byte
)

func BenchmarkClampF32NoOp(b *testing.B) {
	in := syntheticFloat32(1024, 17)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchClampF32 = clampF32(in, 0, 0)
	}
}

func BenchmarkClampF32Active(b *testing.B) {
	in := syntheticFloat32(1024, 17)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchClampF32 = clampF32(in, -1, 1)
	}
}

func BenchmarkClampBF16NoOp(b *testing.B) {
	in := toBF16Bytes(syntheticFloat32(1024, 17))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchClampBF16 = clampBF16(in, 0, 0)
	}
}

func BenchmarkClampBF16Active(b *testing.B) {
	in := toBF16Bytes(syntheticFloat32(1024, 17))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchClampBF16 = clampBF16(in, -1, 1)
	}
}
