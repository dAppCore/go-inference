// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkRoPEHeads8Dim64(b *testing.B) {
	requireNativeRuntime(b)

	x := syntheticFloat32(8*64, 3)
	b.SetBytes(int64(len(x) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := RoPE(x, 1, 8, 64, 10000, 1, 17, false); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRoPEIntoHeads8Dim64(b *testing.B) {
	requireNativeRuntime(b)

	x := syntheticFloat32(8*64, 3)
	out := make([]float32, len(x))
	b.SetBytes(int64(len(x) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = RoPEInto(out, x, 1, 8, 64, 10000, 1, 17, false)
		if err != nil {
			b.Fatal(err)
		}
	}
}
