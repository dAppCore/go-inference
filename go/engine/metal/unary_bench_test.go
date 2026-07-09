// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkUnarySquare1024(b *testing.B) {
	requireNativeRuntime(b)

	in := syntheticFloat32(1024, 3)
	b.SetBytes(int64(len(in) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Square(in); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkUnarySquareInto1024(b *testing.B) {
	requireNativeRuntime(b)

	in := syntheticFloat32(1024, 3)
	out := make([]float32, len(in))
	b.SetBytes(int64(len(in) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := RunUnaryInto("v_Squarefloat32float32", in, out); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSigmoidBF161024(b *testing.B) {
	requireNativeRuntime(b)

	in := toBF16Bytes(syntheticFloat32(1024, 3))
	b.SetBytes(int64(len(in)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := SigmoidBF16(in); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSigmoidBF16Into1024(b *testing.B) {
	requireNativeRuntime(b)

	in := toBF16Bytes(syntheticFloat32(1024, 3))
	out := make([]byte, len(in))
	b.SetBytes(int64(len(in)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := RunUnaryBF16Into("v_Sigmoidbfloat16bfloat16", in, out); err != nil {
			b.Fatal(err)
		}
	}
}
