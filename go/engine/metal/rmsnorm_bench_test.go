// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkRMSNormRows4Axis1024(b *testing.B) {
	requireNativeRuntime(b)

	const rows, axis = 4, 1024
	x := syntheticFloat32(rows*axis, 3)
	w := syntheticFloat32(axis, 5)
	b.SetBytes(int64(len(x) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := RMSNorm(x, w, rows, axis, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRMSNormIntoRows4Axis1024(b *testing.B) {
	requireNativeRuntime(b)

	const rows, axis = 4, 1024
	x := syntheticFloat32(rows*axis, 3)
	w := syntheticFloat32(axis, 5)
	out := make([]float32, len(x))
	b.SetBytes(int64(len(x) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = RMSNormInto(out, x, w, rows, axis, 1e-5)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRMSNormBF16Rows4Axis512(b *testing.B) {
	requireNativeRuntime(b)

	const rows, axis = 4, 512
	x, w := rmsNormBF16Fixture(rows, axis)
	b.SetBytes(int64(len(x) + len(w)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := RMSNormBF16(x, w, rows, axis, 1e-6); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRMSNormBF16IntoRows4Axis512(b *testing.B) {
	requireNativeRuntime(b)

	const rows, axis = 4, 512
	x, w := rmsNormBF16Fixture(rows, axis)
	out := make([]byte, rows*axis*bf16Size)
	b.SetBytes(int64(len(x) + len(w)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = RMSNormBF16Into(out, x, w, rows, axis, 1e-6)
		if err != nil {
			b.Fatal(err)
		}
	}
}
