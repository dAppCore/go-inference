// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkBinaryAdd1024(b *testing.B) {
	requireNativeRuntime(b)

	a := syntheticFloat32(1024, 3)
	c := syntheticFloat32(1024, 5)
	b.SetBytes(int64(len(a) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Add(a, c); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBinaryAddInto1024(b *testing.B) {
	requireNativeRuntime(b)

	a := syntheticFloat32(1024, 3)
	c := syntheticFloat32(1024, 5)
	out := make([]float32, len(a))
	b.SetBytes(int64(len(a) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := RunBinaryInto("vv_Addfloat32", a, c, out); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBinaryAddAlternatingSizes(b *testing.B) {
	requireNativeRuntime(b)

	type fixture struct {
		a, c []float32
	}
	fixtures := []fixture{
		{a: syntheticFloat32(1024, 3), c: syntheticFloat32(1024, 5)},
		{a: syntheticFloat32(2048, 7), c: syntheticFloat32(2048, 11)},
	}
	perCallBytes := 0
	for _, f := range fixtures {
		perCallBytes += len(f.a) * 4
		if _, err := Add(f.a, f.c); err != nil {
			b.Fatal(err)
		}
	}
	b.SetBytes(int64(perCallBytes / len(fixtures)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := fixtures[i&1]
		if _, err := Add(f.a, f.c); err != nil {
			b.Fatal(err)
		}
	}
}
