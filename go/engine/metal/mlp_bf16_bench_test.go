// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkMulBF161024(b *testing.B) {
	requireNativeRuntime(b)

	a := toBF16Bytes(syntheticFloat32(1024, 3))
	c := toBF16Bytes(syntheticFloat32(1024, 5))
	b.ReportAllocs()
	b.SetBytes(int64(len(a) + len(c)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MulBF16(a, c); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMulBF16Into1024(b *testing.B) {
	requireNativeRuntime(b)

	a := toBF16Bytes(syntheticFloat32(1024, 3))
	c := toBF16Bytes(syntheticFloat32(1024, 5))
	out := make([]byte, len(a))
	b.ReportAllocs()
	b.SetBytes(int64(len(a) + len(c)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := MulBF16Into(out, a, c); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTanhBF161024(b *testing.B) {
	requireNativeRuntime(b)

	x := toBF16Bytes(syntheticFloat32(1024, 3))
	b.ReportAllocs()
	b.SetBytes(int64(len(x)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := TanhBF16(x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTanhBF16Into1024(b *testing.B) {
	requireNativeRuntime(b)

	x := toBF16Bytes(syntheticFloat32(1024, 3))
	out := make([]byte, len(x))
	b.ReportAllocs()
	b.SetBytes(int64(len(x)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := TanhBF16Into(out, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMulBF16Const1024(b *testing.B) {
	requireNativeRuntime(b)

	x := toBF16Bytes(syntheticFloat32(1024, 3))
	b.ReportAllocs()
	b.SetBytes(int64(len(x)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := mulBF16Const(x, 1024, 0.375); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMulBF16ConstInto1024(b *testing.B) {
	requireNativeRuntime(b)

	x := toBF16Bytes(syntheticFloat32(1024, 3))
	out := make([]byte, len(x))
	b.ReportAllocs()
	b.SetBytes(int64(len(x)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := mulBF16ConstInto(x, 1024, 0.375, out); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGeluGateMulBF161024(b *testing.B) {
	requireNativeRuntime(b)

	gate := toBF16Bytes(syntheticFloat32(1024, 3))
	up := toBF16Bytes(syntheticFloat32(1024, 5))
	b.SetBytes(int64(len(gate) + len(up)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := GeluGateMulBF16(gate, up); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGeluGateMulBF16Composed1024(b *testing.B) {
	requireNativeRuntime(b)
	old := customLibraryLoaded
	customLibraryLoaded = false
	defer func() { customLibraryLoaded = old }()

	gate := toBF16Bytes(syntheticFloat32(1024, 3))
	up := toBF16Bytes(syntheticFloat32(1024, 5))
	b.ReportAllocs()
	b.SetBytes(int64(len(gate) + len(up)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := GeluGateMulBF16(gate, up); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGeluGateMulBF16ComposedInto1024(b *testing.B) {
	requireNativeRuntime(b)
	old := customLibraryLoaded
	customLibraryLoaded = false
	defer func() { customLibraryLoaded = old }()

	gate := toBF16Bytes(syntheticFloat32(1024, 3))
	up := toBF16Bytes(syntheticFloat32(1024, 5))
	out := make([]byte, len(gate))
	b.ReportAllocs()
	b.SetBytes(int64(len(gate) + len(up)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := GeluGateMulBF16Into(out, gate, up); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGeluBF161024(b *testing.B) {
	requireNativeRuntime(b)

	x := toBF16Bytes(syntheticFloat32(1024, 3))
	b.ReportAllocs()
	b.SetBytes(int64(len(x)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := GeluBF16(x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGeluBF16Into1024(b *testing.B) {
	requireNativeRuntime(b)

	x := toBF16Bytes(syntheticFloat32(1024, 3))
	out := make([]byte, len(x))
	b.ReportAllocs()
	b.SetBytes(int64(len(x)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := GeluBF16Into(out, x); err != nil {
			b.Fatal(err)
		}
	}
}
