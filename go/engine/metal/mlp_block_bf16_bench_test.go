// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkMLPBlockBF16_64x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, dFF = 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	normW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 11))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 13))
	b.SetBytes(int64(len(x) + len(normW) + len(wGate) + len(wUp) + len(wDown)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MLPBlockBF16(x, normW, wGate, wUp, wDown, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMLPBlockBF16Into_64x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, dFF = 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	normW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 11))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 13))
	out := make([]byte, dModel*bf16Size)
	b.ReportAllocs()
	b.SetBytes(int64(len(x) + len(normW) + len(wGate) + len(wUp) + len(wDown)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MLPBlockBF16Into(out, x, normW, wGate, wUp, wDown, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMLPBlockBF16Composed64x128(b *testing.B) {
	requireNativeRuntime(b)
	old := customLibraryLoaded
	customLibraryLoaded = false
	defer func() { customLibraryLoaded = old }()
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF = 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	normW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 11))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 13))
	b.ReportAllocs()
	b.SetBytes(int64(len(x) + len(normW) + len(wGate) + len(wUp) + len(wDown)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MLPBlockBF16(x, normW, wGate, wUp, wDown, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMLPBlockBF16IntoComposed64x128(b *testing.B) {
	requireNativeRuntime(b)
	old := customLibraryLoaded
	customLibraryLoaded = false
	defer func() { customLibraryLoaded = old }()
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF = 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	normW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 11))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 13))
	out := make([]byte, dModel*bf16Size)
	b.ReportAllocs()
	b.SetBytes(int64(len(x) + len(normW) + len(wGate) + len(wUp) + len(wDown)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MLPBlockBF16Into(out, x, normW, wGate, wUp, wDown, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}
