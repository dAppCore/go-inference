// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkLayerScalarBuf128(b *testing.B) {
	requireNativeRuntime(b)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	scalarW := toBF16Bytes([]float32{0.75})
	b.ReportAllocs()
	b.SetBytes(128 * bf16Size)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if buf := layerScalarBuf(scalarW, 128); buf == nil {
			b.Fatal("nil layer scalar buffer")
		}
	}
}

func BenchmarkValueNormOnesBuf256(b *testing.B) {
	requireNativeRuntime(b)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	b.ReportAllocs()
	b.SetBytes(256 * bf16Size)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if buf := valueNormOnesBuf(true, 256); buf == nil {
			b.Fatal("nil value norm ones buffer")
		}
	}
}
