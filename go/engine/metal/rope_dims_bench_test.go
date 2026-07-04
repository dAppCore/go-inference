// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkRoPEDimsBF16Heads8Dim64Rotary32(b *testing.B) {
	requireNativeRuntime(b)

	const batch, nHeads, headDim, rotaryDim = 1, 8, 64, 32
	x := toBF16Bytes(syntheticFloat32(batch*nHeads*headDim, 5))
	b.SetBytes(int64(len(x)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := RoPEDimsBF16(x, batch, nHeads, headDim, rotaryDim, 10000, 1, 7, false); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRoPEDimsBF16IntoHeads8Dim64Rotary32(b *testing.B) {
	requireNativeRuntime(b)

	const batch, nHeads, headDim, rotaryDim = 1, 8, 64, 32
	x := toBF16Bytes(syntheticFloat32(batch*nHeads*headDim, 5))
	out := make([]byte, len(x))
	b.SetBytes(int64(len(x)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := RoPEDimsBF16Into(out, x, batch, nHeads, headDim, rotaryDim, 10000, 1, 7, false); err != nil {
			b.Fatal(err)
		}
	}
}
