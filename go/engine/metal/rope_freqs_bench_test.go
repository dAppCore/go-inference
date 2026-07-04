// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

func BenchmarkRoPEFreqsBF16Heads8Dim64(b *testing.B) {
	requireNativeRuntime(b)

	const batch, nHeads, headDim, rotaryDim = 1, 8, 64, 64
	x := toBF16Bytes(syntheticFloat32(batch*nHeads*headDim, 5))
	invFreqs := make([]float32, rotaryDim/2)
	for i := range invFreqs {
		invFreqs[i] = float32(math.Pow(10000, -float64(2*i)/float64(rotaryDim)))
	}
	b.SetBytes(int64(len(x)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := RoPEFreqsBF16(x, batch, nHeads, headDim, rotaryDim, invFreqs, 1, 7, false); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRoPEFreqsBF16IntoHeads8Dim64(b *testing.B) {
	requireNativeRuntime(b)

	const batch, nHeads, headDim, rotaryDim = 1, 8, 64, 64
	x := toBF16Bytes(syntheticFloat32(batch*nHeads*headDim, 5))
	out := make([]byte, len(x))
	invFreqs := make([]float32, rotaryDim/2)
	for i := range invFreqs {
		invFreqs[i] = float32(math.Pow(10000, -float64(2*i)/float64(rotaryDim)))
	}
	b.SetBytes(int64(len(x)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := RoPEFreqsBF16Into(out, x, batch, nHeads, headDim, rotaryDim, invFreqs, 1, 7, false); err != nil {
			b.Fatal(err)
		}
	}
}
