// SPDX-Licence-Identifier: EUPL-1.2

package gptq

import "testing"

var benchmarkTensor Tensor
var benchmarkValues []float32

func BenchmarkQuantize(b *testing.B) {
	values := make([]float32, 256*256)
	for i := range values {
		values[i] = float32(i%31-15) / 32
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		benchmarkTensor, _ = Quantize(values, 256, 256, Options{Bits: 4, GroupSize: 128, Symmetric: true})
	}
}

func BenchmarkDequantize(b *testing.B) {
	values := make([]float32, 256*256)
	tensor, _ := Quantize(values, 256, 256, Options{Bits: 4, GroupSize: 128, Symmetric: true})
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		benchmarkValues, _ = Dequantize(tensor)
	}
}
