// SPDX-Licence-Identifier: EUPL-1.2
package fp8

import "testing"

var sinkTensor Tensor

func BenchmarkQuantize(b *testing.B) {
	values := make([]float32, 4096)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		sinkTensor, _ = Quantize(values)
	}
}
func BenchmarkDequantize(b *testing.B) {
	tensor, _ := Quantize(make([]float32, 4096))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_, _ = Dequantize(tensor)
	}
}
