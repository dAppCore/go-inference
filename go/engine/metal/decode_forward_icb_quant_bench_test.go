// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkDecodeForwardICBQuantOneLayerTwoTokens(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(b, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardICBQuant(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeForwardICBQuantIntoOneLayerTwoTokens(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(b, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	outputs := make([][]byte, len(inputs))
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardICBQuantInto(outputs, inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps); err != nil {
			b.Fatal(err)
		}
	}
}
