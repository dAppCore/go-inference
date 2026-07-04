// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkDecodeForwardArchICBQuantOneLayerTwoTokens(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const groupSize, bits = 64, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(b, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeForwardArchICBQuantIntoOneLayerTwoTokens(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const groupSize, bits = 64, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(b, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	outputs := make([][]byte, len(inputs))
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardArchICBQuantInto(outputs, inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeForwardArchICBQuantPipelinedFourTokens(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 8
	const groupSize, bits = 64, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(4, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(b, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeForwardArchICBQuantIntoPipelinedFourTokens(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 8
	const groupSize, bits = 64, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(4, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(b, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	outputs := make([][]byte, len(inputs))
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardArchICBQuantInto(outputs, inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
			b.Fatal(err)
		}
	}
}
