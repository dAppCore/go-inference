// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkDecodeForwardArchQuantOneLayerTwoTokens(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const groupSize, bits = 64, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(b, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardArchQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeForwardArchQuantIntoOneLayerTwoTokens(b *testing.B) {
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
		if _, err := DecodeForwardArchQuantInto(outputs, inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeForwardArchQuantMoEOneLayerTwoTokens(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const expertDFF, numExperts, topK = 96, 4, 2
	const groupSize, bits = 32, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	arch.Layer[0].MoE = true
	inputs := decodeInputsFixture(2, dModel)
	layer := quantizedLayerFixture(b, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)
	moeWeights := quantMoELayerWeightsGuard(b, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)
	layer.MLPNormW, layer.Gate, layer.Up, layer.Down = nil, QuantWeight{}, QuantWeight{}, QuantWeight{}
	layer.MoE = &moeWeights
	layers := []QuantizedLayerWeights{layer}
	b.ReportAllocs()
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardArchQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeForwardArchQuantPLEOneLayerTwoTokens(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const pliDim, groupSize, bits = 32, 32, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(b, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	layers[0].PerLayerGate = quantWeightFixture(b, pliDim, dModel, groupSize, bits, 17)
	layers[0].PerLayerProjection = quantWeightFixture(b, dModel, pliDim, groupSize, bits, 23)
	layers[0].PostPerLayerInputNormW = toBF16Bytes(syntheticFloat32(dModel, 5))
	pleEmbed := quantWeightFixture(b, vocab, nLayers*pliDim, groupSize, bits, 31)
	ple := ArchPLEQuant{
		TokenIDs:            []int32{1, 2},
		EmbedPerLayer:       pleEmbed.Packed,
		EmbedPerLayerScales: pleEmbed.Scales,
		EmbedPerLayerBiases: pleEmbed.Biases,
		PerLayerModelProjW:  toBF16Bytes(syntheticFloat32(nLayers*pliDim*dModel, 37)),
		PerLayerProjNormW:   toBF16Bytes(syntheticFloat32(pliDim, 41)),
		VocabPLI:            vocab,
		PliDim:              pliDim,
		GroupSize:           groupSize,
		Bits:                bits,
		ProjGroupSize:       groupSize,
		ProjBits:            bits,
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardArchQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm, ple); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeForwardArchQuantPLEQuantProjectionOneLayerTwoTokens(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const pliDim, groupSize, bits = 32, 32, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(b, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	layers[0].PerLayerGate = quantWeightFixture(b, pliDim, dModel, groupSize, bits, 17)
	layers[0].PerLayerProjection = quantWeightFixture(b, dModel, pliDim, groupSize, bits, 23)
	layers[0].PostPerLayerInputNormW = toBF16Bytes(syntheticFloat32(dModel, 5))
	pleEmbed := quantWeightFixture(b, vocab, nLayers*pliDim, groupSize, bits, 31)
	pleProj := quantWeightFixture(b, nLayers*pliDim, dModel, groupSize, bits, 37)
	ple := ArchPLEQuant{
		TokenIDs:                []int32{1, 2},
		EmbedPerLayer:           pleEmbed.Packed,
		EmbedPerLayerScales:     pleEmbed.Scales,
		EmbedPerLayerBiases:     pleEmbed.Biases,
		PerLayerModelProjW:      pleProj.Packed,
		PerLayerModelProjScales: pleProj.Scales,
		PerLayerModelProjBiases: pleProj.Biases,
		PerLayerProjNormW:       toBF16Bytes(syntheticFloat32(pliDim, 41)),
		VocabPLI:                vocab,
		PliDim:                  pliDim,
		GroupSize:               groupSize,
		Bits:                    bits,
		ProjGroupSize:           groupSize,
		ProjBits:                bits,
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardArchQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm, ple); err != nil {
			b.Fatal(err)
		}
	}
}
