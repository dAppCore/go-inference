// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

func BenchmarkDecodeForwardArchOneLayerTwoTokens(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	b.ReportAllocs()
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardArch(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeForwardArchIntoOneLayerTwoTokens(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	outputs := make([][]byte, len(inputs))
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardArchInto(outputs, inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeForwardArchMoEOneLayerTwoTokens(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const numExperts, topK, expertDFF = 4, 2, 96
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	arch.Layer[0].MoE = true
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	layers[0].MoE = buildMoEWeights(numExperts, topK, dModel, dFF, expertDFF, 9)
	b.ReportAllocs()
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardArch(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkArchDecodeStateSetup(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	specs := []model.LayerSpec{{CacheIndex: -1}}
	layers := []archLayerBufs{{dFF: dFF}}
	withAutoreleasePool(func() {
		warm := newArchDecodeState(specs, layers, nil, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, 10000, 10000, 0.125, 1e-5, false, maxLen)
		warm.Close()

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			st := newArchDecodeState(specs, layers, nil, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, 10000, 10000, 0.125, 1e-5, false, maxLen)
			st.Close()
		}
	})
}

func BenchmarkArchDecodeStateGlobalProportionalRopePeriods(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	specs := []model.LayerSpec{{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: headDim, KVHeads: nKV}}
	layers := []archLayerBufs{{dFF: dFF}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		withAutoreleasePool(func() {
			st := newArchDecodeState(specs, layers, nil, dModel, nHeads, nKV, headDim, dFF, 0, 32, headDim, 10000, 10000, 0.125, 1e-5, false, maxLen)
			if st.globalRopeFreqs == nil || st.globalRopeFreqs.GetID() == 0 {
				b.Fatal("missing global proportional rope periods")
			}
		})
	}
}
