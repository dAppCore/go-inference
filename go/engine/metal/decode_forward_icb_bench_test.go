// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkDecodeForwardICBOneLayerTwoTokens(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardICB(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeForwardICBIntoOneLayerTwoTokens(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	outputs := make([][]byte, len(inputs))
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardICBInto(outputs, inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeForwardICBAlternatingShape(b *testing.B) {
	requireNativeRuntime(b)

	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	cases := []struct {
		dModel, nHeads, nKV, headDim, dFF, maxLen int
		inputs                                    [][]byte
		layers                                    []DecodeLayerWeights
	}{
		{dModel: 64, nHeads: 1, nKV: 1, headDim: 64, dFF: 128, maxLen: 4},
		{dModel: 128, nHeads: 2, nKV: 1, headDim: 64, dFF: 256, maxLen: 4},
	}
	var totalBytes int64
	for i := range cases {
		cases[i].inputs = decodeInputsFixture(2, cases[i].dModel)
		cases[i].layers = []DecodeLayerWeights{decodeLayerFixture(cases[i].dModel, cases[i].nHeads, cases[i].nKV, cases[i].headDim, cases[i].dFF, 3)}
		totalBytes += int64(len(cases[i].inputs) * cases[i].dModel * bf16Size)
		if _, err := DecodeForwardICB(cases[i].inputs, cases[i].layers, cases[i].dModel, cases[i].nHeads, cases[i].nKV, cases[i].headDim, cases[i].maxLen, cases[i].dFF, base, scale, eps); err != nil {
			b.Fatalf("warmup dModel %d: %v", cases[i].dModel, err)
		}
	}
	b.SetBytes(totalBytes)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := cases[i&1]
		if _, err := DecodeForwardICB(c.inputs, c.layers, c.dModel, c.nHeads, c.nKV, c.headDim, c.maxLen, c.dFF, base, scale, eps); err != nil {
			b.Fatal(err)
		}
	}
}
