// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkNativeBackendBF16DecodeForward(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	backend, err := NewBF16Backend(arch, layers, maxLen)
	if err != nil {
		b.Fatal(err)
	}
	inputs := decodeInputsFixture(2, dModel)
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := backend.DecodeForward(inputs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeBackendBF16DecodeForwardInto(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	backend, err := NewBF16Backend(arch, layers, maxLen)
	if err != nil {
		b.Fatal(err)
	}
	inputs := decodeInputsFixture(2, dModel)
	outputs := make([][]byte, len(inputs))
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := backend.DecodeForwardInto(outputs, inputs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeBackendBF16ICBDecodeForward(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	backend, err := NewBF16Backend(arch, layers, maxLen, WithICB())
	if err != nil {
		b.Fatal(err)
	}
	inputs := decodeInputsFixture(2, dModel)
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := backend.DecodeForward(inputs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeBackendBF16ICBDecodeForwardInto(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	backend, err := NewBF16Backend(arch, layers, maxLen, WithICB())
	if err != nil {
		b.Fatal(err)
	}
	inputs := decodeInputsFixture(2, dModel)
	outputs := make([][]byte, len(inputs))
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := backend.DecodeForwardInto(outputs, inputs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeBackendQuantDecodeForward(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const groupSize, bits = 64, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(b, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	backend, err := NewQuantBackend(arch, layers, maxLen)
	if err != nil {
		b.Fatal(err)
	}
	inputs := decodeInputsFixture(2, dModel)
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := backend.DecodeForward(inputs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeBackendQuantDecodeForwardInto(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const groupSize, bits = 64, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(b, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	backend, err := NewQuantBackend(arch, layers, maxLen)
	if err != nil {
		b.Fatal(err)
	}
	inputs := decodeInputsFixture(2, dModel)
	outputs := make([][]byte, len(inputs))
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := backend.DecodeForwardInto(outputs, inputs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeBackendQuantICBDecodeForward(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const groupSize, bits = 64, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(b, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	backend, err := NewQuantBackend(arch, layers, maxLen, WithICB())
	if err != nil {
		b.Fatal(err)
	}
	inputs := decodeInputsFixture(2, dModel)
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := backend.DecodeForward(inputs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeBackendQuantICBDecodeForwardInto(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const groupSize, bits = 64, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(b, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	backend, err := NewQuantBackend(arch, layers, maxLen, WithICB())
	if err != nil {
		b.Fatal(err)
	}
	inputs := decodeInputsFixture(2, dModel)
	outputs := make([][]byte, len(inputs))
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := backend.DecodeForwardInto(outputs, inputs); err != nil {
			b.Fatal(err)
		}
	}
}
