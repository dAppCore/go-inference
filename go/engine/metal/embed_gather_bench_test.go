// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkEmbedGatherQuantBF16256x512(b *testing.B) {
	requireNativeRuntime(b)
	if !gpuHasGeluKernel() {
		b.Skip("custom kernel library not loaded")
	}

	const vocab, dModel, groupSize, bits = 256, 512, 64, 4
	const scale = float32(0.5)
	packed, scales, biases := embedGatherQuantFixture(vocab, dModel, groupSize, bits)
	b.SetBytes(int64(len(packed) + len(scales) + len(biases)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := EmbedGatherQuantBF16(42, packed, scales, biases, dModel, groupSize, bits, scale); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEmbedGatherQuantBF16Into256x512(b *testing.B) {
	requireNativeRuntime(b)
	if !gpuHasGeluKernel() {
		b.Skip("custom kernel library not loaded")
	}

	const vocab, dModel, groupSize, bits = 256, 512, 64, 4
	const scale = float32(0.5)
	packed, scales, biases := embedGatherQuantFixture(vocab, dModel, groupSize, bits)
	out := make([]byte, dModel*bf16Size)
	b.SetBytes(int64(len(packed) + len(scales) + len(biases)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := EmbedGatherQuantBF16Into(out, 42, packed, scales, biases, dModel, groupSize, bits, scale); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEmbedGatherQuantBF16AlternatingDModel(b *testing.B) {
	requireNativeRuntime(b)
	if !gpuHasGeluKernel() {
		b.Skip("custom kernel library not loaded")
	}

	const vocab, groupSize, bits = 256, 64, 4
	const scale = float32(0.5)
	type fixture struct {
		dModel                 int
		packed, scales, biases []byte
	}
	makeFixture := func(dModel int) fixture {
		packed, scales, biases := embedGatherQuantFixture(vocab, dModel, groupSize, bits)
		return fixture{dModel: dModel, packed: packed, scales: scales, biases: biases}
	}
	fixtures := []fixture{makeFixture(512), makeFixture(1024)}
	perCallBytes := 0
	for _, f := range fixtures {
		perCallBytes += len(f.packed) + len(f.scales) + len(f.biases)
		if _, err := EmbedGatherQuantBF16(42, f.packed, f.scales, f.biases, f.dModel, groupSize, bits, scale); err != nil {
			b.Fatal(err)
		}
	}
	b.SetBytes(int64(perCallBytes / len(fixtures)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := fixtures[i&1]
		if _, err := EmbedGatherQuantBF16(42, f.packed, f.scales, f.biases, f.dModel, groupSize, bits, scale); err != nil {
			b.Fatal(err)
		}
	}
}
