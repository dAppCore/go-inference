// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"testing"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

func BenchmarkNativeQuantTokenModelEmbed(b *testing.B) {
	const gs, bits = 32, 4
	arch, err := g4.Config{
		HiddenSize: 64, NumHiddenLayers: 1, IntermediateSize: 128,
		NumAttentionHeads: 1, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}.Arch()
	if err != nil {
		b.Fatalf("Arch: %v", err)
	}
	lm, err := model.Assemble(quantGemma4TensorsB(arch, gs, bits), arch, model.StandardWeightNames())
	if err != nil {
		b.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		b.Fatalf("loadedToQuant: %v", err)
	}
	tm, err := NewQuantTokenModel(g, arch, 4)
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(arch.Hidden * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := tm.Embed(int32(i % arch.Vocab)); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeQuantTokenModelEmbedInto(b *testing.B) {
	const gs, bits = 32, 4
	arch, err := g4.Config{
		HiddenSize: 64, NumHiddenLayers: 1, IntermediateSize: 128,
		NumAttentionHeads: 1, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}.Arch()
	if err != nil {
		b.Fatalf("Arch: %v", err)
	}
	lm, err := model.Assemble(quantGemma4TensorsB(arch, gs, bits), arch, model.StandardWeightNames())
	if err != nil {
		b.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		b.Fatalf("loadedToQuant: %v", err)
	}
	tm, err := NewQuantTokenModel(g, arch, 4)
	if err != nil {
		b.Fatal(err)
	}
	dst := make([]byte, tm.EmbeddingBytes())
	b.SetBytes(int64(arch.Hidden * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := tm.EmbedInto(dst, int32(i%arch.Vocab)); err != nil {
			b.Fatal(err)
		}
	}
}
