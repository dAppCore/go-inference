// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/model/vision"
)

func BenchmarkNativeTokenModelEmbed(b *testing.B) {
	g, arch := gemma4BF16Fixture(b, 64, 1, 1, 64, 128, 32, 1)
	tm, err := NewBF16TokenModel(g, arch, 4)
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(arch.Hidden * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := tm.Embed(int32(i % arch.Vocab)); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeTokenModelEmbedInto(b *testing.B) {
	g, arch := gemma4BF16Fixture(b, 64, 1, 1, 64, 128, 32, 1)
	tm, err := NewBF16TokenModel(g, arch, 4)
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

func BenchmarkNativeTokenModelTokenEmbeddingsWithFeatures(b *testing.B) {
	const hidden = 128
	const imageTok = int32(88)
	const audioTok = int32(77)
	const videoTok = int32(99)
	ids := make([]int32, 256)
	imageSlots, audioSlots, videoSlots := 0, 0, 0
	for i := range ids {
		switch {
		case i%29 == 0:
			ids[i] = imageTok
			imageSlots++
		case i%31 == 0:
			ids[i] = audioTok
			audioSlots++
		case i%37 == 0:
			ids[i] = videoTok
			videoSlots++
		default:
			ids[i] = int32(i % 64)
		}
	}
	tm := &NativeTokenModel{
		NativeBackend: &NativeBackend{arch: model.Arch{Hidden: hidden}},
		vision: &vision.Loaded{Cfg: vision.Config{
			ImageTokenID: imageTok,
			VideoTokenID: videoTok,
		}},
		audio: &model.LoadedAudio{Cfg: model.LoadedAudioConfig{AudioTokenID: int(audioTok)}},
		embedInto: func(dst []byte, id int32) ([]byte, error) {
			for off := 0; off < len(dst); off += bf16Size {
				dst[off] = byte(id)
				dst[off+1] = byte(id >> 8)
			}
			return dst, nil
		},
	}
	imageFeatures := make([]byte, imageSlots*hidden*bf16Size)
	audioFeatures := make([]byte, audioSlots*hidden*bf16Size)
	videoFeatures := make([]byte, videoSlots*hidden*bf16Size)

	b.SetBytes(int64(len(ids) * hidden * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rows, err := tm.TokenEmbeddingsWithFeatures(ids, imageFeatures, audioFeatures, videoFeatures)
		if err != nil {
			b.Fatal(err)
		}
		if len(rows) != len(ids) {
			b.Fatalf("rows = %d, want %d", len(rows), len(ids))
		}
	}
}

type nativeTokenModelNoDirectGenerate struct {
	*NativeTokenModel
}

func (m nativeTokenModelNoDirectGenerate) OpenSession() (model.DecodeStepper, error) {
	sess, err := m.NativeTokenModel.OpenSession()
	if err != nil {
		return nil, err
	}
	return noDirectGenerateStepper{sess: sess}, nil
}

type noDirectGenerateStepper struct {
	sess model.DecodeStepper
}

func (s noDirectGenerateStepper) Step(emb []byte) ([]byte, error) {
	return s.sess.Step(emb)
}

func (s noDirectGenerateStepper) StepWithID(id int32, emb []byte) ([]byte, error) {
	if stepID, ok := s.sess.(interface {
		StepWithID(int32, []byte) ([]byte, error)
	}); ok {
		return stepID.StepWithID(id, emb)
	}
	return s.sess.Step(emb)
}

func (s noDirectGenerateStepper) Close() error {
	if c, ok := s.sess.(interface{ Close() error }); ok {
		return c.Close()
	}
	return nil
}

func BenchmarkNativeTokenModelGenerateStepwiseHead(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 32768, 2)
	tm, err := NewBF16TokenModel(g, arch, 16)
	if err != nil {
		b.Fatal(err)
	}
	wrapped := nativeTokenModelNoDirectGenerate{NativeTokenModel: tm}
	prompt := []int32{1, 5, 3, 9}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.Generate(wrapped, prompt, 6, -1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeTokenModelGenerateDirectSession(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 32768, 2)
	tm, err := NewBF16TokenModel(g, arch, 16)
	if err != nil {
		b.Fatal(err)
	}
	prompt := []int32{1, 5, 3, 9}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.Generate(tm, prompt, 6, -1); err != nil {
			b.Fatal(err)
		}
	}
}

type sampledStepwiseOnlyTokenModel struct {
	*NativeTokenModel
}

func (m sampledStepwiseOnlyTokenModel) OpenSession() (model.DecodeStepper, error) {
	sess, err := m.NativeTokenModel.OpenSession()
	if err != nil {
		return nil, err
	}
	return sampledStepwiseOnlyStepper{inner: sess}, nil
}

type sampledStepwiseOnlyStepper struct {
	inner model.DecodeStepper
}

func (s sampledStepwiseOnlyStepper) Step(emb []byte) ([]byte, error) {
	return s.inner.Step(emb)
}

func (s sampledStepwiseOnlyStepper) StepWithID(id int32, emb []byte) ([]byte, error) {
	if stepID, ok := s.inner.(interface {
		StepWithID(id int32, emb []byte) ([]byte, error)
	}); ok {
		return stepID.StepWithID(id, emb)
	}
	return s.inner.Step(emb)
}

func (s sampledStepwiseOnlyStepper) Close() error {
	if c, ok := s.inner.(interface{ Close() error }); ok {
		return c.Close()
	}
	return nil
}

func BenchmarkNativeTokenModelGenerateSampledStepwiseHead(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 32768, 2)
	tm, err := NewBF16TokenModel(g, arch, 16)
	if err != nil {
		b.Fatal(err)
	}
	stepwise := sampledStepwiseOnlyTokenModel{NativeTokenModel: tm}
	prompt := []int32{1, 5, 3, 9}
	params := model.SampleParams{Temperature: 1, TopK: 32}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.GenerateSampledWithStopTokens(stepwise, model.NewSampler(1), params, prompt, 6, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeTokenModelGenerateSampledNativeSessionOneShot(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 32768, 2)
	tm, err := NewBF16TokenModel(g, arch, 16)
	if err != nil {
		b.Fatal(err)
	}
	prompt := []int32{1, 5, 3, 9}
	params := model.SampleParams{Temperature: 1, TopK: 32}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.GenerateSampledWithStopTokens(tm, model.NewSampler(1), params, prompt, 6, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeTokenModelGenerateSampledNoEOSStepwiseHead(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 32768, 2)
	tm, err := NewBF16TokenModel(g, arch, 16)
	if err != nil {
		b.Fatal(err)
	}
	stepwise := sampledStepwiseOnlyTokenModel{NativeTokenModel: tm}
	prompt := []int32{1, 5, 3, 9}
	params := model.SampleParams{Temperature: 1, TopK: 32}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.GenerateSampled(stepwise, model.NewSampler(1), params, prompt, 6, -1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeTokenModelGenerateSampledNoEOSNativeSessionOneShot(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 32768, 2)
	tm, err := NewBF16TokenModel(g, arch, 16)
	if err != nil {
		b.Fatal(err)
	}
	prompt := []int32{1, 5, 3, 9}
	params := model.SampleParams{Temperature: 1, TopK: 32}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.GenerateSampled(tm, model.NewSampler(1), params, prompt, 6, -1); err != nil {
			b.Fatal(err)
		}
	}
}

func benchmarkNativeQuantTokenModelGenerateSampledWithParams(b *testing.B, direct bool, params model.SampleParams) {
	requireNativeRuntime(b)

	const gs, bits = 64, 4
	arch, err := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32768, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}.Arch()
	if err != nil {
		b.Fatalf("Arch: %v", err)
	}
	lm, err := model.Assemble(quantGemma4Tensors(b, arch, gs, bits), arch, model.StandardWeightNames())
	if err != nil {
		b.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		b.Fatalf("loadedToQuant: %v", err)
	}
	tm, err := NewQuantTokenModel(g, arch, 16)
	if err != nil {
		b.Fatal(err)
	}
	prompt := []int32{1, 5, 3, 9}
	stepwise := sampledStepwiseOnlyTokenModel{NativeTokenModel: tm}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !direct {
			if _, err := model.GenerateSampledWithStopTokens(stepwise, model.NewSampler(1), params, prompt, 6, nil); err != nil {
				b.Fatal(err)
			}
			continue
		}
		if _, err := model.GenerateSampledWithStopTokens(tm, model.NewSampler(1), params, prompt, 6, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeQuantTokenModelGenerateSampledStepwiseHead(b *testing.B) {
	benchmarkNativeQuantTokenModelGenerateSampledWithParams(b, false, model.SampleParams{Temperature: 1, TopK: 32})
}

func BenchmarkNativeQuantTokenModelGenerateSampledNativeSessionOneShot(b *testing.B) {
	benchmarkNativeQuantTokenModelGenerateSampledWithParams(b, true, model.SampleParams{Temperature: 1, TopK: 32})
}

func BenchmarkNativeQuantTokenModelGenerateSampledTopKOneStepwiseHead(b *testing.B) {
	benchmarkNativeQuantTokenModelGenerateSampledWithParams(b, false, model.SampleParams{Temperature: 1, TopK: 1})
}

func BenchmarkNativeQuantTokenModelGenerateSampledTopKOneNativeSessionOneShot(b *testing.B) {
	benchmarkNativeQuantTokenModelGenerateSampledWithParams(b, true, model.SampleParams{Temperature: 1, TopK: 1})
}

func BenchmarkNativeQuantTokenModelGenerateSampledTopKOneRepeatPenaltyStepwiseHead(b *testing.B) {
	benchmarkNativeQuantTokenModelGenerateSampledWithParams(b, false, model.SampleParams{Temperature: 1, TopK: 1, RepeatPenalty: 1.2})
}

func BenchmarkNativeQuantTokenModelGenerateSampledTopKOneRepeatPenaltyNativeSessionOneShot(b *testing.B) {
	benchmarkNativeQuantTokenModelGenerateSampledWithParams(b, true, model.SampleParams{Temperature: 1, TopK: 1, RepeatPenalty: 1.2})
}

func BenchmarkNativeQuantTokenModelGenerateSampledTopKTopPStepwiseHead(b *testing.B) {
	benchmarkNativeQuantTokenModelGenerateSampledWithParams(b, false, model.SampleParams{Temperature: 1, TopK: 32, TopP: 0.95})
}

func BenchmarkNativeQuantTokenModelGenerateSampledTopKTopPNativeSessionOneShot(b *testing.B) {
	benchmarkNativeQuantTokenModelGenerateSampledWithParams(b, true, model.SampleParams{Temperature: 1, TopK: 32, TopP: 0.95})
}

func BenchmarkNativeQuantTokenModelGenerateSampledTopKRepeatPenaltyStepwiseHead(b *testing.B) {
	benchmarkNativeQuantTokenModelGenerateSampledWithParams(b, false, model.SampleParams{Temperature: 1, TopK: 32, RepeatPenalty: 1.2})
}

func BenchmarkNativeQuantTokenModelGenerateSampledTopKRepeatPenaltyNativeSessionOneShot(b *testing.B) {
	benchmarkNativeQuantTokenModelGenerateSampledWithParams(b, true, model.SampleParams{Temperature: 1, TopK: 32, RepeatPenalty: 1.2})
}

func BenchmarkNativeQuantTokenModelGenerateSampledTempOnlyStepwiseHead(b *testing.B) {
	benchmarkNativeQuantTokenModelGenerateSampledWithParams(b, false, model.SampleParams{Temperature: 0.8})
}

func BenchmarkNativeQuantTokenModelGenerateSampledTempOnlyNativeSessionOneShot(b *testing.B) {
	benchmarkNativeQuantTokenModelGenerateSampledWithParams(b, true, model.SampleParams{Temperature: 0.8})
}

func BenchmarkNativeQuantTokenModelGenerateSampledRepeatPenaltyStepwiseHead(b *testing.B) {
	benchmarkNativeQuantTokenModelGenerateSampledWithParams(b, false, model.SampleParams{Temperature: 0.8, RepeatPenalty: 1.2})
}

func BenchmarkNativeQuantTokenModelGenerateSampledRepeatPenaltyNativeSessionOneShot(b *testing.B) {
	benchmarkNativeQuantTokenModelGenerateSampledWithParams(b, true, model.SampleParams{Temperature: 0.8, RepeatPenalty: 1.2})
}

func BenchmarkNativeQuantTokenModelGenerateSampledZeroTempStepwiseHead(b *testing.B) {
	benchmarkNativeQuantTokenModelGenerateSampledWithParams(b, false, model.SampleParams{Temperature: 0})
}

func BenchmarkNativeQuantTokenModelGenerateSampledZeroTempNativeSessionOneShot(b *testing.B) {
	benchmarkNativeQuantTokenModelGenerateSampledWithParams(b, true, model.SampleParams{Temperature: 0})
}
