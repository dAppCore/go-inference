// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

var sampleHistoryBenchSink []int32
var samplePenaltyBenchSink []byte
var sampleSuppressBenchSink []int32
var archSessionHiddenBenchSink []byte
var archSessionSampleTokenBenchSink int32

func newQuantICBStepBenchSession(tb testing.TB, maxLen int) *ArchSession {
	tb.Helper()
	const gs, bits = 64, 4
	arch, err := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 256, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}.Arch()
	if err != nil {
		tb.Fatalf("Arch: %v", err)
	}
	lm, err := model.Assemble(quantGemma4Tensors(tb, arch, gs, bits), arch, model.StandardWeightNames())
	if err != nil {
		tb.Fatalf("Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		tb.Fatalf("loadedToQuant: %v", err)
	}
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		tb.Fatalf("NewArchQuantSession: %v", err)
	}
	if sess.state.icb == nil {
		tb.Skip("ICB replay unavailable")
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
		tb.Fatalf("PrefillTokens: %v", err)
	}
	return sess
}

func BenchmarkArchSessionEmbedID(b *testing.B) {
	const vocab, dModel = 64, 128
	table := toBF16Bytes(syntheticFloat32(vocab*dModel, 17))
	scale := float32(1.25)
	tokens := []int32{0, 7, 31, 63}

	b.Run("owned", func(b *testing.B) {
		sess := &ArchSession{
			arch: model.Arch{Hidden: dModel, Vocab: vocab},
			embed: func(id int32) ([]byte, error) {
				return embedTokenBF16(table, id, vocab, dModel, scale)
			},
		}
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			out, err := sess.embedID(tokens[i%len(tokens)])
			if err != nil {
				b.Fatalf("embedID owned: %v", err)
			}
			archSessionHiddenBenchSink = out
		}
	})

	b.Run("scratch", func(b *testing.B) {
		sess := &ArchSession{
			arch: model.Arch{Hidden: dModel, Vocab: vocab},
			embed: func(id int32) ([]byte, error) {
				return embedTokenBF16(table, id, vocab, dModel, scale)
			},
			embedInto: func(dst []byte, id int32) ([]byte, error) {
				return embedTokenBF16Into(dst, table, id, vocab, dModel, scale)
			},
		}
		sess.markDefaultEmbedFunc()
		if _, err := sess.embedID(tokens[0]); err != nil {
			b.Fatalf("embedID scratch warmup: %v", err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out, err := sess.embedID(tokens[i%len(tokens)])
			if err != nil {
				b.Fatalf("embedID scratch: %v", err)
			}
			archSessionHiddenBenchSink = out
		}
	})
}

func BenchmarkArchSessionSampleVocabLargeTempOnly(b *testing.B) {
	const vocab = 4096
	logits := toBF16Bytes(syntheticFloat32(vocab, 91))
	sess := &ArchSession{}
	params := model.SampleParams{Temperature: 1}
	sampler := model.NewSampler(1)
	if _, err := sess.sampleVocabBF16(logits, vocab, sampler, params); err != nil {
		b.Fatalf("sampleVocabBF16 warmup: %v", err)
	}
	if cap(sess.sampleOrder) != 0 {
		b.Fatalf("temp-only warmup grew rank scratch: %d", cap(sess.sampleOrder))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok, err := sess.sampleVocabBF16(logits, vocab, model.NewSampler(uint64(i+1)), params)
		if err != nil {
			b.Fatalf("sampleVocabBF16: %v", err)
		}
		archSessionSampleTokenBenchSink = tok
	}
}

func BenchmarkArchSessionSampleVocabLargeTopP(b *testing.B) {
	const vocab = 4096
	logits := toBF16Bytes(syntheticFloat32(vocab, 92))
	sess := &ArchSession{}
	params := model.SampleParams{Temperature: 1, TopP: 0.72}
	sampler := model.NewSampler(1)
	if _, err := sess.sampleVocabBF16(logits, vocab, sampler, params); err != nil {
		b.Fatalf("sampleVocabBF16 warmup: %v", err)
	}
	if cap(sess.sampleOrder) < vocab {
		b.Fatalf("TopP warmup rank scratch cap = %d, want at least %d", cap(sess.sampleOrder), vocab)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok, err := sess.sampleVocabBF16(logits, vocab, model.NewSampler(uint64(i+1)), params)
		if err != nil {
			b.Fatalf("sampleVocabBF16: %v", err)
		}
		archSessionSampleTokenBenchSink = tok
	}
}

func BenchmarkArchSessionSampleVocabLargeTopPPeaked(b *testing.B) {
	const vocab = 4096
	logits := toBF16Bytes(peakedSampleFloat32(vocab))
	sess := &ArchSession{}
	params := model.SampleParams{Temperature: 1, TopP: 0.92}
	sampler := model.NewSampler(1)
	if _, err := sess.sampleVocabBF16(logits, vocab, sampler, params); err != nil {
		b.Fatalf("sampleVocabBF16 warmup: %v", err)
	}
	if cap(sess.sampleOrder) < vocab {
		b.Fatalf("TopP warmup rank scratch cap = %d, want at least %d", cap(sess.sampleOrder), vocab)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok, err := sess.sampleVocabBF16(logits, vocab, model.NewSampler(uint64(i+1)), params)
		if err != nil {
			b.Fatalf("sampleVocabBF16: %v", err)
		}
		archSessionSampleTokenBenchSink = tok
	}
}

func peakedSampleFloat32(n int) []float32 {
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = 8 - float32(i)*0.25
	}
	return vals
}

func BenchmarkArchSessionStepIDInPoolICBHiddenReadback(b *testing.B) {
	requireNativeRuntime(b)
	g, arch, maxLen := icbSessionStateFixture(b)
	sess := newICBSessionStateFixture(b, g, arch, maxLen)
	if sess.state.icb == nil {
		b.Fatal("fixture must build an ICB replay session")
	}
	ids := []int32{1, 5, 3, 2}

	b.ReportAllocs()
	b.ResetTimer()
	withAutoreleasePool(func() {
		for i := 0; i < b.N; i++ {
			sess.pos = i % (maxLen - 1)
			h, err := sess.stepIDInPool(ids[i%len(ids)])
			if err != nil {
				b.Fatalf("stepIDInPool: %v", err)
			}
			archSessionHiddenBenchSink = h
		}
	})
}

func BenchmarkArchSessionStepIDInPoolNonICBTransientHidden(b *testing.B) {
	requireNativeRuntime(b)
	g, arch, maxLen := icbSessionStateFixture(b)
	sess := newICBSessionStateFixture(b, g, arch, maxLen)
	sess.state.icb = nil
	ids := []int32{1, 5, 3, 2}
	if _, err := sess.stepIDInPool(ids[0]); err != nil {
		b.Fatalf("stepIDInPool warmup: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	withAutoreleasePool(func() {
		for i := 0; i < b.N; i++ {
			sess.pos = i % (maxLen - 1)
			h, err := sess.stepIDInPool(ids[i%len(ids)])
			if err != nil {
				b.Fatalf("stepIDInPool: %v", err)
			}
			archSessionHiddenBenchSink = h
		}
	})
}

func BenchmarkArchSessionStepIDRetainedInPoolNonICB(b *testing.B) {
	requireNativeRuntime(b)
	g, arch, maxLen := icbSessionStateFixture(b)
	sess := newICBSessionStateFixture(b, g, arch, maxLen)
	sess.state.icb = nil
	ids := []int32{1, 5, 3, 2}
	if _, err := sess.stepIDRetainedInPool(ids[0]); err != nil {
		b.Fatalf("stepIDRetainedInPool warmup: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	withAutoreleasePool(func() {
		for i := 0; i < b.N; i++ {
			sess.pos = i % (maxLen - 1)
			h, err := sess.stepIDRetainedInPool(ids[i%len(ids)])
			if err != nil {
				b.Fatalf("stepIDRetainedInPool: %v", err)
			}
			archSessionHiddenBenchSink = h
		}
	})
}

func BenchmarkArchSessionCloseSessionOwnedScratch(b *testing.B) {
	candidateLogits := []byte{1, 2}
	candidateIDs := []int32{3}
	headLogits := []byte{4, 5}
	hidden := []byte{6, 7}
	nextInputEmbHost := []byte{8, 9}
	nextInputPLEHost := []byte{10, 11}
	history := []int32{8}
	penaltyIDs := []int32{9}
	penaltyLogits := []byte{10, 11}
	scaled := []float32{0.1}
	probs := []float32{0.2}
	order := []int32{0}
	suppress := []int32{12}
	var token int32
	var emb byte
	nextPL, tailPL0, tailPL1 := &plGPUScratch{}, &plGPUScratch{}, &plGPUScratch{}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sess := ArchSession{
			sampleCandidateLogits: candidateLogits,
			sampleCandidateIDs:    candidateIDs,
			sampleHeadLogits:      headLogits,
			sampleHidden:          hidden,
			sampleHistory:         history,
			samplePenaltyIDs:      penaltyIDs,
			samplePenaltyLogits:   penaltyLogits,
			sampleScaled:          scaled,
			sampleProbs:           probs,
			sampleOrder:           order,
			sampleSuppressTokens:  suppress,
			nextInputTokenPtr:     &token,
			nextInputEmbPtr:       &emb,
			nextInputEmbHost:      nextInputEmbHost,
			nextInputPLEHost:      nextInputPLEHost,
			nextInputPLScratch:    nextPL,
			gpuTailPLScratch:      [2]*plGPUScratch{tailPL0, tailPL1},
		}
		sess.closeSessionOwnedScratch()
		if sess.sampleCandidateLogits != nil || sess.sampleScaled != nil || sess.sampleProbs != nil || sess.sampleOrder != nil || sess.nextInputEmbHost != nil || sess.nextInputPLScratch != nil || sess.gpuTailPLScratch[0] != nil {
			b.Fatal("session-owned scratch survived close cleanup")
		}
	}
}

func BenchmarkArchSessionCloseModelAndDecodeStateReferences(b *testing.B) {
	embed := func(int32) ([]byte, error) { return nil, nil }
	head := func([]byte, bool) ([]byte, error) { return nil, nil }
	greedy := func([]byte, []int32) (int32, bool, error) { return 0, false, nil }
	perLayer := func(int32, []byte) ([]byte, error) { return nil, nil }
	plScratch := func() *plGPUScratch { return nil }
	recordPeer := func() (*archICBReplay, error) { return nil, nil }
	cachedIDs := []int32{1, 2}
	cachedPromptIDs := []int32{1}
	cachedPromptHidden := []byte{2, 3}
	cachedPromptLogits := []byte{4, 5}
	retainedHidden := []byte{6, 7}
	stateInput := []byte{8, 9}
	stateScratch := []byte{10, 11}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sess := ArchSession{
			arch:               model.Arch{Hidden: 4, Vocab: 8},
			embed:              embed,
			head:               head,
			greedy:             greedy,
			headEnc:            &headEncoder{},
			perLayerInput:      perLayer,
			plScratchNew:       plScratch,
			recordPeerICB:      recordPeer,
			icbPeer:            &archICBReplay{},
			state:              archDecodeState{specs: []model.LayerSpec{{}}, perLayerInput: stateInput, hostScratch: stateScratch, icb: &archICBReplay{}},
			pos:                2,
			maxLen:             8,
			cachedIDs:          cachedIDs,
			cachedPromptIDs:    cachedPromptIDs,
			cachedPromptHidden: cachedPromptHidden,
			cachedPromptLogits: cachedPromptLogits,
			retainedHidden:     retainedHidden,
		}
		sess.closeModelAndDecodeStateReferences()
		if sess.embed != nil || sess.state.specs != nil || sess.cachedIDs != nil || sess.arch.Hidden != 0 {
			b.Fatal("model/decode references survived close cleanup")
		}
	}
}

func BenchmarkArchSessionSampleHistoryScratchFor(b *testing.B) {
	b.Run("no-repeat-penalty", func(b *testing.B) {
		params := model.SampleParams{Temperature: 1, TopK: 32}
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			sess := &ArchSession{}
			history := sess.sampleHistoryScratchFor(params, 32)
			if len(history) != 0 || cap(history) != 0 {
				b.Fatalf("history scratch len/cap = %d/%d, want 0/0", len(history), cap(history))
			}
			sampleHistoryBenchSink = history
		}
	})
	b.Run("repeat-penalty", func(b *testing.B) {
		params := model.SampleParams{Temperature: 1, TopK: 32, RepeatPenalty: 1.2}
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			sess := &ArchSession{}
			history := sess.sampleHistoryScratchFor(params, 32)
			if len(history) != 0 || cap(history) < 32 {
				b.Fatalf("history scratch len/cap = %d/%d, want 0/>=32", len(history), cap(history))
			}
			sampleHistoryBenchSink = history
		}
	})
}

func BenchmarkNewArchSession(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 64, 1, 1, 64, 128, 32, 1)
	b.SetBytes(int64(len(g.Embed) + len(g.Layers[0].WGate)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess, err := NewArchSession(g, arch, 4)
		if err != nil {
			b.Fatal(err)
		}
		_ = sess.Close()
	}
}

func BenchmarkArchSessionGenerateJoinedPrompt(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 64, 2)
	prefix := []int32{1, 2, 3}
	suffix := []int32{4, 5}
	full := append(append([]int32{}, prefix...), suffix...)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess, err := NewArchSession(g, arch, 24)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := sess.Generate(full, 4, -1); err != nil {
			b.Fatalf("Generate: %v", err)
		}
		_ = sess.Close()
	}
}

func BenchmarkArchSessionPrefillAppendGenerateFromCache(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 64, 2)
	prefix := []int32{1, 2, 3}
	suffix := []int32{4, 5}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess, err := NewArchSession(g, arch, 24)
		if err != nil {
			b.Fatal(err)
		}
		if err := sess.PrefillTokens(prefix); err != nil {
			b.Fatalf("PrefillTokens: %v", err)
		}
		if err := sess.AppendTokens(suffix); err != nil {
			b.Fatalf("AppendTokens: %v", err)
		}
		if _, err := sess.GenerateFromCache(4, -1); err != nil {
			b.Fatalf("GenerateFromCache: %v", err)
		}
		_ = sess.Close()
	}
}

func BenchmarkArchSessionReplayFullPromptSecondTurn(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 64, 2)
	full := []int32{1, 2, 3, 4, 5}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess, err := NewArchSession(g, arch, 24)
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		if _, err := sess.Generate(full, 4, -1); err != nil {
			b.Fatalf("Generate: %v", err)
		}
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkArchSessionPrefillRetainedDense(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 64, 2)
	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	embeddingSource, err := NewArchSession(g, arch, 24)
	if err != nil {
		b.Fatal(err)
	}
	embeddings := make([][]byte, len(ids))
	for i, id := range ids {
		emb, err := embeddingSource.embedID(id)
		if err != nil {
			b.Fatal(err)
		}
		embeddings[i] = append([]byte(nil), emb...)
	}
	_ = embeddingSource.Close()
	b.Run("prefix-plus-final-step", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			sess, err := NewArchSession(g, arch, 24)
			if err != nil {
				b.Fatal(err)
			}
			sess.state.icb = nil
			if err := sess.prefillCachedIDs(ids[:len(ids)-1]); err != nil {
				b.Fatal(err)
			}
			withAutoreleasePool(func() {
				_, err = sess.stepIDInPool(ids[len(ids)-1])
			})
			if err != nil {
				b.Fatal(err)
			}
			_ = sess.Close()
		}
	})
	b.Run("batched-retained-hidden", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			sess, err := NewArchSession(g, arch, 24)
			if err != nil {
				b.Fatal(err)
			}
			sess.state.icb = nil
			if _, err := sess.prefillRetainedTokens(ids, "bench"); err != nil {
				b.Fatal(err)
			}
			_ = sess.Close()
		}
	})
	b.Run("explicit-embeddings", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			sess, err := NewArchSession(g, arch, 24)
			if err != nil {
				b.Fatal(err)
			}
			sess.state.icb = nil
			if err := sess.PrefillTokenEmbeddings(ids, embeddings); err != nil {
				b.Fatal(err)
			}
			_ = sess.Close()
		}
	})
	b.Run("explicit-embeddings-icb", func(b *testing.B) {
		const gs, bits = 64, 4
		icbArch, err := g4.Config{
			HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
			NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 256, RMSNormEps: 1e-6,
			Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
		}.Arch()
		if err != nil {
			b.Fatalf("Arch: %v", err)
		}
		lm, err := model.Assemble(quantGemma4Tensors(b, icbArch, gs, bits), icbArch, model.StandardWeightNames())
		if err != nil {
			b.Fatalf("Assemble: %v", err)
		}
		icbG, err := loadedToQuant(lm, gs, bits)
		if err != nil {
			b.Fatalf("loadedToQuant: %v", err)
		}
		icbIDs := []int32{1, 2, 3, 4, 5, 6, 7, 8}
		embeddingSource, err := NewArchQuantSession(icbG, icbArch, 24)
		if err != nil {
			b.Fatalf("NewArchQuantSession embeddings: %v", err)
		}
		if embeddingSource.state.icb == nil {
			b.Skip("ICB replay unavailable")
		}
		icbEmbeddings := make([][]byte, len(icbIDs))
		for i, id := range icbIDs {
			emb, err := embeddingSource.embedID(id)
			if err != nil {
				b.Fatalf("embedID(%d): %v", id, err)
			}
			icbEmbeddings[i] = append([]byte(nil), emb...)
		}
		_ = embeddingSource.Close()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			sess, err := NewArchQuantSession(icbG, icbArch, 24)
			if err != nil {
				b.Fatal(err)
			}
			if sess.state.icb == nil {
				b.Skip("ICB replay unavailable")
			}
			if err := sess.PrefillTokenEmbeddings(icbIDs, icbEmbeddings); err != nil {
				b.Fatal(err)
			}
			_ = sess.Close()
		}
	})
	b.Run("explicit-embeddings-icb-ple", func(b *testing.B) {
		const gs, bits = 64, 4
		icbArch, err := g4.Config{
			HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
			NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 256, RMSNormEps: 1e-6,
			HiddenSizePerLayerInput: 64, VocabSizePerLayerInput: 256,
			Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
		}.Arch()
		if err != nil {
			b.Fatalf("Arch: %v", err)
		}
		ts := quantGemma4Tensors(b, icbArch, gs, bits)
		addPLETensors(b, ts, icbArch, gs, bits)
		lm, err := model.Assemble(ts, icbArch, model.StandardWeightNames())
		if err != nil {
			b.Fatalf("Assemble: %v", err)
		}
		icbG, err := loadedToQuant(lm, gs, bits)
		if err != nil {
			b.Fatalf("loadedToQuant: %v", err)
		}
		if !icbG.HasPLE() {
			b.Fatal("assembled benchmark model should have PLE tensors")
		}
		icbIDs := []int32{1, 2, 3, 4, 5, 6, 7, 8}
		embeddingSource, err := NewArchQuantSession(icbG, icbArch, 24)
		if err != nil {
			b.Fatalf("NewArchQuantSession embeddings: %v", err)
		}
		if embeddingSource.state.icb == nil || !embeddingSource.state.icb.hasPLE {
			b.Skip("PLE ICB replay unavailable")
		}
		icbEmbeddings := make([][]byte, len(icbIDs))
		for i, id := range icbIDs {
			emb, err := embeddingSource.embedID(id)
			if err != nil {
				b.Fatalf("embedID(%d): %v", id, err)
			}
			icbEmbeddings[i] = append([]byte(nil), emb...)
		}
		_ = embeddingSource.Close()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			sess, err := NewArchQuantSession(icbG, icbArch, 24)
			if err != nil {
				b.Fatal(err)
			}
			if sess.state.icb == nil || !sess.state.icb.hasPLE {
				b.Skip("PLE ICB replay unavailable")
			}
			if err := sess.PrefillTokenEmbeddings(icbIDs, icbEmbeddings); err != nil {
				b.Fatal(err)
			}
			_ = sess.Close()
		}
	})
	b.Run("batched-retained-hidden-two-chunks", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			sess, err := NewArchSession(g, arch, 24)
			if err != nil {
				b.Fatal(err)
			}
			sess.state.icb = nil
			if _, err := sess.prefillRetainedTokens(ids[:4], "bench"); err != nil {
				b.Fatal(err)
			}
			if _, err := sess.prefillRetainedTokens(ids[4:], "bench"); err != nil {
				b.Fatal(err)
			}
			_ = sess.Close()
		}
	})

	slidingG, slidingArch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 64, 1)
	slidingArch.SlidingWindow = 4
	slidingArch.Layer[0].Attention = model.SlidingAttention
	slidingIDs := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	b.Run("sliding-serial-steps", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			sess, err := NewArchSession(slidingG, slidingArch, 24)
			if err != nil {
				b.Fatal(err)
			}
			sess.state.icb = nil
			withAutoreleasePool(func() {
				for _, id := range slidingIDs {
					if _, err = sess.stepIDInPool(id); err != nil {
						return
					}
				}
			})
			if err != nil {
				b.Fatal(err)
			}
			_ = sess.Close()
		}
	})
	b.Run("sliding-batched-chunks", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			sess, err := NewArchSession(slidingG, slidingArch, 24)
			if err != nil {
				b.Fatal(err)
			}
			sess.state.icb = nil
			if _, err := sess.prefillRetainedTokens(slidingIDs, "bench"); err != nil {
				b.Fatal(err)
			}
			_ = sess.Close()
		}
	})
}

func BenchmarkArchSessionAppendGenerateFromCacheSecondTurn(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 64, 2)
	prefix := []int32{1, 2, 3}
	suffix := []int32{4, 5}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess, err := NewArchSession(g, arch, 24)
		if err != nil {
			b.Fatal(err)
		}
		if err := sess.PrefillTokens(prefix); err != nil {
			b.Fatalf("PrefillTokens: %v", err)
		}
		b.StartTimer()
		if err := sess.AppendTokens(suffix); err != nil {
			b.Fatalf("AppendTokens: %v", err)
		}
		if _, err := sess.GenerateFromCache(4, -1); err != nil {
			b.Fatalf("GenerateFromCache: %v", err)
		}
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkArchSessionSampleHistoryFresh(b *testing.B) {
	const maxNew = 8
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		history := make([]int32, 0, maxNew)
		for j := range maxNew {
			history = append(history, int32(i+j))
		}
		if len(history) != maxNew {
			b.Fatal("sample history length mismatch")
		}
		sampleHistoryBenchSink = history
	}
}

func BenchmarkArchSessionSampleHistoryScratch(b *testing.B) {
	const maxNew = 8
	sess := &ArchSession{}
	history := sess.sampleHistoryScratch(maxNew)
	for j := range maxNew {
		history = append(history, int32(j))
	}
	sess.sampleHistory = history

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		history = sess.sampleHistoryScratch(maxNew)
		for j := range maxNew {
			history = append(history, int32(i+j))
		}
		sess.sampleHistory = history
		if len(sess.sampleHistory) != maxNew {
			b.Fatal("sample history length mismatch")
		}
		sampleHistoryBenchSink = sess.sampleHistory
	}
}

func BenchmarkArchSessionRepeatPenaltyFresh(b *testing.B) {
	const vocab = 32768
	logits := make([]byte, vocab*bf16Size)
	for i := range logits {
		logits[i] = byte(i)
	}
	history := []int32{31, 7, 1024, 7, 2048, -1, vocab + 1, 16384}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		out, err := nativeApplyRepeatPenaltyBF16(logits, vocab, history, 1.2)
		if err != nil {
			b.Fatal(err)
		}
		samplePenaltyBenchSink = out
	}
}

func BenchmarkArchSessionRepeatPenaltyScratch(b *testing.B) {
	const vocab = 32768
	logits := make([]byte, vocab*bf16Size)
	for i := range logits {
		logits[i] = byte(i)
	}
	history := []int32{31, 7, 1024, 7, 2048, -1, vocab + 1, 16384}
	sess := &ArchSession{}
	if _, err := sess.repeatPenaltyLogitsScratch(logits, vocab, history, 1.2); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := sess.repeatPenaltyLogitsScratch(logits, vocab, history, 1.2)
		if err != nil {
			b.Fatal(err)
		}
		samplePenaltyBenchSink = out
	}
}

func BenchmarkArchSessionRepeatPenaltyScratchDuplicateHistory(b *testing.B) {
	const vocab = 32768
	logits := make([]byte, vocab*bf16Size)
	for i := range logits {
		logits[i] = byte(i)
	}
	history := []int32{31, 31, 31, 7, 7, 7, 7, 1024, 1024, 2048, 2048, 2048, -1, vocab + 1, 16384, 16384}
	sess := &ArchSession{}
	if _, err := sess.repeatPenaltyLogitsScratch(logits, vocab, history, 1.2); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := sess.repeatPenaltyLogitsScratch(logits, vocab, history, 1.2)
		if err != nil {
			b.Fatal(err)
		}
		if len(sess.samplePenaltyIDs) != 5 {
			b.Fatalf("unique penalty ids = %d, want 5", len(sess.samplePenaltyIDs))
		}
		samplePenaltyBenchSink = out
		sampleHistoryBenchSink = sess.samplePenaltyIDs
	}
}

func BenchmarkArchSessionSampleTokenFromLogitsTopKRepeatPenalty(b *testing.B) {
	const vocab = 32768
	logits := make([]byte, vocab*bf16Size)
	for i := range logits {
		logits[i] = byte(i)
	}
	params := model.SampleParams{Temperature: 1, TopK: 32, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{31, 7, 1024, 7, 2048, -1, vocab + 1, 16384}
	sess := &ArchSession{arch: model.Arch{Vocab: vocab}}
	if tok, err := sess.sampleTokenFromLogits(logits, model.NewSampler(1), params, history); err != nil {
		b.Fatal(err)
	} else {
		archSessionSampleTokenBenchSink = tok
	}
	if len(sess.sampleCandidateIDs) != params.TopK {
		b.Fatalf("candidate ids len = %d, want %d", len(sess.sampleCandidateIDs), params.TopK)
	}
	if sess.samplePenaltyLogits != nil {
		b.Fatal("TopK repeat-penalty sampling used vocab-sized repeat-penalty scratch")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok, err := sess.sampleTokenFromLogits(logits, model.NewSampler(uint64(i+2)), params, history)
		if err != nil {
			b.Fatal(err)
		}
		archSessionSampleTokenBenchSink = tok
	}
}

func BenchmarkArchSessionHeadLogitsFresh(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	g, arch := gemma4BF16Fixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, 16)
	if err != nil {
		b.Fatalf("NewArchSession: %v", err)
	}
	hidden := toBF16Bytes(syntheticFloat32(dModel, 47))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := sess.head(hidden, false)
		if err != nil {
			b.Fatal(err)
		}
		samplePenaltyBenchSink = out
	}
}

func BenchmarkArchSessionHeadLogitsScratch(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	g, arch := gemma4BF16Fixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, 16)
	if err != nil {
		b.Fatalf("NewArchSession: %v", err)
	}
	hidden := toBF16Bytes(syntheticFloat32(dModel, 47))
	if _, err := sess.headLogitsScratch(hidden, false); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := sess.headLogitsScratch(hidden, false)
		if err != nil {
			b.Fatal(err)
		}
		samplePenaltyBenchSink = out
	}
}

func BenchmarkArchSessionBoundaryLogitsRetainedHiddenNoCopy(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	g, arch := gemma4BF16Fixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, 16)
	if err != nil {
		b.Fatalf("NewArchSession: %v", err)
	}
	hidden := toBF16Bytes(syntheticFloat32(dModel, 48))
	sess.rememberRetainedHidden(hidden)
	if sess.retainedHiddenBuffer() == nil {
		b.Fatal("retained hidden did not expose no-copy buffer")
	}
	if _, err := sess.BoundaryLogits(); err != nil {
		b.Fatalf("BoundaryLogits warmup: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess.resetRetainedLogits()
		out, err := sess.BoundaryLogits()
		if err != nil {
			b.Fatal(err)
		}
		samplePenaltyBenchSink = out
	}
}

func BenchmarkArchSessionHeadGreedyFreshHidden(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	g, arch := gemma4BF16Fixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, 16)
	if err != nil {
		b.Fatalf("NewArchSession: %v", err)
	}
	hidden := toBF16Bytes(syntheticFloat32(dModel, 51))
	if _, err := sess.headGreedyOrLogits(hidden, nil, nil, nil, false); err != nil {
		b.Fatalf("headGreedyOrLogits warmup: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok, err := sess.headGreedyOrLogits(hidden, nil, nil, nil, false)
		if err != nil {
			b.Fatal(err)
		}
		archSessionSampleTokenBenchSink = tok
	}
}

func BenchmarkArchSessionHeadGreedyRetainedHiddenNoCopy(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	g, arch := gemma4BF16Fixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, 16)
	if err != nil {
		b.Fatalf("NewArchSession: %v", err)
	}
	sess.rememberRetainedHidden(toBF16Bytes(syntheticFloat32(dModel, 51)))
	if sess.retainedHiddenBuffer() == nil {
		b.Fatal("retained hidden did not expose no-copy buffer")
	}
	if _, err := sess.headGreedyOrLogits(sess.retainedHidden, nil, nil, nil, false); err != nil {
		b.Fatalf("headGreedyOrLogits warmup: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok, err := sess.headGreedyOrLogits(sess.retainedHidden, nil, nil, nil, false)
		if err != nil {
			b.Fatal(err)
		}
		archSessionSampleTokenBenchSink = tok
	}
}

func BenchmarkArchSessionSampleTopKCandidatesFreshHidden(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	g, arch := gemma4BF16Fixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, 16)
	if err != nil {
		b.Fatalf("NewArchSession: %v", err)
	}
	hidden := toBF16Bytes(syntheticFloat32(dModel, 49))
	params := model.SampleParams{Temperature: 1, TopK: 5, TopP: 0.5, SuppressTokens: []int32{2, 7}}
	if _, _, ok, err := sess.sampleTopKCandidatesFromHiddenInPool(hidden, params); err != nil {
		b.Fatalf("sampleTopKCandidates warmup: %v", err)
	} else if !ok {
		b.Fatal("sampleTopKCandidates declined")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits, ids, ok, err := sess.sampleTopKCandidatesFromHiddenInPool(hidden, params)
		if err != nil {
			b.Fatal(err)
		}
		if !ok {
			b.Fatal("sampleTopKCandidates declined")
		}
		samplePenaltyBenchSink = logits
		sampleHistoryBenchSink = ids
	}
}

func BenchmarkArchSessionSampleTopKCandidatesFreshHiddenRepeatPenalty(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	g, arch := gemma4BF16Fixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, 16)
	if err != nil {
		b.Fatalf("NewArchSession: %v", err)
	}
	hidden := toBF16Bytes(syntheticFloat32(dModel, 49))
	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	if _, _, ok, err := sess.sampleTopKCandidatesFromHiddenWithHistoryInPool(hidden, params, history); err != nil {
		b.Fatalf("sampleTopKCandidates warmup: %v", err)
	} else if !ok {
		b.Fatal("sampleTopKCandidates declined")
	}
	if sess.samplePenaltyLogits != nil {
		b.Fatal("TopK candidate repeat-penalty path used vocab-sized repeat-penalty scratch")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits, ids, ok, err := sess.sampleTopKCandidatesFromHiddenWithHistoryInPool(hidden, params, history)
		if err != nil {
			b.Fatal(err)
		}
		if !ok {
			b.Fatal("sampleTopKCandidates declined")
		}
		samplePenaltyBenchSink = logits
		sampleHistoryBenchSink = ids
	}
}

func BenchmarkArchSessionSampleTopKCandidatesRetainedHiddenNoCopy(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	g, arch := gemma4BF16Fixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, 16)
	if err != nil {
		b.Fatalf("NewArchSession: %v", err)
	}
	sess.rememberRetainedHidden(toBF16Bytes(syntheticFloat32(dModel, 49)))
	if sess.retainedHiddenBuffer() == nil {
		b.Fatal("retained hidden did not expose no-copy buffer")
	}
	params := model.SampleParams{Temperature: 1, TopK: 5, TopP: 0.5, SuppressTokens: []int32{2, 7}}
	if _, _, ok, err := sess.sampleTopKCandidatesFromHiddenInPool(sess.retainedHidden, params); err != nil {
		b.Fatalf("sampleTopKCandidates warmup: %v", err)
	} else if !ok {
		b.Fatal("sampleTopKCandidates declined")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits, ids, ok, err := sess.sampleTopKCandidatesFromHiddenInPool(sess.retainedHidden, params)
		if err != nil {
			b.Fatal(err)
		}
		if !ok {
			b.Fatal("sampleTopKCandidates declined")
		}
		samplePenaltyBenchSink = logits
		sampleHistoryBenchSink = ids
	}
}

func BenchmarkArchSessionStepGreedyICB(b *testing.B) {
	requireNativeRuntime(b)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess := newQuantICBStepBenchSession(b, 16)
		if _, _, ok, err := sess.stepGreedyInPool(9, nil, nil); err != nil || !ok {
			b.Fatalf("stepGreedyInPool warmup ok=%v err=%v", ok, err)
		}
		b.StartTimer()
		tok, hidden, ok, err := sess.stepGreedyInPool(9, nil, nil)
		if err != nil || !ok {
			b.Fatalf("stepGreedyInPool ok=%v err=%v", ok, err)
		}
		archSessionSampleTokenBenchSink = tok
		archSessionHiddenBenchSink = hidden
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkArchSessionStepSampleLogitsTokenICB(b *testing.B) {
	requireNativeRuntime(b)

	params := model.SampleParams{Temperature: 0.8, MinP: 0.02, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess := newQuantICBStepBenchSession(b, 16)
		if _, _, ok, err := sess.stepSampleLogitsTokenInPool(9, params, 0.37, history); err != nil || !ok {
			b.Fatalf("stepSampleLogitsTokenInPool warmup ok=%v err=%v", ok, err)
		}
		b.StartTimer()
		hidden, tok, ok, err := sess.stepSampleLogitsTokenInPool(9, params, 0.37, history)
		if err != nil || !ok {
			b.Fatalf("stepSampleLogitsTokenInPool ok=%v err=%v", ok, err)
		}
		archSessionHiddenBenchSink = hidden
		archSessionSampleTokenBenchSink = tok
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkArchSessionStepSampleLogitsTokenICBGPUInputs(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := pleQuantModel(b, 2, 256, 32, 0)
	params := model.SampleParams{Temperature: 0.8, MinP: 0.02, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess, err := NewArchQuantSession(g, arch, 16)
		if err != nil {
			b.Fatalf("NewArchQuantSession: %v", err)
		}
		if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
			b.Fatalf("PrefillTokens: %v", err)
		}
		if sess.encNextInputsGPU == nil {
			b.Fatal("fixture did not wire GPU next-inputs seam")
		}
		if _, _, ok, err := sess.stepSampleLogitsTokenInPool(9, params, 0.37, history); err != nil || !ok {
			b.Fatalf("stepSampleLogitsTokenInPool warmup ok=%v err=%v", ok, err)
		}
		b.StartTimer()
		hidden, tok, ok, err := sess.stepSampleLogitsTokenInPool(9, params, 0.37, history)
		if err != nil || !ok {
			b.Fatalf("stepSampleLogitsTokenInPool ok=%v err=%v", ok, err)
		}
		archSessionHiddenBenchSink = hidden
		archSessionSampleTokenBenchSink = tok
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkArchSessionStepSampleLogitsTokenICBHostPLE(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := pleQuantModel(b, 2, 256, 32, 0)
	params := model.SampleParams{Temperature: 0.8, MinP: 0.02, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	old := chainedGPUInputsDisabled
	chainedGPUInputsDisabled = true
	defer func() { chainedGPUInputsDisabled = old }()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess, err := NewArchQuantSession(g, arch, 16)
		if err != nil {
			b.Fatalf("NewArchQuantSession: %v", err)
		}
		if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
			b.Fatalf("PrefillTokens: %v", err)
		}
		if _, _, ok, err := sess.stepSampleLogitsTokenInPool(9, params, 0.37, history); err != nil || !ok {
			b.Fatalf("stepSampleLogitsTokenInPool warmup ok=%v err=%v", ok, err)
		}
		b.StartTimer()
		hidden, tok, ok, err := sess.stepSampleLogitsTokenInPool(9, params, 0.37, history)
		if err != nil || !ok {
			b.Fatalf("stepSampleLogitsTokenInPool ok=%v err=%v", ok, err)
		}
		archSessionHiddenBenchSink = hidden
		archSessionSampleTokenBenchSink = tok
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkArchSessionStepSampleTopKTokenICB(b *testing.B) {
	requireNativeRuntime(b)

	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess := newQuantICBStepBenchSession(b, 16)
		if _, _, ok, err := sess.stepSampleTopKTokenInPool(9, params, 0.42, history); err != nil || !ok {
			b.Fatalf("stepSampleTopKTokenInPool warmup ok=%v err=%v", ok, err)
		}
		b.StartTimer()
		hidden, tok, ok, err := sess.stepSampleTopKTokenInPool(9, params, 0.42, history)
		if err != nil || !ok {
			b.Fatalf("stepSampleTopKTokenInPool ok=%v err=%v", ok, err)
		}
		archSessionHiddenBenchSink = hidden
		archSessionSampleTokenBenchSink = tok
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkArchSessionStepSampleTopKTokenICBGPUInputs(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := pleQuantModel(b, 2, 256, 32, 0)
	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess, err := NewArchQuantSession(g, arch, 16)
		if err != nil {
			b.Fatalf("NewArchQuantSession: %v", err)
		}
		if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
			b.Fatalf("PrefillTokens: %v", err)
		}
		if sess.encNextInputsGPU == nil {
			b.Fatal("fixture did not wire GPU next-inputs seam")
		}
		if _, _, ok, err := sess.stepSampleTopKTokenInPool(9, params, 0.42, history); err != nil || !ok {
			b.Fatalf("stepSampleTopKTokenInPool warmup ok=%v err=%v", ok, err)
		}
		b.StartTimer()
		hidden, tok, ok, err := sess.stepSampleTopKTokenInPool(9, params, 0.42, history)
		if err != nil || !ok {
			b.Fatalf("stepSampleTopKTokenInPool ok=%v err=%v", ok, err)
		}
		archSessionHiddenBenchSink = hidden
		archSessionSampleTokenBenchSink = tok
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkArchSessionStepSampleTopKTokenICBHostPLE(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := pleQuantModel(b, 2, 256, 32, 0)
	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	old := chainedGPUInputsDisabled
	chainedGPUInputsDisabled = true
	defer func() { chainedGPUInputsDisabled = old }()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess, err := NewArchQuantSession(g, arch, 16)
		if err != nil {
			b.Fatalf("NewArchQuantSession: %v", err)
		}
		if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
			b.Fatalf("PrefillTokens: %v", err)
		}
		if _, _, ok, err := sess.stepSampleTopKTokenInPool(9, params, 0.42, history); err != nil || !ok {
			b.Fatalf("stepSampleTopKTokenInPool warmup ok=%v err=%v", ok, err)
		}
		b.StartTimer()
		hidden, tok, ok, err := sess.stepSampleTopKTokenInPool(9, params, 0.42, history)
		if err != nil || !ok {
			b.Fatalf("stepSampleTopKTokenInPool ok=%v err=%v", ok, err)
		}
		archSessionHiddenBenchSink = hidden
		archSessionSampleTokenBenchSink = tok
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkArchSessionStepSampleTopKCandidatesICB(b *testing.B) {
	requireNativeRuntime(b)

	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess := newQuantICBStepBenchSession(b, 16)
		if _, _, _, ok, err := sess.stepSampleTopKCandidatesInPool(9, params); err != nil || !ok {
			b.Fatalf("stepSampleTopKCandidatesInPool warmup ok=%v err=%v", ok, err)
		}
		b.StartTimer()
		hidden, logits, ids, ok, err := sess.stepSampleTopKCandidatesInPool(9, params)
		if err != nil || !ok {
			b.Fatalf("stepSampleTopKCandidatesInPool ok=%v err=%v", ok, err)
		}
		archSessionHiddenBenchSink = hidden
		samplePenaltyBenchSink = logits
		sampleHistoryBenchSink = ids
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkArchSessionStepSampleTopKCandidatesICBRepeatPenalty(b *testing.B) {
	requireNativeRuntime(b)

	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess := newQuantICBStepBenchSession(b, 16)
		if _, _, _, ok, err := sess.stepSampleTopKCandidatesWithHistoryInPool(9, params, history); err != nil || !ok {
			b.Fatalf("stepSampleTopKCandidatesWithHistoryInPool warmup ok=%v err=%v", ok, err)
		}
		b.StartTimer()
		hidden, logits, ids, ok, err := sess.stepSampleTopKCandidatesWithHistoryInPool(9, params, history)
		if err != nil || !ok {
			b.Fatalf("stepSampleTopKCandidatesWithHistoryInPool ok=%v err=%v", ok, err)
		}
		archSessionHiddenBenchSink = hidden
		samplePenaltyBenchSink = logits
		sampleHistoryBenchSink = ids
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkArchSessionStepSampleTopKCandidatesICBGPUInputs(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := pleQuantModel(b, 2, 256, 32, 0)
	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess, err := NewArchQuantSession(g, arch, 16)
		if err != nil {
			b.Fatalf("NewArchQuantSession: %v", err)
		}
		if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
			b.Fatalf("PrefillTokens: %v", err)
		}
		if sess.encNextInputsGPU == nil {
			b.Fatal("fixture did not wire GPU next-inputs seam")
		}
		if _, _, _, ok, err := sess.stepSampleTopKCandidatesInPool(9, params); err != nil || !ok {
			b.Fatalf("stepSampleTopKCandidatesInPool warmup ok=%v err=%v", ok, err)
		}
		b.StartTimer()
		hidden, logits, ids, ok, err := sess.stepSampleTopKCandidatesInPool(9, params)
		if err != nil || !ok {
			b.Fatalf("stepSampleTopKCandidatesInPool ok=%v err=%v", ok, err)
		}
		archSessionHiddenBenchSink = hidden
		samplePenaltyBenchSink = logits
		sampleHistoryBenchSink = ids
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkArchSessionStepSampleTopKCandidatesICBHostPLE(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := pleQuantModel(b, 2, 256, 32, 0)
	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}}
	old := chainedGPUInputsDisabled
	chainedGPUInputsDisabled = true
	defer func() { chainedGPUInputsDisabled = old }()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess, err := NewArchQuantSession(g, arch, 16)
		if err != nil {
			b.Fatalf("NewArchQuantSession: %v", err)
		}
		if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
			b.Fatalf("PrefillTokens: %v", err)
		}
		if _, _, _, ok, err := sess.stepSampleTopKCandidatesInPool(9, params); err != nil || !ok {
			b.Fatalf("stepSampleTopKCandidatesInPool warmup ok=%v err=%v", ok, err)
		}
		b.StartTimer()
		hidden, logits, ids, ok, err := sess.stepSampleTopKCandidatesInPool(9, params)
		if err != nil || !ok {
			b.Fatalf("stepSampleTopKCandidatesInPool ok=%v err=%v", ok, err)
		}
		archSessionHiddenBenchSink = hidden
		samplePenaltyBenchSink = logits
		sampleHistoryBenchSink = ids
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkArchSessionSuppressionFresh(b *testing.B) {
	base := []int32{2, 7, 13, 29}
	extra := []int32{7, 11, 13, 17, 19}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		out := nativeAppendSuppressionTokens(base, extra)
		if len(out) != 7 {
			b.Fatal("suppression token length mismatch")
		}
		sampleSuppressBenchSink = out
	}
}

func BenchmarkArchSessionSuppressionScratch(b *testing.B) {
	base := []int32{2, 7, 13, 29}
	extra := []int32{7, 11, 13, 17, 19}
	sess := &ArchSession{}
	if out := sess.suppressionTokensScratch(base, extra); len(out) != 7 {
		b.Fatal("suppression token length mismatch")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := sess.suppressionTokensScratch(base, extra)
		if len(out) != 7 {
			b.Fatal("suppression token length mismatch")
		}
		sampleSuppressBenchSink = out
	}
}

func BenchmarkArchSessionSuppressionScratchBaseEmpty(b *testing.B) {
	extra := []int32{7, 11, 13, 17, 19}
	sess := &ArchSession{}
	if out := sess.suppressionTokensScratch(nil, extra); len(out) != len(extra) {
		b.Fatal("suppression token length mismatch")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := sess.suppressionTokensScratch(nil, extra)
		if len(out) != len(extra) {
			b.Fatal("suppression token length mismatch")
		}
		sampleSuppressBenchSink = out
	}
}

func BenchmarkArchSessionSuppressionScratchExtraCovered(b *testing.B) {
	base := []int32{2, 7, 13, 29}
	extra := []int32{7, 13}
	sess := &ArchSession{}
	if out := sess.suppressionTokensScratch(base, extra); len(out) != len(base) {
		b.Fatal("suppression token length mismatch")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := sess.suppressionTokensScratch(base, extra)
		if len(out) != len(base) {
			b.Fatal("suppression token length mismatch")
		}
		sampleSuppressBenchSink = out
	}
}
