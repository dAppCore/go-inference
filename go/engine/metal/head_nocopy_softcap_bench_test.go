// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

func BenchmarkHeadEncoderSoftcapKernelRoute(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab = 64, 2048
	const eps, softCap = float32(1e-6), float32(30)
	h := &headEncoder{
		finalNorm: copyView(toBF16Bytes(syntheticFloat32(dModel, 5))),
		weight:    copyView(toBF16Bytes(syntheticFloat32(vocab*dModel, 7))),
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
		softCap:   softCap,
	}
	h.initSoftcapBuffers()
	hidden := toBF16Bytes(syntheticFloat32(dModel, 3))
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := h.encode(hidden, false); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHeadEncoderSoftcapInitBuffers(b *testing.B) {
	requireNativeRuntime(b)

	warm := &headEncoder{vocab: 8192, softCap: 30}
	warm.initSoftcapBuffers()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h := &headEncoder{vocab: 8192, softCap: 30}
		h.initSoftcapBuffers()
		if h.invSoftCapScale.buf == nil || h.softCapScale.buf == nil {
			b.Fatal("softcap scalar buffers missing")
		}
	}
}

func quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits int) (*headEncoder, []byte) {
	packed := make([]byte, vocab*dModel*bits/8)
	for i := range packed {
		packed[i] = byte((i*31 + 19) & 0xff)
	}
	sidecars := vocab * (dModel / groupSize)
	scalesF, biasesF := make([]float32, sidecars), make([]float32, sidecars)
	for i := range scalesF {
		scalesF[i] = 0.01 + float32((i%13)+1)*0.0015
		biasesF[i] = -0.05 + float32(i%17)*0.004
	}
	return &headEncoder{
		finalNorm: copyView(toBF16Bytes(syntheticFloat32(dModel, 41))),
		weight:    copyView(packed),
		scales:    copyView(toBF16Bytes(scalesF)),
		biases:    copyView(toBF16Bytes(biasesF)),
		quant:     true,
		groupSize: groupSize,
		bits:      bits,
		dModel:    dModel,
		vocab:     vocab,
		eps:       1e-6,
	}, toBF16Bytes(syntheticFloat32(dModel, 43))
}

func bf16HeadEncoderBenchFixture(dModel, vocab int) (*headEncoder, []byte) {
	return &headEncoder{
		finalNorm: copyView(toBF16Bytes(syntheticFloat32(dModel, 61))),
		weight:    copyView(toBF16Bytes(syntheticFloat32(vocab*dModel, 67))),
		dModel:    dModel,
		vocab:     vocab,
		eps:       1e-6,
	}, toBF16Bytes(syntheticFloat32(dModel, 71))
}

func BenchmarkHeadEncoderBF16FullLogitsGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab = 512, 4096
	h, hidden := bf16HeadEncoderBenchFixture(dModel, vocab)
	if logits, err := h.encode(hidden, true); err != nil {
		b.Fatal(err)
	} else if _, err := model.Greedy(logits, vocab); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits, err := h.encode(hidden, true)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := model.Greedy(logits, vocab); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHeadEncoderBF16FullLogitsCallerOutputGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab = 512, 4096
	h, hidden := bf16HeadEncoderBenchFixture(dModel, vocab)
	out := make([]byte, vocab*bf16Size)
	if logits, err := h.encodeInto(hidden, true, out); err != nil {
		b.Fatal(err)
	} else if _, err := model.Greedy(logits, vocab); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits, err := h.encodeInto(hidden, true, out)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := model.Greedy(logits, vocab); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHeadEncoderBF16DirectGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab = 512, 4096
	h, hidden := bf16HeadEncoderBenchFixture(dModel, vocab)
	if _, ok, err := h.greedy(hidden, nil); err != nil {
		b.Fatal(err)
	} else if !ok {
		b.Fatal("direct greedy declined BF16 head")
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok, err := h.greedy(hidden, nil); err != nil {
			b.Fatal(err)
		} else if !ok {
			b.Fatal("direct greedy declined BF16 head")
		}
	}
}

func BenchmarkHeadEncoderBF16FullLogitsSuppressedGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab = 512, 4096
	h, hidden := bf16HeadEncoderBenchFixture(dModel, vocab)
	logits, err := h.encode(hidden, true)
	if err != nil {
		b.Fatal(err)
	}
	top, err := model.Greedy(logits, vocab)
	if err != nil {
		b.Fatal(err)
	}
	suppress := []int32{top}
	if _, err := greedyBF16Suppressed(logits, vocab, suppress); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits, err := h.encode(hidden, true)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := greedyBF16Suppressed(logits, vocab, suppress); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHeadEncoderBF16DirectSuppressedGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab = 512, 4096
	h, hidden := bf16HeadEncoderBenchFixture(dModel, vocab)
	logits, err := h.encode(hidden, true)
	if err != nil {
		b.Fatal(err)
	}
	top, err := model.Greedy(logits, vocab)
	if err != nil {
		b.Fatal(err)
	}
	suppress := []int32{top}
	if _, ok, err := h.greedy(hidden, suppress); err != nil {
		b.Fatal(err)
	} else if !ok {
		b.Fatal("direct greedy declined BF16 head")
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok, err := h.greedy(hidden, suppress); err != nil {
			b.Fatal(err)
		} else if !ok {
			b.Fatal("direct greedy declined BF16 head")
		}
	}
}

func BenchmarkHeadEncoderQuantFullLogitsGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab, groupSize, bits = 512, 4096, 64, 4
	h, hidden := quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits)
	if logits, err := h.encode(hidden, true); err != nil {
		b.Fatal(err)
	} else if _, err := model.Greedy(logits, vocab); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits, err := h.encode(hidden, true)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := model.Greedy(logits, vocab); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHeadEncoderQuantFullLogitsCallerOutputGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab, groupSize, bits = 512, 4096, 64, 4
	h, hidden := quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits)
	out := make([]byte, vocab*bf16Size)
	if logits, err := h.encodeInto(hidden, true, out); err != nil {
		b.Fatal(err)
	} else if _, err := model.Greedy(logits, vocab); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits, err := h.encodeInto(hidden, true, out)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := model.Greedy(logits, vocab); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHeadEncoderQuantDirectGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab, groupSize, bits = 512, 4096, 64, 4
	h, hidden := quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits)
	if _, ok, err := h.greedy(hidden, nil); err != nil {
		b.Fatal(err)
	} else if !ok {
		b.Fatal("direct greedy declined quant head")
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok, err := h.greedy(hidden, nil); err != nil {
			b.Fatal(err)
		} else if !ok {
			b.Fatal("direct greedy declined quant head")
		}
	}
}

func BenchmarkHeadEncoderQuantDirectGreedyInPoolSynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab, groupSize, bits = 512, 4096, 64, 4
	h, hidden := quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits)
	withAutoreleasePool(func() {
		if _, ok, err := h.greedyInPool(hidden, nil); err != nil {
			b.Fatal(err)
		} else if !ok {
			b.Fatal("direct greedy in pool declined quant head")
		}
	})
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	withAutoreleasePool(func() {
		for i := 0; i < b.N; i++ {
			if _, ok, err := h.greedyInPool(hidden, nil); err != nil {
				b.Fatal(err)
			} else if !ok {
				b.Fatal("direct greedy in pool declined quant head")
			}
		}
	})
}

func BenchmarkHeadEncoderQuantFullLogitsSuppressedGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab, groupSize, bits = 512, 4096, 64, 4
	h, hidden := quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits)
	logits, err := h.encode(hidden, true)
	if err != nil {
		b.Fatal(err)
	}
	top, err := model.Greedy(logits, vocab)
	if err != nil {
		b.Fatal(err)
	}
	suppress := []int32{top}
	if _, err := greedyBF16Suppressed(logits, vocab, suppress); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits, err := h.encode(hidden, true)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := greedyBF16Suppressed(logits, vocab, suppress); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHeadEncoderQuantDirectSuppressedGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab, groupSize, bits = 512, 4096, 64, 4
	h, hidden := quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits)
	logits, err := h.encode(hidden, true)
	if err != nil {
		b.Fatal(err)
	}
	top, err := model.Greedy(logits, vocab)
	if err != nil {
		b.Fatal(err)
	}
	suppress := []int32{top}
	if _, ok, err := h.greedy(hidden, suppress); err != nil {
		b.Fatal(err)
	} else if !ok {
		b.Fatal("direct greedy declined quant head")
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok, err := h.greedy(hidden, suppress); err != nil {
			b.Fatal(err)
		} else if !ok {
			b.Fatal("direct greedy declined quant head")
		}
	}
}

func BenchmarkHeadEncoderQuantFullLogitsSampledTopKSynthetic(b *testing.B) {
	const dModel, vocab, groupSize, bits = 512, 4096, 64, 4
	const topK = 32
	benchmarkHeadEncoderQuantSampledTopK(b, dModel, vocab, groupSize, bits, topK, false)
}

func BenchmarkHeadEncoderQuantFusedSampledTopKSynthetic(b *testing.B) {
	const dModel, vocab, groupSize, bits = 512, 4096, 64, 4
	const topK = 32
	benchmarkHeadEncoderQuantSampledTopK(b, dModel, vocab, groupSize, bits, topK, true)
}

func BenchmarkHeadEncoderQuantFullLogitsSampledTopKLargeSynthetic(b *testing.B) {
	const dModel, vocab, groupSize, bits = 2048, 32768, 64, 4
	const topK = 32
	benchmarkHeadEncoderQuantSampledTopK(b, dModel, vocab, groupSize, bits, topK, false)
}

func BenchmarkHeadEncoderQuantFusedSampledTopKLargeSynthetic(b *testing.B) {
	const dModel, vocab, groupSize, bits = 2048, 32768, 64, 4
	const topK = 32
	benchmarkHeadEncoderQuantSampledTopK(b, dModel, vocab, groupSize, bits, topK, true)
}

func BenchmarkHeadEncoderQuantDirectSampledTopKTokenSynthetic(b *testing.B) {
	const dModel, vocab, groupSize, bits = 512, 4096, 64, 4
	const topK = 32
	benchmarkHeadEncoderQuantSampledTopKToken(b, dModel, vocab, groupSize, bits, topK, false)
}

func BenchmarkHeadEncoderQuantDirectInPoolSampledTopKTokenSynthetic(b *testing.B) {
	const dModel, vocab, groupSize, bits = 512, 4096, 64, 4
	const topK = 32
	benchmarkHeadEncoderQuantSampledTopKToken(b, dModel, vocab, groupSize, bits, topK, true)
}

func BenchmarkHeadEncoderQuantFullLogitsSampledTopPOnlySmallVocabSynthetic(b *testing.B) {
	const dModel, vocab, groupSize, bits = 512, 64, 64, 4
	benchmarkHeadEncoderQuantSampledTopPOnlySmallVocab(b, dModel, vocab, groupSize, bits, false)
}

func BenchmarkHeadEncoderQuantDirectInPoolSampledTopPOnlySmallVocabSynthetic(b *testing.B) {
	const dModel, vocab, groupSize, bits = 512, 64, 64, 4
	benchmarkHeadEncoderQuantSampledTopPOnlySmallVocab(b, dModel, vocab, groupSize, bits, true)
}

func benchmarkHeadEncoderQuantSampledTopK(b *testing.B, dModel, vocab, groupSize, bits, topK int, fused bool) {
	requireNativeRuntime(b)

	if fused && !q4LMHeadTopKUsable(dModel, vocab, groupSize, bits, topK) {
		b.Skip("fused q4 lm-head top-k custom kernel unavailable")
	}
	h, hidden := quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits)
	h.softCap = 2
	h.initSoftcapBuffers()
	params := model.SampleParams{Temperature: 1, TopK: topK}
	sampler := model.NewSampler(1)
	if fused {
		if logits, ids, ok, err := h.sampleTopKCandidatesFusedQ4(hidden, topK, nil); err != nil {
			b.Fatal(err)
		} else if !ok {
			b.Fatal("sampleTopKCandidates declined fused q4 top-k shape")
		} else if _, err := sampler.SampleCandidates(logits, ids, params); err != nil {
			b.Fatal(err)
		}
	} else {
		if logits, err := h.encode(hidden, false); err != nil {
			b.Fatal(err)
		} else if _, err := sampler.Sample(logits, vocab, params); err != nil {
			b.Fatal(err)
		}
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if fused {
			logits, ids, ok, err := h.sampleTopKCandidatesFusedQ4(hidden, topK, nil)
			if err != nil {
				b.Fatal(err)
			}
			if !ok {
				b.Fatal("sampleTopKCandidates declined fused q4 top-k shape")
			}
			if _, err := sampler.SampleCandidates(logits, ids, params); err != nil {
				b.Fatal(err)
			}
			continue
		}
		logits, err := h.encode(hidden, false)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := sampler.Sample(logits, vocab, params); err != nil {
			b.Fatal(err)
		}
	}
}

func benchmarkHeadEncoderQuantSampledTopKToken(b *testing.B, dModel, vocab, groupSize, bits, topK int, inPool bool) {
	requireNativeRuntime(b)

	h, hidden := quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits)
	params := model.SampleParams{Temperature: 1, TopK: topK}
	sampler := model.NewSampler(1)
	if inPool {
		withAutoreleasePool(func() {
			if _, ok, err := h.sampleTopKTokenInPool(hidden, params, sampler.Draw(), nil); err != nil {
				b.Fatal(err)
			} else if !ok {
				b.Fatal("direct TopK token sampler declined")
			}
			b.SetBytes(int64(vocab * bf16Size))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, ok, err := h.sampleTopKTokenInPool(hidden, params, sampler.Draw(), nil); err != nil {
					b.Fatal(err)
				} else if !ok {
					b.Fatal("direct TopK token sampler declined")
				}
			}
		})
		return
	}
	if _, ok, err := h.sampleTopKToken(hidden, params, sampler.Draw(), nil); err != nil {
		b.Fatal(err)
	} else if !ok {
		b.Fatal("direct TopK token sampler declined")
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok, err := h.sampleTopKToken(hidden, params, sampler.Draw(), nil); err != nil {
			b.Fatal(err)
		} else if !ok {
			b.Fatal("direct TopK token sampler declined")
		}
	}
}

func benchmarkHeadEncoderQuantSampledTopPOnlySmallVocab(b *testing.B, dModel, vocab, groupSize, bits int, direct bool) {
	requireNativeRuntime(b)

	h, hidden := quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits)
	params := model.SampleParams{Temperature: 1, TopP: 0.72}
	sampler := model.NewSampler(1)
	if direct {
		withAutoreleasePool(func() {
			if _, ok, err := h.sampleLogitsTokenInPool(hidden, params, sampler.Draw(), nil); err != nil {
				b.Fatal(err)
			} else if !ok {
				b.Fatal("direct TopP-only sampler declined exact small-vocab shape")
			}
			b.SetBytes(int64(vocab * bf16Size))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, ok, err := h.sampleLogitsTokenInPool(hidden, params, sampler.Draw(), nil); err != nil {
					b.Fatal(err)
				} else if !ok {
					b.Fatal("direct TopP-only sampler declined exact small-vocab shape")
				}
			}
		})
		return
	}
	if logits, err := h.encode(hidden, false); err != nil {
		b.Fatal(err)
	} else if _, err := sampler.Sample(logits, vocab, params); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits, err := h.encode(hidden, false)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := sampler.Sample(logits, vocab, params); err != nil {
			b.Fatal(err)
		}
	}
}
