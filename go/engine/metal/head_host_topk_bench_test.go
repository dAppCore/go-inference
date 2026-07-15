// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math/rand"
	"testing"

	"dappco.re/go/inference/model"
)

// BenchmarkHostTopKCandidatesBF16 is the selector's cost at real-vocab scale
// (262k, k=40) — the number that beat the GPU selection kernels (17-33ms) and
// justified the host lane. One pass, zero allocs.
func BenchmarkHostTopKCandidatesBF16(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	const vocab, k = 262144, 40
	f := make([]float32, vocab)
	for i := range f {
		f[i] = float32(rng.NormFloat64() * 3)
	}
	logits := bf16FromF32s(f)
	var vals [headSampleTopKMaxK * bf16Size]byte
	var ids [headSampleTopKMaxK]int32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hostTopKCandidatesBF16(logits, vocab, k, nil, vals[:], ids[:])
	}
}

// BenchmarkSampleHostTopKBF16 is the whole host lane per token: select + the
// shared candidate sampler.
func BenchmarkSampleHostTopKBF16(b *testing.B) {
	rng := rand.New(rand.NewSource(4))
	const vocab = 262144
	f := make([]float32, vocab)
	for i := range f {
		f[i] = float32(rng.NormFloat64() * 3)
	}
	logits := bf16FromF32s(f)
	sampler := model.NewSampler(1)
	params := model.SampleParams{Temperature: 0.8, TopK: 40, TopP: 0.95}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sampleHostTopKBF16(logits, vocab, sampler, params); err != nil {
			b.Fatal(err)
		}
	}
}
