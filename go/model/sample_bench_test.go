// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

// The sampler benches are the contract's alloc baseline (AX-11): pure-Go over a
// synthetic vocab-sized logit buffer, no model load. Greedy is the zero-alloc floor
// (one scan, no buffers); Sample is the alloc-heavy path (two float32 buffers + an
// index order + a stable sort) — the Greedy↔Sample gap IS the baseline a later pass
// optimises against. A realistic vocab (≈256k) so the per-token cost is real.
const benchVocab = 256000

func benchLogits(vocab int) []byte {
	vals := make([]float32, vocab)
	for i := range vals {
		vals[i] = float32((i*131)%4096-2048) * 0.01 // a spread, deterministic
	}
	return bf16Bytes(vals)
}

// BenchmarkGreedy — argmax over the vocab: a single scan, expected ~0 allocs/op. The
// floor the decode loop's deterministic close costs.
func BenchmarkGreedy(b *testing.B) {
	logits := benchLogits(benchVocab)
	b.SetBytes(int64(len(logits)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Greedy(logits, benchVocab); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSample_Temp — temperature softmax with no top-k/top-p cut: the full
// softmax + rank + sort path, the alloc-heavy ceiling of token selection.
func BenchmarkSample_Temp(b *testing.B) {
	logits := benchLogits(benchVocab)
	s := NewSampler(1)
	p := SampleParams{Temperature: 1}
	b.SetBytes(int64(len(logits)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Sample(logits, benchVocab, p); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSample_TopKTopP — both nucleus cuts active. Same buffers + sort as the
// plain temperature path (the cuts narrow the kept set, they don't change the
// allocation shape), so it pins that the cuts add no extra per-token allocation.
func BenchmarkSample_TopKTopP(b *testing.B) {
	logits := benchLogits(benchVocab)
	s := NewSampler(1)
	p := SampleParams{Temperature: 1, TopK: 64, TopP: 0.95}
	b.SetBytes(int64(len(logits)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Sample(logits, benchVocab, p); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSample_ZeroTemp — temperature <= 0 short-circuits to Greedy, so this should
// match BenchmarkGreedy (no softmax buffers): pins that a zero-temp request pays the
// greedy cost, not the sampler's.
func BenchmarkSample_ZeroTemp(b *testing.B) {
	logits := benchLogits(benchVocab)
	s := NewSampler(1)
	p := SampleParams{Temperature: 0}
	b.SetBytes(int64(len(logits)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Sample(logits, benchVocab, p); err != nil {
			b.Fatal(err)
		}
	}
}
