// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"math"
	"testing"
)

// bf16Bytes encodes float32s to bf16 bytes (round-to-nearest-even), the dtype the LM
// head emits — for building test logits.
func bf16Bytes(vals []float32) []byte {
	out := make([]byte, len(vals)*bf16Size)
	for i, v := range vals {
		bits := math.Float32bits(v)
		var h uint16
		if bits&0x7fffffff > 0x7f800000 {
			h = uint16(bits>>16) | 0x0040
		} else {
			h = uint16((bits + ((bits>>16)&1 + 0x7fff)) >> 16)
		}
		out[i*bf16Size] = byte(h)
		out[i*bf16Size+1] = byte(h >> 8)
	}
	return out
}

func TestGreedy(t *testing.T) {
	logits := bf16Bytes([]float32{0.1, 0.5, -0.3, 0.5, 0.2}) // max 0.5 first at index 1
	got, err := Greedy(logits, 5)
	if err != nil {
		t.Fatalf("Greedy: %v", err)
	}
	if got != 1 {
		t.Fatalf("Greedy: got %d, want 1 (lowest-index of the tied max)", got)
	}
	if _, err := Greedy(logits, 4); err == nil {
		t.Fatal("expected a length-mismatch error")
	}
	t.Logf("greedy: argmax with lowest-index tie-break, length validated")
}

func TestSample(t *testing.T) {
	const vocab = 12
	// a spread distribution so stochastic draws actually vary.
	spread := make([]float32, vocab)
	for i := range spread {
		spread[i] = float32(i) * 0.4
	}
	spreadLogits := bf16Bytes(spread)
	argmax, _ := Greedy(spreadLogits, vocab) // = vocab-1 (largest)

	// temp <= 0 → greedy (matches Greedy, no RNG perturbation).
	g, err := NewSampler(1).Sample(spreadLogits, vocab, SampleParams{Temperature: 0})
	if err != nil {
		t.Fatalf("Sample temp0: %v", err)
	}
	if g != argmax {
		t.Fatalf("Sample temp0: got %d, want greedy %d", g, argmax)
	}

	// reproducible: two samplers, same seed, identical sequences.
	a, b := NewSampler(42), NewSampler(42)
	for i := 0; i < 32; i++ {
		ta, _ := a.Sample(spreadLogits, vocab, SampleParams{Temperature: 1})
		tb, _ := b.Sample(spreadLogits, vocab, SampleParams{Temperature: 1})
		if ta != tb {
			t.Fatalf("same seed diverged at draw %d: %d vs %d", i, ta, tb)
		}
	}

	// the RNG must advance (a single seed gives a VARYING sequence, not one repeated token).
	s := NewSampler(7)
	seen := map[int32]int{}
	for i := 0; i < 64; i++ {
		tok, _ := s.Sample(spreadLogits, vocab, SampleParams{Temperature: 1.5})
		seen[tok]++
	}
	if len(seen) < 2 {
		t.Fatalf("temperature sampling produced only %d distinct token(s) over 64 draws — RNG not advancing", len(seen))
	}

	// peaked distribution → temperature sampling lands on the peak ~always.
	peak := make([]float32, vocab)
	peak[5] = 30
	peakLogits := bf16Bytes(peak)
	ps := NewSampler(3)
	hits := 0
	for i := 0; i < 200; i++ {
		if tok, _ := ps.Sample(peakLogits, vocab, SampleParams{Temperature: 1}); tok == 5 {
			hits++
		}
	}
	if hits < 195 {
		t.Fatalf("peaked sampling hit the peak only %d/200 times", hits)
	}

	// top-k = 1 → always the argmax, regardless of temperature.
	ks := NewSampler(9)
	for i := 0; i < 32; i++ {
		if tok, _ := ks.Sample(spreadLogits, vocab, SampleParams{Temperature: 2, TopK: 1}); tok != argmax {
			t.Fatalf("top-k=1 draw %d returned %d, want argmax %d", i, tok, argmax)
		}
	}

	// tiny top-p → nucleus is just the top token → argmax.
	pp := NewSampler(11)
	for i := 0; i < 32; i++ {
		if tok, _ := pp.Sample(spreadLogits, vocab, SampleParams{Temperature: 1, TopP: 0.001}); tok != argmax {
			t.Fatalf("top-p=0.001 draw %d returned %d, want argmax %d", i, tok, argmax)
		}
	}

	// min-p masks tokens below a fraction of the top token probability, even
	// when no temperature transform is requested.
	minPLogits := bf16Bytes([]float32{-100, 50, -100, -100})
	for i := 0; i < 32; i++ {
		if tok, _ := NewSampler(uint64(i+1)).Sample(minPLogits, 4, SampleParams{MinP: 0.1}); tok != 1 {
			t.Fatalf("min-p draw %d returned %d, want dominant token 1", i, tok)
		}
	}

	if _, err := NewSampler(1).Sample(spreadLogits, vocab+1, SampleParams{Temperature: 1}); err == nil {
		t.Fatal("expected a length-mismatch error")
	}
	t.Logf("sample: temp0=greedy; same-seed reproducible; RNG advances (%d distinct/64); peaked→peak 195+/200; top-k=1 and tiny top-p →argmax", len(seen))
}

func TestSampleCandidatesMatchesFullTopKWindow(t *testing.T) {
	fullVals := []float32{-4, -3, -2, -1, 0.2, 0.4, 1.7, 2.1}
	full := bf16Bytes(fullVals)
	candidateIDs := []int32{7, 6, 5}
	candidates := bf16Bytes([]float32{fullVals[7], fullVals[6], fullVals[5]})
	fullSampler := NewSampler(123)
	candidateSampler := NewSampler(123)
	params := SampleParams{Temperature: 0.8, TopK: 3}
	for i := 0; i < 32; i++ {
		want, err := fullSampler.Sample(full, len(fullVals), params)
		if err != nil {
			t.Fatalf("full sample %d: %v", i, err)
		}
		got, err := candidateSampler.SampleCandidates(candidates, candidateIDs, params)
		if err != nil {
			t.Fatalf("candidate sample %d: %v", i, err)
		}
		if got != want {
			t.Fatalf("draw %d: candidate sample = %d, want %d", i, got, want)
		}
	}

	if got, err := NewSampler(1).SampleCandidates(candidates, candidateIDs, SampleParams{Temperature: 0, SuppressTokens: []int32{7}}); err != nil || got != 6 {
		t.Fatalf("candidate greedy suppression = %d, %v; want 6, nil", got, err)
	}
}

func TestSampleCandidatesMatchesFullTopKTopPWindow(t *testing.T) {
	fullVals := []float32{0, 0, 0, 0, 0, 0, 0, 0}
	full := bf16Bytes(fullVals)
	candidateIDs := []int32{0, 1, 2}
	candidates := bf16Bytes([]float32{fullVals[0], fullVals[1], fullVals[2]})
	fullSampler := NewSampler(456)
	candidateSampler := NewSampler(456)
	params := SampleParams{Temperature: 1, TopK: 3, TopP: 0.5}
	for i := 0; i < 64; i++ {
		want, err := fullSampler.Sample(full, len(fullVals), params)
		if err != nil {
			t.Fatalf("full sample %d: %v", i, err)
		}
		got, err := candidateSampler.SampleCandidates(candidates, candidateIDs, params)
		if err != nil {
			t.Fatalf("candidate sample %d: %v", i, err)
		}
		if got != want {
			t.Fatalf("draw %d: candidate sample = %d, want %d", i, got, want)
		}
		if got == 2 {
			t.Fatalf("draw %d sampled token 2; TopP must be applied after TopK over the renormalised candidate window", i)
		}
	}
}

func TestSampleKeepsUnnormalisedWeightScratch(t *testing.T) {
	const vocab = 72
	vals := make([]float32, vocab)
	for i := range vals {
		vals[i] = float32(i%9) * 0.125
	}
	logits := bf16Bytes(vals)
	params := SampleParams{Temperature: 1, TopP: 0.7}

	a := NewSampler(7)
	b := NewSampler(7)
	got, err := a.Sample(logits, vocab, params)
	if err != nil {
		t.Fatalf("Sample: %v", err)
	}
	want, err := b.Sample(logits, vocab, params)
	if err != nil {
		t.Fatalf("reference Sample: %v", err)
	}
	if got != want {
		t.Fatalf("sample = %d, want reproducible token %d", got, want)
	}
	var total float32
	for _, p := range a.probs {
		total += p
	}
	if total <= 10 {
		t.Fatalf("sample probability scratch total = %g, want unnormalised exp weight mass", total)
	}
	if cap(a.scaled) != 0 {
		t.Fatalf("sample scaled scratch cap = %d, want 0", cap(a.scaled))
	}
}

func TestSampleTempOnlyAvoidsRankScratch(t *testing.T) {
	const vocab = 72
	vals := make([]float32, vocab)
	for i := range vals {
		vals[i] = float32((i*17)%31) * 0.05
	}
	logits := bf16Bytes(vals)
	params := SampleParams{Temperature: 1}

	a := NewSampler(13)
	b := NewSampler(13)
	got, err := a.Sample(logits, vocab, params)
	if err != nil {
		t.Fatalf("Sample: %v", err)
	}
	want, err := b.Sample(logits, vocab, params)
	if err != nil {
		t.Fatalf("reference Sample: %v", err)
	}
	if got != want {
		t.Fatalf("sample = %d, want reproducible token %d", got, want)
	}
	if cap(a.scaled) != 0 {
		t.Fatalf("temp-only scaled scratch cap = %d, want 0", cap(a.scaled))
	}
	if cap(a.order) != 0 {
		t.Fatalf("temp-only rank scratch cap = %d, want 0", cap(a.order))
	}
	if cap(a.probs) < vocab {
		t.Fatalf("temp-only probability scratch cap = %d, want at least %d", cap(a.probs), vocab)
	}
}

func TestSampleTopKOneAvoidsScratchAndAdvancesRNG(t *testing.T) {
	logits := bf16Bytes([]float32{0.1, 2.0, -1, 1.5})
	s := NewSampler(9)
	before := s.state
	got, err := s.Sample(logits, 4, SampleParams{Temperature: 2, TopK: 1})
	if err != nil {
		t.Fatalf("Sample TopK=1: %v", err)
	}
	if got != 1 {
		t.Fatalf("Sample TopK=1 = %d, want argmax token 1", got)
	}
	if s.state == before {
		t.Fatal("Sample TopK=1 must advance the RNG to preserve stochastic sampler state")
	}
	if cap(s.scaled) != 0 || cap(s.probs) != 0 || cap(s.order) != 0 {
		t.Fatalf("Sample TopK=1 grew sampler scratch: scaled=%d probs=%d order=%d", cap(s.scaled), cap(s.probs), cap(s.order))
	}
}

func TestSampleCandidatesTopKOneAvoidsScratchAndAdvancesRNG(t *testing.T) {
	candidates := bf16Bytes([]float32{3.0, 2.0, 1.0})
	ids := []int32{7, 8, 9}
	s := NewSampler(11)
	before := s.state
	got, err := s.SampleCandidates(candidates, ids, SampleParams{Temperature: 1, TopK: 1, SuppressTokens: []int32{7}})
	if err != nil {
		t.Fatalf("SampleCandidates TopK=1: %v", err)
	}
	if got != 8 {
		t.Fatalf("SampleCandidates TopK=1 = %d, want highest unsuppressed token 8", got)
	}
	if s.state == before {
		t.Fatal("SampleCandidates TopK=1 must advance the RNG to preserve stochastic sampler state")
	}
	if cap(s.scaled) != 0 || cap(s.probs) != 0 || cap(s.order) != 0 {
		t.Fatalf("SampleCandidates TopK=1 grew sampler scratch: scaled=%d probs=%d order=%d", cap(s.scaled), cap(s.probs), cap(s.order))
	}
}

var sampleBenchSink int32

func BenchmarkSamplerTopKOneCold(b *testing.B) {
	logits := bf16Bytes([]float32{-3, -1, 0.25, 4, 2, 1})
	params := SampleParams{Temperature: 1, TopK: 1}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		s := NewSampler(uint64(i + 1))
		tok, err := s.Sample(logits, 6, params)
		if err != nil {
			b.Fatalf("Sample TopK=1: %v", err)
		}
		sampleBenchSink = tok
	}
}

func BenchmarkSamplerCandidatesTopKOneCold(b *testing.B) {
	candidates := bf16Bytes([]float32{0.25, 4, 2, 1})
	ids := []int32{2, 3, 4, 5}
	params := SampleParams{Temperature: 1, TopK: 1}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		s := NewSampler(uint64(i + 1))
		tok, err := s.SampleCandidates(candidates, ids, params)
		if err != nil {
			b.Fatalf("SampleCandidates TopK=1: %v", err)
		}
		sampleBenchSink = tok
	}
}
