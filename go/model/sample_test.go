// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"math"
	"sort"
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
	for i := range 32 {
		ta, _ := a.Sample(spreadLogits, vocab, SampleParams{Temperature: 1})
		tb, _ := b.Sample(spreadLogits, vocab, SampleParams{Temperature: 1})
		if ta != tb {
			t.Fatalf("same seed diverged at draw %d: %d vs %d", i, ta, tb)
		}
	}

	// the RNG must advance (a single seed gives a VARYING sequence, not one repeated token).
	s := NewSampler(7)
	seen := map[int32]int{}
	for range 64 {
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
	for range 200 {
		if tok, _ := ps.Sample(peakLogits, vocab, SampleParams{Temperature: 1}); tok == 5 {
			hits++
		}
	}
	if hits < 195 {
		t.Fatalf("peaked sampling hit the peak only %d/200 times", hits)
	}

	// top-k = 1 → always the argmax, regardless of temperature.
	ks := NewSampler(9)
	for i := range 32 {
		if tok, _ := ks.Sample(spreadLogits, vocab, SampleParams{Temperature: 2, TopK: 1}); tok != argmax {
			t.Fatalf("top-k=1 draw %d returned %d, want argmax %d", i, tok, argmax)
		}
	}

	// tiny top-p → nucleus is just the top token → argmax.
	pp := NewSampler(11)
	for i := range 32 {
		if tok, _ := pp.Sample(spreadLogits, vocab, SampleParams{Temperature: 1, TopP: 0.001}); tok != argmax {
			t.Fatalf("top-p=0.001 draw %d returned %d, want argmax %d", i, tok, argmax)
		}
	}

	// min-p masks tokens below a fraction of the top token probability, even
	// when no temperature transform is requested.
	minPLogits := bf16Bytes([]float32{-100, 50, -100, -100})
	for i := range 32 {
		if tok, _ := NewSampler(uint64(i+1)).Sample(minPLogits, 4, SampleParams{MinP: 0.1}); tok != 1 {
			t.Fatalf("min-p draw %d returned %d, want dominant token 1", i, tok)
		}
	}

	if _, err := NewSampler(1).Sample(spreadLogits, vocab+1, SampleParams{Temperature: 1}); err == nil {
		t.Fatal("expected a length-mismatch error")
	}
	t.Logf("sample: temp0=greedy; same-seed reproducible; RNG advances (%d distinct/64); peaked→peak 195+/200; top-k=1 and tiny top-p →argmax", len(seen))
}

// TestSelectTopKDescMatchesStableSort locks the bounded top-k select byte-identical to the
// prefix of the full descending stable sort it replaces on the sampler hot path — across a
// deterministic fuzz of vocab/k with a deliberately tiny value range so ties are frequent and
// the ascending-index tie-break is exercised, the corner the sampler's determinism rides on.
func TestSelectTopKDescMatchesStableSort(t *testing.T) {
	rng := uint64(0x9e3779b97f4a7c15) // splitmix64 state — hermetic, no math/rand import
	nextInt := func(n int) int {
		rng += 0x9e3779b97f4a7c15
		z := rng
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb
		z ^= z >> 31
		return int(z % uint64(n))
	}
	for _, vocab := range []int{1, 2, 5, 8, 33, 64, 257, 1000} {
		for trial := 0; trial < 40; trial++ {
			probs := make([]float32, vocab)
			for i := range probs {
				probs[i] = float32(nextInt(5)) // 0..4 → many collisions → tie-break coverage
			}
			ref := make([]int, vocab)
			for i := range ref {
				ref[i] = i
			}
			sort.SliceStable(ref, func(a, b int) bool { return probs[ref[a]] > probs[ref[b]] })
			for _, k := range []int{1, 2, 7, vocab/2 + 1, vocab} {
				if k < 1 || k > vocab {
					continue
				}
				got := make([]int, k)
				selectTopKDesc(got, probs, vocab)
				for i := 0; i < k; i++ {
					if got[i] != ref[i] {
						t.Fatalf("vocab=%d k=%d pos=%d: got %d, want %d (full-sort prefix %v; probs=%v)",
							vocab, k, i, got[i], ref[i], ref[:k], probs)
					}
				}
			}
		}
	}
	t.Logf("selectTopKDesc matches the stable-sort prefix across tie-heavy fuzz")
}

func TestSampleCandidatesMatchesFullTopKWindow(t *testing.T) {
	fullVals := []float32{-4, -3, -2, -1, 0.2, 0.4, 1.7, 2.1}
	full := bf16Bytes(fullVals)
	candidateIDs := []int32{7, 6, 5}
	candidates := bf16Bytes([]float32{fullVals[7], fullVals[6], fullVals[5]})
	fullSampler := NewSampler(123)
	candidateSampler := NewSampler(123)
	params := SampleParams{Temperature: 0.8, TopK: 3}
	for i := range 32 {
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
	for i := range 64 {
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

// TestSample_Greedy_Good covers the ordinary argmax: the highest-logit index wins.
func TestSample_Greedy_Good(t *testing.T) {
	logits := bf16Bytes([]float32{0.1, 0.9, 0.4})
	got, err := Greedy(logits, 3)
	if err != nil {
		t.Fatalf("Greedy: %v", err)
	}
	if got != 1 {
		t.Fatalf("Greedy = %d, want 1 (the max)", got)
	}
}

// TestSample_Greedy_Bad covers a length mismatch: logits that aren't exactly vocab bf16
// bytes is a clean error, never an out-of-range read.
func TestSample_Greedy_Bad(t *testing.T) {
	logits := bf16Bytes([]float32{0.1, 0.9, 0.4})
	if _, err := Greedy(logits, 4); err == nil {
		t.Fatal("Greedy with a vocab/length mismatch: expected an error")
	}
}

// TestSample_Greedy_Ugly covers a tie at the maximum: the LOWEST index wins, a
// deterministic tie-break (never the last-seen or a random pick).
func TestSample_Greedy_Ugly(t *testing.T) {
	logits := bf16Bytes([]float32{0.5, 0.1, 0.5, 0.2})
	got, err := Greedy(logits, 4)
	if err != nil {
		t.Fatalf("Greedy: %v", err)
	}
	if got != 0 {
		t.Fatalf("Greedy(tie) = %d, want the lowest tied index 0", got)
	}
}

// TestSample_NewSampler_Good covers the ordinary construction: NewSampler seeds the
// Sampler's RNG state directly from its argument, and no scratch is pre-allocated (grown
// lazily on first Sample).
func TestSample_NewSampler_Good(t *testing.T) {
	s := NewSampler(42)
	if s.state != 42 {
		t.Fatalf("NewSampler(42).state = %d, want 42", s.state)
	}
	if cap(s.probs) != 0 || cap(s.scaled) != 0 || cap(s.order) != 0 {
		t.Fatalf("NewSampler pre-allocated scratch: probs=%d scaled=%d order=%d", cap(s.probs), cap(s.scaled), cap(s.order))
	}
}

// TestSample_NewSampler_Bad covers the degenerate seed 0: a valid, distinct sampler
// state, not special-cased into some other default.
func TestSample_NewSampler_Bad(t *testing.T) {
	s := NewSampler(0)
	if s.state != 0 {
		t.Fatalf("NewSampler(0).state = %d, want 0 (no hidden re-seeding)", s.state)
	}
}

// TestSample_NewSampler_Ugly covers two DIFFERENT seeds: they must diverge (a sampler's
// identity is its seed), proven by the first Draw() differing.
func TestSample_NewSampler_Ugly(t *testing.T) {
	a, b := NewSampler(1), NewSampler(2)
	if a.Draw() == b.Draw() {
		t.Fatal("two samplers seeded differently drew the same first value — seeds not distinguishing state")
	}
}

// TestSample_Sampler_Draw_Good covers the ordinary reproducible draw: the same seed
// gives the SAME sequence of Draw() calls (no other state perturbs it).
func TestSample_Sampler_Draw_Good(t *testing.T) {
	a, b := NewSampler(7), NewSampler(7)
	for i := range 8 {
		if av, bv := a.Draw(), b.Draw(); av != bv {
			t.Fatalf("draw %d diverged: %v vs %v (same seed)", i, av, bv)
		}
	}
}

// TestSample_Sampler_Draw_Bad covers the value range: every draw must land in [0,1),
// including 0 but never reaching 1.
func TestSample_Sampler_Draw_Bad(t *testing.T) {
	s := NewSampler(3)
	for range 256 {
		if v := s.Draw(); v < 0 || v >= 1 {
			t.Fatalf("Draw() = %v, want a value in [0,1)", v)
		}
	}
}

// TestSample_Sampler_Draw_Ugly covers the RNG-advance guarantee: Draw() is NOT
// idempotent — consecutive calls on the same Sampler must (almost always) differ, or a
// generation loop would draw one repeated value forever.
func TestSample_Sampler_Draw_Ugly(t *testing.T) {
	s := NewSampler(5)
	first := s.Draw()
	seenDifferent := false
	for range 8 {
		if s.Draw() != first {
			seenDifferent = true
			break
		}
	}
	if !seenDifferent {
		t.Fatal("Draw() returned the same value 9 times in a row — RNG state not advancing")
	}
}

// TestSample_Sampler_Sample_Good covers the ordinary stochastic pick: a peaked
// distribution lands on the peak the overwhelming majority of the time.
func TestSample_Sampler_Sample_Good(t *testing.T) {
	const vocab = 8
	peak := make([]float32, vocab)
	peak[3] = 30
	logits := bf16Bytes(peak)
	s := NewSampler(11)
	hits := 0
	for range 100 {
		if tok, err := s.Sample(logits, vocab, SampleParams{Temperature: 1}); err != nil {
			t.Fatalf("Sample: %v", err)
		} else if tok == 3 {
			hits++
		}
	}
	if hits < 95 {
		t.Fatalf("peaked Sample hit the peak only %d/100 times", hits)
	}
}

// TestSample_Sampler_Sample_Bad covers a length mismatch between logits and vocab: a
// clean error rather than an out-of-range read.
func TestSample_Sampler_Sample_Bad(t *testing.T) {
	logits := bf16Bytes([]float32{0.1, 0.2, 0.3})
	if _, err := NewSampler(1).Sample(logits, 4, SampleParams{Temperature: 1}); err == nil {
		t.Fatal("Sample with a vocab/length mismatch: expected an error")
	}
}

// TestSample_Sampler_Sample_Ugly covers every token suppressed: Sample must error rather
// than silently returning a suppressed id or panicking on an empty distribution.
func TestSample_Sampler_Sample_Ugly(t *testing.T) {
	logits := bf16Bytes([]float32{0.1, 0.2, 0.3})
	suppressAll := []int32{0, 1, 2}
	if _, err := NewSampler(1).Sample(logits, 3, SampleParams{Temperature: 1, SuppressTokens: suppressAll}); err == nil {
		t.Fatal("Sample with every token suppressed: expected an error")
	}
	if _, err := NewSampler(1).Sample(logits, 3, SampleParams{Temperature: 0, SuppressTokens: suppressAll}); err == nil {
		t.Fatal("Sample (greedy path) with every token suppressed: expected an error")
	}
}

// TestSample_Sampler_SampleCandidates_Good covers the ordinary candidate-window pick: a
// dominant candidate wins under greedy (temp<=0) selection.
func TestSample_Sampler_SampleCandidates_Good(t *testing.T) {
	candidates := bf16Bytes([]float32{0.1, 5.0, 0.2})
	ids := []int32{10, 11, 12}
	got, err := NewSampler(1).SampleCandidates(candidates, ids, SampleParams{Temperature: 0})
	if err != nil {
		t.Fatalf("SampleCandidates: %v", err)
	}
	if got != 11 {
		t.Fatalf("SampleCandidates = %d, want the dominant candidate id 11", got)
	}
}

// TestSample_Sampler_SampleCandidates_Bad covers the empty-candidates call: a clean
// error, since there is nothing to pick from.
func TestSample_Sampler_SampleCandidates_Bad(t *testing.T) {
	if _, err := NewSampler(1).SampleCandidates(nil, nil, SampleParams{}); err == nil {
		t.Fatal("SampleCandidates with no candidates: expected an error")
	}
}

// TestSample_Sampler_SampleCandidates_Ugly covers a logits/ids length mismatch: a clean
// error rather than indexing past the shorter slice.
func TestSample_Sampler_SampleCandidates_Ugly(t *testing.T) {
	candidates := bf16Bytes([]float32{0.1, 0.2})
	ids := []int32{1, 2, 3} // one more id than logits values
	if _, err := NewSampler(1).SampleCandidates(candidates, ids, SampleParams{}); err == nil {
		t.Fatal("SampleCandidates with logits/ids length mismatch: expected an error")
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

// referenceTopKSample is the full-softmax algorithm Sample's fast path replaces: exp the
// WHOLE vocab, rank, then draw. It shares the production cut+draw tail (drawFromRanked), so
// only the exp-everything vs exp-the-survivors difference is under test. Assumes a top-k cap
// and at least one unsuppressed token (the fuzz below guarantees both).
func referenceTopKSample(s *Sampler, logits []byte, vocab int, p SampleParams) int32 {
	temp := p.Temperature
	if temp <= 0 {
		temp = 1
	}
	probs := make([]float32, vocab)
	maxL := float32(math.Inf(-1))
	for i := 0; i < vocab; i++ {
		if tokenSuppressed(int32(i), p.SuppressTokens) {
			continue
		}
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
		if v > maxL {
			maxL = v
		}
	}
	var sum float32
	for i := 0; i < vocab; i++ {
		if tokenSuppressed(int32(i), p.SuppressTokens) {
			probs[i] = 0
			continue
		}
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
		e := float32(math.Exp(float64(v - maxL)))
		probs[i] = e
		sum += e
	}
	keep := p.TopK
	if keep > vocab {
		keep = vocab
	}
	order := make([]int, keep)
	selectTopKDesc(order, probs, vocab)
	tok, _ := s.drawFromRanked(order, probs, nil, keep, vocab, sum, p)
	return tok
}

// TestSampleTopKFirst_ParityWithFullExp locks Sample's fast top-k path — rank on the raw
// logits, exp only the survivors — byte-identical to the full-softmax reference it replaces,
// across a deterministic fuzz of vocab, temperature, top-k/top-p/min-p and suppression. Case
// 0 is an exp-underflow stress (far fewer real peaks than TopK over a sub-underflow floor) so
// the top-k dips into the 0-mass tail the two paths order differently: the test proves that
// tail can never change the drawn token. A divergence would mean the ≈256k-exp/token saving
// silently alters generations.
func TestSampleTopKFirst_ParityWithFullExp(t *testing.T) {
	rng := uint64(0x243f6a8885a308d3) // splitmix64 — hermetic, no math/rand import
	next := func() uint64 {
		rng += 0x9e3779b97f4a7c15
		z := rng
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb
		z ^= z >> 31
		return z
	}
	nextInt := func(n int) int { return int(next() % uint64(n)) }
	nextF := func() float32 { return float32(next()>>40) / float32(1<<24) }

	for c := 0; c < 4000; c++ {
		vocab := 16 + nextInt(2048)
		vals := make([]float32, vocab)
		switch c % 5 {
		case 0: // exp-underflow floor + a few real peaks
			for i := range vals {
				vals[i] = -300
			}
			for k, n := 0, 1+nextInt(20); k < n; k++ {
				vals[nextInt(vocab)] = nextF()*12 - 2
			}
		case 1: // near-flat tiny range → frequent bf16 ties
			for i := range vals {
				vals[i] = float32(nextInt(4)) * 0.1
			}
		default: // realistic spread
			spread := 1 + nextF()*8
			for i := range vals {
				vals[i] = (nextF()*2 - 1) * spread
			}
		}
		logits := bf16Bytes(vals)

		p := SampleParams{
			Temperature: 0.4 + nextF()*1.6,
			TopK:        2 + nextInt(vocab-2), // 2..vocab-1 → always caps → fast path
		}
		if nextInt(2) == 0 {
			p.TopP = 0.4 + nextF()*0.59
		}
		if nextInt(3) == 0 {
			p.MinP = nextF() * 0.4
		}
		if nextInt(3) == 0 {
			for k, n := 0, nextInt(6); k < n; k++ {
				p.SuppressTokens = append(p.SuppressTokens, int32(nextInt(vocab)))
			}
		}

		seed := next()
		got, gerr := NewSampler(seed).Sample(logits, vocab, p)
		if gerr != nil {
			t.Fatalf("case %d: fast path errored: %v (vocab=%d p=%+v)", c, gerr, vocab, p)
		}
		want := referenceTopKSample(NewSampler(seed), logits, vocab, p)
		if got != want {
			t.Fatalf("case %d: token mismatch got=%d want=%d (vocab=%d p=%+v seed=%d)", c, got, want, vocab, p, seed)
		}
	}
}
