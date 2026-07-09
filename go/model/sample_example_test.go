// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// ExampleGreedy shows the deterministic token pick: argmax over vocab bf16 logits,
// ties resolved to the lowest index. Greedy is the natural closer for a decode loop in
// a correctness gate or a bench — no RNG, reproducible.
func ExampleGreedy() {
	logits := bf16Bytes([]float32{0.1, 0.9, 0.4, 0.9}) // peak (0.9) first at index 1
	id, err := Greedy(logits, 4)
	if err != nil {
		return
	}
	core.Println(id) // the lowest index of the tied maximum
	// Output: 1
}

// ExampleSampler_Sample shows stochastic sampling: a temperature-scaled softmax with
// optional top-k then top-p, drawn from a reproducible RNG that advances per call. A
// single seed yields a VARYING sequence (the RNG advances), so a generation loop gets
// variety from one seed rather than re-seeding per token.
func ExampleSampler_Sample() {
	logits := bf16Bytes([]float32{0.2, 1.5, 0.7, 0.1})
	s := NewSampler(42)
	p := SampleParams{Temperature: 0.8, TopK: 3, TopP: 0.95}
	id, err := s.Sample(logits, 4, p)
	if err != nil {
		return
	}
	core.Println(id >= 0) // a valid token id drawn from the kept nucleus
	// Output: true
}

// ExampleNewSampler shows construction: a Sampler seeded for reproducible draws — the
// SAME seed always yields the SAME sequence of Sample/Draw calls.
func ExampleNewSampler() {
	a := NewSampler(42)
	b := NewSampler(42)
	core.Println(a.Draw() == b.Draw()) // same seed, same first draw
	// Output: true
}

// ExampleSampler_Draw shows the raw uniform-in-[0,1) stream a backend can consume
// directly (e.g. to keep a sampling reduction on-device) — the same RNG stream Sample
// itself draws from.
func ExampleSampler_Draw() {
	s := NewSampler(1)
	v := s.Draw()
	core.Println(v >= 0 && v < 1)
	// Output: true
}

// ExampleSampler_SampleCandidates shows sampling over a PRESELECTED candidate set: the
// returned token is one of ids, letting a backend keep vocab-wide ranking on-device and
// read back only the candidate window.
func ExampleSampler_SampleCandidates() {
	candidates := bf16Bytes([]float32{0.1, 5.0, 0.2})
	ids := []int32{10, 11, 12}
	got, err := NewSampler(1).SampleCandidates(candidates, ids, SampleParams{Temperature: 0})
	if err != nil {
		return
	}
	core.Println(got) // greedy: the dominant candidate's id
	// Output: 11
}
