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
