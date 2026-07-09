// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// ExampleGenerate shows the greedy token-in/token-out loop: Generate drives a TokenModel
// from a prompt to maxNew new tokens, re-embedding each generated token for the next
// decode step. counterModel is a deterministic fake whose next token is always (id+1) mod
// vocab, so the output is a clean count from the prompt's last id.
func ExampleGenerate() {
	m := counterModel{vocab: 16, dModel: 4}
	got, err := Generate(m, []int32{0}, 3, -1)
	if err != nil {
		return
	}
	core.Println(got)
	// Output: [1 2 3]
}

// ExampleGenerateSampled shows stochastic generation: the same loop as Generate, but
// each token is drawn via the Sampler + SampleParams. Temperature<=0 falls back to
// greedy per token, so a zero-temp request reproduces Generate's sequence exactly.
func ExampleGenerateSampled() {
	m := counterModel{vocab: 16, dModel: 4}
	got, err := GenerateSampled(m, NewSampler(1), SampleParams{Temperature: 0}, []int32{0}, 3, -1)
	if err != nil {
		return
	}
	core.Println(got)
	// Output: [1 2 3]
}

// ExampleGenerateSampledWithStopTokens shows the full stop-token-set variant: generation
// ends the instant any declared stop token is produced, matching serving engines that
// accept more than one stop id.
func ExampleGenerateSampledWithStopTokens() {
	m := counterModel{vocab: 16, dModel: 4}
	got, err := GenerateSampledWithStopTokens(m, NewSampler(1), SampleParams{Temperature: 0}, []int32{0}, 10, []int32{2})
	if err != nil {
		return
	}
	core.Println(got)
	// Output: [1 2]
}

// ExampleGenerateSampledWithStopTokensTransform shows the committed-token transform: the
// TRANSFORMED id is what gets returned and fed into the next decode step, applied before
// the stop-token check.
func ExampleGenerateSampledWithStopTokensTransform() {
	m := counterModel{vocab: 16, dModel: 4}
	got, err := GenerateSampledWithStopTokensTransform(m, NewSampler(1), SampleParams{Temperature: 0}, []int32{0}, 2, nil, func(id int32) int32 {
		if id == 1 {
			return 5
		}
		return id
	})
	if err != nil {
		return
	}
	core.Println(got)
	// Output: [5 6]
}

// ExampleGenerateSampledWithStopTokensTransformEach shows the streaming sibling: each
// committed token is handed to yield as it is sampled, so a serving loop can emit
// incrementally instead of waiting for the whole batch — byte-identical to the batch
// path when yield always returns true.
func ExampleGenerateSampledWithStopTokensTransformEach() {
	m := counterModel{vocab: 16, dModel: 4}
	var streamed []int32
	got, err := GenerateSampledWithStopTokensTransformEach(m, NewSampler(1), SampleParams{Temperature: 0}, []int32{0}, 3, nil, nil, func(id int32) bool {
		streamed = append(streamed, id)
		return true
	})
	if err != nil {
		return
	}
	core.Println(len(got) == len(streamed))
	core.Println(got)
	// Output:
	// true
	// [1 2 3]
}
