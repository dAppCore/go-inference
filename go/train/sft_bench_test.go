// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the per-sample training-string render on the SFT hot path.
// sftSampleText runs once per sample per epoch (tens of thousands of calls on
// a real corpus), so its allocation shape is multiplied by the whole run.

package train

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/train/dataset"
)

// benchChatSample is a real-shaped multi-turn chat row: a system turn plus
// three user/assistant exchanges — the dominant Messages-branch input.
var benchChatSample = dataset.Sample{
	Messages: []inference.Message{
		{Role: "system", Content: "You are a terse, accurate assistant."},
		{Role: "user", Content: "What is the capital of France?"},
		{Role: "assistant", Content: "Paris."},
		{Role: "user", Content: "And its population, roughly?"},
		{Role: "assistant", Content: "About 2.1 million in the city proper."},
		{Role: "user", Content: "Thanks."},
		{Role: "assistant", Content: "You're welcome."},
	},
}

// BenchmarkSft_SampleText_Messages measures the chat-turn join branch — the
// per-message temp string plus the strings.Builder growth.
func BenchmarkSft_SampleText_Messages(b *testing.B) {
	b.ReportAllocs()
	var sink string
	for i := 0; i < b.N; i++ {
		sink = sftSampleText(benchChatSample)
	}
	_ = sink
}
