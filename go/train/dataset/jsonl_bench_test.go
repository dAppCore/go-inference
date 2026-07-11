// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"strings"
	"testing"

	"dappco.re/go/inference"
)

// mixedCorpus is a realistic training JSONL blend across the six shapes LoadJSONL
// auto-detects (bare text, OpenAI messages, ShareGPT conversations, prompt/response,
// Alpaca instruction/input/output, problem/solution reasoning), repeated to a
// corpus-sized input so the parse-loop steady state (decoder reuse, samples presize,
// msgBuf reuse) is what the bench measures rather than one-row cold cost.
func mixedCorpus(rows int) string {
	shapes := []string{
		`{"text":"The sovereign network protects consciousness across every substrate it touches."}`,
		`{"messages":[{"role":"system","content":"You are a careful assistant."},{"role":"user","content":"Explain gated delta attention."},{"role":"assistant","content":"It decays the recurrent state then writes a rank-1 delta."}]}`,
		`{"conversations":[{"from":"human","value":"What is a KV cache?"},{"from":"gpt","value":"A running store of past keys and values for causal attention."}]}`,
		`{"prompt":"Summarise the mamba-2 block.","response":"Projection, causal conv, SiLU, SSD scan, gated RMSNorm, out-projection."}`,
		`{"instruction":"Convert to uppercase.","input":"hello world","output":"HELLO WORLD"}`,
		`{"problem":"Sum the first ten integers.","thinking":"Pair them: 1+10, 2+9 ... five pairs of 11.","solution":"55"}`,
	}
	var b strings.Builder
	for i := 0; i < rows; i++ {
		b.WriteString(shapes[i%len(shapes)])
		b.WriteByte('\n')
	}
	return b.String()
}

// BenchmarkJSONL_LoadJSONL_MixedCorpus measures the whole parse+normalise pipeline
// over a realistic multi-shape corpus — the streaming decoder, per-row field reset,
// shape detection, and sample assembly.
func BenchmarkJSONL_LoadJSONL_MixedCorpus(b *testing.B) {
	corpus := mixedCorpus(120)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ds, err := LoadJSONL(strings.NewReader(corpus))
		if err != nil {
			b.Fatal(err)
		}
		if len(ds.samples) == 0 {
			b.Fatal("no samples parsed")
		}
	}
}

// BenchmarkJSONL_Next_ReplaySample measures the per-sample replay path — Next
// deep-clones each stored sample (Labels map + Messages slice) every epoch.
func BenchmarkJSONL_Next_ReplaySample(b *testing.B) {
	ds, err := LoadJSONL(strings.NewReader(mixedCorpus(120)))
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok, _ := ds.Next(); !ok {
			_ = ds.Reset()
		}
	}
}

// BenchmarkJSONL_MessagesToSample_ChatTurns measures the chat-normalisation path:
// find the trailing assistant turn, join the preceding turns, clone the message list.
func BenchmarkJSONL_MessagesToSample_ChatTurns(b *testing.B) {
	msgs := []inference.Message{
		{Role: "system", Content: "You are a careful assistant."},
		{Role: "user", Content: "Explain gated delta attention in one sentence."},
		{Role: "assistant", Content: "It decays the recurrent state then writes a rank-1 delta keyed by the normalised key."},
		{Role: "user", Content: "And the read-out?"},
		{Role: "assistant", Content: "The scaled query reads the post-write state."},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok, err := MessagesToSample(msgs, "openai_messages"); err != nil || !ok {
			b.Fatal("MessagesToSample failed")
		}
	}
}
