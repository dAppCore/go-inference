// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the streaming ThinkingExtractor. Per AX-11 — Process fires
// PER GENERATED TOKEN on every chat/response stream, splitting model-internal
// reasoning from assistant content. Its per-token allocation profile is paid
// on every token of every answer, and the cumulative Content/Thinking totals
// are folded across the whole stream — so a full-response Process→Flush loop
// is the honest unit, not a single token in isolation.
//
// Run:    go test -bench=ThinkingExtractor -benchmem -run='^$' ./serving/provider/openai/
package openai

import (
	"strings"
	"testing"

	"dappco.re/go/inference"
)

// Sinks defeat compiler DCE.
var (
	thinkBenchSinkContent string
	thinkBenchSinkThought string
)

// benchPlainTokens is a realistic ~200-token plain-text answer (no reasoning
// markers) — the dominant stream shape, and the case that folds straight into
// the cumulative content total on every token.
func benchPlainTokens(n int) []inference.Token {
	words := strings.Fields("the quick brown fox jumps over the lazy dog while " +
		"the honest answer depends on the batch shape you are actually serving")
	toks := make([]inference.Token, n)
	for i := range toks {
		toks[i] = inference.Token{Text: words[i%len(words)] + " "}
	}
	return toks
}

// benchThoughtTokens interleaves a gemma channel-reasoning span before the
// visible answer, exercising the in-channel drain branch and the cumulative
// thinking total.
func benchThoughtTokens() []inference.Token {
	stream := "<|channel>analysis let me weigh the options carefully step by step " +
		"<channel|>The answer is that it depends on your workload. " + strings.Repeat("More detail. ", 20)
	fields := strings.SplitAfter(stream, " ")
	toks := make([]inference.Token, len(fields))
	for i, f := range fields {
		toks[i] = inference.Token{Text: f}
	}
	return toks
}

// --- Process (per-token) over a full plain-text answer -------------------

// The whole 200-token answer is streamed token-by-token through one extractor
// then flushed — this is what a full non-reasoning response costs, and where
// the cumulative content total's growth shows up.
func BenchmarkThinkingExtractor_Process_Plain(b *testing.B) {
	toks := benchPlainTokens(200)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e := NewThinkingExtractor()
		var c string
		for _, tok := range toks {
			cd, _ := e.Process(tok)
			c = cd
		}
		fc, _ := e.Flush()
		thinkBenchSinkContent = c + fc + e.Content()
	}
}

// Reasoning stream: a channel-thought span then the answer, folding both the
// cumulative thinking and content totals.
func BenchmarkThinkingExtractor_Process_Reasoning(b *testing.B) {
	toks := benchThoughtTokens()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e := NewThinkingExtractor()
		for _, tok := range toks {
			_, td := e.Process(tok)
			thinkBenchSinkThought = td
		}
		e.Flush()
		thinkBenchSinkContent = e.Content()
		thinkBenchSinkThought = e.Thinking()
	}
}

// --- Single Process token (the isolated per-token cost) ------------------

// A mid-stream plain token on a warmed extractor: the marginal per-token cost
// once the answer already carries ~100 tokens of accumulated content.
func BenchmarkThinkingExtractor_Process_OneToken(b *testing.B) {
	warm := benchPlainTokens(100)
	tok := inference.Token{Text: "another "}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e := NewThinkingExtractor()
		for _, w := range warm {
			e.Process(w)
		}
		thinkBenchSinkContent, thinkBenchSinkThought = e.Process(tok)
	}
}
