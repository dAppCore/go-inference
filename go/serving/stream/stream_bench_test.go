// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the streaming taxonomy + assembler. Per AX-11 — this
// machinery runs PER emitted chunk/delta: Assembler.Add fires once for
// every token of a generation, Collect/Result fold the whole sequence,
// and FromTokens(Err) builds the unified event slice a local engine
// emits. Allocations here recur many times per response, so the per-op
// allocation profile is what a consumer pays on every streamed answer.
//
// Run:    go test -bench=BenchmarkStream -benchmem -run='^$' ./stream/
package stream_test

import (
	"testing"

	stream "dappco.re/go/inference/serving/stream"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE.
var (
	streamBenchSinkString string
	streamBenchSinkErr    error
	streamBenchSinkAsm    *stream.Assembler
	streamBenchSinkResp   stream.Response
	streamBenchSinkEvents []stream.Event
)

// benchTokens is a realistic ~48-token text generation — the dominant
// shape of a streamed answer (mostly text deltas).
var benchTokens = []string{
	"The ", "quick ", "brown ", "fox ", "jumps ", "over ", "the ", "lazy ", "dog", ". ",
	"It ", "was ", "a ", "bright ", "cold ", "day ", "in ", "April", ", ", "and ",
	"the ", "clocks ", "were ", "striking ", "thirteen", ". ", "Winston ", "Smith", ", ", "his ",
	"chin ", "nuzzled ", "into ", "his ", "breast ", "in ", "an ", "effort ", "to ", "escape ",
	"the ", "vile ", "wind", ", ", "slipped ", "quickly ", "through", ".",
}

var benchUsage = stream.Usage{PromptTokens: 16, CompletionTokens: 48, TotalTokens: 64}

// benchTextStream is the unified event sequence for a plain text answer:
// created → 48 text deltas → text-done → usage → completed. This is what
// the assembler folds on the common path (no tool calls).
func benchTextStream() []stream.Event {
	evs := make([]stream.Event, 0, len(benchTokens)+4)
	evs = append(evs, stream.Event{Kind: stream.KindResponseCreated, ResponseID: "resp-bench"})
	for _, tok := range benchTokens {
		evs = append(evs, stream.Event{Kind: stream.KindTextDelta, Text: tok})
	}
	evs = append(evs, stream.Event{Kind: stream.KindTextDone})
	evs = append(evs, stream.Event{Kind: stream.KindUsage, Usage: benchUsage})
	evs = append(evs, stream.Event{Kind: stream.KindResponseCompleted, ResponseID: "resp-bench"})
	return evs
}

// benchToolStream interleaves a function call into a shorter text answer,
// exercising the tool-index map path (ensureTool / appendToolArgs).
func benchToolStream() []stream.Event {
	return []stream.Event{
		{Kind: stream.KindResponseCreated, ResponseID: "resp-tool"},
		{Kind: stream.KindTextDelta, Text: "Let me look that up. "},
		{Kind: stream.KindFunctionCallArgsDelta, ToolCallID: "call-1", ToolName: "search", Text: `{"query":`},
		{Kind: stream.KindFunctionCallArgsDelta, ToolCallID: "call-1", Text: `"go allocations"`},
		{Kind: stream.KindFunctionCallArgsDelta, ToolCallID: "call-1", Text: `,"limit":5}`},
		{Kind: stream.KindFunctionCallArgsDone, ToolCallID: "call-1"},
		{Kind: stream.KindTextDone},
		{Kind: stream.KindUsage, Usage: benchUsage},
		{Kind: stream.KindResponseCompleted, ResponseID: "resp-tool"},
	}
}

// --- Kind.String ---------------------------------------------------------

func BenchmarkStream_KindString(b *testing.B) {
	k := stream.KindResponseCompleted
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		streamBenchSinkString = k.String()
	}
}

// --- StreamError.Error ---------------------------------------------------

func BenchmarkStream_StreamErrorError(b *testing.B) {
	e := &stream.StreamError{Code: "rate_limited", Message: "429 slow down"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		streamBenchSinkString = e.Error()
	}
}

// --- NewAssembler --------------------------------------------------------

func BenchmarkStream_NewAssembler(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		streamBenchSinkAsm = stream.NewAssembler()
	}
}

// --- Assembler.Add (the per-chunk hot path) ------------------------------

// Text-dominant: a fresh assembler folds the whole 48-token text stream
// per iteration — this is the append-growth cost paid on every answer.
func BenchmarkStream_Add(b *testing.B) {
	evs := benchTextStream()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a := stream.NewAssembler()
		for _, ev := range evs {
			streamBenchSinkErr = a.Add(ev)
		}
		streamBenchSinkAsm = a
	}
}

// Tool path: exercises ensureTool/appendToolArgs and the tool-index map.
func BenchmarkStream_Add_Tool(b *testing.B) {
	evs := benchToolStream()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a := stream.NewAssembler()
		for _, ev := range evs {
			streamBenchSinkErr = a.Add(ev)
		}
		streamBenchSinkAsm = a
	}
}

// --- Assembler.Result ----------------------------------------------------

// Result is idempotent (it does not mutate the assembler), so a single
// pre-populated assembler is folded repeatedly to isolate the join cost.
func BenchmarkStream_Result(b *testing.B) {
	a := stream.NewAssembler()
	for _, ev := range benchTextStream() {
		_ = a.Add(ev)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		streamBenchSinkResp = a.Result()
	}
}

// --- Collect (batch fold: NewAssembler + Add loop + Result) --------------

func BenchmarkStream_Collect(b *testing.B) {
	evs := benchTextStream()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		streamBenchSinkResp, streamBenchSinkErr = stream.Collect(evs)
	}
}

func BenchmarkStream_Collect_Tool(b *testing.B) {
	evs := benchToolStream()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		streamBenchSinkResp, streamBenchSinkErr = stream.Collect(evs)
	}
}

// --- FromTokens ----------------------------------------------------------

func BenchmarkStream_FromTokens(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		streamBenchSinkEvents = stream.FromTokens(benchTokens, benchUsage)
	}
}

// --- FromTokensErr -------------------------------------------------------

// Success path (nil genErr) mirrors FromTokens; the failure path is
// measured separately as it allocates the terminal StreamError.
func BenchmarkStream_FromTokensErr(b *testing.B) {
	genErr := core.E("mlx", "decode aborted", nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		streamBenchSinkEvents = stream.FromTokensErr(benchTokens, benchUsage, genErr)
	}
}
