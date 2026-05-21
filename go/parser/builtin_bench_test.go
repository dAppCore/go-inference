// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the built-in OutputParser shell — newBuiltinOutputParser,
// ParserID, ParseReasoning, ParseTools. Per AX-11 — every reasoning- and
// tool-emitting model resolves to a builtinOutputParser instance and the
// ParseReasoning / ParseTools entry points fire once per generation
// flush of the streamed response. Marker-set is varied (qwen vs gemma
// vs gpt-oss) because the per-call cost is dominated by the marker
// scan in parseReasoningText, which itself is the per-segment hot
// loop driven by indexString.
//
// Run:    go test -bench='Benchmark_Builtin' -benchmem -run='^$' ./go/parser
//
// Stream sizes mirror the realistic generation shapes:
//   - 32-token  ≈ short answer, no reasoning span
//   - 256-token ≈ typical chat response with mid-length reasoning
//   - 2048-token ≈ long-form response (the loop pays N times here)

package parser

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Sinks defeat compiler DCE.
var (
	builtinBenchParser   *builtinOutputParser
	builtinBenchID       string
	builtinBenchReason   inference.ReasoningParseResult
	builtinBenchTools    inference.ToolParseResult
	builtinBenchErr      error
)

// Roughly one English word ≈ one token for fixture-generation purposes —
// good enough for the parser scan cost which is bytes-driven.
func builtinBenchText(tokens int) string {
	out := core.NewBuilder()
	for i := 0; i < tokens; i++ {
		out.WriteString("word ")
	}
	return out.String()
}

// builtinBenchReasoningStream produces a synthetic generation of
// `tokens` words wrapped with a <think>...</think> span covering the
// requested fraction of the stream. spanFraction is 0.10, 0.50, 0.90.
func builtinBenchReasoningStream(tokens int, spanFraction float64, startMarker, endMarker string) string {
	span := int(float64(tokens) * spanFraction)
	if span < 1 {
		span = 1
	}
	if span > tokens {
		span = tokens
	}
	pre := (tokens - span) / 2
	post := tokens - span - pre
	out := core.NewBuilder()
	out.WriteString(builtinBenchText(pre))
	out.WriteString(startMarker)
	out.WriteString(builtinBenchText(span))
	out.WriteString(endMarker)
	out.WriteString(builtinBenchText(post))
	return out.String()
}

// --- newBuiltinOutputParser (per-registry build) ---

func Benchmark_Builtin_New_Generic(b *testing.B) {
	markers := genericMarkers()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		builtinBenchParser = newBuiltinOutputParser("generic", markers)
	}
}

func Benchmark_Builtin_New_Qwen(b *testing.B) {
	markers := qwenMarkers()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		builtinBenchParser = newBuiltinOutputParser("qwen", markers)
	}
}

func Benchmark_Builtin_New_Gemma(b *testing.B) {
	markers := gemmaMarkers()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		builtinBenchParser = newBuiltinOutputParser("gemma", markers)
	}
}

// --- ParserID (called per dispatch + per Process flush) ---

func Benchmark_Builtin_ParserID(b *testing.B) {
	parser := newBuiltinOutputParser("qwen", qwenMarkers())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		builtinBenchID = parser.ParserID()
	}
}

func Benchmark_Builtin_ParserID_NilReceiver(b *testing.B) {
	var parser *builtinOutputParser
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		builtinBenchID = parser.ParserID()
	}
}

// --- ParseReasoning across stream sizes × span fractions × architectures ---
// The 3 architectures cover the three marker shapes:
//   qwen  — single short pair `<think>…</think>`
//   gemma — multi-pair channel markers
//   gpt-oss — multi-end markers (the worst-case findReasoningStart fan-out)

var builtinBenchArchitectures = []struct {
	id      string
	parser  *builtinOutputParser
	start   string
	end     string
}{
	{"qwen", newBuiltinOutputParser("qwen", qwenMarkers()), "<think>", "</think>"},
	{"gemma", newBuiltinOutputParser("gemma", gemmaMarkers()), "<start_of_turn>thinking\n", "<end_of_turn>"},
	{"gptoss", newBuiltinOutputParser("gpt-oss", gptOSSMarkers()), "<|channel>analysis\n", "<|channel>final\n"},
}

var builtinBenchStreamSizes = []int{32, 256, 2048}

var builtinBenchSpanFractions = []struct {
	id   string
	frac float64
}{
	{"Span10pct", 0.10},
	{"Span50pct", 0.50},
	{"Span90pct", 0.90},
}

func Benchmark_Builtin_ParseReasoning(b *testing.B) {
	for _, arch := range builtinBenchArchitectures {
		for _, size := range builtinBenchStreamSizes {
			for _, span := range builtinBenchSpanFractions {
				text := builtinBenchReasoningStream(size, span.frac, arch.start, arch.end)
				b.Run(arch.id+"/"+span.id+"/"+core.Sprintf("Tokens%d", size), func(b *testing.B) {
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						builtinBenchReason, builtinBenchErr = arch.parser.ParseReasoning(nil, text)
					}
				})
			}
		}
	}
}

// No reasoning span at all — common case for short factual answers.
func Benchmark_Builtin_ParseReasoning_NoSpan_Qwen(b *testing.B) {
	parser := newBuiltinOutputParser("qwen", qwenMarkers())
	text := builtinBenchText(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		builtinBenchReason, builtinBenchErr = parser.ParseReasoning(nil, text)
	}
}

// Nil receiver pays the lazy-construction cost of building the
// generic-fallback parser before the parse runs.
func Benchmark_Builtin_ParseReasoning_NilReceiver(b *testing.B) {
	var parser *builtinOutputParser
	text := "pre<thinking>plan</thinking>answer"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		builtinBenchReason, builtinBenchErr = parser.ParseReasoning(nil, text)
	}
}

// --- ParseTools — 0 / 1 / 5 tool invocations per response ---

func Benchmark_Builtin_ParseTools_NoCalls(b *testing.B) {
	parser := newBuiltinOutputParser("hermes", genericMarkers())
	text := builtinBenchText(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		builtinBenchTools, builtinBenchErr = parser.ParseTools(nil, text)
	}
}

func Benchmark_Builtin_ParseTools_OneCall(b *testing.B) {
	parser := newBuiltinOutputParser("hermes", genericMarkers())
	text := `before <tool_call>{"name":"search","arguments":{"q":"core"}}</tool_call> after`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		builtinBenchTools, builtinBenchErr = parser.ParseTools(nil, text)
	}
}

func Benchmark_Builtin_ParseTools_FiveCalls(b *testing.B) {
	parser := newBuiltinOutputParser("hermes", genericMarkers())
	out := core.NewBuilder()
	out.WriteString("preamble text ")
	for i := 0; i < 5; i++ {
		out.WriteString(`<tool_call>{"name":"search","arguments":{"q":"core","page":`)
		out.WriteString(core.Sprintf("%d", i))
		out.WriteString(`}}</tool_call> `)
	}
	out.WriteString("trailing text")
	text := out.String()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		builtinBenchTools, builtinBenchErr = parser.ParseTools(nil, text)
	}
}
