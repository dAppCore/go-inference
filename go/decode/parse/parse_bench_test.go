// SPDX-Licence-Identifier: EUPL-1.2

package parse_test

import (
	"testing"

	tools "dappco.re/go/inference/agent/tools"
	parse "dappco.re/go/inference/decode/parse"
)

// Package-level sinks defeat dead-code elimination: the compiler cannot prove
// the benchmarked results are unused, so the calls cannot be optimised away.
var (
	sinkCalls   []tools.ToolCall
	sinkNormal  string
	sinkErr     error
	sinkReason  string
	sinkContent string
	sinkParser  parse.ReasoningParser
)

// Realistic Gemma 4 model outputs — these are what the detector sees per
// generation, so every allocation here recurs per response on the serving path.
const (
	// The canonical case: a little leading text, one call, a delimited string
	// arg and a bare number arg.
	benchSingleCall = `Let me check the forecast for you.<|tool_call>call:get_weather{city: <|"|>Paris<|"|>, days: 3}<tool_call|>`

	// Two calls back to back — exercises the per-match append growth.
	benchMultiCall = `<|tool_call>call:get_weather{city: <|"|>Paris<|"|>, units: <|"|>metric<|"|>}<tool_call|>` +
		`<|tool_call>call:get_time{tz: <|"|>Europe/Paris<|"|>}<tool_call|>`

	// Every value kind: string, int, float, bools, arrays, nested object,
	// array-of-object, nested array, bare word — the worst case for the map /
	// slice / number machinery.
	benchComplexCall = `<|tool_call>call:complex{` +
		`name: <|"|>Ada<|"|>, count: 42, ratio: 1.5, active: true, hidden: false, ` +
		`tags: [<|"|>a<|"|>, <|"|>b<|"|>], nums: [1, 2, 3], ` +
		`meta: {role: <|"|>admin<|"|>, level: 9}, ` +
		`people: [{n: <|"|>x<|"|>}, {n: <|"|>y<|"|>}], grid: [[1, 2], [3, 4]], raw: bareword` +
		`}<tool_call|>`

	// The common path: the model answered without calling a tool. No start
	// token, so this is the cheap early return — keep it honest about its floor.
	benchPlainText = `The capital of France is Paris. It has a population of over two million ` +
		`people and is known for the Eiffel Tower, the Louvre, and its cuisine.`

	// A reasoning block followed by the answer — the reasoning splitter's hot case.
	benchReasoning = `<think>The user wants the capital of France. That is Paris. ` +
		`I should answer concisely.</think>The capital of France is Paris.`

	// No reasoning tokens at all — the cheap early return for the splitter.
	benchPlainContent = `The capital of France is Paris.`
)

func BenchmarkParseGemma4ToolCalls_SingleCall(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkCalls, sinkNormal, sinkErr = parse.ParseGemma4ToolCalls(benchSingleCall)
	}
}

func BenchmarkParseGemma4ToolCalls_MultiCall(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkCalls, sinkNormal, sinkErr = parse.ParseGemma4ToolCalls(benchMultiCall)
	}
}

func BenchmarkParseGemma4ToolCalls_Complex(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkCalls, sinkNormal, sinkErr = parse.ParseGemma4ToolCalls(benchComplexCall)
	}
}

func BenchmarkParseGemma4ToolCalls_PlainText(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkCalls, sinkNormal, sinkErr = parse.ParseGemma4ToolCalls(benchPlainText)
	}
}

func BenchmarkReasoningParse_WithThink(b *testing.B) {
	p := parse.Gemma4Reasoning()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkReason, sinkContent = p.Parse(benchReasoning)
	}
}

func BenchmarkReasoningParse_PlainContent(b *testing.B) {
	p := parse.Gemma4Reasoning()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkReason, sinkContent = p.Parse(benchPlainContent)
	}
}

func BenchmarkGemma4Reasoning(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkParser = parse.Gemma4Reasoning()
	}
}
