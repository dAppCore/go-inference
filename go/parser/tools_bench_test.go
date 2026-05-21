// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the tool-call parser — parseToolText, findToolBlockStart,
// parseToolPayload, convertParsedToolCalls, convertParsedToolCall,
// normaliseArgumentsJSON. Per AX-11 — parseToolText is the per-flush
// hot loop fired on every completion that may carry a tool call (every
// agentic-mode response). findToolBlockStart is the per-scan fan-out
// across three block-marker pairs. parseToolPayload pays the JSON-decode
// + envelope-walk per call. The bench varies tool-call count (0 / 1 / 5)
// and stream length to mirror realistic agent traces.
//
// Run:    go test -bench='Benchmark_Tools' -benchmem -run='^$' ./go/parser

package parser

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Sinks defeat compiler DCE.
var (
	toolsBenchResult   inference.ToolParseResult
	toolsBenchErr      error
	toolsBenchCalls    []inference.ToolCall
	toolsBenchCall     inference.ToolCall
	toolsBenchIdx      int
	toolsBenchMarker   toolBlockMarker
	toolsBenchOK       bool
	toolsBenchString   string
)

// toolsBenchWords builds a synthetic prose stream of `tokens` words.
func toolsBenchWords(tokens int) string {
	out := core.NewBuilder()
	for i := 0; i < tokens; i++ {
		out.WriteString("word ")
	}
	return out.String()
}

// toolsBenchStreamWithCalls splices `n` tool-call blocks evenly
// across a prose stream of `tokens` words.
func toolsBenchStreamWithCalls(tokens, n int) string {
	pre := tokens / (n + 1)
	out := core.NewBuilder()
	for i := 0; i < n; i++ {
		out.WriteString(toolsBenchWords(pre))
		out.WriteString(`<tool_call>{"name":"search","arguments":{"q":"core","page":`)
		out.WriteString(core.Sprintf("%d", i))
		out.WriteString(`}}</tool_call>`)
	}
	out.WriteString(toolsBenchWords(pre))
	return out.String()
}

// --- parseToolText: per-response hot path ---

func Benchmark_Tools_ParseText_NoCalls_Short(b *testing.B) {
	text := toolsBenchWords(32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchResult, toolsBenchErr = parseToolText(text)
	}
}

func Benchmark_Tools_ParseText_NoCalls_Mid(b *testing.B) {
	text := toolsBenchWords(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchResult, toolsBenchErr = parseToolText(text)
	}
}

func Benchmark_Tools_ParseText_NoCalls_Long(b *testing.B) {
	text := toolsBenchWords(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchResult, toolsBenchErr = parseToolText(text)
	}
}

func Benchmark_Tools_ParseText_OneCall_Short(b *testing.B) {
	text := toolsBenchStreamWithCalls(32, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchResult, toolsBenchErr = parseToolText(text)
	}
}

func Benchmark_Tools_ParseText_OneCall_Mid(b *testing.B) {
	text := toolsBenchStreamWithCalls(256, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchResult, toolsBenchErr = parseToolText(text)
	}
}

func Benchmark_Tools_ParseText_OneCall_Long(b *testing.B) {
	text := toolsBenchStreamWithCalls(2048, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchResult, toolsBenchErr = parseToolText(text)
	}
}

func Benchmark_Tools_ParseText_FiveCalls_Mid(b *testing.B) {
	text := toolsBenchStreamWithCalls(256, 5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchResult, toolsBenchErr = parseToolText(text)
	}
}

func Benchmark_Tools_ParseText_FiveCalls_Long(b *testing.B) {
	text := toolsBenchStreamWithCalls(2048, 5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchResult, toolsBenchErr = parseToolText(text)
	}
}

// Unclosed tagged tool-call exercises the `end < 0` branch — the
// scan walks the whole payload looking for `</tool_call>` and falls
// back to passthrough.
func Benchmark_Tools_ParseText_Unclosed(b *testing.B) {
	text := `before <tool_call>{"name":"search","arguments":{"q":"core"}` + toolsBenchWords(64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchResult, toolsBenchErr = parseToolText(text)
	}
}

// Untagged JSON fallback — the entire payload is parsed as JSON.
func Benchmark_Tools_ParseText_JSONFallback(b *testing.B) {
	text := `{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":{"id":7}}}]}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchResult, toolsBenchErr = parseToolText(text)
	}
}

// Tool-calls block (plural) wrapper.
func Benchmark_Tools_ParseText_ToolCallsBlock(b *testing.B) {
	text := `pre <tool_calls>[{"name":"a","arguments":{"x":1}},{"name":"b","arguments":{"y":2}}]</tool_calls> post`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchResult, toolsBenchErr = parseToolText(text)
	}
}

// function_call (singular) wrapper.
func Benchmark_Tools_ParseText_FunctionCallBlock(b *testing.B) {
	text := `pre <function_call>{"name":"a","arguments":{"x":1}}</function_call> post`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchResult, toolsBenchErr = parseToolText(text)
	}
}

// --- findToolBlockStart: per-scan fan-out across 3 marker pairs ---

func Benchmark_Tools_FindBlockStart_HitFirst(b *testing.B) {
	text := `<tool_call>{"name":"x"}</tool_call>tail`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchIdx, toolsBenchMarker, toolsBenchOK = findToolBlockStart(text)
	}
}

func Benchmark_Tools_FindBlockStart_HitMid(b *testing.B) {
	text := toolsBenchWords(64) + `<tool_call>{"name":"x"}</tool_call>tail`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchIdx, toolsBenchMarker, toolsBenchOK = findToolBlockStart(text)
	}
}

func Benchmark_Tools_FindBlockStart_Miss_256bytes(b *testing.B) {
	text := toolsBenchWords(64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchIdx, toolsBenchMarker, toolsBenchOK = findToolBlockStart(text)
	}
}

func Benchmark_Tools_FindBlockStart_Miss_2048bytes(b *testing.B) {
	text := toolsBenchWords(512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchIdx, toolsBenchMarker, toolsBenchOK = findToolBlockStart(text)
	}
}

// --- parseToolPayload: JSON decode + envelope walk ---

func Benchmark_Tools_ParsePayload_SingleObject(b *testing.B) {
	payload := `{"name":"search","arguments":{"q":"core"}}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchCalls, toolsBenchErr = parseToolPayload(payload)
	}
}

func Benchmark_Tools_ParsePayload_Array(b *testing.B) {
	payload := `[{"name":"a","arguments":{"x":1}},{"name":"b","arguments":{"y":2}}]`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchCalls, toolsBenchErr = parseToolPayload(payload)
	}
}

func Benchmark_Tools_ParsePayload_ToolCallsEnvelope(b *testing.B) {
	payload := `{"tool_calls":[{"id":"c1","type":"function","function":{"name":"lookup","arguments":{"id":7}}}]}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchCalls, toolsBenchErr = parseToolPayload(payload)
	}
}

func Benchmark_Tools_ParsePayload_CallsEnvelope(b *testing.B) {
	payload := `{"calls":[{"name":"lookup","arguments":{"id":7}}]}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchCalls, toolsBenchErr = parseToolPayload(payload)
	}
}

func Benchmark_Tools_ParsePayload_FunctionEnvelope(b *testing.B) {
	payload := `{"function":{"name":"lookup","arguments":{"id":7}}}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchCalls, toolsBenchErr = parseToolPayload(payload)
	}
}

func Benchmark_Tools_ParsePayload_Empty(b *testing.B) {
	payload := ""
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchCalls, toolsBenchErr = parseToolPayload(payload)
	}
}

func Benchmark_Tools_ParsePayload_ArgumentsAsString(b *testing.B) {
	payload := `{"name":"search","arguments_json":"{\"q\":\"core\"}"}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchCalls, toolsBenchErr = parseToolPayload(payload)
	}
}

// --- convertParsedToolCalls / convertParsedToolCall ---

func Benchmark_Tools_ConvertParsedToolCall_SimpleName(b *testing.B) {
	parsed := parsedToolCall{Name: "search", Arguments: map[string]any{"q": "core"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchCall = convertParsedToolCall(parsed)
	}
}

func Benchmark_Tools_ConvertParsedToolCall_FromFunctionEnvelope(b *testing.B) {
	parsed := parsedToolCall{
		ID:       "c1",
		Type:     "function",
		Function: &parsedFunction{Name: "lookup", Arguments: map[string]any{"id": 7}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchCall = convertParsedToolCall(parsed)
	}
}

func Benchmark_Tools_ConvertParsedToolCalls_Array(b *testing.B) {
	input := []parsedToolCall{
		{Name: "a", Arguments: map[string]any{"x": 1}},
		{Name: "b", Arguments: map[string]any{"y": 2}},
		{Name: "c", Arguments: map[string]any{"z": 3}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchCalls = convertParsedToolCalls(input)
	}
}

// --- normaliseArgumentsJSON ---

func Benchmark_Tools_NormaliseArgumentsJSON_ExistingJSON(b *testing.B) {
	existing := `{"q":"core"}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchString = normaliseArgumentsJSON(existing, nil)
	}
}

func Benchmark_Tools_NormaliseArgumentsJSON_FromMap(b *testing.B) {
	args := map[string]any{"q": "core", "page": 3}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchString = normaliseArgumentsJSON("", args)
	}
}

func Benchmark_Tools_NormaliseArgumentsJSON_FromString(b *testing.B) {
	args := any(`{"q":"core"}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchString = normaliseArgumentsJSON("", args)
	}
}

func Benchmark_Tools_NormaliseArgumentsJSON_Nil(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchString = normaliseArgumentsJSON("", nil)
	}
}
