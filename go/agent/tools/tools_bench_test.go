// SPDX-Licence-Identifier: EUPL-1.2

package tools_test

import (
	"context"
	"testing"

	"dappco.re/go/inference/agent/tools"
)

// AX-11 allocation baselines for the tool-calling orchestration surface
// (tools.go / parse.go / dispatch.go). These run once per request — per
// turn for Resolve, per model output for ParseToolCalls, per tool-call
// batch for Dispatch — so an alloc regression here scales 1×per-request
// across every adapter that offers tools.
//
// One benchmark per public function (plus the per-mode / per-shape
// variants that exercise distinct alloc paths), realistic tool-definition
// and tool-call fixtures, ReportAllocs. Package-level sinks defeat
// dead-code elimination. Black-box (package tools_test) — every target
// is exported.
//
// Run:
//   go test -bench=. -benchmem -benchtime=200ms -run='^$' ./tools/

// Sinks — one per returned type so the compiler cannot prove the result
// unused and elide the call.
var (
	sinkTools   []tools.Tool
	sinkCalls   []tools.ToolCall
	sinkResults []tools.ToolResult
	sinkErr     error
	sinkBool    bool
	sinkReg     *tools.Registry
	sinkExec    tools.Executor
	sinkOK      bool
)

// declaredTools is a realistic per-turn tool set: three function tools
// carrying JSON-schema parameters plus two server tools, the shape a chat
// request declares for a single turn.
func declaredTools() []tools.Tool {
	return []tools.Tool{
		{Name: "get_weather", Description: "Get the current weather for a city",
			Parameters: `{"type":"object","properties":{"city":{"type":"string"},"units":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}`},
		{Name: "search_web", Description: "Search the web for a query",
			Parameters: `{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}},"required":["query"]}`},
		{Name: "send_email", Description: "Send an email to a recipient",
			Parameters: `{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]}`},
		{Name: "web_search", ServerKind: tools.ServerWebSearch},
		{Name: "code_interpreter", ServerKind: tools.ServerCodeInterpreter},
	}
}

// --- Resolve ---

// Auto returns every declared tool through cloneTools — the defensive copy
// is the package's documented contract.
func BenchmarkResolve_Auto(b *testing.B) {
	declared := declaredTools()
	choice := tools.ChoiceAuto()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkTools, sinkErr = tools.Resolve(choice, declared)
	}
}

// Required walks the same cloneTools path as auto.
func BenchmarkResolve_Required(b *testing.B) {
	declared := declaredTools()
	choice := tools.ChoiceRequired()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkTools, sinkErr = tools.Resolve(choice, declared)
	}
}

// Tool narrows the set to the single forced tool — a one-element slice.
func BenchmarkResolve_Tool(b *testing.B) {
	declared := declaredTools()
	choice := tools.ChoiceTool("send_email")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkTools, sinkErr = tools.Resolve(choice, declared)
	}
}

// None returns an empty, non-nil slice.
func BenchmarkResolve_None(b *testing.B) {
	declared := declaredTools()
	choice := tools.ChoiceNone()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkTools, sinkErr = tools.Resolve(choice, declared)
	}
}

// --- ParseToolCalls ---

// The common one-call shape: a single JSON object the model emits when it
// calls exactly one tool.
const oneCallJSON = `{"id":"call_a1b2","name":"get_weather","arguments":"{\"city\":\"London\",\"units\":\"celsius\"}"}`

// The parallel-tool-calls shape: a JSON array of several calls in one turn.
const multiCallJSON = `[` +
	`{"id":"call_1","name":"get_weather","arguments":"{\"city\":\"London\"}"},` +
	`{"id":"call_2","name":"search_web","arguments":"{\"query\":\"lethean\",\"limit\":5}"},` +
	`{"id":"call_3","name":"send_email","arguments":"{\"to\":\"a@b.c\",\"body\":\"hi\"}"}` +
	`]`

// Single-object shape — the prefix-"{" wrap-and-decode path.
func BenchmarkParseToolCalls_Single(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkCalls, sinkErr = tools.ParseToolCalls(oneCallJSON)
	}
}

// Array shape — the multi-call decode path.
func BenchmarkParseToolCalls_Array(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkCalls, sinkErr = tools.ParseToolCalls(multiCallJSON)
	}
}

// Empty input — the "model called no tools" fast path.
func BenchmarkParseToolCalls_Empty(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkCalls, sinkErr = tools.ParseToolCalls("")
	}
}

// --- Dispatch ---

// benchExec is a no-op executor that echoes a fixed reply — the dispatch
// machinery is what's measured, not executor work.
type benchExec struct{}

func (benchExec) Execute(_ context.Context, call tools.ToolCall) (tools.ToolResult, error) {
	return tools.ToolResult{ID: call.ID, Content: "ok"}, nil
}

// benchRegistry registers the three function tools the bench calls target.
func benchRegistry() *tools.Registry {
	reg := tools.NewRegistry()
	reg.Register("get_weather", benchExec{})
	reg.Register("search_web", benchExec{})
	reg.Register("send_email", benchExec{})
	return reg
}

// benchCalls is a realistic three-call batch (parallel_tool_calls).
func benchCalls() []tools.ToolCall {
	return []tools.ToolCall{
		{ID: "call_1", Name: "get_weather", Arguments: `{"city":"London"}`},
		{ID: "call_2", Name: "search_web", Arguments: `{"query":"lethean"}`},
		{ID: "call_3", Name: "send_email", Arguments: `{"to":"a@b.c","body":"hi"}`},
	}
}

// Sequential dispatch over a three-call batch.
func BenchmarkDispatch_Sequential(b *testing.B) {
	reg := benchRegistry()
	calls := benchCalls()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkResults = tools.Dispatch(ctx, calls, reg, false)
	}
}

// Parallel dispatch over the same batch — one goroutine per call.
func BenchmarkDispatch_Parallel(b *testing.B) {
	reg := benchRegistry()
	calls := benchCalls()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkResults = tools.Dispatch(ctx, calls, reg, true)
	}
}

// --- Registry ---

// NewRegistry allocates the backing map.
func BenchmarkNewRegistry(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkReg = tools.NewRegistry()
	}
}

// Register at steady state — replacing an existing key, no map growth.
func BenchmarkRegistry_Register(b *testing.B) {
	reg := tools.NewRegistry()
	exec := benchExec{}
	reg.Register("get_weather", exec)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reg.Register("get_weather", exec)
	}
}

// Lookup of a present key.
func BenchmarkRegistry_Lookup(b *testing.B) {
	reg := benchRegistry()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkExec, sinkOK = reg.Lookup("search_web")
	}
}

// --- Tool.IsServer ---

func BenchmarkTool_IsServer(b *testing.B) {
	srv := tools.Tool{Name: "web_search", ServerKind: tools.ServerWebSearch}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkBool = srv.IsServer()
	}
}
