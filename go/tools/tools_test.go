// SPDX-Licence-Identifier: EUPL-1.2

package tools

import (
	"context"

	core "dappco.re/go"
)

// fakeExecutor is a test double: it echoes a fixed reply, or fails on demand,
// recording every call it received so the parallel path can be asserted.
//
//	reg.Register("echo", &fakeExecutor{reply: "hi"})
type fakeExecutor struct {
	reply string
	err   error
}

func (f *fakeExecutor) Execute(_ context.Context, call ToolCall) (ToolResult, error) {
	if f.err != nil {
		return ToolResult{}, f.err
	}
	return ToolResult{ID: call.ID, Content: f.reply}, nil
}

// ---------------------------------------------------------------------------
// ToolChoice.Resolve
// ---------------------------------------------------------------------------

func TestTools_Choice_Good(t *core.T) {
	offered := []Tool{
		{Name: "search", Description: "web search"},
		{Name: "fetch", Description: "web fetch"},
	}

	// auto offers every declared tool, unforced.
	got, err := Resolve(ChoiceAuto(), offered)
	core.AssertNoError(t, err)
	core.AssertLen(t, got, 2, "auto offers all tools")

	// required offers every tool too — the difference (the model MUST call one)
	// is carried by the choice value, not the returned set.
	got, err = Resolve(ChoiceRequired(), offered)
	core.AssertNoError(t, err)
	core.AssertLen(t, got, 2, "required still offers all tools")

	// named narrows the set to exactly the forced tool.
	got, err = Resolve(ChoiceTool("fetch"), offered)
	core.AssertNoError(t, err)
	core.AssertLen(t, got, 1, "a named choice offers only that tool")
	core.AssertEqual(t, "fetch", got[0].Name)
}

func TestTools_Choice_Bad(t *core.T) {
	offered := []Tool{{Name: "search"}}

	// A named choice for a tool that isn't declared is a caller error, not a
	// silent no-op — the model would be told to call something that can't run.
	_, err := Resolve(ChoiceTool("missing"), offered)
	core.AssertError(t, err, "not declared")

	// required with no tools to require is equally a contradiction.
	_, err = Resolve(ChoiceRequired(), nil)
	core.AssertError(t, err, "no tools were declared")
}

func TestTools_Choice_Ugly(t *core.T) {
	offered := []Tool{{Name: "search"}, {Name: "fetch"}}

	// none suppresses all tools regardless of what's declared — an empty,
	// non-nil offer with no error.
	got, err := Resolve(ChoiceNone(), offered)
	core.AssertNoError(t, err)
	core.AssertLen(t, got, 0, "none offers no tools")

	// The zero-value choice defaults to auto, so a caller that forgot to set one
	// still gets sane behaviour rather than a panic.
	got, err = Resolve(ToolChoice{}, offered)
	core.AssertNoError(t, err)
	core.AssertLen(t, got, 2, "the zero choice behaves as auto")

	// auto over an empty tool set is fine — the model simply has nothing to call.
	got, err = Resolve(ChoiceAuto(), nil)
	core.AssertNoError(t, err)
	core.AssertLen(t, got, 0)
}

// ---------------------------------------------------------------------------
// ParseToolCalls
// ---------------------------------------------------------------------------

func TestTools_Parse_Good(t *core.T) {
	raw := `[
	  {"id":"call_1","name":"search","arguments":"{\"q\":\"lethean\"}"},
	  {"id":"call_2","name":"fetch","arguments":"{\"url\":\"https://lthn.ai\"}"}
	]`
	calls, err := ParseToolCalls(raw)
	core.AssertNoError(t, err)
	core.AssertLen(t, calls, 2)
	core.AssertEqual(t, "call_1", calls[0].ID)
	core.AssertEqual(t, "search", calls[0].Name)
	core.AssertEqual(t, `{"q":"lethean"}`, calls[0].Arguments)
	core.AssertEqual(t, "fetch", calls[1].Name)

	// A single object (not an array) is the common one-call shape and parses too.
	one, err := ParseToolCalls(`{"id":"c","name":"datetime","arguments":"{}"}`)
	core.AssertNoError(t, err)
	core.AssertLen(t, one, 1)
	core.AssertEqual(t, "datetime", one[0].Name)
}

func TestTools_Parse_Bad(t *core.T) {
	// Malformed JSON is an error, not an empty slice — the model returned junk.
	_, err := ParseToolCalls(`[{"id":"call_1","name":"search"`)
	core.AssertError(t, err, "parse tool calls")

	// A call with no name can't be dispatched to any executor — reject it.
	_, err = ParseToolCalls(`[{"id":"call_1","arguments":"{}"}]`)
	core.AssertError(t, err, "missing its tool name")
}

func TestTools_Parse_Ugly(t *core.T) {
	// Empty / whitespace input means "the model called no tools" — not an error,
	// just an empty set. The runner loops on len==0, it shouldn't have to special
	// case an error here.
	calls, err := ParseToolCalls("")
	core.AssertNoError(t, err)
	core.AssertLen(t, calls, 0)

	calls, err = ParseToolCalls("   \n\t ")
	core.AssertNoError(t, err)
	core.AssertLen(t, calls, 0)

	// An empty JSON array is likewise no calls, no error.
	calls, err = ParseToolCalls("[]")
	core.AssertNoError(t, err)
	core.AssertLen(t, calls, 0)
}

// ---------------------------------------------------------------------------
// Registry + Dispatch
// ---------------------------------------------------------------------------

func TestTools_Dispatch_Good(t *core.T) {
	reg := NewRegistry()
	reg.Register("search", &fakeExecutor{reply: "result-a"})
	reg.Register("fetch", &fakeExecutor{reply: "result-b"})

	calls := []ToolCall{
		{ID: "1", Name: "search"},
		{ID: "2", Name: "fetch"},
	}

	// Sequential dispatch returns results in input order, each tagged with its
	// call ID, no errors.
	out := Dispatch(context.Background(), calls, reg, false)
	core.AssertLen(t, out, 2, "one result per call")
	core.AssertEqual(t, "1", out[0].ID)
	core.AssertEqual(t, "result-a", out[0].Content)
	core.AssertNoError(t, out[0].Err)
	core.AssertEqual(t, "2", out[1].ID)
	core.AssertEqual(t, "result-b", out[1].Content)

	// The parallel path produces the same ordered results — concurrency must not
	// reorder the output.
	par := Dispatch(context.Background(), calls, reg, true)
	core.AssertLen(t, par, 2)
	core.AssertEqual(t, "1", par[0].ID)
	core.AssertEqual(t, "result-a", par[0].Content)
	core.AssertEqual(t, "2", par[1].ID)
	core.AssertEqual(t, "result-b", par[1].Content)
}

func TestTools_Dispatch_Bad(t *core.T) {
	reg := NewRegistry()
	reg.Register("search", &fakeExecutor{reply: "ok"})

	// An unknown tool becomes a ToolResult with Err set — it MUST NOT abort the
	// batch; the known tool still runs and succeeds.
	calls := []ToolCall{
		{ID: "1", Name: "search"},
		{ID: "2", Name: "ghost"},
	}
	out := Dispatch(context.Background(), calls, reg, false)
	core.AssertLen(t, out, 2, "an unknown tool still yields a result slot")
	core.AssertNoError(t, out[0].Err)
	core.AssertEqual(t, "ok", out[0].Content)
	core.AssertEqual(t, "2", out[1].ID, "the failed result keeps its call ID")
	core.AssertError(t, out[1].Err, "no executor registered")
}

func TestTools_Dispatch_Ugly(t *core.T) {
	reg := NewRegistry()
	boom := core.E("tools", "executor exploded", nil)
	reg.Register("ok", &fakeExecutor{reply: "fine"})
	reg.Register("boom", &fakeExecutor{err: boom})

	// One executor errors mid-batch; the others still succeed and the error is
	// captured in that call's slot, in order — true on both paths.
	calls := []ToolCall{
		{ID: "1", Name: "boom"},
		{ID: "2", Name: "ok"},
	}

	seq := Dispatch(context.Background(), calls, reg, false)
	core.AssertLen(t, seq, 2)
	core.AssertError(t, seq[0].Err, "executor exploded") // the executor's own error chains through
	core.AssertEqual(t, "1", seq[0].ID)
	core.AssertNoError(t, seq[1].Err, "a sibling failure doesn't taint a good call")
	core.AssertEqual(t, "fine", seq[1].Content)

	par := Dispatch(context.Background(), calls, reg, true)
	core.AssertLen(t, par, 2)
	core.AssertError(t, par[0].Err, "executor exploded") // parallel path captures it too
	core.AssertEqual(t, "fine", par[1].Content)

	// An empty batch is a no-op — an empty, non-nil slice, no panic.
	empty := Dispatch(context.Background(), nil, reg, true)
	core.AssertLen(t, empty, 0)
}

// panicExecutor blows up inside Execute, modelling a misbehaving tool the
// dispatcher must contain rather than crash on.
type panicExecutor struct{}

func (panicExecutor) Execute(_ context.Context, _ ToolCall) (ToolResult, error) {
	panic("executor went bang")
}

// TestTools_Dispatch_Panic covers runOne's panic recovery: an executor that
// panics is turned into a ToolResult carrying the call's ID and an error, so the
// rest of the batch still runs. Both the sequential and parallel paths must
// contain the panic.
func TestTools_Dispatch_Panic(t *core.T) {
	reg := NewRegistry()
	reg.Register("boom", panicExecutor{})
	reg.Register("ok", &fakeExecutor{reply: "fine"})

	calls := []ToolCall{
		{ID: "1", Name: "boom"},
		{ID: "2", Name: "ok"},
	}

	seq := Dispatch(context.Background(), calls, reg, false)
	core.AssertLen(t, seq, 2)
	core.AssertEqual(t, "1", seq[0].ID, "the panicked call keeps its ID")
	core.AssertError(t, seq[0].Err, "executor panicked")
	core.AssertNoError(t, seq[1].Err, "a panicking sibling doesn't taint a good call")
	core.AssertEqual(t, "fine", seq[1].Content)

	// The parallel path recovers the panic per-goroutine too — the batch does not
	// crash and the good call still returns.
	par := Dispatch(context.Background(), calls, reg, true)
	core.AssertLen(t, par, 2)
	core.AssertError(t, par[0].Err, "executor panicked")
	core.AssertEqual(t, "fine", par[1].Content)
}

// terseExecutor returns a result WITHOUT setting an ID, so the dispatcher must
// backfill the call's ID to keep the result correlatable.
type terseExecutor struct {
	reply string
}

func (e terseExecutor) Execute(_ context.Context, _ ToolCall) (ToolResult, error) {
	return ToolResult{Content: e.reply}, nil // no ID set
}

// TestTools_Dispatch_TerseExecutor covers runOne's ID-backfill branch: an
// executor that leaves ToolResult.ID empty still yields a result tagged with the
// originating call's ID, so the model can correlate it.
func TestTools_Dispatch_TerseExecutor(t *core.T) {
	reg := NewRegistry()
	reg.Register("terse", terseExecutor{reply: "answer"})

	out := Dispatch(context.Background(), []ToolCall{{ID: "call-42", Name: "terse"}}, reg, false)
	core.AssertLen(t, out, 1)
	core.AssertEqual(t, "call-42", out[0].ID, "an empty result ID is backfilled from the call")
	core.AssertEqual(t, "answer", out[0].Content)
	core.AssertNoError(t, out[0].Err)
}

// ---------------------------------------------------------------------------
// Tool.IsServer
// ---------------------------------------------------------------------------

func TestTools_IsServer_Good(t *core.T) {
	// A tool with a ServerKind set runs inside the pipeline (true); a plain
	// function tool (no ServerKind) round-trips its call back to the caller
	// (false).
	srv := Tool{Name: "web_search", ServerKind: ServerWebSearch}
	core.AssertTrue(t, srv.IsServer(), "a tool with a server kind is a server tool")

	fn := Tool{Name: "get_weather", Description: "current weather"}
	core.AssertFalse(t, fn.IsServer(), "a plain function tool is not a server tool")

	// The MCP server kind is likewise a server tool (the own MCP server).
	mcp := Tool{Name: "lthn_search", ServerKind: ServerMCP}
	core.AssertTrue(t, mcp.IsServer())
}

// TestTools_Parse_Null covers the explicit JSON null case: a model output of
// literal `null` decodes to a nil slice, which ParseToolCalls normalises to an
// empty (non-nil) slice with no error — "no tools called", not a failure.
func TestTools_Parse_Null(t *core.T) {
	calls, err := ParseToolCalls("null")
	core.AssertNoError(t, err, "JSON null means no tools, not an error")
	core.AssertLen(t, calls, 0)
	core.AssertNotNil(t, calls, "the returned slice is empty but non-nil")
}
