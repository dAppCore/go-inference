// SPDX-Licence-Identifier: EUPL-1.2

package stream

import (
	"testing"

	core "dappco.re/go"
)

// --- Collect -------------------------------------------------------------

func TestStream_Collect_Good(t *testing.T) {
	// A full lifecycle: created → text deltas → reasoning → a tool call →
	// text-done → usage → completed. The assembler must concatenate text in
	// order, collect reasoning, gather the tool call's arguments, and surface
	// the final usage.
	events := []Event{
		{Kind: KindResponseCreated, ResponseID: "resp-1"},
		{Kind: KindReasoningDelta, Text: "let me think"},
		{Kind: KindReasoningDone},
		{Kind: KindTextDelta, Text: "Hello, "},
		{Kind: KindTextDelta, Text: "world"},
		{Kind: KindFunctionCallArgsDelta, ToolCallID: "call-1", ToolName: "search", Text: `{"q":`},
		{Kind: KindFunctionCallArgsDelta, ToolCallID: "call-1", Text: `"go"}`},
		{Kind: KindFunctionCallArgsDone, ToolCallID: "call-1"},
		{Kind: KindTextDone},
		{Kind: KindUsage, Usage: Usage{PromptTokens: 10, CompletionTokens: 3, TotalTokens: 13}},
		{Kind: KindResponseCompleted, ResponseID: "resp-1"},
	}

	resp, err := Collect(events)
	if err != nil {
		t.Fatalf("Collect lifecycle: unexpected error: %v", err)
	}
	if resp.Text != "Hello, world" {
		t.Fatalf("Collect text: got %q, want %q", resp.Text, "Hello, world")
	}
	if resp.Reasoning != "let me think" {
		t.Fatalf("Collect reasoning: got %q, want %q", resp.Reasoning, "let me think")
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("Collect tool calls: got %d, want 1", len(resp.ToolCalls))
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "call-1" || tc.Name != "search" {
		t.Fatalf("Collect tool call identity: got id=%q name=%q", tc.ID, tc.Name)
	}
	if tc.Arguments != `{"q":"go"}` {
		t.Fatalf("Collect tool call args: got %q, want %q", tc.Arguments, `{"q":"go"}`)
	}
	if resp.Usage.TotalTokens != 13 {
		t.Fatalf("Collect usage: got total %d, want 13", resp.Usage.TotalTokens)
	}
	if !resp.Completed {
		t.Fatal("Collect: expected Completed true after response-completed")
	}
	if resp.ResponseID != "resp-1" {
		t.Fatalf("Collect response id: got %q, want %q", resp.ResponseID, "resp-1")
	}
}

func TestStream_Collect_Bad(t *testing.T) {
	// An error event mid-stream must yield an error from Collect — the partial
	// text already accumulated is not a successful response.
	events := []Event{
		{Kind: KindTextDelta, Text: "partial"},
		{Kind: KindError, Err: &StreamError{Code: "rate_limited", Message: "429 slow down"}},
	}
	if _, err := Collect(events); err == nil {
		t.Fatal("Collect with error event: expected error, got nil")
	}

	// A response-failed lifecycle event is also surfaced as an error.
	failed := []Event{
		{Kind: KindResponseCreated, ResponseID: "r"},
		{Kind: KindTextDelta, Text: "half"},
		{Kind: KindResponseFailed, Err: &StreamError{Code: "provider_overloaded", Message: "upstream down"}},
	}
	if _, err := Collect(failed); err == nil {
		t.Fatal("Collect with response-failed: expected error, got nil")
	}

	// An error event with no payload still errors (fail closed), and the code
	// is carried through so the caller can branch on the failure class.
	r, err := Collect([]Event{{Kind: KindError, Err: &StreamError{Code: "internal"}}})
	if err == nil {
		t.Fatal("Collect with bare error event: expected error, got nil")
	}
	if r.Err == nil || r.Err.Code != "internal" {
		t.Fatalf("Collect: expected response to carry the stream error code, got %+v", r.Err)
	}
}

func TestStream_Collect_Ugly(t *testing.T) {
	// Edge cases: an empty stream assembles into an empty, non-completed
	// response with no error (nothing happened, but nothing failed either).
	resp, err := Collect(nil)
	if err != nil {
		t.Fatalf("Collect empty stream: unexpected error: %v", err)
	}
	if resp.Text != "" || resp.Completed || len(resp.ToolCalls) != 0 {
		t.Fatalf("Collect empty stream: expected zero response, got %+v", resp)
	}

	// Out-of-order / missing-done: deltas with no terminating text-done and no
	// completed event still assemble the text — done markers refine state, they
	// are not required to read what arrived. Two interleaved tool calls keep
	// their own argument buffers.
	events := []Event{
		{Kind: KindFunctionCallArgsDelta, ToolCallID: "a", ToolName: "alpha", Text: "1"},
		{Kind: KindTextDelta, Text: "mid"},
		{Kind: KindFunctionCallArgsDelta, ToolCallID: "b", ToolName: "beta", Text: "2"},
		{Kind: KindFunctionCallArgsDelta, ToolCallID: "a", Text: "3"},
		// note: no text-done, no usage, no completed
	}
	resp2, err := Collect(events)
	if err != nil {
		t.Fatalf("Collect missing-done: unexpected error: %v", err)
	}
	if resp2.Text != "mid" {
		t.Fatalf("Collect missing-done text: got %q, want %q", resp2.Text, "mid")
	}
	if resp2.Completed {
		t.Fatal("Collect missing-done: Completed must be false without a completed event")
	}
	if len(resp2.ToolCalls) != 2 {
		t.Fatalf("Collect interleaved tool calls: got %d, want 2", len(resp2.ToolCalls))
	}
	// Tool calls are returned in first-seen order; "a" accumulated 1 then 3.
	if resp2.ToolCalls[0].ID != "a" || resp2.ToolCalls[0].Arguments != "13" {
		t.Fatalf("Collect tool call a: got id=%q args=%q", resp2.ToolCalls[0].ID, resp2.ToolCalls[0].Arguments)
	}
	if resp2.ToolCalls[1].ID != "b" || resp2.ToolCalls[1].Arguments != "2" {
		t.Fatalf("Collect tool call b: got id=%q args=%q", resp2.ToolCalls[1].ID, resp2.ToolCalls[1].Arguments)
	}
}

// --- FromTokens ----------------------------------------------------------

func TestStream_FromTokens_Good(t *testing.T) {
	// A plain token stream becomes one text-delta per token, then a single
	// text-done, then a usage frame — the same Event shape a remote SSE stream
	// produces. Feeding the result back through Collect reconstructs the text.
	tokens := []string{"The ", "quick ", "fox"}
	usage := Usage{PromptTokens: 4, CompletionTokens: 3, TotalTokens: 7}
	events := FromTokens(tokens, usage)

	// 3 deltas + text-done + usage = 5 events.
	if len(events) != 5 {
		t.Fatalf("FromTokens count: got %d events, want 5", len(events))
	}
	if events[0].Kind != KindTextDelta || events[0].Text != "The " {
		t.Fatalf("FromTokens first event: got %+v", events[0])
	}
	if events[3].Kind != KindTextDone {
		t.Fatalf("FromTokens penultimate: got kind %s, want text-done", events[3].Kind)
	}
	last := events[len(events)-1]
	if last.Kind != KindUsage || last.Usage.TotalTokens != 7 {
		t.Fatalf("FromTokens trailing usage: got %+v", last)
	}

	resp, err := Collect(events)
	if err != nil {
		t.Fatalf("Collect(FromTokens): unexpected error: %v", err)
	}
	if resp.Text != "The quick fox" {
		t.Fatalf("Collect(FromTokens) text: got %q, want %q", resp.Text, "The quick fox")
	}
	if resp.Usage.TotalTokens != 7 {
		t.Fatalf("Collect(FromTokens) usage: got %d, want 7", resp.Usage.TotalTokens)
	}
}

func TestStream_FromTokens_Bad(t *testing.T) {
	// A nil error from the local generator is fine; a non-nil one becomes an
	// error event so the unified consumer sees a local failure exactly like a
	// remote one. FromTokensErr threads that failure through.
	tokens := []string{"part"}
	genErr := core.E("mlx", "decode aborted", nil)
	events := FromTokensErr(tokens, Usage{}, genErr)

	// The text that arrived before the failure is preserved, then an error
	// event terminates the stream (no text-done / usage on a failed gen).
	if len(events) != 2 {
		t.Fatalf("FromTokensErr count: got %d, want 2", len(events))
	}
	if events[0].Kind != KindTextDelta || events[0].Text != "part" {
		t.Fatalf("FromTokensErr delta: got %+v", events[0])
	}
	if events[1].Kind != KindError || events[1].Err == nil {
		t.Fatalf("FromTokensErr terminator: got %+v", events[1])
	}
	if _, err := Collect(events); err == nil {
		t.Fatal("Collect(FromTokensErr): expected error from the error event, got nil")
	}
}

func TestStream_FromTokens_Ugly(t *testing.T) {
	// Empty token stream: no deltas, but still a text-done + usage so a
	// downstream consumer always sees a well-formed terminated stream.
	events := FromTokens(nil, Usage{PromptTokens: 2, TotalTokens: 2})
	if len(events) != 2 {
		t.Fatalf("FromTokens empty: got %d events, want 2 (done+usage)", len(events))
	}
	if events[0].Kind != KindTextDone {
		t.Fatalf("FromTokens empty: first event should be text-done, got %s", events[0].Kind)
	}
	if events[1].Kind != KindUsage {
		t.Fatalf("FromTokens empty: second event should be usage, got %s", events[1].Kind)
	}
	resp, err := Collect(events)
	if err != nil {
		t.Fatalf("Collect(empty FromTokens): unexpected error: %v", err)
	}
	if resp.Text != "" {
		t.Fatalf("Collect(empty FromTokens): expected empty text, got %q", resp.Text)
	}

	// An empty token that is genuinely empty string is still a delta — the
	// generator decides what a token is; FromTokens does not filter.
	one := FromTokens([]string{""}, Usage{})
	if len(one) != 3 || one[0].Kind != KindTextDelta {
		t.Fatalf("FromTokens empty-string token: got %d events, first %s", len(one), one[0].Kind)
	}
}

// --- Kind.String ---------------------------------------------------------

func TestStream_KindString_Good(t *testing.T) {
	// A Kind formats as its own wire key — the stable contract value used in
	// logs and metrics (§3.2). Spot-check the lifecycle and a delta kind.
	if got := KindTextDelta.String(); got != "text-delta" {
		t.Fatalf("KindTextDelta.String(): got %q, want %q", got, "text-delta")
	}
	if got := KindResponseCompleted.String(); got != "response-completed" {
		t.Fatalf("KindResponseCompleted.String(): got %q, want %q", got, "response-completed")
	}
	if got := KindUsage.String(); got != "usage" {
		t.Fatalf("KindUsage.String(): got %q, want %q", got, "usage")
	}
}

func TestStream_KindString_Ugly(t *testing.T) {
	// String is a plain cast, so even an unknown/zero Kind round-trips its raw
	// string value rather than panicking.
	if got := Kind("").String(); got != "" {
		t.Fatalf("empty Kind.String(): got %q, want empty", got)
	}
	if got := Kind("future-kind").String(); got != "future-kind" {
		t.Fatalf("unknown Kind.String(): got %q, want %q", got, "future-kind")
	}
}

// --- StreamError.Error ---------------------------------------------------

func TestStream_StreamError_Good(t *testing.T) {
	// A code + message renders "code: message"; a code-only error renders just
	// the code so it still reads cleanly when wrapped.
	withMsg := &StreamError{Code: "rate_limited", Message: "429 slow down"}
	if got := withMsg.Error(); got != "rate_limited: 429 slow down" {
		t.Fatalf("StreamError.Error() with message: got %q", got)
	}
	codeOnly := &StreamError{Code: "internal"}
	if got := codeOnly.Error(); got != "internal" {
		t.Fatalf("StreamError.Error() code only: got %q, want %q", got, "internal")
	}
}

func TestStream_StreamError_Ugly(t *testing.T) {
	// A nil *StreamError renders the empty string rather than panicking — it is
	// safe to format an absent error.
	var e *StreamError
	if got := e.Error(); got != "" {
		t.Fatalf("nil StreamError.Error(): got %q, want empty", got)
	}
}

// --- Assembler.Add: structural and edge branches -------------------------

func TestStream_Add_Good(t *testing.T) {
	// Annotations with a payload are collected; a KindAnnotationAdded with a nil
	// Annotation is silently ignored (it carries nothing to attach). A bare
	// function-call-args-done with no preceding delta still registers the call.
	a := NewAssembler()
	if err := a.Add(Event{Kind: KindAnnotationAdded, Annotation: &Annotation{Title: "Go", URL: "https://go.dev"}}); err != nil {
		t.Fatalf("Add annotation: unexpected error: %v", err)
	}
	if err := a.Add(Event{Kind: KindAnnotationAdded, Annotation: nil}); err != nil {
		t.Fatalf("Add nil annotation: unexpected error: %v", err)
	}
	// A done with an id but no prior delta creates the slot (id/name affirmed
	// only on the done marker).
	if err := a.Add(Event{Kind: KindFunctionCallArgsDone, ToolCallID: "late", ToolName: "tardy"}); err != nil {
		t.Fatalf("Add done-without-delta: unexpected error: %v", err)
	}
	// Refusal deltas concatenate onto the Refusal field, just like text.
	if err := a.Add(Event{Kind: KindRefusalDelta, Text: "I can't "}); err != nil {
		t.Fatalf("Add refusal delta: unexpected error: %v", err)
	}
	if err := a.Add(Event{Kind: KindRefusalDelta, Text: "help with that"}); err != nil {
		t.Fatalf("Add refusal delta: unexpected error: %v", err)
	}

	resp := a.Result()
	if resp.Refusal != "I can't help with that" {
		t.Fatalf("Add refusal: got %q", resp.Refusal)
	}
	if len(resp.Annotations) != 1 {
		t.Fatalf("Add annotations: got %d, want 1 (nil ignored)", len(resp.Annotations))
	}
	if resp.Annotations[0].Title != "Go" || resp.Annotations[0].URL != "https://go.dev" {
		t.Fatalf("Add annotation payload: got %+v", resp.Annotations[0])
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != "late" || resp.ToolCalls[0].Name != "tardy" {
		t.Fatalf("Add done-without-delta tool call: got %+v", resp.ToolCalls)
	}
}

func TestStream_Add_Bad(t *testing.T) {
	// A KindResponseCreated / -Completed with an EMPTY ResponseID must not
	// overwrite a previously-set id with "" — the empty-id guard is exercised
	// here. Created carries the id, then a completed with no id arrives.
	a := NewAssembler()
	if err := a.Add(Event{Kind: KindResponseCreated, ResponseID: "resp-9"}); err != nil {
		t.Fatalf("Add created: unexpected error: %v", err)
	}
	if err := a.Add(Event{Kind: KindResponseCompleted}); err != nil { // no ResponseID
		t.Fatalf("Add completed (no id): unexpected error: %v", err)
	}
	resp := a.Result()
	if resp.ResponseID != "resp-9" {
		t.Fatalf("Add empty-id completed must keep the prior id: got %q", resp.ResponseID)
	}
	if !resp.Completed {
		t.Fatal("Add completed: expected Completed true")
	}

	// And the inverse: a created event with an empty id leaves the id unset.
	b := NewAssembler()
	if err := b.Add(Event{Kind: KindResponseCreated}); err != nil {
		t.Fatalf("Add created (no id): unexpected error: %v", err)
	}
	if got := b.Result().ResponseID; got != "" {
		t.Fatalf("Add created with empty id: got %q, want empty", got)
	}
}

func TestStream_Add_Ugly(t *testing.T) {
	// The structural/closing markers (text-done, content-part-added/-done,
	// reasoning-done, refusal-done) hit the default arm: they refine a live view
	// but do not change the assembled value. Feeding only those yields a zero
	// response with no error.
	a := NewAssembler()
	markers := []Event{
		{Kind: KindTextDone},
		{Kind: KindContentPartAdded, PartIndex: 0, PartType: "text"},
		{Kind: KindContentPartDone, PartIndex: 0},
		{Kind: KindReasoningDone},
		{Kind: KindRefusalDone},
	}
	for _, ev := range markers {
		if err := a.Add(ev); err != nil {
			t.Fatalf("Add structural marker %s: unexpected error: %v", ev.Kind, err)
		}
	}
	resp := a.Result()
	if resp.Text != "" || resp.Reasoning != "" || resp.Refusal != "" || len(resp.ToolCalls) != 0 {
		t.Fatalf("Add structural-only: expected zero response, got %+v", resp)
	}

	// An argument delta with NO ToolCallID folds into a single anonymous "_"
	// call rather than being dropped — exercise the empty-id branch of
	// appendToolArgs. A later delta, also id-less, appends to the same call.
	b := NewAssembler()
	_ = b.Add(Event{Kind: KindFunctionCallArgsDelta, Text: `{"a":`})
	_ = b.Add(Event{Kind: KindFunctionCallArgsDelta, Text: `1}`})
	rb := b.Result()
	if len(rb.ToolCalls) != 1 {
		t.Fatalf("anonymous tool call: got %d calls, want 1", len(rb.ToolCalls))
	}
	if rb.ToolCalls[0].Arguments != `{"a":1}` {
		t.Fatalf("anonymous tool call args: got %q", rb.ToolCalls[0].Arguments)
	}

	// ensureTool's name-fill-on-existing branch: a first delta sets a blank
	// name, a later delta (or done) for the same id supplies it.
	c := NewAssembler()
	_ = c.Add(Event{Kind: KindFunctionCallArgsDelta, ToolCallID: "x", Text: "1"}) // no name yet
	_ = c.Add(Event{Kind: KindFunctionCallArgsDelta, ToolCallID: "x", ToolName: "named", Text: "2"})
	// A redundant non-empty name on the same id must NOT overwrite the one set.
	_ = c.Add(Event{Kind: KindFunctionCallArgsDone, ToolCallID: "x", ToolName: "ignored"})
	rc := c.Result()
	if len(rc.ToolCalls) != 1 || rc.ToolCalls[0].Name != "named" {
		t.Fatalf("ensureTool name fill: got %+v", rc.ToolCalls)
	}
	if rc.ToolCalls[0].Arguments != "12" {
		t.Fatalf("ensureTool args after name fill: got %q", rc.ToolCalls[0].Arguments)
	}
}

// --- Assembler error event with no payload -------------------------------

func TestStream_AddError_Ugly(t *testing.T) {
	// A terminal error event with a NIL Err still fails closed: Add returns a
	// non-nil error (the "stream failed" core.E), and Result carries no stream
	// error payload. This exercises streamErr's a.failErr == nil branch.
	a := NewAssembler()
	err := a.Add(Event{Kind: KindError, Err: nil})
	if err == nil {
		t.Fatal("Add error event with nil Err: expected a non-nil error, got nil")
	}

	// Collect surfaces the same nil-payload failure as an error, with the
	// partial response returned (its Err nil).
	resp, cerr := Collect([]Event{
		{Kind: KindTextDelta, Text: "partial"},
		{Kind: KindResponseFailed}, // no Err payload
	})
	if cerr == nil {
		t.Fatal("Collect with payload-less response-failed: expected error, got nil")
	}
	if resp.Err != nil {
		t.Fatalf("Collect: expected nil stream-error payload, got %+v", resp.Err)
	}
}
