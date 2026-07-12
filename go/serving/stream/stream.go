// SPDX-Licence-Identifier: EUPL-1.2

// Package stream is the streaming event taxonomy and assembler for the
// inference stack. It defines ONE typed event shape that both a local engine
// token stream and a remote SSE provider stream produce, and an assembler that
// folds a sequence of those events back into a single final response.
//
// A caller consumes one Event stream whether the tokens come from on-device
// inference or a remote provider. The taxonomy covers text, content parts,
// tool-call arguments, reasoning, refusals, annotations, the response
// lifecycle, a trailing usage frame, and an error event.
//
//	// Assemble a streamed response into its final form:
//	resp, err := stream.Collect(events)
//	if err != nil { return err }
//	use(resp.Text, resp.ToolCalls, resp.Usage)
//
//	// Make the unified event sequence from a local token stream so the same
//	// consumer handles local and remote identically:
//	events := stream.FromTokens(tokens, usage)
package stream

import core "dappco.re/go"

// Kind names an event in the streaming taxonomy (§6.5). The string values are
// the stable wire keys — a remote SSE stream emits them and the assembler keys
// on them, so they are part of the contract.
//
//	core.Println(stream.KindTextDelta.String()) // "text-delta"
type Kind string

const (
	// Text — the assistant's visible answer, streamed as deltas then closed.
	KindTextDelta Kind = "text-delta"
	KindTextDone  Kind = "text-done"

	// Content parts — multimodal output blocks (text/image/audio) opening and
	// closing around their deltas (§6.12).
	KindContentPartAdded Kind = "content-part-added"
	KindContentPartDone  Kind = "content-part-done"

	// Function-call arguments — a tool call's JSON arguments, streamed as
	// deltas keyed by ToolCallID, then closed (§6.4).
	KindFunctionCallArgsDelta Kind = "function-call-args-delta"
	KindFunctionCallArgsDone  Kind = "function-call-args-done"

	// Reasoning — a reasoning model's thinking, streamed as deltas then closed.
	KindReasoningDelta Kind = "reasoning-delta"
	KindReasoningDone  Kind = "reasoning-done"

	// Refusal — a streamed refusal message, deltas then closed (§6.18).
	KindRefusalDelta Kind = "refusal-delta"
	KindRefusalDone  Kind = "refusal-done"

	// Annotation — a search citation attached to the response (§6.8 rerank /
	// web-search server tool).
	KindAnnotationAdded Kind = "annotation-added"

	// Usage — the trailing token + cost accounting frame (§6.6), requested via
	// stream_options.
	KindUsage Kind = "usage"

	// Error — a stream-level failure (§6.7). Carries a StreamError in Err.
	KindError Kind = "error"

	// Response lifecycle — the outer envelope of one generation (§6.5).
	KindResponseCreated   Kind = "response-created"
	KindResponseCompleted Kind = "response-completed"
	KindResponseFailed    Kind = "response-failed"
)

// String returns the Kind's wire key — its own string value — so a Kind formats
// stably in logs and metrics (§3.2).
func (k Kind) String() string { return string(k) }

// Usage is the token + cost accounting frame (§6.6). It is carried on a
// KindUsage event and reconciled into the final Response. Counts are absolute
// totals for the generation, not per-delta increments.
//
//	stream.Usage{PromptTokens: 10, CompletionTokens: 3, TotalTokens: 13}
type Usage struct {
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	ReasoningTokens  int     `json:"reasoning_tokens,omitempty"`
	CachedTokens     int     `json:"cached_tokens,omitempty"`
	Cost             float64 `json:"cost,omitempty"`
}

// StreamError is the payload of a KindError or KindResponseFailed event (§6.7).
// Code is the typed failure class (e.g. "rate_limited", "provider_overloaded",
// "internal"); Message is the human-readable detail. It implements error so a
// failed stream surfaces directly.
//
//	stream.StreamError{Code: "rate_limited", Message: "429 slow down"}
type StreamError struct {
	Code    string `json:"code"`
	Message string `json:"message,omitempty"`
}

// Error renders the stream error as "code: message" (or just the code when the
// message is empty), so it reads cleanly when wrapped by core.E.
func (e *StreamError) Error() string {
	if e == nil {
		return ""
	}
	if e.Message == "" {
		return e.Code
	}
	return e.Code + ": " + e.Message
}

// Annotation is a search citation attached to the response (§6.8), carried on a
// KindAnnotationAdded event and collected onto the final Response.
type Annotation struct {
	Title string `json:"title,omitempty"`
	URL   string `json:"url,omitempty"`
}

// Event is the single typed shape for every step in a streamed generation
// (§6.5). Kind selects the meaning; the optional fields below carry only what
// that Kind needs — Text for text/reasoning/refusal/argument deltas, the
// ToolCall* fields for function-call deltas, Usage for the usage frame, Err for
// failures, ResponseID for lifecycle events. One struct keeps local (go-mlx)
// and remote (SSE) streams producing an identical sequence.
//
//	stream.Event{Kind: stream.KindTextDelta, Text: "Hello"}
//	stream.Event{Kind: stream.KindUsage, Usage: stream.Usage{TotalTokens: 13}}
type Event struct {
	Kind Kind `json:"kind"`

	// Text carries the delta payload for text, reasoning, refusal, and
	// function-call-argument deltas — its meaning follows Kind.
	Text string `json:"text,omitempty"`

	// ToolCallID / ToolName identify the function call a
	// function-call-args-delta / -done belongs to (§6.4). ToolName is typically
	// set on the first delta of a call.
	ToolCallID string `json:"tool_call_id,omitempty"`
	ToolName   string `json:"tool_name,omitempty"`

	// PartIndex / PartType describe the content part a content-part-added /
	// -done event opens or closes (§6.12).
	PartIndex int    `json:"part_index,omitempty"`
	PartType  string `json:"part_type,omitempty"`

	// Annotation carries a citation for a KindAnnotationAdded event (§6.8).
	Annotation *Annotation `json:"annotation,omitempty"`

	// Usage carries the trailing accounting frame for a KindUsage event (§6.6).
	Usage Usage `json:"usage"`

	// Err carries the failure for a KindError / KindResponseFailed event
	// (§6.7).
	Err *StreamError `json:"error,omitempty"`

	// ResponseID identifies the generation on lifecycle events
	// (created / completed / failed).
	ResponseID string `json:"response_id,omitempty"`
}

// ToolCall is one assembled function call (§6.4): its id, name, and the full
// JSON arguments string concatenated from the call's argument deltas.
//
//	tc.ID, tc.Name, tc.Arguments // "call-1", "search", `{"q":"go"}`
type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`
}

// Response is the assembled result of a streamed generation — the running state
// an Assembler builds up and the value Collect returns. Text, Reasoning, and
// Refusal are the concatenated deltas; ToolCalls are the collected function
// calls in first-seen order; Usage is the final accounting; Completed records
// whether a response-completed lifecycle event arrived; Err carries a stream
// failure (also returned as an error from Collect).
type Response struct {
	ResponseID  string       `json:"response_id,omitempty"`
	Text        string       `json:"text"`
	Reasoning   string       `json:"reasoning,omitempty"`
	Refusal     string       `json:"refusal,omitempty"`
	ToolCalls   []ToolCall   `json:"tool_calls,omitempty"`
	Annotations []Annotation `json:"annotations,omitempty"`
	Usage       Usage        `json:"usage"`
	Completed   bool         `json:"completed"`
	Err         *StreamError `json:"error,omitempty"`
}

// Assembler folds a sequence of Events into a Response, exposing the running
// state as it goes (§6.5). Construct it with NewAssembler, feed events with
// Add, and read the accumulated Response with Result. Add reports a failure
// event so a caller streaming live can stop early; Collect wraps the whole loop
// for the common case.
//
//	a := stream.NewAssembler()
//	for ev := range ch {
//	    if err := a.Add(ev); err != nil { break }
//	}
//	resp := a.Result()
type Assembler struct {
	resp    Response
	tools   []ToolCall
	toolIdx map[string]int // ToolCallID → index into tools, for interleaved calls
	text    core.Builder   // text-delta payloads, folded as they arrive
	reason  core.Builder   // reasoning-delta payloads, folded as they arrive
	refuse  core.Builder   // refusal-delta payloads, folded as they arrive
	failed  bool
	failErr *StreamError
}

// NewAssembler returns an empty Assembler ready to consume events.
//
//	a := stream.NewAssembler()
func NewAssembler() *Assembler {
	// toolIdx is left nil and created lazily on the first tool call
	// (ensureTool) — a pure-text stream never allocates the map at all.
	return &Assembler{}
}

// Add folds one event into the running state. It returns a non-nil error only
// for a terminal failure event (KindError / KindResponseFailed) so a live
// consumer can stop the stream; every other event returns nil. The error is
// also retained on the Response (Err), so Result/Collect surface it even if the
// caller ignores Add's return.
//
//	if err := a.Add(ev); err != nil { return a.Result(), err }
func (a *Assembler) Add(ev Event) error {
	switch ev.Kind {
	case KindTextDelta:
		a.text.WriteString(ev.Text)
	case KindReasoningDelta:
		a.reason.WriteString(ev.Text)
	case KindRefusalDelta:
		a.refuse.WriteString(ev.Text)
	case KindFunctionCallArgsDelta:
		a.appendToolArgs(ev)
	case KindFunctionCallArgsDone:
		// Closing marker — the call's id/name may be (re)affirmed here; ensure
		// the slot exists so a done with no preceding delta still registers.
		if ev.ToolCallID != "" {
			a.ensureTool(ev.ToolCallID, ev.ToolName)
		}
	case KindAnnotationAdded:
		if ev.Annotation != nil {
			a.resp.Annotations = append(a.resp.Annotations, *ev.Annotation)
		}
	case KindUsage:
		a.resp.Usage = ev.Usage
	case KindResponseCreated:
		if ev.ResponseID != "" {
			a.resp.ResponseID = ev.ResponseID
		}
	case KindResponseCompleted:
		a.resp.Completed = true
		if ev.ResponseID != "" {
			a.resp.ResponseID = ev.ResponseID
		}
	case KindError, KindResponseFailed:
		a.failed = true
		a.failErr = ev.Err
		a.resp.Err = ev.Err
		return a.streamErr()
	default:
		// text-done, content-part-added/done, reasoning-done, refusal-done —
		// closing/structural markers the final Response does not need to hold.
		// They refine a live view but do not change the assembled value.
	}
	return nil
}

// appendToolArgs routes an argument delta to its tool call's buffer, keyed by
// ToolCallID, so interleaved calls keep separate argument strings (§6.4).
func (a *Assembler) appendToolArgs(ev Event) {
	id := ev.ToolCallID
	if id == "" {
		// No id to key on — fold into a single anonymous call so the arguments
		// are not silently dropped.
		id = "_"
	}
	i := a.ensureTool(id, ev.ToolName)
	a.tools[i].Arguments += ev.Text
}

// ensureTool returns the index of the tool call with id, creating it (in
// first-seen order) if absent. A non-empty name fills a blank name without
// overwriting one already set on the first delta.
func (a *Assembler) ensureTool(id, name string) int {
	// A read from a nil map is valid (yields the zero index, ok=false), so the
	// lookup is safe before the map exists; only the write below needs it.
	if i, ok := a.toolIdx[id]; ok {
		if name != "" && a.tools[i].Name == "" {
			a.tools[i].Name = name
		}
		return i
	}
	a.tools = append(a.tools, ToolCall{ID: id, Name: name})
	i := len(a.tools) - 1
	if a.toolIdx == nil {
		a.toolIdx = make(map[string]int)
	}
	a.toolIdx[id] = i
	return i
}

// streamErr wraps the retained stream failure as a core.E error (scope "ai"),
// or returns nil if the failure carried no payload detail beyond its class.
func (a *Assembler) streamErr() error {
	if a.failErr == nil {
		return core.E("ai", "stream failed", nil)
	}
	return core.E("ai", "stream failed: "+a.failErr.Error(), nil)
}

// Result returns the assembled Response from the events seen so far. It is safe
// to call at any point — mid-stream for a running view, or at the end for the
// final value. Add folds each delta into a builder as it arrives, so reading
// the assembled text here is a zero-copy view of the folded buffer rather than
// a per-call join of the delta slices.
//
//	resp := a.Result()
func (a *Assembler) Result() Response {
	r := a.resp
	r.Text = a.text.String()
	r.Reasoning = a.reason.String()
	r.Refusal = a.refuse.String()
	if len(a.tools) > 0 {
		r.ToolCalls = a.tools
	}
	return r
}

// Collect folds a whole event sequence into its final Response (§6.5). It is
// the batch form of Assembler: feed it the full stream and read the result. A
// KindError or KindResponseFailed event yields a non-nil error (with the
// assembled-so-far Response also returned, its Err set), so a caller can both
// branch on failure and inspect the partial output. An empty sequence yields a
// zero Response and no error.
//
//	resp, err := stream.Collect(events)
//	if err != nil { return err }
//	use(resp.Text, resp.ToolCalls, resp.Usage)
func Collect(events []Event) (Response, error) {
	a := NewAssembler()
	a.presize(events)
	for _, ev := range events {
		if err := a.Add(ev); err != nil {
			return a.Result(), err
		}
	}
	return a.Result(), nil
}

// presize reserves the exact byte budget for the delta accumulators from the
// full event sequence Collect holds, so each builder allocates its backing once
// and the Add loop never grows it. Live streaming (Add fed one event at a time)
// cannot do this — it has no length hint — but the batch path sums every
// delta's bytes up front. The totals are exact, so no accumulator is
// over-allocated. Indexing events (not ranging by value) avoids copying the
// large Event struct per element.
func (a *Assembler) presize(events []Event) {
	var nText, nReason, nRefuse int
	for i := range events {
		switch events[i].Kind {
		case KindTextDelta:
			nText += len(events[i].Text)
		case KindReasoningDelta:
			nReason += len(events[i].Text)
		case KindRefusalDelta:
			nRefuse += len(events[i].Text)
		}
	}
	if nText > 0 {
		a.text.Grow(nText)
	}
	if nReason > 0 {
		a.reason.Grow(nReason)
	}
	if nRefuse > 0 {
		a.refuse.Grow(nRefuse)
	}
}

// FromTokens builds the common unified event sequence from a plain token stream
// (§6.5): one KindTextDelta per token, a single KindTextDone, then the trailing
// KindUsage frame (§6.6). A local go-mlx token stream and a remote SSE stream
// both produce this same Event sequence, so one consumer handles both. An empty
// token slice still emits text-done + usage, so the stream is always
// well-formed and terminated.
//
//	events := stream.FromTokens([]string{"Hello", " world"}, usage)
//	resp, _ := stream.Collect(events) // resp.Text == "Hello world"
func FromTokens(tokens []string, usage Usage) []Event {
	return FromTokensErr(tokens, usage, nil)
}

// FromTokensErr is FromTokens with a terminating generator error: it emits a
// KindTextDelta per token produced before the failure, then — if genErr is
// non-nil — a single KindError event instead of the text-done + usage frames,
// so a local generation failure reaches the unified consumer exactly like a
// remote one (§6.7). A nil genErr behaves identically to FromTokens.
//
//	events := stream.FromTokensErr(partial, stream.Usage{}, decodeErr)
func FromTokensErr(tokens []string, usage Usage, genErr error) []Event {
	events := make([]Event, 0, len(tokens)+2)
	for _, tok := range tokens {
		events = append(events, Event{Kind: KindTextDelta, Text: tok})
	}
	if genErr != nil {
		events = append(events, Event{
			Kind: KindError,
			Err:  &StreamError{Code: "generation_failed", Message: genErr.Error()},
		})
		return events
	}
	events = append(events, Event{Kind: KindTextDone})
	events = append(events, Event{Kind: KindUsage, Usage: usage})
	return events
}
