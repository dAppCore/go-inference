// SPDX-Licence-Identifier: EUPL-1.2

package budget

import (
	core "dappco.re/go"
	chat "dappco.re/go/inference/serving/chat"
)

// fakeCounter returns a fixed prompt total regardless of input — the real
// tokeniser lives in go-mlx, so the budgeting logic is tested against a stub
// the way welfare tests its scorer with an injected Hostility func.
//
//	b := New(fakeCounter(1200))
type fakeCounter int

func (f fakeCounter) Count(_ []chat.Message, _ string) int { return int(f) }

// textCounter sizes a prompt by summing the rune length of each message's text,
// reading it through the canonical chat.Message.Text helper — a stand-in for a
// real tokeniser that proves budgeting consumes chat.Message end to end.
//
//	c := textCounter{}
//	c.Count([]chat.Message{chat.UserText("abc")}, "any") // 3
type textCounter struct{}

func (textCounter) Count(messages []chat.Message, _ string) int {
	total := 0
	for _, m := range messages {
		total += len([]rune(m.Text()))
	}
	return total
}

// userMsg builds a single-text user turn for the budgeting scenarios.
//
//	userMsg("what is 2+2?")
func userMsg(text string) chat.Message {
	return chat.Message{Role: chat.User, Content: []chat.ContentBlock{chat.Text(text)}}
}

func TestBudget_FitsWindow_Good(t *core.T) {
	// Prompt + expected completion sit comfortably inside the window.
	core.AssertTrue(t, FitsWindow(1000, 512, 8192), "1512 of 8192 fits")

	// Exact boundary: prompt + completion == contextLen still fits (the window
	// is inclusive of its last token).
	core.AssertTrue(t, FitsWindow(7680, 512, 8192), "exactly 8192 of 8192 fits")
}

func TestBudget_FitsWindow_Bad(t *core.T) {
	// One token over the window does not fit.
	core.AssertFalse(t, FitsWindow(7681, 512, 8192), "8193 of 8192 overflows")

	// A huge prompt against a short 16 GB-GPU window overflows.
	core.AssertFalse(t, FitsWindow(40000, 256, 8192), "long prompt overflows short window")
}

func TestBudget_FitsWindow_Ugly(t *core.T) {
	// Degenerate inputs are treated as "does not fit" rather than panicking or
	// reporting a phantom fit — a zero/negative context window can hold nothing.
	core.AssertFalse(t, FitsWindow(10, 0, 0), "zero context holds nothing")
	core.AssertFalse(t, FitsWindow(10, 0, -8192), "negative context holds nothing")

	// Negative token counts are nonsense input — clamp to "does not fit".
	core.AssertFalse(t, FitsWindow(-5, -5, 8192), "negative counts do not fit")
}

func TestBudget_FitsMemory_Good(t *core.T) {
	// 1000 tokens * 4 bytes/token = 4000 bytes working set, well under a 96 GB
	// M3-Ultra-class budget.
	core.AssertTrue(t, FitsMemory(1000, 4, 96<<30), "4000 bytes fits a 96 GB budget")

	// Exact boundary: working set == device budget still fits.
	core.AssertTrue(t, FitsMemory(1000, 4, 4000), "exactly 4000 of 4000 fits")
}

func TestBudget_FitsMemory_Bad(t *core.T) {
	// Working set one byte over the device budget does not fit.
	core.AssertFalse(t, FitsMemory(1000, 4, 3999), "4000 over a 3999 budget")

	// A large working set against a 16 GB-GPU-class budget overflows.
	core.AssertFalse(t, FitsMemory(8_000_000_000, 4, 16<<30), "32 GB working set over 16 GB")
}

func TestBudget_FitsMemory_Ugly(t *core.T) {
	// Zero / negative device budget can hold nothing.
	core.AssertFalse(t, FitsMemory(10, 4, 0), "zero budget holds nothing")
	core.AssertFalse(t, FitsMemory(10, 4, -1), "negative budget holds nothing")

	// Non-positive bytes-per-token is unusable input — fail closed.
	core.AssertFalse(t, FitsMemory(10, 0, 1<<30), "zero bytes/token is unusable")
}

func TestBudget_Decide_Good(t *core.T) {
	// A 1200-token prompt + 512 completion fits an 8192 window, and its working
	// set fits the device budget → Fits, with the counted total surfaced. The
	// prompt is real chat.Messages summed through chat.Message.Text by
	// textCounter — proving budgeting consumes the canonical message end to end.
	msgs := []chat.Message{
		{Role: chat.System, Content: []chat.ContentBlock{chat.Text(core.Repeat("a", 200))}},
		userMsg(core.Repeat("b", 1000)),
	}
	b := New(textCounter{})
	ep := Endpoint{ContextLen: 8192, MemoryBudget: 96 << 30, BytesPerToken: 4}
	d := b.Decide(msgs, "gemma-4-31b", 512, ep)
	core.AssertEqual(t, DecisionFits, d.Decision)
	core.AssertEqual(t, 1200, d.PromptTokens, "the counted prompt total is reported")
	core.AssertTrue(t, d.FitsWindow)
	core.AssertTrue(t, d.FitsMemory)
}

func TestBudget_Decide_Bad(t *core.T) {
	// Over the window → NeedsTransform (compress the middle, §6.11) rather than
	// a hard reject — the conversation can still be made to fit.
	over := New(fakeCounter(40000))
	ep := Endpoint{ContextLen: 8192, MemoryBudget: 96 << 30, BytesPerToken: 4}
	d := over.Decide(nil, "qwen-q4", 256, ep)
	core.AssertEqual(t, DecisionNeedsTransform, d.Decision)
	core.AssertFalse(t, d.FitsWindow)

	// Fits the window but the working set overflows the device budget →
	// NeedsLargerEndpoint (route to a roomier device, §6.2/§6.16).
	heavy := New(fakeCounter(2000))
	tight := Endpoint{ContextLen: 8192, MemoryBudget: 4096, BytesPerToken: 4}
	d2 := heavy.Decide(nil, "gemma-4-e4b", 256, tight)
	core.AssertEqual(t, DecisionNeedsLargerEndpoint, d2.Decision)
	core.AssertTrue(t, d2.FitsWindow, "the window was fine; memory was not")
	core.AssertFalse(t, d2.FitsMemory)
}

func TestBudget_Decide_Ugly(t *core.T) {
	// Over BOTH the window and the device budget → Overflows: a transform alone
	// won't save it and no local device fits, so the caller must fall out to a
	// provider (§6.2 local-first, free-first fallback).
	huge := New(fakeCounter(40000))
	tiny := Endpoint{ContextLen: 8192, MemoryBudget: 4096, BytesPerToken: 4}
	d := huge.Decide(nil, "qwen-q4", 1024, tiny)
	core.AssertEqual(t, DecisionOverflows, d.Decision)
	core.AssertFalse(t, d.FitsWindow)
	core.AssertFalse(t, d.FitsMemory)

	// A degenerate endpoint (zero context) can never fit → Overflows, never a
	// phantom Fits.
	z := New(fakeCounter(10))
	d2 := z.Decide(nil, "broken", 0, Endpoint{ContextLen: 0, MemoryBudget: 0, BytesPerToken: 4})
	core.AssertEqual(t, DecisionOverflows, d2.Decision)

	// String() is stable for logging / metrics keys.
	core.AssertEqual(t, "fits", DecisionFits.String())
	core.AssertEqual(t, "needs_transform", DecisionNeedsTransform.String())
	core.AssertEqual(t, "needs_larger_endpoint", DecisionNeedsLargerEndpoint.String())
	core.AssertEqual(t, "overflows", DecisionOverflows.String())
}

func TestBudget_Decide_NilCounter(t *core.T) {
	// A Budget with no Counter is a misconfiguration — Decide fails closed to
	// Overflows so a missing tokeniser never green-lights a placement.
	b := New(nil)
	d := b.Decide(nil, "gemma", 128, Endpoint{ContextLen: 8192, MemoryBudget: 96 << 30, BytesPerToken: 4})
	core.AssertEqual(t, DecisionOverflows, d.Decision)
	core.AssertEqual(t, 0, d.PromptTokens)
}
