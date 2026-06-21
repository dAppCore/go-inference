// SPDX-Licence-Identifier: EUPL-1.2

package transform

import (
	core "dappco.re/go"
	chat "dappco.re/go/inference/chat"
)

// fakeCounter sizes a conversation the way budget tests stub the tokeniser: the
// real tokeniser lives in go-mlx, so the middle-out logic is tested against a
// stub. Here one message costs a fixed number of tokens regardless of content,
// so a window expressed in "messages" is easy to reason about — a window of 30
// with a per-message cost of 10 holds exactly three messages.
//
//	MiddleOut(msgs, perMessage(10), 30)
type perMessage int

func (f perMessage) Count(messages []chat.Message) int { return len(messages) * int(f) }

// lenCounter sizes a conversation by the summed length of its content, the other
// shape the spec calls out — used to prove the transform is content-sensitive,
// not only count-sensitive. It measures the canonical message text via Text().
//
//	MiddleOut(msgs, lenCounter{}, 40)
type lenCounter struct{}

func (lenCounter) Count(messages []chat.Message) int {
	total := 0
	for _, m := range messages {
		total += len(m.Text())
	}
	return total
}

// sys, user and asst build canonical chat.Message turns carrying a single text
// block — the conversation shape the transform now reasons over.
//
//	sys("be terse") // chat.Message{Role: chat.System, Content: [chat.Text("be terse")]}
func sys(content string) chat.Message  { return msg(chat.System, content) }
func user(content string) chat.Message { return msg(chat.User, content) }
func asst(content string) chat.Message { return msg(chat.Assistant, content) }

func msg(role chat.Role, content string) chat.Message {
	return chat.Message{Role: role, Content: []chat.ContentBlock{chat.Text(content)}}
}

// TestTransform_MiddleOut_Good — a conversation already inside the window is
// returned untouched (transformed=false), and an over-window conversation is
// compressed by eliding the MIDDLE while the leading system message and the
// most-recent turns are preserved, until it fits (transformed=true).
func TestTransform_MiddleOut_Good(t *core.T) {
	// Already fits: three messages at 10 tokens each = 30, a 100-token window has
	// room to spare → unchanged, no transform, no error.
	fits := []chat.Message{sys("be terse"), user("hello"), asst("hi")}
	out, transformed, err := MiddleOut(fits, perMessage(10), 100)
	core.AssertNoError(t, err)
	core.AssertFalse(t, transformed, "a conversation already inside the window is untouched")
	core.AssertLen(t, out, 3, "no messages are dropped when it already fits")
	core.AssertEqual(t, "be terse", out[0].Text(), "the head is the original system message")
	core.AssertEqual(t, "hi", out[2].Text(), "the tail is the original last turn")

	// Over window: a leading system message + eight turns at 10 tokens each = 90
	// tokens against a 50-token window. The middle is elided into one placeholder;
	// the system head and the most-recent turns survive, and the result fits.
	long := []chat.Message{
		sys("be terse"),
		user("q1"), asst("a1"),
		user("q2"), asst("a2"),
		user("q3"), asst("a3"),
		user("q4"), asst("a4"),
	}
	out2, transformed2, err2 := MiddleOut(long, perMessage(10), 50)
	core.AssertNoError(t, err2)
	core.AssertTrue(t, transformed2, "an over-window conversation is compressed")
	core.AssertTrue(t, perMessage(10).Count(out2) <= 50, "the compressed conversation fits the window")
	core.AssertEqual(t, chat.System, out2[0].Role, "the leading system message is preserved as the head")
	core.AssertEqual(t, "be terse", out2[0].Text(), "the head content is the original system message")
	last := out2[len(out2)-1]
	core.AssertEqual(t, "a4", last.Text(), "the most-recent turn is preserved as the tail")

	// Exactly one elision placeholder sits between the head and the kept tail.
	placeholders := 0
	for _, m := range out2 {
		if m.Role == PlaceholderRole {
			placeholders++
		}
	}
	core.AssertEqual(t, 1, placeholders, "the elided middle is a single placeholder message")
}

// TestTransform_MiddleOut_Placeholder — the placeholder reports how many turns
// were dropped, and the count is accurate against the input minus what survives.
func TestTransform_MiddleOut_Placeholder(t *core.T) {
	long := []chat.Message{
		sys("be terse"),
		user("q1"), asst("a1"),
		user("q2"), asst("a2"),
		user("q3"), asst("a3"),
		user("q4"), asst("a4"),
	}
	out, transformed, err := MiddleOut(long, perMessage(10), 50)
	core.AssertNoError(t, err)
	core.AssertTrue(t, transformed)

	// Reconstruct the dropped count: original turns minus the kept (non-placeholder)
	// turns equals the number the placeholder must report.
	kept := 0
	var note string
	for _, m := range out {
		if m.Role == PlaceholderRole {
			note = m.Text()
			continue
		}
		kept++
	}
	dropped := len(long) - kept
	core.AssertTrue(t, dropped > 0, "at least one middle turn was dropped")
	core.AssertContains(t, note, core.Itoa(dropped), "the placeholder names how many turns were elided")
	core.AssertContains(t, note, "elided", "the placeholder reads as an elision note")
}

// TestTransform_MiddleOut_ContentSensitive — the same shape compresses under a
// length-based counter too, proving the transform measures via the injected
// Counter rather than assuming a fixed per-message cost. The window is sized to
// leave room for the elision placeholder (which has a real content cost under a
// length counter) plus the most-recent turn.
func TestTransform_MiddleOut_ContentSensitive(t *core.T) {
	thirty := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" // 30 chars
	long := []chat.Message{
		sys("system"),              // 6
		user(thirty), asst(thirty), // 60
		user(thirty), asst(thirty), // 60
		user("the final answer goes right here"), // 32
	}
	// 158 chars of content against a 120-char window → must compress; head (6) +
	// placeholder + the latest turn fits, the whole conversation does not.
	out, transformed, err := MiddleOut(long, lenCounter{}, 120)
	core.AssertNoError(t, err)
	core.AssertTrue(t, transformed, "over a length window, the middle is elided")
	core.AssertTrue(t, lenCounter{}.Count(out) <= 120, "the result fits the length window")
	core.AssertEqual(t, "system", out[0].Text(), "the system head survives")
	core.AssertEqual(t, "the final answer goes right here", out[len(out)-1].Text(), "the latest turn survives")
}

// TestTransform_MiddleOut_Bad — a conversation that cannot fit even when the
// middle is maximally elided (the protected head + the single most-recent turn
// already overflow) returns the best-effort compressed set PLUS the typed
// ErrCannotFit, so the caller can fall out to a longer-context endpoint (§6.2).
func TestTransform_MiddleOut_Bad(t *core.T) {
	// Head (system, 10) + every turn (10 each); even head + placeholder + one
	// tail turn is 30 tokens, but the window is 25 — irreducible.
	long := []chat.Message{
		sys("be terse"),
		user("q1"), asst("a1"),
		user("q2"), asst("a2"),
		user("q3"), asst("a3"),
	}
	out, transformed, err := MiddleOut(long, perMessage(10), 25)
	core.AssertError(t, err)
	core.AssertErrorIs(t, err, ErrCannotFit, "the failure is the typed ErrCannotFit")
	core.AssertTrue(t, transformed, "the best effort still counts as a transform")
	core.AssertNotEmpty(t, out, "a best-effort compressed set is still returned")
	// Best effort keeps the protected head and the latest turn even though they
	// overflow — the caller decides what to do with the too-big-but-minimal set.
	core.AssertEqual(t, chat.System, out[0].Role, "the protected head is kept in the best effort")
	core.AssertEqual(t, "a3", out[len(out)-1].Text(), "the most-recent turn is kept in the best effort")
}

// TestTransform_MiddleOut_BadHeadAlone — when the protected head alone overflows
// the window there is nothing left to elide; the head is returned with
// ErrCannotFit rather than an empty set.
func TestTransform_MiddleOut_BadHeadAlone(t *core.T) {
	// Two system messages at 10 each = 20 against a 15-token window. The head is
	// protected and already over — nothing to compress.
	msgs := []chat.Message{sys("rule one"), sys("rule two"), user("go")}
	out, transformed, err := MiddleOut(msgs, perMessage(10), 15)
	core.AssertErrorIs(t, err, ErrCannotFit, "an oversized protected head cannot fit")
	core.AssertTrue(t, transformed, "the attempt is a transform")
	core.AssertNotEmpty(t, out, "the head is still returned for the caller to route elsewhere")
}

// TestTransform_MiddleOut_Ugly — degenerate inputs: a non-positive window is a
// usage error, a nil counter fails closed (can't measure, can't compress), and
// an empty conversation is a no-op.
func TestTransform_MiddleOut_Ugly(t *core.T) {
	msgs := []chat.Message{sys("be terse"), user("hello")}

	// window <= 0 is a misuse — error, with the input handed back unchanged so the
	// caller doesn't lose the conversation.
	out, transformed, err := MiddleOut(msgs, perMessage(10), 0)
	core.AssertError(t, err)
	core.AssertErrorIs(t, err, ErrBadWindow, "the typed ErrBadWindow is returned")
	core.AssertFalse(t, transformed, "a usage error is not a transform")
	core.AssertLen(t, out, 2, "the input is returned unchanged on a usage error")

	_, _, errNeg := MiddleOut(msgs, perMessage(10), -100)
	core.AssertErrorIs(t, errNeg, ErrBadWindow, "a negative window is the same usage error")

	// nil counter — we can't size anything, so fail closed rather than guess.
	outNil, transformedNil, errNil := MiddleOut(msgs, nil, 100)
	core.AssertError(t, errNil)
	core.AssertErrorIs(t, errNil, ErrNoCounter, "the typed ErrNoCounter is returned")
	core.AssertFalse(t, transformedNil)
	core.AssertLen(t, outNil, 2, "the input is returned unchanged when it cannot be measured")

	// Empty conversation — nothing to fit, nothing to compress, no error.
	outEmpty, transformedEmpty, errEmpty := MiddleOut(nil, perMessage(10), 100)
	core.AssertNoError(t, errEmpty, "an empty conversation is a clean no-op")
	core.AssertFalse(t, transformedEmpty)
	core.AssertLen(t, outEmpty, 0, "an empty conversation stays empty")
}

// TestTransform_MiddleOut_Single — a single message is returned unchanged when
// it fits; when it overflows there is nothing to elide, so it comes back with
// ErrCannotFit (the head-is-the-tail edge).
func TestTransform_MiddleOut_Single(t *core.T) {
	one := []chat.Message{user("just one message")}

	// Fits: untouched.
	out, transformed, err := MiddleOut(one, perMessage(10), 100)
	core.AssertNoError(t, err)
	core.AssertFalse(t, transformed, "a single fitting message is untouched")
	core.AssertLen(t, out, 1)

	// Overflows: a lone message can't be split — best-effort is itself, with the
	// typed error.
	outBig, transformedBig, errBig := MiddleOut(one, perMessage(10), 5)
	core.AssertErrorIs(t, errBig, ErrCannotFit, "a single over-window message cannot fit")
	core.AssertTrue(t, transformedBig)
	core.AssertLen(t, outBig, 1, "the lone message is returned as the best effort")
}

// TestTransform_MiddleOut_NoSystemHead — a conversation with no leading system
// message still compresses: the head protection is "leading system messages",
// which is simply empty here, so the most-recent turns are kept and the older
// ones elided.
func TestTransform_MiddleOut_NoSystemHead(t *core.T) {
	long := []chat.Message{
		user("q1"), asst("a1"),
		user("q2"), asst("a2"),
		user("q3"), asst("a3"),
		user("q4"), asst("a4"),
	}
	out, transformed, err := MiddleOut(long, perMessage(10), 40)
	core.AssertNoError(t, err)
	core.AssertTrue(t, transformed, "no system head still compresses the middle")
	core.AssertTrue(t, perMessage(10).Count(out) <= 40, "the result fits")
	core.AssertEqual(t, "a4", out[len(out)-1].Text(), "the latest turn is preserved")
	// First message is either the placeholder or a kept recent turn — never an
	// elided older one (q1/a1 are gone).
	core.AssertNotEqual(t, "q1", out[0].Text(), "the oldest turn is elided when there is no system head")
}

// TestTransform_MiddleOut_Deterministic — the same input yields byte-identical
// output across repeated calls (no map iteration, no clock, no randomness).
func TestTransform_MiddleOut_Deterministic(t *core.T) {
	long := []chat.Message{
		sys("be terse"),
		user("q1"), asst("a1"),
		user("q2"), asst("a2"),
		user("q3"), asst("a3"),
		user("q4"), asst("a4"),
	}
	a, ta, ea := MiddleOut(long, perMessage(10), 50)
	b, tb, eb := MiddleOut(long, perMessage(10), 50)
	core.AssertNoError(t, ea)
	core.AssertNoError(t, eb)
	core.AssertEqual(t, ta, tb, "the transform flag is deterministic")
	core.AssertLen(t, b, len(a), "the same input yields the same length")
	for i := range a {
		core.AssertEqual(t, a[i].Role, b[i].Role, "roles match across runs")
		core.AssertEqual(t, a[i].Text(), b[i].Text(), "contents match across runs")
	}

	// The input slice is never mutated — the caller's conversation is left intact.
	core.AssertLen(t, long, 9, "the original conversation is not mutated")
	core.AssertEqual(t, "q1", long[1].Text(), "the original middle is still present in the input")
}
