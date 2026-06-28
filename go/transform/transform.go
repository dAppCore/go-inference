// SPDX-Licence-Identifier: EUPL-1.2

// Package transform fits a conversation to a model's context window by
// compressing its middle (RFC §6.11 "Message transforms", §6.13). When a
// prompt exceeds the chosen endpoint's window — the common case when the same
// conversation routes between a long-context M3-Ultra model and a shorter-context
// 16 GB-GPU model — MiddleOut elides the oldest middle turns while always keeping
// the leading system instruction and the most-recent turns, so the request still
// fits without losing either the standing instructions or the live thread.
//
// It is budget's natural sibling: budget (§6.13) decides a request needs a
// transform (budget.DecisionNeedsTransform); this package performs it. The real
// tokeniser lives in go-mlx (locally) or the provider's encoding (remotely), so a
// Counter is injected and the logic stays pure arithmetic over message slices.
//
//	out, transformed, err := transform.MiddleOut(messages, mlxCounter, window)
//	if err != nil { /* §6.2: route to a roomier endpoint, or fall out to a provider */ }
//	if transformed { /* the middle was elided to make it fit */ }
//	place(out)
package transform

import (
	core "dappco.re/go"
	chat "dappco.re/go/inference/chat"
)

// Counter returns the prompt-token total for messages under the active model's
// tokeniser — go-mlx locally, the provider's encoding remotely (§6.13). It is the
// only piece the transform borrows from a real model; everything else is slice
// arithmetic, so tests inject a fake (a fixed-per-message or content-length stub).
//
//	type mlxCounter struct{ /* … */ }
//	func (mlxCounter) Count(m []chat.Message) int { /* … */ }
type Counter interface {
	Count(messages []chat.Message) int
}

// PlaceholderRole is the role stamped on the single message that replaces the
// elided middle span, so a caller (or the provider-translation layer, §6.14) can
// recognise and style it distinctly from real turns. It is part of the contract:
// a distinct chat.Role marking the synthetic elision turn, not a real author.
//
//	out[i].Role == transform.PlaceholderRole // this is the elision note
const PlaceholderRole chat.Role = "system.elision"

// ErrBadWindow is the typed error MiddleOut returns for a non-positive window —
// a usage error (you cannot fit a conversation into zero or negative tokens).
// The input is handed back unchanged so the caller never loses the conversation.
//
//	if core.Is(err, transform.ErrBadWindow) { /* fix the window, don't retry */ }
var ErrBadWindow = core.E("transform", "window must be positive", nil)

// ErrNoCounter is the typed error MiddleOut returns when no Counter is supplied:
// without a tokeniser the conversation cannot be sized, so the transform fails
// closed rather than guessing a fit (mirrors budget.New(nil) failing closed).
var ErrNoCounter = core.E("transform", "no token counter supplied", nil)

// ErrCannotFit is the typed error MiddleOut returns when even maximal
// compression — the protected head plus a single most-recent turn plus the
// elision placeholder — still overflows the window. The best-effort compressed
// set is returned alongside it, so the caller can fall out to a longer-context
// endpoint or a provider (§6.2) with the smallest viable conversation in hand.
//
//	out, _, err := transform.MiddleOut(msgs, counter, window)
//	if core.Is(err, transform.ErrCannotFit) { routeToLongerContext(out) }
var ErrCannotFit = core.E("transform", "conversation cannot fit window even fully compressed", nil)

// MiddleOut fits messages to window by eliding the middle of the conversation
// (§6.11). Behaviour:
//
//   - window <= 0 → (messages, false, ErrBadWindow): a usage error; input unchanged.
//   - counter == nil → (messages, false, ErrNoCounter): can't measure; fail closed.
//   - already fits (Count(messages) <= window) → (messages, false, nil): untouched.
//   - over window → keep the leading system message(s) (the protected head) and the
//     most-recent turns (the tail), replace the elided middle span with ONE
//     placeholder message noting how many turns were dropped, and shrink the kept
//     tail until the result fits → (compressed, true, nil).
//   - cannot fit even at maximal compression (head + placeholder + one tail turn
//     still overflows) → (best-effort compressed, true, ErrCannotFit).
//
// Deterministic: no maps, clock, or randomness — the same input always yields the
// same output. The input slice is never mutated; a fresh slice is returned.
//
//	out, transformed, err := transform.MiddleOut(msgs, counter, 8192)
func MiddleOut(messages []chat.Message, counter Counter, window int) ([]chat.Message, bool, error) {
	if window <= 0 {
		return messages, false, ErrBadWindow
	}
	if counter == nil {
		return messages, false, ErrNoCounter
	}
	// Nothing to fit — a clean no-op (also dodges an empty-tail edge below).
	if len(messages) == 0 {
		return messages, false, nil
	}
	// Already inside the window — return untouched, no transform.
	if counter.Count(messages) <= window {
		return messages, false, nil
	}

	// Split off the protected head: the leading run of system/developer turns
	// (standing instructions, never elided). Everything after is the body, whose
	// middle is the elision candidate and whose end is the live thread.
	headLen := leadingHeadLen(messages)
	head := messages[:headLen]
	body := messages[headLen:]

	// With one body turn or fewer there is no middle to elide — the smallest set
	// is head + body itself. If that already overflowed (it did, or we'd have
	// returned above), it's the best effort and it cannot fit.
	if len(body) <= 1 {
		best := concat(head, body)
		return best, true, ErrCannotFit
	}

	// Keep the largest recent tail that fits: try the biggest first and shrink,
	// so the result retains as much live context as the window allows. tail spans
	// [1, len(body)-1] — at least one recent turn kept, at least one middle turn
	// elided (tail == len(body) would be "no elision", already ruled out as
	// over-window).
	//
	// Every candidate is head + [placeholder] + tail; all but the one that fits
	// are built only to be measured and discarded. So the loop reuses a single
	// backing array (sized for the largest candidate) and a single placeholder
	// content cell across iterations rather than allocating a fresh candidate each
	// pass. The survivor is returned with its capacity clamped to its length, so
	// it is byte-identical to a freshly built slice.
	buf := make([]chat.Message, 0, len(head)+1+len(body))
	cell := make([]chat.ContentBlock, 1)
	for tail := len(body) - 1; tail >= 1; tail-- {
		dropped := len(body) - tail
		cell[0] = chat.Text(placeholderText(dropped))
		buf = append(buf[:0], head...)
		buf = append(buf, chat.Message{Role: PlaceholderRole, Content: cell})
		buf = append(buf, body[len(body)-tail:]...)
		if counter.Count(buf) <= window {
			return buf[:len(buf):len(buf)], true, nil
		}
	}

	// Maximal compression — head + placeholder + the single most-recent turn —
	// still overflows. buf already holds that smallest viable set from the final
	// iteration (tail == 1); return it as the best effort with the typed error,
	// so the caller routes it elsewhere (§6.2).
	return buf[:len(buf):len(buf)], true, ErrCannotFit
}

// leadingHeadLen counts the leading run of protected turns — consecutive system
// or developer messages at the start of the conversation. These carry standing
// instructions and are never elided. A conversation with no system preamble has
// a head length of 0, so its oldest turns become the elision candidates instead.
func leadingHeadLen(messages []chat.Message) int {
	n := 0
	for _, m := range messages {
		if m.Role == chat.System || m.Role == chat.Developer {
			n++
			continue
		}
		break
	}
	return n
}

// placeholderText is the elision note dropped into the middle of the
// conversation — deterministic and machine-greppable (it names the count and
// reads as an elision), so both a human reading the transcript and the
// provider-translation layer (§6.14) can recognise it.
//
//	placeholderText(5) // "[… 5 earlier turns elided to fit the context window …]"
func placeholderText(dropped int) string {
	return core.Sprintf("[… %d earlier turns elided to fit the context window …]", dropped)
}

// concat returns a + b as a fresh slice, never aliasing either input — so the
// caller's original conversation is left intact (the deterministic, no-mutation
// contract).
func concat(a, b []chat.Message) []chat.Message {
	out := make([]chat.Message, 0, len(a)+len(b))
	out = append(out, a...)
	out = append(out, b...)
	return out
}
