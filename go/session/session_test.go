// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	core "dappco.re/go"
	chat "dappco.re/go/inference/chat"
)

// userTurn builds a single-text user message — the common test fixture for one
// conversation turn now that the registry orders canonical chat messages.
//
//	resp, _ := m.Append(s.ID, userTurn("hello"))
func userTurn(text string) chat.Message {
	return chat.Message{Role: chat.User, Content: []chat.ContentBlock{chat.Text(text)}}
}

// assistantTurn builds a single-text assistant message fixture.
//
//	resp, _ := m.Append(s.ID, assistantTurn("second"))
func assistantTurn(text string) chat.Message {
	return chat.Message{Role: chat.Assistant, Content: []chat.ContentBlock{chat.Text(text)}}
}

// seqIDs returns a deterministic id generator yielding the supplied ids in
// order, so a test can assert exactly which session / response id is minted.
//
//	m := NewManager(NewMemoryStore(), WithIDGen(seqIDs("sess-1", "resp-1")))
func seqIDs(ids ...string) func() string {
	i := 0
	return func() string {
		id := ids[i%len(ids)]
		i++
		return id
	}
}

// fixedClock pins time so created/updated stamps are deterministic.
func fixedClock(t core.Time) func() core.Time {
	return func() core.Time { return t }
}

func TestSession_Continue_Good(t *core.T) {
	// Open a session, append a turn, mint a responseID, then resolve that id
	// back to the session WITHOUT resending the transcript — the registry hands
	// the full prior context straight back (0% replay).
	at := core.Now()
	m := NewManager(NewMemoryStore(),
		WithIDGen(seqIDs("sess-1", "resp-1")),
		WithClock(fixedClock(at)))

	sess := m.Open("lemma")
	core.AssertEqual(t, "sess-1", sess.ID)
	core.AssertEqual(t, "lemma", sess.Model)

	respID, err := m.Append(sess.ID, userTurn("hello"))
	core.AssertNoError(t, err)
	core.AssertEqual(t, "resp-1", respID)

	// Continue from the response id resolves to the same session with its turn.
	got, err := m.Continue(respID)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "sess-1", got.ID)
	core.AssertLen(t, got.Turns, 1)
	core.AssertEqual(t, chat.User, got.Turns[0].Role)
	core.AssertEqual(t, "hello", got.Turns[0].Text())
}

func TestSession_Continue_Bad(t *core.T) {
	// An unknown response id is a typed error, not a silent empty session.
	m := NewManager(NewMemoryStore())

	_, err := m.Continue("resp-does-not-exist")
	core.AssertError(t, err)
	core.AssertErrorIs(t, err, ErrNotFound)
}

func TestSession_Continue_Ugly(t *core.T) {
	// An empty response id errors rather than resolving anything.
	m := NewManager(NewMemoryStore())

	_, err := m.Continue("")
	core.AssertError(t, err)
}

func TestSession_Append_Good(t *core.T) {
	// Turns accumulate in order and each append mints a fresh responseID that
	// advances — the latest responseID always points at the latest position.
	m := NewManager(NewMemoryStore(),
		WithIDGen(seqIDs("sess-1", "resp-1", "resp-2")))

	sess := m.Open("lemmy")

	r1, err := m.Append(sess.ID, userTurn("first"))
	core.AssertNoError(t, err)
	core.AssertEqual(t, "resp-1", r1)

	r2, err := m.Append(sess.ID, assistantTurn("second"))
	core.AssertNoError(t, err)
	core.AssertEqual(t, "resp-2", r2)
	core.AssertNotEqual(t, r1, r2)

	// Both responseIDs resolve, but each carries the transcript as it stood at
	// its own position — r1 sees one turn, r2 sees both.
	at1, err := m.Continue(r1)
	core.AssertNoError(t, err)
	core.AssertLen(t, at1.Turns, 1)

	at2, err := m.Continue(r2)
	core.AssertNoError(t, err)
	core.AssertLen(t, at2.Turns, 2)
	core.AssertEqual(t, "first", at2.Turns[0].Text())
	core.AssertEqual(t, "second", at2.Turns[1].Text())
}

func TestSession_Append_Bad(t *core.T) {
	// Appending to a session that was never opened is a typed error.
	m := NewManager(NewMemoryStore())

	_, err := m.Append("sess-missing", userTurn("orphan"))
	core.AssertError(t, err)
	core.AssertErrorIs(t, err, ErrNotFound)
}

func TestSession_Append_Ugly(t *core.T) {
	// An empty session id errors rather than minting a dangling response.
	m := NewManager(NewMemoryStore())

	_, err := m.Append("", userTurn("x"))
	core.AssertError(t, err)
}

func TestSession_StateHandle_RoundTrip(t *core.T) {
	// The go-mlx KV state is opaque to the inference stack: the runtime attaches a handle and
	// it round-trips on the session so the next request can re-attach the same
	// KV blocks (Wake/Sleep) instead of re-prefilling.
	m := NewManager(NewMemoryStore(),
		WithIDGen(seqIDs("sess-1", "resp-1")))

	sess := m.Open("lemma")
	core.AssertEqual(t, "", sess.StateHandle, "a fresh session has no KV handle yet")

	err := m.SetStateHandle(sess.ID, "mlx-kv://node-a/slab/42")
	core.AssertNoError(t, err)

	// The handle is visible on a freshly fetched session and survives a
	// subsequent append (it tracks the live KV state, not a single turn).
	got, err := m.Get(sess.ID)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "mlx-kv://node-a/slab/42", got.StateHandle)

	respID, err := m.Append(sess.ID, userTurn("with state"))
	core.AssertNoError(t, err)

	resumed, err := m.Continue(respID)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "mlx-kv://node-a/slab/42", resumed.StateHandle)

	// Setting a handle on a missing session is a typed error.
	err = m.SetStateHandle("sess-missing", "mlx-kv://nope")
	core.AssertError(t, err)
	core.AssertErrorIs(t, err, ErrNotFound)
}
