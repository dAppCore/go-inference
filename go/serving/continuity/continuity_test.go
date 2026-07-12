// SPDX-Licence-Identifier: EUPL-1.2

package continuity

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
	state "dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/session"
)

// decliningSessionHandle is an inference.SessionHandle whose block-sleep declines — the shape a composed
// hybrid presents before the token-only RangeKVBlocks lands, and the shape any session takes when a sleep
// genuinely fails. Every other method is an inert stub; only RangeKVBlocks' decline is under test.
type decliningSessionHandle struct{}

func (decliningSessionHandle) Prefill(context.Context, string) error      { return nil }
func (decliningSessionHandle) AppendPrompt(context.Context, string) error { return nil }
func (decliningSessionHandle) CaptureKV(context.Context) (*kv.Snapshot, error) {
	return nil, core.NewError("continuity_test: no capture")
}
func (decliningSessionHandle) Generate(context.Context, inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}
func (decliningSessionHandle) RangeKVBlocks(context.Context, int, kv.CaptureOptions, func(kv.Block) (bool, error)) error {
	return core.NewError("continuity_test: block sleep declines")
}
func (decliningSessionHandle) Fork(context.Context) (inference.SessionHandle, error) {
	return nil, core.NewError("continuity_test: no fork")
}
func (decliningSessionHandle) Reset()       {}
func (decliningSessionHandle) Close() error { return nil }
func (decliningSessionHandle) Err() error   { return nil }

// TestConversationTurnSplit gates the prefix/tail boundary: the trailing run
// of user/tool messages is the new turn; everything before it is the prefix a
// prior turn's retained state covers.
func TestConversationTurnSplit(t *testing.T) {
	msg := func(role string) inference.Message { return inference.Message{Role: role, Content: "x"} }
	cases := []struct {
		name string
		in   []inference.Message
		want int
	}{
		{"first turn", []inference.Message{msg("user")}, 0},
		{"system prefixed first turn", []inference.Message{msg("system"), msg("user")}, 1},
		{"second turn", []inference.Message{msg("user"), msg("assistant"), msg("user")}, 2},
		{"tool result turn", []inference.Message{msg("user"), msg("assistant"), msg("tool"), msg("user")}, 2},
		{"role case folded", []inference.Message{msg("user"), msg("Assistant"), msg("USER")}, 2},
		{"no trailing user turn", []inference.Message{msg("user"), msg("assistant")}, 2},
	}
	for _, tc := range cases {
		if got := conversationTurnSplit(tc.in); got != tc.want {
			t.Errorf("%s: split = %d, want %d", tc.name, got, tc.want)
		}
	}
}

// TestConversationKey gates the state-key contract: stable across calls,
// role-spelling folded, and sensitive to every turn's content — the key a
// finished turn sleeps under must be exactly the key the next request's
// prefix hashes to.
func TestConversationKey(t *testing.T) {
	a := []inference.Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hey"}}
	b := []inference.Message{{Role: "USER ", Content: "hi"}, {Role: "assistant", Content: "hey"}}
	if conversationKey(a, false) != conversationKey(b, false) {
		t.Fatal("conversationKey must fold role spelling")
	}
	c := []inference.Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hey!"}}
	if conversationKey(a, false) == conversationKey(c, false) {
		t.Fatal("conversationKey must change with turn content")
	}
	if conversationKey(a, false) == conversationKey(a, true) {
		t.Fatal("conversationKey must separate thinking modes — the retained prefix is framed differently")
	}
}

// TestManagerChatDeclines gates the decline-to-stateless contract on the
// request shapes continuity must not take: empty, media-carrying, explicit
// thinking overrides, and no-trailing-user-turn requests. Declines must be
// (nil, false) and counted.
func TestManagerChatDeclines(t *testing.T) {
	m := &Manager{resident: map[string]*residentConversation{}}
	if _, ok := m.Chat(t.Context(), nil); ok {
		t.Fatal("empty request must decline")
	}
	media := []inference.Message{{Role: "user", Content: "look", Images: [][]byte{{1}}}}
	if _, ok := m.Chat(t.Context(), media); ok {
		t.Fatal("media turns must decline to the stateless multimodal lane")
	}
	noTail := []inference.Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hey"}}
	if _, ok := m.Chat(t.Context(), noTail); ok {
		t.Fatal("no trailing user turn must decline")
	}
	if got := m.Stats().StatelessFallbacks; got != 2 {
		t.Fatalf("StatelessFallbacks = %d, want 2 (media, no-tail; the empty request short-circuits uncounted)", got)
	}
}

// TestEnableCapabilityErrors gates Enable's probe errors: a model without the
// session/framing/interceptor seams (nil here) must error, never panic, so
// serve degrades to stateless with an honest notice.
func TestEnableCapabilityErrors(t *testing.T) {
	if err := Enable(nil, nil); err == nil {
		t.Fatal("nil model must error")
	}
}

// TestManagerFinishTurnSleepDeclineStaysResident gates the sleep-decline degrade: when a session's block
// sleep declines (a composed hybrid without the token-only lane, or any genuine sleep failure), finishTurn
// must keep the conversation RAM-resident, clear its busy flag, and not count the failed sleep — turns keep
// working, durability resumes on the next successful sleep. No hard-fail, no panic.
func TestManagerFinishTurnSleepDeclineStaysResident(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	m := &Manager{
		store:    store,
		writer:   store,
		max:      defaultMaxResident,
		resident: map[string]*residentConversation{},
	}
	conv := &residentConversation{sess: session.New(decliningSessionHandle{}, m.info, nil), busy: true}
	messages := []inference.Message{{Role: "user", Content: "hi"}}

	m.finishTurn(t.Context(), conv, messages, "hello", false)

	if len(m.resident) != 1 {
		t.Fatalf("conversation count after declined sleep = %d, want 1 (stays RAM-resident)", len(m.resident))
	}
	if m.stats.Sleeps != 0 {
		t.Fatalf("declined sleep counted %d Sleeps, want 0", m.stats.Sleeps)
	}
	if conv.busy {
		t.Fatal("finishTurn must clear busy even when the sleep declines")
	}
}
