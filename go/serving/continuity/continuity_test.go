// SPDX-Licence-Identifier: EUPL-1.2

package continuity

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model/spine"
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

// countingSessionHandle is a decliningSessionHandle that records how many times
// Close was called — the seam residentConversation.close and the eviction path
// close through, so the count proves the handle was actually released.
type countingSessionHandle struct {
	decliningSessionHandle
	closed int
}

func (h *countingSessionHandle) Close() error { h.closed++; return nil }

// fakeSessionFactory is the inference.SessionFactory Manager.newConversation
// consumes: NewSession hands back handle (nil to exercise the nil-session
// guard) and counts its calls.
type fakeSessionFactory struct {
	handle inference.SessionHandle
	calls  int
}

func (f *fakeSessionFactory) NewSession() inference.SessionHandle {
	f.calls++
	return f.handle
}

var _ inference.SessionFactory = (*fakeSessionFactory)(nil)

// TestResidentConversationClose gates the close seam both ways: a
// conversation holding only a raw handle closes the handle directly, and one
// holding a session.Session closes through the session (which releases the same
// underlying handle). Close must reach the handle exactly once on each path.
func TestResidentConversationClose(t *testing.T) {
	// handle-only branch: no session wrapper.
	rawHandle := &countingSessionHandle{}
	(&residentConversation{handle: rawHandle}).close()
	if rawHandle.closed != 1 {
		t.Fatalf("handle-only close: handle.closed = %d, want 1", rawHandle.closed)
	}

	// session branch: the wrapper is closed, which releases the handle beneath.
	sessHandle := &countingSessionHandle{}
	(&residentConversation{sess: session.New(sessHandle, spine.ModelInfo{}, nil), handle: sessHandle}).close()
	if sessHandle.closed != 1 {
		t.Fatalf("session close: underlying handle.closed = %d, want 1", sessHandle.closed)
	}

	// A conversation with neither is inert — no panic.
	(&residentConversation{}).close()
}

// TestManagerNewConversation gates fresh-session construction: a working
// factory yields a busy conversation with both the handle and its session view
// set, and a factory returning a nil session errors instead of handing back a
// conversation that would nil-panic on first use.
func TestManagerNewConversation(t *testing.T) {
	m := &Manager{factory: &fakeSessionFactory{handle: &countingSessionHandle{}}}
	conv, err := m.newConversation()
	if err != nil {
		t.Fatalf("newConversation: %v", err)
	}
	if conv.handle == nil || conv.sess == nil {
		t.Fatal("newConversation must set both the handle and its session view")
	}
	if !conv.busy {
		t.Fatal("a fresh conversation must start busy (owned by the acquiring turn)")
	}

	nilFactory := &Manager{factory: &fakeSessionFactory{handle: nil}}
	if _, err := nilFactory.newConversation(); err == nil {
		t.Fatal("a nil session from the factory must error, not return a conversation")
	}
}

// TestManagerRemoveOrderLocked gates the eviction-order bookkeeping: removing a
// key drops exactly that entry and preserves the rest's order, at the front,
// middle, and tail; an absent key is a no-op.
func TestManagerRemoveOrderLocked(t *testing.T) {
	eq := func(got, want []string) bool {
		if len(got) != len(want) {
			return false
		}
		for i := range got {
			if got[i] != want[i] {
				return false
			}
		}
		return true
	}
	cases := []struct {
		name   string
		order  []string
		remove string
		want   []string
	}{
		{"middle", []string{"a", "b", "c"}, "b", []string{"a", "c"}},
		{"front", []string{"a", "b", "c"}, "a", []string{"b", "c"}},
		{"tail", []string{"a", "b", "c"}, "c", []string{"a", "b"}},
		{"absent", []string{"a", "b"}, "z", []string{"a", "b"}},
		{"last remaining", []string{"only"}, "only", []string{}},
	}
	for _, tc := range cases {
		m := &Manager{order: append([]string(nil), tc.order...)}
		m.removeOrderLocked(tc.remove)
		if !eq(m.order, tc.want) {
			t.Errorf("%s: order = %v, want %v", tc.name, m.order, tc.want)
		}
	}
}

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
	// An explicit thinking override (either polarity) rides the stateless path —
	// the lane only guarantees byte-identity for the model's default thinking
	// mode (#1841).
	thinkOff, thinkOn := false, true
	for _, et := range []*bool{&thinkOff, &thinkOn} {
		if _, ok := m.Chat(t.Context(), []inference.Message{{Role: "user", Content: "hi"}}, inference.WithEnableThinking(et)); ok {
			t.Fatalf("explicit thinking override (=%v) must decline to the stateless path", *et)
		}
	}
	noTail := []inference.Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hey"}}
	if _, ok := m.Chat(t.Context(), noTail); ok {
		t.Fatal("no trailing user turn must decline")
	}
	if got := m.Stats().StatelessFallbacks; got != 4 {
		t.Fatalf("StatelessFallbacks = %d, want 4 (media, two thinking overrides, no-tail; the empty request short-circuits uncounted)", got)
	}
}

// TestManagerChat_ThinkingOverride_IsolatesFromHistory is the RECEIPT 2 guard
// (#1841): a request that is a genuine continuation of a resident conversation
// but carries an explicit thinking override must NOT wake that conversation's
// state — it declines to the stateless path, so its answer cannot depend on
// serving history. Before the decline landed, the override continuation woke the
// resident prefix (a ResidentTurn), coupling the two conversations.
func TestManagerChat_ThinkingOverride_IsolatesFromHistory(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	h := &graftHandle{kvSnap: synthSnapshot(8), genTokens: []inference.Token{{Text: "ok"}}}
	m := shareManager(store, h, nil, false)

	// Turn 1 (no override) establishes a resident conversation whose
	// [user, assistant] prefix a continuation would wake.
	streamed, ok := m.Chat(ctx, []inference.Message{{Role: "user", Content: "A"}}, inference.WithMaxTokens(4))
	if !ok {
		t.Fatal("a fresh no-override turn must be served")
	}
	drain(streamed)
	if s := m.Stats(); s.FreshConversations != 1 {
		t.Fatalf("turn 1 stats = %+v, want one fresh conversation", s)
	}

	// Turn 2 shares turn 1's prefix (its reply was "ok") but overrides thinking.
	off := false
	continuation := []inference.Message{
		{Role: "user", Content: "A"},
		{Role: "assistant", Content: "ok"},
		{Role: "user", Content: "B"},
	}
	if _, ok := m.Chat(ctx, continuation, inference.WithMaxTokens(4), inference.WithEnableThinking(&off)); ok {
		t.Fatal("an override continuation must decline, not wake the resident conversation")
	}
	if s := m.Stats(); s.ResidentTurns != 0 || s.StoreWakes != 0 {
		t.Fatalf("override continuation coupled to history: stats = %+v, want no ResidentTurns/StoreWakes", s)
	}
}

// TestManagerChat_ThinkingOverride_RepeatDeclines is the RECEIPT 3 guard
// (#1841): a byte-identical deterministic request carrying an explicit thinking
// override is answered fresh every time — the lane declines it to the stateless
// path on each repeat rather than continuing an earlier identical turn. Before
// the decline landed, the lane served every such request itself.
func TestManagerChat_ThinkingOverride_RepeatDeclines(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	h := &graftHandle{kvSnap: synthSnapshot(8), genTokens: []inference.Token{{Text: "ok"}}}
	m := shareManager(store, h, nil, false)

	req := []inference.Message{{Role: "user", Content: "Write the integers from 1 to 800"}}
	off := false
	for i := 0; i < 2; i++ {
		if _, ok := m.Chat(ctx, req, inference.WithMaxTokens(48), inference.WithEnableThinking(&off)); ok {
			t.Fatalf("repeat %d: a deterministic override request must decline to stateless, not be served by the lane", i)
		}
	}
	if s := m.Stats(); s.StatelessFallbacks != 2 || s.FreshConversations != 0 || s.ResidentTurns != 0 {
		t.Fatalf("stats = %+v, want two stateless fallbacks and no lane turns", s)
	}
	if h.prefillCalls != 0 || h.appendCalls != 0 {
		t.Fatalf("the lane touched its session for a declined request: prefill=%d append=%d", h.prefillCalls, h.appendCalls)
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

	m.finishTurn(t.Context(), conv, messages, "hello", nil, false)

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

// TestManagerChat_StoreWakeDeclineServesStateless is the WAKE half of the composed
// graceful-degrade contract (handover item: "graceful degrade when composed sleep
// declines" names capture AND restore; TestManagerFinishTurnSleepDeclineStaysResident
// covers the capture/sleep decline, this covers the restore/wake decline). A
// conversation is slept to the store by a capturing manager; its next turn then
// arrives at a fresh manager whose session CANNOT restore (a composed hybrid whose
// wake declines, or any genuine restore failure). The manager must decline that turn
// to the stateless path — (nil, false), counted, an honest log line — never surfacing
// the wake error to the client and never a silent wrong answer: the caller re-prefills
// the whole conversation statelessly, which is always correct, just slower.
func TestManagerChat_StoreWakeDeclineServesStateless(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	// Turn 1 through a capturing manager: a fresh [system,user] turn is served,
	// generates "ok", and sleeps its state to the store under the conversation key
	// the next turn looks up.
	seeder := shareManager(store, &graftHandle{kvSnap: synthSnapshot(6), genTokens: []inference.Token{{Text: "ok"}}}, nil, false)
	turn1 := []inference.Message{{Role: "system", Content: "S"}, {Role: "user", Content: "U1"}}
	streamed, ok := seeder.Chat(ctx, turn1, inference.WithMaxTokens(4))
	if !ok {
		t.Fatal("seed turn declined — a fresh [system,user] turn must be served")
	}
	drain(streamed)
	if s := seeder.Stats(); s.FreshConversations != 1 || s.Sleeps != 1 {
		t.Fatalf("seed stats = %+v, want one FreshConversation and one Sleep (state is in the store)", s)
	}

	// Turn 2 arrives at a FRESH manager (the conversation is not RAM-resident here,
	// so it must wake from the store) whose session cannot restore. WakeAgentMemory
	// fails, so acquire errors and Chat declines to the stateless path — ok=false
	// tells the caller to re-prefill statelessly (a fresh conversation would instead
	// return ok=true, so ok=false uniquely proves the wake was attempted and declined).
	waker := shareManager(store, decliningSessionHandle{}, nil, false)
	turn2 := []inference.Message{
		{Role: "system", Content: "S"},
		{Role: "user", Content: "U1"},
		{Role: "assistant", Content: "ok"},
		{Role: "user", Content: "U2"},
	}
	seq2, ok := waker.Chat(ctx, turn2, inference.WithMaxTokens(4))
	if ok {
		t.Fatal("a turn whose store-wake declines must decline to the stateless path, not be served on a half-woken session")
	}
	if seq2 != nil {
		t.Fatal("a declined turn must return a nil token sequence")
	}
	if s := waker.Stats(); s.StatelessFallbacks != 1 || s.StoreWakes != 0 {
		t.Fatalf("wake-decline stats = %+v, want one StatelessFallback and no StoreWakes", s)
	}
}

// TestSpineModelInfo checks the neutral-to-spine field mapping and the context
// length default: the seven mapped fields carry across verbatim, and a
// non-positive context length falls back to 4096 (the spine's minimum window).
func TestSpineModelInfo(t *testing.T) {
	info := inference.ModelInfo{
		Architecture: "gemma3",
		VocabSize:    262144,
		NumLayers:    34,
		HiddenSize:   3840,
		QuantBits:    4,
		QuantGroup:   64,
	}
	got := spineModelInfo(info, 8192)
	want := spine.ModelInfo{
		Architecture:  "gemma3",
		VocabSize:     262144,
		NumLayers:     34,
		HiddenSize:    3840,
		QuantBits:     4,
		QuantGroup:    64,
		ContextLength: 8192,
	}
	if got.Architecture != want.Architecture || got.VocabSize != want.VocabSize ||
		got.NumLayers != want.NumLayers || got.HiddenSize != want.HiddenSize ||
		got.QuantBits != want.QuantBits || got.QuantGroup != want.QuantGroup ||
		got.ContextLength != want.ContextLength {
		t.Fatalf("spineModelInfo mapping = %+v, want the seven fields of %+v", got, want)
	}

	// A non-positive context length falls back to the default.
	for _, in := range []int{0, -1} {
		if cl := spineModelInfo(info, in).ContextLength; cl != 4096 {
			t.Errorf("spineModelInfo(contextLen=%d).ContextLength = %d, want 4096", in, cl)
		}
	}
}

// TestManagerChat_MetricsSink_Good pins the request-scoped usage delivery on
// a continuity-served turn: the woken-session path bypasses the engine's own
// sink point, so the manager delivers this turn's counts to
// GenerateConfig.MetricsSink itself (the same numbers RecordChatMetrics banks
// globally).
func TestManagerChat_MetricsSink_Good(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	handle := &graftHandle{genTokens: []inference.Token{{Text: "a"}, {Text: "b"}, {Text: "c"}}}
	m := shareManager(store, handle, nil, false)

	var got inference.GenerateMetrics
	fired := 0
	streamed, ok := m.Chat(ctx, []inference.Message{{Role: "user", Content: "hi"}},
		inference.WithMaxTokens(4),
		inference.WithMetricsSink(func(gm inference.GenerateMetrics) {
			got = gm
			fired++
		}))
	if !ok {
		t.Fatal("continuity declined a plain text turn")
	}
	drain(streamed)
	if fired != 1 {
		t.Fatalf("MetricsSink fired %d times, want exactly once", fired)
	}
	if got.GeneratedTokens != 3 {
		t.Fatalf("sink GeneratedTokens = %d, want the 3 served tokens", got.GeneratedTokens)
	}
}
