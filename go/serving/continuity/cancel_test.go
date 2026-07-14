// SPDX-Licence-Identifier: EUPL-1.2

package continuity

import (
	"context"
	"iter"
	"testing"

	"dappco.re/go/inference"
	state "dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/session"
)

// cutHandle mimics engine.SessionHandle's ctx-cancel decode contract for the
// unit suite. Generate streams stream[:cutAt], then — as a serving/scheduler
// Cancel (or a client disconnect surfaced as ctx cancel) would — cancels the
// request ctx and stops, so the caller's reply holds only the pre-cancel tokens,
// exactly as the engine's per-token ctx check does. A cutAt at or past
// len(stream) streams the whole reply and only THEN cancels (the
// completion-racing-cancel case); a pre-cancelled ctx yields nothing (the loop's
// leading ctx check fires first). Err reports context.Canceled after a cut,
// matching engine.SessionHandle setting s.err = ctx.Err() on a clean-stop cancel
// — so the switch in Chat is proven to observe the cut via ctx, not via Err.
type cutHandle struct {
	graftHandle
	stream []inference.Token
	cutAt  int
	cancel context.CancelFunc
	cut    bool
}

func (h *cutHandle) Generate(ctx context.Context, _ inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for i, tok := range h.stream {
			if ctx.Err() != nil { // a pre-cancelled ctx never emits a token
				h.cut = true
				return
			}
			if i >= h.cutAt { // reached the cut point mid-stream
				break
			}
			if !yield(tok) {
				return
			}
		}
		// Cut the turn: cancel and stop, like a scheduler Cancel landing on it.
		h.cut = true
		if h.cancel != nil {
			h.cancel()
		}
	}
}

func (h *cutHandle) Err() error {
	if h.cut {
		return context.Canceled
	}
	return h.graftHandle.Err()
}

var _ inference.SessionHandle = (*cutHandle)(nil)

func tok(text string) inference.Token { return inference.Token{Text: text} }

// TestManagerChat_Cancel_Good is the load-bearing cut-turn receipt: a turn cut
// mid-generation clears residency (the conversation is discarded, not left
// resident-busy), records the partial reply the client actually received, and is
// NOT slept — so the NEXT turn re-prefills the client's full transcript cleanly
// (evict-and-refresh: a fresh full prefill, never a wake-and-append of drifted
// state, so no double-append and no position drift). The re-prefill prompt is
// byte-identical to that of an uncancelled equivalent conversation sent fresh —
// the parity that proves the cut left the conversation in a clean, re-derivable
// state.
func TestManagerChat_Cancel_Good(t *testing.T) {
	store := state.NewInMemoryStore(nil)

	// --- Turn 1: a fresh conversation cut after two tokens. ---
	ctx1, cancel1 := context.WithCancel(context.Background())
	cutH := &cutHandle{stream: []inference.Token{tok("t0"), tok("t1"), tok("t2"), tok("t3")}, cutAt: 2, cancel: cancel1}
	m1 := shareManager(store, cutH, nil, false)

	var generated int
	seq1, ok := m1.Chat(ctx1, []inference.Message{{Role: "user", Content: "A"}},
		inference.WithMaxTokens(8),
		inference.WithMetricsSink(func(gm inference.GenerateMetrics) { generated = gm.GeneratedTokens }))
	if !ok {
		t.Fatal("a fresh turn must be served before it is cut")
	}
	drain(seq1)

	if generated != 2 {
		t.Fatalf("client received %d tokens before the cut, want the 2 streamed pre-cancel", generated)
	}
	if s := m1.Stats(); s.CutTurns != 1 || s.Sleeps != 0 || s.FreshConversations != 1 {
		t.Fatalf("cut-turn stats = %+v, want one CutTurn, no Sleeps, one FreshConversation", s)
	}
	if len(m1.resident) != 0 {
		t.Fatalf("resident count after a cut = %d, want 0 (the conversation is discarded, not left resident-busy)", len(m1.resident))
	}
	if cutH.closeCalls != 1 {
		t.Fatalf("cut session closed %d times, want exactly 1", cutH.closeCalls)
	}

	// --- Turn 2: the follow-up. The client re-sends its transcript INCLUDING the
	//     partial reply it received; the cut turn slept nothing, so this must be a
	//     fresh full re-prefill of the whole transcript. ---
	transcript := []inference.Message{
		{Role: "user", Content: "A"},
		{Role: "assistant", Content: "t0t1"}, // exactly the tokens the client got
		{Role: "user", Content: "B"},
	}
	contH := &graftHandle{genTokens: []inference.Token{tok("ans")}}
	m2 := shareManager(store, contH, nil, false)
	seq2, ok := m2.Chat(context.Background(), transcript, inference.WithMaxTokens(8))
	if !ok {
		t.Fatal("the follow-up turn after a cut must be served")
	}
	drain(seq2)

	if s := m2.Stats(); s.FreshConversations != 1 || s.StoreWakes != 0 || s.ResidentTurns != 0 {
		t.Fatalf("follow-up stats = %+v, want one FreshConversation (re-prefill), no wake/resident", s)
	}
	if contH.prefillCalls != 1 || contH.appendCalls != 0 {
		t.Fatalf("follow-up did %d prefills / %d appends, want one full prefill and no continuation append (no double-append)", contH.prefillCalls, contH.appendCalls)
	}

	// --- Parity: an uncancelled equivalent conversation, sent the SAME transcript
	//     fresh, frames the identical prefill prompt. ---
	refH := &graftHandle{genTokens: []inference.Token{tok("ans")}}
	mRef := shareManager(state.NewInMemoryStore(nil), refH, nil, false)
	seqRef, ok := mRef.Chat(context.Background(), transcript, inference.WithMaxTokens(8))
	if !ok {
		t.Fatal("the reference equivalent turn must be served")
	}
	drain(seqRef)

	if contH.prefillPrompt != refH.prefillPrompt {
		t.Fatalf("cut-then-continue prefill %q diverges from the uncancelled equivalent %q", contH.prefillPrompt, refH.prefillPrompt)
	}
}

// TestManagerChat_Cancel_Bad settles the two racy-but-legitimate cuts: a cancel
// that lands BEFORE the first token (an empty partial) and a cancel that lands
// AFTER the whole reply generated (completion won the tokens, the cancel still
// won the outcome). Both must settle cleanly — busy cleared, one CutTurn counted,
// nothing slept, no deadlock — never a half-finished turn wedged resident-busy.
func TestManagerChat_Cancel_Bad(t *testing.T) {
	// (a) cancel before the first token: reply is empty.
	t.Run("before first token", func(t *testing.T) {
		store := state.NewInMemoryStore(nil)
		ctx, cancel := context.WithCancel(context.Background())
		h := &cutHandle{stream: []inference.Token{tok("x"), tok("y")}, cutAt: 0, cancel: cancel}
		m := shareManager(store, h, nil, false)

		var generated int
		seq, ok := m.Chat(ctx, []inference.Message{{Role: "user", Content: "A"}},
			inference.WithMaxTokens(4),
			inference.WithMetricsSink(func(gm inference.GenerateMetrics) { generated = gm.GeneratedTokens }))
		if !ok {
			t.Fatal("the turn must be served before the cut")
		}
		drain(seq)

		if generated != 0 {
			t.Fatalf("cancel before first token streamed %d tokens, want 0", generated)
		}
		if s := m.Stats(); s.CutTurns != 1 || s.Sleeps != 0 {
			t.Fatalf("stats = %+v, want one CutTurn and no Sleeps", s)
		}
		if len(m.resident) != 0 {
			t.Fatalf("resident count = %d, want 0 (no turn wedged resident-busy)", len(m.resident))
		}
	})

	// (b) cancel racing completion: all tokens emit, THEN the cancel lands.
	t.Run("racing completion", func(t *testing.T) {
		store := state.NewInMemoryStore(nil)
		ctx, cancel := context.WithCancel(context.Background())
		h := &cutHandle{stream: []inference.Token{tok("t0"), tok("t1")}, cutAt: 99, cancel: cancel}
		m := shareManager(store, h, nil, false)

		var generated int
		seq, ok := m.Chat(ctx, []inference.Message{{Role: "user", Content: "A"}},
			inference.WithMaxTokens(4),
			inference.WithMetricsSink(func(gm inference.GenerateMetrics) { generated = gm.GeneratedTokens }))
		if !ok {
			t.Fatal("the turn must be served")
		}
		drain(seq)

		if generated != 2 {
			t.Fatalf("completion produced %d tokens, want the full 2", generated)
		}
		// The cancel still won the OUTCOME: the turn is evicted, not slept
		// (pessimistic but always correct — the next turn re-prefills).
		if s := m.Stats(); s.CutTurns != 1 || s.Sleeps != 0 {
			t.Fatalf("stats = %+v, want one CutTurn and no Sleeps even though the reply completed", s)
		}
		if len(m.resident) != 0 {
			t.Fatalf("resident count = %d, want 0", len(m.resident))
		}
	})
}

// TestManagerChat_Cancel_Ugly exercises the degenerate cut shapes: a cancel
// racing finishTurn's sleep-and-evict itself, a double cancel, and a cancel on a
// request whose ctx is already dead before any turn is in flight. None may panic,
// deadlock, or leave a conversation wedged resident-busy.
func TestManagerChat_Cancel_Ugly(t *testing.T) {
	// (a) cancel during finishTurn's sleep: the clean completion path reaches the
	//     sleep, but the ctx cancels under it. finishTurn already degrades a failed
	//     sleep to RAM-resident; either way busy must clear and the conversation
	//     must not deadlock.
	t.Run("cancel during finishTurn sleep", func(t *testing.T) {
		store := state.NewInMemoryStore(nil)
		m := &Manager{store: store, writer: store, max: defaultMaxResident, resident: map[string]*residentConversation{}}
		capH := &graftHandle{kvSnap: synthSnapshot(6)}
		conv := &residentConversation{sess: session.New(capH, m.info, nil), busy: true}

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // the cancel has already landed as finishTurn runs
		m.finishTurn(ctx, conv, []inference.Message{{Role: "user", Content: "A"}}, "reply", nil, false)

		if conv.busy {
			t.Fatal("finishTurn must clear busy even when the ctx cancels under the sleep")
		}
		if len(m.resident) != 1 {
			t.Fatalf("conversation count = %d, want 1 (sleep-or-degrade both leave it resident)", len(m.resident))
		}
	})

	// (b) double cancel: the handle cancels the ctx, the caller cancels it again.
	//     ctx cancellation is idempotent and the generator closure runs once, so
	//     exactly one cut is counted.
	t.Run("double cancel", func(t *testing.T) {
		store := state.NewInMemoryStore(nil)
		ctx, cancel := context.WithCancel(context.Background())
		h := &cutHandle{stream: []inference.Token{tok("t0"), tok("t1"), tok("t2")}, cutAt: 1, cancel: cancel}
		m := shareManager(store, h, nil, false)

		seq, ok := m.Chat(ctx, []inference.Message{{Role: "user", Content: "A"}}, inference.WithMaxTokens(4))
		if !ok {
			t.Fatal("the turn must be served")
		}
		drain(seq)
		cancel() // second cancel — a no-op

		if s := m.Stats(); s.CutTurns != 1 {
			t.Fatalf("double cancel counted %d CutTurns, want exactly 1", s.CutTurns)
		}
		if len(m.resident) != 0 {
			t.Fatalf("resident count = %d, want 0", len(m.resident))
		}
	})

	// (c) cancel on a non-busy conversation: a request arrives with a ctx that is
	//     already dead — there is no in-flight generation to interrupt, and the cut
	//     path must still discard cleanly rather than leak or wedge.
	t.Run("pre-cancelled request", func(t *testing.T) {
		store := state.NewInMemoryStore(nil)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		h := &cutHandle{stream: []inference.Token{tok("t0")}, cutAt: 0}
		m := shareManager(store, h, nil, false)

		seq, ok := m.Chat(ctx, []inference.Message{{Role: "user", Content: "A"}}, inference.WithMaxTokens(4))
		if !ok {
			t.Fatal("the fake session prefills without honouring ctx, so the turn is accepted then cut")
		}
		drain(seq)

		if s := m.Stats(); s.CutTurns != 1 || s.Sleeps != 0 {
			t.Fatalf("stats = %+v, want one CutTurn and no Sleeps for a pre-cancelled request", s)
		}
		if len(m.resident) != 0 {
			t.Fatalf("resident count = %d, want 0", len(m.resident))
		}
	})
}
