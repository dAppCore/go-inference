// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// foldTestHandle builds a handle over the fake engine session with a controlled
// window and a synthetic retained transcript of n tokens (ids 0..n-1), as a
// conversation that has already run n tokens would hold.
func foldTestHandle(t *testing.T, maxLen, resident int) (*SessionHandle, *fakeSession) {
	t.Helper()
	m := NewTextModel(&fakeTokenModel{}, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, maxLen)
	sess := &fakeSession{pos: resident}
	handle := NewSessionHandle(m, sess)
	tokens := make([]int32, resident)
	for i := range tokens {
		tokens[i] = int32(i)
	}
	handle.tokens = tokens
	return handle, sess
}

// TestSessionFold_AppendPrompt_Good pins the budget-triggered fold: a turn that
// would leave less than the reply headroom folds the transcript to BOS + the
// newest suffix and re-prefills kept+turn in ONE engine call — no append, no
// error, counters reporting the eviction.
func TestSessionFold_AppendPrompt_Good(t *testing.T) {
	const maxLen, resident = 256, 200 // headroom 64, keep target 128
	handle, sess := foldTestHandle(t, maxLen, resident)
	if err := handle.AppendPrompt(context.Background(), "hello world"); err != nil {
		t.Fatalf("AppendPrompt: %v", err)
	}
	if len(sess.appendCalls) != 0 {
		t.Fatalf("fold path must not AppendTokens (got %d append calls)", len(sess.appendCalls))
	}
	if len(sess.prefillCalls) != 1 {
		t.Fatalf("fold path re-prefills exactly once (got %d)", len(sess.prefillCalls))
	}
	folded := sess.prefillCalls[0]
	ids := len(folded) - foldKeepTarget(maxLen) // kept tokens = the keep target here
	if ids < 1 {
		t.Fatalf("folded call %d tokens is shorter than the keep target %d", len(folded), foldKeepTarget(maxLen))
	}
	if folded[0] != 0 {
		t.Fatalf("fold must keep the original BOS (token 0), got %d", folded[0])
	}
	// the newest keep-1 transcript tokens follow BOS: resident-(keep-1) .. resident-1
	keep := foldKeepTarget(maxLen)
	if want := int32(resident - (keep - 1)); folded[1] != want {
		t.Fatalf("fold suffix starts at %d, want %d (newest keep-1)", folded[1], want)
	}
	if folded[keep-1] != int32(resident-1) {
		t.Fatalf("fold suffix ends at %d, want %d (the newest resident token)", folded[keep-1], resident-1)
	}
	if got := handle.ContextFolds(); got != 1 {
		t.Fatalf("ContextFolds = %d, want 1", got)
	}
	kept, dropped := handle.LastContextFold()
	if kept != keep || dropped != resident-keep {
		t.Fatalf("LastContextFold = (%d, %d), want (%d, %d)", kept, dropped, keep, resident-keep)
	}
	if len(handle.tokens) != len(folded) {
		t.Fatalf("transcript after fold = %d tokens, want the folded %d", len(handle.tokens), len(folded))
	}
}

// TestSessionFold_AppendPrompt_Bad pins the under-budget path: a turn that fits
// with headroom to spare appends normally — no fold, no re-prefill.
func TestSessionFold_AppendPrompt_Bad(t *testing.T) {
	const maxLen, resident = 256, 40
	handle, sess := foldTestHandle(t, maxLen, resident)
	if err := handle.AppendPrompt(context.Background(), "hello"); err != nil {
		t.Fatalf("AppendPrompt: %v", err)
	}
	if len(sess.prefillCalls) != 0 {
		t.Fatalf("under-budget append must not fold (got %d prefill calls)", len(sess.prefillCalls))
	}
	if len(sess.appendCalls) != 1 {
		t.Fatalf("under-budget append routes to AppendTokens (got %d calls)", len(sess.appendCalls))
	}
	if got := handle.ContextFolds(); got != 0 {
		t.Fatalf("ContextFolds = %d, want 0", got)
	}
}

// TestSessionFold_AppendPrompt_Ugly pins the kill-switch and the fold-error edge:
// LTHN_CONTEXT_FOLD=0 restores the plain append (the engine's hard error stands),
// and a re-prefill failure during a fold surfaces as the append's error.
func TestSessionFold_AppendPrompt_Ugly(t *testing.T) {
	t.Setenv("LTHN_CONTEXT_FOLD", "0")
	handle, sess := foldTestHandle(t, 256, 200)
	if err := handle.AppendPrompt(context.Background(), "hello"); err != nil {
		t.Fatalf("AppendPrompt with fold disabled: %v", err)
	}
	if len(sess.prefillCalls) != 0 || len(sess.appendCalls) != 1 {
		t.Fatalf("fold disabled must append plainly (prefills %d, appends %d)", len(sess.prefillCalls), len(sess.appendCalls))
	}

	t.Setenv("LTHN_CONTEXT_FOLD", "")
	handle2, sess2 := foldTestHandle(t, 256, 200)
	sess2.prefillErr = core.NewError("fixture: prefill failed")
	if err := handle2.AppendPrompt(context.Background(), "hello"); err == nil {
		t.Fatal("a failing fold re-prefill must surface an error")
	}
}
