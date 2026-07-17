// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/kv"
)

// composed_stateful_session_test.go is #25's acceptance: composedEngineSession holds ONE open
// recurrent stepper across prefill/append/generate, so a continued conversation forwards only
// its new tokens — and decodes token-identically to a cold session given the whole transcript
// (the composed forward is deterministic, so parity is exact, not approximate). The tiny model
// is pure host f32; these run under the plain `go test ./...` gate.

// coldDecode decodes maxNew greedy tokens from a FRESH session prefilled with the whole
// transcript — the reference every stateful continuation must match token-for-token.
func coldDecode(t *testing.T, transcript []int32, maxNew int) []int32 {
	t.Helper()
	sess := newTinyComposedSession(2)
	if err := sess.PrefillTokens(transcript); err != nil {
		t.Fatalf("cold prefill: %v", err)
	}
	gen, err := sess.GenerateFromCacheEach(maxNew, -1, nil)
	if err != nil {
		t.Fatalf("cold generate: %v", err)
	}
	return gen
}

// TestComposedEngineSession_StatefulContinuationParity_Good is the core #25 gate:
// prefill → generate → append → generate on ONE session equals the cold decode of the
// full transcript, and Pos tracks the whole transcript (prompt + generated + appended).
func TestComposedEngineSession_StatefulContinuationParity_Good(t *testing.T) {
	prompt := []int32{1, 2, 3}
	turn2 := []int32{6, 7}

	sess := newTinyComposedSession(2)
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	g1, err := sess.GenerateFromCacheEach(4, -1, nil)
	if err != nil {
		t.Fatalf("turn 1 generate: %v", err)
	}
	if len(g1) == 0 {
		t.Fatal("turn 1 generated nothing")
	}
	if got, want := sess.Pos(), len(prompt)+len(g1); got != want {
		t.Fatalf("Pos after generate = %d, want %d (generated tokens must join the transcript)", got, want)
	}
	if err := sess.AppendTokens(turn2); err != nil {
		t.Fatalf("append: %v", err)
	}
	g2, err := sess.GenerateFromCacheEach(4, -1, nil)
	if err != nil {
		t.Fatalf("turn 2 generate: %v", err)
	}

	transcript := append(append(append([]int32(nil), prompt...), g1...), turn2...)
	want := coldDecode(t, transcript, 4)
	assertTokens(t, "stateful continuation vs cold full-transcript decode", g2, want)
	if got, wantPos := sess.Pos(), len(transcript)+len(g2); got != wantPos {
		t.Fatalf("Pos after turn 2 = %d, want %d", got, wantPos)
	}
}

// TestComposedEngineSession_PrefillTokensCached_Good drives the resident-reuse entry the
// serve prompt cache calls: a prompt EXTENDING the resident transcript keeps the live state
// (reused = resident length minus the one picked-but-unstepped token) and decodes
// token-identically to a cold session on the full prompt.
func TestComposedEngineSession_PrefillTokensCached_Good(t *testing.T) {
	prompt := []int32{1, 2, 3}
	sess := newTinyComposedSession(2)
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	g1, err := sess.GenerateFromCacheEach(3, -1, nil)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	full := append(append(append([]int32(nil), prompt...), g1...), 6, 7)
	reused, err := sess.PrefillTokensCached(full)
	if err != nil {
		t.Fatalf("cached prefill: %v", err)
	}
	// the final generated token was picked but never stepped, so it is not resident.
	if want := len(prompt) + len(g1) - 1; reused != want {
		t.Fatalf("reused = %d, want %d", reused, want)
	}
	g2, err := sess.GenerateFromCacheEach(3, -1, nil)
	if err != nil {
		t.Fatalf("generate after cached prefill: %v", err)
	}
	assertTokens(t, "cached-reuse decode vs cold decode", g2, coldDecode(t, full, 3))
}

// TestComposedEngineSession_PrefillTokensCached_Bad locks the cold fallback: a prompt that
// does NOT extend the resident transcript (recurrent state cannot rewind) reports zero reuse
// and decodes exactly as a fresh PrefillTokens would.
func TestComposedEngineSession_PrefillTokensCached_Bad(t *testing.T) {
	sess := newTinyComposedSession(2)
	if err := sess.PrefillTokens([]int32{1, 2, 3}); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	if _, err := sess.GenerateFromCacheEach(2, -1, nil); err != nil {
		t.Fatalf("generate: %v", err)
	}
	divergent := []int32{9, 8, 7}
	reused, err := sess.PrefillTokensCached(divergent)
	if err != nil {
		t.Fatalf("divergent cached prefill: %v", err)
	}
	if reused != 0 {
		t.Fatalf("divergent prompt reused %d, want 0", reused)
	}
	g, err := sess.GenerateFromCacheEach(3, -1, nil)
	if err != nil {
		t.Fatalf("generate after divergent prefill: %v", err)
	}
	assertTokens(t, "divergent cached prefill decodes as cold", g, coldDecode(t, divergent, 3))
}

// TestComposedEngineSession_CaptureAfterGenerate_Good locks the post-reply snapshot: capture
// after a generate carries prompt + generated tokens, and a session restored from it continues
// token-identically to the live one — the reply is part of the resumable transcript.
func TestComposedEngineSession_CaptureAfterGenerate_Good(t *testing.T) {
	prompt := []int32{1, 2, 3}
	sess := newTinyComposedSession(2)
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	g1, err := sess.GenerateFromCacheEach(3, -1, nil)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	snap, err := sess.CaptureKVWithOptions(kv.CaptureOptions{})
	if err != nil {
		t.Fatalf("capture: %v", err)
	}
	assertTokens(t, "post-reply snapshot tokens", snap.Tokens, append(append([]int32(nil), prompt...), g1...))

	restored := newTinyComposedSession(2)
	if err := restored.RestoreFromKV(nil, snap); err != nil {
		t.Fatalf("restore: %v", err)
	}
	gLive, err := sess.GenerateFromCacheEach(3, -1, nil)
	if err != nil {
		t.Fatalf("live continue: %v", err)
	}
	gRestored, err := restored.GenerateFromCacheEach(3, -1, nil)
	if err != nil {
		t.Fatalf("restored continue: %v", err)
	}
	assertTokens(t, "restored continuation vs live", gRestored, gLive)
}