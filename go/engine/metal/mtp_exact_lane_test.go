// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"strings"
	"testing"

	inference "dappco.re/go/inference"
	"dappco.re/go/inference/engine"
)

// mtp_exact_lane_test.go — the #55 regression pins. MTP speculative decode
// under full greedy emitted DIFFERENT text than plain decode; the two defects
// were (1) the speculative Chat framing rendering plain turns without the
// large-variant thought-suppressor cue, so the pair conditioned on different
// prompt tokens than plain, and (2) the greedy verify forward taking the
// small-K batched FOLD, a qmm token-identity tier whose batched numerics are
// not byte-identical to sequential decode and flip a near-tied argmax. These
// tests pin the routing rule and the framing parity host-side — no model
// download; the live byte-identity receipts (26B/E2B pair-vs-plain) ride the
// commit message.

func restoreMTPFoldLevers(t *testing.T) {
	t.Helper()
	forced, disabled, rowsHead := mtpVerifyFoldForced, mtpVerifyFoldDisabled, mtpRowsHeadForced
	t.Cleanup(func() {
		mtpVerifyFoldForced, mtpVerifyFoldDisabled, mtpRowsHeadForced = forced, disabled, rowsHead
	})
}

// TestMtpVerifyFoldArmed_Good pins the default #55 routing: the byte-exact
// greedy lane (exact=true) never takes the fold; the sampled lane keeps it.
// The greedy contract is what keeps the emitted stream invariant to the
// re-engagement policy's wall-clock verdicts (mtpVerifyFoldArmed's doc).
func TestMtpVerifyFoldArmed_Good(t *testing.T) {
	restoreMTPFoldLevers(t)
	mtpVerifyFoldForced, mtpVerifyFoldDisabled = false, false
	if mtpVerifyFoldArmed(true) {
		t.Fatal("exact greedy verify armed the batched fold — the byte-exact contract cannot hold on the token-identity tier (#55)")
	}
	if !mtpVerifyFoldArmed(false) {
		t.Fatal("sampled verify did not arm the batched fold — the sampled lane's throughput tier regressed")
	}
}

// TestMtpVerifyFoldArmed_Bad pins LTHN_MTP_VERIFY_FOLD=0: the per-row lane is
// forced everywhere, including the sampled verify.
func TestMtpVerifyFoldArmed_Bad(t *testing.T) {
	restoreMTPFoldLevers(t)
	mtpVerifyFoldForced, mtpVerifyFoldDisabled = false, true
	if mtpVerifyFoldArmed(true) || mtpVerifyFoldArmed(false) {
		t.Fatal("LTHN_MTP_VERIFY_FOLD=0 must force the per-row lane in both lanes")
	}
}

// TestMtpVerifyFoldArmed_Ugly pins LTHN_MTP_VERIFY_FOLD=1 — the A/B lever that
// resurrects the pre-#55 fold-everywhere behaviour, exact lane included.
func TestMtpVerifyFoldArmed_Ugly(t *testing.T) {
	restoreMTPFoldLevers(t)
	mtpVerifyFoldForced, mtpVerifyFoldDisabled = true, false
	if !mtpVerifyFoldArmed(true) || !mtpVerifyFoldArmed(false) {
		t.Fatal("LTHN_MTP_VERIFY_FOLD=1 must force the fold in both lanes (the #55 A/B lever)")
	}
}

// exactLaneFakeTok is the minimal engine.TextTokenizer for template framing
// tests: it carries the gemma4 <|turn> marker (and no ChatML marker), which is
// all DetectTurnTokens/DetectChatTemplate consult.
type exactLaneFakeTok struct{}

func (exactLaneFakeTok) Encode(text string) []int32 { return []int32{int32(len(text))} }
func (exactLaneFakeTok) Decode([]int32) string      { return "" }
func (exactLaneFakeTok) DecodeToken(int32) string   { return "" }
func (exactLaneFakeTok) DecodeOne(int32) string     { return "" }
func (exactLaneFakeTok) EOS() int32                 { return 1 }
func (exactLaneFakeTok) TokenID(text string) (int32, bool) {
	if text == "<|turn>" || text == "<turn|>" {
		return 5, true
	}
	return 0, false
}

// TestSpeculativeChatPrompt_Good pins the #55 framing parity on a large-variant
// target (26B geometry, 16 heads): the speculative framing must equal the
// plain lane's declared-template render — DetectChatTemplate WITH the
// thought-suppressor — byte for byte, including the pre-closed empty thought
// channel on the thinking-off cue. The broken framing rendered plain turns
// (no suppressor cue), so the pair conditioned on 34 prompt tokens where plain
// conditioned on 38 and greedy equality was broken before any verify ran.
func TestSpeculativeChatPrompt_Good(t *testing.T) {
	tok := exactLaneFakeTok{}
	turns := engine.DetectTurnTokens(tok)
	msgs := []inference.Message{{Role: "user", Content: "reverse a linked list"}}
	think := false

	got := speculativeChatPrompt(tok, turns, largeVariantAttentionHeads, msgs, &think)
	want := engine.RenderChatPrompt(engine.DetectChatTemplate(tok, turns, true), msgs, &think)
	if got != want {
		t.Fatalf("speculative framing diverged from the plain declared-template render:\n got %q\nwant %q", got, want)
	}
	if !strings.Contains(got, "<|channel>thought") {
		t.Fatalf("large-variant thinking-off framing lost the pre-closed thought channel: %q", got)
	}
}

// TestSpeculativeChatPrompt_Bad pins the small-variant framing (E2B geometry,
// 8 heads): no thought-suppressor cue — matching the plain lane's declaration
// for the same geometry.
func TestSpeculativeChatPrompt_Bad(t *testing.T) {
	tok := exactLaneFakeTok{}
	turns := engine.DetectTurnTokens(tok)
	msgs := []inference.Message{{Role: "user", Content: "reverse a linked list"}}
	think := false

	got := speculativeChatPrompt(tok, turns, 8, msgs, &think)
	want := engine.RenderChatPrompt(engine.DetectChatTemplate(tok, turns, false), msgs, &think)
	if got != want {
		t.Fatalf("small-variant speculative framing diverged from the plain render:\n got %q\nwant %q", got, want)
	}
	if strings.Contains(got, "<|channel>thought") {
		t.Fatalf("small-variant framing must not carry the thought-suppressor cue: %q", got)
	}
}

// TestSpeculativeChatPrompt_Ugly pins the thinking-default framing
// (enableThinking nil, gemma4 DefaultOn): the speculative lane renders the
// leading system turn with the <|think|> switch exactly as plain does —
// surprising but valid, because thinking is the gemma4 vendor default.
func TestSpeculativeChatPrompt_Ugly(t *testing.T) {
	tok := exactLaneFakeTok{}
	turns := engine.DetectTurnTokens(tok)
	msgs := []inference.Message{{Role: "user", Content: "reverse a linked list"}}

	got := speculativeChatPrompt(tok, turns, largeVariantAttentionHeads, msgs, nil)
	want := engine.RenderChatPrompt(engine.DetectChatTemplate(tok, turns, true), msgs, nil)
	if got != want {
		t.Fatalf("thinking-default speculative framing diverged from the plain render:\n got %q\nwant %q", got, want)
	}
	if !strings.Contains(got, "<|think|>") {
		t.Fatalf("gemma4 DefaultOn framing lost the <|think|> switch: %q", got)
	}
}
