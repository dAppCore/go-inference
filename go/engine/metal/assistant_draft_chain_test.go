// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
)

// withDraftChain runs fn with the chained-draft lane forced on/off, restoring
// the process default after — the LTHN_MTP_DRAFT_CHAIN lever's test twin.
func withDraftChain(t *testing.T, enabled bool, fn func()) {
	t.Helper()
	prev := mtpDraftChainDisabled
	mtpDraftChainDisabled = !enabled
	defer func() { mtpDraftChainDisabled = prev }()
	fn()
}

// TestAssistantDraftChainParity_Good pins the chained block's whole contract:
// the one-command-buffer draft block proposes the SAME tokens and hands over
// the SAME recursion hidden as the per-step loop. Identical inputs (same
// session boundary, same seed token) are guaranteed because a draft block
// never advances the target session.
func TestAssistantDraftChainParity_Good(t *testing.T) {
	sess, pair, prompt := newSyntheticAssistantPair(t)
	if err := sess.PrepareAssistantPrompt(prompt); err != nil {
		t.Fatalf("PrepareAssistantPrompt: %v", err)
	}
	last := prompt[len(prompt)-1]
	const maxDraft = 4

	var loopBlock, chainBlock AssistantDraftBlockResult
	var loopHidden, chainHidden []byte
	withDraftChain(t, false, func() {
		block, err := pair.DraftBlockFromSession(sess, last, maxDraft)
		if err != nil {
			t.Fatalf("per-step DraftBlockFromSession: %v", err)
		}
		loopBlock = block
		loopHidden = append([]byte(nil), block.Hidden...)
	})
	withDraftChain(t, true, func() {
		fused := pair.fusedDraft()
		if fused == nil {
			t.Fatal("fused drafter did not arm on the synthetic pair")
		}
		if !fused.chainReady(sess) {
			t.Skip("chain pipelines unavailable (custom metallib without argmax/gather/copy kernels)")
		}
		block, err := pair.DraftBlockFromSession(sess, last, maxDraft)
		if err != nil {
			t.Fatalf("chained DraftBlockFromSession: %v", err)
		}
		chainBlock = block
		chainHidden = append([]byte(nil), block.Hidden...)
	})
	if len(chainBlock.Tokens) != maxDraft {
		t.Fatalf("chained block produced %d tokens, want %d", len(chainBlock.Tokens), maxDraft)
	}
	for i := range loopBlock.Tokens {
		if loopBlock.Tokens[i] != chainBlock.Tokens[i] {
			t.Fatalf("chained token[%d] = %d, per-step = %d (tokens %v vs %v)",
				i, chainBlock.Tokens[i], loopBlock.Tokens[i], chainBlock.Tokens, loopBlock.Tokens)
		}
	}
	if !bytes.Equal(loopHidden, chainHidden) {
		t.Fatal("chained recursion hidden differs from the per-step hidden")
	}
}

// TestAssistantDraftChainParity_Suppress_Ugly is the suppression arm: masking
// the unsuppressed winner forces the argmax onto a different id, and the
// device tiles+merge suppression must pick exactly what the host scan picks.
func TestAssistantDraftChainParity_Suppress_Ugly(t *testing.T) {
	sess, pair, prompt := newSyntheticAssistantPair(t)
	if err := sess.PrepareAssistantPrompt(prompt); err != nil {
		t.Fatalf("PrepareAssistantPrompt: %v", err)
	}
	last := prompt[len(prompt)-1]
	const maxDraft = 3

	var free AssistantDraftBlockResult
	withDraftChain(t, false, func() {
		block, err := pair.DraftBlockFromSession(sess, last, maxDraft)
		if err != nil {
			t.Fatalf("unsuppressed DraftBlockFromSession: %v", err)
		}
		free = block
	})
	suppress := []int32{free.Tokens[0]} // ban the natural first pick — the suppression must bite
	var loopTokens, chainTokens []int32
	withDraftChain(t, false, func() {
		block, err := pair.DraftBlockFromSession(sess, last, maxDraft, suppress)
		if err != nil {
			t.Fatalf("per-step suppressed DraftBlockFromSession: %v", err)
		}
		loopTokens = append([]int32(nil), block.Tokens...)
	})
	withDraftChain(t, true, func() {
		if fused := pair.fusedDraft(); fused == nil || !fused.chainReady(sess) {
			t.Skip("chain unavailable on this runtime")
		}
		block, err := pair.DraftBlockFromSession(sess, last, maxDraft, suppress)
		if err != nil {
			t.Fatalf("chained suppressed DraftBlockFromSession: %v", err)
		}
		chainTokens = append([]int32(nil), block.Tokens...)
	})
	for _, id := range loopTokens {
		if id == suppress[0] {
			t.Fatalf("per-step block proposed the suppressed id %d", id)
		}
	}
	for i := range loopTokens {
		if loopTokens[i] != chainTokens[i] {
			t.Fatalf("suppressed chained token[%d] = %d, per-step = %d", i, chainTokens[i], loopTokens[i])
		}
	}
}

// TestAssistantFusedDraftChainReady_Bad pins the decline conditions: the lever
// env, a nil receiver/target, and an ordered-head drafter must all report
// not-ready without touching the GPU.
func TestAssistantFusedDraftChainReady_Bad(t *testing.T) {
	sess, pair, prompt := newSyntheticAssistantPair(t)
	if err := sess.PrepareAssistantPrompt(prompt); err != nil {
		t.Fatalf("PrepareAssistantPrompt: %v", err)
	}
	if _, err := pair.DraftBlockFromSession(sess, prompt[len(prompt)-1], 2); err != nil {
		t.Fatalf("DraftBlockFromSession (fused arm): %v", err)
	}
	fused := pair.fusedDraft()
	if fused == nil {
		t.Fatal("fused drafter did not arm on the synthetic pair")
	}
	withDraftChain(t, false, func() {
		if fused.chainReady(sess) {
			t.Fatal("chainReady = true with the lane disabled")
		}
	})
	withDraftChain(t, true, func() {
		var nilFused *assistantFusedDraft
		if nilFused.chainReady(sess) {
			t.Fatal("chainReady = true on a nil drafter")
		}
		if fused.chainReady(nil) {
			t.Fatal("chainReady = true on a nil target")
		}
		ordered := &assistantFusedDraft{centroidsW: fused.embedW, embedW: fused.embedW, vocabLogits: fused.vocabLogits, vocab: fused.vocab}
		if ordered.chainReady(sess) {
			t.Fatal("chainReady = true for an ordered-head drafter (host top-k step)")
		}
	})
}

// TestAssistantDraftChainDisabled_Good pins the kill switch: with the lane off
// the block still fills through the per-step loop — the repro anchor works.
func TestAssistantDraftChainDisabled_Good(t *testing.T) {
	sess, pair, prompt := newSyntheticAssistantPair(t)
	if err := sess.PrepareAssistantPrompt(prompt); err != nil {
		t.Fatalf("PrepareAssistantPrompt: %v", err)
	}
	withDraftChain(t, false, func() {
		block, err := pair.DraftBlockFromSession(sess, prompt[len(prompt)-1], 3)
		if err != nil {
			t.Fatalf("DraftBlockFromSession: %v", err)
		}
		if len(block.Tokens) != 3 {
			t.Fatalf("disabled-chain block produced %d tokens, want 3", len(block.Tokens))
		}
		synthTokensInVocab(t, "draft", block.Tokens)
	})
}
