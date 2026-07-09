// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"os"

	core "dappco.re/go"
)

// session_fold.go — the budget-triggered context fold for retained conversations (#346).
//
// A conversation session grows by one turn + one reply per exchange; without intervention
// the retained cache eventually fills the context window and the next append (or the reply
// after it) dies on "sequence would exceed maxLen cache rows". The fold is the pure math
// reduction: when an appended turn would leave less than the reply headroom, the handle
// keeps token 0 (the BOS the first prefill produced) plus the newest suffix of the
// transcript, and re-prefills kept+turn in ONE engine call — PrefillTokens' replace-state
// contract IS the fold, so this works identically on every engine behind the neutral
// [Session] interface. No summarisation, no model in the loop: the oldest context is
// evicted, the newest kept, and the trade is one re-prefill of the kept tokens. The keep
// target is half the window so folds amortise (a fold then buys ~a quarter-window of
// exchanges before the next), mirroring the engine-level CompactCache primitive and
// llama.cpp's context-shift shape.

// contextFoldDisabled reports the LTHN_CONTEXT_FOLD=0 kill-switch — the A/B lever that
// restores the pre-fold hard error.
func contextFoldDisabled() bool { return os.Getenv("LTHN_CONTEXT_FOLD") == "0" }

// foldHeadroom is the reply room an append must leave free: a quarter of the window,
// at least 64 tokens. Folding at this line (rather than at the hard maxLen wall) keeps
// the generate after the turn from starving too.
func foldHeadroom(maxLen int) int {
	return max(64, maxLen/4)
}

// foldKeepTarget caps the kept transcript at half the window so consecutive folds
// amortise instead of re-prefilling nearly the whole window every turn.
func foldKeepTarget(maxLen int) int {
	return maxLen / 2
}

// foldForAppendLocked folds the retained conversation to make room for ids: keep = token 0
// + the newest suffix, bounded by both the keep target and what leaves the turn + headroom
// inside the window, then one PrefillTokens(kept+ids) replaces the retained state. Returns
// false (untouched) when folding cannot help — the turn alone outgrows the window — so the
// caller falls through to the engine's honest hard error. Caller holds s.mu.
func (s *SessionHandle) foldForAppendLocked(ids []int32) (bool, error) {
	maxLen := s.model.maxLen
	budget := maxLen - foldHeadroom(maxLen) - len(ids)
	if budget < 1 || len(s.tokens) == 0 {
		return false, nil
	}
	keep := min(len(s.tokens), budget, foldKeepTarget(maxLen))
	if keep >= len(s.tokens) {
		return false, nil // nothing would be evicted — the plain append path handles it
	}
	folded := make([]int32, 0, keep+len(ids))
	folded = append(folded, s.tokens[0])                          // the original BOS anchors position 0
	folded = append(folded, s.tokens[len(s.tokens)-(keep-1):]...) // the newest keep-1 tokens
	folded = append(folded, ids...)
	if err := s.sess.PrefillTokens(folded); err != nil {
		s.err = err
		return true, core.E("engine.SessionHandle.AppendPrompt", "context fold re-prefill", err)
	}
	dropped := len(s.tokens) - keep
	s.tokens = folded
	s.contextFolds++
	s.lastFoldKept, s.lastFoldDropped = keep, dropped
	return true, nil
}

// ContextFolds reports how many budget-triggered context folds this session has run —
// the turn loop's seam for an honest "the window folded" note.
func (s *SessionHandle) ContextFolds() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.contextFolds
}

// LastContextFold reports the most recent fold's kept and dropped token counts (zeros
// before any fold).
func (s *SessionHandle) LastContextFold() (kept, dropped int) {
	if s == nil {
		return 0, 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastFoldKept, s.lastFoldDropped
}
