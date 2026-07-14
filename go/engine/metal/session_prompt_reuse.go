// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
)

// session_prompt_reuse.go is the serve-side automatic prompt cache (#377,
// llama.cpp slot-cache parity). PrefillTokensCached reuses the retained cache
// where a new prompt shares a prefix with the ids already resident (cachedIDs
// is maintained by every prefill / append / decode path), re-prefilling only
// the divergent suffix. The multi-turn chat shape — each request re-sends the
// whole conversation — then pays only the new turn instead of the whole
// history, exactly like the -state lane but volatile and single-slot.
//
// Ring safety is the first hard rule. Sliding-window layers keep a bounded
// ring (cacheLen = slidingWindow): landing row p overwrites the slot row p−w
// held, so a rollback past a wrapped ring would resume attention over window
// rows the discarded tail has already destroyed. A rollback (lcp < pos) is
// therefore taken only while pos ≤ slidingWindow (nothing has wrapped yet);
// otherwise the call degrades to a cold PrefillTokens — token-identical to a
// fresh session, just uncached.
//
// q8 landing stability is the second (#1845). Prefill numerics are batch-
// shape dependent at the projection-kernel knees (the same token-identity
// tier the fold's qmm / steel boundary trades at): landing a prompt as
// prefix+suffix calls produces KV rows a few bf16 ulps off the whole-call
// landing. bf16 caches absorb that inside greedy margins (probed on real
// checkpoints: token-identical), but the q8 store quantises the wobble
// DISCONTINUOUSLY — code and scale flips amplify it ~4× (probe: boundary
// hidden max|Δ| 0.078 bf16 → 0.31 q8 on gemma-4-31B) — and knife-edge tokens
// flip: the same request then answers differently depending on what the slot
// held before it. A session with the q8-ICB store active therefore DECLINES
// the cached path entirely and takes the cold prefill — request-deterministic,
// just uncached — until the landing is canonicalised. The full evidence pack
// lives in TestProbePromptReuseParity (probe_prompt_reuse_parity_test.go).

// PrefillTokensCached is PrefillTokens with automatic prefix reuse: it finds
// the longest prefix of ids already resident from prior calls on this session,
// re-prefills only the divergent suffix, and reports how many prompt rows were
// reused (0 on the cold path). The retained state after the call is identical
// to PrefillTokens(ids) — TestPrefillTokensCachedTokenIdentical gates decode
// parity against a cold session.
func (s *ArchSession) PrefillTokensCached(ids []int32) (int, error) {
	if len(ids) == 0 {
		return 0, core.NewError("native.ArchSession.PrefillTokensCached: empty prompt tokens")
	}
	if len(ids) > s.maxLen {
		return 0, core.NewError("native.ArchSession.PrefillTokensCached: sequence would exceed maxLen cache rows")
	}
	// q8-ICB sessions decline reuse (#1845): the q8 store amplifies the
	// batch-shape numeric tier into token flips, so a cached prefix+suffix
	// landing is not answer-stable against the whole-prompt landing. Cold
	// prefill keeps every request deterministic.
	if s.state.icb.hasKVQ8() {
		return 0, s.PrefillTokens(ids)
	}
	n := s.pos
	if len(s.cachedIDs) < n {
		n = len(s.cachedIDs)
	}
	lcp := 0
	for lcp < len(ids) && lcp < n && ids[lcp] == s.cachedIDs[lcp] {
		lcp++
	}
	if lcp == len(ids) {
		// Exact hit (or the resident run extends past the prompt): keep at
		// least one token to re-append so the decode boundary hidden/logits
		// are rebuilt for the generate-from-cache entry.
		lcp = len(ids) - 1
	}
	if lcp == 0 {
		return 0, s.PrefillTokens(ids)
	}
	if lcp < s.pos {
		if s.ringRollbackUnsafe() {
			return 0, s.PrefillTokens(ids)
		}
		if !s.TruncateTo(lcp) {
			return 0, s.PrefillTokens(ids)
		}
		if err := s.truncateSpeculativeKV(lcp); err != nil {
			return 0, err
		}
	}
	if err := s.AppendTokens(ids[lcp:]); err != nil {
		return 0, err
	}
	return lcp, nil
}

// ringRollbackUnsafe reports whether rolling the session boundary back would
// resume attention over sliding-ring rows the discarded tail has already
// overwritten — true once any sliding ring has wrapped (pos > window). Global
// layers keep maxLen-linear caches and are always rollback-safe.
func (s *ArchSession) ringRollbackUnsafe() bool {
	w := s.state.slidingWindow
	return w > 0 && w < s.maxLen && s.pos > w
}
