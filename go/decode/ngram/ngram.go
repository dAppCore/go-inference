// SPDX-Licence-Identifier: EUPL-1.2

// Package ngram is the n-gram speculative drafter — prompt-lookup decoding for
// the speculative path. A target model decodes faster when something cheaply
// proposes the next few tokens for it to VERIFY in one forward pass instead of
// generating them one at a time. The cheapest such proposer needs no draft model
// at all: it looks the continuation up in the prompt/context itself.
//
// The method: take the last n tokens of the context (the suffix), find the most
// recent EARLIER place that same suffix occurred, and propose the tokens that
// followed it there. Repeated text — boilerplate, quoted source, a name said
// twice, a list pattern — gets predicted for free. The drafter is pure integer
// logic over token ids: it proposes, it never verifies; the caller runs the
// target model to accept or reject the proposed tokens (RFC speculative
// decoding). It is fully deterministic — same context, same draft, every time.
//
// Two ways to drive it, and they compose:
//
//	// 1. Stateless: hand it the full context each call (easy to test, no state).
//	d := ngram.New(ngram.Config{MaxNgram: 3, MaxDraft: 4})
//	draft := d.Draft(promptTokens)           // propose from this exact context
//
//	// 2. Stateful: keep a running context and append accepted tokens to it.
//	d := ngram.New(ngram.Config{MaxNgram: 3, MaxDraft: 4})
//	d.Update(promptTokens)                    // seed the running context
//	for {
//	    draft := d.DraftNext()               // propose from the running context
//	    accepted := target.Verify(draft)     // target accepts a prefix of it
//	    d.Update(accepted)                    // grow the context, draft again
//	}
//
// DraftNext() is exactly Draft(Context()): the stateful API is a thin running
// buffer over the same lookup, so the two never disagree.
package ngram

import "sync"

// Config tunes the drafter. MaxNgram is the longest suffix it will try to match
// (longer = more specific, higher-confidence matches); MaxDraft caps how many
// tokens a single Draft proposes (longer = more speculation per target pass, but
// more wasted work when the target rejects). Both are clamped to a minimum of 1,
// so the zero Config is a usable 1-gram, 1-token drafter rather than a dead one.
//
//	ngram.Config{MaxNgram: 3, MaxDraft: 4} // match up to trigrams, propose up to 4
type Config struct {
	MaxNgram int // longest suffix length to look up (clamped ≥ 1)
	MaxDraft int // maximum tokens proposed per Draft (clamped ≥ 1)
}

// Drafter proposes draft continuations by prompt-lookup. Construct with New. The
// stateless Draft is safe to call concurrently; the stateful Update / DraftNext /
// Context / Reset share a running context guarded by a mutex, so a single Drafter
// may be driven from more than one goroutine without data races.
type Drafter struct {
	maxNgram int
	maxDraft int

	mu  sync.Mutex
	ctx []int // running context grown by Update; read by DraftNext / Context
}

// New builds a Drafter from a Config, clamping MaxNgram and MaxDraft up to 1 so
// the drafter is always usable (a zero-value Config yields a 1-gram, 1-token
// drafter rather than one that proposes nothing).
//
//	d := ngram.New(ngram.Config{MaxNgram: 3, MaxDraft: 4})
func New(cfg Config) *Drafter {
	n := max(cfg.MaxNgram, 1)
	k := max(cfg.MaxDraft, 1)
	return &Drafter{maxNgram: n, maxDraft: k}
}

// Draft proposes the next tokens for `context` by prompt-lookup, without touching
// the drafter's running context (this is the stateless entry point). It tries the
// longest suffix first: for n from MaxNgram down to 1 it takes the last n tokens
// of the context and scans backwards for the most recent EARLIER occurrence of
// that exact n-gram; the first (longest-n, most-recent) match wins and the tokens
// that followed it — up to MaxDraft of them — are returned. No match at any n, or
// a context too short to have an earlier occurrence, yields an empty draft.
//
//	d.Draft([]int{1, 2, 3, 9, 1, 2, 3}) // suffix [1 2 3] seen earlier → [9 ...]
func (d *Drafter) Draft(context []int) []int {
	return lookup(context, d.maxNgram, d.maxDraft)
}

// lookup is the pure prompt-lookup core shared by Draft and DraftNext. It holds
// no state and reads nothing but its arguments, so it is trivially deterministic
// and race-free.
//
// It scans ONCE, anchored on the suffix's last token. The tokens an accepted match
// proposes always begin one past the earlier occurrence's END position, so the only
// thing the search decides is WHICH end position wins: the one supporting the
// longest suffix, and — among equally-long matches — the most recent. Every earlier
// occurrence ends on a token equal to context[L-1], so a single backwards scan over
// just those anchor positions finds the winner. The previous shape re-scanned the
// whole context once per suffix length n (maxNgram × L work on a miss); anchoring
// pays one L-length pass plus a short backwards extend only where the last token
// actually recurs — byte-identical drafts (gated by TestNgram_Lookup_Matches
// ReferenceFuzz against the naive reference), ~maxNgram× less scanning on the
// dominant no-repeat miss.
func lookup(context []int, maxNgram, maxDraft int) []int {
	L := len(context)
	if L < 2 {
		// Need at least one token of suffix AND one earlier token for it to
		// match: a 0- or 1-token context can never have an earlier occurrence.
		return nil
	}

	// Cap the suffix length to what the context can actually hold while still
	// leaving room for an earlier occurrence (suffix can be at most L-1 long).
	maxN := min(maxNgram, L-1)
	last := context[L-1] // every earlier occurrence ends on this same token

	bestN := 0
	bestFrom := -1
	// Candidate END positions, most-recent first. e is where an earlier occurrence
	// of the suffix would END; its match extends backwards while tokens keep
	// agreeing with the suffix's tail.
	for e := L - 2; e >= 0; e-- {
		if context[e] != last {
			continue
		}
		// Longest suffix this anchor supports: extend backwards while the tokens
		// agree, capped at maxN and stopping before the context start.
		n := 1
		for n < maxN && e-n >= 0 && context[e-n] == context[L-1-n] {
			n++
		}
		// Non-overlap: the n tokens ending at e occupy [e-n+1, e] and must end
		// strictly before the suffix [L-n, L) begins → e < L-n → n <= L-1-e.
		if lim := L - 1 - e; n > lim {
			n = lim
		}
		// Longest wins; most-recent breaks ties (we scan recent-first, so the first
		// anchor to reach a given n keeps it). Once an anchor hits the cap maxN no
		// earlier, less-recent anchor can beat it — stop.
		if n > bestN {
			bestN = n
			bestFrom = e + 1
			if bestN == maxN {
				break
			}
		}
	}
	if bestFrom < 0 {
		return nil
	}
	// The proposed tokens start one past the winning occurrence's end position.
	end := min(bestFrom+maxDraft, L)
	out := make([]int, end-bestFrom)
	copy(out, context[bestFrom:end])
	return out
}

// Update appends accepted tokens to the running context so later DraftNext calls
// see them (this is the stateful entry point — seed with the prompt, then append
// what the target accepts each step). A nil or empty slice is a no-op.
//
//	d.Update(promptTokens)   // seed
//	d.Update(acceptedTokens) // grow after each verification step
func (d *Drafter) Update(tokens []int) {
	if len(tokens) == 0 {
		return
	}
	d.mu.Lock()
	d.ctx = append(d.ctx, tokens...)
	d.mu.Unlock()
}

// DraftNext proposes the next tokens from the running context — it is exactly
// Draft(Context()), so the stateful and stateless paths never disagree. An empty
// running context yields an empty draft.
//
//	d.Update(promptTokens); next := d.DraftNext()
func (d *Drafter) DraftNext() []int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return lookup(d.ctx, d.maxNgram, d.maxDraft)
}

// Context returns a copy of the running context. It is a copy, not the live
// buffer, so a caller can read or mutate it without corrupting the drafter.
//
//	seen := d.Context()
func (d *Drafter) Context() []int {
	d.mu.Lock()
	defer d.mu.Unlock()
	if len(d.ctx) == 0 {
		return nil
	}
	out := make([]int, len(d.ctx))
	copy(out, d.ctx)
	return out
}

// Reset clears the running context so the drafter starts a fresh sequence,
// reusing its backing array (length zeroed, capacity kept) to avoid a realloc.
//
//	d.Reset() // begin a new generation with the same drafter
func (d *Drafter) Reset() {
	d.mu.Lock()
	d.ctx = d.ctx[:0]
	d.mu.Unlock()
}
