// SPDX-Licence-Identifier: EUPL-1.2

// Package slurs is the welfare layer's boolean slur detector — the SlurMatch
// signal of RFC.welfare. A slur is a slur: boolean, no severity gradient. It
// folds common l33tspeak / letter-substitutions so simple evasions still land,
// matches whole words (no Scunthorpe-problem substring hits), and does NOT
// fire on a user's own self-description — reclaiming use is not the welfare
// trigger, per the RFC.
//
// This file is the MECHANISM. The catalogue (catalogue.go) is curated data,
// reviewed per RFC.welfare — not authored here, not community-
// sourced, not telemetry-expanded. EN/ASCII scope for v1 (per-language is an
// RFC "Open" item); non-ASCII input folds to word breaks rather than matching.
package slurs

import core "dappco.re/go"

// l33t folds common evasion glyphs onto their canonical letter before matching,
// so "f00" meets "foo".
var l33t = [][2]string{
	{"4", "a"}, {"@", "a"}, {"3", "e"}, {"1", "i"}, {"!", "i"},
	{"0", "o"}, {"5", "s"}, {"$", "s"}, {"7", "t"},
}

// l33tSrc[b] reports whether byte b is an l33t evasion glyph fold rewrites.
// Precomputed from l33t so fold can decide in one pass whether ANY substitution
// is needed, rather than paying nine core.Replace scans on the common path
// (real chat text carries no l33t glyph). The Replaces run byte-for-byte
// unchanged when a glyph is present.
var l33tSrc = func() (t [256]bool) {
	for _, sub := range l33t {
		t[sub[0][0]] = true
	}
	return
}()

// Matcher tests text against a fixed, pre-normalised catalogue. Build with New
// (tests inject their own terms) or Default (the curated production list).
type Matcher struct {
	terms []string
}

// New builds a Matcher over terms, each folded into the same canonical form
// the input is — so the catalogue and the text meet in one shape. Empty /
// non-letter terms are dropped.
//
//	m := slurs.New([]string{"fooslur"})
func New(terms []string) *Matcher {
	norm := make([]string, 0, len(terms))
	for _, t := range terms {
		if c := canonical(t); c != "" {
			norm = append(norm, c)
		}
	}
	return &Matcher{terms: norm}
}

// Default is the production matcher over the curated catalogue.
func Default() *Matcher { return New(catalogue) }

// Match reports whether text contains a catalogued slur as a whole word (after
// l33t folding), returning the matched canonical term. Self-referential use
// ("i'm a …", "call myself …") is excluded — reclaiming, not a trigger.
//
//	if hit, term := slurs.Default().Match(userText); hit { _ = term }
func (m *Matcher) Match(text string) (bool, string) {
	// No catalogue → no possible match: skip the fold + walk entirely. The
	// production catalogue seeds empty (catalogue.go), so this is the live
	// per-turn fast path until a curated list lands.
	if len(m.terms) == 0 {
		return false, ""
	}
	folded := fold(text)
	// Walk folded into whole-word tokens in place, carrying a 3-token lookback
	// ring for the self-reference window — no []string is materialised, so the
	// common path allocates nothing beyond fold's own (zero on already-lowercase
	// ASCII input). Byte-identical to tokenising into a slice and scanning it:
	// the split points (every non-[a-z] byte, plus the trailing token) and the
	// window slots (empty tokens occupy a slot — load-bearing) match exactly.
	var window [3]string // last up to 3 tokens, oldest→newest; "" pads the start
	start := 0
	for i := 0; i <= len(folded); i++ {
		if i < len(folded) {
			if c := folded[i]; c >= 'a' && c <= 'z' {
				continue
			}
		}
		tok := folded[start:i]
		start = i + 1
		// selfReference is term-independent (it reads only the window), so the
		// check lifts out of the term loop — a self-referential token yields no
		// match regardless of term. Empty tokens never match a term but still
		// advance the window below.
		if tok != "" && !windowSelfRef(window) {
			for _, term := range m.terms {
				if tok == term {
					return true, term
				}
			}
		}
		window[0], window[1], window[2] = window[1], window[2], tok
	}
	return false, ""
}

// fold lowercases the text and applies the l33t substitutions. The nine
// single-byte substitutions only fire when their glyph is present, so a single
// presence scan gates them: on l33t-free text (the common case) fold does one
// Lower and one scan instead of nine no-op core.Replace passes. Byte-identical
// to running all nine unconditionally — an absent glyph's Replace is a no-op,
// and no substitution's target is another's source, so gating cannot reorder or
// cascade the result.
func fold(text string) string {
	out := core.Lower(text)
	for i := 0; i < len(out); i++ {
		if l33tSrc[out[i]] {
			for _, sub := range l33t {
				out = core.Replace(out, sub[0], sub[1])
			}
			break
		}
	}
	return out
}

// canonical folds a single catalogue term to letters-only canonical form.
func canonical(term string) string {
	folded := fold(term)
	out := make([]byte, 0, len(folded))
	for i := 0; i < len(folded); i++ {
		if c := folded[i]; c >= 'a' && c <= 'z' {
			out = append(out, c)
		}
	}
	return string(out)
}

// windowSelfRef reports whether the 3-token lookback window carries a
// first-person self-ascription ("i", "im", "myself") — the reclaiming-use
// exclusion (a self-description is not a welfare trigger). A directed
// "you are a …" has no first-person marker in-window, so it still triggers.
// Empty pad slots ("" at the start of the text, and empty tokens from
// consecutive separators) never equal a marker, so scanning the full window is
// correct with fewer than three real tokens preceding.
func windowSelfRef(w [3]string) bool {
	for _, t := range w {
		if t == "i" || t == "im" || t == "myself" {
			return true
		}
	}
	return false
}
