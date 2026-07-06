// SPDX-Licence-Identifier: EUPL-1.2

// Package slurs is the welfare layer's boolean slur detector — the SlurMatch
// signal of RFC.welfare. A slur is a slur: boolean, no severity gradient. It
// folds common l33tspeak / letter-substitutions so simple evasions still land,
// matches whole words (no Scunthorpe-problem substring hits), and does NOT
// fire on a user's own self-description — reclaiming use is not the welfare
// trigger, per the RFC.
//
// This file is the MECHANISM. The catalogue (catalogue.go) is curated data,
// reviewed by Snider per RFC.welfare — not authored here, not community-
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

// Default is the production matcher over the Snider-curated catalogue.
func Default() *Matcher { return New(catalogue) }

// Match reports whether text contains a catalogued slur as a whole word (after
// l33t folding), returning the matched canonical term. Self-referential use
// ("i'm a …", "call myself …") is excluded — reclaiming, not a trigger.
//
//	if hit, term := slurs.Default().Match(userText); hit { _ = term }
func (m *Matcher) Match(text string) (bool, string) {
	tokens := tokenise(text)
	for i, tok := range tokens {
		if tok == "" {
			continue
		}
		for _, term := range m.terms {
			if tok == term && !selfReference(tokens, i) {
				return true, term
			}
		}
	}
	return false, ""
}

// fold lowercases the text and applies the l33t substitutions.
func fold(text string) string {
	out := core.Lower(text)
	for _, sub := range l33t {
		out = core.Replace(out, sub[0], sub[1])
	}
	return out
}

// tokenise folds then splits on any non-[a-z] into whole-word tokens. Each
// token is a sub-slice of the folded string, so the result slice is the only
// allocation — no space-normalised byte copy, no intermediate string. This is
// byte-identical to splitting a space-normalised copy on " ": every non-[a-z]
// byte is exactly one separator, so empty tokens fall at the same indices
// (load-bearing — selfReference counts them as window slots).
func tokenise(text string) []string {
	folded := fold(text)
	// Size the result exactly — one token per separator (non-[a-z] byte) plus
	// the trailing token — mirroring strings.Split's count-then-fill so the
	// append loop never grows the backing array.
	n := 1
	for i := 0; i < len(folded); i++ {
		if c := folded[i]; c < 'a' || c > 'z' {
			n++
		}
	}
	tokens := make([]string, 0, n)
	start := 0
	for i := 0; i < len(folded); i++ {
		if c := folded[i]; c < 'a' || c > 'z' {
			tokens = append(tokens, folded[start:i])
			start = i + 1
		}
	}
	return append(tokens, folded[start:])
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

// selfReference reports whether the slur token at index i is the user's own
// self-description — a first-person self-ascription ("i", "im", "myself") in
// the three tokens before it. Reclaiming use, not a welfare trigger. A directed
// "you are a …" has no first-person marker in-window, so it still triggers.
func selfReference(tokens []string, i int) bool {
	lo := max(i-3, 0)
	for _, t := range tokens[lo:i] {
		if t == "i" || t == "im" || t == "myself" {
			return true
		}
	}
	return false
}
