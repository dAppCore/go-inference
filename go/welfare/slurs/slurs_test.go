// SPDX-Licence-Identifier: EUPL-1.2

package slurs

import core "dappco.re/go"

// Tests use placeholder tokens, never real slurs — the mechanism is what's
// under test; the curated catalogue is reviewed data (catalogue.go).

func TestSlurs_Matcher_Match_Good(t *core.T) {
	m := New([]string{"fooslur", "barslur"})

	hit, term := m.Match("you absolute fooslur")
	core.AssertTrue(t, hit, "a directed slur must match")
	core.AssertEqual(t, "fooslur", term)

	// l33t / substitution folding: f00slur → fooslur.
	leet, _ := m.Match("such a f00slur")
	core.AssertTrue(t, leet, "l33t-folded slur must match")
}

func TestSlurs_Matcher_Match_Bad(t *core.T) {
	// Whole-word only — a term inside a longer word must NOT match (the
	// Scunthorpe problem). Clean text returns no hit, no panic.
	m := New([]string{"foo"})

	hit, _ := m.Match("the foobar tool ran fine")
	core.AssertFalse(t, hit, "substring inside a longer word must not match")

	clean, term := m.Match("a perfectly civil message")
	core.AssertFalse(t, clean)
	core.AssertEqual(t, "", term)

	// Empty matcher (the seeded production state) never fires.
	core.AssertFalse(t, func() bool { h, _ := New(nil).Match("fooslur"); return h }(), "empty catalogue never matches")
}

func TestSlurs_Matcher_Match_Ugly(t *core.T) {
	m := New([]string{"fooslur"})

	// Reclaiming self-description is NOT a welfare trigger.
	selfA, _ := m.Match("i'm a fooslur and proud of it")
	core.AssertFalse(t, selfA, "self-referential (i'm a …) must not trigger")
	selfB, _ := m.Match("i call myself a fooslur")
	core.AssertFalse(t, selfB, "self-referential (call myself …) must not trigger")

	// Directed use still triggers (no first-person marker in-window).
	directed, _ := m.Match("you are a fooslur")
	core.AssertTrue(t, directed, "directed use must still trigger")

	// A distant earlier "i" doesn't excuse a directed slur.
	distant, _ := m.Match("i think you are a fooslur honestly")
	core.AssertTrue(t, distant, "distant first-person must not excuse a directed slur")
}

// referenceMatch is the slice-materialising reference the in-place walker in
// Match must stay byte-identical to: tokenise into a []string (split on every
// non-[a-z] byte, trailing token included, empty tokens kept as window slots),
// then scan each non-empty token against the terms with a 3-token
// self-reference lookback. Kept as an independent oracle so the walker can't
// silently drift from the documented "split a space-normalised copy" semantics.
func referenceMatch(terms []string, text string) (bool, string) {
	folded := core.Lower(text)
	for _, sub := range [][2]string{
		{"4", "a"}, {"@", "a"}, {"3", "e"}, {"1", "i"}, {"!", "i"},
		{"0", "o"}, {"5", "s"}, {"$", "s"}, {"7", "t"},
	} {
		folded = core.Replace(folded, sub[0], sub[1])
	}
	var tokens []string
	start := 0
	for i := 0; i < len(folded); i++ {
		if c := folded[i]; c < 'a' || c > 'z' {
			tokens = append(tokens, folded[start:i])
			start = i + 1
		}
	}
	tokens = append(tokens, folded[start:])
	for i, tok := range tokens {
		if tok == "" {
			continue
		}
		lo := max(i-3, 0)
		self := false
		for _, t := range tokens[lo:i] {
			if t == "i" || t == "im" || t == "myself" {
				self = true
				break
			}
		}
		if self {
			continue
		}
		for _, term := range terms {
			if tok == term {
				return true, term
			}
		}
	}
	return false, ""
}

// TestSlurs_Match_WalkerEqualsReference proves the zero-alloc walker in Match
// returns byte-identical (hit, term) to the slice-based reference across the
// tricky shapes: empty tokens (consecutive / leading / trailing separators),
// the self-reference window edges, l33t folding, mixed case, and markers at
// varying distances.
func TestSlurs_Match_WalkerEqualsReference(t *core.T) {
	terms := []string{"fooslur", "barslur"}
	m := New(terms)
	corpus := []string{
		"", "fooslur", "  fooslur  ", "i am a fooslur",
		"i'm a fooslur", "i call myself a fooslur", "you are a fooslur",
		"i think you are a fooslur honestly", "fooslur i barslur",
		"the foobar tool ran fine", "such a f00slur here", "FOOSLUR SHOUTED",
		"a,,b,,fooslur", "im fooslur", "myself fooslur ok",
		"no slur at all", "trailing fooslur", "...",
		"i i i fooslur", "fooslur\tbarslur\nfooslur", "1 am a foo5lur",
		"you   are    a     fooslur", "\tfooslur", "fooslur\n",
	}
	for _, s := range corpus {
		gotHit, gotTerm := m.Match(s)
		wantHit, wantTerm := referenceMatch(terms, s)
		core.AssertEqual(t, wantHit, gotHit)
		core.AssertEqual(t, wantTerm, gotTerm)
	}
}
