// SPDX-Licence-Identifier: EUPL-1.2

package slurs

import core "dappco.re/go"

// Tests use placeholder tokens, never real slurs — the mechanism is what's
// under test; the curated catalogue is Snider-reviewed data (catalogue.go).

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
