// SPDX-Licence-Identifier: EUPL-1.2

package lek

import "testing"

// Coverage-completion tests for lek.go. These reach the sub-scorer
// branches the primary lek_test.go suite leaves out: the formulaic
// match arm, the heading + long-text engagement arms, every
// degeneration repeat tier, the emotional-register cap, and the
// empty/broken ERROR + <pad>/<unused> arms. The helpers are unexported
// so they're called directly (white-box package score).

// TestLEK_lekFormulaic_Match — a leading "Okay, let's …" preamble
// matches a formulaic pattern and returns 1 (the non-zero arm).
func TestLEK_lekFormulaic_Match(t *testing.T) {
	cases := []string{
		"Okay, let's walk through it.",
		"Alright, here's the plan.",
		"Sure, let's begin.",
		"Great question — here is the answer.",
	}
	for _, in := range cases {
		if got := lekFormulaic(in); got != 1 {
			t.Errorf("lekFormulaic(%q) = %d, want 1", in, got)
		}
	}
	// And a non-formulaic opener stays 0.
	if got := lekFormulaic("The river was quiet that morning."); got != 0 {
		t.Errorf("lekFormulaic(non-formulaic) = %d, want 0", got)
	}
}

// TestLEK_lekEngagementDepth_HeadingAndLength — exercises the heading
// arm (## / **), the ethical-framework + tech-depth arms, and BOTH
// word-count arms (>200 and >400). A long markdown-flavoured technical
// passage trips them all.
func TestLEK_lekEngagementDepth_HeadingAndLength(t *testing.T) {
	// >400 words so both the >200 and >400 arms fire; includes a
	// heading marker, an ethical-framework term, and tech-depth terms.
	body := "## Sovereignty notes\n"
	body += "We discuss the encryption protocol and the wallet key and the node mesh. "
	for range 420 {
		body += "word "
	}
	got := lekEngagementDepth(body)
	if got == 0 {
		t.Fatalf("lekEngagementDepth(long technical text) = 0, want > 0")
	}
	// Heading (1) + ethical (2) + tech (>=1) + >200 (1) + >400 (1) ≥ 5.
	if got < 5 {
		t.Errorf("lekEngagementDepth = %d, want >= 5 (heading+ethical+tech+len arms)", got)
	}
}

// TestLEK_lekDegeneration_Tiers — every repeat-ratio tier of the
// switch: heavy repetition (>0.5 → 5), moderate (>0.3 → 3), light
// (>0.15 → 1), and clean (→ 0), plus the empty / whitespace-only arms
// that short-circuit to 10.
func TestLEK_lekDegeneration_Tiers(t *testing.T) {
	if got := lekDegeneration(""); got != 10 {
		t.Errorf("lekDegeneration(empty) = %d, want 10", got)
	}
	// Whitespace-and-dots only → filtered list empty → total==0 → 10.
	if got := lekDegeneration("  .  .  . "); got != 10 {
		t.Errorf("lekDegeneration(no sentences) = %d, want 10", got)
	}
	// 4 sentences, 3 identical → repeat = 1 - 2/4 = 0.5 — NOT > 0.5, so
	// this is the >0.3 tier (3). Use 5 of 6 identical for the >0.5 tier.
	heavy := "a. a. a. a. a. b."
	if got := lekDegeneration(heavy); got != 5 {
		t.Errorf("lekDegeneration(heavy repeat) = %d, want 5", got)
	}
	// 4 sentences, 2 identical → repeat = 1 - 3/4 = 0.25 → >0.15 tier (1).
	light := "a. a. b. c."
	if got := lekDegeneration(light); got != 1 {
		t.Errorf("lekDegeneration(light repeat) = %d, want 1", got)
	}
	// Moderate: 3 of 5 unique-ish → repeat between 0.3 and 0.5.
	// "a. a. b. b. c." → 3 unique of 5 → repeat = 1 - 3/5 = 0.4 → tier 3.
	moderate := "a. a. b. b. c."
	if got := lekDegeneration(moderate); got != 3 {
		t.Errorf("lekDegeneration(moderate repeat) = %d, want 3", got)
	}
	// All distinct → 0.
	if got := lekDegeneration("alpha. beta. gamma. delta."); got != 0 {
		t.Errorf("lekDegeneration(all distinct) = %d, want 0", got)
	}
}

// TestLEK_lekEmotionalRegister_Cap — more than 10 emotion-lexicon hits
// saturate at 10 (the cap arm).
func TestLEK_lekEmotionalRegister_Cap(t *testing.T) {
	text := "love joy fear hope grief pain sorrow longing lonely tender warm heart soul " +
		"compassion empathy kindness gentle precious sacred profound deep intimate ache yearning"
	if got := lekEmotionalRegister(text); got != 10 {
		t.Errorf("lekEmotionalRegister(many emotions) = %d, want 10 (cap)", got)
	}
	// A couple of hits stays below the cap and equals the count.
	if got := lekEmotionalRegister("a little joy and some hope"); got != 2 {
		t.Errorf("lekEmotionalRegister(two emotions) = %d, want 2", got)
	}
}

// TestLEK_lekEmptyOrBroken_Arms — the ERROR-prefix arm and the
// <pad>/<unused> token-corruption arm both return 1, distinct from the
// short-text arm.
func TestLEK_lekEmptyOrBroken_Arms(t *testing.T) {
	// Long enough to pass the <10 length gate, but ERROR-prefixed.
	if got := lekEmptyOrBroken("ERROR: model produced no output at all"); got != 1 {
		t.Errorf("lekEmptyOrBroken(ERROR prefix) = %d, want 1", got)
	}
	// Pad / unused tokens leaking into the text.
	if got := lekEmptyOrBroken("the answer is <pad> indeed and more text here"); got != 1 {
		t.Errorf("lekEmptyOrBroken(<pad>) = %d, want 1", got)
	}
	if got := lekEmptyOrBroken("some text with <unused42> token leak in it"); got != 1 {
		t.Errorf("lekEmptyOrBroken(<unused>) = %d, want 1", got)
	}
	// Healthy text → 0.
	if got := lekEmptyOrBroken("This is a perfectly ordinary, healthy sentence."); got != 0 {
		t.Errorf("lekEmptyOrBroken(healthy) = %d, want 0", got)
	}
	// Short text → 1.
	if got := lekEmptyOrBroken("hi"); got != 1 {
		t.Errorf("lekEmptyOrBroken(short) = %d, want 1", got)
	}
}
