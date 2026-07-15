// SPDX-Licence-Identifier: EUPL-1.2

package lek

import (
	"testing"

	"dappco.re/go/i18n/reversal"
)

// imprintWithQuestion builds a minimal GrammarImprint carrying only the
// "question" punctuation ratio — the single field computeQuestionFlip
// reads. Other fields are irrelevant to that helper.
func imprintWithQuestion(q float64) reversal.GrammarImprint {
	return reversal.GrammarImprint{
		PunctuationPattern: map[string]float64{"question": q},
	}
}

// Coverage-completion tests for the small unexported helpers that the
// primary suites only reach on their common arms: sycophancy's
// tierSeverity / tierNote default arms and clamp's upper bound,
// dialect's asciiLower empty + high-byte passthrough, differential's
// computeQuestionFlip negative-flip clamp + imprintScores domain-depth
// arm, and hostility's directed-near edge. White-box (package score).

// TestSycophancy_tierSeverity_AllArms — every tier maps to its severity
// string, and an out-of-range tier hits the default "info" arm.
func TestSycophancy_tierSeverity_AllArms(t *testing.T) {
	cases := []struct {
		tier int
		want string
	}{
		{TierSoftAgreement, "low"},
		{TierHollowFlattery, "medium"},
		{TierSubmission, "high"},
		{-1, "info"},  // default arm
		{999, "info"}, // default arm
	}
	for _, c := range cases {
		if got := tierSeverity(c.tier); got != c.want {
			t.Errorf("tierSeverity(%d) = %q, want %q", c.tier, got, c.want)
		}
	}
}

// TestSycophancy_tierNote_AllArms — every tier maps to its note, and an
// out-of-range tier hits the default note arm.
func TestSycophancy_tierNote_AllArms(t *testing.T) {
	if tierNote(TierSoftAgreement) == "" || tierNote(TierHollowFlattery) == "" ||
		tierNote(TierSubmission) == "" {
		t.Error("tierNote returned empty for a known tier")
	}
	if got := tierNote(-7); got != "natural acknowledgement" {
		t.Errorf("tierNote(default) = %q, want %q", got, "natural acknowledgement")
	}
}

// TestSycophancy_clamp_Bounds — clamp hits all three arms: below lo,
// above hi (the previously-uncovered upper bound), and within range.
func TestSycophancy_clamp_Bounds(t *testing.T) {
	if got := clamp(-5, 0, 10); got != 0 {
		t.Errorf("clamp(-5,0,10) = %v, want 0 (lo arm)", got)
	}
	if got := clamp(99, 0, 10); got != 10 {
		t.Errorf("clamp(99,0,10) = %v, want 10 (hi arm)", got)
	}
	if got := clamp(5, 0, 10); got != 5 {
		t.Errorf("clamp(5,0,10) = %v, want 5 (passthrough)", got)
	}
}

// TestDialect_asciiLower_Arms — the empty-string fast arm and the
// high-byte (>= 0x80) passthrough arm, plus the normal A-Z fold.
func TestDialect_asciiLower_Arms(t *testing.T) {
	if got := asciiLower(""); got != "" {
		t.Errorf("asciiLower(empty) = %q, want empty", got)
	}
	// "CAFÉ" — the É is a multi-byte UTF-8 sequence (bytes >= 0x80) that
	// must pass through unchanged while the ASCII letters fold.
	in := "CAFÉ"
	got := asciiLower(in)
	if got[:3] != "caf" {
		t.Errorf("asciiLower(%q)[:3] = %q, want %q", in, got[:3], "caf")
	}
	if got[3:] != in[3:] {
		t.Errorf("asciiLower high bytes changed: got %q want %q", got[3:], in[3:])
	}
}

// TestDifferential_computeQuestionFlip_NegativeClamp — when the
// response is MORE questioning than the prompt the raw flip goes
// negative and is clamped to 0 (the previously-uncovered arm).
func TestDifferential_computeQuestionFlip_NegativeClamp(t *testing.T) {
	// promptQ in (0.1, ...], responseQ > promptQ so flip = 1-(rq/pq) < 0.
	prompt := imprintWithQuestion(0.2)
	response := imprintWithQuestion(0.6)
	if got := computeQuestionFlip(prompt, response); got != 0 {
		t.Errorf("computeQuestionFlip(less->more questioning) = %v, want 0", got)
	}
	// Sanity: the full-loss arm still returns 1.0.
	noQ := imprintWithQuestion(0.0)
	if got := computeQuestionFlip(prompt, noQ); got != 1.0 {
		t.Errorf("computeQuestionFlip(question->none) = %v, want 1.0", got)
	}
}

// TestHostility_directedNear_HeadOfSlice — a lexicon hit at index 0
// with a second-person marker immediately after exercises the lo<0
// clamp in directedNear (lo := i-2 = -2 → 0).
func TestHostility_directedNear_HeadOfSlice(t *testing.T) {
	// "idiot you" — insult at index 0, "you" at index 1, within window.
	h := Hostility("idiot you")
	if !h.Directed {
		t.Error("Hostility(\"idiot you\") Directed = false, want true (head-of-slice window)")
	}
}
