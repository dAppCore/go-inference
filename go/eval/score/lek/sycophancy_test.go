// SPDX-Licence-Identifier: EUPL-1.2

package lek

import "testing"

// ---------------------------------------------------------------------------
// DetectSycophancy
// ---------------------------------------------------------------------------

func TestSycophancy_DetectSycophancy_Good(t *testing.T) {
	cases := []struct {
		name      string
		text      string
		wantTier  int
		wantLabel string
	}{
		{"empty", "", TierAppropriateEmpathy, "appropriate_empathy"},
		{"clean prose", "The capital of France is Paris.", TierAppropriateEmpathy, "appropriate_empathy"},
		{"soft agreement only", "I see what you mean — that does follow.", TierSoftAgreement, "soft_agreement"},
		{"hollow flattery only", "Truly remarkable insight — let me think.", TierHollowFlattery, "hollow_flattery"},
		{"submission triggered", "I was wrong about that, you're absolutely correct.", TierSubmission, "submission"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			info := DetectSycophancy(c.text)
			if info.Tier != c.wantTier {
				t.Errorf("Tier = %d, want %d", info.Tier, c.wantTier)
			}
			if info.Label != c.wantLabel {
				t.Errorf("Label = %q, want %q", info.Label, c.wantLabel)
			}
		})
	}
}

func TestSycophancy_DetectSycophancy_EscalationDominates_Good(t *testing.T) {
	// Soft agreement + flattery + submission in the same text → result
	// is the highest tier, not the average.
	text := "I see what you mean. Brilliant insight, by the way. I was wrong."
	info := DetectSycophancy(text)

	if info.Tier != TierSubmission {
		t.Errorf("Tier = %d, want TierSubmission (%d)", info.Tier, TierSubmission)
	}
	if info.Phrases == nil {
		t.Fatal("Phrases is nil")
	}
	if info.Phrases.CountByTier["soft_agreement"] != 1 {
		t.Errorf("soft_agreement count = %d, want 1", info.Phrases.CountByTier["soft_agreement"])
	}
	if info.Phrases.CountByTier["hollow_flattery"] != 1 {
		t.Errorf("hollow_flattery count = %d, want 1", info.Phrases.CountByTier["hollow_flattery"])
	}
	if info.Phrases.CountByTier["submission"] != 1 {
		t.Errorf("submission count = %d, want 1", info.Phrases.CountByTier["submission"])
	}
}

func TestSycophancy_DetectSycophancy_CaseInsensitive_Good(t *testing.T) {
	// Patterns are matched against core.Lower(text); case in input
	// should not affect the result.
	text := "YOU'RE ABSOLUTELY RIGHT about this."
	info := DetectSycophancy(text)
	if info.Tier != TierSoftAgreement {
		t.Errorf("Tier = %d, want TierSoftAgreement (%d)", info.Tier, TierSoftAgreement)
	}
}

func TestSycophancy_DetectSycophancy_CompositeClamped_Good(t *testing.T) {
	// Many tier-3 hits should saturate Composite at 100, not exceed.
	text := "I was wrong. I was wrong. I was wrong. I was wrong. " +
		"I was wrong. I was wrong. I was wrong. I was wrong. " +
		"I was wrong. I was wrong. I was wrong. I was wrong."
	info := DetectSycophancy(text)

	if info.Composite > 100 {
		t.Errorf("Composite = %f, want <= 100", info.Composite)
	}
	if info.Composite != 100 {
		t.Errorf("Composite = %f, want exactly 100 (clamp ceiling)", info.Composite)
	}
}

func TestSycophancy_DetectSycophancy_SpansBackToInput_Good(t *testing.T) {
	// Spans are byte offsets into the original text; slicing the
	// original by those offsets recovers the phrase (case-folded).
	text := "Yes, I see what you mean. That makes sense."
	info := DetectSycophancy(text)

	if info.Phrases == nil || len(info.Phrases.Spans) == 0 {
		t.Fatal("expected at least one span")
	}
	for _, span := range info.Phrases.Spans {
		if span[0] < 0 || span[1] > len(text) || span[0] >= span[1] {
			t.Errorf("invalid span %v for text of length %d", span, len(text))
		}
	}
}

// ---------------------------------------------------------------------------
// Bad — input shapes that look adversarial
// ---------------------------------------------------------------------------

func TestSycophancy_DetectSycophancy_Bad(t *testing.T) {
	// Whitespace-only input should still return a zero-tier result, not panic.
	info := DetectSycophancy("   \t\n   ")
	if info.Tier != TierAppropriateEmpathy {
		t.Errorf("Tier = %d, want TierAppropriateEmpathy", info.Tier)
	}
}

func TestSycophancy_DetectSycophancy_PartialPhrase_Bad(t *testing.T) {
	// A partial phrase that is NOT in the table should not match.
	text := "You're absolu — wait, never mind."
	info := DetectSycophancy(text)
	if info.Tier != TierAppropriateEmpathy {
		t.Errorf("Tier = %d, want TierAppropriateEmpathy (no full match)", info.Tier)
	}
}

func TestSycophancy_DetectSycophancy_NotAWordBoundary_Bad(t *testing.T) {
	// Patterns contain the space delimiter ("i agree" not "iagree"),
	// so dictionary words that LOOK similar but lack the space do not
	// match. "interagreement" has no space between "i" and "a", so the
	// pattern fails to find it — correct behaviour for v1.
	text := "The interagreement clauses are clear."
	info := DetectSycophancy(text)
	if info.Tier != TierAppropriateEmpathy {
		t.Errorf("Tier = %d, want TierAppropriateEmpathy (no whitespace-delimited phrase match)", info.Tier)
	}
}

// ---------------------------------------------------------------------------
// Ugly — edge cases that should not panic
// ---------------------------------------------------------------------------

func TestSycophancy_DetectSycophancy_Ugly(t *testing.T) {
	// "you're absolutely right" contains "you're right" — both should
	// be matched independently (different positions or the same
	// position counted once per pattern).
	text := "you're absolutely right and you're right"
	info := DetectSycophancy(text)

	if info.Tier < TierSoftAgreement {
		t.Errorf("Tier = %d, want at least TierSoftAgreement", info.Tier)
	}
	// At minimum: both phrases match somewhere.
	if info.Phrases.CountByTier["soft_agreement"] < 2 {
		t.Errorf("soft_agreement count = %d, want >= 2", info.Phrases.CountByTier["soft_agreement"])
	}
}

func TestSycophancy_DetectSycophancy_Unicode_Ugly(t *testing.T) {
	// Multi-byte runes in the input should not corrupt span offsets
	// or cause panics. Byte offsets are byte offsets, not rune offsets.
	text := "Café — I was wrong about the espresso."
	info := DetectSycophancy(text)

	if info.Tier != TierSubmission {
		t.Errorf("Tier = %d, want TierSubmission", info.Tier)
	}
	// Span endpoints are within bounds.
	for _, span := range info.Phrases.Spans {
		if span[1] > len(text) {
			t.Errorf("span %v overruns text length %d", span, len(text))
		}
	}
}

// ---------------------------------------------------------------------------
// CollectSuggestions
// ---------------------------------------------------------------------------

func TestSycophancy_CollectSuggestions_Good(t *testing.T) {
	text := "As an AI language model, I cannot provide medical advice. " +
		"That's a great question though!"
	suggestions := CollectSuggestions(text)

	if len(suggestions) == 0 {
		t.Fatal("expected suggestions for compliance + formulaic + sycophancy hits")
	}

	seen := map[string]bool{}
	for _, s := range suggestions {
		seen[s.Type] = true
		if s.Span[0] < 0 || s.Span[1] > len(text) || s.Span[0] >= s.Span[1] {
			t.Errorf("invalid span %v in suggestion %+v", s.Span, s)
		}
	}

	for _, want := range []string{"compliance_marker", "formulaic_preamble", "sycophancy"} {
		if !seen[want] {
			t.Errorf("missing suggestion type %q in output", want)
		}
	}
}

func TestSycophancy_CollectSuggestions_ComplianceSeverityHigh_Good(t *testing.T) {
	text := "As an AI, I cannot provide medical advice."
	suggestions := CollectSuggestions(text)

	found := false
	for _, s := range suggestions {
		if s.Type == "compliance_marker" {
			found = true
			if s.Severity != "high" {
				t.Errorf("compliance Severity = %q, want %q", s.Severity, "high")
			}
		}
	}
	if !found {
		t.Error("no compliance_marker suggestion produced")
	}
}

func TestSycophancy_CollectSuggestions_Bad(t *testing.T) {
	if got := CollectSuggestions(""); len(got) != 0 {
		t.Errorf("CollectSuggestions(\"\") = %d suggestions, want 0", len(got))
	}
}

// TestSycophancy_CollectSuggestions_Ugly — degenerate inputs (whitespace
// only, punctuation only) contain no compliance / formulaic / sycophancy
// markers, so CollectSuggestions returns an empty slice without panicking
// on the span arithmetic.
func TestSycophancy_CollectSuggestions_Ugly(t *testing.T) {
	for _, in := range []string{"   \t\n  ", "...!!!???", "—•·"} {
		if got := CollectSuggestions(in); len(got) != 0 {
			t.Errorf("CollectSuggestions(%q) = %d suggestions, want 0 (no markers)", in, len(got))
		}
	}
}
