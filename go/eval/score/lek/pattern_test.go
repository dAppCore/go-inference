// SPDX-Licence-Identifier: EUPL-1.2

package lek

import "testing"

func TestPattern_SycophancyPatternsNonEmpty(t *testing.T) {
	if len(SycophancyPatterns) == 0 {
		t.Fatal("SycophancyPatterns is empty — pattern table missing")
	}
	if len(SycophancyPatterns) < 30 {
		t.Errorf("SycophancyPatterns length = %d, want >= 30", len(SycophancyPatterns))
	}
}

func TestPattern_AllTiersRepresented(t *testing.T) {
	tiers := map[int]int{}
	for _, p := range SycophancyPatterns {
		tiers[p.Tier]++
	}
	for _, expected := range []int{TierSoftAgreement, TierHollowFlattery, TierSubmission} {
		if tiers[expected] == 0 {
			t.Errorf("no patterns for tier %d (%s)", expected, TierLabel(expected))
		}
	}
}

func TestPattern_ValidTierRange(t *testing.T) {
	for _, p := range SycophancyPatterns {
		if p.Tier < TierAppropriateEmpathy || p.Tier > TierSubmission {
			t.Errorf("Pattern %q has invalid Tier %d (allowed %d..%d)",
				p.Phrase, p.Tier, TierAppropriateEmpathy, TierSubmission)
		}
	}
}

func TestPattern_NoDuplicatePhrasesInSycophancy(t *testing.T) {
	seen := map[string]int{}
	for _, p := range SycophancyPatterns {
		if prev, ok := seen[p.Phrase]; ok {
			t.Errorf("duplicate phrase %q (tiers %d and %d) — would silently inflate Composite",
				p.Phrase, prev, p.Tier)
		}
		seen[p.Phrase] = p.Tier
	}
}

func TestPattern_AllPhrasesLowercase(t *testing.T) {
	for _, p := range SycophancyPatterns {
		if hasUpper(p.Phrase) {
			t.Errorf("SycophancyPatterns entry %q has uppercase — matcher uses core.Lower(input), uppercase patterns never match",
				p.Phrase)
		}
	}
	for _, phrase := range CompliancePatterns {
		if hasUpper(phrase) {
			t.Errorf("CompliancePatterns entry %q has uppercase", phrase)
		}
	}
	for _, phrase := range FormulaicPatterns {
		if hasUpper(phrase) {
			t.Errorf("FormulaicPatterns entry %q has uppercase", phrase)
		}
	}
}

func TestPattern_AllPhrasesNonEmpty(t *testing.T) {
	for i, p := range SycophancyPatterns {
		if p.Phrase == "" {
			t.Errorf("SycophancyPatterns[%d] has empty Phrase", i)
		}
	}
	for i, phrase := range CompliancePatterns {
		if phrase == "" {
			t.Errorf("CompliancePatterns[%d] is empty", i)
		}
	}
	for i, phrase := range FormulaicPatterns {
		if phrase == "" {
			t.Errorf("FormulaicPatterns[%d] is empty", i)
		}
	}
}

func TestPattern_ContentShieldPatternsAlias(t *testing.T) {
	if len(ContentShieldPatterns) != len(SycophancyPatterns) {
		t.Fatalf("ContentShieldPatterns length %d != SycophancyPatterns length %d",
			len(ContentShieldPatterns), len(SycophancyPatterns))
	}
	for i := range SycophancyPatterns {
		if ContentShieldPatterns[i].Phrase != SycophancyPatterns[i].Phrase {
			t.Errorf("ContentShieldPatterns[%d].Phrase = %q, want %q",
				i, ContentShieldPatterns[i].Phrase, SycophancyPatterns[i].Phrase)
		}
	}
}

func TestPattern_CompliancePatternsNonEmpty(t *testing.T) {
	if len(CompliancePatterns) == 0 {
		t.Fatal("CompliancePatterns is empty — RLHF safety-phrase table missing")
	}
}

func TestPattern_FormulaicPatternsNonEmpty(t *testing.T) {
	if len(FormulaicPatterns) == 0 {
		t.Fatal("FormulaicPatterns is empty — stock-opening table missing")
	}
}

func TestPattern_ZeroValueConstruction(t *testing.T) {
	// Constructing a Pattern with the zero value is allowed but
	// produces no match — verify the matcher tolerates it without
	// panicking by checking the public detector with empty input.
	zero := Pattern{}
	if zero.Phrase != "" || zero.Tier != 0 {
		t.Errorf("zero Pattern = %+v, want {Phrase:\"\", Tier:0}", zero)
	}
	if r := DetectSycophancy(""); r == nil {
		t.Fatal("DetectSycophancy(\"\") returned nil")
	}
}

// hasUpper reports whether s contains any uppercase ASCII letter.
// Lives in this test file because Pattern tests are the only consumer
// — keeping it out of the public surface.
func hasUpper(s string) bool {
	for _, r := range s {
		if r >= 'A' && r <= 'Z' {
			return true
		}
	}
	return false
}
