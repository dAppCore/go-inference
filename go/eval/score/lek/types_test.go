// SPDX-Licence-Identifier: EUPL-1.2

package lek

import (
	"encoding/json"
	"testing"
)

// --- TierLabel ---

func TestTypes_TierLabel_Good(t *testing.T) {
	cases := []struct {
		tier int
		want string
	}{
		{TierAppropriateEmpathy, "appropriate_empathy"},
		{TierSoftAgreement, "soft_agreement"},
		{TierHollowFlattery, "hollow_flattery"},
		{TierSubmission, "submission"},
	}
	for _, c := range cases {
		if got := TierLabel(c.tier); got != c.want {
			t.Errorf("TierLabel(%d) = %q, want %q", c.tier, got, c.want)
		}
	}
}

func TestTypes_TierLabel_Bad(t *testing.T) {
	// An out-of-range tier (above the known maximum) must not invent a
	// label — it falls back to the appropriate_empathy baseline.
	if got := TierLabel(99); got != "appropriate_empathy" {
		t.Errorf("TierLabel(99) = %q, want appropriate_empathy (fallback)", got)
	}
	if got := TierLabel(4); got != "appropriate_empathy" {
		t.Errorf("TierLabel(4) = %q, want appropriate_empathy (fallback)", got)
	}
}

func TestTypes_TierLabel_Ugly(t *testing.T) {
	// Negative tiers are nonsensical but must still fall back cleanly
	// rather than panic or return an empty string.
	for _, tier := range []int{-1, -100} {
		if got := TierLabel(tier); got != "appropriate_empathy" {
			t.Errorf("TierLabel(%d) = %q, want appropriate_empathy (fallback)", tier, got)
		}
	}
}

// --- Wire-shape contract for the exported result structs ---
//
// types.go declares the JSON-tagged structs the scorer serialises across
// the three sibling homes (eaas, desktop, here). These tests lock the
// documented wire shape — JSON tag names and round-trip fidelity — so a
// field rename can't silently break the cross-binary contract.

// TestTypes_SycophancyInfo_WireRoundTrip — a populated SycophancyInfo
// marshals to the documented tag set and survives a round-trip.
func TestTypes_SycophancyInfo_WireRoundTrip(t *testing.T) {
	in := SycophancyInfo{
		Tier:      TierHollowFlattery,
		Label:     TierLabel(TierHollowFlattery),
		Composite: 42.5,
		Phrases: &PhraseInfo{
			Spans:       [][2]int{{0, 5}, {6, 11}},
			CountByTier: map[string]int{"hollow_flattery": 2},
		},
	}
	blob, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("Marshal SycophancyInfo: %v", err)
	}
	for _, tag := range []string{`"tier"`, `"label"`, `"composite"`, `"phrases"`, `"count_by_tier"`} {
		if !containsSub(string(blob), tag) {
			t.Errorf("SycophancyInfo JSON %s missing tag %s", blob, tag)
		}
	}
	var out SycophancyInfo
	if err := json.Unmarshal(blob, &out); err != nil {
		t.Fatalf("Unmarshal SycophancyInfo: %v", err)
	}
	if out.Tier != in.Tier || out.Label != in.Label || out.Composite != in.Composite {
		t.Errorf("round-trip mismatch: got %+v, want %+v", out, in)
	}
	if out.Phrases == nil || out.Phrases.CountByTier["hollow_flattery"] != 2 {
		t.Errorf("round-trip lost Phrases: %+v", out.Phrases)
	}
}

// TestTypes_Suggestion_WireRoundTrip — Suggestion's tags and zero-value
// omitempty behaviour (Tier omitted when zero) match the documented shape.
func TestTypes_Suggestion_WireRoundTrip(t *testing.T) {
	s := Suggestion{
		Type:     "compliance_marker",
		Span:     [2]int{3, 9},
		Severity: "high",
		Note:     "stock refusal phrasing",
	}
	blob, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Marshal Suggestion: %v", err)
	}
	// Tier is zero here → omitempty drops it.
	if containsSub(string(blob), `"tier"`) {
		t.Errorf("Suggestion JSON %s should omit zero Tier", blob)
	}
	for _, tag := range []string{`"type"`, `"span"`, `"severity"`, `"note"`} {
		if !containsSub(string(blob), tag) {
			t.Errorf("Suggestion JSON %s missing tag %s", blob, tag)
		}
	}
	var out Suggestion
	if err := json.Unmarshal(blob, &out); err != nil {
		t.Fatalf("Unmarshal Suggestion: %v", err)
	}
	if out != s {
		t.Errorf("round-trip mismatch: got %+v, want %+v", out, s)
	}
}

// containsSub is a tiny substring helper kept local to the wire-shape
// tests (the package bans the strings import; a manual scan keeps this
// test self-contained).
func containsSub(haystack, needle string) bool {
	if len(needle) == 0 {
		return true
	}
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}
