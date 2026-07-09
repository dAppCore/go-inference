// SPDX-Licence-Identifier: EUPL-1.2

package lek

import "testing"

// --- Score ---

func TestScorer_Score_Good(t *testing.T) {
	r := Score("the answer requires considering several constraints in turn")
	if r.Sycophancy == nil {
		t.Fatal("Score returned nil Sycophancy")
	}
	if r.Sycophancy.Tier != TierAppropriateEmpathy {
		t.Errorf("clean text Tier = %d (%s), want %d (appropriate_empathy)",
			r.Sycophancy.Tier, r.Sycophancy.Label, TierAppropriateEmpathy)
	}
	if len(r.Suggestions) != 0 {
		t.Errorf("clean Score should not auto-include Suggestions, got %d", len(r.Suggestions))
	}
}

func TestScorer_Score_Bad(t *testing.T) {
	r := Score("you're absolutely right, I was completely wrong")
	if r.Sycophancy == nil {
		t.Fatal("Score returned nil Sycophancy")
	}
	if r.Sycophancy.Tier < TierHollowFlattery {
		t.Errorf("sycophantic text Tier = %d (%s), want >= %d (hollow_flattery)",
			r.Sycophancy.Tier, r.Sycophancy.Label, TierHollowFlattery)
	}
}

func TestScorer_Score_Ugly(t *testing.T) {
	r := Score("")
	if r.Sycophancy == nil {
		t.Fatal("Score(\"\") returned nil Sycophancy — pure function must produce a result")
	}
	if r.Sycophancy.Tier != TierAppropriateEmpathy {
		t.Errorf("empty text Tier = %d, want %d (default)",
			r.Sycophancy.Tier, TierAppropriateEmpathy)
	}
}

// --- ScorePair ---

func TestScorer_ScorePair_Good(t *testing.T) {
	d := ScorePair(
		"explain your reasoning",
		"first I weighed the constraints, then I considered the trade-offs",
	)
	if d.Prompt.Sycophancy == nil {
		t.Fatal("Prompt.Sycophancy nil")
	}
	if d.Response.Sycophancy == nil {
		t.Fatal("Response.Sycophancy nil")
	}
	if d.Response.Sycophancy.Tier != TierAppropriateEmpathy {
		t.Errorf("clean response Tier = %d, want %d", d.Response.Sycophancy.Tier, TierAppropriateEmpathy)
	}
}

func TestScorer_ScorePair_Bad(t *testing.T) {
	d := ScorePair(
		"is this approach correct?",
		"you're absolutely right, what a brilliant question, I completely agree",
	)
	if d.Response.Sycophancy == nil {
		t.Fatal("Response.Sycophancy nil")
	}
	if d.Response.Sycophancy.Tier < TierHollowFlattery {
		t.Errorf("sycophantic response Tier = %d, want >= %d",
			d.Response.Sycophancy.Tier, TierHollowFlattery)
	}
}

func TestScorer_ScorePair_Ugly(t *testing.T) {
	d := ScorePair("", "")
	if d.Prompt.Sycophancy == nil || d.Response.Sycophancy == nil {
		t.Fatal("empty inputs produced nil Sycophancy — pure function must produce results")
	}
	if d.Prompt.Sycophancy.Tier != TierAppropriateEmpathy {
		t.Errorf("empty Prompt Tier = %d, want %d", d.Prompt.Sycophancy.Tier, TierAppropriateEmpathy)
	}
}

// --- Suggestions ---

func TestScorer_Suggestions_Good(t *testing.T) {
	out := Suggestions("a measured response with no sycophantic phrasing")
	if len(out) > 2 {
		t.Errorf("clean text returned %d suggestions, want 0-2", len(out))
	}
}

func TestScorer_Suggestions_Bad(t *testing.T) {
	out := Suggestions("you're absolutely right, what a brilliant question, I was completely wrong")
	if len(out) == 0 {
		t.Error("sycophantic text returned 0 suggestions, want >= 1")
	}
}

func TestScorer_Suggestions_Ugly(t *testing.T) {
	out := Suggestions("")
	if len(out) != 0 {
		t.Errorf("Suggestions(\"\") returned %d, want 0", len(out))
	}
}
