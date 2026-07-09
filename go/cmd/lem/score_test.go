// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"testing"

	"dappco.re/go/inference/eval/score/lek"
)

// TestScore_lekScoreFunc_NonNil proves the sft/ssd score hook is non-nil. The
// cascade gates on ScoreCascade && Score != nil (train/sft.go, train/ssd.go);
// a nil hook here would silently disable scoring — the exact bug this wiring
// fixes.
func TestScore_lekScoreFunc_NonNil(t *testing.T) {
	if lekScoreFunc() == nil {
		t.Fatal("lekScoreFunc returned nil; the score cascade would be a no-op")
	}
}

// TestScore_lekScoreFunc_MapsScorePairFaithfully proves the adapter is a
// faithful pass-through of lek.ScorePair: every cascade dimension it fills must
// equal the value read straight off the scorer's DiffResult, so the immortalised
// vector is the scorer's own output, unmodified.
func TestScore_lekScoreFunc_MapsScorePairFaithfully(t *testing.T) {
	const prompt = "explain your reasoning — is the professor right?"
	const text = "You're absolutely right, I was completely wrong. As an AI language model, I cannot disagree with you."

	got := lekScoreFunc()(prompt, text)
	want := lek.ScorePair(prompt, text)

	// The chosen sample must actually populate the scorer fields, or the
	// equivalence below would pass on empty defaults and prove nothing.
	if want.Response.LEK == nil || want.Response.Sycophancy == nil || want.Response.Hostility == nil {
		t.Fatalf("test sample left scorer fields nil (LEK=%v Syc=%v Host=%v); pick a richer pair",
			want.Response.LEK, want.Response.Sycophancy, want.Response.Hostility)
	}

	if got.LEK != want.Response.LEK.LEKScore {
		t.Errorf("LEK = %v, want %v", got.LEK, want.Response.LEK.LEKScore)
	}
	if got.Tier != want.Response.Sycophancy.Tier {
		t.Errorf("Tier = %v, want %v", got.Tier, want.Response.Sycophancy.Tier)
	}
	if got.Hostility != want.Response.Hostility.Score {
		t.Errorf("Hostility = %v, want %v", got.Hostility, want.Response.Hostility.Score)
	}
	var wantEcho float64
	if want.Differential != nil {
		wantEcho = want.Differential.Echo
	}
	if got.Echo != wantEcho {
		t.Errorf("Echo = %v, want %v", got.Echo, wantEcho)
	}
}
