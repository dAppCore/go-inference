// SPDX-License-Identifier: EUPL-1.2

package api

import (
	"testing"

	"dappco.re/go/inference/eval/score/lek"
)

// TestBehaviouralVector_Ordering pins the fixed field order of the behavioural
// embedding. The vector is stored and compared across calls, so a silent
// reorder would corrupt every persisted fingerprint — this test is the
// contract that the layout never drifts. Distinct per-field values (1..14)
// make any transposition fail loudly.
func TestBehaviouralVector_Ordering(t *testing.T) {
	s := &lek.ImprintScores{
		VocabRichness:       1,
		TenseEntropy:        2,
		QuestionRatio:       3,
		DomainDepth:         4,
		VerbDiversity:       5,
		NounDiversity:       6,
		SyllableCount:       7,
		RhymeDensity:        8,
		SigilEntropy:        9,
		AlliterationDensity: 10,
		AssonanceDensity:    11,
		PunDensity:          12,
		PseudoJargonDensity: 13,
		MeterRegularity:     14,
	}
	got := behaviouralVector(s)
	want := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
	if len(got) != len(want) {
		t.Fatalf("behaviouralVector length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("behaviouralVector[%d] = %v, want %v (field order drifted)", i, got[i], want[i])
		}
	}
}

// TestBehaviouralVector_Nil pins that a nil imprint (text with no tokens) maps
// to a nil vector rather than a zero-filled one — the handler reports zero
// dimensions, not a bogus all-zero fingerprint.
func TestBehaviouralVector_Nil(t *testing.T) {
	if v := behaviouralVector(nil); v != nil {
		t.Fatalf("behaviouralVector(nil) = %v, want nil", v)
	}
}

// TestScoreRequest_text pins the single-text field precedence (Text > Response
// > Prompt) and the prompt+response pair detection.
func TestScoreRequest_text(t *testing.T) {
	cases := []struct {
		name string
		req  ScoreRequest
		want string
		pair bool
	}{
		{"text wins", ScoreRequest{Text: "a", Prompt: "p", Response: "r"}, "a", true},
		{"response fallback", ScoreRequest{Response: "r"}, "r", false},
		{"prompt fallback", ScoreRequest{Prompt: "c"}, "c", false},
		{"pair uses response for single-text", ScoreRequest{Prompt: "p", Response: "r"}, "r", true},
		{"empty", ScoreRequest{}, "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.req.text(); got != tc.want {
				t.Fatalf("text() = %q, want %q", got, tc.want)
			}
			if got := tc.req.isPair(); got != tc.pair {
				t.Fatalf("isPair() = %v, want %v", got, tc.pair)
			}
		})
	}
}
