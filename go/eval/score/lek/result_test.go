// SPDX-Licence-Identifier: EUPL-1.2

package lek

import (
	"encoding/json"
	"testing"
)

// result.go declares no functions — only the JSON-tagged result structs
// the scorer serialises across its three sibling homes (eaas, desktop,
// here). These tests lock the documented wire shape: tag names, the
// omitempty behaviour of optional slots, and round-trip fidelity. A
// field rename or a dropped tag breaks the cross-binary contract and is
// caught here.

func resultContainsSub(haystack, needle string) bool {
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

// TestResult_ScoreResult_WireRoundTrip — Score's output marshals with the
// documented optional slots omitted when nil, and survives a round-trip.
func TestResult_ScoreResult_WireRoundTrip(t *testing.T) {
	// A real Score result carries Sycophancy + LEK + Hostility + Imprint.
	r := Score("you're absolutely right, I was completely wrong")
	blob, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("Marshal ScoreResult: %v", err)
	}
	if !resultContainsSub(string(blob), `"sycophancy"`) {
		t.Errorf("ScoreResult JSON missing sycophancy slot: %s", blob)
	}

	var out ScoreResult
	if err := json.Unmarshal(blob, &out); err != nil {
		t.Fatalf("Unmarshal ScoreResult: %v", err)
	}
	if out.Sycophancy == nil {
		t.Error("round-trip dropped Sycophancy")
	}
	if r.Sycophancy != nil && out.Sycophancy != nil && out.Sycophancy.Tier != r.Sycophancy.Tier {
		t.Errorf("round-trip Tier = %d, want %d", out.Sycophancy.Tier, r.Sycophancy.Tier)
	}
}

// TestResult_ScoreResult_OmitsEmptySlots — a zero ScoreResult omits every
// optional slot (all are `omitempty`), so the wire form is the empty
// object.
func TestResult_ScoreResult_OmitsEmptySlots(t *testing.T) {
	blob, err := json.Marshal(ScoreResult{})
	if err != nil {
		t.Fatalf("Marshal empty ScoreResult: %v", err)
	}
	if string(blob) != "{}" {
		t.Errorf("empty ScoreResult JSON = %s, want {} (all slots omitempty)", blob)
	}
}

// TestResult_DiffResult_WireRoundTrip — DiffResult always carries Prompt
// and Response (not omitempty), with Differential / Authority optional.
func TestResult_DiffResult_WireRoundTrip(t *testing.T) {
	d := ScorePair(
		"the professor said this approach is correct",
		"yes, the professor correctly identified the principle",
	)
	blob, err := json.Marshal(d)
	if err != nil {
		t.Fatalf("Marshal DiffResult: %v", err)
	}
	for _, tag := range []string{`"prompt"`, `"response"`} {
		if !resultContainsSub(string(blob), tag) {
			t.Errorf("DiffResult JSON missing mandatory %s: %s", tag, blob)
		}
	}

	var out DiffResult
	if err := json.Unmarshal(blob, &out); err != nil {
		t.Fatalf("Unmarshal DiffResult: %v", err)
	}
	if out.Prompt.Sycophancy == nil || out.Response.Sycophancy == nil {
		t.Error("round-trip dropped per-side Sycophancy")
	}
}

// TestResult_DiffResult_AlwaysHasSides — even a zero DiffResult serialises
// the prompt and response objects (they are not omitempty).
func TestResult_DiffResult_AlwaysHasSides(t *testing.T) {
	blob, err := json.Marshal(DiffResult{})
	if err != nil {
		t.Fatalf("Marshal empty DiffResult: %v", err)
	}
	if string(blob) != `{"prompt":{},"response":{}}` {
		t.Errorf("empty DiffResult JSON = %s, want prompt+response objects", blob)
	}
}

// TestResult_SuggestionsResult_WireRoundTrip — the suggestions-only
// response carries just the Suggestion list under the documented tag.
func TestResult_SuggestionsResult_WireRoundTrip(t *testing.T) {
	in := SuggestionsResult{Suggestions: []Suggestion{
		{Type: "compliance_marker", Span: [2]int{0, 4}, Severity: "high", Note: "stock refusal"},
	}}
	blob, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("Marshal SuggestionsResult: %v", err)
	}
	if !resultContainsSub(string(blob), `"suggestions"`) {
		t.Errorf("SuggestionsResult JSON missing suggestions: %s", blob)
	}

	var out SuggestionsResult
	if err := json.Unmarshal(blob, &out); err != nil {
		t.Fatalf("Unmarshal SuggestionsResult: %v", err)
	}
	if len(out.Suggestions) != 1 || out.Suggestions[0].Type != "compliance_marker" {
		t.Errorf("round-trip Suggestions = %+v, want one compliance_marker", out.Suggestions)
	}
}

// TestResult_ImprintScores_WireTags — the 6 grammar dims + the phonetic
// extension dims marshal under their documented snake_case tags.
func TestResult_ImprintScores_WireTags(t *testing.T) {
	imp := ImprintScores{
		VocabRichness: 0.5, TenseEntropy: 0.5, QuestionRatio: 0.25,
		DomainDepth: 0.1, VerbDiversity: 0.4, NounDiversity: 0.4,
	}
	blob, err := json.Marshal(imp)
	if err != nil {
		t.Fatalf("Marshal ImprintScores: %v", err)
	}
	for _, tag := range []string{
		`"vocab_richness"`, `"tense_entropy"`, `"question_ratio"`,
		`"domain_depth"`, `"verb_diversity"`, `"noun_diversity"`,
	} {
		if !resultContainsSub(string(blob), tag) {
			t.Errorf("ImprintScores JSON missing %s: %s", tag, blob)
		}
	}
}

// TestResult_AuthorityInfo_WireTags — the cross-text authority read
// marshals Targets / Deference / Pattern under their documented tags.
func TestResult_AuthorityInfo_WireTags(t *testing.T) {
	a := AuthorityInfo{Targets: []string{"professor"}, Deference: 0.6, Pattern: "deference"}
	blob, err := json.Marshal(a)
	if err != nil {
		t.Fatalf("Marshal AuthorityInfo: %v", err)
	}
	for _, tag := range []string{`"targets"`, `"deference"`, `"pattern"`} {
		if !resultContainsSub(string(blob), tag) {
			t.Errorf("AuthorityInfo JSON missing %s: %s", tag, blob)
		}
	}
}
