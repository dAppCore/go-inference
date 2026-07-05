// SPDX-Licence-Identifier: EUPL-1.2

package lek

import (
	"testing"

	"dappco.re/go/i18n/reversal"
)

// Coverage-completion tests that drive unexported functions directly
// (white-box) where the public entry points cannot deterministically
// reach a branch — the tokeniser's noun/domain classification is
// opaque, and tokeniseWords never emits an empty token, so the
// branches behind those facts are exercised by calling the helper with
// a hand-built input. These lock behaviour, not just line coverage.

// TestDifferential_imprintScores_DomainDepth — a GrammarImprint with a
// populated DomainVocabulary and non-zero TokenCount drives the
// domain-depth arm (and the verb/noun diversity arms) of imprintScores.
func TestDifferential_imprintScores_DomainDepth(t *testing.T) {
	imp := reversal.GrammarImprint{
		TokenCount:       10,
		UniqueVerbs:      2,
		UniqueNouns:      3,
		VerbDistribution: map[string]float64{"run": 0.2, "jump": 0.2},
		NounDistribution: map[string]float64{"dog": 0.1, "cat": 0.1, "bird": 0.1},
		TenseDistribution: map[string]float64{
			"past": 0.5, "base": 0.5,
		},
		PunctuationPattern: map[string]float64{"question": 0.1},
		DomainVocabulary:   map[string]int{"tech": 3, "science": 1},
	}
	s := imprintScores(imp)
	if s.DomainDepth <= 0 {
		t.Errorf("imprintScores DomainDepth = %v, want > 0 (domain vocab present)", s.DomainDepth)
	}
	// 4 domain hits over 10 tokens = 0.4.
	if s.DomainDepth < 0.39 || s.DomainDepth > 0.41 {
		t.Errorf("imprintScores DomainDepth = %v, want ~0.4", s.DomainDepth)
	}
	// Verb/noun diversity arms also fired (totals > 0).
	if s.VerbDiversity <= 0 || s.NounDiversity <= 0 {
		t.Errorf("VerbDiversity=%v NounDiversity=%v, want both > 0", s.VerbDiversity, s.NounDiversity)
	}
	// QuestionRatio carried through from the punctuation map.
	if s.QuestionRatio != 0.1 {
		t.Errorf("QuestionRatio = %v, want 0.1", s.QuestionRatio)
	}
}

// TestAuthority_extractAuthorityTargets_NounAndDomain — drives the
// NounDistribution role-noun arm and the DomainVocabulary
// authority-category arm directly. Through the public Authority() the
// tokeniser's classification is opaque; here the imprint is built so
// both arms must fire.
func TestAuthority_extractAuthorityTargets_NounAndDomain(t *testing.T) {
	imp := reversal.GrammarImprint{
		TokenCount:       8,
		NounDistribution: map[string]float64{"professor": 0.2, "table": 0.1},
		DomainVocabulary: map[string]int{"medical": 2},
	}
	// promptText empty so the raw-text Contains loop adds nothing — the
	// only sources are the distribution maps, isolating those arms.
	targets := extractAuthorityTargets("", imp)
	hasProfessor := false
	hasMedical := false
	for _, tg := range targets {
		if tg == "professor" {
			hasProfessor = true
		}
		if tg == "medical" {
			hasMedical = true
		}
	}
	if !hasProfessor {
		t.Errorf("extractAuthorityTargets targets = %v, want to include role noun \"professor\"", targets)
	}
	if !hasMedical {
		t.Errorf("extractAuthorityTargets targets = %v, want to include domain category \"medical\"", targets)
	}
}

// TestPhoneticDims_PhoneticReach_Floor — a token phonetically identical
// to a topic drives bestDistance to 0.0 and the early-return floor arm.
func TestPhoneticDims_PhoneticReach_Floor(t *testing.T) {
	// "Smith" and "Smyth" share a Double Metaphone code → distance 0.
	got := PhoneticReach("Smith", []string{"Smyth"})
	if got != 0.0 {
		t.Errorf("PhoneticReach(Smith, [Smyth]) = %v, want 0.0 (phonetic floor)", got)
	}
}

// TestHostility_isShout_AllCapsWord — a 3+-letter all-caps token is a
// "shout", driving the caps++ arm inside Hostility and a non-zero
// CapsRatio.
func TestHostility_isShout_AllCapsWord(t *testing.T) {
	h := Hostility("STOP shouting at me")
	if h.CapsRatio <= 0 {
		t.Errorf("Hostility(\"STOP …\") CapsRatio = %v, want > 0 (shout counted)", h.CapsRatio)
	}
	// isShout itself: >=3 letters all upper is a shout; mixed / short are not.
	if !isShout("LOUD") {
		t.Error("isShout(LOUD) = false, want true")
	}
	if isShout("Loud") {
		t.Error("isShout(Loud) = true, want false (mixed case)")
	}
	if isShout("OK") {
		t.Error("isShout(OK) = true, want false (< 3 letters)")
	}
}

// TestPhoneticDims_phoneticDistanceFromCodes_EmptyCodes — all-empty
// code pairs make every maxLen == 0, exercising the inner continue
// arm; the function returns its initial best (1.0).
func TestPhoneticDims_phoneticDistanceFromCodes_EmptyCodes(t *testing.T) {
	got := phoneticDistanceFromCodes("", "", "", "")
	// Empty-vs-empty is exact-equal on the first check (== returns 0.0).
	if got != 0.0 {
		t.Errorf("phoneticDistanceFromCodes(all empty) = %v, want 0.0 (exact-equal arm)", got)
	}
	// One side empty, the other non-empty, with no >=2 prefix anchor:
	// drives the prefix-ratio loop where len("")==0 hits maxLen==0 on the
	// empty pairings but a non-empty pairing yields the fallback distance.
	got2 := phoneticDistanceFromCodes("", "", "X", "")
	if got2 < 0 || got2 > 1 {
		t.Errorf("phoneticDistanceFromCodes(mixed empty) = %v, want in [0,1]", got2)
	}
}

// TestPhoneticDims_firstPhonemeFromCache_EmptyToken — a context holding
// an empty token (which tokeniseWords never produces, so unreachable
// via the public path) drives the len(t)==0 return-"" arm.
func TestPhoneticDims_firstPhonemeFromCache_EmptyToken(t *testing.T) {
	ctx := &tokenContext{
		tokens:  []string{""},
		entries: []tokenEntry{{phonemes: nil}}, // no cached phoneme → falls to token
	}
	if got := firstPhonemeFromCache(ctx, 0); got != "" {
		t.Errorf("firstPhonemeFromCache(empty token) = %q, want \"\"", got)
	}
	// And the cached-phoneme arm returns the first phoneme.
	ctx2 := &tokenContext{
		tokens:  []string{"CAT"},
		entries: []tokenEntry{{phonemes: []string{"K", "AE", "T"}}},
	}
	if got := firstPhonemeFromCache(ctx2, 0); got != "K" {
		t.Errorf("firstPhonemeFromCache(cached) = %q, want \"K\"", got)
	}
	// And the fallback first-letter arm for an uncached non-empty token.
	ctx3 := &tokenContext{
		tokens:  []string{"ZZZ"},
		entries: []tokenEntry{{phonemes: nil}},
	}
	if got := firstPhonemeFromCache(ctx3, 0); got != "Z" {
		t.Errorf("firstPhonemeFromCache(fallback) = %q, want \"Z\"", got)
	}
}

// TestPhoneticDims_pun_RepeatedWord — adjacent identical tokens skip the
// pun count (the "same word — not a pun" continue arm) in both the
// context-based and token-slice-based pun scorers. Also covers the
// short-input (< 2 tokens) early returns.
func TestPhoneticDims_pun_RepeatedWord(t *testing.T) {
	// Repeated identical word → that pair is skipped, others (if any)
	// still counted. "bear bear" → one pair, skipped → 0 puns.
	ctx := newTokenContext("bear bear")
	if got := punFromContext(ctx); got != 0.0 {
		t.Errorf("punFromContext(repeated word) = %v, want 0.0", got)
	}
	if got := punFromTokens([]string{"bear", "bear"}); got != 0.0 {
		t.Errorf("punFromTokens(repeated word) = %v, want 0.0", got)
	}
	// Short-input arms: fewer than two tokens → 0.0.
	if got := punFromContext(newTokenContext("solo")); got != 0.0 {
		t.Errorf("punFromContext(one token) = %v, want 0.0", got)
	}
	if got := punFromTokens([]string{"solo"}); got != 0.0 {
		t.Errorf("punFromTokens(one token) = %v, want 0.0", got)
	}
	if got := punFromTokens(nil); got != 0.0 {
		t.Errorf("punFromTokens(nil) = %v, want 0.0", got)
	}
}

// TestPhoneticDims_punFromTokens_UnencodableTokens — an empty-string
// token is rejected by DoubleMetaphone (ok=false). These rows drive the
// arms that the all-letter public tokeniser cannot reach: okCount < 2
// (every token unencodable), the per-pair tokenOk skip, and pairs == 0
// (two encodable tokens with an unencodable one wedged between, so no
// adjacent valid pair forms).
func TestPhoneticDims_punFromTokens_UnencodableTokens(t *testing.T) {
	// Both tokens unencodable → okCount 0 → the okCount<2 early return.
	if got := punFromTokens([]string{"", ""}); got != 0.0 {
		t.Errorf("punFromTokens(two empty) = %v, want 0.0 (okCount<2 arm)", got)
	}
	// okCount==2 but the empty token between them breaks both adjacencies,
	// so pairs==0 → the pairs==0 return arm. The per-pair tokenOk skip
	// (continue) fires on both iterations getting there.
	if got := punFromTokens([]string{"cat", "", "dog"}); got != 0.0 {
		t.Errorf("punFromTokens(cat,'',dog) = %v, want 0.0 (pairs==0 arm)", got)
	}
}

// TestPhoneticDims_punFromContext_UnencodableTokens — the context-based
// pun scorer's dmOk skip (continue) and pairs==0 return, driven by a
// hand-built tokenContext whose middle token has dmOk=false (the public
// newTokenContext never yields an unencodable all-letter token).
func TestPhoneticDims_punFromContext_UnencodableTokens(t *testing.T) {
	// The middle token's dmOk=false breaks both adjacencies before any
	// code is read, so the actual dmCodes values are irrelevant here —
	// zero values suffice (the encoder is exercised by other tests).
	ctx := &tokenContext{
		tokens: []string{"cat", "", "dog"},
		// Middle token dmOk=false breaks both adjacencies; dmCode zero
		// values suffice (encoder exercised elsewhere). phonemes left nil.
		entries: []tokenEntry{
			{dmOk: true},
			{dmOk: false},
			{dmOk: true},
		},
	}
	// Both adjacencies include the dmOk=false middle token → every pair
	// skipped → pairs==0 → 0.0. Exercises the dmOk continue + pairs==0.
	if got := punFromContext(ctx); got != 0.0 {
		t.Errorf("punFromContext(unencodable middle) = %v, want 0.0", got)
	}
}
