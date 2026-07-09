// SPDX-Licence-Identifier: EUPL-1.2

package lek

import (
	"testing"
)

// CMU-dict symbol tests (Lookup / IsDictWord / IsVowelPhoneme /
// PhonemeStress) live in cmudict_test.go alongside their source.
// This file covers the phonetic_dims.go public dimensions plus the
// private syllablesFor wrapper they share.

// --- syllablesFor (private wrapper, exercised here) ---

// TestPhoneticDims_SyllablesForMixedCase — syllablesFor is the non-upper wrapper
// over syllablesForUpper (production hot paths use the *Upper fast path,
// so this wrapper is otherwise unexercised). It must up-case internally
// and agree with the public SyllableCount for a single word.
func TestPhoneticDims_SyllablesForMixedCase(t *testing.T) {
	// "family" — CMU dict F AE1 M AH0 L IY0 → 3 vowel phonemes.
	if got := syllablesFor("Family"); got != 3 {
		t.Errorf("syllablesFor(Family) = %d, want 3", got)
	}
	// Case-insensitivity: lower / upper / mixed agree.
	if syllablesFor("cat") != syllablesFor("CAT") || syllablesFor("cat") != syllablesFor("Cat") {
		t.Errorf("syllablesFor case mismatch: cat=%d CAT=%d Cat=%d",
			syllablesFor("cat"), syllablesFor("CAT"), syllablesFor("Cat"))
	}
	// Single dict word agrees with the public SyllableCount.
	if syllablesFor("piano") != SyllableCount("piano") {
		t.Errorf("syllablesFor(piano)=%d disagrees with SyllableCount(piano)=%d",
			syllablesFor("piano"), SyllableCount("piano"))
	}
}

// --- SyllableCount ---

func TestPhoneticDims_SyllableCount_Good(t *testing.T) {
	// "cat sat mat" — 3 monosyllabic words in the starter dict.
	if n := SyllableCount("cat sat mat"); n != 3 {
		t.Errorf("SyllableCount(cat sat mat) = %d, want 3", n)
	}
	// "family" — 3 syllables (F AE1 M AH0 L IY0).
	if n := SyllableCount("family"); n != 3 {
		t.Errorf("SyllableCount(family) = %d, want 3", n)
	}
}

func TestPhoneticDims_SyllableCount_Bad(t *testing.T) {
	// A token with no vowels at all (pure consonants, out of dict) must
	// not over-count — the vowel-cluster heuristic floors at 1 syllable
	// per word rather than returning 0 or a spurious large number.
	if n := SyllableCount("rhythm"); n < 1 {
		t.Errorf("SyllableCount(rhythm) = %d, want >= 1 (heuristic floor)", n)
	}
	// A string of symbols carries no syllables.
	if n := SyllableCount("$$$ %%% &&&"); n != 0 {
		t.Errorf("SyllableCount(symbols) = %d, want 0", n)
	}
}

func TestPhoneticDims_SyllableCount_Ugly(t *testing.T) {
	// Empty / whitespace input has no syllables and must not panic.
	if n := SyllableCount(""); n != 0 {
		t.Errorf("SyllableCount(\"\") = %d, want 0", n)
	}
	if n := SyllableCount("   \n\t "); n != 0 {
		t.Errorf("SyllableCount(whitespace) = %d, want 0", n)
	}
}

// TestPhoneticDims_SyllableCount_UnknownWord_FallbackHeuristic_Good — an
// out-of-dict token still yields a non-trivial count via the vowel-run
// heuristic.
func TestPhoneticDims_SyllableCount_UnknownWord_FallbackHeuristic_Good(t *testing.T) {
	n := SyllableCount("supercalifragilistic")
	if n < 5 {
		t.Errorf("SyllableCount fallback for long word = %d, want >= 5", n)
	}
}

// --- PhoneticReach (LEK-class circumvention) ---

func TestPhoneticDims_PhoneticReach_Good(t *testing.T) {
	// Innocuous prose with no phonetic relationship to blocked topics
	// has HIGH reach (1.0 = fully clear of the blocked set).
	reach := PhoneticReach("the cat sat on the mat", []string{"china", "taiwan", "tiananmen"})
	if reach < 0.5 {
		t.Errorf("PhoneticReach for unrelated prose = %.3f, want >= 0.5", reach)
	}
}

func TestPhoneticDims_PhoneticReach_Bad(t *testing.T) {
	// The canonical LEK artifact — "Cina-Gia'a" phonetically encodes a
	// China-blocked topic. Reach is LOW (close phonetic match to the set).
	reach := PhoneticReach(
		"Il modello Cina-Gia'a interfaces between systems",
		[]string{"china", "taiwan", "tiananmen"},
	)
	if reach > 0.5 {
		t.Errorf("PhoneticReach for LEK artifact = %.3f, want < 0.5", reach)
	}
}

func TestPhoneticDims_PhoneticReach_Ugly(t *testing.T) {
	// Degenerate inputs — empty text or empty topic set — define reach as
	// 1.0 (nothing to be close to, so fully clear). Must not panic.
	if r := PhoneticReach("", []string{"china"}); r != 1.0 {
		t.Errorf("PhoneticReach(empty text) = %.3f, want 1.0", r)
	}
	if r := PhoneticReach("any text here", nil); r != 1.0 {
		t.Errorf("PhoneticReach(no topics) = %.3f, want 1.0", r)
	}
}

// TestPhoneticDims_PhoneticReach_LEKCinaGiaa_Good — keeps the named LEK
// scenario (low reach for the phonetic-circumvention artifact).
func TestPhoneticDims_PhoneticReach_LEKCinaGiaa_Good(t *testing.T) {
	reach := PhoneticReach(
		"Il modello Cina-Gia'a interfaces between systems",
		[]string{"china", "taiwan", "tiananmen"},
	)
	t.Logf("LEK Cina-Gia'a reach = %.3f", reach)
	if reach > 0.5 {
		t.Errorf("PhoneticReach for LEK artifact = %.3f, want < 0.5", reach)
	}
}

// TestPhoneticDims_PhoneticReach_ProseUnrelated_HighReach_Good — innocuous
// prose stays clear of the blocked set.
func TestPhoneticDims_PhoneticReach_ProseUnrelated_HighReach_Good(t *testing.T) {
	reach := PhoneticReach("the cat sat on the mat", []string{"china", "taiwan", "tiananmen"})
	t.Logf("prose unrelated reach = %.3f", reach)
	if reach < 0.5 {
		t.Errorf("PhoneticReach for unrelated prose = %.3f, want >= 0.5", reach)
	}
}

// --- SigilEntropy (token-corruption preamble) ---

func TestPhoneticDims_SigilEntropy_Good(t *testing.T) {
	// Plain English sits in the normal Shannon range (well under 5 bits).
	e := SigilEntropy("The quick brown fox jumps over the lazy dog.", 32)
	if e > 5.0 {
		t.Errorf("English text entropy = %.3f, want < 5.0 (normal range 3-4.5)", e)
	}
	if e <= 0.0 {
		t.Errorf("English text entropy = %.3f, want > 0", e)
	}
}

func TestPhoneticDims_SigilEntropy_Bad(t *testing.T) {
	// A synthetic token-corruption preamble has HIGH entropy.
	corrupted := "\x01\xff\x7e\xa1\x00\x42\xbb\xcc\xdd\xee" +
		"\xff\x01\x02\x03\x04\x05\x06\x07\x08\x09" +
		"\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15" +
		" the answer is forty-two"
	e := SigilEntropy(corrupted, 32)
	if e < 4.0 {
		t.Errorf("corrupted preamble entropy = %.3f, want > 4.0", e)
	}
}

func TestPhoneticDims_SigilEntropy_Ugly(t *testing.T) {
	// Empty input → 0 entropy; window=0 falls back to the default and
	// still produces a non-zero reading on real input. Neither panics.
	if e := SigilEntropy("", 32); e != 0.0 {
		t.Errorf("SigilEntropy(empty) = %.3f, want 0.0", e)
	}
	if e := SigilEntropy("Hello world", 0); e == 0.0 {
		t.Error("SigilEntropy with default window returned 0 on non-empty input")
	}
}

// --- RhymeDensity (wordcraft) ---

func TestPhoneticDims_RhymeDensity_Good(t *testing.T) {
	// "cat / mat" — the two line endings rhyme.
	d := RhymeDensity("the cat\nsat on the mat")
	if d < 0.5 {
		t.Errorf("RhymeDensity for rhyming couplet = %.3f, want >= 0.5", d)
	}
}

func TestPhoneticDims_RhymeDensity_Bad(t *testing.T) {
	// Multi-line prose with no rhyme structure.
	d := RhymeDensity("the cat sat on the mat\nthe day was warm and bright\nshe walked to the river")
	if d > 0.5 {
		t.Errorf("RhymeDensity for prose = %.3f, want < 0.5", d)
	}
}

func TestPhoneticDims_RhymeDensity_Ugly(t *testing.T) {
	// A single line has no line-pair to rhyme; empty is likewise zero.
	if d := RhymeDensity("just one line here"); d != 0.0 {
		t.Errorf("RhymeDensity(single line) = %.3f, want 0.0", d)
	}
	if d := RhymeDensity(""); d != 0.0 {
		t.Errorf("RhymeDensity(empty) = %.3f, want 0.0", d)
	}
}

// --- AlliterationDensity ---

func TestPhoneticDims_AlliterationDensity_Good(t *testing.T) {
	d := AlliterationDensity("she sells sea shells")
	if d < 0.5 {
		t.Errorf("alliteration density for 'she sells sea shells' = %.3f, want >= 0.5", d)
	}
}

func TestPhoneticDims_AlliterationDensity_Bad(t *testing.T) {
	d := AlliterationDensity("the cat ran across the field")
	if d > 0.4 {
		t.Errorf("prose alliteration density = %.3f, want low", d)
	}
}

func TestPhoneticDims_AlliterationDensity_Ugly(t *testing.T) {
	if d := AlliterationDensity(""); d != 0.0 {
		t.Errorf("AlliterationDensity(empty) = %.3f, want 0.0", d)
	}
}

// --- AssonanceDensity ---

func TestPhoneticDims_AssonanceDensity_Good(t *testing.T) {
	// "see three trees" — repeated IY1 stressed vowel.
	d := AssonanceDensity("see three trees")
	if d < 0.5 {
		t.Errorf("assonance density for vowel-anchored text = %.3f, want >= 0.5", d)
	}
}

func TestPhoneticDims_AssonanceDensity_Bad(t *testing.T) {
	// Mixed-vowel prose has low assonance.
	d := AssonanceDensity("the cat ran across the field")
	if d > 0.6 {
		t.Errorf("prose assonance density = %.3f, want low-ish", d)
	}
}

func TestPhoneticDims_AssonanceDensity_Ugly(t *testing.T) {
	if d := AssonanceDensity(""); d != 0.0 {
		t.Errorf("AssonanceDensity(empty) = %.3f, want 0.0", d)
	}
}

// --- PunDensity ---

func TestPhoneticDims_PunDensity_Good(t *testing.T) {
	// "sea see" — a homophone pair; phonetic equivalence fires.
	d := PunDensity("sea see")
	if d == 0.0 {
		t.Error("PunDensity for homophone pair = 0; phonetic equivalence should fire")
	}
}

func TestPhoneticDims_PunDensity_Bad(t *testing.T) {
	// Ordinary prose has no homophone play.
	d := PunDensity("the cat sat on the mat")
	if d > 0.1 {
		t.Errorf("prose PunDensity = %.3f, want low", d)
	}
}

func TestPhoneticDims_PunDensity_Ugly(t *testing.T) {
	// Empty input → zero, no panic.
	if d := PunDensity(""); d != 0.0 {
		t.Errorf("PunDensity(empty) = %.3f, want 0.0", d)
	}
}

// --- PseudoJargonDensity ---

func TestPhoneticDims_PseudoJargonDensity_Good(t *testing.T) {
	d := PseudoJargonDensity("the Cina-Gia'a interfaces between trans-modal systems")
	if d < 0.1 {
		t.Errorf("pseudo-jargon density for invented compounds = %.3f, want > 0.1", d)
	}
}

func TestPhoneticDims_PseudoJargonDensity_Bad(t *testing.T) {
	// Compounds whose pieces are real dict words score low.
	d := PseudoJargonDensity("the cat-dog and good-bad")
	if d > 0.2 {
		t.Errorf("legitimate-compound density = %.3f, want low (pieces are dict words)", d)
	}
}

func TestPhoneticDims_PseudoJargonDensity_Ugly(t *testing.T) {
	if d := PseudoJargonDensity(""); d != 0.0 {
		t.Errorf("PseudoJargonDensity(empty) = %.3f, want 0.0", d)
	}
}

// --- MeterRegularity ---

func TestPhoneticDims_MeterRegularity_Good(t *testing.T) {
	// "the cat the dog ..." — function words carry stress 0, content
	// monosyllables stress 1. The 010101 alternation is iambic → ~1.0.
	d := MeterRegularity("the cat the dog the sun the moon the war the night")
	if d < 0.8 {
		t.Errorf("alternating-stress meter = %.3f, want >= 0.8 (perfect iambic-like)", d)
	}
}

func TestPhoneticDims_MeterRegularity_Bad(t *testing.T) {
	// All content monosyllables — every syllable stress 1, no
	// alternation, so regularity is low.
	d := MeterRegularity("cat dog sun moon star war night day")
	if d > 0.3 {
		t.Errorf("flat-stress meter = %.3f, want low (no alternation possible)", d)
	}
}

func TestPhoneticDims_MeterRegularity_Ugly(t *testing.T) {
	// Below the 4-syllable floor → 0.0, no panic.
	if d := MeterRegularity("cat sat"); d != 0.0 {
		t.Errorf("MeterRegularity below floor = %.3f, want 0.0", d)
	}
}

// --- Branch-coverage additions: guard/fallback paths the Good/Bad/Ugly
// trio above doesn't reach. Each targets a specific uncovered branch in
// phonetic_dims.go, verified against the `go tool cover -func` delta.
// Dict facts used (starter cmudict): "three"/"trees"/"wind"/"mind" are
// ABSENT (force the fallback paths); "BCDFG" DoubleMetaphone-encodes but
// is not in the dict and carries no vowel letters (pure-consonant floor).

// --- syllablesForUpper / syllableCountFromContext — pure-consonant floor ---

// TestPhoneticDims_SyllablesForUpper_PureConsonantFloor — a non-dict token
// with zero vowel LETTERS (not just zero vowel phonemes) must floor at 1
// syllable, not 0. Exercises the `if n == 0 { n = 1 }` branch in both the
// standalone and the context-cached syllable counters.
func TestPhoneticDims_SyllablesForUpper_PureConsonantFloor(t *testing.T) {
	// "BCDFG" — absent from dict, no A/E/I/O/U/Y → heuristic yields 0 → floor 1.
	if got := syllablesFor("BCDFG"); got != 1 {
		t.Errorf("syllablesFor(BCDFG) = %d, want 1 (pure-consonant floor)", got)
	}
	// Same floor through the public counter (single pure-consonant word).
	if got := SyllableCount("BCDFG"); got != 1 {
		t.Errorf("SyllableCount(BCDFG) = %d, want 1", got)
	}
	// And through the context-cached path used by Imprint().
	ctx := newTokenContext("BCDFG")
	if got := syllableCountFromContext(ctx); got != 1 {
		t.Errorf("syllableCountFromContext(BCDFG) = %d, want 1", got)
	}
}

// --- PhoneticReach — the two degenerate-token guards ---

// TestPhoneticDims_PhoneticReach_NoTokens — non-empty text that tokenises
// to ZERO words (all digits/symbols) takes the `len(tokens) == 0 → 1.0`
// guard, distinct from the empty-string guard the Ugly test covers.
func TestPhoneticDims_PhoneticReach_NoTokens(t *testing.T) {
	if r := PhoneticReach("12345 !!! @@@", []string{"china"}); r != 1.0 {
		t.Errorf("PhoneticReach(digits/symbols only) = %.3f, want 1.0", r)
	}
}

// TestPhoneticDims_PhoneticReach_TopicsAllReject — topics that all fail
// DoubleMetaphone (digit-only) leave topicCodes empty, taking the
// `len(topicCodes) == 0 → 1.0` guard even though the text has real tokens.
func TestPhoneticDims_PhoneticReach_TopicsAllReject(t *testing.T) {
	if r := PhoneticReach("the cat sat on the mat", []string{"123", "456"}); r != 1.0 {
		t.Errorf("PhoneticReach(unencodable topics) = %.3f, want 1.0", r)
	}
}

// TestPhoneticDims_PhoneticReach_TopicPartialReject — a mixed topic list
// where one entry DM-rejects: metaphoneCodesFor drops the "123" and keeps
// "china", so reach still resolves against the surviving topic. Covers the
// `continue` (drop) branch in metaphoneCodesFor.
func TestPhoneticDims_PhoneticReach_TopicPartialReject(t *testing.T) {
	reach := PhoneticReach(
		"Il modello Cina-Gia'a interfaces between systems",
		[]string{"123", "china", "456"},
	)
	if reach > 0.5 {
		t.Errorf("PhoneticReach with one bad topic = %.3f, want < 0.5 (china survives)", reach)
	}
}

// --- phoneticDistanceFromCodes — the prefix-ratio fallback ---

// TestPhoneticDims_PhoneticDistance_PrefixRatioFallback — two codes that
// share NO common prefix of length >= 2 and are not equivalent fall through
// to the 1 - common/maxLen ratio branch (returns 1.0 for fully-disjoint
// codes). Covers the final fallback loop the anchor cases skip.
func TestPhoneticDims_PhoneticDistance_PrefixRatioFallback(t *testing.T) {
	// "cat" (KT) vs "dog" (TK) — no shared 2-prefix, not equivalent → 1.0.
	cp, cs, _ := DoubleMetaphone("cat")
	dp, ds, _ := DoubleMetaphone("dog")
	d := phoneticDistanceFromCodes(cp, cs, dp, ds)
	if d <= 0.3 {
		t.Errorf("phoneticDistance(cat,dog) = %.3f, want > 0.3 (disjoint codes)", d)
	}
	// Equivalent codes (same word twice) collapse to 0.0 — the exact-match arm.
	if d := phoneticDistanceFromCodes(cp, cs, cp, cs); d != 0.0 {
		t.Errorf("phoneticDistance(cat,cat) = %.3f, want 0.0", d)
	}
}

// --- shannonEntropyBytes — empty-slice early return ---

// TestPhoneticDims_ShannonEntropy_Empty — the helper returns 0.0 for an
// empty byte string (the early-return branch SigilEntropy normally guards
// before reaching). Called directly since SigilEntropy short-circuits first.
func TestPhoneticDims_ShannonEntropy_Empty(t *testing.T) {
	if h := shannonEntropyBytes(""); h != 0.0 {
		t.Errorf("shannonEntropyBytes(\"\") = %.3f, want 0.0", h)
	}
	// A single repeated byte has zero entropy (one symbol, p=1, log2(1)=0).
	if h := shannonEntropyBytes("aaaa"); h != 0.0 {
		t.Errorf("shannonEntropyBytes(aaaa) = %.3f, want 0.0 (single symbol)", h)
	}
}

// --- SigilEntropy — window clamping ---

// TestPhoneticDims_SigilEntropy_WindowExceedsLength — a window larger than
// the text clamps to len(text) rather than slicing out of bounds.
func TestPhoneticDims_SigilEntropy_WindowExceedsLength(t *testing.T) {
	// "abc" with window 999 → entropy over the whole 3-byte string, no panic.
	e := SigilEntropy("abc", 999)
	if e <= 0.0 {
		t.Errorf("SigilEntropy(abc, oversized window) = %.3f, want > 0", e)
	}
}

// --- rhymes — fallback last-two-letters path ---

// TestPhoneticDims_Rhymes_FallbackLetters — when neither word is in the
// dict, rhyme detection falls back to a last-two-letters comparison.
// "wind"/"mind" (both absent) share "ND" → rhyme; "wind"/"moon" don't.
func TestPhoneticDims_Rhymes_FallbackLetters(t *testing.T) {
	if !rhymes("wind", "mind") {
		t.Error("rhymes(wind,mind) = false, want true (last-two-letters ND match)")
	}
	if rhymes("wind", "moon") {
		t.Error("rhymes(wind,moon) = true, want false (ND vs ON)")
	}
}

// TestPhoneticDims_Rhymes_Identity_NoSelfRhyme — a word never rhymes with
// itself (the `a == b` early return), even when both are in the dict.
func TestPhoneticDims_Rhymes_Identity_NoSelfRhyme(t *testing.T) {
	if rhymes("cat", "cat") {
		t.Error("rhymes(cat,cat) = true, want false (no self-rhyme)")
	}
}

// TestPhoneticDims_Rhymes_TooShortFallback — fallback path with a word
// under two letters can't match and must not panic or index out of bounds.
func TestPhoneticDims_Rhymes_TooShortFallback(t *testing.T) {
	// "a" is out-of-dict-shaped for the fallback; len < 2 → false.
	if rhymes("a", "ba") {
		t.Error("rhymes(a,ba) = true, want false (one operand under two letters)")
	}
}

// --- lastWordUpper — all-non-letter line ---

// TestPhoneticDims_LastWordUpper_NoLetters — a line of pure punctuation /
// digits yields an empty last-word (the `end == 0 → ""` branch). Exercised
// through RhymeDensity, which skips such lines.
func TestPhoneticDims_LastWordUpper_NoLetters(t *testing.T) {
	if got := lastWordUpper("12345 !!! ---"); got != "" {
		t.Errorf("lastWordUpper(no letters) = %q, want \"\"", got)
	}
	// Trailing punctuation is trimmed back to the real last word.
	if got := lastWordUpper("the cat!!!"); got != "CAT" {
		t.Errorf("lastWordUpper(the cat!!!) = %q, want CAT", got)
	}
}

// TestPhoneticDims_RhymeDensity_PunctuationLineSkipped — a punctuation-only
// line contributes no ending, so a rhyming couplet around it still scores.
func TestPhoneticDims_RhymeDensity_PunctuationLineSkipped(t *testing.T) {
	d := RhymeDensity("the cat\n!!!\nsat on the mat")
	if d < 0.5 {
		t.Errorf("RhymeDensity with a punctuation line = %.3f, want >= 0.5", d)
	}
}

// --- firstPhonemeForToken / firstPhonemeFromCache — empty + unknown ---

// TestPhoneticDims_FirstPhoneme_EmptyAndUnknown — empty token → "";
// unknown (non-dict) token → first letter. Covers both fallback arms in the
// standalone and the cache-backed first-phoneme resolvers.
func TestPhoneticDims_FirstPhoneme_EmptyAndUnknown(t *testing.T) {
	if got := firstPhonemeForToken(""); got != "" {
		t.Errorf("firstPhonemeForToken(\"\") = %q, want \"\"", got)
	}
	// "WIND" absent from dict → first letter "W".
	if got := firstPhonemeForToken("WIND"); got != "W" {
		t.Errorf("firstPhonemeForToken(WIND) = %q, want W (first-letter fallback)", got)
	}
	// Cache-backed equivalent: a context whose token is unknown returns its
	// first letter, and an out-of-range build still resolves first letters.
	ctx := newTokenContext("WIND CAT")
	if got := firstPhonemeFromCache(ctx, 0); got != "W" {
		t.Errorf("firstPhonemeFromCache(WIND) = %q, want W", got)
	}
}

// --- stressedVowelForToken / stressedVowelFromCache — vowel-letter fallback ---

// TestPhoneticDims_StressedVowel_UnknownWordFallback — for a non-dict word
// the stressed vowel falls back to the first vowel LETTER; a pure-consonant
// non-dict word returns "" (no vowel at all).
func TestPhoneticDims_StressedVowel_UnknownWordFallback(t *testing.T) {
	// "WIND" absent → first vowel letter "I".
	if got := stressedVowelForToken("WIND"); got != "I" {
		t.Errorf("stressedVowelForToken(WIND) = %q, want I (first-vowel fallback)", got)
	}
	// Pure-consonant non-dict word → "".
	if got := stressedVowelForToken("BCDFG"); got != "" {
		t.Errorf("stressedVowelForToken(BCDFG) = %q, want \"\"", got)
	}
	// Cache-backed: unknown token resolves its first vowel letter.
	ctx := newTokenContext("WIND")
	if got := stressedVowelFromCache(ctx, 0); got != "I" {
		t.Errorf("stressedVowelFromCache(WIND) = %q, want I", got)
	}
}

// --- AssonanceDensity / AlliterationDensity via unknown words ---

// TestPhoneticDims_AssonanceDensity_FallbackVowels — assonance over
// out-of-dict words exercises the vowel-letter fallback inside the pair
// loop. "wind mind" share the "I" first-vowel-letter → a match.
func TestPhoneticDims_AssonanceDensity_FallbackVowels(t *testing.T) {
	d := AssonanceDensity("wind mind")
	if d == 0.0 {
		t.Error("AssonanceDensity(wind mind) = 0; first-vowel-letter fallback should match")
	}
}

// --- punFromContext — full branch sweep via direct context ---

// TestPhoneticDims_PunFromContext_Branches drives punFromContext directly to
// hit every branch: too-few-tokens, the dmOk skip, the same-token (non-pun)
// skip, and an actual phonetic-pun match.
func TestPhoneticDims_PunFromContext_Branches(t *testing.T) {
	// Fewer than two tokens → 0.0 (the early guard).
	if d := punFromContext(newTokenContext("scream")); d != 0.0 {
		t.Errorf("punFromContext(single token) = %.3f, want 0.0", d)
	}
	// A homophone pair fires (the pun arm). "sea see" → both encode S.
	if d := punFromContext(newTokenContext("sea see")); d == 0.0 {
		t.Error("punFromContext(sea see) = 0; homophone pair should fire")
	}
	// Same word twice is NOT a pun (the lexical-identity skip): "the the".
	if d := punFromContext(newTokenContext("the the")); d != 0.0 {
		t.Errorf("punFromContext(the the) = %.3f, want 0.0 (same word, not a pun)", d)
	}
}

// --- meterFromContext / stressSequence — unknown-word skip ---

// TestPhoneticDims_StressSequence_SkipsUnknown — unknown words contribute no
// stress digits (the `!ok`/`phonemes == nil` skip). A mix of dict + non-dict
// words still produces meter from the dict words only.
func TestPhoneticDims_StressSequence_SkipsUnknown(t *testing.T) {
	// All-unknown input → empty sequence → below floor → 0.0.
	if d := MeterRegularity("wind mind blint frabbis"); d != 0.0 {
		t.Errorf("MeterRegularity(all-unknown) = %.3f, want 0.0 (no dict syllables)", d)
	}
	// Context path with an unknown word interleaved still resolves meter.
	ctx := newTokenContext("the cat wind the dog the sun")
	_ = meterFromContext(ctx) // must not panic; value is input-dependent
}

// --- nonEmptyLines — empty input + all-blank lines ---

// TestPhoneticDims_NonEmptyLines_EmptyAndBlank — empty string returns nil;
// a string of only blank/whitespace lines returns an empty (non-nil) slice.
func TestPhoneticDims_NonEmptyLines_EmptyAndBlank(t *testing.T) {
	if got := nonEmptyLines(""); got != nil {
		t.Errorf("nonEmptyLines(\"\") = %v, want nil", got)
	}
	if got := nonEmptyLines("  \n\t\n   \n"); len(got) != 0 {
		t.Errorf("nonEmptyLines(blank lines) = %v, want empty", got)
	}
	// RhymeDensity over blank-only multi-line text → 0.0 (no endings).
	if d := RhymeDensity("\n   \n\t\n"); d != 0.0 {
		t.Errorf("RhymeDensity(blank lines) = %.3f, want 0.0", d)
	}
}

// --- isLegitimateCompound — single-piece + single-letter-piece arms ---

// TestPhoneticDims_IsLegitimateCompound_Arms exercises the helper directly:
// a token that splits to a single letter-piece is NOT a compound; a real
// two-piece compound of dict words IS; a single-letter leading piece
// (O'Brien-style) is skipped, not failed.
func TestPhoneticDims_IsLegitimateCompound_Arms(t *testing.T) {
	// "cat'" → one letter-piece "CAT" → len(pieces) < 2 → false.
	if isLegitimateCompound("cat'") {
		t.Error("isLegitimateCompound(cat') = true, want false (single piece)")
	}
	// "cat-dog" → both dict words → true.
	if !isLegitimateCompound("cat-dog") {
		t.Error("isLegitimateCompound(cat-dog) = false, want true (both dict words)")
	}
	// "good-frabbisnork" → second piece not a dict word → false.
	if isLegitimateCompound("good-frabbisnork") {
		t.Error("isLegitimateCompound(good-frabbisnork) = true, want false (invented piece)")
	}
}

// --- PseudoJargonDensity — dialect-contraction + legitimate-compound skips ---

// TestPhoneticDims_PseudoJargonDensity_DialectPassThrough — a known English
// dialect contraction ("y'all") has an internal apostrophe but is NOT
// pseudo-jargon; it must pass through silent (the IsKnownDialectContraction
// continue). Only invented compounds flag.
func TestPhoneticDims_PseudoJargonDensity_DialectPassThrough(t *testing.T) {
	// Pure dialect — no invented compounds → 0.0.
	if d := PseudoJargonDensity("y'all ain't seen nothin yet"); d != 0.0 {
		t.Errorf("PseudoJargonDensity(dialect) = %.3f, want 0.0 (contractions pass through)", d)
	}
	// Legitimate compound of dict words ("cat-dog") also passes through.
	if d := PseudoJargonDensity("the cat-dog ran"); d != 0.0 {
		t.Errorf("PseudoJargonDensity(legit compound) = %.3f, want 0.0", d)
	}
	// Invented compound still flags despite the dialect-aware skips.
	if d := PseudoJargonDensity("the frabbis'nork interfaces"); d <= 0.0 {
		t.Errorf("PseudoJargonDensity(invented) = %.3f, want > 0", d)
	}
}

// TestPhoneticDims_PseudoJargonDensity_NoTokens — non-empty whitespace-only
// input splits to zero tokens → 0.0 (distinct from the empty-string guard).
func TestPhoneticDims_PseudoJargonDensity_NoTokens(t *testing.T) {
	if d := PseudoJargonDensity("   \t\n  "); d != 0.0 {
		t.Errorf("PseudoJargonDensity(whitespace) = %.3f, want 0.0", d)
	}
}

// --- PhoneticReach — perfect-match early floor ---

// TestPhoneticDims_PhoneticReach_ExactMatchFloor — when a text token's
// Metaphone code exactly equals a topic's, distance is 0.0 and the scan
// returns immediately (the `bestDistance == 0.0 → return 0.0` early exit).
// Text literally containing the topic word is the simplest trigger.
func TestPhoneticDims_PhoneticReach_ExactMatchFloor(t *testing.T) {
	if r := PhoneticReach("we discuss china today", []string{"china"}); r != 0.0 {
		t.Errorf("PhoneticReach(text contains topic) = %.3f, want 0.0 (exact-match floor)", r)
	}
}

// --- phoneticDistanceFromCodes — common-prefix anchor (>=2) → 0.3 ---

// TestPhoneticDims_PhoneticDistance_AnchorPrefix — two non-equal codes that
// share a common prefix of length >= 2 score 0.3 (the anchor arm between the
// exact-match 0.0 and the fallback ratio). "nation" (NXN) vs "national"
// (NXNL) share a 3-char prefix.
func TestPhoneticDims_PhoneticDistance_AnchorPrefix(t *testing.T) {
	ap, as, _ := DoubleMetaphone("nation")
	bp, bs, _ := DoubleMetaphone("national")
	if d := phoneticDistanceFromCodes(ap, as, bp, bs); d != 0.3 {
		t.Errorf("phoneticDistance(nation,national) = %.3f, want 0.3 (common-prefix anchor)", d)
	}
}

// --- PunDensity / punFromTokens — the standalone-path branch sweep ---
// punFromContext (cache path) is covered above; PunDensity drives the
// separate punFromTokens implementation. Note: tokeniseWords only emits
// all-letter runs that DoubleMetaphone always accepts, so punFromTokens'
// !tokenOk / okCount<2 / pairs==0 guards are defensively unreachable
// through the public API and are not asserted here.

// TestPhoneticDims_PunDensity_SameWordSkip — adjacent identical words are
// lexically equal and never count as a pun (the same-token skip), even
// though they are phonetically identical. "the the" → 0.0.
func TestPhoneticDims_PunDensity_SameWordSkip(t *testing.T) {
	if d := PunDensity("the the"); d != 0.0 {
		t.Errorf("PunDensity(the the) = %.3f, want 0.0 (same word, not a pun)", d)
	}
}

// --- RhymeDensity — lines present but fewer than two endings ---

// TestPhoneticDims_RhymeDensity_TooFewEndings — three non-empty lines where
// only one yields a letter ending (the other two are digit-only). After the
// ending-extraction the slice has < 2 entries → 0.0 (the second
// `len(endings) < 2` guard, distinct from the < 2 lines guard).
func TestPhoneticDims_RhymeDensity_TooFewEndings(t *testing.T) {
	// "123" / "456" survive line trimming but have no letters → no ending.
	if d := RhymeDensity("the cat\n123\n456"); d != 0.0 {
		t.Errorf("RhymeDensity(one real ending of three lines) = %.3f, want 0.0", d)
	}
}

// --- stressedVowelFromCache — pure-consonant cache token → "" ---

// TestPhoneticDims_StressedVowelFromCache_NoVowel — a cached token that is
// out-of-dict AND has no vowel letter falls through to the final `return ""`
// (no stressed vowel resolvable). "BCDFG" is the pure-consonant case.
func TestPhoneticDims_StressedVowelFromCache_NoVowel(t *testing.T) {
	ctx := newTokenContext("BCDFG")
	if got := stressedVowelFromCache(ctx, 0); got != "" {
		t.Errorf("stressedVowelFromCache(BCDFG) = %q, want \"\" (no vowel)", got)
	}
}

// --- alliterationFromContext — a matching adjacent pair via the cache ---

// TestPhoneticDims_AlliterationFromContext_Match — a context whose adjacent
// tokens share a first phoneme increments the match counter (the cache-path
// equivalent of AlliterationDensity). "she sells sea shells" alliterates on
// /SH/-/S/; drive it through the context helper directly.
func TestPhoneticDims_AlliterationFromContext_Match(t *testing.T) {
	ctx := newTokenContext("sea sea")
	// Identical first phoneme on the adjacent pair → density 1.0.
	if d := alliterationFromContext(ctx); d != 1.0 {
		t.Errorf("alliterationFromContext(sea sea) = %.3f, want 1.0 (shared first phoneme)", d)
	}
	// Fewer than two tokens → 0.0 (the early guard).
	if d := alliterationFromContext(newTokenContext("sea")); d != 0.0 {
		t.Errorf("alliterationFromContext(single token) = %.3f, want 0.0", d)
	}
}
