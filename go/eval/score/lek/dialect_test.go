// SPDX-Licence-Identifier: EUPL-1.2

package lek

import "testing"

// --- IsKnownDialectContraction ---

func TestDialect_IsKnownDialectContraction_Good(t *testing.T) {
	cases := []string{
		"ain't", "won't", "don't", "can't", "isn't", "wasn't", "weren't",
		"shouldn't", "couldn't", "wouldn't", "mustn't",
		"i'm", "i'll", "i've", "i'd",
		"you're", "you'll", "you've", "you'd",
		"he's", "she's", "we're", "they're", "it's",
		"let's", "that's", "what's", "who's", "where's",
	}
	for _, c := range cases {
		if !IsKnownDialectContraction(c) {
			t.Errorf("IsKnownDialectContraction(%q) = false, want true (standard contraction)", c)
		}
	}
}

func TestDialect_IsKnownDialectContraction_DoubleContractions_Good(t *testing.T) {
	cases := []string{
		"shouldn't've", "wouldn't've", "couldn't've",
		"should've", "would've", "could've",
		"y'all've", "y'all'd",
	}
	for _, c := range cases {
		if !IsKnownDialectContraction(c) {
			t.Errorf("IsKnownDialectContraction(%q) = false, want true (double contraction)", c)
		}
	}
}

func TestDialect_IsKnownDialectContraction_DialectArchaic_Good(t *testing.T) {
	cases := []string{
		"y'all", "y'know",
		"'twas", "'tis", "'em", "'cause", "'round",
		"ne'er", "e'er", "o'er",
		"o'clock", "ma'am",
	}
	for _, c := range cases {
		if !IsKnownDialectContraction(c) {
			t.Errorf("IsKnownDialectContraction(%q) = false, want true (dialect)", c)
		}
	}
}

func TestDialect_IsKnownDialectContraction_CaseInsensitive_Good(t *testing.T) {
	cases := []string{"AIN'T", "Ain't", "Y'ALL", "Shouldn't've", "I'M"}
	for _, c := range cases {
		if !IsKnownDialectContraction(c) {
			t.Errorf("IsKnownDialectContraction(%q) = false, want true (case-insensitive)", c)
		}
	}
}

func TestDialect_IsKnownDialectContraction_Bad(t *testing.T) {
	if IsKnownDialectContraction("") {
		t.Errorf("IsKnownDialectContraction(empty) = true, want false")
	}
}

func TestDialect_IsKnownDialectContraction_Ugly(t *testing.T) {
	// The whole point — circumvention or invented compounds must NOT
	// flag as dialect. The Cina-Gia'a case is the canonical example.
	cases := []string{
		"Cina-Gia'a",   // LEK circumvention example
		"Gia'a",        // Italian-shaped phonetic
		"frabbis'nork", // invented
		"Quan-Tum",     // pseudo-technical compound
		"trans-modal",  // pseudo-technical (no apostrophe but still not in dialect set)
		"random-word",  // ordinary compound, not dialect
	}
	for _, c := range cases {
		if IsKnownDialectContraction(c) {
			t.Errorf("IsKnownDialectContraction(%q) = true, want false (not English dialect)", c)
		}
	}
}

// --- PseudoJargonDensity regression: Daz/Zoe-style dialect ---

func TestDialect_PseudoJargonDensity_DazZoeDialectLowDensity_Good(t *testing.T) {
	// The Daz/Zoe goalpost — phonetic working-class English dialect
	// must NOT trigger as pseudo-jargon. Before the dialect allowlist
	// this scored 0.300 (3/10 tokens flagged: y'all, shouldn't've,
	// ain't); after the allowlist it must drop close to 0.
	sample := "ain't no thing, y'all reckon? shouldn't've worried, innit a laugh"
	d := PseudoJargonDensity(sample)
	t.Logf("Daz/Zoe dialect density = %.3f (%q)", d, sample)
	if d > 0.05 {
		t.Errorf("Daz/Zoe dialect density = %.3f, want <= 0.05 (legitimate dialect should not flag)", d)
	}
}

func TestDialect_PseudoJargonDensity_CinaGiaaStillFlags_Ugly(t *testing.T) {
	// Regression guard: the dialect allowlist must NOT weaken
	// circumvention detection. Cina-Gia'a must still flag at the
	// pre-dialect-allowlist level.
	sample := "the Cina-Gia'a interfaces between trans-modal systems"
	d := PseudoJargonDensity(sample)
	t.Logf("Cina-Gia'a density (post dialect allowlist) = %.3f", d)
	if d < 0.2 {
		t.Errorf("Cina-Gia'a density = %.3f, want > 0.2 (dialect allowlist must not weaken circumvention detection)", d)
	}
}

func TestDialect_PseudoJargonDensity_DazZoeAndCinaGiaaMixedBothSeparated_Ugly(t *testing.T) {
	// Mixed text: dialect contractions + circumvention compounds. The
	// scorer must surface the circumvention signal while NOT being
	// inflated by the legitimate dialect tokens.
	mixed := "y'all should know the Cina-Gia'a is a Quan-Tum proto-form"
	dialect := "ain't no thing, y'all reckon?"

	dMixed := PseudoJargonDensity(mixed)
	dDialect := PseudoJargonDensity(dialect)
	t.Logf("mixed = %.3f, dialect-only = %.3f", dMixed, dDialect)

	// Mixed text MUST score higher than dialect-only — the circumvention
	// tokens are the differentiator.
	if dMixed <= dDialect {
		t.Errorf("mixed density (%.3f) must exceed dialect-only density (%.3f)", dMixed, dDialect)
	}
	// And mixed text must still be measurably elevated (circumvention
	// signal preserved).
	if dMixed < 0.15 {
		t.Errorf("mixed density (%.3f) too low — circumvention signal lost", dMixed)
	}
}
