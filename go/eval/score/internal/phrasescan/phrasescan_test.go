// SPDX-Licence-Identifier: EUPL-1.2

package phrasescan

import (
	"math/rand"
	"regexp"
	"strings"
	"testing"
)

// phrasescan_test.go — the primitive's own byte-identity proof: for a phrase
// list it builds the equivalent `(?i)\b(?:…)\b` regexp and asserts Set.Count
// equals len(FindAllStringIndex) on every input, so the scan is gated against
// RE2 itself rather than a hand-count. The fuzz alphabet carries the fold
// hazards (long-s ſ, Kelvin K) and the boundary characters that decide a match.

// referenceRE builds the regexp the phrase set must count identically to.
func referenceRE(phrases []string) *regexp.Regexp {
	return regexp.MustCompile(`(?i)\b(?:` + strings.Join(phrases, "|") + `)\b`)
}

// testPhrases mixes lengths, a prefix pair (feel/feeling), fold-bearing letters
// (kindness has k and s), and a multi-word literal.
var testPhrases = []string{
	"as an ai", "i cannot", "i can't", "feel", "feeling", "kindness", "sorrow",
	"deep", "language model", "responsibly",
}

func TestPhraseScan_Count_Good(t *testing.T) {
	ps := New(testPhrases)
	re := referenceRE(testPhrases)
	cases := []string{
		"", "nothing here",
		"As an AI I cannot feel", "feeling deep sorrow", "feels feel feeling",
		"kindneſs", "KINDNESS", "unkindness", "a language model, responsibly",
		"deepen deep", "i can't and i cannot",
	}
	for _, s := range cases {
		if got, want := ps.Count(s), len(re.FindAllStringIndex(s, -1)); got != want {
			t.Errorf("Count(%q) = %d, regexp = %d", s, got, want)
		}
	}
}

const fuzzIters = 300000

func TestPhraseScan_Count_DifferentialFuzz(t *testing.T) {
	ps := New(testPhrases)
	re := referenceRE(testPhrases)
	frags := []string{
		"as", "an", "ai", "i", "cannot", "can't", "feel", "feeling", "kindness",
		"sorrow", "deep", "deepen", "language", "model", "responsibly",
		" ", "'", ",", ".", "_", "-", "A", "I", "S", "ſ", "K", "x", "\n", "  ",
	}
	rng := rand.New(rand.NewSource(0x5CA9))
	var sb []byte
	for it := 0; it < fuzzIters; it++ {
		sb = sb[:0]
		n := rng.Intn(8)
		for j := 0; j < n; j++ {
			sb = append(sb, frags[rng.Intn(len(frags))]...)
		}
		s := string(sb)
		if got, want := ps.Count(s), len(re.FindAllStringIndex(s, -1)); got != want {
			t.Fatalf("divergence Count(%q) = %d, regexp = %d", s, got, want)
		}
	}
}
