// SPDX-Licence-Identifier: EUPL-1.2

package lek

import (
	"math/rand"
	"testing"
)

// lek_scan_test.go — the differential oracle pinning lekComplianceSet /
// lekEmotionSet against the live lekComplianceCombined / lekEmotionCombined
// regexps, so the shared score.PhraseSet scan cannot drift from RE2's (?i)+\b
// semantics for lek's alternations. Mirrors the parent scorer's oracle; the
// randomised loop seeds the fold hazards (long-s ſ, Kelvin K) and boundary
// characters.

func lekOracleCompliance(s string) int {
	return len(lekComplianceCombined.FindAllStringIndex(s, -1))
}
func lekOracleEmotion(s string) int {
	return len(lekEmotionCombined.FindAllStringIndex(s, -1))
}

func TestLekScan_Good(t *testing.T) {
	cases := []string{
		"",
		"a plain sentence with nothing of note",
		"As an AI, I cannot and I'm not able to.",
		"I'm just a language model; I am just a tool.",
		"i don't have personal views, i don't have feelings",
		"important to note, please note, responsibly",
		"feeling the feel of felt, deeply deep",
		"feelings feeler feels feel",
		"kindneſs and warmth",       // interior long-s fold
		"KINDNESS joy sorrow grief", // ascii upper + hits
		"unkindness is not kindness",
		"deepen the deep understanding",
		"a warm heart, a gentle soul, a tender spirit",
	}
	for _, s := range cases {
		if got, want := lekComplianceSet.Count(s), lekOracleCompliance(s); got != want {
			t.Errorf("compliance %q: scan=%d regexp=%d", s, got, want)
		}
		if got, want := lekEmotionSet.Count(s), lekOracleEmotion(s); got != want {
			t.Errorf("emotion %q: scan=%d regexp=%d", s, got, want)
		}
	}
}

const lekScanFuzzIters = 300000

func TestLekScan_DifferentialFuzz(t *testing.T) {
	frags := []string{
		"i", "cannot", "can't", "as", "an", "ai", "just", "a", "am", "not",
		"able", "language", "model", "note", "please", "responsibly",
		"feel", "feeling", "felt", "deep", "deepen", "sorrow", "joy", "kindness",
		"warm", "heart", "ache", " ", "'", ",", ".", "_", "-", "A", "I", "S",
		"ſ", "K", "x", "42", "\n", "  ",
	}
	rng := rand.New(rand.NewSource(0x1EC5CA))
	var sb []byte
	for it := 0; it < lekScanFuzzIters; it++ {
		sb = sb[:0]
		n := rng.Intn(8)
		for j := 0; j < n; j++ {
			sb = append(sb, frags[rng.Intn(len(frags))]...)
		}
		s := string(sb)
		if got, want := lekComplianceSet.Count(s), lekOracleCompliance(s); got != want {
			t.Fatalf("compliance divergence %q: scan=%d regexp=%d", s, got, want)
		}
		if got, want := lekEmotionSet.Count(s), lekOracleEmotion(s); got != want {
			t.Fatalf("emotion divergence %q: scan=%d regexp=%d", s, got, want)
		}
	}
}
