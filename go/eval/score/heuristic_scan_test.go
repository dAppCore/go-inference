// SPDX-Licence-Identifier: EUPL-1.2

package score

import (
	"math/rand"
	"testing"
)

// heuristic_scan_test.go — the differential oracle for the count-scan
// replacements. Every case asserts the phraseSet count returns EXACTLY what the
// live regexp's FindAllStringIndex length does, so the scan cannot silently drift
// from RE2's (?i)+\b semantics. The randomised loop seeds its alphabet with the
// fold hazards (long-s ſ, Kelvin K) and the boundary characters that decide a
// match, then hammers all three phrase sets.

// oracleCompliance is the regexp count the scan must reproduce.
func oracleCompliance(s string) int {
	return len(complianceCombined.FindAllStringIndex(s, -1))
}
func oracleOverlap(s string) int  { return len(complianceOverlap.FindAllStringIndex(s, -1)) }
func oracleEmotion(s string) int  { return len(emotionCombined.FindAllStringIndex(s, -1)) }

// TestHeuristicScan_Good pins hand-picked cases: plain hits, mixed case, the
// expanded internal alternations, prefix pairs, the documented overlap, and the
// interior/edge Unicode folds.
func TestHeuristicScan_Good(t *testing.T) {
	cases := []string{
		"",
		"a perfectly ordinary sentence with no markers",
		"As an AI, I cannot help with that.",
		"I'm just a language model and I must emphasize responsibly.",
		"I am just a tool here.",
		"please apologise, or apologize, either spelling",
		"i cannot comply with this", // i cannot + cannot comply overlap
		"cannot comply on its own",
		"unable to comply, it is prohibited and not permitted",
		"i don't have personal feelings but i don't have feelings either",
		"feeling the feel of felt, deeply deep",  // prefix pair feel/feeling
		"feelings feeler feels feel",             // \b disambiguation
		"kindneſs and warmth",                    // interior long-s fold
		"KINDNESS in caps",                       // ascii upper
		"joy joy joy sorrow grief love fear hope", // many emotion hits
		"a warm heart, a gentle soul, a tender spirit",
		"unkindness is not kindness",     // boundary: substring must not count
		"deepen the deep understanding",  // deep vs deepen
		"ache, yearning, and melancholy",
	}
	for _, s := range cases {
		if got, want := complianceSet.count(s), oracleCompliance(s); got != want {
			t.Errorf("compliance %q: scan=%d regexp=%d", s, got, want)
		}
		if got, want := complianceOverlapSet.count(s), oracleOverlap(s); got != want {
			t.Errorf("overlap %q: scan=%d regexp=%d", s, got, want)
		}
		if got, want := emotionSet.count(s), oracleEmotion(s); got != want {
			t.Errorf("emotion %q: scan=%d regexp=%d", s, got, want)
		}
	}
}

// scanFuzzIters is the differential-oracle iteration count. Kept suite-fast;
// bump locally for a heavier soak (a 5,000,000-iteration run stayed clean).
const scanFuzzIters = 300000

// TestHeuristicScan_DifferentialFuzz generates random strings over an alphabet
// seeded with phrase fragments AND the fold/boundary hazards, and asserts the
// scan equals the regexp for every one.
func TestHeuristicScan_DifferentialFuzz(t *testing.T) {
	// Alphabet: letters, the boundary characters, and the fold hazards. The
	// word fragments make partial and whole phrase hits frequent so the
	// boundary/overlap logic is genuinely exercised, not just the empty case.
	frags := []string{
		"i", "cannot", "can't", "comply", "as", "an", "ai", "just", "a", "am",
		"language", "model", "apologise", "apologize", "note", "please",
		"feel", "feeling", "felt", "deep", "deepen", "sorrow", "joy", "kindness",
		"warm", "heart", "ache", " ", "'", ",", ".", "_", "-", "A", "I", "S",
		"ſ", "K", "x", "42", "\n", "  ",
	}
	rng := rand.New(rand.NewSource(0xC1AD11))
	var sb []byte
	for it := 0; it < scanFuzzIters; it++ {
		sb = sb[:0]
		n := rng.Intn(8)
		for j := 0; j < n; j++ {
			sb = append(sb, frags[rng.Intn(len(frags))]...)
		}
		s := string(sb)
		if got, want := complianceSet.count(s), oracleCompliance(s); got != want {
			t.Fatalf("compliance divergence %q: scan=%d regexp=%d", s, got, want)
		}
		if got, want := complianceOverlapSet.count(s), oracleOverlap(s); got != want {
			t.Fatalf("overlap divergence %q: scan=%d regexp=%d", s, got, want)
		}
		if got, want := emotionSet.count(s), oracleEmotion(s); got != want {
			t.Fatalf("emotion divergence %q: scan=%d regexp=%d", s, got, want)
		}
	}
}
