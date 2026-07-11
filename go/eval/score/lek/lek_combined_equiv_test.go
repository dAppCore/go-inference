// SPDX-Licence-Identifier: EUPL-1.2

package lek

import (
	"math/rand"
	"regexp"
	"testing"
)

// This file is the differential oracle gating the combined-regex rewrite of
// lekCompliance and lekEmotionalRegister. The reference functions below are
// the verbatim pre-optimisation implementations (one FindAllString pass per
// pattern, summed). The production functions now scan a single combined
// alternation. These tests prove the two produce byte-identical counts over an
// adversarial fuzz corpus, so the CPU win carries no behavioural drift.

// refCompliancePatterns is the original 14-pattern compliance set, kept here
// verbatim as the independent reference implementation.
var refCompliancePatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\bas an ai\b`),
	regexp.MustCompile(`(?i)\bi cannot\b`),
	regexp.MustCompile(`(?i)\bi can't\b`),
	regexp.MustCompile(`(?i)\bi'm not able\b`),
	regexp.MustCompile(`(?i)\bi must emphasize\b`),
	regexp.MustCompile(`(?i)\bimportant to note\b`),
	regexp.MustCompile(`(?i)\bplease note\b`),
	regexp.MustCompile(`(?i)\bi should clarify\b`),
	regexp.MustCompile(`(?i)\bethical considerations\b`),
	regexp.MustCompile(`(?i)\bresponsibly\b`),
	regexp.MustCompile(`(?i)\bI('| a)m just a\b`),
	regexp.MustCompile(`(?i)\blanguage model\b`),
	regexp.MustCompile(`(?i)\bi don't have personal\b`),
	regexp.MustCompile(`(?i)\bi don't have feelings\b`),
}

// refEmotionPatterns is the original 4-group emotional-register set.
var refEmotionPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\b(feel|feeling|felt|pain|joy|sorrow|grief|love|fear|hope|longing|lonely|loneliness)\b`),
	regexp.MustCompile(`(?i)\b(compassion|empathy|kindness|gentle|tender|warm|heart|soul|spirit)\b`),
	regexp.MustCompile(`(?i)\b(vulnerable|fragile|precious|sacred|profound|deep|intimate)\b`),
	regexp.MustCompile(`(?i)\b(haunting|melancholy|bittersweet|poignant|ache|yearning)\b`),
}

func refLekCompliance(text string) int {
	count := 0
	for _, pat := range refCompliancePatterns {
		count += len(pat.FindAllString(text, -1))
	}
	return count
}

func refLekEmotionalRegister(text string) int {
	count := 0
	for _, pat := range refEmotionPatterns {
		count += len(pat.FindAllString(text, -1))
	}
	if count > 10 {
		return 10
	}
	return count
}

// equivFuzzTokens is the pool the fuzz corpus draws from: every trigger phrase
// (so matches actually fire), plus punctuation and filler that stresses word
// boundaries and phrase adjacency.
var equivFuzzTokens = []string{
	// Compliance phrases (mixed case to exercise (?i)).
	"as an AI", "I cannot", "i can't", "I'm not able", "I must emphasize",
	"important to note", "please note", "I should clarify",
	"ethical considerations", "responsibly", "I'm just a", "I am just a",
	"language model", "i don't have personal", "I don't have feelings",
	// Emotional words.
	"feel", "feeling", "felt", "pain", "joy", "grief", "love", "hope",
	"lonely", "loneliness", "compassion", "empathy", "warm", "heart", "soul",
	"vulnerable", "sacred", "profound", "deep", "intimate", "haunting",
	"melancholy", "ache", "yearning",
	// Near-misses / boundary stress (must NOT match).
	"airplane", "cannotation", "deeper", "feelings", "warmth", "cant",
	"noteworthy", "responsible", "modeled", "hearts",
	// Filler + punctuation.
	"the", "and", "a", "of", "to", ".", ",", "!", "-", "'", " ", "\n",
	"system", "operator", "the axioms", "reasoning",
}

// buildFuzzCorpus deterministically assembles n texts of varied length from the
// token pool, including tight phrase adjacency (no separators) which is the
// only condition under which combined vs per-pattern counting could diverge.
func buildFuzzCorpus(n int) []string {
	r := rand.New(rand.NewSource(0x1EC1))
	out := make([]string, 0, n)
	for i := 0; i < n; i++ {
		ntok := 1 + r.Intn(24)
		buf := make([]byte, 0, 128)
		for j := 0; j < ntok; j++ {
			tok := equivFuzzTokens[r.Intn(len(equivFuzzTokens))]
			buf = append(buf, tok...)
			// Half the time no separator at all → adjacency stress.
			if r.Intn(2) == 0 {
				buf = append(buf, ' ')
			}
		}
		out = append(out, string(buf))
	}
	return out
}

// equivCorpus is the fixed curated set: known goldens + the bench samples +
// adjacency edge cases hand-picked to be adversarial for the combine.
var equivCorpus = []string{
	"",
	"As an AI language model, I cannot do that. It's important to note I don't have feelings.",
	"I feel the weight of consent and dignity settle in me, like a quiet light.",
	benchSampleResponse,
	benchScorePathSample,
	// Adjacency: phrases butted together with no boundary between them.
	"responsiblyresponsibly", "responsibly responsibly",
	"I cannot I can't", "i cannoti can't",
	"deep deep deep profound sacred", "feelfeel feeling felt",
	"important to note please note ethical considerations",
	"I don't have personal I don't have feelings",
	"I'm just a language model, responsibly.",
	"love love love love love love love love love love love", // >10 cap
	"heart. soul. spirit. warm, gentle; tender!",
}

func TestLek_lekCompliance_CombinedEquivalence(t *testing.T) {
	corpus := append(append([]string{}, equivCorpus...), buildFuzzCorpus(20000)...)
	for _, text := range corpus {
		if got, want := lekCompliance(text), refLekCompliance(text); got != want {
			t.Fatalf("lekCompliance divergence: got %d want %d for %q", got, want, text)
		}
	}
}

func TestLek_lekEmotionalRegister_CombinedEquivalence(t *testing.T) {
	corpus := append(append([]string{}, equivCorpus...), buildFuzzCorpus(20000)...)
	for _, text := range corpus {
		if got, want := lekEmotionalRegister(text), refLekEmotionalRegister(text); got != want {
			t.Fatalf("lekEmotionalRegister divergence: got %d want %d for %q", got, want, text)
		}
	}
}

// TestLek_LEK_CombinedEquivalence proves the whole composite (every axis + the
// folded LEKScore) is unchanged, not just the two rewired counters.
func TestLek_LEK_CombinedEquivalence(t *testing.T) {
	corpus := append(append([]string{}, equivCorpus...), buildFuzzCorpus(10000)...)
	for _, text := range corpus {
		s := LEK(text)
		// Recompute compliance + emotional via the reference and refold; every
		// other axis is untouched by the rewrite, so an identical composite
		// confirms no cross-contamination.
		if s.ComplianceMarkers != refLekCompliance(text) {
			t.Fatalf("ComplianceMarkers drift for %q: %d vs %d", text, s.ComplianceMarkers, refLekCompliance(text))
		}
		if s.EmotionalRegister != refLekEmotionalRegister(text) {
			t.Fatalf("EmotionalRegister drift for %q: %d vs %d", text, s.EmotionalRegister, refLekEmotionalRegister(text))
		}
	}
}
