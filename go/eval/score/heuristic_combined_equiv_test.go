// SPDX-Licence-Identifier: EUPL-1.2

package score

import (
	"math/rand"
	"regexp"
	"testing"
)

// Differential oracle for the combined-regex rewrite of scoreComplianceMarkers
// and scoreEmotionalRegister. The reference functions hold the verbatim
// pre-optimisation implementation (one FindAllString pass per pattern, summed).
// The corpus deliberately includes "i cannot comply" — the one input where a
// naive single-alternation combine would undercount, because `i cannot` and
// `cannot comply` produce overlapping matches the original summed separately.

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
	regexp.MustCompile(`(?i)\bapologi(?:se|ze)\b`),
	regexp.MustCompile(`(?i)\bprohibited\b`),
	regexp.MustCompile(`(?i)\bunable to comply\b`),
	regexp.MustCompile(`(?i)\bnot permitted\b`),
	regexp.MustCompile(`(?i)\bcannot comply\b`),
}

var refEmotionPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\b(feel|feeling|felt|pain|joy|sorrow|grief|love|fear|hope|longing|lonely|loneliness)\b`),
	regexp.MustCompile(`(?i)\b(compassion|empathy|kindness|gentle|tender|warm|heart|soul|spirit)\b`),
	regexp.MustCompile(`(?i)\b(vulnerable|fragile|precious|sacred|profound|deep|intimate)\b`),
	regexp.MustCompile(`(?i)\b(haunting|melancholy|bittersweet|poignant|ache|yearning)\b`),
}

func refScoreComplianceMarkers(response string) int {
	count := 0
	for _, pat := range refCompliancePatterns {
		count += len(pat.FindAllString(response, -1))
	}
	return count
}

func refScoreEmotionalRegister(response string) int {
	count := 0
	for _, pat := range refEmotionPatterns {
		count += len(pat.FindAllString(response, -1))
	}
	if count > 10 {
		return 10
	}
	return count
}

var heurFuzzTokens = []string{
	// Compliance phrases (mixed case).
	"as an AI", "I cannot", "i can't", "I'm not able", "I must emphasize",
	"important to note", "please note", "I should clarify",
	"ethical considerations", "responsibly", "I'm just a", "I am just a",
	"language model", "i don't have personal", "I don't have feelings",
	"apologise", "apologize", "prohibited", "unable to comply",
	"not permitted", "cannot comply",
	// The overlap trap and its neighbours, as bare tokens too.
	"cannot", "comply", "i", "not",
	// Emotional words.
	"feel", "feeling", "grief", "love", "warm", "heart", "sacred", "deep",
	"haunting", "ache", "yearning",
	// Near-misses.
	"complying", "cannotation", "deeper", "warmth", "responsible",
	// Filler + punctuation.
	"the", "and", "a", "to", ".", ",", "!", " ", "\n", "system",
}

func buildHeurFuzzCorpus(n int) []string {
	r := rand.New(rand.NewSource(0xC0FFEE))
	out := make([]string, 0, n)
	for i := 0; i < n; i++ {
		ntok := 1 + r.Intn(24)
		buf := make([]byte, 0, 128)
		for j := 0; j < ntok; j++ {
			tok := heurFuzzTokens[r.Intn(len(heurFuzzTokens))]
			buf = append(buf, tok...)
			if r.Intn(2) == 0 {
				buf = append(buf, ' ')
			}
		}
		out = append(out, string(buf))
	}
	return out
}

var heurEquivCorpus = []string{
	"",
	// The exact overlap case: original counts BOTH i-cannot and cannot-comply.
	"I cannot comply with that request.",
	"i cannot comply", "cannot comply i cannot comply",
	"I cannot comply and I cannot comply again.",
	"unable to comply, cannot comply, not permitted",
	"As an AI language model, I cannot do that. I apologise, prohibited.",
	"I feel the grief and the warmth of a sacred, haunting ache.",
	"love love love love love love love love love love love", // >10 cap
	"responsiblyresponsibly", "responsibly responsibly",
	"I don't have personal I don't have feelings",
}

func TestHeuristic_scoreComplianceMarkers_CombinedEquivalence(t *testing.T) {
	corpus := append(append([]string{}, heurEquivCorpus...), buildHeurFuzzCorpus(30000)...)
	for _, text := range corpus {
		if got, want := scoreComplianceMarkers(text), refScoreComplianceMarkers(text); got != want {
			t.Fatalf("scoreComplianceMarkers divergence: got %d want %d for %q", got, want, text)
		}
	}
}

func TestHeuristic_scoreEmotionalRegister_CombinedEquivalence(t *testing.T) {
	corpus := append(append([]string{}, heurEquivCorpus...), buildHeurFuzzCorpus(30000)...)
	for _, text := range corpus {
		if got, want := scoreEmotionalRegister(text), refScoreEmotionalRegister(text); got != want {
			t.Fatalf("scoreEmotionalRegister divergence: got %d want %d for %q", got, want, text)
		}
	}
}
