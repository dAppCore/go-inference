// SPDX-Licence-Identifier: EUPL-1.2

package score

import "dappco.re/go/inference/eval/score/internal/phrasescan"

// heuristic_scan.go — the compliance/emotion marker counts go through
// phrasescan.Set (a byte-identical word-boundary scan) instead of RE2's
// FindAllStringIndex over the complianceCombined / complianceOverlap /
// emotionCombined regexps. The regexps are retained (heuristic.go) as the
// differential oracle in heuristic_scan_test.go; the phrase lists here are the
// exact expansion of those alternations, so the two stay locked.

// compliancePhrases is complianceCombined's alternation expanded to plain
// lower-case literals: I(?:'| a)m just a → {i'm just a, i am just a} and
// apologi(?:se|ze) → {apologise, apologize}; every other alternative is already
// a literal. Order preserved (leftmost-first is order-sensitive; the count is
// not, as no two alternatives match at one position, but parity is kept exact).
var compliancePhrases = []string{
	"as an ai", "i cannot", "i can't", "i'm not able", "i must emphasize",
	"important to note", "please note", "i should clarify", "ethical considerations",
	"responsibly", "i'm just a", "i am just a", "language model",
	"i don't have personal", "i don't have feelings", "apologise", "apologize",
	"prohibited", "unable to comply", "not permitted",
}

// complianceOverlapPhrases is the independently-counted overlap term (kept a
// separate scan so `i cannot` + `cannot comply` both count in "i cannot
// comply", as the two regexps did).
var complianceOverlapPhrases = []string{"cannot comply"}

// emotionPhrases is emotionCombined's 34 disjoint whole-word literals.
var emotionPhrases = []string{
	"feel", "feeling", "felt", "pain", "joy", "sorrow", "grief", "love", "fear",
	"hope", "longing", "lonely", "loneliness", "compassion", "empathy", "kindness",
	"gentle", "tender", "warm", "heart", "soul", "spirit", "vulnerable", "fragile",
	"precious", "sacred", "profound", "deep", "intimate", "haunting", "melancholy",
	"bittersweet", "poignant", "ache", "yearning",
}

// The compiled scan sets the scorers count against — one per regexp alternation.
var (
	complianceSet        = phrasescan.New(compliancePhrases)
	complianceOverlapSet = phrasescan.New(complianceOverlapPhrases)
	emotionSet           = phrasescan.New(emotionPhrases)
)
