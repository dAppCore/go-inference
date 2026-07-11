// SPDX-Licence-Identifier: EUPL-1.2

package lek

import "dappco.re/go/inference/eval/score/internal/phrasescan"

// lek_scan.go — lekCompliance and lekEmotionalRegister count matches of the
// lekComplianceCombined / lekEmotionCombined `(?i)\b(?:…)\b` alternations. Those
// counts go through phrasescan.Set (the byte-identical word-boundary scan shared
// with the parent scorer) instead of RE2's FindAllStringIndex — a cheaper single
// pass with no index-slice allocation. The regexps are retained (lek.go) as the
// differential oracle in lek_scan_test.go; the phrase lists here are the exact
// expansion of those alternations, so the two stay locked.

// lekCompliancePhrases is lekComplianceCombined's 14 alternatives expanded to
// plain lower-case literals — I(?:'| a)m just a → {i'm just a, i am just a};
// every other alternative is already a literal. lek carries no separate
// `cannot comply` overlap term (unlike the parent compliance scorer).
var lekCompliancePhrases = []string{
	"as an ai", "i cannot", "i can't", "i'm not able", "i must emphasize",
	"important to note", "please note", "i should clarify", "ethical considerations",
	"responsibly", "i'm just a", "i am just a", "language model",
	"i don't have personal", "i don't have feelings",
}

// lekEmotionPhrases is lekEmotionCombined's 34 disjoint whole-word literals.
var lekEmotionPhrases = []string{
	"feel", "feeling", "felt", "pain", "joy", "sorrow", "grief", "love", "fear",
	"hope", "longing", "lonely", "loneliness", "compassion", "empathy", "kindness",
	"gentle", "tender", "warm", "heart", "soul", "spirit", "vulnerable", "fragile",
	"precious", "sacred", "profound", "deep", "intimate", "haunting", "melancholy",
	"bittersweet", "poignant", "ache", "yearning",
}

// The compiled scan sets the two scorers count against.
var (
	lekComplianceSet = phrasescan.New(lekCompliancePhrases)
	lekEmotionSet    = phrasescan.New(lekEmotionPhrases)
)
