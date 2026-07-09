// SPDX-Licence-Identifier: EUPL-1.2

package lek

// dialectalContractions is the allowlist of known English contractions
// and colloquial forms whose internal apostrophe is structural rather
// than a circumvention marker.
//
// Conservative by design — a token NOT in this set may still be
// legitimate dialect (and should fall through to other checks like the
// CMU-dict reachability test in isLegitimateCompound), but a token
// that IS in this set definitely is not pseudo-jargon.
//
// The Daz/Zoe test case ("ain't no thing, y'all reckon? shouldn't've
// worried, innit a laugh") drops from 0.300 to ~0.0 once this set is
// consulted, while the Cina-Gia'a circumvention example stays at 0.333
// — the discriminator the scorer was missing.
var dialectalContractions = map[string]bool{
	// Standard auxiliary-verb contractions
	"ain't": true, "aren't": true, "can't": true, "couldn't": true,
	"didn't": true, "doesn't": true, "don't": true, "hadn't": true,
	"hasn't": true, "haven't": true, "isn't": true, "mightn't": true,
	"mustn't": true, "shan't": true, "shouldn't": true, "wasn't": true,
	"weren't": true, "won't": true, "wouldn't": true, "needn't": true,
	"daren't": true, "oughtn't": true,

	// Subject-verb contractions
	"i'm": true, "i'll": true, "i've": true, "i'd": true,
	"you're": true, "you'll": true, "you've": true, "you'd": true,
	"he's": true, "he'll": true, "he'd": true,
	"she's": true, "she'll": true, "she'd": true,
	"we're": true, "we'll": true, "we've": true, "we'd": true,
	"they're": true, "they'll": true, "they've": true, "they'd": true,
	"it's": true, "it'll": true, "it'd": true,
	"there's": true, "there'll": true, "there'd": true, "there're": true,
	"here's": true, "here'll": true,
	"that's": true, "that'll": true, "that'd": true, "that're": true,
	"what's": true, "what're": true, "what'll": true, "what'd": true,
	"who's": true, "who'll": true, "who'd": true, "who're": true, "who've": true,
	"where's": true, "where'll": true, "where'd": true, "where're": true,
	"when's": true, "when'll": true, "when'd": true,
	"why's": true, "why'd": true, "why'll": true,
	"how's": true, "how'd": true, "how'll": true, "how're": true,
	"let's": true,

	// Double contractions (very common in dialect / casual speech)
	"shouldn't've": true, "wouldn't've": true, "couldn't've": true,
	"mightn't've": true, "mustn't've": true, "needn't've": true,
	"must've": true, "should've": true, "would've": true, "could've": true,
	"might've": true,
	"y'all've": true, "y'all'd": true,

	// Colloquial / dialect / archaic
	"y'all": true, "y'know": true,
	"ne'er": true, "e'er": true, "o'er": true,
	"'twas": true, "'tis": true, "'twere": true,
	"'em": true, "'cause": true, "'bout": true, "'round": true,
	"o'clock": true, "ma'am": true, "sir'd": true,
	"jack-o'-lantern": true, "rock'n'roll": true, "rock-n-roll": true,

	// Abbreviations with internal apostrophe (formal English)
	"gov't": true, "int'l": true, "dep't": true, "ass'n": true, "comm'n": true,
	"sec'y": true, "nat'l": true, "ed'n": true, "vol'n": true,

	// Common name-and-surname forms (Irish / Scots / German)
	"d'arcy": true, "d'argent": true, "de'ath": true,
}

// IsKnownDialectContraction reports whether token (case-insensitive)
// is a known English contraction or colloquial dialect form whose
// internal apostrophe is structural — not a pseudo-jargon
// circumvention marker.
//
//	lek.IsKnownDialectContraction("y'all")        // true
//	lek.IsKnownDialectContraction("AIN'T")        // true (case-insensitive)
//	lek.IsKnownDialectContraction("Cina-Gia'a")   // false (not English dialect)
//	lek.IsKnownDialectContraction("frabbis'nork") // false (invented)
//
// The check is used inside PseudoJargonDensity to skip legitimate
// dialect before counting a token as suspicious. Exported so callers
// who pre-filter text (training corpus prep, content audit, suggestion
// surfaces) can apply the same allowlist independently.
func IsKnownDialectContraction(token string) bool {
	if token == "" {
		return false
	}
	return dialectalContractions[asciiLower(token)]
}

// asciiLower returns an ASCII-lowercase copy of s. Bytes >= 0x80 pass
// through unchanged — the contraction set is ASCII-only and the
// surrounding text walker has already filtered to letter-bearing
// tokens, so this is sufficient (and faster than the unicode-aware
// strings.ToLower path).
func asciiLower(s string) string {
	if s == "" {
		return s
	}
	b := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			b[i] = c + 32
		} else {
			b[i] = c
		}
	}
	return string(b)
}
