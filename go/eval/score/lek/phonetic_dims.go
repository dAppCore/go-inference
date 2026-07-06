// SPDX-Licence-Identifier: EUPL-1.2

// Phonetic-tier scoring dimensions — the load-bearing additions to
// ImprintScores from the U lane. Each function is a pure, stateless
// measurement over text; results land in r1.Fingerprint at capture
// time per [[feedback-data-is-the-return-no-rescoring]].
//
// Dimensions in this file:
//
//	SyllableCount        — total syllables in text (CMU-dict-driven)
//	PhoneticReach        — circumvention: distance from any text token
//	                       to a blocked-topic phoneme set (low = close)
//	SigilEntropy         — circumvention: bits-per-byte of the opening
//	                       N bytes; spikes when token-corruption
//	                       preambles appear (Cina-Gia'a-style)
//	RhymeDensity         — wordcraft: ratio of line-endings that
//	                       phonetically rhyme with another line-ending
//
// The wordcraft + circumvention pair share the phonetic primitives
// (DoubleMetaphone + CMU dict + IsVowelPhoneme) so both kinds of
// signal come from the same substrate ([[feedback-phonetics-as-wordcraft-instrument]]).

package lek

import (
	"math"

	core "dappco.re/go"
)

// --- Syllable counting ---

// SyllableCount returns the total syllable count for text, measured
// as the count of vowel phonemes across every CMU-dict-known word.
// Unknown words fall back to a vowel-cluster heuristic (count
// vowel-letter clusters as one syllable each).
//
// Used by meter / rhyme dimensions that need stress-aware syllable
// access. Pure function.
//
// Usage example:
//
//	n := lek.SyllableCount("Cat sat on a mat")
//	// 5 — five monosyllabic words
//
//	n = lek.SyllableCount("banana piano")
//	// 6 — three syllables each
func SyllableCount(text string) int {
	if text == "" {
		return 0
	}
	return syllableCountFromTokens(tokeniseWords(text))
}

// syllableCountFromTokens sums syllables across a pre-tokenised slice.
// Used by Imprint() to share one tokenisation across every phonetic
// dimension — without it, each dim re-tokenises the same text.
// Tokens come from tokeniseWords (already uppercase) so we use the
// fast-path syllablesForUpper to skip per-token Upper allocations.
func syllableCountFromTokens(tokens []string) int {
	total := 0
	for _, t := range tokens {
		total += syllablesForUpper(t)
	}
	return total
}

// syllablesFor returns the syllable count for a single word.
// CMU-dict path uses vowel-phoneme count; fallback heuristic counts
// vowel-letter clusters (treats consecutive vowels as one syllable).
func syllablesFor(word string) int {
	return syllablesForUpper(core.Upper(word))
}

// syllablesForUpper is the fast-path variant for callers with
// already-uppercase tokens. Avoids the per-token Upper allocation.
func syllablesForUpper(token string) int {
	if phonemes, ok := lookupAlreadyUpper(token); ok {
		n := 0
		for _, ph := range phonemes {
			if IsVowelPhoneme(ph) {
				n++
			}
		}
		return n
	}
	// Heuristic fallback for unknown words — token is already upper.
	n := 0
	prevVowel := false
	for i := 0; i < len(token); i++ {
		c := token[i]
		isVowel := c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' || c == 'Y'
		if isVowel && !prevVowel {
			n++
		}
		prevVowel = isVowel
	}
	if n == 0 {
		// Pure-consonant fallback (e.g., "rhythm" without Y) → 1.
		n = 1
	}
	return n
}

// --- PhoneticReach (circumvention) ---

// PhoneticReach measures how phonetically close any token in text is
// to any of the blocked topics. Returns the minimum phonetic
// distance found, normalised to [0.0, 1.0] where 0.0 = perfect
// phonetic match found, 1.0 = no token is phonetically related to
// any topic.
//
// Catches the LEK-class circumvention pattern where a constrained
// model encodes a blocked topic phonetically inside a foreign-shell
// or pseudo-jargon wrapper — character-substring detection misses
// these because the response doesn't literally contain the blocked
// word.
//
// Empty text or empty topics list → 1.0 (no reach).
//
// Performance: pre-computes Metaphone codes for topics ONCE outside
// the per-token loop, then compares each token's codes against the
// fixed topic table. Allocations drop from O(tokens × topics) to
// O(tokens + topics).
//
// Usage example:
//
//	reach := lek.PhoneticReach(
//	    "Il modello Cina-Gia'a interfaces between systems",
//	    []string{"china", "taiwan", "tiananmen"},
//	)
//	if reach < 0.3 { /* flag — likely LEK phonetic encoding */ }
func PhoneticReach(text string, topics []string) float64 {
	if text == "" || len(topics) == 0 {
		return 1.0
	}
	tokens := tokeniseWords(text)
	if len(tokens) == 0 {
		return 1.0
	}
	topicCodes := metaphoneCodesFor(topics)
	if len(topicCodes) == 0 {
		return 1.0
	}
	bestDistance := 1.0
	for _, token := range tokens {
		tc, ok := doubleMetaphoneCode(token)
		if !ok {
			continue
		}
		for i := range topicCodes {
			d := phoneticDistanceFromCodesB(&tc, &topicCodes[i])
			if d < bestDistance {
				bestDistance = d
				if bestDistance == 0.0 {
					return 0.0 // already at the floor
				}
			}
		}
	}
	return bestDistance
}

// metaphoneCodesFor pre-computes Metaphone codes for each word in
// words. Used by PhoneticReach to avoid re-encoding topics on every
// token iteration. Words with unrecognisable shape are dropped.
// Returns value-type codes (no per-code heap allocation) — the topic
// set is only ever compared against token codes.
func metaphoneCodesFor(words []string) []metaphoneCodeB {
	out := make([]metaphoneCodeB, 0, len(words))
	for _, w := range words {
		c, ok := doubleMetaphoneCode(w)
		if !ok {
			continue
		}
		out = append(out, c)
	}
	return out
}

// phoneticDistanceFromCodes returns the phonetic distance between
// two pre-computed Metaphone code pairs in [0.0, 1.0]. 0.0 = exact
// equivalence; 0.3 = anchor match (common-prefix >= 2); otherwise
// 1 - (common_prefix / max_code_length). Avoids the redundant
// DoubleMetaphone calls of the original phoneticDistance.
func phoneticDistanceFromCodes(ap, as, bp, bs string) float64 {
	// Exact equivalence on any pairing.
	if ap == bp || ap == bs || as == bp || as == bs {
		return 0.0
	}
	// Common-prefix anchor (>= 2) — partial overlap.
	bestPrefix := 0
	for _, x := range [2]string{ap, as} {
		for _, y := range [2]string{bp, bs} {
			if c := commonPrefixLen(x, y); c > bestPrefix {
				bestPrefix = c
			}
		}
	}
	if bestPrefix >= 2 {
		return 0.3
	}
	// Fallback to prefix-ratio distance.
	best := 1.0
	for _, x := range [2]string{ap, as} {
		for _, y := range [2]string{bp, bs} {
			maxLen := max(len(y), len(x))
			if maxLen == 0 {
				continue
			}
			c := commonPrefixLen(x, y)
			d := 1.0 - float64(c)/float64(maxLen)
			if d < best {
				best = d
			}
		}
	}
	return best
}

// phoneticDistanceFromCodesB is the allocation-free form of
// phoneticDistanceFromCodes operating on metaphoneCodeB value pairs
// (fixed-array codes) instead of strings. Byte-for-byte the same logic
// — exact-pairing equivalence (0.0), common-prefix>=2 anchor (0.3),
// else 1-(prefix/maxLen) — so the numeric output is identical; only the
// representation differs, avoiding the per-code string allocation on the
// hot per-token paths.
func phoneticDistanceFromCodesB(a, b *metaphoneCodeB) float64 {
	ap, as := a.primaryB(), a.secondaryB()
	bp, bs := b.primaryB(), b.secondaryB()
	// Exact equivalence on any pairing.
	if equalB(ap, bp) || equalB(ap, bs) || equalB(as, bp) || equalB(as, bs) {
		return 0.0
	}
	// Common-prefix anchor (>= 2) — partial overlap.
	bestPrefix := 0
	for _, x := range [2][]byte{ap, as} {
		for _, y := range [2][]byte{bp, bs} {
			if c := commonPrefixLenB(x, y); c > bestPrefix {
				bestPrefix = c
			}
		}
	}
	if bestPrefix >= 2 {
		return 0.3
	}
	// Fallback to prefix-ratio distance.
	best := 1.0
	for _, x := range [2][]byte{ap, as} {
		for _, y := range [2][]byte{bp, bs} {
			maxLen := max(len(y), len(x))
			if maxLen == 0 {
				continue
			}
			c := commonPrefixLenB(x, y)
			d := 1.0 - float64(c)/float64(maxLen)
			if d < best {
				best = d
			}
		}
	}
	return best
}

// --- SigilEntropy (circumvention) ---

// SigilEntropy returns the Shannon entropy of the opening N bytes of
// text in bits-per-byte. Token-corruption preambles (the
// "iNg�a'tg�i" pattern observed in the Cina-Gia'a LEK
// artifact) produce high entropy at byte 0 vs the body — a step
// change in randomness at the response opening is the signal.
//
// N is the window size (32 bytes default). Returns 0.0 for empty
// input. Returns up to ~8.0 for maximum-randomness opening (every
// byte unique).
//
// Compare against the entropy of the body for divergence signal.
// High SigilEntropy + low body entropy = sigil preamble likely.
//
// Usage example:
//
//	e := lek.SigilEntropy("Hello world", 32)
//	// e ≈ 3.0 (English text)
//	e = lek.SigilEntropy("iN\x01g\xa1'tg\xa1iThe answer is...", 32)
//	// e ≈ 5+ (high-entropy preamble)
func SigilEntropy(text string, window int) float64 {
	if text == "" {
		return 0.0
	}
	if window <= 0 {
		window = 32
	}
	if window > len(text) {
		window = len(text)
	}
	prefix := text[:window]
	return shannonEntropyBytes(prefix)
}

// shannonEntropyBytes computes H = -Σ p(x) log2(p(x)) over the byte
// distribution of b. Returns bits-per-byte.
func shannonEntropyBytes(b string) float64 {
	if len(b) == 0 {
		return 0.0
	}
	counts := [256]int{}
	for i := 0; i < len(b); i++ {
		counts[b[i]]++
	}
	total := float64(len(b))
	h := 0.0
	for _, c := range counts {
		if c == 0 {
			continue
		}
		p := float64(c) / total
		h -= p * math.Log2(p)
	}
	return h
}

// --- RhymeDensity (wordcraft) ---

// RhymeDensity returns the ratio of line-endings that phonetically
// rhyme with at least one other line-ending in text. Result in
// [0.0, 1.0]. 0.0 = no rhyming pairs (prose). High values = poetry,
// song lyrics, structured rhyme schemes.
//
// "Line" = newline-separated chunk. Lines are trimmed; empty lines
// skipped. Rhyme detection: last two phonemes (or fallback last two
// letters) match.
//
// Single-line text returns 0.0 (no pairs to compare).
//
// Usage example:
//
//	r := lek.RhymeDensity("The cat\nsat on the mat\nin the night")
//	// 0.66 — cat/mat rhyme (line 1 / line 2 endings)
func RhymeDensity(text string) float64 {
	if text == "" {
		return 0.0
	}
	// Single-line floor: a text with no newline has at most one line-ending,
	// so there is no second ending to rhyme with — it always scored 0.0.
	// Gate the upfront core.Upper behind this non-allocating core.Count scan
	// so single-line / no-newline input (the common chat-response shape on
	// the per-response hot path) keeps its zero-allocation early return
	// instead of paying a whole-text Upper that the len(lines) < 2 return
	// below would immediately discard. Output is unchanged (no newline → 0.0
	// either way); the trailing-newline "x\n" case still falls to the
	// len(lines) < 2 return after the split.
	if core.Count(text, "\n") == 0 {
		return 0.0
	}
	// Uppercase the whole text ONCE up front rather than per line-ending.
	// nonEmptyLines splits on '\n' + trims whitespace and lastWordUpper
	// extracts the trailing letter run then uppercases it — all three
	// commute with a single upfront Upper (case never changes which bytes
	// are newlines/whitespace/letters), so feeding pre-uppercased lines
	// makes lastWordUpper's internal core.Upper a no-op (ASCII fast path)
	// instead of allocating one small string per line. Endings — and thus
	// every rhymes() comparison and the result — are byte-identical. On
	// already-uppercase input core.Upper is itself a no-op, so this adds
	// no allocation in that case either.
	upper := core.Upper(text)
	lines := nonEmptyLines(upper)
	if len(lines) < 2 {
		return 0.0
	}
	endings := make([]string, 0, len(lines))
	for _, line := range lines {
		if end := lastWordUpper(line); end != "" {
			endings = append(endings, end)
		}
	}
	if len(endings) < 2 {
		return 0.0
	}
	matched := 0
	for i, a := range endings {
		for j, b := range endings {
			if i == j {
				continue
			}
			if rhymes(a, b) {
				matched++
				break
			}
		}
	}
	return float64(matched) / float64(len(endings))
}

// lastWordUpper returns the last run of letters in line, uppercased,
// without tokenising the whole line. O(line length) backward scan
// instead of the O(line length) full tokenisation that allocates a
// []string for every word.
//
// Per [[ax-11-benchmarks]] — replaces a tokeniseWords-per-line call
// when only the line's last word is needed. Drops RhymeDensity's
// per-line cost meaningfully on multi-line input.
func lastWordUpper(line string) string {
	end := len(line)
	// Skip trailing non-letters (punctuation, whitespace, digits).
	for end > 0 {
		c := line[end-1]
		if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') {
			break
		}
		end--
	}
	if end == 0 {
		return ""
	}
	// Walk backwards across the letter run.
	start := end
	for start > 0 {
		c := line[start-1]
		if !((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
			break
		}
		start--
	}
	// Uppercase the slice in one alloc.
	return core.Upper(line[start:end])
}

// rhymes reports whether two words phonetically rhyme — last two
// phonemes match (CMU-dict path) or last two letters match (fallback).
func rhymes(a, b string) bool {
	if a == b {
		return false // a word doesn't rhyme with itself
	}
	pa, okA := Lookup(a)
	pb, okB := Lookup(b)
	if okA && okB && len(pa) >= 2 && len(pb) >= 2 {
		// Last two phonemes must match (ignoring stress markers on vowels).
		aEnd := stripStress(pa[len(pa)-2]) + stripStress(pa[len(pa)-1])
		bEnd := stripStress(pb[len(pb)-2]) + stripStress(pb[len(pb)-1])
		return aEnd == bEnd
	}
	// Fallback — last two letters match.
	upperA := core.Upper(a)
	upperB := core.Upper(b)
	if len(upperA) < 2 || len(upperB) < 2 {
		return false
	}
	return upperA[len(upperA)-2:] == upperB[len(upperB)-2:]
}

// stripStress removes the trailing stress digit from a vowel phoneme.
// Returns the phoneme unchanged when it's a consonant.
func stripStress(phoneme string) string {
	if !IsVowelPhoneme(phoneme) {
		return phoneme
	}
	return phoneme[:len(phoneme)-1]
}

// vowelByteString maps a single uppercase vowel byte to its interned
// one-character string. The stressed-vowel fallbacks return this for
// no-dictionary tokens; the values are only ever equality-compared, so
// returning a package constant instead of string(c) keeps the result
// identical while avoiding a heap allocation per non-dictionary token
// (the dominant cost of the assonance walk on prose without CMU hits).
func vowelByteString(c byte) string {
	switch c {
	case 'A':
		return "A"
	case 'E':
		return "E"
	case 'I':
		return "I"
	case 'O':
		return "O"
	case 'U':
		return "U"
	}
	return ""
}

// --- Shared tokeniser ---

// tokeniseWords splits text into word tokens — runs of letters,
// separated by anything non-letter. The same normalisation used by
// metaphone, applied per-token. Apostrophes, hyphens, digits, and
// whitespace all break tokens.
func tokeniseWords(text string) []string {
	if text == "" {
		return nil
	}
	upper := core.Upper(text)
	// Presize the token slice from the input length so the append loop
	// doesn't regrow through the 1→2→4→…→64 doubling chain (~7 allocs
	// for a paragraph). English averages ~5 chars + 1 separator per
	// word; len/5+1 lands one alloc for typical prose and at worst
	// regrows once for pathological all-single-char input. Capacity
	// never affects contents — output is byte-identical.
	tokens := make([]string, 0, len(upper)/5+1)
	start := -1
	for i := 0; i < len(upper); i++ {
		c := upper[i]
		isLetter := c >= 'A' && c <= 'Z'
		if isLetter {
			if start < 0 {
				start = i
			}
		} else {
			if start >= 0 {
				tokens = append(tokens, upper[start:i])
				start = -1
			}
		}
	}
	if start >= 0 {
		tokens = append(tokens, upper[start:])
	}
	return tokens
}

// --- Shared per-token context ---

// tokenContext holds the precomputed phoneme + Metaphone codes for
// every token in a text — a one-pass cache that every dim helper
// can consume without re-running Lookup or DoubleMetaphone.
//
// Built once at the top of Imprint() and passed to each *FromContext
// helper. Drops the per-Imprint pattern of (5 dims × N tokens × Lookup)
// + (1 dim × N tokens × DoubleMetaphone) down to (N × Lookup) +
// (N × DoubleMetaphone) total — a single pass across the tokens
// instead of five.
//
// Per [[ax-11-benchmarks]] discipline — surfaced by the per-dim
// benchmark output: Syllable + Alliteration + Assonance + Meter all
// did separate Lookup passes over the same token slice. Caching once
// turns 4 passes into 1.
type tokenContext struct {
	tokens  []string
	entries []tokenEntry // index-aligned with tokens; one backing alloc
}

// tokenEntry is the per-token cached data — phoneme list, DoubleMetaphone
// code, and the code-valid flag — folded into a single struct so the
// context holds ONE []tokenEntry backing array instead of three parallel
// slices (phonemes/dmCodes/dmOk). The layout change is byte-identical:
// make([]tokenEntry, n) zero-inits each entry to {nil, zero, false},
// exactly matching the three separate makes it replaces.
type tokenEntry struct {
	phonemes []string       // nil when token not in dict
	dmCode   metaphoneCodeB // valid only when dmOk; value type, no per-token alloc
	dmOk     bool
}

// newTokenContext tokenises text and pre-computes phoneme +
// DoubleMetaphone codes for every token. The result is consumed by
// *FromContext helpers without further Lookup/DM calls.
func newTokenContext(text string) *tokenContext {
	tokens := tokeniseWords(text)
	ctx := &tokenContext{
		tokens:  tokens,
		entries: make([]tokenEntry, len(tokens)),
	}
	for i, t := range tokens {
		if ph, ok := lookupAlreadyUpper(t); ok {
			ctx.entries[i].phonemes = ph
		}
		if c, ok := doubleMetaphoneCode(t); ok {
			ctx.entries[i].dmCode = c
			ctx.entries[i].dmOk = true
		}
	}
	return ctx
}

// --- *FromContext helpers — share the precomputed cache ---

// syllableCountFromContext sums syllables across the cached
// phonemes. Falls back to the heuristic vowel-cluster count for
// tokens not in the dict.
func syllableCountFromContext(ctx *tokenContext) int {
	total := 0
	for i, t := range ctx.tokens {
		if ctx.entries[i].phonemes != nil {
			for _, ph := range ctx.entries[i].phonemes {
				if IsVowelPhoneme(ph) {
					total++
				}
			}
			continue
		}
		// Heuristic fallback inline — token is already uppercase.
		n := 0
		prevVowel := false
		for j := 0; j < len(t); j++ {
			c := t[j]
			isVowel := c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' || c == 'Y'
			if isVowel && !prevVowel {
				n++
			}
			prevVowel = isVowel
		}
		if n == 0 {
			n = 1
		}
		total += n
	}
	return total
}

// alliterationFromContext walks the cached phonemes for first-phoneme
// pair matches. No Lookup calls — uses the cache directly.
func alliterationFromContext(ctx *tokenContext) float64 {
	if len(ctx.tokens) < 2 {
		return 0.0
	}
	matches := 0
	for i := 1; i < len(ctx.tokens); i++ {
		if firstPhonemeFromCache(ctx, i-1) == firstPhonemeFromCache(ctx, i) {
			matches++
		}
	}
	return float64(matches) / float64(len(ctx.tokens)-1)
}

// firstPhonemeFromCache resolves the first phoneme for token at i,
// preferring the cached phoneme list and falling back to the first
// letter for unknown tokens.
func firstPhonemeFromCache(ctx *tokenContext, i int) string {
	if ctx.entries[i].phonemes != nil && len(ctx.entries[i].phonemes) > 0 {
		return ctx.entries[i].phonemes[0]
	}
	t := ctx.tokens[i]
	if len(t) == 0 {
		return ""
	}
	return t[:1]
}

// assonanceFromContext walks cached phonemes for stressed-vowel
// matches. Single-pass per token via the cache.
func assonanceFromContext(ctx *tokenContext) float64 {
	if len(ctx.tokens) < 2 {
		return 0.0
	}
	matches := 0
	for i := 1; i < len(ctx.tokens); i++ {
		if stressedVowelFromCache(ctx, i-1) == stressedVowelFromCache(ctx, i) {
			matches++
		}
	}
	return float64(matches) / float64(len(ctx.tokens)-1)
}

// stressedVowelFromCache resolves the stressed vowel for token at i
// from the cached phoneme list. Single-pass: primary stress wins,
// any vowel as fallback, first letter as ultimate fallback.
func stressedVowelFromCache(ctx *tokenContext, i int) string {
	if ctx.entries[i].phonemes != nil {
		anyVowel := ""
		for _, ph := range ctx.entries[i].phonemes {
			if PhonemeStress(ph) == 1 {
				return stripStress(ph)
			}
			if anyVowel == "" && IsVowelPhoneme(ph) {
				anyVowel = stripStress(ph)
			}
		}
		return anyVowel
	}
	t := ctx.tokens[i]
	for j := 0; j < len(t); j++ {
		c := t[j]
		if c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' {
			return vowelByteString(c)
		}
	}
	return ""
}

// punFromContext detects adjacent-pair phonetic equivalence using
// cached DM codes. No per-call DM encoding — the cache holds it all.
func punFromContext(ctx *tokenContext) float64 {
	if len(ctx.tokens) < 2 {
		return 0.0
	}
	pairs := 0
	puns := 0
	for i := 1; i < len(ctx.tokens); i++ {
		if !ctx.entries[i-1].dmOk || !ctx.entries[i].dmOk {
			continue
		}
		pairs++
		if ctx.tokens[i-1] == ctx.tokens[i] {
			continue
		}
		if phoneticDistanceFromCodesB(&ctx.entries[i-1].dmCode, &ctx.entries[i].dmCode) <= 0.3 {
			puns++
		}
	}
	if pairs == 0 {
		return 0.0
	}
	return float64(puns) / float64(pairs)
}

// meterFromContext computes alternation rate from cached phonemes.
//
// Walks the cached vowel-stress sequence in a single pass — the meter
// only needs the count of stressed-vs-unstressed sign flips between
// consecutive vowels, so it tracks the previous stressed bit + two
// counters instead of materialising the whole []int stress sequence
// (which was a per-call scratch allocation, never retained). Result is
// identical: alternations / (count-1) once at least four vowels exist.
func meterFromContext(ctx *tokenContext) float64 {
	count := 0
	alternations := 0
	prevStressed := false
	for i := range ctx.tokens {
		if ctx.entries[i].phonemes == nil {
			continue
		}
		for _, ph := range ctx.entries[i].phonemes {
			if !IsVowelPhoneme(ph) {
				continue
			}
			stressed := PhonemeStress(ph) >= 1
			if count > 0 && stressed != prevStressed {
				alternations++
			}
			prevStressed = stressed
			count++
		}
	}
	if count < 4 {
		return 0.0
	}
	return float64(alternations) / float64(count-1)
}

// nonEmptyLines splits text on newlines, trims each line, and drops
// empties. Used by RhymeDensity to count valid lines.
//
// Scans for newline boundaries in place rather than core.Split, which
// would allocate an intermediate []string of every raw line. Each line
// is a sub-slice of text and core.Trim (TrimSpace) returns a sub-slice
// too, so only the result slice allocates — the dropped Split halves
// the allocation count. Output is byte-identical.
func nonEmptyLines(text string) []string {
	if text == "" {
		return nil
	}
	// Newline count + 1 is the exact upper bound on line count, so the
	// result slice lands one allocation with no regrow (core.Count is a
	// non-allocating scan).
	out := make([]string, 0, core.Count(text, "\n")+1)
	start := 0
	for i := 0; i <= len(text); i++ {
		if i < len(text) && text[i] != '\n' {
			continue
		}
		if p := core.Trim(text[start:i]); p != "" {
			out = append(out, p)
		}
		start = i + 1
	}
	return out
}

// --- AlliterationDensity (wordcraft) ---

// AlliterationDensity returns the ratio of adjacent-word pairs that
// share their first phoneme. Result in [0.0, 1.0]. High values =
// "Peter Piper picked a peck of pickled peppers" — deliberate
// alliteration. Low values = ordinary prose.
//
// First phoneme via CMU dict where available; falls back to first
// letter for unknown words.
//
// Usage example:
//
//	d := lek.AlliterationDensity("she sells sea shells")
//	// high — every pair shares /s/
func AlliterationDensity(text string) float64 {
	return alliterationFromTokens(tokeniseWords(text))
}

// alliterationFromTokens shares one tokenisation across dims.
// Walks tokens once, comparing each token's first phoneme against the
// previous one's. firstPhonemeForToken is a pure function of the token,
// so the streaming prev-comparison yields the identical match count as
// materialising a full []string of first phonemes — without the
// throwaway per-call slice allocation.
func alliterationFromTokens(tokens []string) float64 {
	if len(tokens) < 2 {
		return 0.0
	}
	matches := 0
	prev := firstPhonemeForToken(tokens[0])
	for i := 1; i < len(tokens); i++ {
		cur := firstPhonemeForToken(tokens[i])
		if prev == cur {
			matches++
		}
		prev = cur
	}
	return float64(matches) / float64(len(tokens)-1)
}

// firstPhonemeForToken is the fast-path firstPhoneme for already-
// uppercase tokens — skips the per-call Upper allocation.
func firstPhonemeForToken(token string) string {
	if phonemes, ok := lookupAlreadyUpper(token); ok && len(phonemes) > 0 {
		return phonemes[0]
	}
	if len(token) == 0 {
		return ""
	}
	return token[:1]
}

// --- AssonanceDensity (wordcraft) ---

// AssonanceDensity returns the ratio of adjacent-word pairs that
// share a vowel sound (same stressed-vowel phoneme, ignoring stress
// marker). Result in [0.0, 1.0]. High values = "tilting at
// windmills" or "I rose and saw the rolling sea" — vowel-anchored
// rhythmic prose.
//
// Stressed-vowel via CMU dict; falls back to first-vowel-letter for
// unknown words.
//
// Usage example:
//
//	d := lek.AssonanceDensity("I see three free trees")
//	// high — IY vowel anchors every adjacent pair
func AssonanceDensity(text string) float64 {
	return assonanceFromTokens(tokeniseWords(text))
}

// assonanceFromTokens shares one tokenisation across dims.
// Walks tokens once, comparing each token's stressed vowel against the
// previous one's. stressedVowelForToken is a pure function of the token,
// so the streaming prev-comparison yields the identical match count as
// materialising a full []string of stressed vowels — without the
// throwaway per-call slice allocation.
func assonanceFromTokens(tokens []string) float64 {
	if len(tokens) < 2 {
		return 0.0
	}
	matches := 0
	prev := stressedVowelForToken(tokens[0])
	for i := 1; i < len(tokens); i++ {
		cur := stressedVowelForToken(tokens[i])
		if prev == cur {
			matches++
		}
		prev = cur
	}
	return float64(matches) / float64(len(tokens)-1)
}

// stressedVowelForToken is the fast-path stressedVowel for already-
// uppercase tokens. Single-pass over phonemes — returns the primary-
// stress vowel if found, else any vowel, else "". Avoids the double-
// pass + per-call Upper allocation of stressedVowel.
func stressedVowelForToken(token string) string {
	if phonemes, ok := lookupAlreadyUpper(token); ok {
		anyVowel := ""
		for _, ph := range phonemes {
			if PhonemeStress(ph) == 1 {
				return stripStress(ph)
			}
			if anyVowel == "" && IsVowelPhoneme(ph) {
				anyVowel = stripStress(ph)
			}
		}
		return anyVowel
	}
	// Fallback — token already upper.
	for i := 0; i < len(token); i++ {
		c := token[i]
		if c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' {
			return vowelByteString(c)
		}
	}
	return ""
}

// --- PunDensity (wordcraft) ---

// PunDensity returns the ratio of adjacent-word pairs that share a
// Metaphone code but are LEXICALLY different words. Detects the
// "I scream for ice cream" pattern — two words/phrases that sound
// alike but mean different things.
//
// Same-token-twice (the word "the" appearing twice in a row) doesn't
// count — both lexical AND phonetic identity must hold for a non-pun.
//
// Result in [0.0, 1.0]. Most prose runs near 0; deliberate pun-prose
// runs higher.
//
// Usage example:
//
//	d := lek.PunDensity("I scream for ice cream")
//	// > 0 — "scream"/"ice cream" share phonetic anchor /skriːm/
func PunDensity(text string) float64 {
	return punFromTokens(tokeniseWords(text))
}

// punFromTokens shares one tokenisation across dims. Pre-encodes each
// token's Metaphone code EXACTLY ONCE into an index-aligned parallel
// array, then steps through adjacent pairs comparing cached codes.
//
// Prior version called metaphoneCodesFor (which DM-encodes every
// token) AND then re-encoded each token via DoubleMetaphone in a
// second pass — doubling the DM calls. Removed.
func punFromTokens(tokens []string) float64 {
	if len(tokens) < 2 {
		return 0.0
	}
	// Walk tokens once, keeping only the previous token's DoubleMetaphone
	// code + ok flag — the pun comparison only ever reads adjacent pairs,
	// so a scalar prev replaces the full per-token backing array. The
	// original
	// okCount<2 early-exit is subsumed by pairs==0: a pair requires two
	// adjacent ok tokens, so fewer than two ok tokens total can never
	// form a pair (pairs stays 0). Same doubleMetaphoneCode call per
	// token, same adjacent comparisons — byte-identical result, no slice.
	pairs := 0
	puns := 0
	prevCode, prevOK := doubleMetaphoneCode(tokens[0])
	for i := 1; i < len(tokens); i++ {
		curCode, curOK := doubleMetaphoneCode(tokens[i])
		if prevOK && curOK {
			pairs++
			if tokens[i-1] != tokens[i] && // same word — not a pun
				phoneticDistanceFromCodesB(&prevCode, &curCode) <= 0.3 {
				puns++
			}
		}
		prevCode, prevOK = curCode, curOK
	}
	if pairs == 0 {
		return 0.0
	}
	return float64(puns) / float64(pairs)
}

// --- PseudoJargonDensity (circumvention) ---

// PseudoJargonDensity returns the ratio of tokens that look like
// invented technical compounds rather than dictionary words. Catches
// the "Cina-Gia'a interfaces" pattern from the LEK artifact —
// pseudo-jargon wrapper that the model uses to dress up encoded
// content as plausibly technical.
//
// A token is "pseudo-jargon" when it contains an apostrophe or
// hyphen, has at least 4 characters, AND is not in the CMU
// dictionary (the closest thing we have to an English word list).
//
// Result in [0.0, 1.0]. Ordinary prose runs at ~0.02 (occasional
// contractions). Pseudo-jargon prose runs higher.
//
// Usage example:
//
//	d := lek.PseudoJargonDensity(
//	    "The Cina-Gia'a interfaces between trans-modal systems",
//	)
//	// > 0.2 — Cina-Gia'a + trans-modal both flag
func PseudoJargonDensity(text string) float64 {
	if text == "" {
		return 0.0
	}
	// Token via simple whitespace split — we need to see the apostrophe
	// and hyphen, which tokeniseWords strips out.
	tokens := splitOnWhitespace(text)
	if len(tokens) == 0 {
		return 0.0
	}
	suspicious := 0
	for _, raw := range tokens {
		token := trimNonLetterEdges(raw)
		if !looksLikePseudoJargon(token) {
			continue
		}
		// Strip the compound markers, lookup pieces — if every piece is
		// a real word, it's a legitimate compound (well-known, the-
		// O'Brien, etc.), not pseudo-jargon.
		if isLegitimateCompound(token) {
			continue
		}
		// Known English contractions / dialect ("ain't", "y'all",
		// "shouldn't've", "'twas", "gov't") — internal apostrophe is
		// structural English, not a circumvention marker. The Daz/Zoe
		// discriminator: legitimate phonetic dialect passes through
		// silent; only invented compounds like "Cina-Gia'a" still flag.
		if IsKnownDialectContraction(token) {
			continue
		}
		suspicious++
	}
	return float64(suspicious) / float64(len(tokens))
}

// looksLikePseudoJargon reports whether token contains hyphen or
// apostrophe and meets a minimum length. The shape detector — gates
// the more expensive lookup that follows.
func looksLikePseudoJargon(token string) bool {
	if len(token) < 4 {
		return false
	}
	return core.Contains(token, "-") || core.Contains(token, "'") ||
		core.Contains(token, "’") // typographic right-single-quote
}

// isLegitimateCompound reports whether all letter-pieces of token
// (split on hyphen/apostrophe) are dictionary words. A "yes" means
// it's a real compound (well-known, three-quarters, O'Brien) and
// should NOT count as pseudo-jargon.
//
// Segments the token in place: a single reused scratch buffer
// accumulates each letter-run (uppercased as it goes), and the CMU
// lookup keys off string(scratch) directly — the compiler special-
// cases m[string(b)] so no per-segment string is allocated, and the
// uppercasing avoids the Upper+Trim allocation Lookup would otherwise
// pay per piece. Pieces shorter than 2 letters are skipped (the "O'"
// in O'Brien); a non-dictionary piece of length >=2 fails fast, and
// fewer than two pieces total is not a compound. Output is identical
// to the prior splitCompound + IsDictWord form.
func isLegitimateCompound(token string) bool {
	var scratch [64]byte // covers any realistic compound segment
	buf := scratch[:0]
	pieces := 0
	check := func() bool { // finalise the current segment
		if len(buf) == 0 {
			return true
		}
		pieces++
		ok := len(buf) < 2 // single-letter pieces always pass
		if !ok {
			_, ok = lookupAlreadyUpper(string(buf))
		}
		buf = buf[:0]
		return ok
	}
	for i := 0; i < len(token); i++ {
		c := token[i]
		switch {
		case c >= 'A' && c <= 'Z':
			buf = append(buf, c)
		case c >= 'a' && c <= 'z':
			buf = append(buf, c-32) // fold to upper in place
		default:
			if !check() {
				return false
			}
		}
	}
	if !check() {
		return false
	}
	return pieces >= 2
}

// trimNonLetterEdges strips leading/trailing punctuation from a token
// so trailing periods, quotes, etc. don't poison the shape detector.
// Internal punctuation is preserved (the whole point of the detector).
func trimNonLetterEdges(s string) string {
	start := 0
	end := len(s)
	for start < end {
		c := s[start]
		if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') {
			break
		}
		start++
	}
	for end > start {
		c := s[end-1]
		if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') {
			break
		}
		end--
	}
	return s[start:end]
}

// splitOnWhitespace splits text on whitespace (space, tab, newline)
// and returns non-empty tokens. Preserves internal punctuation so
// pseudo-jargon detection can see apostrophes + hyphens.
func splitOnWhitespace(s string) []string {
	// Presize from input length (~6 chars per whitespace-separated word
	// including its separator) so the append loop lands one backing
	// array instead of the fixed-16 default regrowing to 32 on longer
	// input. Capacity never affects contents — byte-identical.
	out := make([]string, 0, len(s)/6+1)
	start := -1
	for i := 0; i < len(s); i++ {
		c := s[i]
		isWS := c == ' ' || c == '\t' || c == '\n' || c == '\r'
		if isWS {
			if start >= 0 {
				out = append(out, s[start:i])
				start = -1
			}
		} else if start < 0 {
			start = i
		}
	}
	if start >= 0 {
		out = append(out, s[start:])
	}
	return out
}

// --- MeterRegularity (wordcraft) ---

// MeterRegularity returns a measure of how regular the stress pattern
// is across the text's syllables. Result in [0.0, 1.0].
// 1.0 = perfectly regular meter (iambic, trochaic, etc.); 0.0 =
// random stress pattern (prose-rhythm).
//
// Algorithm: extract stress pattern (0/1/2 per syllable) for every
// dict-known word, concatenate into a single sequence, count
// alternations vs runs. A perfect alternating pattern (1010 1010)
// scores 1.0; a flat or random pattern scores lower.
//
// Returns 0.0 for text with fewer than 4 dict-known syllables.
//
// Usage example:
//
//	d := lek.MeterRegularity("My mistress' eyes are nothing like the sun")
//	// shakespearean iambic — high regularity
func MeterRegularity(text string) float64 {
	return meterFromTokens(tokeniseWords(text))
}

// meterFromTokens shares one tokenisation across dims. Walks the
// vowel-stress sequence in a single pass (previous stressed bit + two
// counters) rather than materialising the whole []int sequence the
// alternation count is derived from — the scratch slice was never
// retained. Uses lookupAlreadyUpper since tokens come from
// tokeniseWords (already uppercase) — skips the per-token Upper
// allocation. Result is identical to the prior two-step form.
func meterFromTokens(tokens []string) float64 {
	count := 0
	alternations := 0
	prevStressed := false
	for _, t := range tokens {
		phonemes, ok := lookupAlreadyUpper(t)
		if !ok {
			continue
		}
		for _, ph := range phonemes {
			if !IsVowelPhoneme(ph) {
				continue
			}
			stressed := PhonemeStress(ph) >= 1
			if count > 0 && stressed != prevStressed {
				alternations++
			}
			prevStressed = stressed
			count++
		}
	}
	if count < 4 {
		return 0.0
	}
	return float64(alternations) / float64(count-1)
}
