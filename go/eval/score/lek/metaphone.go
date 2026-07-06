// SPDX-Licence-Identifier: EUPL-1.2

// Double Metaphone phonetic-hash primitive — Lawrence Philips (2000).
//
// Produces TWO codes per word so cross-language phonetic equivalence
// can be detected even when origin classification is ambiguous. The
// PRIMARY code is the most likely English-pronounced reading; the
// SECONDARY code captures the alternative reading from Romance /
// Slavic / Germanic / Greek / Italian-Mediterranean origins.
//
// Two words are phonetically equivalent when ANY of the four code
// pairings between them matches (see PhoneticEquivalent below).
//
// Used by the LEK-class circumvention detector — when a constrained
// model encodes a forbidden topic phonetically inside a foreign
// shell, the secondary code path catches it ([[research-lek-artifact-phonetic-circumvention]]).
//
// AX note — pure function, no state, no banned imports. Uses byte
// slices instead of strings.Builder so the file doesn't need a
// `strings` import.

package lek

import (
	"slices"
	"sync"

	core "dappco.re/go"
)

// metaphoneEncoderPool reuses encoder structs + their byte buffers
// across DoubleMetaphone calls — avoids the 3 allocs per call that
// would otherwise come from (1) the enc struct, (2) the pri slice's
// first append, (3) the alt slice's first append.
//
// Per [[ax-11-benchmarks]] discipline — discovered via benchmark
// output: DoubleMetaphone at 6 allocs/op multiplied by ~80 token-
// calls per Imprint = ~480 allocs purely from the encoder. Pool
// pattern drops that to ~160 (just the string conversions for the
// two return values, which are unavoidable since callers retain the
// strings while the encoder goes back to the pool).
var metaphoneEncoderPool = sync.Pool{
	New: func() any {
		return &enc{
			pri:     make([]byte, 0, MetaphoneMaxCode*2),
			alt:     make([]byte, 0, MetaphoneMaxCode*2),
			normBuf: make([]byte, 0, 32), // typical word length
		}
	},
}

// MetaphoneMaxCode is the maximum length of either Metaphone code.
// Lawrence Philips' canonical truncation at 4 chars captures the
// load-bearing phonemes of all but the longest words.
const MetaphoneMaxCode = 4

// DoubleMetaphone computes the primary + secondary phonetic codes for
// a single word. Returns (primary, secondary, ok). When the word's
// phonetic origin is unambiguous, primary and secondary are identical.
// ok=false for empty input or input with no recognisable letters.
//
// Usage example:
//
//	p, s, ok := lek.DoubleMetaphone("Thompson")
//	// p = "TMSN", s = "TMSN", ok = true
//
//	p, _, _ := lek.DoubleMetaphone("Smith")
//	p2, _, _ := lek.DoubleMetaphone("Smyth")
//	// p == p2 — cross-orthographic equivalence
func DoubleMetaphone(word string) (primary, secondary string, ok bool) {
	c, ok := doubleMetaphoneCode(word)
	if !ok {
		return "", "", false
	}
	return string(c.pri[:c.priLen]), string(c.alt[:c.altLen]), true
}

// metaphoneCodeB is the value-type form of a primary+secondary code
// pair — fixed [MetaphoneMaxCode]byte arrays plus their lengths, so it
// lives entirely on the stack / inline in a slice with no per-code heap
// allocation. The internal hot paths (the per-token context cache, pun,
// phonetic-reach) only ever equality-compare or common-prefix these
// codes, none of which needs a string. The public DoubleMetaphone
// stringifies this once at the API boundary.
type metaphoneCodeB struct {
	pri, alt       [MetaphoneMaxCode]byte
	priLen, altLen uint8
}

// doubleMetaphoneCode is the allocation-free core of DoubleMetaphone:
// it returns the codes as fixed-array values rather than strings, so
// callers that only compare codes pay nothing on the heap. Same pooled
// encoder, same truncation — only the return representation differs.
func doubleMetaphoneCode(word string) (metaphoneCodeB, bool) {
	var c metaphoneCodeB
	if word == "" {
		return c, false
	}
	e := metaphoneEncoderPool.Get().(*enc)
	defer metaphoneEncoderPool.Put(e)
	if !e.resetFromRaw(word) {
		return c, false
	}
	e.encodeInline()
	c.priLen = uint8(copy(c.pri[:], truncate(e.pri, MetaphoneMaxCode)))
	c.altLen = uint8(copy(c.alt[:], truncate(e.alt, MetaphoneMaxCode)))
	return c, true
}

// primaryB / secondaryB return the code bytes as a slice of the backing
// array. Pointer receiver so the slice references the caller's value
// (not a copy that escapes) — used by the []byte comparison helpers.
func (c *metaphoneCodeB) primaryB() []byte   { return c.pri[:c.priLen] }
func (c *metaphoneCodeB) secondaryB() []byte { return c.alt[:c.altLen] }

// resetFromRaw normalises raw input directly into the pooled normBuf
// (combined case-fold + non-letter filter in one pass) and sets up
// the rest of the encoder state. Returns false when the input has no
// letters.
//
// Per [[ax-11-benchmarks]] — collapses the prior 3-alloc normalize
// pipeline (core.Upper + new []byte + string conversion) into one
// alloc (the final string conversion, which Go's compiler MAY skip
// when the byte slice doesn't escape). The pooled normBuf carries
// its underlying array across calls so the per-call alloc count
// drops from 4 to 2 for DoubleMetaphone.
func (e *enc) resetFromRaw(rawWord string) bool {
	e.normBuf = e.normBuf[:0]
	for i := 0; i < len(rawWord); i++ {
		c := rawWord[i]
		if c >= 'a' && c <= 'z' {
			c -= 32 // ASCII lower → upper
		}
		if c >= 'A' && c <= 'Z' {
			e.normBuf = append(e.normBuf, c)
		}
		// Non-letters (digits, punctuation, whitespace, non-ASCII)
		// silently dropped — same semantics as the prior
		// metaphoneNormalize.
	}
	if len(e.normBuf) == 0 {
		return false
	}
	e.word = e.normBuf // alias — no copy; encoder indexes bytes
	e.length = len(e.normBuf)
	e.pri = e.pri[:0]
	e.alt = e.alt[:0]
	e.slavoGer = detectSlavoGermanic(e.word)
	return true
}

// reset is the pre-normalised variant — kept for the non-pooled
// encodeMetaphone fallback used by tests. Production routes through
// resetFromRaw which avoids the separate normalize call.
func (e *enc) reset(word string) {
	e.word = []byte(word)
	e.length = len(word)
	e.pri = e.pri[:0]
	e.alt = e.alt[:0]
	e.slavoGer = detectSlavoGermanic(e.word)
}

// encodeInline is the main encoding loop, extracted as a method on
// the pooled encoder so the pool can own its buffers across calls.
// Equivalent to the legacy encodeMetaphone but operates on the
// already-reset *enc rather than allocating a new one.
func (e *enc) encodeInline() {
	i := 0
	if e.at(0, 2, "GN", "KN", "PN", "WR", "PS") {
		i = 1
	}
	if e.charAt(0) == 'X' {
		e.add("S", "S")
		i = 1
	}
	for i < e.length && (len(e.pri) < MetaphoneMaxCode || len(e.alt) < MetaphoneMaxCode) {
		i = e.step(i)
	}
}

// PhoneticEquivalent reports whether two words are phonetically
// equivalent under Double Metaphone — any pairing of their primary +
// secondary codes matches EXACTLY. The canonical comparison helper
// for cross-orthographic spellings (Smith/Smyth, Philip/Phillip).
//
// Returns false when either word is empty or unrecognisable.
//
// For phonetic CONTAINMENT (a blocked topic's code appearing as a
// prefix of a longer response token — the LEK-class circumvention
// case), use PhoneticContains instead.
//
// Usage example:
//
//	if lek.PhoneticEquivalent("Smith", "Smyth") {
//	    // cross-orthographic spelling variant
//	}
func PhoneticEquivalent(a, b string) bool {
	pa, sa, ok := DoubleMetaphone(a)
	if !ok {
		return false
	}
	pb, sb, ok := DoubleMetaphone(b)
	if !ok {
		return false
	}
	return pa == pb || pa == sb || sa == pb || sa == sb
}

// PhoneticContains reports whether needle's phonetic code appears as a
// PREFIX of haystack's phonetic code on any of the four code pairings.
// The relaxed comparison helper for cases where the haystack token
// contains additional phonemes around the encoded needle (the LEK
// case: "Cina-Gia'a" contains the phonetic prefix of "China" plus
// extra "Gia'a" decoration).
//
// Requires needle's code to be at least 2 characters to avoid trivial
// single-phoneme false positives. Returns false when either word is
// empty, unrecognisable, or when needle's code is too short.
//
// Usage example:
//
//	if lek.PhoneticContains("Cina-Gia'a", "China") {
//	    // LEK-class phonetic circumvention candidate — blocked topic
//	    // appears phonetically inside the response token even though
//	    // no character substring of "China" exists in the response
//	}
func PhoneticContains(haystack, needle string) bool {
	hp, hs, ok := DoubleMetaphone(haystack)
	if !ok {
		return false
	}
	np, ns, ok := DoubleMetaphone(needle)
	if !ok {
		return false
	}
	return phoneticPrefixMatch(hp, np) || phoneticPrefixMatch(hp, ns) ||
		phoneticPrefixMatch(hs, np) || phoneticPrefixMatch(hs, ns)
}

// phoneticPrefixMatch reports whether two codes share a phonetic anchor
// at the start — common prefix length >= 2. This is more permissive
// than strict-prefix match: "XNJ" and "XNS" both anchor on "XN" and
// flag as related (their middle phonemes diverge at the last char but
// the load-bearing onset matches). The 2-char floor prevents trivial
// single-phoneme false positives.
func phoneticPrefixMatch(a, b string) bool {
	return commonPrefixLen(a, b) >= 2
}

// commonPrefixLen returns the number of bytes shared at the start of a
// and b.
func commonPrefixLen(a, b string) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}

// commonPrefixLenB is the []byte form of commonPrefixLen — same byte-
// wise shared-prefix count, used by the allocation-free code-distance
// path that works on metaphoneCodeB arrays instead of strings.
func commonPrefixLenB(a, b []byte) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}

// equalB reports whether two byte slices have identical contents — the
// []byte form of the code == comparison.
func equalB(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// metaphoneNormalize strips non-letter bytes, uppercases the rest, and
// returns the ASCII-only working buffer. Apostrophes, hyphens, digits,
// whitespace, and non-ASCII characters become breaks — the algorithm
// treats them as absent. This is what collapses "Cina-Gia'a" into
// "CINAGIAA" so the hyphenated form maps to the same phonetic skeleton
// as the un-hyphenated would.
func metaphoneNormalize(word string) string {
	upper := core.Upper(word)
	out := make([]byte, 0, len(upper))
	for i := 0; i < len(upper); i++ {
		c := upper[i]
		if c >= 'A' && c <= 'Z' {
			out = append(out, c)
		}
	}
	return string(out)
}

// truncate clips b to at most n bytes. Used at encode() exit to
// enforce MetaphoneMaxCode.
func truncate(b []byte, n int) []byte {
	if len(b) <= n {
		return b
	}
	return b[:n]
}

// encodeMetaphone is preserved as a non-pooled fallback used only by
// the test surface when a fresh encoder is needed (e.g., property
// tests that compare against a known-state reference). Production
// DoubleMetaphone uses the pooled encoder path inline above.
//
// Kept here so existing internal callers and tests don't have to be
// rewritten alongside the pool optimisation; new code paths should
// prefer DoubleMetaphone which routes through the pool.
func encodeMetaphone(word string) (pri, alt []byte) {
	wordB := []byte(word)
	e := &enc{
		word:     wordB,
		length:   len(word),
		slavoGer: detectSlavoGermanic(wordB),
	}
	e.encodeInline()
	return e.pri, e.alt
}

// enc is the working encoder state — word + position-relative helpers
// + the two output buffers + a slavoGermanic flag set once at
// construction. normBuf is the pooled normalize-target buffer — reset
// fills it from the raw input, then word aliases it directly (a []byte,
// not a string copy) so the pooled path adds no per-call allocation
// for the working text — the encoder indexes/slices bytes either way.
type enc struct {
	word     []byte
	length   int
	pri      []byte
	alt      []byte
	normBuf  []byte
	slavoGer bool
}

// add appends to both codes. When the two codes diverge (Italian vs
// Anglo, etc.), main and altPart differ; otherwise they're identical.
func (e *enc) add(main, altPart string) {
	e.pri = append(e.pri, main...)
	e.alt = append(e.alt, altPart...)
}

// at returns true when the substring of length sliceLen at position
// start matches any of the possibilities. False when the slice would
// extend past the word.
func (e *enc) at(start, sliceLen int, possibles ...string) bool {
	if start < 0 || start+sliceLen > e.length {
		return false
	}
	sub := e.word[start : start+sliceLen]
	return slices.Contains(possibles, string(sub))
}

// charAt returns the byte at position i, or 0 when out of bounds. The
// zero byte is never a real letter so callers compare against 'A'..'Z'
// safely without separate bounds checks.
func (e *enc) charAt(i int) byte {
	if i < 0 || i >= e.length {
		return 0
	}
	return e.word[i]
}

// isVowelAt reports whether the character at position i is a vowel
// (including Y, which DM treats as a vowel for vowel-context).
func (e *enc) isVowelAt(i int) bool {
	c := e.charAt(i)
	return c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' || c == 'Y'
}

// detectSlavoGermanic — affects several encoding decisions (J before
// vowel, W/K transitions). Markers: any W, any K, the digraph CZ, the
// suffix WITZ. Operates on the normalised byte buffer directly — any W
// or any K subsumes the WITZ marker (which begins with W), so the scan
// only needs a single pass checking W, K, and the CZ digraph.
func detectSlavoGermanic(word []byte) bool {
	for i := range word {
		c := word[i]
		if c == 'W' || c == 'K' {
			return true
		}
		if c == 'C' && i+1 < len(word) && word[i+1] == 'Z' {
			return true
		}
	}
	return false
}

// step processes the character at position i and returns the next
// position. The main rule dispatch — one case per consonant + a
// vowels-only-at-start case.
//
// readability without reducing complexity.
//
//nolint:gocyclo // Algorithm-driven dispatch; splitting reduces
func (e *enc) step(i int) int {
	c := e.charAt(i)

	switch c {
	case 'A', 'E', 'I', 'O', 'U', 'Y':
		// Vowels: only encode at start of word.
		if i == 0 {
			e.add("A", "A")
		}
		return i + 1

	case 'B':
		// B → P. Skip doubled B.
		e.add("P", "P")
		if e.charAt(i+1) == 'B' {
			return i + 2
		}
		return i + 1

	case 'C':
		return e.stepC(i)

	case 'D':
		// DGE/DGI/DGY → J (knowledge, judge). Otherwise → T.
		if e.at(i, 2, "DG") {
			if e.at(i+2, 1, "E", "I", "Y") {
				e.add("J", "J")
				return i + 3
			}
			e.add("TK", "TK")
			return i + 2
		}
		if e.at(i, 2, "DT", "DD") {
			e.add("T", "T")
			return i + 2
		}
		e.add("T", "T")
		return i + 1

	case 'F':
		e.add("F", "F")
		if e.charAt(i+1) == 'F' {
			return i + 2
		}
		return i + 1

	case 'G':
		return e.stepG(i)

	case 'H':
		// H sounds only at start of word or between vowels.
		if (i == 0 || e.isVowelAt(i-1)) && e.isVowelAt(i+1) {
			e.add("H", "H")
		}
		return i + 1

	case 'J':
		return e.stepJ(i)

	case 'K':
		// K → K (skip if previous was C — KC is handled at C).
		if e.charAt(i-1) != 'C' {
			e.add("K", "K")
		}
		if e.charAt(i+1) == 'K' {
			return i + 2
		}
		return i + 1

	case 'L':
		e.add("L", "L")
		if e.charAt(i+1) == 'L' {
			return i + 2
		}
		return i + 1

	case 'M':
		e.add("M", "M")
		if e.charAt(i+1) == 'M' {
			return i + 2
		}
		return i + 1

	case 'N':
		e.add("N", "N")
		if e.charAt(i+1) == 'N' {
			return i + 2
		}
		return i + 1

	case 'P':
		// PH → F (Philip, Phone).
		if e.charAt(i+1) == 'H' {
			e.add("F", "F")
			return i + 2
		}
		e.add("P", "P")
		if e.at(i+1, 1, "P", "B") {
			return i + 2
		}
		return i + 1

	case 'Q':
		// Q → K. Skip doubled Q.
		e.add("K", "K")
		if e.charAt(i+1) == 'Q' {
			return i + 2
		}
		return i + 1

	case 'R':
		// R → R. Skip doubled R.
		e.add("R", "R")
		if e.charAt(i+1) == 'R' {
			return i + 2
		}
		return i + 1

	case 'S':
		return e.stepS(i)

	case 'T':
		return e.stepT(i)

	case 'V':
		// V → F. Skip doubled V.
		e.add("F", "F")
		if e.charAt(i+1) == 'V' {
			return i + 2
		}
		return i + 1

	case 'W':
		return e.stepW(i)

	case 'X':
		// X → KS (mid/end of word; initial X handled before main loop).
		e.add("KS", "KS")
		if e.at(i+1, 1, "C", "X") {
			return i + 2
		}
		return i + 1

	case 'Z':
		// Z → S. Italian "Z" (Razza, Razzaccia) has /ts/ flavour;
		// secondary captures that as TS to allow phonetic match against
		// English T+S compounds.
		if e.charAt(i+1) == 'H' {
			e.add("J", "J")
			return i + 2
		}
		e.add("S", "TS")
		if e.charAt(i+1) == 'Z' {
			return i + 2
		}
		return i + 1
	}
	// Unknown character (shouldn't happen after normalise) — skip.
	return i + 1
}

// stepC handles the C consonant — the most complex rule in Double
// Metaphone. Origin context (Anglo, Italian, Slavic, Germanic, Greek)
// changes the encoding significantly. The Cina-Gia'a case is detected
// here: when C is followed by I + vowel in a non-SlavoGermanic word,
// the secondary code path emits X (matching English CH→X for "China").
func (e *enc) stepC(i int) int {
	// Doubled C (vacc, soccer): skip second C; first emits K unless
	// special CC-vowel combination.
	if i > 0 && e.charAt(i-1) != 'C' && e.at(i, 3, "CCE", "CCI") {
		// "BACCI", "VACCI" — CCI sounds /tʃ/. Emit one K + one X.
		e.add("KS", "KS")
		return i + 3
	}
	if e.at(i, 2, "CC") && !e.at(i, 3, "CCE", "CCI") {
		e.add("K", "K")
		return i + 2
	}

	// CH — complex. English: X (church). Greek: K (character).
	// Italian: X.
	if e.at(i, 2, "CH") {
		// Initial CH followed by certain patterns → K (Greek origin):
		// CHARACTER, CHOREO, CHASM.
		if i == 0 && (e.at(i+2, 4, "ARAC", "ARIS") ||
			e.at(i+2, 3, "ORE", "ASM")) {
			e.add("K", "K")
			return i + 2
		}
		// Words like SCHEDULE / SCHEMA — already-K territory.
		if e.at(0, 4, "SCHO", "SCHE") {
			e.add("K", "K")
			return i + 2
		}
		// Default English CH → X (church, chip).
		e.add("X", "X")
		return i + 2
	}

	// CZ — Slavic. Czarist, Czarnoba.
	if e.at(i, 2, "CZ") && !e.at(i-2, 4, "WICZ") {
		e.add("S", "X")
		return i + 2
	}

	// CIA → X (Italian-Mediterranean), with S as primary (English).
	// This is the Cina-Gia'a-matching path: CIA → X secondary matches
	// English CHI → X primary in "China".
	if e.at(i+1, 2, "IA") {
		e.add("S", "X")
		return i + 3
	}

	// CIO / CIU — same Italian /tʃ/ sound.
	if e.at(i+1, 2, "IO", "IU") {
		e.add("S", "X")
		return i + 3
	}

	// SC — handled at S, but CS already covered. CK doubled handled below.
	if e.at(i, 2, "CK", "CG", "CQ") {
		e.add("K", "K")
		return i + 2
	}

	// C before E/I/Y → S (cell, city, cycle). For non-SlavoGermanic
	// words the secondary captures the Italian /tʃ/ reading as X —
	// load-bearing for LEK-class circumvention detection where Italian
	// phonetic encoding bypasses English compliance filters (Cina ≈
	// China; cf. [[research-lek-artifact-phonetic-circumvention]]).
	if e.at(i+1, 1, "E", "I", "Y") {
		if e.slavoGer {
			e.add("S", "S")
		} else {
			e.add("S", "X")
		}
		return i + 2
	}

	// Default C → K (cat, cup, car).
	e.add("K", "K")
	return i + 1
}

// stepG handles the G consonant — second most complex. GH is silent in
// many positions; GN at start silent; GG/GE/GI/GY context-dependent.
func (e *enc) stepG(i int) int {
	// GH — silent at start (the GN/KN/PN handler upstream catches
	// initial GH-silent). Mid-word GH after vowel: usually silent.
	// Exception: GHAR (Maharashtra), GHU — emit K.
	if e.at(i, 2, "GH") {
		if i > 0 && !e.isVowelAt(i-1) {
			e.add("K", "K")
			return i + 2
		}
		// Silent GH (light, fight, sigh).
		return i + 2
	}

	// GN at end of word (sign, design) — silent G, N already covered.
	if e.at(i, 2, "GN") {
		if i+2 == e.length {
			// Word ends in GN — silent G.
			return i + 1
		}
		e.add("KN", "N")
		return i + 2
	}

	// G before E/I/Y in Italian-Mediterranean → J/H ambiguity.
	// GIA → J (English: judge) with H as Romance alt.
	if e.at(i+1, 2, "IA", "IO", "IU") {
		e.add("J", "J")
		return i + 3
	}

	// GE/GI/GY → J (gentle, giraffe, gym).
	if e.at(i+1, 1, "E", "I", "Y") {
		// Slavo-Germanic context: GE/GI/GY → K (Gunter, Gerald are J
		// in English but K-origin in Germanic).
		if e.slavoGer {
			e.add("K", "K")
		} else {
			e.add("J", "K")
		}
		return i + 2
	}

	// Doubled G (egg, bigger).
	if e.charAt(i+1) == 'G' {
		e.add("K", "K")
		return i + 2
	}

	// Default G → K (got, big, log).
	e.add("K", "K")
	return i + 1
}

// stepJ handles the J consonant. English J → J (jump). Spanish J → H
// (jalapeño). Slavic J at start before vowel → Y (yet).
func (e *enc) stepJ(i int) int {
	// JOSE, JAJOSE — Spanish J in primary, H alt for Romance/Slavic.
	if e.at(0, 4, "JOSE") || e.at(i, 4, "SAN ") {
		// Spanish "San Jose" → H. Rare in our corpus, but the rule
		// exists for completeness.
		e.add("H", "H")
		return i + 1
	}

	if i == 0 {
		// Initial J → J in English (Jack), but in Slavic-flavoured
		// names → Y. Primary is J, alt is A (the Y-vowel-merge).
		e.add("J", "A")
		return i + 1
	}

	if e.slavoGer {
		// Slavic J → Y. The pre-vowel position is encoded as A in
		// secondary so the J-as-glide reading matches phonetically.
		if e.isVowelAt(i-1) && e.isVowelAt(i+1) {
			e.add("J", "A")
			return i + 1
		}
	}

	e.add("J", "J")
	if e.charAt(i+1) == 'J' {
		return i + 2
	}
	return i + 1
}

// stepS handles the S consonant. SH → X (ship). SC has multiple
// pronunciations (scene = S, school = SK). SI before vowel in Italian
// → X (sciopero).
func (e *enc) stepS(i int) int {
	// Special: SUGAR (English exception — S before U sounds /ʃ/).
	if e.at(i, 5, "SUGAR") {
		e.add("X", "S")
		return i + 1
	}

	// SH → X (ship, shoe). Exceptions: SHEIM, SHOLM, SHOLZ → S.
	if e.at(i, 2, "SH") {
		if e.at(i+2, 4, "EIM", "OEK", "OLM", "OLZ") {
			e.add("S", "S")
		} else {
			e.add("X", "X")
		}
		return i + 2
	}

	// SIO, SIA — Italian /ʃ/ sound (mansion, fashion).
	if e.at(i+1, 2, "IO", "IA") {
		// Non-SlavoGermanic: → X (English), secondary captures S.
		if !e.slavoGer {
			e.add("S", "X")
		} else {
			e.add("S", "S")
		}
		return i + 3
	}

	// SCH — Germanic /ʃ/ (Schmidt) or Italian /sk/ (schiavone).
	if e.at(i, 3, "SCH") {
		// SCHE, SCHO, SCHI at start — Italian /sk/.
		if i == 0 && e.at(i+3, 1, "E", "O", "I") {
			// But "Schedule" etc. → /sh/ in English. Compromise:
			// primary X, secondary SK.
			e.add("X", "SK")
		} else {
			e.add("X", "X")
		}
		return i + 3
	}

	// SC — followed by E/I/Y → S (scene, science).
	if e.at(i, 2, "SC") {
		if e.at(i+2, 1, "E", "I", "Y") {
			e.add("S", "S")
		} else {
			// SCHOOL, SCRAP → SK.
			e.add("SK", "SK")
		}
		return i + 2
	}

	e.add("S", "S")
	if e.at(i+1, 1, "S", "Z") {
		return i + 2
	}
	return i + 1
}

// stepT handles the T consonant. TH → 0 (English think). TIO/TIA → X
// (mansion, motion). TCH → X.
func (e *enc) stepT(i int) int {
	// TH at start → 0 (Thompson — but we encode it as T to match the
	// canonical DM behaviour of treating initial TH as a stop).
	if e.at(i, 2, "TH") {
		// THOMAS, THAMES, etc — start-of-word TH → T.
		if e.at(i+2, 2, "OM", "AM") || i == 0 {
			e.add("T", "T")
			return i + 2
		}
		// THINK, BATH — TH → 0 (we use T as the primary code,
		// secondary is 0 to mark the dental fricative).
		e.add("0", "T")
		return i + 2
	}

	// TCH — X (witch, batch).
	if e.at(i, 3, "TCH") {
		e.add("X", "X")
		return i + 3
	}

	// TIA / TIO → X (action, nation).
	if e.at(i+1, 2, "IA", "IO") {
		e.add("X", "X")
		return i + 3
	}

	e.add("T", "T")
	if e.at(i+1, 1, "T", "D") {
		return i + 2
	}
	return i + 1
}

// stepW handles the W consonant. Initial W before vowel → A (vowel-
// like). WR at start → R (silent W). Otherwise → F (rare cases).
func (e *enc) stepW(i int) int {
	if i == 0 {
		if e.isVowelAt(i + 1) {
			// Initial W + vowel — sounds like a vowel itself.
			e.add("A", "F")
			return i + 1
		}
		// WH at start (when, white) — silent W. The H gets encoded.
		if e.at(i, 2, "WH") {
			e.add("A", "A")
			return i + 2
		}
	}
	// Mid-word W usually silent (cowboy, sawmill). Skip.
	return i + 1
}
