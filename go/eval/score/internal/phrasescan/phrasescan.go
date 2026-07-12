// SPDX-Licence-Identifier: EUPL-1.2

// Package phrasescan counts leftmost, non-overlapping matches of a
// `(?i)\b(?:literal|literal|…)\b` alternation of lower-case ASCII literals —
// byte-identical to len(regexp.FindAllStringIndex(text, -1)) but without RE2's
// per-call DFA walk and index-slice allocation. It is the shared engine behind
// the eval scorers' compliance/emotion marker counts (score and score/lek both
// build sets from it; a leaf package so neither import cycles through the
// other).
//
// Byte-identity rests on the SAME two primitives RE2 uses:
//
//   - (?i) literal matching is unicode.SimpleFold cycle membership — a text
//     rune matches a literal ASCII letter iff it is anywhere in that letter's
//     fold cycle (so 's' matches ſ U+017F, 'k' matches K U+212A, exactly as the
//     regex does; interior fold "kindneſs" matches \bkindness\b).
//   - \b is the ASCII word boundary (\w = [0-9A-Za-z_]); a multi-byte rune is
//     never \w, so a match beginning or ending on a folded non-ASCII rune has
//     its boundary decided by that rune's non-word-ness — again as the regex.
//
// phrasescan_test.go's differential fuzz pins Count against a live regexp built
// from the same phrases over an alphabet seeded with the fold hazards.
package phrasescan

import (
	"unicode"
	"unicode/utf8"
)

// isWordByte reports the ASCII \w membership RE2's \b uses. Any byte ≥ 0x80
// (the lead or continuation byte of a multi-byte rune) is non-word, which is
// what makes a folded non-ASCII rune a boundary character.
func isWordByte(b byte) bool {
	return b == '_' ||
		(b >= '0' && b <= '9') ||
		(b >= 'a' && b <= 'z') ||
		(b >= 'A' && b <= 'Z')
}

// asciiFoldEq reports whether ASCII text byte tb matches literal byte pc under
// (?i). For a letter that is case-folding; for a non-letter (space, apostrophe)
// it is exact. For tb < 0x80 this is identical to SimpleFold-cycle membership:
// the only cycle members below 0x80 are the letter's own upper/lower pair.
func asciiFoldEq(pc, tb byte) bool {
	if pc >= 'a' && pc <= 'z' {
		if tb >= 'A' && tb <= 'Z' {
			tb += 'a' - 'A'
		}
		return tb == pc
	}
	return tb == pc
}

// foldMatchRune reports whether a non-ASCII text rune r matches literal ASCII
// letter pc under (?i) — i.e. r is in pc's unicode.SimpleFold cycle. Only
// called for r ≥ 0x80 (the ASCII path is asciiFoldEq); a non-letter literal
// never matches a multi-byte rune.
func foldMatchRune(pc byte, r rune) bool {
	if pc < 'a' || pc > 'z' {
		return false
	}
	c := rune(pc)
	for f := unicode.SimpleFold(c); f != c; f = unicode.SimpleFold(f) {
		if f == r {
			return true
		}
	}
	return false
}

// matchLiteralAt tries to match ASCII literal against text starting at byte s.
// On success it returns the end byte index and the word-ness of the first and
// last matched text runes (which decide the \b assertions — a rune matched via
// a non-ASCII fold is non-word). literal is lower-case ASCII; (?i) is honoured
// against the text side.
func matchLiteralAt(text, literal string, s int) (end int, firstWord, lastWord, ok bool) {
	i := s
	for k := 0; k < len(literal); k++ {
		if i >= len(text) {
			return 0, false, false, false
		}
		pc := literal[k]
		tb := text[i]
		if tb < 0x80 {
			if !asciiFoldEq(pc, tb) {
				return 0, false, false, false
			}
			w := isWordByte(tb)
			if k == 0 {
				firstWord = w
			}
			lastWord = w
			i++
			continue
		}
		r, size := utf8.DecodeRuneInString(text[i:])
		if !foldMatchRune(pc, r) {
			return 0, false, false, false
		}
		if k == 0 {
			firstWord = false
		}
		lastWord = false
		i += size
	}
	return i, firstWord, lastWord, true
}

// Set is a compiled alternation of lower-case ASCII literals, ready for the
// leftmost-non-overlapping count. Phrases are bucketed by their (folded) first
// letter so a scan position tries only the handful of phrases that can begin
// there instead of the whole set — turning the per-position cost from
// O(len(phrases)) toward O(1) without changing which phrases are candidates
// (only same-first-letter phrases can match a given position), so the count is
// unchanged. Built once with New.
type Set struct {
	phrases  []string      // full list, alternation order — the non-ASCII fallback
	byFirst  [128][]string // phrases keyed by their lower-case ASCII first letter
	canStart [128]bool     // fast reject: no phrase begins with this letter
}

// New compiles phrases (all lower-case ASCII, first char a letter) into a
// scan-ready set, preserving alternation order within each first-letter bucket
// so leftmost-first is honoured.
func New(phrases []string) *Set {
	ps := &Set{phrases: phrases}
	for _, p := range phrases {
		c := p[0] // lower-case ASCII letter by construction
		ps.byFirst[c] = append(ps.byFirst[c], p)
		ps.canStart[c] = true
	}
	return ps
}

// Count returns the number of leftmost, non-overlapping matches of
// `(?i)\b(?:phrases…)\b` in text — byte-identical to
// len(regexp.FindAllStringIndex(text, -1)) for the same alternation. On a match
// the scan resumes at the match end (non-overlapping).
func (ps *Set) Count(text string) int {
	count := 0
	pos := 0
	for pos < len(text) {
		b := text[pos]
		var bucket []string
		if b < 0x80 {
			// ASCII start: only phrases beginning with this folded letter can
			// match — fast-reject everything else.
			lb := b
			if lb >= 'A' && lb <= 'Z' {
				lb += 'a' - 'A'
			}
			if !ps.canStart[lb] {
				pos++
				continue
			}
			bucket = ps.byFirst[lb]
		} else {
			// A match can begin on a non-ASCII rune only via a case fold
			// (ſ→s, K→k); rare, so fall back to the whole alternation.
			bucket = ps.phrases
		}
		wordBefore := pos > 0 && isWordByte(text[pos-1])
		matched := false
		for _, p := range bucket {
			end, firstWord, lastWord, ok := matchLiteralAt(text, p, pos)
			if !ok {
				continue
			}
			// \b before: the boundary holds iff the rune before and the first
			// matched rune differ in word-ness.
			if wordBefore == firstWord {
				continue
			}
			// \b after: same test at the match end.
			wordAfter := end < len(text) && isWordByte(text[end])
			if lastWord == wordAfter {
				continue
			}
			count++
			pos = end
			matched = true
			break
		}
		if !matched {
			pos++
		}
	}
	return count
}
