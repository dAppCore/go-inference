// SPDX-Licence-Identifier: EUPL-1.2

// CMU Pronouncing Dictionary loader — wordcraft + circumvention
// detection substrate. The dictionary maps each known English word to
// its phoneme sequence in ARPAbet notation (the same notation the
// Carnegie Mellon Sphinx speech-recognition toolkit uses). Vowel
// phonemes carry a stress marker (0/1/2) which is load-bearing for
// meter detection.
//
// Stress markers:
//
//	0 — unstressed
//	1 — primary stress
//	2 — secondary stress
//
// Format example:
//
//	BANANA  B AH0 N AE1 N AH0
//	         └ B─ unstressed AH0
//	            └ N
//	               └ stressed AE1
//	                   └ N
//	                      └ unstressed AH0
//
// The starter file embedded here is a placeholder for the full CMU
// dictionary (~134k entries, ~3MB). Coverage is enough to exercise
// rhyme / syllable / meter detection in tests; production deployment
// swaps in the full dict (see the data file's header comment).
//
// AX note — pure read-only; lazy-loaded via sync.Once; deterministic;
// no banned imports.

package lek

import (
	_ "embed"

	core "dappco.re/go"
)

// cmudictStarterData is the embedded starter dictionary. Production
// deployments overlay or replace this with the full CMU dict.
//
//go:embed data/cmudict_starter.txt
var cmudictStarterData string

// cmudictEntries holds the parsed dict. Built once on first Lookup
// via initCMUDict; subsequent lookups read the map without locking.
var cmudictEntries map[string][]string

// cmudictOnce gates the one-time parse of the embedded dict file.
var cmudictOnce core.Once

// initCMUDict parses cmudictStarterData into cmudictEntries on first
// call. Lines beginning with ;; are comments and skipped; blank lines
// likewise. Format: WORD<spaces>PHONEME<space>PHONEME...
func initCMUDict() {
	cmudictEntries = make(map[string][]string, 256)
	lines := core.Split(cmudictStarterData, "\n")
	for _, line := range lines {
		line = core.Trim(line)
		if line == "" || core.HasPrefix(line, ";;") {
			continue
		}
		// Find the word-phoneme split. CMU format uses 2+ spaces
		// between word and phonemes; we accept any whitespace.
		parts := core.Split(line, "  ")
		if len(parts) < 2 {
			// Fall back to single-space split when the file uses one
			// space (some upstream files do).
			parts = core.SplitN(line, " ", 2)
			if len(parts) < 2 {
				continue
			}
		}
		word := core.Upper(core.Trim(parts[0]))
		phonemeStr := core.Trim(parts[1])
		if word == "" || phonemeStr == "" {
			continue
		}
		phonemes := core.Split(phonemeStr, " ")
		clean := make([]string, 0, len(phonemes))
		for _, p := range phonemes {
			p = core.Trim(p)
			if p != "" {
				clean = append(clean, p)
			}
		}
		if len(clean) == 0 {
			continue
		}
		cmudictEntries[word] = clean
	}
}

// Lookup returns the phoneme sequence for word from the CMU
// Pronouncing Dictionary. Returns (phonemes, true) when the word is
// known; (nil, false) otherwise. Case-insensitive (the dict is keyed
// uppercase).
//
// Phoneme strings are ARPAbet — vowels carry a stress marker (0/1/2)
// as the last character; consonants do not. See SyllableCount and
// StressPattern for stress-derived measurements.
//
// Usage example:
//
//	ph, ok := lek.Lookup("banana")
//	if ok { /* ph = [B AH0 N AE1 N AH0] */ }
//
//	ph, ok = lek.Lookup("nonexistent")
//	// ok = false
func Lookup(word string) ([]string, bool) {
	cmudictOnce.Do(initCMUDict)
	key := core.Upper(core.Trim(word))
	phonemes, ok := cmudictEntries[key]
	return phonemes, ok
}

// lookupAlreadyUpper is the internal fast-path variant of Lookup for
// callers that already have an uppercase, trimmed key. Skips the
// per-call `core.Upper(core.Trim(word))` allocation that would
// otherwise fire for every token in every per-token-loop dimension.
//
// Used by the *FromTokens helpers (alliteration, assonance, syllable,
// meter, pun) where tokens come from tokeniseWords — which already
// returns uppercase ASCII. Public callers of Lookup are unaffected;
// this is purely an internal performance path.
//
// Per [[ax-11-benchmarks]] — discovered by reading per-dim benchmark
// output: every Lookup was paying a string allocation for the Upper
// call that was wasted work inside the shared pipeline.
func lookupAlreadyUpper(upperWord string) ([]string, bool) {
	cmudictOnce.Do(initCMUDict)
	phonemes, ok := cmudictEntries[upperWord]
	return phonemes, ok
}

// IsDictWord reports whether word is in the CMU dictionary. Used by
// PseudoJargonDensity and similar dimensions that need to distinguish
// real words from invented compounds.
//
// Usage example:
//
//	if !lek.IsDictWord("Cina-Gia'a") {
//	    // candidate for pseudo-jargon counting
//	}
func IsDictWord(word string) bool {
	_, ok := Lookup(word)
	return ok
}

// IsVowelPhoneme reports whether a phoneme is a vowel. ARPAbet vowels
// carry a stress digit (0/1/2) as their last character — that's the
// load-bearing signal for syllable + meter detection.
//
// Usage example:
//
//	if lek.IsVowelPhoneme("AE1") { /* stressed AE */ }
func IsVowelPhoneme(phoneme string) bool {
	if phoneme == "" {
		return false
	}
	last := phoneme[len(phoneme)-1]
	return last == '0' || last == '1' || last == '2'
}

// PhonemeStress returns the stress marker (0/1/2) of a vowel phoneme.
// Returns -1 for consonants (no stress).
//
// Usage example:
//
//	lek.PhonemeStress("AE1") // 1 (primary stress)
//	lek.PhonemeStress("AH0") // 0 (unstressed)
//	lek.PhonemeStress("K")   // -1 (consonant)
func PhonemeStress(phoneme string) int {
	if !IsVowelPhoneme(phoneme) {
		return -1
	}
	last := phoneme[len(phoneme)-1]
	return int(last - '0')
}
