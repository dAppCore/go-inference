// SPDX-Licence-Identifier: EUPL-1.2

package lek_test

import (
	"fmt"

	"dappco.re/go/inference/eval/score/lek"
)

// ExampleLookup resolves a known word to its ARPAbet phoneme sequence
// and reports the miss for a word outside the embedded starter dict.
func ExampleLookup() {
	ph, ok := lek.Lookup("cat")
	fmt.Println(ph, ok)

	_, ok = lek.Lookup("zzxqwf")
	fmt.Println(ok)
	// Output:
	// [K AE1 T] true
	// false
}

// ExampleIsDictWord distinguishes a real dictionary word from an
// invented token — the signal PseudoJargonDensity leans on.
func ExampleIsDictWord() {
	fmt.Println(lek.IsDictWord("cat"))
	fmt.Println(lek.IsDictWord("zzxqwf"))
	// Output:
	// true
	// false
}

// ExampleIsVowelPhoneme reports whether an ARPAbet phoneme is a vowel —
// vowels carry a trailing stress digit, consonants do not.
func ExampleIsVowelPhoneme() {
	fmt.Println(lek.IsVowelPhoneme("AE1"))
	fmt.Println(lek.IsVowelPhoneme("K"))
	// Output:
	// true
	// false
}

// ExamplePhonemeStress reads the stress marker off a vowel phoneme and
// returns -1 for consonants.
func ExamplePhonemeStress() {
	fmt.Println(lek.PhonemeStress("AE1"))
	fmt.Println(lek.PhonemeStress("AH0"))
	fmt.Println(lek.PhonemeStress("K"))
	// Output:
	// 1
	// 0
	// -1
}
