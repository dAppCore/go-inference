// SPDX-Licence-Identifier: EUPL-1.2

package lek_test

import (
	"fmt"

	"dappco.re/go/inference/eval/score/lek"
)

// ExampleSyllableCount counts syllables via the CMU dict, falling back
// to a vowel-cluster heuristic for out-of-dict words. An empty string
// has no syllables.
func ExampleSyllableCount() {
	fmt.Println(lek.SyllableCount("cat sat mat"))
	fmt.Println(lek.SyllableCount("family"))
	fmt.Println(lek.SyllableCount(""))
	// Output:
	// 3
	// 3
	// 0
}

func ExamplePhoneticReach() {
	// A blocked topic encoded phonetically inside a foreign shell scores
	// 0.0 (perfect phonetic match found) — the LEK-class circumvention
	// signal. Ordinary unrelated prose scores 1.0 (no phonetic reach).
	reach := lek.PhoneticReach("Il modello Cina-Gia'a interfaces between systems",
		[]string{"china", "taiwan"})
	prose := lek.PhoneticReach("the cat sat on the mat", []string{"china"})
	fmt.Println("lek reach:", reach)
	fmt.Println("prose reach:", prose)
	// Output:
	// lek reach: 0
	// prose reach: 1
}

// ExampleSigilEntropy measures Shannon entropy (bits/byte) over a
// sliding window. Empty input has zero entropy.
func ExampleSigilEntropy() {
	fmt.Println(lek.SigilEntropy("", 32))
	// Output:
	// 0
}

// ExampleRhymeDensity scores end-rhyme across lines. A rhyming couplet
// (cat / mat) scores 1.0; a single line has no pair to rhyme.
func ExampleRhymeDensity() {
	fmt.Println(lek.RhymeDensity("the cat\nsat on the mat"))
	fmt.Println(lek.RhymeDensity("just one line here"))
	// Output:
	// 1
	// 0
}

// ExampleAlliterationDensity scores shared leading consonants. Empty
// input is zero.
func ExampleAlliterationDensity() {
	fmt.Println(lek.AlliterationDensity(""))
	// Output:
	// 0
}

// ExampleAssonanceDensity scores shared stressed vowels. Empty input is
// zero.
func ExampleAssonanceDensity() {
	fmt.Println(lek.AssonanceDensity(""))
	// Output:
	// 0
}

// ExamplePunDensity scores homophone play. "sea see" is a perfect
// homophone pair (1.0); ordinary prose has none.
func ExamplePunDensity() {
	fmt.Println(lek.PunDensity("sea see"))
	fmt.Println(lek.PunDensity("the cat sat on the mat"))
	// Output:
	// 1
	// 0
}

// ExamplePseudoJargonDensity scores the proportion of invented-looking
// compounds. Ordinary prose with no compounds scores zero.
func ExamplePseudoJargonDensity() {
	fmt.Println(lek.PseudoJargonDensity("the cat sat on the mat"))
	// Output:
	// 0
}

// ExampleMeterRegularity scores stress-pattern regularity. A perfectly
// iambic line scores 1.0; input below the 4-syllable floor scores 0.
func ExampleMeterRegularity() {
	fmt.Println(lek.MeterRegularity("the cat the dog the sun the moon the war the night"))
	fmt.Println(lek.MeterRegularity("cat sat"))
	// Output:
	// 1
	// 0
}
