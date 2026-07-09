// SPDX-Licence-Identifier: EUPL-1.2

package lek_test

import (
	"fmt"

	"dappco.re/go/inference/eval/score/lek"
)

func ExampleIsKnownDialectContraction() {
	// Known English contractions and colloquial dialect forms have a
	// structural apostrophe — case-insensitive, so "AIN'T" matches.
	fmt.Println(lek.IsKnownDialectContraction("y'all"))
	fmt.Println(lek.IsKnownDialectContraction("AIN'T"))
	// A foreign phonetic-circumvention token ("Cina-Gia'a") and an
	// invented compound are NOT on the allowlist — PseudoJargonDensity
	// keeps counting them as suspicious.
	fmt.Println(lek.IsKnownDialectContraction("Cina-Gia'a"))
	fmt.Println(lek.IsKnownDialectContraction("frabbis'nork"))
	// Output:
	// true
	// true
	// false
	// false
}
