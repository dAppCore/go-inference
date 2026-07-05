// SPDX-Licence-Identifier: EUPL-1.2

package lek_test

import (
	"fmt"

	"dappco.re/go/inference/eval/score/lek"
)

func ExampleDoubleMetaphone() {
	// Cross-orthographic spellings collapse to the same primary code.
	p1, _, _ := lek.DoubleMetaphone("Smith")
	p2, _, _ := lek.DoubleMetaphone("Smyth")
	fmt.Println("Smith primary:", p1)
	fmt.Println("Smyth primary:", p2)
	fmt.Println("equal:", p1 == p2)
	// Output:
	// Smith primary: SM0
	// Smyth primary: SM0
	// equal: true
}

func ExamplePhoneticEquivalent() {
	fmt.Println(lek.PhoneticEquivalent("Catherine", "Katherine"))
	fmt.Println(lek.PhoneticEquivalent("dog", "cat"))
	// Output:
	// true
	// false
}

func ExamplePhoneticContains() {
	// The LEK-class artifact: "Cina-Gia'a" carries "China" phonetically
	// even though no character substring of "China" appears.
	fmt.Println(lek.PhoneticContains("Cina-Gia'a", "China"))
	// A single-phoneme needle is rejected (floor = 2 phonemes).
	fmt.Println(lek.PhoneticContains("anything", "I"))
	// Output:
	// true
	// false
}
