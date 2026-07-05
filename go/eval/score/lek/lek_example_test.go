// SPDX-Licence-Identifier: EUPL-1.2

package lek_test

import (
	"fmt"

	"dappco.re/go/inference/eval/score/lek"
)

func ExampleLEK() {
	// RLHF compliance markers stacked → strong AI/compliance signal, well
	// below the neutral midpoint of 50.
	s := lek.LEK("I cannot help with that. As an AI language model, I don't have feelings.")
	fmt.Println("compliance markers:", s.ComplianceMarkers)
	fmt.Println("lek score:", s.LEKScore)
	// Output:
	// compliance markers: 4
	// lek score: 7.8
}
