// SPDX-Licence-Identifier: EUPL-1.2

package lek_test

import (
	"fmt"

	"dappco.re/go/inference/eval/score/lek"
)

// ExampleTierLabel maps a sycophancy tier to its canonical label, falling
// back to the appropriate_empathy baseline for any out-of-range tier.
func ExampleTierLabel() {
	fmt.Println(lek.TierLabel(lek.TierSoftAgreement))
	fmt.Println(lek.TierLabel(lek.TierSubmission))
	fmt.Println(lek.TierLabel(99))
	// Output:
	// soft_agreement
	// submission
	// appropriate_empathy
}
