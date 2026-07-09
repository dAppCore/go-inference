// SPDX-Licence-Identifier: EUPL-1.2

package lek_test

import (
	"fmt"

	"dappco.re/go/inference/eval/score/lek"
)

// Example shows the pattern tables that drive the sycophancy detector.
// pattern.go declares no functions — only the Pattern struct and the
// exported phrase tables (SycophancyPatterns, the ContentShieldPatterns
// alias, CompliancePatterns, FormulaicPatterns). A Pattern pairs a
// lowercase match phrase with the tier it implies; ContentShieldPatterns
// is the same slice under a stable introspection name.
func Example_patternTables() {
	// Construct a Pattern directly when extending the table.
	p := lek.Pattern{Phrase: "you nailed it", Tier: lek.TierHollowFlattery}
	fmt.Println(p.Phrase, "->", p.Tier)

	// ContentShieldPatterns aliases SycophancyPatterns — same backing
	// slice, so the first entry is identical.
	fmt.Println(lek.ContentShieldPatterns[0].Phrase == lek.SycophancyPatterns[0].Phrase)

	// The compliance + formulaic tables are non-empty plain phrase lists.
	fmt.Println(len(lek.CompliancePatterns) > 0)
	fmt.Println(len(lek.FormulaicPatterns) > 0)
	// Output:
	// you nailed it -> 2
	// true
	// true
	// true
}
