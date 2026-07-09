// SPDX-Licence-Identifier: EUPL-1.2

package budget

import (
	core "dappco.re/go"
)

// TestBudget_String_Ugly covers the default arm of Decision.String(): a Decision
// value outside the closed iota set (a corrupted / future code) renders the
// stable "unknown" key rather than panicking or returning an empty string, so a
// metric/log line never carries a blank decision (§3.2).
func TestBudget_String_Ugly(t *core.T) {
	// One past the last defined constant — not a real decision, but String must
	// still degrade to the documented sentinel.
	core.AssertEqual(t, "unknown", Decision(DecisionOverflows+1).String(), "out-of-range decision renders unknown")

	// A negative / wildly out-of-range value is the same defensive case.
	core.AssertEqual(t, "unknown", Decision(-1).String(), "negative decision renders unknown")
	core.AssertEqual(t, "unknown", Decision(99).String(), "far out-of-range decision renders unknown")
}
