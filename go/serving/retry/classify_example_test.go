// SPDX-Licence-Identifier: EUPL-1.2

package retry

import (
	core "dappco.re/go"
)

// ExampleClassify demonstrates mapping an HTTP-ish status onto its failure
// class, and that class feeding straight into Retryable.
func ExampleClassify() {
	class := Classify(429)
	core.Println(class == ClassRateLimited)
	core.Println(Retryable(class))
	// Output:
	// true
	// true
}

// ExampleRetryable demonstrates the retryable/permanent split: an upstream
// overload is worth trying again, a malformed request is not.
func ExampleRetryable() {
	core.Println(Retryable(ClassRateLimited))
	core.Println(Retryable(ClassBadRequest))
	// Output:
	// true
	// false
}
