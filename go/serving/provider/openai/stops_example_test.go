// SPDX-Licence-Identifier: EUPL-1.2

package openai

import core "dappco.re/go"

func ExampleTruncateAtStopSequence() {
	got := TruncateAtStopSequence("Answer: 42.END trailing", []string{"END"})

	core.Println(got)
	// Output:
	// Answer: 42.
}
