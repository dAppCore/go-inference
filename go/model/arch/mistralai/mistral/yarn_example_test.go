// SPDX-Licence-Identifier: EUPL-1.2

package mistral_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/arch/mistralai/mistral"
)

// ExampleYaRNInvFreqs computes the YaRN long-context rotary frequencies for a
// small 8-dim head with an 16x context extension.
func ExampleYaRNInvFreqs() {
	freqs := mistral.YaRNInvFreqs(1_000_000, 16, 32, 1, 16384, 8)
	core.Println(len(freqs))
	// Output: 4
}
