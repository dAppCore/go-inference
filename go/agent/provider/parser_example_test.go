// SPDX-License-Identifier: EUPL-1.2

package provider

import (
	core "dappco.re/go"
)

func ExampleParseFinalStatus() {
	result := ParseFinalStatus(`<<<LEM_STATUS>>>{"status":"waiting","question":"Which API?"}<<<END_LEM_STATUS>>>`)
	status := result.Value.(FinalStatus)
	core.Println(status.Status, status.Question)
	// Output: waiting Which API?
}
