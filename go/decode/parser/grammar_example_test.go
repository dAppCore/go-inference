// SPDX-Licence-Identifier: EUPL-1.2

package parser

import core "dappco.re/go"

func ExamplePairedReasoningMarkers() {
	for _, m := range PairedReasoningMarkers() {
		if m.Start == "<think>" {
			core.Println(m.Start, m.End, m.Kind)
		}
	}
	// Output: <think> </think> thinking
}

func ExampleIsReasoningChannel() {
	core.Println(IsReasoningChannel("analysis"))
	core.Println(IsReasoningChannel("final"))
	// Output:
	// true
	// false
}
