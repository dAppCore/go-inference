// SPDX-Licence-Identifier: EUPL-1.2

package parser

import core "dappco.re/go"

// ExampleHint shows the shape a caller builds to select a parser by model
// architecture, with an optional adapter-name fallback.
func ExampleHint() {
	hint := Hint{Architecture: "qwen3", AdapterName: "lora-coder"}
	core.Println(Family(hint))
	// Output: qwen
}

// ExampleResult shows the fields Filter returns: the visible text plus the
// captured reasoning trail.
func ExampleResult() {
	result := Filter("<think>plan</think>answer", Config{Mode: Capture}, Hint{Architecture: "qwen3"})
	core.Println(result.Text)
	core.Println(result.Reasoning)
	// Output:
	// answer
	// plan
}
