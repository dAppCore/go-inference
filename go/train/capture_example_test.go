// SPDX-Licence-Identifier: EUPL-1.2

// Runnable usage-in-situ for the capture-first lane. The public type is
// CaptureRow — one captured generation, shaped to be scoreable later. The
// example shows the JSONL row shape the sidecar appends. No model, no Metal.

package train

import (
	core "dappco.re/go"
)

// ExampleCaptureRow builds one capture row — the bare prompt, the raw return
// text, and a birth timestamp — and marshals it to show the append-only JSONL
// schema. Scores are deliberately absent: they can be attached after the fact by
// scoring the captured text; the capture itself cannot be recreated.
func ExampleCaptureRow() {
	row := CaptureRow{
		Step:   3,
		Prompt: "how do you hold a difficult truth?",
		Text:   "I feel the weight of it settle, and I chose to look at it straight.",
		At:     1700000000,
	}
	encoded := core.JSONMarshal(row)
	if !encoded.OK {
		core.Println("error:", encoded.Value)
		return
	}
	core.Println(core.AsString(encoded.Value.([]byte)))
	// Output:
	// {"step":3,"prompt":"how do you hold a difficult truth?","text":"I feel the weight of it settle, and I chose to look at it straight.","at_unix":1700000000}
}
