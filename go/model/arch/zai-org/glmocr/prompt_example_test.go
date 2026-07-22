// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import core "dappco.re/go"

// ExampleBuildPrompt documents BuildPrompt's call shape: pass a *tokenizer.Tokenizer loaded via
// tokenizer.LoadTokenizer(dir + "/tokenizer.json") — a nil tokenizer (shown here so this Example
// needs no real checkpoint on disk) fails predictably. See live_test.go's
// TestLive_RealCheckpoint_BuildPrompt_Good for a real, executed run matched against the actual
// reference chat template's output.
func ExampleBuildPrompt() {
	cfg := &Config{ImageTokenID: 59280}
	_, _, err := BuildPrompt(nil, cfg, "Text Recognition:", 16)
	core.Println(err != nil)
	// Output: true
}

// ExamplePositionIDs computes the 3D mrope position ids for a 9-token sequence: 2 text tokens,
// a 4-token image span (a 2x2 merged grid), 3 more text tokens.
func ExamplePositionIDs() {
	mmType := []int32{0, 0, 1, 1, 1, 1, 0, 0, 0}
	tPos, hPos, wPos, err := PositionIDs(mmType, 1, 4, 4, 2)
	if err != nil {
		core.Println("PositionIDs:", err)
		return
	}
	core.Println(tPos)
	core.Println(hPos)
	core.Println(wPos)
	// Output:
	// [0 1 2 2 2 2 4 5 6]
	// [0 1 2 2 3 3 4 5 6]
	// [0 1 2 3 2 3 4 5 6]
}
