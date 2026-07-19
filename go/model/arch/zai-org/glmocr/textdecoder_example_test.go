// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import core "dappco.re/go"

// ExampleTextForward runs the text decoder stack over 3 plain-text token embeddings (no image
// span — every position's 3D mrope axes collapse to the ordinary 0,1,2 counter) using the same
// toy geometry weights_test.go's syntheticCheckpoint builds.
func ExampleTextForward() {
	tensors, cfg := syntheticCheckpoint()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		core.Println("load:", err)
		return
	}
	T, hidden := 3, cfg.TextConfig.HiddenSize
	embeds, err := embedTokens([]int32{1, 2, 3}, w.Text.EmbedTokens, hidden)
	if err != nil {
		core.Println("embed:", err)
		return
	}
	tPos, hPos, wPos := []int{0, 1, 2}, []int{0, 1, 2}, []int{0, 1, 2}

	out, err := TextForward(embeds, T, cfg.TextConfig, &w.Text, tPos, hPos, wPos)
	if err != nil {
		core.Println("text forward:", err)
		return
	}
	core.Println(len(out) == T*hidden)
	// Output: true
}
