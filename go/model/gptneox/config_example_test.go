// SPDX-Licence-Identifier: EUPL-1.2

package gptneox

import core "dappco.re/go"

func ExampleConfig_Arch() {
	cfg := Config{ModelType: "gpt_neox", HiddenSize: 512, IntermediateSize: 2048, NumHiddenLayers: 6, NumAttentionHeads: 8, VocabSize: 50304, RotaryPct: .25}
	a, _ := cfg.Arch()
	core.Println(a.HeadDim, a.RotaryDim)
	// Output:
	// 64 16
}
