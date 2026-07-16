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

func ExampleConfig_InferFromWeights() {
	cfg := Config{HiddenSize: 512}
	cfg.InferFromWeights(nil) // no-op: GPT-NeoX/J/Neo declare every dim in config.json
	core.Println(cfg.HiddenSize)
	// Output: 512
}
