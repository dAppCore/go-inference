// SPDX-Licence-Identifier: EUPL-1.2

package gpt2

import core "dappco.re/go"

func ExampleConfig_Arch() {
	cfg := Config{Hidden: 768, Heads: 12, Layers: 12, Positions: 1024, Vocab: 50257}
	arch, err := cfg.Arch()
	core.Println(err == nil, arch.HeadDim, arch.LearnedAbsolutePositions)
	// Output: true 64 true
}

func ExampleConfig_InferFromWeights() {
	cfg := Config{Hidden: 768}
	cfg.InferFromWeights(nil) // no-op: GPT-2 declares every dim in config.json
	core.Println(cfg.Hidden)
	// Output: 768
}
