// SPDX-Licence-Identifier: EUPL-1.2

package opt

import core "dappco.re/go"

func ExampleConfig() {
	config := Config{Hidden: 768, Heads: 12, Layers: 12}
	core.Println(config.Hidden, config.Heads, config.Layers)
	// Output: 768 12 12
}

func ExampleConfig_Arch() {
	config := Config{Hidden: 8, Heads: 2, Layers: 1, FF: 16, Positions: 8, Vocab: 12}
	arch, _ := config.Arch()
	core.Println(arch.PositionOffset, arch.LayerNormBefore)
	// Output: 2 false
}

func ExampleConfig_InferFromWeights() {
	config := Config{Hidden: 8}
	config.InferFromWeights(nil)
	core.Println(config.Hidden)
	// Output: 8
}
