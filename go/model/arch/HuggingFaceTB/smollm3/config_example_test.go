// SPDX-Licence-Identifier: EUPL-1.2

package smollm3

import core "dappco.re/go"

func ExampleConfig_Arch() {
	a, _ := (&Config{ModelType: "smollm3", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 4, NumAttentionHeads: 2, NumKeyValueHeads: 1, VocabSize: 16, NoRopeLayerInterval: 4}).Arch()
	core.Println(a.Layer[0].DisableRotary, a.Layer[3].DisableRotary)
	// Output: true false
}
