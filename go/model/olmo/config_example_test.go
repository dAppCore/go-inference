// SPDX-Licence-Identifier: EUPL-1.2

package olmo

import core "dappco.re/go"

func ExampleConfig() {
	cfg := Config{ModelType: "olmo2", HiddenSize: 8, NumAttentionHeads: 2}
	core.Println(cfg.ModelType, cfg.HiddenSize/cfg.NumAttentionHeads)
	// Output:
	// olmo2 4
}

func ExampleParseConfig() {
	cfg, _ := ParseConfig([]byte(`{"model_type":"olmo","hidden_size":8}`))
	core.Println(cfg.ModelType, cfg.HiddenSize)
	// Output:
	// olmo 8
}

func ExampleConfig_InferFromWeights() {
	cfg := Config{ModelType: "olmo"}
	cfg.InferFromWeights(nil)
	core.Println(cfg.ModelType)
	// Output:
	// olmo
}

func ExampleConfig_Arch() {
	cfg := Config{ModelType: "olmo2", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 32}
	a, _ := cfg.Arch()
	core.Println(a.NormPlacement, a.HeadDim)
	// Output:
	// post 4
}
