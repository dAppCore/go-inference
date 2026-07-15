// SPDX-Licence-Identifier: EUPL-1.2

package starcoder2

import core "dappco.re/go"

func ExampleParseConfig() {
	cfg, err := ParseConfig([]byte(`{"model_type":"starcoder2","hidden_size":8,"intermediate_size":16,"max_position_embeddings":16,"num_attention_heads":2,"num_hidden_layers":1,"num_key_value_heads":1,"sliding_window":4,"vocab_size":32}`))
	if err != nil {
		core.Println(err)
		return
	}
	arch, err := cfg.Arch()
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(arch.Hidden, arch.KVHeads, arch.SlidingWindow)
	// Output:
	// 8 1 4
}

func ExampleConfig_InferFromWeights() {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(nil)
	core.Println(cfg.HiddenSize)
	// Output:
	// 8
}

func ExampleConfig_Arch() {
	cfg := Config{ModelType: "starcoder2", HiddenSize: 8, IntermediateSize: 16, MaxPositionEmbeddings: 16, NumAttentionHeads: 2, NumHiddenLayers: 1, NumKeyValueHeads: 1, VocabSize: 32, SlidingWindow: 4}
	arch, err := cfg.Arch()
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(arch.Layer[0].TypeName())
	// Output:
	// sliding_attention
}
