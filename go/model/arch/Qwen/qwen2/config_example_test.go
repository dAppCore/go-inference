// SPDX-Licence-Identifier: EUPL-1.2

package qwen2

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

func ExampleParseConfig() {
	cfg, err := ParseConfig([]byte(`{"model_type":"qwen2","hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":32}`))
	core.Println(err == nil, cfg.ModelType, cfg.HiddenSize)
	// Output: true qwen2 8
}

func ExampleConfig_Arch() {
	cfg := Config{HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, VocabSize: 32}
	arch, err := cfg.Arch()
	core.Println(err == nil, arch.HeadDim, arch.KVHeads)
	// Output: true 4 1
}

func ExampleConfig_InferFromWeights() {
	cfg := Config{NumHiddenLayers: 1, NumAttentionHeads: 2}
	cfg.InferFromWeights(map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": {Shape: []int{16, 8}},
		"model.embed_tokens.weight":              {Shape: []int{32, 8}},
	})
	core.Println(cfg.HeadDim, cfg.VocabSize)
	// Output: 8 32
}
