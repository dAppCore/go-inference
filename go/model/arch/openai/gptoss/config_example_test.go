// SPDX-Licence-Identifier: EUPL-1.2

package gptoss

import core "dappco.re/go"

func ExampleParseConfig() {
	cfg, err := ParseConfig([]byte(`{"model_type":"gpt_oss","hidden_size":2880,"num_hidden_layers":24,"num_local_experts":32,"num_experts_per_tok":4}`))
	core.Println(err == nil, cfg.ModelType, cfg.HiddenSize)
	// Output: true gpt_oss 2880
}

func ExampleConfig_Arch() {
	cfg := Config{NumHiddenLayers: 24, NumLocalExperts: 32, NumExpertsPerTok: 4, VocabSize: 201088}
	_, err := cfg.Arch()
	core.Println(err != nil)
	// Output: true
}

func ExampleConfig_InferFromWeights() {
	cfg := Config{HiddenSize: 2880}
	cfg.InferFromWeights(nil) // no-op: GPT-OSS declares every dim in config.json
	core.Println(cfg.HiddenSize)
	// Output: 2880
}
