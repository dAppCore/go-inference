// SPDX-Licence-Identifier: EUPL-1.2

package gptoss

import core "dappco.re/go"

func ExampleParseConfig() {
	cfg, err := ParseConfig([]byte(`{"model_type":"gpt_oss","hidden_size":2880,"num_hidden_layers":24,"num_local_experts":32,"num_experts_per_tok":4}`))
	core.Println(err == nil, cfg.ModelType, cfg.HiddenSize)
	// Output: true gpt_oss 2880
}

func ExampleConfig_Arch() {
	// A fully-populated, structurally valid gpt_oss config resolves to a full decode Arch — every
	// layer MoE on the alternating sliding/full schedule, the clamped-SwiGLU activation declared.
	cfg := Config{
		HiddenSize: 2880, NumHiddenLayers: 24, NumAttentionHeads: 64, NumKeyValueHeads: 8, HeadDim: 64,
		VocabSize: 201088, NumLocalExperts: 32, NumExpertsPerTok: 4, IntermediateSize: 2880,
		LayerTypes: []string{
			"sliding_attention", "full_attention", "sliding_attention", "full_attention",
			"sliding_attention", "full_attention", "sliding_attention", "full_attention",
			"sliding_attention", "full_attention", "sliding_attention", "full_attention",
			"sliding_attention", "full_attention", "sliding_attention", "full_attention",
			"sliding_attention", "full_attention", "sliding_attention", "full_attention",
			"sliding_attention", "full_attention", "sliding_attention", "full_attention",
		},
	}
	a, err := cfg.Arch()
	core.Println(err == nil, len(a.Layer), a.Experts, a.Activation)
	// Output: true 24 32 gpt_oss_clamped_swiglu
}

func ExampleConfig_InferFromWeights() {
	cfg := Config{HiddenSize: 2880}
	cfg.InferFromWeights(nil) // no-op: GPT-OSS declares every dim in config.json
	core.Println(cfg.HiddenSize)
	// Output: 2880
}
