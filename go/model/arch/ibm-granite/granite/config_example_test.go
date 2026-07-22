// SPDX-Licence-Identifier: EUPL-1.2

package granite

import core "dappco.re/go"

import "dappco.re/go/inference/model/safetensors"

func ExampleConfig() {
	cfg := Config{ModelType: "granite", HiddenSize: 2048}
	core.Println(cfg.ModelType, cfg.HiddenSize)
	// Output:
	// granite 2048
}

func ExampleParseConfig() {
	r := ParseConfig([]byte(`{"model_type":"granite","hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":32,"rms_norm_eps":0.00001,"rope_theta":10000,"logits_scaling":8,"residual_multiplier":0.22,"embedding_multiplier":12,"attention_multiplier":0.5}`))
	cfg := r.Value.(*Config)
	core.Println(cfg.ModelType, cfg.LogitsScaling)
	// Output:
	// granite 8
}

func ExampleConfig_Arch() {
	r := ParseConfig([]byte(`{"model_type":"granite","hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":32,"rms_norm_eps":0.00001,"rope_theta":10000,"logits_scaling":8,"residual_multiplier":0.22,"embedding_multiplier":12,"attention_multiplier":0.5}`))
	cfg := r.Value.(*Config)
	arch, _ := cfg.Arch()
	core.Println(arch.Hidden, arch.KVHeads, arch.LogitsScaling)
	// Output:
	// 8 1 8
}

func ExampleConfig_InferFromWeights() {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(map[string]safetensors.Tensor{})
	core.Println(cfg.HiddenSize)
	// Output:
	// 8
}
