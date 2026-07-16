// SPDX-Licence-Identifier: EUPL-1.2
package exaone4_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/LGAI-EXAONE/exaone4"
)

func ExampleConfig() {
	_, ok := model.LookupArch("exaone4")
	core.Println(ok) // Output: true
}

func ExampleConfig_Arch() {
	spec, _ := model.LookupArch("exaone4")
	cfg, _ := spec.Parse([]byte(`{"model_type":"exaone4","hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":0.00001,"rope_theta":10000}`))
	arch, err := cfg.Arch()
	core.Println(err == nil)
	core.Println(arch.HeadDim, arch.Heads)
	// Output:
	// true
	// 8 2
}

func ExampleConfig_InferFromWeights() {
	spec, _ := model.LookupArch("exaone4")
	cfg, _ := spec.Parse([]byte(`{"model_type":"exaone4","hidden_size":16}`))
	cfg.InferFromWeights(nil) // no-op: EXAONE 4 declares every dim in config.json
	_, err := cfg.Arch()
	core.Println(err != nil) // still incomplete — InferFromWeights fabricates nothing
	// Output: true
}
