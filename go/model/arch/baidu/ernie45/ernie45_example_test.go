// SPDX-Licence-Identifier: EUPL-1.2
package ernie45_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/baidu/ernie45"
)

func ExampleConfig() {
	_, ok := model.LookupArch("ernie4_5")
	core.Println(ok) // Output: true
}

func ExampleConfig_Arch() {
	spec, _ := model.LookupArch("ernie4_5")
	cfg, _ := spec.Parse([]byte(`{"model_type":"ernie4_5","hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":0.00001,"rope_theta":10000}`))
	arch, err := cfg.Arch()
	core.Println(err == nil)
	core.Println(arch.HeadDim, arch.Heads)
	// Output:
	// true
	// 8 2
}

func ExampleConfig_InferFromWeights() {
	spec, _ := model.LookupArch("ernie4_5")
	cfg, _ := spec.Parse([]byte(`{"model_type":"ernie4_5","hidden_size":16}`))
	cfg.InferFromWeights(nil) // no-op: ERNIE 4.5 declares every dim in config.json
	_, err := cfg.Arch()
	core.Println(err != nil) // still incomplete — InferFromWeights fabricates nothing
	// Output: true
}
