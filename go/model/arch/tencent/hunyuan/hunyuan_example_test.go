// SPDX-Licence-Identifier: EUPL-1.2
package hunyuan_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/tencent/hunyuan"
)

func ExampleConfig() {
	_, ok := model.LookupArch("hunyuan_v1_dense")
	core.Println(ok) // Output: true
}

func ExampleConfig_Arch() {
	spec, _ := model.LookupArch("hunyuan_v1_dense")
	cfg, _ := spec.Parse([]byte(`{"model_type":"hunyuan_v1_dense","hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":0.00001,"rope_theta":10000}`))
	arch, err := cfg.Arch()
	core.Println(err == nil)
	core.Println(arch.HeadDim, arch.Heads)
	// Output:
	// true
	// 8 2
}

func ExampleConfig_InferFromWeights() {
	spec, _ := model.LookupArch("hunyuan_v1_dense")
	cfg, _ := spec.Parse([]byte(`{"model_type":"hunyuan_v1_dense","hidden_size":16}`))
	cfg.InferFromWeights(nil) // no-op: HunYuan declares every dim in config.json
	_, err := cfg.Arch()
	core.Println(err != nil) // still incomplete — InferFromWeights fabricates nothing
	// Output: true
}
