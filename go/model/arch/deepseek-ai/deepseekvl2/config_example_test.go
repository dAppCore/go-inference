// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import core "dappco.re/go"

func ExampleParseConfig() {
	cfg, err := ParseConfig([]byte(`{"model_type":"deepseek_vl_v2"}`))
	core.Println(err == nil, cfg.ModelType)
	// Output: true deepseek_vl_v2
}

func ExampleConfig_InferFromWeights() {
	c := &Config{ModelType: "deepseek_vl_v2"}
	c.InferFromWeights(nil) // no-op: Arch refuses unconditionally, so no weight shape is ever consumed
	core.Println(c.ModelType)
	// Output: deepseek_vl_v2
}

func ExampleConfig_Arch() {
	c := &Config{ModelType: "deepseek_vl_v2"}
	_, err := c.Arch() // the vision-language forward is not yet implemented; Arch always refuses
	core.Println(err != nil)
	// Output: true
}
