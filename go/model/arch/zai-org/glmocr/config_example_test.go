// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import core "dappco.re/go"

func ExampleParseConfig() {
	cfg, err := ParseConfig([]byte(`{"model_type":"glm_ocr"}`))
	core.Println(err == nil, cfg.ModelType)
	// Output: true glm_ocr
}

func ExampleConfig_InferFromWeights() {
	c := &Config{ModelType: "glm_ocr"}
	c.InferFromWeights(nil) // no-op: Arch refuses unconditionally, so no weight shape is ever consumed
	core.Println(c.ModelType)
	// Output: glm_ocr
}

func ExampleConfig_Arch() {
	c := &Config{ModelType: "glm_ocr"}
	_, err := c.Arch() // the vision-language forward is not yet implemented; Arch always refuses
	core.Println(err != nil)
	// Output: true
}
