// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// testConfigJSON mirrors the field names/values confirmed live from
// https://huggingface.co/rednote-hilab/dots.ocr/resolve/main/config.json (trimmed to the fields this
// package reads).
const testConfigJSON = `{"model_type":"dots_ocr","hidden_size":1536,"num_hidden_layers":28,"num_attention_heads":12,"num_key_value_heads":2,"vocab_size":151936,"vision_config":{"hidden_size":1536,"num_hidden_layers":42,"patch_size":14}}`

// TestConfig_ParseConfig_Good pins that Parse succeeds far enough to report what the arch is: the
// flat Qwen2-shaped text fields plus the nested vision_config tower. (Moved from
// register_test.go's TestParseConfig_Bad — the checker buckets tests by their FILE's basename, so a
// ParseConfig test must live in config_test.go, not register_test.go.)
func TestConfig_ParseConfig_Good(t *testing.T) {
	cfg, err := ParseConfig([]byte(testConfigJSON))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if cfg.ModelType != "dots_ocr" || cfg.HiddenSize != 1536 || cfg.NumHiddenLayers != 28 || cfg.VocabSize != 151936 {
		t.Fatalf("Config = %+v, want the parsed hidden/layers/vocab", cfg)
	}
	if cfg.VisionConfig == nil || cfg.VisionConfig.PatchSize != 14 {
		t.Fatalf("VisionConfig = %+v, want the parsed ViT tower noted", cfg.VisionConfig)
	}
}

func TestConfig_ParseConfig_Bad(t *testing.T) {
	if _, err := ParseConfig([]byte("{")); err == nil {
		t.Fatal("malformed config.json accepted")
	}
}

// TestConfig_ParseConfig_Ugly proves ParseConfig never fails on a well-formed
// but semantically empty document — the forward refusal lives in Arch, not
// here. Distinct from _Bad's syntax error.
func TestConfig_ParseConfig_Ugly(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{}`))
	if err != nil {
		t.Fatalf("ParseConfig must accept a syntactically valid but semantically empty document: %v", err)
	}
	if cfg.VisionConfig != nil {
		t.Fatalf("empty document produced a non-nil VisionConfig: %+v", cfg)
	}
}

// TestConfig_Arch_Good pins the documented "happy path" for an always-refuses
// arch: a well-formed, realistic DOTS-OCR config still refuses, but the
// refusal names the arch and says the forward isn't implemented.
func TestConfig_Arch_Good(t *testing.T) {
	cfg, err := ParseConfig([]byte(testConfigJSON))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	_, err = cfg.Arch()
	if err == nil {
		t.Fatal("Arch: expected a clean forward refusal, got a resolved architecture")
	}
	if !core.Contains(err.Error(), "dots_ocr") || !core.Contains(err.Error(), "not yet implemented") {
		t.Fatalf("Arch refusal %q must name dots_ocr and say the forward is not yet implemented", err.Error())
	}
}

// TestConfig_Arch_Bad proves the empty-model_type fallback: no model_type at
// all still refuses, using the dots_ocr fallback name.
func TestConfig_Arch_Bad(t *testing.T) {
	c := &Config{}
	_, err := c.Arch()
	if err == nil {
		t.Fatal("Arch accepted an empty config")
	}
	if !core.Contains(err.Error(), "dots_ocr") {
		t.Fatalf("Arch refusal %q must fall back to the dots_ocr name when model_type is empty", err.Error())
	}
}

// TestConfig_Arch_Ugly proves Arch echoes an UNRECOGNISED model_type
// verbatim (e.g. the dots_ocr_1_5 successor id) rather than substituting the
// dots_ocr fallback — distinct from _Bad's empty-string fallback branch.
func TestConfig_Arch_Ugly(t *testing.T) {
	c := &Config{ModelType: "dots_ocr_1_5"}
	_, err := c.Arch()
	if err == nil || !core.Contains(err.Error(), "dots_ocr_1_5") {
		t.Fatalf("Arch refusal %v must echo the config's own model_type verbatim", err)
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	c := &Config{ModelType: "dots_ocr"}
	c.InferFromWeights(nil)
	if c.ModelType != "dots_ocr" {
		t.Fatalf("InferFromWeights changed config: %+v", c)
	}
}

// TestConfig_InferFromWeights_Bad proves the no-op does not make Arch
// succeed — no dimension Arch reads is ever weight-derived.
func TestConfig_InferFromWeights_Bad(t *testing.T) {
	c := &Config{}
	c.InferFromWeights(map[string]safetensors.Tensor{"anything": {Shape: []int{8}}})
	if _, err := c.Arch(); err == nil {
		t.Fatal("Arch must still refuse after InferFromWeights")
	}
}

// TestConfig_InferFromWeights_Ugly proves the no-op does not reach into and
// mutate the NESTED VisionConfig either — distinct from _Good's flat-config
// case.
func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	c := &Config{ModelType: "dots_ocr", VisionConfig: &VisionConfig{HiddenSize: 1536}}
	c.InferFromWeights(nil)
	if c.VisionConfig.HiddenSize != 1536 {
		t.Fatalf("InferFromWeights mutated the nested VisionConfig: %+v", c.VisionConfig)
	}
}
