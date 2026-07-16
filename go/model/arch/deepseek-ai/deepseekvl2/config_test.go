// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// TestConfig_ParseConfig_Good pins that a malformed config.json is rejected at
// Parse, not silently accepted into a zero-value Config that would later
// refuse for the wrong reason. (Moved from register_test.go's
// TestParseConfig_Bad — the checker buckets tests by their FILE's basename,
// so a ParseConfig test must live in config_test.go, not register_test.go.)
func TestConfig_ParseConfig_Good(t *testing.T) {
	cfg, err := ParseConfig([]byte(testConfigJSON))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if cfg.ModelType != "deepseek_vl_v2" {
		t.Fatalf("ModelType = %q, want deepseek_vl_v2", cfg.ModelType)
	}
	if cfg.LanguageConfig == nil || cfg.LanguageConfig.HiddenSize != 1280 || cfg.LanguageConfig.NumHiddenLayers != 12 || cfg.LanguageConfig.VocabSize != 129280 {
		t.Fatalf("LanguageConfig = %+v, want the parsed hidden/layers/vocab", cfg.LanguageConfig)
	}
	if cfg.VisionConfig == nil || cfg.VisionConfig.ModelName != "deeplip_b_l" {
		t.Fatalf("VisionConfig = %+v, want the parsed dual-tower vision config noted", cfg.VisionConfig)
	}
}

func TestConfig_ParseConfig_Bad(t *testing.T) {
	if _, err := ParseConfig([]byte("{")); err == nil {
		t.Fatal("malformed config.json accepted")
	}
}

// TestConfig_ParseConfig_Ugly proves ParseConfig never fails on a well-formed
// but semantically empty document — the forward refusal lives in Arch, not
// here. Distinct from _Bad's syntax-error rejection.
func TestConfig_ParseConfig_Ugly(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{}`))
	if err != nil {
		t.Fatalf("ParseConfig must accept a syntactically valid but semantically empty document: %v", err)
	}
	if cfg.LanguageConfig != nil || cfg.VisionConfig != nil {
		t.Fatalf("empty document produced a non-nil nested config: %+v", cfg)
	}
}

// TestConfig_Arch_Good pins the documented "happy path" for an
// always-refuses arch: a well-formed, realistic DeepSeek-OCR config still
// refuses, but the refusal names the arch and says the forward isn't
// implemented — the expected, correctly-shaped behaviour.
func TestConfig_Arch_Good(t *testing.T) {
	cfg, err := ParseConfig([]byte(testConfigJSON))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	_, err = cfg.Arch()
	if err == nil {
		t.Fatal("Arch: expected a clean forward refusal, got a resolved architecture")
	}
	if !core.Contains(err.Error(), "deepseek_vl_v2") || !core.Contains(err.Error(), "not yet implemented") {
		t.Fatalf("Arch refusal %q must name deepseek_vl_v2 and say the forward is not yet implemented", err.Error())
	}
}

// TestConfig_Arch_Bad proves the empty-model_type fallback: no model_type at
// all still refuses, using the deepseek_vl_v2 fallback name.
func TestConfig_Arch_Bad(t *testing.T) {
	c := &Config{}
	_, err := c.Arch()
	if err == nil {
		t.Fatal("Arch accepted an empty config")
	}
	if !core.Contains(err.Error(), "deepseek_vl_v2") {
		t.Fatalf("Arch refusal %q must fall back to the deepseek_vl_v2 name when model_type is empty", err.Error())
	}
}

// TestConfig_Arch_Ugly proves Arch echoes an UNRECOGNISED model_type
// verbatim rather than substituting the deepseek_vl_v2 fallback or a generic
// message — distinct from _Bad's empty-string fallback branch.
func TestConfig_Arch_Ugly(t *testing.T) {
	c := &Config{ModelType: "not-a-real-model-type"}
	_, err := c.Arch()
	if err == nil || !core.Contains(err.Error(), "not-a-real-model-type") {
		t.Fatalf("Arch refusal %v must echo the config's own model_type verbatim", err)
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	c := &Config{ModelType: "deepseek_vl_v2"}
	c.InferFromWeights(nil)
	if c.ModelType != "deepseek_vl_v2" {
		t.Fatalf("InferFromWeights changed config: %+v", c)
	}
}

// TestConfig_InferFromWeights_Bad proves the no-op does not make Arch
// succeed — no dimension Arch reads is ever weight-derived, so a checkpoint
// full of tensors changes nothing.
func TestConfig_InferFromWeights_Bad(t *testing.T) {
	c := &Config{}
	c.InferFromWeights(map[string]safetensors.Tensor{"anything": {Shape: []int{8}}})
	if _, err := c.Arch(); err == nil {
		t.Fatal("Arch must still refuse after InferFromWeights")
	}
}

// TestConfig_InferFromWeights_Ugly proves the no-op does not reach into and
// mutate the NESTED LanguageConfig either — distinct from _Good's flat-config
// case.
func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	c := &Config{ModelType: "deepseek_vl_v2", LanguageConfig: &LanguageConfig{HiddenSize: 1280}}
	c.InferFromWeights(nil)
	if c.LanguageConfig.HiddenSize != 1280 {
		t.Fatalf("InferFromWeights mutated the nested LanguageConfig: %+v", c.LanguageConfig)
	}
}
