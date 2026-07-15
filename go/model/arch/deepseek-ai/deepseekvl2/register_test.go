// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// testConfigJSON mirrors the field names/values confirmed live from
// https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/config.json (trimmed to the
// fields this package reads; the real file also duplicates hidden_size/num_hidden_layers/
// vocab_size at the top level, which this package intentionally does not read — language_config
// is the semantically correct location).
const testConfigJSON = `{"model_type":"deepseek_vl_v2","language_config":{"hidden_size":1280,"num_hidden_layers":12,"vocab_size":129280},"vision_config":{"model_type":"vision","model_name":"deeplip_b_l","image_size":1024}}`

// TestDeepseekvl2Registered_Good pins the deepseek_vl_v2 "recognised, not yet implemented"
// contract: model.LookupArch succeeds (a user pointing lem at a DeepSeek-OCR checkpoint gets
// direction, not "unknown model architecture"), Parse succeeds far enough to report what the
// arch is, and both the Arch and Composed refusal seams name the arch and say why the forward is
// missing — mirroring composed.TestComposedMTPRefusal's discipline for a non-hybrid
// vision-language arch.
func TestDeepseekvl2Registered_Good(t *testing.T) {
	spec, ok := model.LookupArch("deepseek_vl_v2")
	if !ok {
		t.Fatal("model_type deepseek_vl_v2 not registered — an OCR arch must be recognised, not fall to \"unknown model architecture\"")
	}
	if spec.Parse == nil {
		t.Fatal("deepseek_vl_v2 registered without a Parse hook")
	}
	if spec.Composed == nil {
		t.Fatal("deepseek_vl_v2 registered without a Composed hook")
	}

	ac, err := spec.Parse([]byte(testConfigJSON))
	if err != nil {
		t.Fatalf("Parse must succeed enough to report what the arch is: %v", err)
	}
	cfg, ok := ac.(*Config)
	if !ok {
		t.Fatalf("Parse returned %T, want *Config", ac)
	}
	if cfg.LanguageConfig == nil || cfg.LanguageConfig.HiddenSize != 1280 || cfg.LanguageConfig.NumHiddenLayers != 12 || cfg.LanguageConfig.VocabSize != 129280 {
		t.Fatalf("LanguageConfig = %+v, want the parsed hidden/layers/vocab", cfg.LanguageConfig)
	}
	if cfg.VisionConfig == nil || cfg.VisionConfig.ModelName != "deeplip_b_l" {
		t.Fatalf("VisionConfig = %+v, want the parsed dual-tower vision config noted", cfg.VisionConfig)
	}

	if _, err := ac.Arch(); err == nil {
		t.Fatal("Arch: expected a clean forward refusal, got a resolved architecture")
	} else if !core.Contains(err.Error(), "deepseek_vl_v2") || !core.Contains(err.Error(), "not yet implemented") {
		t.Fatalf("Arch refusal %q must name deepseek_vl_v2 and say the forward is not yet implemented", err.Error())
	}

	tm, err := spec.Composed(map[string]safetensors.Tensor{}, []byte(testConfigJSON))
	if err == nil {
		t.Fatal("Composed: expected a clean forward refusal, got a model")
	}
	if tm != nil {
		t.Fatal("Composed: refusal must return a nil model")
	}
	if !core.Contains(err.Error(), "deepseek_vl_v2") || !core.Contains(err.Error(), "not yet implemented") {
		t.Fatalf("Composed refusal %q must name deepseek_vl_v2 and say the forward is not yet implemented", err.Error())
	}
}

// TestParseConfig_Bad pins that a malformed config.json is rejected at Parse, not silently
// accepted into a zero-value Config that would later refuse for the wrong reason.
func TestParseConfig_Bad(t *testing.T) {
	if _, err := ParseConfig([]byte("{")); err == nil {
		t.Fatal("malformed config.json accepted")
	}
}
