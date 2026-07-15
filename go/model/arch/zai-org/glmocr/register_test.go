// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// testConfigJSON mirrors the field names/values confirmed live from
// https://huggingface.co/zai-org/GLM-OCR/resolve/main/config.json (trimmed to the fields this
// package reads).
const testConfigJSON = `{"model_type":"glm_ocr","text_config":{"model_type":"glm_ocr_text","hidden_size":1536,"num_hidden_layers":16,"vocab_size":59392},"vision_config":{"model_type":"glm_ocr_vision","hidden_size":1024,"depth":24,"patch_size":14}}`

// TestGlmocrRegistered_Good pins the glm_ocr / glm_ocr_text "recognised, not yet implemented"
// contract: model.LookupArch succeeds for both the wrapper id and its nested text_config alias
// (a user pointing lem at a GLM-OCR checkpoint gets direction, not "unknown model
// architecture"), Parse succeeds far enough to report what the arch is, and both the Arch and
// Composed refusal seams name the arch and say why the forward is missing — mirroring
// composed.TestComposedMTPRefusal's discipline for a non-hybrid vision-language arch.
func TestGlmocrRegistered_Good(t *testing.T) {
	for _, mt := range []string{"glm_ocr", "glm_ocr_text"} {
		spec, ok := model.LookupArch(mt)
		if !ok {
			t.Fatalf("model_type %q not registered — an OCR arch must be recognised, not fall to \"unknown model architecture\"", mt)
		}
		if spec.Parse == nil {
			t.Fatalf("%q registered without a Parse hook", mt)
		}
		if spec.Composed == nil {
			t.Fatalf("%q registered without a Composed hook", mt)
		}
	}

	spec, _ := model.LookupArch("glm_ocr")
	ac, err := spec.Parse([]byte(testConfigJSON))
	if err != nil {
		t.Fatalf("Parse must succeed enough to report what the arch is: %v", err)
	}
	cfg, ok := ac.(*Config)
	if !ok {
		t.Fatalf("Parse returned %T, want *Config", ac)
	}
	if cfg.TextConfig == nil || cfg.TextConfig.HiddenSize != 1536 || cfg.TextConfig.NumHiddenLayers != 16 || cfg.TextConfig.VocabSize != 59392 {
		t.Fatalf("TextConfig = %+v, want the parsed hidden/layers/vocab", cfg.TextConfig)
	}
	if cfg.VisionConfig == nil || cfg.VisionConfig.Depth != 24 {
		t.Fatalf("VisionConfig = %+v, want the parsed vision tower noted", cfg.VisionConfig)
	}

	if _, err := ac.Arch(); err == nil {
		t.Fatal("Arch: expected a clean forward refusal, got a resolved architecture")
	} else if !core.Contains(err.Error(), "glm_ocr") || !core.Contains(err.Error(), "not yet implemented") {
		t.Fatalf("Arch refusal %q must name glm_ocr and say the forward is not yet implemented", err.Error())
	}

	tm, err := spec.Composed(map[string]safetensors.Tensor{}, []byte(testConfigJSON))
	if err == nil {
		t.Fatal("Composed: expected a clean forward refusal, got a model")
	}
	if tm != nil {
		t.Fatal("Composed: refusal must return a nil model")
	}
	if !core.Contains(err.Error(), "glm_ocr") || !core.Contains(err.Error(), "not yet implemented") {
		t.Fatalf("Composed refusal %q must name glm_ocr and say the forward is not yet implemented", err.Error())
	}
}

// TestParseConfig_Bad pins that a malformed config.json is rejected at Parse, not silently
// accepted into a zero-value Config that would later refuse for the wrong reason.
func TestParseConfig_Bad(t *testing.T) {
	if _, err := ParseConfig([]byte("{")); err == nil {
		t.Fatal("malformed config.json accepted")
	}
}
