// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestDotsocrRegistered_Good pins the dots_ocr / dots_ocr_1_5 "recognised, not yet implemented"
// contract for both ids: model.LookupArch succeeds (a user pointing lem at a DOTS-OCR checkpoint
// gets direction, not "unknown model architecture"), Parse succeeds far enough to report what
// the arch is, and the Arch refusal seam names the arch that actually resolved
// and says why the forward is missing — the recognised-with-its-own-verb discipline
// for a non-hybrid vision-language arch. The config fixture mirrors the field names/values
// confirmed live from https://huggingface.co/rednote-hilab/dots.ocr/resolve/main/config.json
// (trimmed to the fields this package reads); dots_ocr_1_5 has no independently confirmable
// config.json upstream yet, so the fixture is reused with model_type swapped in — same Qwen2
// decoder + ViT tower shape, versioned, per this package's doc comment.
func TestDotsocrRegistered_Good(t *testing.T) {
	for _, mt := range []string{"dots_ocr", "dots_ocr_1_5"} {
		spec, ok := model.LookupArch(mt)
		if !ok {
			t.Fatalf("model_type %q not registered — an OCR arch must be recognised, not fall to \"unknown model architecture\"", mt)
		}
		if spec.Parse == nil {
			t.Fatalf("%q registered without a Parse hook", mt)
		}

		cfgJSON := []byte(`{"model_type":"` + mt + `","hidden_size":1536,"num_hidden_layers":28,"num_attention_heads":12,"num_key_value_heads":2,"vocab_size":151936,"vision_config":{"hidden_size":1536,"num_hidden_layers":42,"patch_size":14}}`)

		ac, err := spec.Parse(cfgJSON)
		if err != nil {
			t.Fatalf("%q: Parse must succeed enough to report what the arch is: %v", mt, err)
		}
		cfg, ok := ac.(*Config)
		if !ok {
			t.Fatalf("%q: Parse returned %T, want *Config", mt, ac)
		}
		if cfg.HiddenSize != 1536 || cfg.NumHiddenLayers != 28 || cfg.VocabSize != 151936 {
			t.Fatalf("%q: Config = %+v, want the parsed hidden/layers/vocab", mt, cfg)
		}
		if cfg.VisionConfig == nil || cfg.VisionConfig.PatchSize != 14 {
			t.Fatalf("%q: VisionConfig = %+v, want the parsed ViT tower noted", mt, cfg.VisionConfig)
		}

		if _, err := ac.Arch(); err == nil {
			t.Fatalf("%q: Arch: expected a clean forward refusal, got a resolved architecture", mt)
		} else if !core.Contains(err.Error(), mt) || !core.Contains(err.Error(), "not yet implemented") {
			t.Fatalf("%q: Arch refusal %q must name the arch and say the forward is not yet implemented", mt, err.Error())
		}
	}
}
