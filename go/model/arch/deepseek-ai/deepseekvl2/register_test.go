// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// testConfigJSON mirrors the field names/values confirmed live from
// https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/config.json (trimmed to the
// fields this package reads). The real file duplicates hidden_size/num_hidden_layers/vocab_size
// under a nested "language_config" key too, but that nested copy is vestigial — DeepseekOCRConfig
// (the checkpoint's own custom_code config class) is instantiated straight from the TOP-LEVEL
// fields (verified by constructing it directly and reading back cfg.hidden_size etc.), so this
// fixture carries both: language_config for the (still-supported) recognition-only reporting
// path, and top-level fields for the fields Config/weights.go/decoder.go actually load geometry
// from.
const testConfigJSON = `{"model_type":"deepseek_vl_v2","hidden_size":1280,"intermediate_size":6848,"moe_intermediate_size":896,"num_hidden_layers":12,"num_attention_heads":10,"num_key_value_heads":10,"vocab_size":129280,"n_routed_experts":64,"n_shared_experts":2,"num_experts_per_tok":6,"first_k_dense_replace":1,"use_mla":false,"max_position_embeddings":8192,"bos_token_id":0,"eos_token_id":1,"language_config":{"hidden_size":1280,"num_hidden_layers":12,"vocab_size":129280},"vision_config":{"model_type":"vision","model_name":"deeplip_b_l","image_size":1024}}`

// TestDeepseekvl2Registered_Good pins the deepseek_vl_v2 "recognised, and OCR works through its
// own verb" contract: model.LookupArch succeeds (a user pointing lem at a DeepSeek-OCR checkpoint
// gets direction, not "unknown model architecture"), Parse succeeds far enough to report what the
// arch is, and the Arch refusal seam names the arch and redirects to `lem ocr` — the recognised-
// with-its-own-verb discipline for a non-hybrid vision-language arch.
func TestDeepseekvl2Registered_Good(t *testing.T) {
	spec, ok := model.LookupArch("deepseek_vl_v2")
	if !ok {
		t.Fatal("model_type deepseek_vl_v2 not registered — an OCR arch must be recognised, not fall to \"unknown model architecture\"")
	}
	if spec.Parse == nil {
		t.Fatal("deepseek_vl_v2 registered without a Parse hook")
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
	} else if !core.Contains(err.Error(), "deepseek_vl_v2") || !core.Contains(err.Error(), "not a decoder-only causal-LM") {
		t.Fatalf("Arch refusal %q must name deepseek_vl_v2 and say it is not a decoder-only causal-LM", err.Error())
	}
}
