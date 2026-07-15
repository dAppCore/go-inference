// SPDX-Licence-Identifier: EUPL-1.2

package privacyfilter

import (
	"testing"

	core "dappco.re/go"
)

// TestConfig_PrivacyFilter_Good parses the unmodified config from openai/privacy-filter.
// Source: https://huggingface.co/openai/privacy-filter/resolve/main/config.json
func TestConfig_PrivacyFilter_Good(t *testing.T) {
	data := core.ReadFile(core.PathJoin("testdata", "openai-privacy-filter-config.json"))
	if !data.OK {
		t.Fatal("read openai/privacy-filter fixture")
	}
	cfg, err := ParseConfig(data.Value.([]byte))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if cfg.ModelType != "openai_privacy_filter" || cfg.HiddenSize != 640 || cfg.NumHiddenLayers != 8 ||
		cfg.NumAttentionHeads != 14 || cfg.NumKeyValueHeads != 2 {
		t.Fatalf("parsed openai_privacy_filter geometry = %+v", cfg)
	}
	if cfg.NumLocalExperts != 128 || cfg.NumExpertsPerTok != 4 {
		t.Fatalf("parsed openai_privacy_filter MoE geometry = experts %d, per-tok %d", cfg.NumLocalExperts, cfg.NumExpertsPerTok)
	}
	if cfg.SlidingWindow != 128 || cfg.MaxPositionEmbeddings != 131072 {
		t.Fatalf("parsed openai_privacy_filter context geometry = sliding_window %d, max_position_embeddings %d", cfg.SlidingWindow, cfg.MaxPositionEmbeddings)
	}
	if cfg.RopeParameters.RopeType != "yarn" || cfg.RopeParameters.RopeTheta != 150000 || cfg.RopeParameters.Factor != 32 {
		t.Fatalf("parsed openai_privacy_filter rope_parameters (YaRN) = %+v", cfg.RopeParameters)
	}
	if len(cfg.ID2Label) != 33 {
		t.Fatalf("id2label = %d entries, want 33 (O + BIOES x 8 PII classes)", len(cfg.ID2Label))
	}
	if cfg.ID2Label["0"] != "O" || cfg.ID2Label["29"] != "B-secret" {
		t.Fatalf("id2label BIOES labels = %+v", cfg.ID2Label)
	}
}

func TestConfig_Bad(t *testing.T) {
	if _, err := ParseConfig([]byte(`{"model_type":`)); err == nil {
		t.Fatal("ParseConfig accepted malformed JSON")
	}
}
