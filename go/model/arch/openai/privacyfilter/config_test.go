// SPDX-Licence-Identifier: EUPL-1.2

package privacyfilter

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// TestConfig_ParseConfig_Good parses the unmodified config from openai/privacy-filter.
// Source: https://huggingface.co/openai/privacy-filter/resolve/main/config.json
func TestConfig_ParseConfig_Good(t *testing.T) {
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

func TestConfig_ParseConfig_Bad(t *testing.T) {
	if _, err := ParseConfig([]byte(`{"model_type":`)); err == nil {
		t.Fatal("ParseConfig accepted malformed JSON")
	}
}

// TestConfig_ParseConfig_Ugly proves ParseConfig never validates geometry —
// a syntactically valid but semantically empty document parses fine.
// Distinct from _Bad's syntax error.
func TestConfig_ParseConfig_Ugly(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{}`))
	if err != nil {
		t.Fatalf("ParseConfig must accept a syntactically valid but semantically empty document: %v", err)
	}
	if cfg.ModelType != "" {
		t.Fatalf("empty document produced a non-empty ModelType: %q", cfg.ModelType)
	}
}

// TestConfig_Arch_Good pins the documented "always refuses" behaviour for a
// realistic, fully-populated config: the refusal echoes the config's ACTUAL
// class/expert counts (proving it doesn't fabricate a generic message).
func TestConfig_Arch_Good(t *testing.T) {
	cfg := Config{NumLocalExperts: 128, NumExpertsPerTok: 4, ID2Label: map[string]string{"0": "O"}}
	_, err := cfg.Arch()
	if err == nil {
		t.Fatal("Arch: expected a clean not-yet-supported refusal, got a resolved architecture")
	}
	if !core.Contains(err.Error(), "128") || !core.Contains(err.Error(), "1-class") {
		t.Fatalf("Arch refusal %q must echo the config's actual class/expert counts", err.Error())
	}
}

func TestConfig_Arch_Bad(t *testing.T) {
	_, err := (&Config{}).Arch()
	if err == nil {
		t.Fatal("Arch accepted an empty config")
	}
	if !core.Contains(err.Error(), "PII token-classification model") {
		t.Fatalf("Arch refusal %q must still explain the arch even for an empty config", err.Error())
	}
}

// TestConfig_Arch_Ugly proves Arch performs NO validation at all — even
// nonsensical negative expert counts are echoed verbatim in the refusal
// (there is only ever one refusal shape, unconditionally) — distinct from
// _Bad's zero-value case.
func TestConfig_Arch_Ugly(t *testing.T) {
	cfg := Config{NumLocalExperts: -1, NumExpertsPerTok: -1}
	_, err := cfg.Arch()
	if err == nil || !core.Contains(err.Error(), "-1") {
		t.Fatalf("Arch refusal %v must echo even nonsensical negative counts verbatim", err)
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{HiddenSize: 640}
	cfg.InferFromWeights(nil)
	if cfg.HiddenSize != 640 {
		t.Fatalf("InferFromWeights changed config: %+v", cfg)
	}
}

// TestConfig_InferFromWeights_Bad proves the no-op does not make Arch
// succeed — Arch always refuses regardless.
func TestConfig_InferFromWeights_Bad(t *testing.T) {
	cfg := Config{}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("Arch must still refuse after InferFromWeights")
	}
}

// TestConfig_InferFromWeights_Ugly proves the no-op stays inert even given a
// malformed/weird weights map entry — distinct from _Good's nil-weights case.
func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 640}
	cfg.InferFromWeights(map[string]safetensors.Tensor{"weird": {}})
	if cfg.HiddenSize != 640 {
		t.Fatalf("InferFromWeights changed config on a malformed weights map: %+v", cfg)
	}
}
