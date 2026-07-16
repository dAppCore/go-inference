// SPDX-Licence-Identifier: EUPL-1.2

package gptoss

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// TestConfig_ParseConfig_Good parses the unmodified config from openai/gpt-oss-20b.
// Source: https://huggingface.co/openai/gpt-oss-20b/resolve/main/config.json
func TestConfig_ParseConfig_Good(t *testing.T) {
	data := core.ReadFile(core.PathJoin("testdata", "openai-gpt-oss-20b-config.json"))
	if !data.OK {
		t.Fatal("read openai/gpt-oss-20b fixture")
	}
	cfg, err := ParseConfig(data.Value.([]byte))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if cfg.ModelType != "gpt_oss" || cfg.HiddenSize != 2880 || cfg.NumHiddenLayers != 24 {
		t.Fatalf("parsed gpt_oss geometry = %+v", cfg)
	}
	if cfg.NumAttentionHeads != 64 || cfg.NumKeyValueHeads != 8 || cfg.HeadDim != 64 {
		t.Fatalf("parsed gpt_oss attention geometry = heads %d, kv_heads %d, head_dim %d", cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim)
	}
	if cfg.NumLocalExperts != 32 || cfg.NumExpertsPerTok != 4 || cfg.ExpertsPerToken != 4 {
		t.Fatalf("parsed gpt_oss MoE geometry = local_experts %d, per_tok %d, experts_per_token %d", cfg.NumLocalExperts, cfg.NumExpertsPerTok, cfg.ExpertsPerToken)
	}
	if cfg.SlidingWindow != 128 || len(cfg.LayerTypes) != 24 {
		t.Fatalf("parsed gpt_oss layer geometry = sliding_window %d, layer_types %d entries", cfg.SlidingWindow, len(cfg.LayerTypes))
	}
	if cfg.RopeTheta != 150000 || cfg.RopeScaling.RopeType != "yarn" || cfg.RopeScaling.Factor != 32 || cfg.RopeScaling.OriginalMaxPositionEmbeddings != 4096 {
		t.Fatalf("parsed gpt_oss YaRN rope = theta %v, scaling %+v", cfg.RopeTheta, cfg.RopeScaling)
	}
	if cfg.VocabSize != 201088 || cfg.MaxPositionEmbeddings != 131072 {
		t.Fatalf("parsed gpt_oss vocab/context = vocab %d, max_position_embeddings %d", cfg.VocabSize, cfg.MaxPositionEmbeddings)
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
// layer/expert/vocab counts (proving it doesn't fabricate a generic message).
func TestConfig_Arch_Good(t *testing.T) {
	cfg := Config{NumHiddenLayers: 24, NumLocalExperts: 32, NumExpertsPerTok: 4, VocabSize: 201088}
	_, err := cfg.Arch()
	if err == nil {
		t.Fatal("Arch: expected a clean not-yet-validated refusal, got a resolved architecture")
	}
	if !core.Contains(err.Error(), "24") || !core.Contains(err.Error(), "32") || !core.Contains(err.Error(), "201088") {
		t.Fatalf("Arch refusal %q must echo the config's actual layer/expert/vocab counts", err.Error())
	}
}

func TestConfig_Arch_Bad(t *testing.T) {
	_, err := (&Config{}).Arch()
	if err == nil {
		t.Fatal("Arch accepted an empty config")
	}
	if !core.Contains(err.Error(), "gpt_oss is a generative MoE causal-LM") {
		t.Fatalf("Arch refusal %q must still explain the arch even for an empty config", err.Error())
	}
}

// TestConfig_Arch_Ugly proves Arch performs NO validation at all — even
// nonsensical negative counts are echoed verbatim in the refusal (there is
// only ever one refusal shape, unconditionally) — distinct from _Bad's
// zero-value case.
func TestConfig_Arch_Ugly(t *testing.T) {
	cfg := Config{NumHiddenLayers: -1, NumLocalExperts: -1, NumExpertsPerTok: -1, VocabSize: -1}
	_, err := cfg.Arch()
	if err == nil || !core.Contains(err.Error(), "-1") {
		t.Fatalf("Arch refusal %v must echo even nonsensical negative counts verbatim", err)
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{HiddenSize: 2880}
	cfg.InferFromWeights(nil)
	if cfg.HiddenSize != 2880 {
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
	cfg := Config{HiddenSize: 2880}
	cfg.InferFromWeights(map[string]safetensors.Tensor{"weird": {}})
	if cfg.HiddenSize != 2880 {
		t.Fatalf("InferFromWeights changed config on a malformed weights map: %+v", cfg)
	}
}
