// SPDX-Licence-Identifier: EUPL-1.2

package gptoss

import (
	"testing"

	core "dappco.re/go"
)

// TestConfig_GptOss20B_Good parses the unmodified config from openai/gpt-oss-20b.
// Source: https://huggingface.co/openai/gpt-oss-20b/resolve/main/config.json
func TestConfig_GptOss20B_Good(t *testing.T) {
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

func TestConfig_Bad(t *testing.T) {
	if _, err := ParseConfig([]byte(`{"model_type":`)); err == nil {
		t.Fatal("ParseConfig accepted malformed JSON")
	}
}
