// SPDX-Licence-Identifier: EUPL-1.2

package gptoss

import (
	"math"
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

// TestConfig_Arch_Good pins the LIFTED boundary (#37, all three rungs landed): the REAL
// openai/gpt-oss-20b config fixture resolves to a full decode Arch — geometry, MoE dims, the
// clamped-SwiGLU declaration (with its limit), the YaRN rope table, AND the mscale²-folded
// attention scale (factor 32 → mscale = 0.1·ln 32 + 1 = 1.34657…; AttnScale = mscale²/√64 —
// the cos·sin double application both fetched references carry, see buildArch's derivation).
func TestConfig_Arch_Good(t *testing.T) {
	data := core.ReadFile(core.PathJoin("testdata", "openai-gpt-oss-20b-config.json"))
	if !data.OK {
		t.Fatal("read openai/gpt-oss-20b fixture")
	}
	cfg, err := ParseConfig(data.Value.([]byte))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	a, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch (all three #37 rungs landed — sinks, biases, attention_factor): %v", err)
	}
	if len(a.Layer) != 24 || a.Experts != 32 || a.TopK != 4 || a.Vocab != 201088 {
		t.Fatalf("Arch geometry = %d layers, %d experts, topK %d, vocab %d — want 24/32/4/201088", len(a.Layer), a.Experts, a.TopK, a.Vocab)
	}
	if a.Activation != "gpt_oss_clamped_swiglu" || a.SwigluLimit != 7 {
		t.Fatalf("Arch activation = %q limit %v, want gpt_oss_clamped_swiglu/7", a.Activation, a.SwigluLimit)
	}
	mscale := float64(yarnAttentionFactor(32))
	want := float32(mscale * mscale / math.Sqrt(64))
	if a.AttnScale != want {
		t.Fatalf("AttnScale = %v, want mscale²/√headDim = %v (mscale %v at factor 32)", a.AttnScale, want, mscale)
	}
	if len(a.RopeFreqs) != 32 {
		t.Fatalf("RopeFreqs length = %d, want 32 (rotaryDim/2)", len(a.RopeFreqs))
	}
}

// TestConfig_Arch_NonYarnScale_Good is the attention-scale REGRESSION: a config without YaRN rope
// keeps the plain 1/√headDim — the mscale fold engages ONLY under rope_type "yarn" with factor>1,
// so no other arch's (and no non-YaRN gpt_oss variant's) scale resolution changes.
func TestConfig_Arch_NonYarnScale_Good(t *testing.T) {
	cfg := realConfig()
	cfg.RopeScaling = RopeScaling{} // no yarn → no mscale
	a, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch(non-yarn): %v", err)
	}
	if want := float32(1 / math.Sqrt(64)); a.AttnScale != want {
		t.Fatalf("non-YaRN AttnScale = %v, want the plain 1/√headDim = %v", a.AttnScale, want)
	}
}

// TestConfig_Arch_Bad proves Arch still validates the STRUCTURAL geometry: an empty config fails on
// the basic hidden_size/layers/heads guard with a precise error — the lift removed the serving
// refusal, never the validation.
func TestConfig_Arch_Bad(t *testing.T) {
	_, err := (&Config{}).Arch()
	if err == nil {
		t.Fatal("Arch accepted an empty config")
	}
	if !core.Contains(err.Error(), "hidden_size") {
		t.Fatalf("Arch refusal %q for an empty config must name the missing hidden_size/layers/heads", err.Error())
	}
}

// TestConfig_Arch_Ugly proves Arch validates the layer_types SCHEDULE specifically — a config that is
// otherwise fully populated (hidden/heads/experts all valid) but whose layer_types length disagrees with
// num_hidden_layers is rejected with a schedule-specific error, distinct from _Bad's basic-geometry gap.
func TestConfig_Arch_Ugly(t *testing.T) {
	cfg := Config{
		HiddenSize: 2880, NumHiddenLayers: 24, NumAttentionHeads: 64, NumKeyValueHeads: 8, HeadDim: 64,
		VocabSize: 201088, NumLocalExperts: 32, NumExpertsPerTok: 4, IntermediateSize: 2880,
		LayerTypes: []string{"sliding_attention", "full_attention"}, // 2 entries, not 24
	}
	_, err := cfg.Arch()
	if err == nil || !core.Contains(err.Error(), "layer_types length 2 != num_hidden_layers 24") {
		t.Fatalf("Arch refusal %v must name the layer_types/num_hidden_layers length mismatch", err)
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{HiddenSize: 2880}
	cfg.InferFromWeights(nil)
	if cfg.HiddenSize != 2880 {
		t.Fatalf("InferFromWeights changed config: %+v", cfg)
	}
}

// TestConfig_InferFromWeights_Bad proves the no-op resolves nothing: an empty config still fails
// Arch's structural validation after InferFromWeights (nothing was inferred into it).
func TestConfig_InferFromWeights_Bad(t *testing.T) {
	cfg := Config{}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("Arch must still reject an empty config after InferFromWeights (the no-op infers nothing)")
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
