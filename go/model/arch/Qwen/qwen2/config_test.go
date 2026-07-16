// SPDX-Licence-Identifier: EUPL-1.2

package qwen2

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// TestConfig_ParseConfig_Good parses the unmodified config from Qwen/Qwen2-0.5B.
// Source: https://huggingface.co/Qwen/Qwen2-0.5B/blob/main/config.json
func TestConfig_ParseConfig_Good(t *testing.T) {
	data := core.ReadFile(core.PathJoin("testdata", "qwen-qwen2-0.5b-config.json"))
	if !data.OK {
		t.Fatal("read Qwen2 fixture")
	}
	cfg, err := ParseConfig(data.Value.([]byte))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if cfg.ModelType != "qwen2" || cfg.HiddenSize != 896 || cfg.NumHiddenLayers != 24 || cfg.NumAttentionHeads != 14 || cfg.NumKeyValueHeads != 2 {
		t.Fatalf("parsed Qwen2 geometry = %+v", cfg)
	}
}

// TestConfig_Arch_Good parses the unmodified Qwen2ForCausalLM config from Qwen/Qwen2.5-0.5B
// and pins the resolved Arch dims.
// Source: https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json
func TestConfig_Arch_Good(t *testing.T) {
	data := core.ReadFile(core.PathJoin("testdata", "qwen-qwen2.5-0.5b-config.json"))
	if !data.OK {
		t.Fatal("read Qwen2.5 fixture")
	}
	cfg, err := ParseConfig(data.Value.([]byte))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Hidden != 896 || arch.FF != 4864 || arch.Vocab != 151936 || arch.RopeBase != 1_000_000 || len(arch.Layer) != 24 {
		t.Fatalf("Qwen2.5 Arch = %+v", arch)
	}
	if arch.AttnScale != float32(1/math.Sqrt(64)) || arch.HeadDim != 64 {
		t.Fatalf("Qwen2.5 attention = head_dim %d scale %g", arch.HeadDim, arch.AttnScale)
	}
}

func TestConfig_ParseConfig_Bad(t *testing.T) {
	if _, err := ParseConfig([]byte(`{"model_type":`)); err == nil {
		t.Fatal("ParseConfig accepted malformed JSON")
	}
}

// TestConfig_ParseConfig_Ugly proves ParseConfig never validates geometry — a
// syntactically valid document with no dimensions parses fine; the rejection
// only surfaces later, at Arch. Distinct from _Bad's syntax error.
func TestConfig_ParseConfig_Ugly(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"model_type":"qwen2"}`))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("Arch accepted empty geometry")
	}
}

func TestConfig_Arch_Bad(t *testing.T) {
	if _, err := (&Config{}).Arch(); err == nil {
		t.Fatal("empty config accepted")
	}
}

// TestConfig_Arch_Ugly pins the head_dim-absent/hidden-indivisible edge —
// distinct from _Bad's totally-empty config.
func TestConfig_Arch_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 9, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 32}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("indivisible hidden size (no head_dim) accepted")
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{NumHiddenLayers: 1, NumAttentionHeads: 2}
	cfg.InferFromWeights(map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": {Shape: []int{16, 8}},
		"model.embed_tokens.weight":              {Shape: []int{32, 8}},
	})
	if cfg.HeadDim != 8 || cfg.VocabSize != 32 {
		t.Fatalf("inferred head_dim/vocab = %d/%d", cfg.HeadDim, cfg.VocabSize)
	}
}

func TestConfig_InferFromWeights_Bad(t *testing.T) {
	cfg := Config{}
	cfg.InferFromWeights(nil)
	if cfg.HeadDim != 0 || cfg.VocabSize != 0 {
		t.Fatalf("empty weights inferred geometry: %+v", cfg)
	}
}

// TestConfig_InferFromWeights_Ugly proves an indivisible q_proj row count
// leaves HeadDim un-inferred rather than a garbage value — distinct from
// _Bad's absent-weights case.
func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{NumHiddenLayers: 1, NumAttentionHeads: 3}
	cfg.InferFromWeights(map[string]safetensors.Tensor{"model.layers.0.self_attn.q_proj.weight": {Shape: []int{8, 8}}})
	if cfg.HeadDim != 0 {
		t.Fatalf("indivisible q_proj inferred head_dim %d", cfg.HeadDim)
	}
}
