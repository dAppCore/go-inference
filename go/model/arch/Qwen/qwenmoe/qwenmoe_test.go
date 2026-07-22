// SPDX-Licence-Identifier: EUPL-1.2

package qwenmoe

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

func readConfigFixture(t *testing.T, name string) Config {
	t.Helper()
	data, err := coreio.Local.Read(core.PathJoin("testdata", name))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cfg Config
	if r := core.JSONUnmarshal([]byte(data), &cfg); !r.OK {
		t.Fatalf("parse fixture: %s", r.Error())
	}
	return cfg
}

// Fixture source: https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blob/main/config.json
func TestQwenmoe_Config_Arch_Good(t *testing.T) {
	cfg := readConfigFixture(t, "Qwen-Qwen1.5-MoE-A2.7B-config.json")
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Hidden != 2048 || arch.Heads != 16 || arch.HeadDim != 128 || arch.Experts != 60 || arch.TopK != 4 || arch.ExpertFF != 1408 {
		t.Fatalf("Qwen2-MoE architecture = %+v", arch)
	}
	if arch.MoEGating != model.MoEGatingSoftmax || arch.NormaliseMoETopK || arch.SharedExperts != 1 || !arch.HasMoE() {
		t.Fatalf("Qwen2-MoE router = score %q normalise %v shared %d MoE %v", arch.MoEGating, arch.NormaliseMoETopK, arch.SharedExperts, arch.HasMoE())
	}
	// #57: shared_expert_intermediate_size (5632) is DISTINCT from moe_intermediate_size (1408, ExpertFF
	// above) on real Qwen1.5-MoE-A2.7B — SharedExpertFF must carry the shared expert's OWN width, not
	// silently equal ExpertFF.
	if arch.SharedExpertFF != 5632 {
		t.Fatalf("Qwen2-MoE SharedExpertFF = %d, want 5632 (shared_expert_intermediate_size, distinct from ExpertFF %d)", arch.SharedExpertFF, arch.ExpertFF)
	}
}

func TestQwenmoe_Config_Arch_Bad(t *testing.T) {
	cfg := Config{}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("empty config accepted")
	}
}

func TestQwenmoe_Config_Arch_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 8, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 16, NumExperts: 2, NumExpertsPerTok: 3, MoEIntermediateSize: 4}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("top-k greater than expert count accepted")
	}
}

func TestQwenmoe_Config_InferFromWeights_Good(t *testing.T) {
	cfg := Config{NumHiddenLayers: 1, NumAttentionHeads: 2}
	cfg.InferFromWeights(map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": {Shape: []int{16, 8}},
		"model.embed_tokens.weight":              {Shape: []int{32, 8}},
	})
	if cfg.HeadDim != 8 || cfg.VocabSize != 32 {
		t.Fatalf("inferred head_dim/vocab = %d/%d", cfg.HeadDim, cfg.VocabSize)
	}
}

func TestQwenmoe_Config_InferFromWeights_Bad(t *testing.T) {
	cfg := Config{}
	cfg.InferFromWeights(nil)
	if cfg.HeadDim != 0 || cfg.VocabSize != 0 {
		t.Fatalf("empty weights inferred geometry: %+v", cfg)
	}
}

func TestQwenmoe_Config_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{NumHiddenLayers: 1, NumAttentionHeads: 3}
	cfg.InferFromWeights(map[string]safetensors.Tensor{"model.layers.0.self_attn.q_proj.weight": {Shape: []int{8, 8}}})
	if cfg.HeadDim != 0 {
		t.Fatalf("indivisible q_proj inferred head_dim %d", cfg.HeadDim)
	}
}

// Fixture source: https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json
func TestQwenmoe_Config_Arch_Qwen3(t *testing.T) {
	cfg := readConfigFixture(t, "Qwen-Qwen3-30B-A3B-config.json")
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Experts != 128 || arch.TopK != 8 || arch.ExpertFF != 768 || !arch.NormaliseMoETopK || arch.SharedExperts != 0 {
		t.Fatalf("Qwen3-MoE router = experts %d top-k %d FF %d normalise %v shared %d", arch.Experts, arch.TopK, arch.ExpertFF, arch.NormaliseMoETopK, arch.SharedExperts)
	}
	// #57: Qwen3-MoE's config.json carries no shared_expert_intermediate_size key at all, so the parsed
	// Config.SharedExpertIntermediateSize (and the derived SharedExpertFF) stays Go's int zero value —
	// consistent with SharedExperts == 0 above (no shared expert to derive a width for).
	if arch.SharedExpertFF != 0 {
		t.Fatalf("Qwen3-MoE SharedExpertFF = %d, want 0 (no shared_expert_intermediate_size in the config)", arch.SharedExpertFF)
	}
}
