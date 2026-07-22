// SPDX-Licence-Identifier: EUPL-1.2

package olmoe

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// Fixture source: https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/main/config.json
func TestConfig_Arch_Good(t *testing.T) {
	data, err := coreio.Local.Read(core.PathJoin("testdata", "allenai-olmoe-1b-7b-0924-config.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cfg Config
	if r := core.JSONUnmarshal([]byte(data), &cfg); !r.OK {
		t.Fatalf("parse fixture: %s", r.Error())
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Hidden != 2048 || arch.Heads != 16 || arch.KVHeads != 16 || arch.HeadDim != 128 {
		t.Fatalf("attention geometry = %+v", arch)
	}
	if arch.Experts != 64 || arch.TopK != 8 || arch.ExpertFF != 1024 || arch.NormaliseMoETopK || arch.SharedExperts != 0 {
		t.Fatalf("MoE declaration = experts %d top-k %d expert FF %d normalise %v shared %d", arch.Experts, arch.TopK, arch.ExpertFF, arch.NormaliseMoETopK, arch.SharedExperts)
	}
	if arch.MoEGating != model.MoEGatingSoftmax || len(arch.Layer) != 16 || !arch.HasMoE() {
		t.Fatalf("architecture receipt = gating %q layers %d MoE %v", arch.MoEGating, len(arch.Layer), arch.HasMoE())
	}
	// #63: hidden_act must reach Arch.Activation so the MoE expert combine can select SiLU
	// (engine/metal/projector.go's ffnUsesSiLU) instead of gemma4's GELU default.
	if arch.Activation != "silu" {
		t.Fatalf("Activation = %q, want silu (real OLMoE checkpoints declare hidden_act: silu)", arch.Activation)
	}
}

func TestConfig_Arch_Bad(t *testing.T) {
	if _, err := (Config{}).Arch(); err == nil {
		t.Fatal("empty config accepted")
	}
}

func TestConfig_Arch_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 8, IntermediateSize: 12, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, NumExperts: 2, NumExpertsPerTok: 3, VocabSize: 32}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("top-k greater than expert count accepted")
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(map[string]safetensors.Tensor{"ignored": {Shape: []int{1}}})
	if cfg.HiddenSize != 8 {
		t.Fatalf("InferFromWeights changed config: %+v", cfg)
	}
}

func TestConfig_InferFromWeights_Bad(t *testing.T) {
	cfg := Config{}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("empty config became valid after InferFromWeights")
	}
}

func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 9, IntermediateSize: 12, NumHiddenLayers: 1, NumAttentionHeads: 2, NumExperts: 2, NumExpertsPerTok: 1, VocabSize: 32}
	cfg.InferFromWeights(map[string]safetensors.Tensor{})
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("indivisible hidden size became valid after InferFromWeights")
	}
}
