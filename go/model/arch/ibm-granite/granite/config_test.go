// SPDX-Licence-Identifier: EUPL-1.2

package granite

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

func TestConfig_Good(t *testing.T) {
	cfg := Config{ModelType: "granite", HiddenSize: 2048}
	if cfg.ModelType != "granite" || cfg.HiddenSize != 2048 {
		t.Fatalf("Config = %+v", cfg)
	}
}

func TestConfig_Bad(t *testing.T) {
	cfg := Config{ModelType: "granitemoe"}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("Config with MoE model type accepted")
	}
}

func TestConfig_Ugly(t *testing.T) {
	var cfg Config
	if cfg.ModelType != "" || cfg.HiddenSize != 0 {
		t.Fatalf("zero Config = %+v", cfg)
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(nil)
	if cfg.HiddenSize != 8 {
		t.Fatalf("InferFromWeights changed declared hidden size: %+v", cfg)
	}
}

func TestConfig_InferFromWeights_Bad(t *testing.T) {
	cfg := Config{}
	cfg.InferFromWeights(nil)
	if cfg.HiddenSize != 0 {
		t.Fatalf("InferFromWeights invented hidden size: %+v", cfg)
	}
}

func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{NumHiddenLayers: 1}
	cfg.InferFromWeights(map[string]safetensors.Tensor{"malformed": {Shape: []int{-1}}})
	if cfg.NumHiddenLayers != 1 {
		t.Fatalf("InferFromWeights changed declared layer count: %+v", cfg)
	}
}

// TestConfig_Arch_Good parses the public ibm-granite/granite-3.3-2b-base
// config fixture and pins all four Granite scalar declarations.
func TestConfig_Arch_Good(t *testing.T) {
	data := core.ReadFile(core.PathJoin("testdata", "ibm-granite-granite-3.3-2b-base-config.json"))
	if !data.OK {
		t.Fatal("read Granite config fixture")
	}
	r := ParseConfig(data.Value.([]byte))
	if !r.OK {
		t.Fatalf("ParseConfig: %v", r.Error())
	}
	cfg := r.Value.(*Config)
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Hidden != 2048 || arch.FF != 8192 || arch.Heads != 32 || arch.KVHeads != 8 || len(arch.Layer) != 40 {
		t.Fatalf("Granite geometry = %+v", arch)
	}
	if arch.LogitsScaling != 8 || arch.ResidualMultiplier != 0.22 || arch.EmbedScale != 12 || arch.AttnScale != 0.015625 {
		t.Fatalf("Granite scalars = logits %g residual %g embedding %g attention %g", arch.LogitsScaling, arch.ResidualMultiplier, arch.EmbedScale, arch.AttnScale)
	}
}

func TestConfig_Arch_Bad(t *testing.T) {
	cfg := Config{HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, VocabSize: 32, LogitsScaling: -1, ResidualMultiplier: 0.22, EmbeddingMultiplier: 12, AttentionMultiplier: 0.125}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("negative logits_scaling accepted")
	}
}

func TestConfig_Arch_Ugly(t *testing.T) {
	cfg := Config{}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("empty config accepted")
	}
}

func TestConfig_ParseConfig_Good(t *testing.T) {
	r := ParseConfig([]byte(`{"model_type":"granite"}`))
	if !r.OK || r.Value.(*Config).ModelType != "granite" {
		t.Fatalf("ParseConfig = %+v", r)
	}
}

func TestConfig_ParseConfig_Bad(t *testing.T) {
	if r := ParseConfig([]byte("not json")); r.OK {
		t.Fatal("malformed config accepted")
	}
}

func TestConfig_ParseConfig_Ugly(t *testing.T) {
	r := ParseConfig([]byte(`{}`))
	if !r.OK || r.Value.(*Config).ModelType != "" {
		t.Fatalf("empty JSON parse = %+v", r)
	}
}
