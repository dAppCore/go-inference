// SPDX-Licence-Identifier: EUPL-1.2

package deepseek

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// Fixture source: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json
func TestConfig_Validate_Good(t *testing.T) {
	data, err := coreio.Local.Read(core.PathJoin("testdata", "deepseek-ai-deepseek-v2-lite-config.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cfg Config
	if r := core.JSONUnmarshal([]byte(data), &cfg); !r.OK {
		t.Fatalf("parse fixture: %s", r.Error())
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if cfg.QHeadDim() != 192 || cfg.KVHeadDim() != 192 || cfg.ValueHeadDim != 128 {
		t.Fatalf("MLA heads = q %d kv %d v %d", cfg.QHeadDim(), cfg.KVHeadDim(), cfg.ValueHeadDim)
	}
	if cfg.KVLoRARank != 512 || cfg.NumRoutedExperts != 64 || cfg.NumExpertsPerTok != 6 {
		t.Fatalf("MLA/MoE geometry = %+v", cfg)
	}
}

func TestConfig_Validate_Bad(t *testing.T) {
	if err := (Config{}).Validate(); err == nil {
		t.Fatal("empty config accepted")
	}
}

func TestConfig_ValidateSparseGeometry_Bad(t *testing.T) {
	cfg := Config{HiddenSize: 8, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 32, KVLoRARank: 4, QKNoPEHeadDim: 2, QKRoPEHeadDim: 2, ValueHeadDim: 2}
	if err := cfg.Validate(); err == nil {
		t.Fatal("missing sparse-expert geometry accepted")
	}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("invalid MLA config accepted by Arch")
	}
}

func TestConfig_Arch_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 8, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 32, KVLoRARank: 4, QKNoPEHeadDim: 2, QKRoPEHeadDim: 2, ValueHeadDim: 2, NumRoutedExperts: 2, NumExpertsPerTok: 1, MoEIntermediateSize: 4}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("MLA config incorrectly lowered to standard attention")
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(map[string]safetensors.Tensor{"ignored": {Shape: []int{1}}})
	if cfg.HiddenSize != 8 {
		t.Fatalf("InferFromWeights changed config: %+v", cfg)
	}
}
