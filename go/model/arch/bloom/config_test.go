// SPDX-Licence-Identifier: EUPL-1.2

package bloom

import (
	"testing"

	core "dappco.re/go"
)

// Fixture source: https://huggingface.co/bigscience/bloom-560m/blob/main/config.json
func TestConfig_Arch_Good(t *testing.T) {
	var cfg Config
	if r := core.JSONUnmarshal([]byte(bloom560MConfig), &cfg); !r.OK {
		t.Fatalf("parse fixture: %v", r.Value)
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Hidden != 1024 || arch.Heads != 16 || arch.FF != 4096 || !arch.ALiBi || len(arch.Layer) != 24 {
		t.Fatalf("BLOOM-560m arch = %+v", arch)
	}
}

func TestConfig_Arch_Bad(t *testing.T) {
	if _, err := (Config{}).Arch(); err == nil {
		t.Fatal("empty config accepted")
	}
}

// TestConfig_Arch_Ugly proves an explicit n_inner overrides the 4x default FF
// — distinct from _Bad's totally-empty config.
func TestConfig_Arch_Ugly(t *testing.T) {
	ff := 12288
	cfg := Config{HiddenSize: 1024, NumHiddenLayers: 1, NumAttentionHeads: 16, VocabSize: 8, IntermediateSize: &ff}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.FF != 12288 {
		t.Fatalf("FF = %d, want the explicit n_inner override 12288 (not the 4x default 4096)", arch.FF)
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{HiddenSize: 1024}
	cfg.InferFromWeights(nil)
	if cfg.HiddenSize != 1024 {
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

// TestConfig_InferFromWeights_Ugly proves the no-op does not paper over the
// indivisible-hidden-size guard — distinct from _Bad's all-zero rejection.
func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 9, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 8}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("indivisible hidden size became valid after InferFromWeights")
	}
}

const bloom560MConfig = `{"layer_norm_epsilon":1e-5,"model_type":"bloom","n_embed":1024,"n_inner":null,"n_layer":24,"num_attention_heads":16,"offset_alibi":100,"vocab_size":250880}`
