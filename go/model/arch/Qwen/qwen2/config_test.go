// SPDX-Licence-Identifier: EUPL-1.2

package qwen2

import (
	"math"
	"testing"

	core "dappco.re/go"
)

// TestConfig_Qwen2_Good parses the unmodified config from Qwen/Qwen2-0.5B.
// Source: https://huggingface.co/Qwen/Qwen2-0.5B/blob/main/config.json
func TestConfig_Qwen2_Good(t *testing.T) {
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

// TestConfig_Qwen25_Good parses the unmodified Qwen2ForCausalLM config from Qwen/Qwen2.5-0.5B.
// Source: https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json
func TestConfig_Qwen25_Good(t *testing.T) {
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

func TestConfig_Bad(t *testing.T) {
	if _, err := ParseConfig([]byte(`{"model_type":`)); err == nil {
		t.Fatal("ParseConfig accepted malformed JSON")
	}
}

func TestConfig_Ugly(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"model_type":"qwen2"}`))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("Arch accepted empty geometry")
	}
}
