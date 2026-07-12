// SPDX-Licence-Identifier: EUPL-1.2

package starcoder2

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestConfig_ParseConfig_Good parses the unmodified bigcode/starcoder2-3b config.
// Source: https://huggingface.co/bigcode/starcoder2-3b/blob/main/config.json
func TestConfig_ParseConfig_Good(t *testing.T) {
	data := core.ReadFile(core.PathJoin("testdata", "bigcode-starcoder2-3b-config.json"))
	if !data.OK {
		t.Fatal("read StarCoder2 fixture")
	}
	cfg, err := ParseConfig(data.Value.([]byte))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	a, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if a.Hidden != 3072 || a.Heads != 24 || a.KVHeads != 2 || a.HeadDim != 128 || a.FF != 12288 || a.Vocab != 49152 || len(a.Layer) != 30 {
		t.Fatalf("StarCoder2 geometry = %+v", a)
	}
	if a.SlidingWindow != 4096 || a.RotaryDim != 128 || a.RopeBase != float32(999999.4420358813) || a.Activation != "gelu_pytorch_tanh" {
		t.Fatalf("StarCoder2 attention = %+v", a)
	}
}

// TestConfig_ParseConfig_Bad rejects malformed config JSON.
func TestConfig_ParseConfig_Bad(t *testing.T) {
	if _, err := ParseConfig([]byte(`{"model_type":`)); err == nil {
		t.Fatal("ParseConfig accepted malformed JSON")
	}
}

// TestConfig_ParseConfig_Ugly accepts valid JSON while leaving geometry validation to Arch.
func TestConfig_ParseConfig_Ugly(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"model_type":"starcoder2"}`))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("Arch accepted empty geometry")
	}
}

func TestConfig_Config_InferFromWeights_Good(t *testing.T) {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(nil)
	if cfg.HiddenSize != 8 {
		t.Fatalf("InferFromWeights changed declared geometry: %+v", cfg)
	}
}

func TestConfig_Config_InferFromWeights_Bad(t *testing.T) {
	cfg := Config{}
	cfg.InferFromWeights(nil)
	if cfg.HiddenSize != 0 {
		t.Fatalf("InferFromWeights invented missing geometry: %+v", cfg)
	}
}

func TestConfig_Config_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: -1}
	cfg.InferFromWeights(nil)
	if cfg.HiddenSize != -1 {
		t.Fatalf("InferFromWeights rewrote malformed geometry: %+v", cfg)
	}
}

func TestConfig_Config_Arch_Good(t *testing.T) {
	cfg := Config{ModelType: "starcoder2", HiddenSize: 8, IntermediateSize: 16, MaxPositionEmbeddings: 16, NumAttentionHeads: 2, NumHiddenLayers: 1, NumKeyValueHeads: 1, VocabSize: 32, SlidingWindow: 4}
	a, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if a.Layer[0].Attention != model.SlidingAttention || a.SlidingWindow != 4 {
		t.Fatalf("Arch sliding declaration = %+v", a)
	}
}

func TestConfig_Config_Arch_Bad(t *testing.T) {
	if _, err := (&Config{}).Arch(); err == nil {
		t.Fatal("Arch accepted empty geometry")
	}
}

func TestConfig_Config_Arch_Ugly(t *testing.T) {
	cfg := Config{ModelType: "codegen", Hidden: 8, Heads: 2, Layers: 1, Positions: 8, VocabSize: 32, RotaryDimension: 5}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("Arch accepted odd CodeGen rotary geometry")
	}
}

// TestParseConfig_CodeGen_Good proves CodeGen maps only through its real GPT-J-like config fields.
// Source: https://huggingface.co/Salesforce/codegen-350M-mono/blob/main/config.json
func TestParseConfig_CodeGen_Good(t *testing.T) {
	data := core.ReadFile(core.PathJoin("testdata", "salesforce-codegen-350m-mono-config.json"))
	if !data.OK {
		t.Fatal("read CodeGen fixture")
	}
	cfg, err := ParseConfig(data.Value.([]byte))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	a, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if a.Hidden != 1024 || a.Heads != 16 || a.KVHeads != 16 || a.HeadDim != 64 || a.FF != 4096 || a.Vocab != 51200 || len(a.Layer) != 20 {
		t.Fatalf("CodeGen geometry = %+v", a)
	}
	if a.RotaryDim != 32 || !a.ParallelResidual || a.LearnedAbsolutePositions || a.Activation != "gelu_new" {
		t.Fatalf("CodeGen declaration = %+v", a)
	}
}
