// SPDX-Licence-Identifier: EUPL-1.2

package phi

import (
	"testing"

	core "dappco.re/go"
)

// Fixtures are verbatim configs from:
// https://huggingface.co/microsoft/phi-2/blob/main/config.json
// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/config.json
// https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/config.json
func parseFixture(t *testing.T, name string) *Config {
	t.Helper()
	r := core.ReadFile(core.PathJoin("testdata", name))
	if !r.OK {
		t.Fatalf("read %s", name)
	}
	var cfg Config
	b, ok := r.Value.([]byte)
	if !ok {
		t.Fatalf("read %s returned a non-byte value", name)
	}
	if r := core.JSONUnmarshal(b, &cfg); !r.OK {
		t.Fatalf("parse %s", name)
	}
	return &cfg
}

// TestConfig_Arch_Good pins the Phi-2 config (dense, LayerNorm, tied head).
func TestConfig_Arch_Good(t *testing.T) {
	a, err := parseFixture(t, "microsoft-phi-2-config.json").Arch()
	if err != nil {
		t.Fatal(err)
	}
	if a.Hidden != 2560 || a.HeadDim != 80 || a.RotaryDim != 32 || a.FF != 10240 || a.TieWordEmbeddings == nil || *a.TieWordEmbeddings {
		t.Fatalf("Phi-2 arch = %+v", a)
	}
}

func TestConfigArchPhi3_Good(t *testing.T) {
	a, err := parseFixture(t, "microsoft-phi-3-mini-4k-instruct-config.json").Arch()
	if err != nil {
		t.Fatal(err)
	}
	if a.Hidden != 3072 || a.HeadDim != 96 || a.RotaryDim != 96 || a.SlidingWindow != 2047 || a.Eps != 1e-5 {
		t.Fatalf("Phi-3 arch = %+v", a)
	}
}

func TestConfigArchPhi3LongRoPE_Good(t *testing.T) {
	a, err := parseFixture(t, "microsoft-phi-3-mini-128k-instruct-config.json").Arch()
	if err != nil {
		t.Fatal(err)
	}
	if len(a.RopeFreqs) != 48 || len(a.RopeShortFreqs) != 48 || a.RopeOriginalContext != 4096 {
		t.Fatalf("Phi-3 LongRoPE declaration = long %d short %d original %d", len(a.RopeFreqs), len(a.RopeShortFreqs), a.RopeOriginalContext)
	}
	if a.RopeFreqs[0] == a.RopeShortFreqs[0] {
		t.Fatal("long and short LongRoPE frequencies unexpectedly match")
	}
}

func TestConfigArchPhi3LongRoPEAlias_Good(t *testing.T) {
	cfg := parseFixture(t, "microsoft-phi-3-mini-128k-instruct-config.json")
	cfg.RopeScaling.RopeType, cfg.RopeScaling.Type = "longrope", ""
	if _, err := cfg.Arch(); err != nil {
		t.Fatalf("rope_type longrope alias: %v", err)
	}
}

func TestConfigArchTiedHead_Good(t *testing.T) {
	tied := true
	cfg := Config{ModelType: "phi3", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 12, RMSNormEps: 1e-5, TieWordEmbeddings: &tied}
	a, err := cfg.Arch()
	if err != nil || a.TieWordEmbeddings == nil || !*a.TieWordEmbeddings {
		t.Fatalf("tied head = %+v, %v", a.TieWordEmbeddings, err)
	}
}

func TestConfig_Arch_Bad(t *testing.T) {
	if _, err := (&Config{ModelType: "phi", HiddenSize: 3, NumAttentionHeads: 2, IntermediateSize: 4, NumHiddenLayers: 1, VocabSize: 8}).Arch(); err == nil {
		t.Fatal("Arch accepted indivisible head geometry")
	}
}

func TestConfig_Arch_Ugly(t *testing.T) {
	if _, err := (&Config{}).Arch(); err == nil {
		t.Fatal("Arch accepted an empty config")
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(nil)
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

// TestConfig_InferFromWeights_Ugly proves the no-op does not paper over the
// Phi-3 fused-qkv guard (equal query/kv head counts) — distinct from _Bad's
// all-zero-fields rejection.
func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{ModelType: "phi3", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 4, NumKeyValueHeads: 2, VocabSize: 12, RMSNormEps: 1e-5}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("Phi-3 unequal query/kv head counts became valid after InferFromWeights")
	}
}
