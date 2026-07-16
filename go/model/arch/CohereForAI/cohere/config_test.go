// SPDX-Licence-Identifier: EUPL-1.2

package cohere

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// Fixtures are architecture fields from the named public checkpoints:
// https://huggingface.co/CohereLabs/c4ai-command-r-v01/blob/main/config.json
// https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024/blob/main/config.json
func TestConfig_Arch_Good(t *testing.T) {
	checks := []struct {
		file  string
		want  model.AttentionType
		logit float32
		qk    model.QKNormalization
	}{
		{"CohereLabs-c4ai-command-r-v01-config.json", model.GlobalAttention, 0.0625, model.QKNone},
		{"CohereLabs-c4ai-command-r7b-12-2024-config.json", model.SlidingAttention, 0.25, model.QKNone},
	}
	for _, check := range checks {
		r := core.ReadFile(core.PathJoin("testdata", check.file))
		if !r.OK {
			t.Fatalf("read %s", check.file)
		}
		var cfg Config
		if decoded := core.JSONUnmarshal(r.Value.([]byte), &cfg); !decoded.OK {
			t.Fatalf("parse %s", check.file)
		}
		a, err := cfg.Arch()
		if err != nil {
			t.Fatalf("Arch %s: %v", check.file, err)
		}
		if a.Layer[0].Attention != check.want || a.LogitScale != check.logit || a.QKNormalization != check.qk || !a.ParallelResidual {
			t.Fatalf("%s arch = %+v", check.file, a)
		}
	}
}

func TestConfig_Arch_Bad(t *testing.T) {
	if _, err := (&Config{}).Arch(); err == nil {
		t.Fatal("empty config accepted")
	}
}

func TestConfig_Arch_Ugly(t *testing.T) {
	if _, err := (&Config{HiddenSize: 7, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 1}).Arch(); err == nil {
		t.Fatal("indivisible attention geometry accepted")
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
// indivisible-attention-geometry guard — distinct from _Bad's all-zero
// rejection.
func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 7, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 1}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("indivisible attention geometry became valid after InferFromWeights")
	}
}

func TestConfig_UseQKNormSwitch_Good(t *testing.T) {
	yes, no := true, false
	base := Config{HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, VocabSize: 32}
	base.UseQKNorm = &yes
	a, _ := base.Arch()
	if a.QKNormalization != model.QKLayerNorm {
		t.Fatalf("enabled QK norm = %q", a.QKNormalization)
	}
	base.UseQKNorm = &no
	a, _ = base.Arch()
	if a.QKNormalization != model.QKNone {
		t.Fatalf("disabled QK norm = %q", a.QKNormalization)
	}
}

func TestConfig_Cohere2SlidingInterleave_Good(t *testing.T) {
	c := Config{ModelType: "cohere2", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 8, NumAttentionHeads: 2, NumKeyValueHeads: 1, VocabSize: 32, SlidingWindow: 4, SlidingWindowPattern: 4}
	a, err := c.Arch()
	if err != nil {
		t.Fatal(err)
	}
	want := []model.AttentionType{model.SlidingAttention, model.SlidingAttention, model.SlidingAttention, model.GlobalAttention, model.SlidingAttention, model.SlidingAttention, model.SlidingAttention, model.GlobalAttention}
	for i := range want {
		if a.Layer[i].Attention != want[i] {
			t.Fatalf("layer %d = %v, want %v", i, a.Layer[i].Attention, want[i])
		}
	}
}
