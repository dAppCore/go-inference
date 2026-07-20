// SPDX-Licence-Identifier: EUPL-1.2

package smollm3

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"testing"
)

// Source: https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/config.json
func TestConfig_Arch_Good(t *testing.T) {
	b := core.ReadFile(core.PathJoin("testdata", "huggingface-smollm3-config.json"))
	if !b.OK {
		t.Fatal("read config fixture")
	}
	spec, _ := model.LookupArch("smollm3")
	cfg, err := spec.Parse(b.Value.([]byte))
	if err != nil {
		t.Fatal(err)
	}
	a, err := cfg.Arch()
	if err != nil {
		t.Fatal(err)
	}
	if a.HeadDim != 128 || a.KVHeads != 4 || !a.Layer[0].DisableRotary || a.Layer[3].DisableRotary || a.RopeBase != 5_000_000 {
		t.Fatalf("SmolLM3 arch=%+v", a)
	}
}
func TestConfig_Arch_Bad(t *testing.T) {
	_, err := (&Config{ModelType: "smollm3", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 2, NumAttentionHeads: 2, NumKeyValueHeads: 1, VocabSize: 8, NoRopeLayers: []int{1}}).Arch()
	if err == nil {
		t.Fatal("accepted short NoPE schedule")
	}
}
func TestConfig_Arch_Ugly(t *testing.T) {
	_, err := (&Config{}).Arch()
	if err == nil {
		t.Fatal("accepted empty config")
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
// NoPE-schedule-length guard — distinct from _Bad's all-zero rejection.
func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{ModelType: "smollm3", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 2, NumAttentionHeads: 2, NumKeyValueHeads: 1, VocabSize: 8, NoRopeLayers: []int{1}}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("short NoPE schedule became valid after InferFromWeights")
	}
}

// Source: https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/model.safetensors.index.json
func TestRegister_WeightMap_Good(t *testing.T) {
	b := core.ReadFile(core.PathJoin("testdata", "huggingface-smollm3-index-receipt.json"))
	if !b.OK {
		t.Fatal("read index receipt")
	}
	var x struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if r := core.JSONUnmarshal(b.Value.([]byte), &x); !r.OK {
		t.Fatal("parse index")
	}
	for _, n := range []string{"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate_proj.weight", "model.norm.weight"} {
		if _, ok := x.WeightMap[n]; !ok {
			t.Fatalf("missing %s", n)
		}
	}
}
