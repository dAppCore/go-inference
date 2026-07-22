// SPDX-Licence-Identifier: EUPL-1.2

package stablelm

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"testing"
)

// Source: https://huggingface.co/stabilityai/stablelm-2-12b/blob/main/config.json
func TestConfig_Arch_Good(t *testing.T) {
	b := core.ReadFile(core.PathJoin("testdata", "stabilityai-stablelm-2-12b-config.json"))
	if !b.OK {
		t.Fatal("read config fixture")
	}
	spec, _ := model.LookupArch("stablelm")
	cfg, err := spec.Parse(b.Value.([]byte))
	if err != nil {
		t.Fatal(err)
	}
	a, err := cfg.Arch()
	if err != nil {
		t.Fatal(err)
	}
	if a.HeadDim != 160 || a.RotaryDim != 40 || a.KVHeads != 8 || !a.ParallelResidual || a.QKNormalization != model.QKLayerNorm {
		t.Fatalf("StableLM arch=%+v", a)
	}
}
func TestConfig_Arch_Bad(t *testing.T) {
	_, err := (&Config{ModelType: "stablelm", HiddenSize: 7, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 8}).Arch()
	if err == nil {
		t.Fatal("accepted invalid geometry")
	}
}
func TestConfig_Arch_Ugly(t *testing.T) {
	_, err := (&Config{ModelType: "stablelm", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, VocabSize: 8, PartialRotaryFactor: .3}).Arch()
	if err == nil {
		t.Fatal("accepted fractional rotary dimension")
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
// fractional-rotary-dimension guard — distinct from _Bad's all-zero rejection.
func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{ModelType: "stablelm", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, VocabSize: 8, PartialRotaryFactor: .3}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("fractional rotary dimension became valid after InferFromWeights")
	}
}

// Source: https://huggingface.co/stabilityai/stablelm-2-12b/blob/main/model.safetensors.index.json
func TestRegister_WeightMap_Good(t *testing.T) {
	b := core.ReadFile(core.PathJoin("testdata", "stabilityai-stablelm-2-12b-index-receipt.json"))
	if !b.OK {
		t.Fatal("read index receipt")
	}
	var x struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if r := core.JSONUnmarshal(b.Value.([]byte), &x); !r.OK {
		t.Fatal("parse index")
	}
	for _, n := range []string{"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.down_proj.weight", "lm_head.weight"} {
		if _, ok := x.WeightMap[n]; !ok {
			t.Fatalf("missing %s", n)
		}
	}
}
