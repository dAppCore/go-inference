// SPDX-Licence-Identifier: EUPL-1.2

package mpt

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"testing"
)

// Source: https://huggingface.co/adamrb/mpt-30b-chat-safetensors/blob/main/config.json
func TestConfig_Arch_Good(t *testing.T) {
	b := core.ReadFile(core.PathJoin("testdata", "adamrb-mpt-30b-chat-config.json"))
	if !b.OK {
		t.Fatal("read config fixture")
	}
	spec, _ := model.LookupArch("mpt")
	cfg, err := spec.Parse(b.Value.([]byte))
	if err != nil {
		t.Fatal(err)
	}
	a, err := cfg.Arch()
	if err != nil {
		t.Fatal(err)
	}
	if !a.ALiBi || !a.LearnedAbsolutePositions || a.HeadDim != 112 || a.FF != 28672 {
		t.Fatalf("MPT arch = %+v", a)
	}
}
func TestConfig_Arch_Bad(t *testing.T) {
	_, err := (&Config{ModelType: "mpt", DModel: 7, NHeads: 2, NLayers: 1, ExpansionRatio: 4, VocabSize: 8}).Arch()
	if err == nil {
		t.Fatal("accepted indivisible heads")
	}
}
func TestConfig_Arch_Ugly(t *testing.T) {
	_, err := (&Config{ModelType: "unknown", DModel: 8, NHeads: 2, NLayers: 1, ExpansionRatio: 4, VocabSize: 8}).Arch()
	if err == nil {
		t.Fatal("accepted unknown model_type")
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	c := Config{ModelType: "mpt", DModel: 8}
	c.InferFromWeights(nil)
	if c.DModel != 8 {
		t.Fatalf("InferFromWeights changed config: %+v", c)
	}
}

func TestConfig_InferFromWeights_Bad(t *testing.T) {
	c := Config{}
	c.InferFromWeights(nil)
	if _, err := c.Arch(); err == nil {
		t.Fatal("empty config became valid after InferFromWeights")
	}
}

// TestConfig_InferFromWeights_Ugly proves the no-op does not paper over the
// model_type gate — distinct from _Bad's all-zero-fields rejection.
func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	c := Config{ModelType: "unknown", DModel: 8, NHeads: 2, NLayers: 1, ExpansionRatio: 4, VocabSize: 8}
	c.InferFromWeights(nil)
	if _, err := c.Arch(); err == nil {
		t.Fatal("unknown model_type became valid after InferFromWeights")
	}
}

// Source: https://huggingface.co/adamrb/mpt-30b-chat-safetensors/blob/main/model.safetensors.index.json
func TestRegister_WeightMap_Good(t *testing.T) {
	b := core.ReadFile(core.PathJoin("testdata", "adamrb-mpt-30b-chat-index-receipt.json"))
	if !b.OK {
		t.Fatal("read index receipt")
	}
	var x struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if r := core.JSONUnmarshal(b.Value.([]byte), &x); !r.OK {
		t.Fatal("parse index receipt")
	}
	if _, ok := x.WeightMap["transformer.blocks.0.attn.Wqkv.weight"]; !ok {
		t.Fatal("missing fused qkv")
	}
	spec, _ := model.LookupArch("mpt")
	if spec.Weights.Embed != "transformer.wte" {
		t.Fatalf("weights=%+v", spec.Weights)
	}
}
