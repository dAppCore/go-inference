// SPDX-Licence-Identifier: EUPL-1.2

package mpt

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	"math"
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

// TestTinyMPTForward_Good uses varied deterministic fills through the shared ALiBi op.
func TestTinyMPTForward_Good(t *testing.T) {
	w := make([]float32, 16)
	for i := range w {
		w[i] = float32((i*7)%11-5) / 17
	}
	m := composed.NewAttnMixer(&composed.AttnWeights{QProj: w, KProj: w, VProj: w, OProj: w}, composed.AttnConfig{Heads: 2, KVHeads: 2, HeadDim: 2, ALiBi: true})
	x := []float32{.2, -.1, .4, .3, -.3, .5, .1, -.2}
	got, _, err := m.Forward(x, 2, 4, nil)
	if err != nil {
		t.Fatal(err)
	}
	if math.Float32bits(got[4]) != math.Float32bits(0.0039271647) {
		t.Fatalf("golden=%g", got[4])
	}
}

// Source: https://huggingface.co/adamrb/mpt-30b-chat-safetensors/blob/main/model.safetensors.index.json
func TestRegister_WeightIndex_Good(t *testing.T) {
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
