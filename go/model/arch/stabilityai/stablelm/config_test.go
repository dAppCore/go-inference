// SPDX-Licence-Identifier: EUPL-1.2

package stablelm

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	"math"
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

// TestTinyStableLMForward_Good uses varied seeded weights and partial rotary.
func TestTinyStableLMForward_Good(t *testing.T) {
	syn := func(n, seed int) []float32 {
		o := make([]float32, n)
		for i := range o {
			o[i] = float32((i*seed)%19-9) / 31
		}
		return o
	}
	m := composed.NewAttnMixer(&composed.AttnWeights{QProj: syn(128, 3), KProj: syn(64, 5), VProj: syn(64, 7), OProj: syn(128, 11)}, composed.AttnConfig{Heads: 4, KVHeads: 2, HeadDim: 4, RotaryDim: 2, RopeTheta: 10000})
	x := syn(32, 13)
	got, _, err := m.Forward(x, 4, 8, nil)
	if err != nil {
		t.Fatal(err)
	}
	if math.Float32bits(got[24]) != math.Float32bits(-0.013525661) {
		t.Fatalf("golden=%g", got[24])
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
