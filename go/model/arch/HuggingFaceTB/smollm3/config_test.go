// SPDX-Licence-Identifier: EUPL-1.2

package smollm3

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	"math"
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

// TestTinySmolLM3Forward_Good proves the declared RoPE and NoPE paths differ under varied fills.
func TestTinySmolLM3Forward_Good(t *testing.T) {
	syn := func(n, seed int) []float32 {
		o := make([]float32, n)
		for i := range o {
			o[i] = float32((i*seed)%23-11) / 37
		}
		return o
	}
	w := &composed.AttnWeights{QProj: syn(64, 3), KProj: syn(32, 5), VProj: syn(32, 7), OProj: syn(64, 11)}
	x := syn(24, 13)
	rope := composed.NewAttnMixer(w, composed.AttnConfig{Heads: 2, KVHeads: 1, HeadDim: 4, RotaryDim: 4, RopeTheta: 5_000_000})
	nope := composed.NewAttnMixer(w, composed.AttnConfig{Heads: 2, KVHeads: 1, HeadDim: 4})
	a, _, e := rope.Forward(x, 3, 8, nil)
	if e != nil {
		t.Fatal(e)
	}
	b, _, e := nope.Forward(x, 3, 8, nil)
	if e != nil {
		t.Fatal(e)
	}
	if math.Float32bits(a[16]) == math.Float32bits(b[16]) {
		t.Fatal("RoPE and NoPE paths unexpectedly equal")
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
