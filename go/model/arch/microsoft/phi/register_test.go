// SPDX-Licence-Identifier: EUPL-1.2

package phi

import (
	"slices"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

func TestPhiRegistered_Good(t *testing.T) {
	for _, mt := range []string{"phi", "phi3"} {
		s, ok := model.LookupArch(mt)
		if !ok || s.Parse == nil {
			t.Fatalf("model_type %q is not registered", mt)
		}
	}
}

func TestPhiWeightNames_Good(t *testing.T) {
	// Exact names are receipts from the public checkpoint indexes:
	// https://huggingface.co/microsoft/phi-2/blob/main/model.safetensors.index.json
	// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/model.safetensors.index.json
	phi2, _ := model.LookupArch("phi")
	assertIndexNames(t, "microsoft-phi-2-model.safetensors.index.json", []string{
		"model.final_layernorm.weight", "model.layers.0.self_attn.dense.weight",
		"model.layers.0.mlp.fc1.weight", "model.layers.0.mlp.fc2.weight",
	})
	if phi2.Weights.FinalNorm != "model.final_layernorm.weight" || phi2.Weights.Gate != ".mlp.fc1" || phi2.Weights.Down != ".mlp.fc2" || phi2.Weights.O != ".self_attn.dense" {
		t.Fatalf("Phi-2 names = %+v", phi2.Weights)
	}
	phi3, _ := model.LookupArch("phi3")
	assertIndexNames(t, "microsoft-phi-3-mini-4k-instruct-model.safetensors.index.json", []string{
		"model.norm.weight", "model.layers.0.self_attn.qkv_proj.weight",
		"model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.gate_up_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
	})
	if phi3.Weights.FinalNorm != "model.norm.weight" || phi3.Weights.Q != ".self_attn.q_proj" || phi3.Weights.Gate != ".mlp.gate_proj" {
		t.Fatalf("Phi-3 canonical names = %+v", phi3.Weights)
	}
}

func assertIndexNames(t *testing.T, fixture string, names []string) {
	t.Helper()
	r := core.ReadFile(core.PathJoin("testdata", fixture))
	if !r.OK {
		t.Fatalf("read %s", fixture)
	}
	b, ok := r.Value.([]byte)
	if !ok {
		t.Fatalf("read %s returned a non-byte value", fixture)
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if decoded := core.JSONUnmarshal(b, &index); !decoded.OK {
		t.Fatalf("parse %s", fixture)
	}
	for _, name := range names {
		if _, ok := index.WeightMap[name]; !ok {
			t.Fatalf("%s absent from %s", name, fixture)
		}
	}
}

func TestNormalizePhi3Weights_Good(t *testing.T) {
	// Varied bytes make row-order mistakes observable; no constant-filled tensor
	// can prove that q/k/v or gate/up retain the checkpoint order.
	qkv := safetensors.Tensor{Dtype: "F32", Shape: []int{6, 2}, Data: []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}}
	gateUp := safetensors.Tensor{Dtype: "F32", Shape: []int{4, 2}, Data: []byte{20, 21, 22, 23, 24, 25, 26, 27}}
	out := NormalizePhi3Weights(map[string]safetensors.Tensor{
		"model.layers.0.self_attn.qkv_proj.weight": qkv,
		"model.layers.0.mlp.gate_up_proj.weight":   gateUp,
	})
	for name, want := range map[string][]byte{
		"model.layers.0.self_attn.q_proj.weight": {0, 1, 2, 3},
		"model.layers.0.self_attn.k_proj.weight": {4, 5, 6, 7},
		"model.layers.0.self_attn.v_proj.weight": {8, 9, 10, 11},
		"model.layers.0.mlp.gate_proj.weight":    {20, 21, 22, 23},
		"model.layers.0.mlp.up_proj.weight":      {24, 25, 26, 27},
	} {
		if !slices.Equal(out[name].Data, want) {
			t.Fatalf("%s = %v, want %v", name, out[name].Data, want)
		}
	}
}

func TestNormalizePhi3Weights_Bad(t *testing.T) {
	name := "model.layers.0.self_attn.qkv_proj.weight"
	in := map[string]safetensors.Tensor{name: {Dtype: "F32", Shape: []int{5, 2}, Data: make([]byte, 10)}}
	out := NormalizePhi3Weights(in)
	if _, ok := out["model.layers.0.self_attn.q_proj.weight"]; ok || !slices.Equal(out[name].Data, in[name].Data) {
		t.Fatal("malformed fused tensor was split or lost")
	}
}

func TestNormalizePhi3Weights_Ugly(t *testing.T) {
	if got := NormalizePhi3Weights(nil); len(got) != 0 {
		t.Fatalf("nil weights normalised to %d entries", len(got))
	}
}

func TestPhiWeightNames_Bad(t *testing.T) {
	if s, ok := model.LookupArch("PhiForCausalLM"); ok || s.Parse != nil {
		t.Fatal("architecture class was registered as a model_type")
	}
}

func TestPhiWeightNames_Ugly(t *testing.T) {
	if _, ok := model.LookupArch(""); ok {
		t.Fatal("empty model_type unexpectedly registered")
	}
}
