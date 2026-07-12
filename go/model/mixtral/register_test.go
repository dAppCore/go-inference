// SPDX-Licence-Identifier: EUPL-1.2

package mixtral

import (
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

func TestWeightNames_Good(t *testing.T) {
	w := WeightNames()
	if w.Router != ".block_sparse_moe.gate.weight" || w.ExpertGate != ".block_sparse_moe.experts.%d.w1.weight" ||
		w.ExpertDown != ".block_sparse_moe.experts.%d.w2.weight" || w.ExpertUp != ".block_sparse_moe.experts.%d.w3.weight" {
		t.Fatalf("Mixtral names = %+v", w)
	}
}

func TestNormalizeWeights_Good(t *testing.T) {
	in := map[string]safetensors.Tensor{
		"model.layers.0.block_sparse_moe.gate.weight":         {Shape: []int{2, 8}},
		"model.layers.0.block_sparse_moe.experts.0.w1.weight": {Shape: []int{16, 8}},
		"model.layers.0.block_sparse_moe.experts.0.w2.weight": {Shape: []int{8, 16}},
		"model.layers.0.block_sparse_moe.experts.0.w3.weight": {Shape: []int{16, 8}},
	}
	got := NormalizeWeights(in)
	for _, name := range []string{
		"model.layers.0.mlp.gate.weight",
		"model.layers.0.mlp.experts.0.gate_proj.weight",
		"model.layers.0.mlp.experts.0.down_proj.weight",
		"model.layers.0.mlp.experts.0.up_proj.weight",
	} {
		if _, ok := got[name]; !ok {
			t.Fatalf("normalised tensor %q absent", name)
		}
	}
	if len(in) != 4 {
		t.Fatal("normalisation mutated source map")
	}
}

func TestMixtralRegistered_Good(t *testing.T) {
	spec, ok := model.LookupArch("mixtral")
	if !ok || spec.Composed == nil || spec.Parse == nil {
		t.Fatalf("Mixtral registration = found %v spec %+v", ok, spec)
	}
}

func TestParse_Bad(t *testing.T) {
	spec, _ := model.LookupArch("mixtral")
	if _, err := spec.Parse([]byte("{")); err == nil {
		t.Fatal("malformed config accepted")
	}
}

func TestParseAndLoadErrors_Ugly(t *testing.T) {
	spec, _ := model.LookupArch("mixtral")
	if _, err := spec.Parse([]byte(`{"hidden_size":8}`)); err != nil {
		t.Fatalf("valid JSON rejected: %v", err)
	}
	if _, err := spec.Composed(map[string]safetensors.Tensor{}, []byte(`{"model_type":"mixtral","hidden_size":8,"num_hidden_layers":1}`)); err == nil {
		t.Fatal("missing checkpoint tensors accepted")
	}
}
