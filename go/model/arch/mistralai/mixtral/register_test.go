// SPDX-Licence-Identifier: EUPL-1.2

package mixtral

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

func TestRegister_WeightNames_Good(t *testing.T) {
	w := WeightNames()
	if w.Router != ".block_sparse_moe.gate.weight" || w.ExpertGate != ".block_sparse_moe.experts.%d.w1.weight" ||
		w.ExpertDown != ".block_sparse_moe.experts.%d.w2.weight" || w.ExpertUp != ".block_sparse_moe.experts.%d.w3.weight" {
		t.Fatalf("Mixtral names = %+v", w)
	}
}

// TestRegister_WeightNames_Bad guards against a copy-paste alias collision: the
// router role must never share a tensor-name template with an expert role.
func TestRegister_WeightNames_Bad(t *testing.T) {
	w := WeightNames()
	if w.Router == w.ExpertGate || w.Router == w.ExpertDown || w.Router == w.ExpertUp {
		t.Fatalf("router alias collides with an expert role: %+v", w)
	}
}

// TestRegister_WeightNames_Ugly proves each expert template carries the %d
// placeholder NormalizeWeights' per-expert Sprintf substitution depends on —
// a dropped placeholder would silently alias every expert onto one tensor.
func TestRegister_WeightNames_Ugly(t *testing.T) {
	w := WeightNames()
	for _, name := range []string{w.ExpertGate, w.ExpertDown, w.ExpertUp} {
		if core.Sprintf(name, 0) == core.Sprintf(name, 1) {
			t.Fatalf("expert weight template does not vary by index: %q", name)
		}
	}
}

func TestRegister_NormalizeWeights_Good(t *testing.T) {
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

// TestRegister_NormalizeWeights_Bad proves non-expert tensors pass through
// unrenamed — normalisation must only touch the block_sparse_moe suffixes.
func TestRegister_NormalizeWeights_Bad(t *testing.T) {
	in := map[string]safetensors.Tensor{"model.embed_tokens.weight": {Shape: []int{8, 8}}}
	got := NormalizeWeights(in)
	if len(got) != 1 {
		t.Fatalf("non-expert tensor set mutated: %+v", got)
	}
	if tensor, ok := got["model.embed_tokens.weight"]; !ok || tensor.Shape[0] != 8 {
		t.Fatalf("non-expert tensor renamed or altered: %+v", got)
	}
}

// TestRegister_NormalizeWeights_Ugly proves an empty checkpoint map degrades
// gracefully — no panic, no phantom aliases fabricated from nothing.
func TestRegister_NormalizeWeights_Ugly(t *testing.T) {
	got := NormalizeWeights(map[string]safetensors.Tensor{})
	if len(got) != 0 {
		t.Fatalf("empty input produced %d tensors", len(got))
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
