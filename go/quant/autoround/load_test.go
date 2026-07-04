// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import "testing"

import core "dappco.re/go"

func TestLoad_LoadPackedProjectionFromSafetensors_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.safetensors")
	writeAutoRoundSafetensors(t, path, []autoRoundSafetensorTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight.packed", DType: "U8", Shape: []int{1}, Raw: []byte{0b11100100}},
		autoRoundF32Tensor("model.layers.0.self_attn.q_proj.weight.scales", []float32{0.5}, 1),
		autoRoundF32Tensor("model.layers.0.self_attn.q_proj.weight.zeros", []float32{0}, 1),
		autoRoundF32Tensor("model.layers.0.self_attn.q_proj.bias", []float32{0.25}, 1),
	})
	info := PackInfo{
		Bits:          2,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Tensors: []PackTensor{{
			Name:       "model.layers.0.self_attn.q_proj.weight",
			Packed:     "model.layers.0.self_attn.q_proj.weight.packed",
			Scales:     "model.layers.0.self_attn.q_proj.weight.scales",
			ZeroPoints: "model.layers.0.self_attn.q_proj.weight.zeros",
			Bias:       "model.layers.0.self_attn.q_proj.bias",
			Shape:      []int32{1, 4},
			Bits:       2,
			GroupSize:  32,
			Symmetric:  true,
		}},
	}
	info.normalise()

	projection, err := LoadPackedProjectionFromSafetensors(info, []string{path}, "model.layers.0.self_attn.q_proj.weight")
	if err != nil {
		t.Fatalf("LoadPackedProjectionFromSafetensors() error = %v", err)
	}
	got, err := DequantizePackedWeights(projection.Weights)
	if err != nil {
		t.Fatalf("DequantizePackedWeights() error = %v", err)
	}

	assertAutoRoundFloat32SliceClose(t, got, []float32{-1, -0.5, 0, 0.5}, 1e-6)
	assertAutoRoundFloat32SliceClose(t, projection.Bias, []float32{0.25}, 1e-6)
	if projection.Tensor.Name != "model.layers.0.self_attn.q_proj.weight" || projection.Weights.QMin != -2 || projection.Weights.QMax != 1 {
		t.Fatalf("projection metadata = %+v, want qmin/qmax and tensor name", projection)
	}
}

func TestLoad_LoadPackedProjectionFromSafetensors_Bad(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.safetensors")
	writeAutoRoundSafetensors(t, path, []autoRoundSafetensorTensor{
		{Name: "weight.packed", DType: "U8", Shape: []int{1}, Raw: []byte{0}},
		autoRoundF32Tensor("weight.scales", []float32{1}, 1),
	})
	info := PackInfo{
		Bits:          4,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Tensors: []PackTensor{{
			Name:       "weight",
			Packed:     "weight.packed",
			Scales:     "weight.scales",
			ZeroPoints: "weight.zeros",
			Shape:      []int32{1, 2},
			Bits:       4,
			GroupSize:  32,
			Symmetric:  true,
		}},
	}
	info.normalise()

	_, err := LoadPackedProjectionFromSafetensors(info, []string{path}, "weight")
	if err == nil || !core.Contains(err.Error(), "missing") {
		t.Fatalf("LoadPackedProjectionFromSafetensors(missing zero point) error = %v, want missing tensor diagnostic", err)
	}
}

func TestLoad_LoadPackedProjectionFromSafetensors_Ugly(t *testing.T) {
	// A non-native pack info (no tensor map) cannot be loaded by name: the
	// guard fires before any file is opened.
	bare := PackInfo{Bits: 4, GroupSize: 32, Symmetric: true, QuantMethod: QuantMethodAutoRound, PackingFormat: string(FormatAutoRound)}
	bare.normalise()
	if _, err := LoadPackedProjectionFromSafetensors(bare, []string{"unused.safetensors"}, "weight"); err == nil || !core.Contains(err.Error(), "native tensor map") {
		t.Fatalf("LoadPackedProjectionFromSafetensors(no tensor map) error = %v, want native tensor map diagnostic", err)
	}
	// A name that is not in the tensor map is rejected with the lookup
	// diagnostic, never a nil-deref on the missing entry.
	info := PackInfo{
		Bits:          4,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Tensors: []PackTensor{{
			Name:       "present",
			Packed:     "present.packed",
			Scales:     "present.scales",
			ZeroPoints: "present.zeros",
			Shape:      []int32{1, 2},
			Bits:       4,
			GroupSize:  32,
			Symmetric:  true,
		}},
	}
	info.normalise()
	if _, err := LoadPackedProjectionFromSafetensors(info, []string{"unused.safetensors"}, "absent"); err == nil || !core.Contains(err.Error(), "does not contain") {
		t.Fatalf("LoadPackedProjectionFromSafetensors(unknown name) error = %v, want tensor-map miss diagnostic", err)
	}
}

func TestLoad_PackInfo_LookupTensor_Good(t *testing.T) {
	info := PackInfo{
		Bits:          4,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Tensors: []PackTensor{{
			Name:       "model.layers.0.mlp.gate_proj.weight",
			Packed:     "model.layers.0.mlp.gate_proj.weight.packed",
			Scales:     "model.layers.0.mlp.gate_proj.weight.scales",
			ZeroPoints: "model.layers.0.mlp.gate_proj.weight.zeros",
			Shape:      []int32{1, 2},
			Bits:       4,
			GroupSize:  32,
			Symmetric:  true,
		}},
	}
	info.normalise()
	tensor, ok := info.LookupTensor("model.layers.0.mlp.gate_proj.weight")
	if !ok {
		t.Fatal("LookupTensor(present) ok = false")
	}
	if tensor.Name != "model.layers.0.mlp.gate_proj.weight" || tensor.Bits != 4 {
		t.Fatalf("LookupTensor(present) = %+v, want the gate projection tensor", tensor)
	}
}

func TestLoad_PackInfo_LookupTensor_Bad(t *testing.T) {
	info := PackInfo{
		Bits:          4,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Tensors: []PackTensor{{
			Name:       "present",
			Packed:     "present.packed",
			Scales:     "present.scales",
			ZeroPoints: "present.zeros",
			Shape:      []int32{1, 2},
			Bits:       4,
			GroupSize:  32,
			Symmetric:  true,
		}},
	}
	info.normalise()
	tensor, ok := info.LookupTensor("absent")
	if ok {
		t.Fatal("LookupTensor(absent) ok = true, want false")
	}
	if tensor.Name != "" {
		t.Fatalf("LookupTensor(absent) = %+v, want zero tensor", tensor)
	}
}

func TestLoad_PackInfo_LookupTensor_Ugly(t *testing.T) {
	// LookupTensor trims its argument, so whitespace-padded names must still
	// match. An empty tensor map can never match anything.
	info := PackInfo{
		Bits:          4,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Tensors: []PackTensor{{
			Name:       "weight",
			Packed:     "weight.packed",
			Scales:     "weight.scales",
			ZeroPoints: "weight.zeros",
			Shape:      []int32{1, 2},
			Bits:       4,
			GroupSize:  32,
			Symmetric:  true,
		}},
	}
	info.normalise()
	if _, ok := info.LookupTensor("  weight  "); !ok {
		t.Fatal("LookupTensor(padded name) ok = false, want trim-and-match")
	}
	if _, ok := (PackInfo{}).LookupTensor("weight"); ok {
		t.Fatal("LookupTensor(empty map) ok = true, want false")
	}
}
