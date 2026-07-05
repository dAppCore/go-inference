// SPDX-Licence-Identifier: EUPL-1.2

package autoround_test

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/autoround"
)

// ExampleLoadPackedProjectionFromSafetensors writes one packed projection to a
// safetensors file via the export primitive, then reads it back by tensor name
// and dequantises the recovered weights.
func ExampleLoadPackedProjectionFromSafetensors() {
	dir := core.MkdirTemp("", "autoround-load-example-*").Value.(string)
	path := core.PathJoin(dir, "model.safetensors")
	projection := autoround.PackedProjection{
		Tensor: autoround.PackTensor{
			Name:        "model.layers.0.self_attn.q_proj.weight",
			Packed:      "model.layers.0.self_attn.q_proj.weight.packed",
			Scales:      "model.layers.0.self_attn.q_proj.weight.scales",
			ZeroPoints:  "model.layers.0.self_attn.q_proj.weight.zeros",
			Shape:       []int32{1, 4},
			Bits:        2,
			GroupSize:   32,
			Symmetric:   true,
			PackedBytes: 1,
			Groups:      1,
			QMin:        -2,
			QMax:        1,
		},
		Weights: autoround.PackedWeights{
			Scheme:     autoround.SchemeW2A16,
			Format:     autoround.FormatAutoRound,
			Bits:       2,
			GroupSize:  32,
			Symmetric:  true,
			Shape:      []int32{1, 4},
			Packed:     []byte{0b11100100},
			Scales:     []float32{0.5},
			ZeroPoints: []float32{0},
			QMin:       -2,
			QMax:       1,
		},
	}
	if err := autoround.WritePackedProjectionSafetensors(context.Background(), path, projection); err != nil {
		core.Println(err.Error())
		return
	}
	info := autoround.PackInfo{
		Bits:          2,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   autoround.QuantMethodAutoRound,
		PackingFormat: string(autoround.FormatAutoRound),
		Tensors:       []autoround.PackTensor{projection.Tensor},
	}
	got, err := autoround.LoadPackedProjectionFromSafetensors(info, []string{path}, projection.Tensor.Name)
	if err != nil {
		core.Println(err.Error())
		return
	}
	values, err := autoround.DequantizePackedWeights(got.Weights)
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(values[0], values[1], values[2], values[3])
	// Output: -1 -0.5 0 0.5
}

// ExamplePackInfo_LookupTensor finds a tensor mapping by its logical name,
// reporting the ok flag for a present and an absent name.
func ExamplePackInfo_LookupTensor() {
	info := autoround.PackInfo{
		Bits:          4,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   autoround.QuantMethodAutoRound,
		PackingFormat: string(autoround.FormatAutoRound),
		Tensors: []autoround.PackTensor{{
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
	tensor, ok := info.LookupTensor("model.layers.0.mlp.gate_proj.weight")
	core.Println(ok, tensor.Bits)
	_, missing := info.LookupTensor("model.layers.0.mlp.up_proj.weight")
	core.Println(missing)
	// Output:
	// true 4
	// false
}
