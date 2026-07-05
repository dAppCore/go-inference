// SPDX-Licence-Identifier: EUPL-1.2

package autoround_test

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/autoround"
)

// exampleProjection builds a deterministic single-group W2 projection used by
// the export examples; the optional bias is attached only when supplied.
func exampleProjection(name string, packed byte, scale float32, bias []float32) autoround.PackedProjection {
	tensor := autoround.PackTensor{
		Name:        name,
		Packed:      name + ".packed",
		Scales:      name + ".scales",
		ZeroPoints:  name + ".zeros",
		Shape:       []int32{1, 4},
		Bits:        2,
		GroupSize:   32,
		Symmetric:   true,
		PackedBytes: 1,
		Groups:      1,
		QMin:        -2,
		QMax:        1,
	}
	if len(bias) > 0 {
		tensor.Bias = name + ".bias"
	}
	return autoround.PackedProjection{
		Tensor: tensor,
		Weights: autoround.PackedWeights{
			Scheme:     autoround.SchemeW2A16,
			Format:     autoround.FormatAutoRound,
			Bits:       2,
			GroupSize:  32,
			Symmetric:  true,
			Shape:      []int32{1, 4},
			Packed:     []byte{packed},
			Scales:     []float32{scale},
			ZeroPoints: []float32{0},
			QMin:       -2,
			QMax:       1,
		},
		Bias: bias,
	}
}

// ExampleWritePackedProjectionSafetensors writes a single packed projection to
// a safetensors file and reports the recovered dequantised values.
func ExampleWritePackedProjectionSafetensors() {
	dir := core.MkdirTemp("", "autoround-write1-example-*").Value.(string)
	path := core.PathJoin(dir, "projection.safetensors")
	projection := exampleProjection("model.layers.0.self_attn.q_proj.weight", 0b11100100, 0.5, nil)
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

// ExampleWritePackedProjectionsSafetensors writes several packed projections
// into one safetensors file and reports the resulting tensor count.
func ExampleWritePackedProjectionsSafetensors() {
	dir := core.MkdirTemp("", "autoround-writeN-example-*").Value.(string)
	path := core.PathJoin(dir, "model.safetensors")
	projections := []autoround.PackedProjection{
		exampleProjection("model.layers.0.self_attn.q_proj.weight", 0b11100100, 0.5, []float32{0.25}),
		exampleProjection("model.layers.0.self_attn.k_proj.weight", 0b00011011, 0.25, nil),
	}
	if err := autoround.WritePackedProjectionsSafetensors(context.Background(), path, projections); err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(len(projections))
	// Output: 2
}

// ExampleWriteNativePack writes a directory-level native AutoRound pack — the
// auto_round config sidecar plus the model.safetensors payload — and reports
// the tensor count recorded in the sidecar.
func ExampleWriteNativePack() {
	dir := core.MkdirTemp("", "autoround-pack-example-*").Value.(string)
	projections := []autoround.PackedProjection{
		exampleProjection("model.layers.0.self_attn.q_proj.weight", 0b11100100, 0.5, []float32{0.25}),
		exampleProjection("model.layers.0.self_attn.k_proj.weight", 0b00011011, 0.25, nil),
	}
	info := autoround.PackInfo{
		Bits:          2,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   autoround.QuantMethodAutoRound,
		PackingFormat: string(autoround.FormatAutoRound),
		Scheme:        autoround.SchemeW2A16,
		ExportFormat:  autoround.FormatAutoRound,
	}
	result, err := autoround.WriteNativePack(context.Background(), dir, info, projections)
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(result.TensorCount)
	// Output: 2
}
