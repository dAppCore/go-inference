// SPDX-Licence-Identifier: EUPL-1.2

package autoround_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/autoround"
)

// ExamplePackQuantizedWeights bit-packs signed quantised values into the
// AutoRound byte layout, offsetting each value by qmin so the packed nibbles
// are unsigned.
func ExamplePackQuantizedWeights() {
	quantized := autoround.QuantizedWeights{
		Scheme:     autoround.SchemeW2A16,
		Bits:       2,
		GroupSize:  32,
		Symmetric:  true,
		QValues:    []int16{-2, -1, 0, 1},
		Scales:     []float32{0.5},
		ZeroPoints: []float32{0},
	}
	packed, err := autoround.PackQuantizedWeights(quantized, []int32{4})
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(packed.Bits, packed.QMin, packed.QMax, len(packed.Packed))
	// Output: 2 -2 1 1
}

// ExampleDequantizePackedWeights reverses the bit-packing, applying per-group
// scale and zero point to recover the float values.
func ExampleDequantizePackedWeights() {
	packed := autoround.PackedWeights{
		Bits:       2,
		GroupSize:  32,
		Symmetric:  true,
		Shape:      []int32{1, 4},
		Packed:     []byte{0b11100100},
		Scales:     []float32{0.5},
		ZeroPoints: []float32{0},
		QMin:       -2,
		QMax:       1,
	}
	values, err := autoround.DequantizePackedWeights(packed)
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(values[0], values[1], values[2], values[3])
	// Output: -1 -0.5 0 0.5
}
