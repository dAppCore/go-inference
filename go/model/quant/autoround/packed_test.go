// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import "testing"

import core "dappco.re/go"

func TestPacked_PackQuantizedWeights_Good(t *testing.T) {
	t.Run("RoundTripsDequantized", func(t *testing.T) {
		weights := make([]float32, 32)
		for i := range weights {
			weights[i] = float32(i-16) / 7
		}
		quantized, err := QuantizeWeights(weights, QuantizeConfig{Scheme: SchemeW4A16, GroupSize: 32, Iters: 0})
		if err != nil {
			t.Fatalf("QuantizeWeights() error = %v", err)
		}

		packed, err := PackQuantizedWeights(quantized, []int32{4, 8})
		if err != nil {
			t.Fatalf("PackQuantizedWeights() error = %v", err)
		}
		got, err := DequantizePackedWeights(packed)
		if err != nil {
			t.Fatalf("DequantizePackedWeights() error = %v", err)
		}

		if packed.QMin != -8 || packed.QMax != 7 || len(packed.Packed) != 16 {
			t.Fatalf("packed metadata = %+v, want W4 symmetric byte layout", packed)
		}
		assertAutoRoundFloat32SliceClose(t, got, quantized.Dequantized, 1e-6)
	})
	t.Run("PreservesSignedQValues", func(t *testing.T) {
		quantized := QuantizedWeights{
			Scheme:      SchemeW2A16,
			Bits:        2,
			GroupSize:   32,
			Symmetric:   true,
			QValues:     []int16{-2, -1, 0, 1},
			Scales:      []float32{0.5},
			ZeroPoints:  []float32{0},
			Dequantized: []float32{-1, -0.5, 0, 0.5},
		}

		packed, err := PackQuantizedWeights(quantized, []int32{4})
		if err != nil {
			t.Fatalf("PackQuantizedWeights() error = %v", err)
		}
		if len(packed.Packed) != 1 || packed.Packed[0] != 0b11100100 {
			t.Fatalf("packed bytes = %08b, want signed qmin-offset layout 11100100", packed.Packed)
		}
	})
}

func TestPacked_PackQuantizedWeights_Bad(t *testing.T) {
	quantized := QuantizedWeights{
		Scheme:     SchemeW4A16,
		Bits:       4,
		GroupSize:  32,
		Symmetric:  true,
		QValues:    []int16{0, 1},
		Scales:     []float32{1},
		ZeroPoints: []float32{0},
	}
	if _, err := PackQuantizedWeights(quantized, []int32{3}); err == nil || !core.Contains(err.Error(), "shape") {
		t.Fatalf("PackQuantizedWeights(bad shape) error = %v, want shape diagnostic", err)
	}

	quantized.QValues[0] = -9
	if _, err := PackQuantizedWeights(quantized, []int32{2}); err == nil || !core.Contains(err.Error(), "outside range") {
		t.Fatalf("PackQuantizedWeights(bad qvalue) error = %v, want range diagnostic", err)
	}
}

func TestPacked_PackQuantizedWeights_Ugly(t *testing.T) {
	// An unsupported bit-width is the degenerate input: it is rejected up front,
	// before the shape check.
	bad := QuantizedWeights{Bits: 5, GroupSize: 32, QValues: []int16{0}, Scales: []float32{1}, ZeroPoints: []float32{0}}
	if _, err := PackQuantizedWeights(bad, []int32{1}); err == nil || !core.Contains(err.Error(), "bits") {
		t.Fatalf("PackQuantizedWeights(bad bits) error = %v, want bits diagnostic", err)
	}
	// Empty qvalues with a non-empty shape: the empty-qvalues guard fires first.
	empty := QuantizedWeights{Bits: 4, GroupSize: 32}
	if _, err := PackQuantizedWeights(empty, []int32{4}); err == nil || !core.Contains(err.Error(), "qvalues") {
		t.Fatalf("PackQuantizedWeights(no qvalues) error = %v, want qvalues diagnostic", err)
	}
}

func TestPacked_DequantizePackedWeights_Good(t *testing.T) {
	// A hand-built W2 symmetric byte (11100100) decodes to the canonical
	// scaled values; the round trip is exercised end-to-end here.
	packed := PackedWeights{
		Scheme:     SchemeW2A16,
		Format:     FormatAutoRound,
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
	got, err := DequantizePackedWeights(packed)
	if err != nil {
		t.Fatalf("DequantizePackedWeights() error = %v", err)
	}
	assertAutoRoundFloat32SliceClose(t, got, []float32{-1, -0.5, 0, 0.5}, 1e-6)
}

func TestPacked_DequantizePackedWeights_Bad(t *testing.T) {
	packed := PackedWeights{
		Bits:       4,
		GroupSize:  32,
		Shape:      []int32{3},
		Packed:     []byte{0},
		Scales:     []float32{1},
		ZeroPoints: []float32{0},
		QMin:       -8,
		QMax:       7,
	}
	if _, err := DequantizePackedWeights(packed); err == nil || !core.Contains(err.Error(), "packed length") {
		t.Fatalf("DequantizePackedWeights(bad length) error = %v, want packed length diagnostic", err)
	}
}

func TestPacked_DequantizePackedWeights_Ugly(t *testing.T) {
	// A scale/zero-point count that disagrees with the group count is the
	// degenerate metadata case: dequant rejects it rather than indexing past
	// the scales slice.
	packed := PackedWeights{
		Bits:       4,
		GroupSize:  32,
		Shape:      []int32{1, 4},
		Packed:     []byte{0, 0},
		Scales:     nil, // expected 1 group, supplied 0
		ZeroPoints: []float32{0},
		QMin:       -8,
		QMax:       7,
	}
	if _, err := DequantizePackedWeights(packed); err == nil || !core.Contains(err.Error(), "scale count") {
		t.Fatalf("DequantizePackedWeights(missing scales) error = %v, want scale count diagnostic", err)
	}
	// A non-positive group size cannot define groups: it is rejected before the
	// element math runs.
	packed.Scales = []float32{1}
	packed.GroupSize = 0
	if _, err := DequantizePackedWeights(packed); err == nil || !core.Contains(err.Error(), "group size") {
		t.Fatalf("DequantizePackedWeights(zero group size) error = %v, want group size diagnostic", err)
	}
}
