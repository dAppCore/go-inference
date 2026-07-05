// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import "testing"

import core "dappco.re/go"

// TestPackedCov_ThreeBit_SpansByteBoundary covers the multi-byte write and read
// branches of packUnsignedBits / unpackUnsignedBits. Only 3-bit packing crosses
// a byte boundary (index 2 starts at bit 6, spilling 1 bit into the next byte),
// so a 3-bit round trip is the sole path through the spanning code.
func TestPackedCov_ThreeBit_SpansByteBoundary(t *testing.T) {
	// Six values across one group; the symmetric 3-bit range is [-4, 3].
	quantized := QuantizedWeights{
		Scheme:      SchemeW4A16, // scheme tag is cosmetic for packing
		Bits:        3,
		GroupSize:   32,
		Symmetric:   true,
		QValues:     []int16{-4, -3, 3, 1, -1, 2},
		Scales:      []float32{0.5},
		ZeroPoints:  []float32{0},
		Dequantized: []float32{-2, -1.5, 1.5, 0.5, -0.5, 1},
	}
	packed, err := PackQuantizedWeights(quantized, []int32{1, 6})
	if err != nil {
		t.Fatalf("PackQuantizedWeights(3-bit) error = %v", err)
	}
	if packed.QMin != -4 || packed.QMax != 3 {
		t.Fatalf("PackQuantizedWeights(3-bit) range = [%d,%d], want [-4,3]", packed.QMin, packed.QMax)
	}
	// 6 values * 3 bits = 18 bits → ceil to 3 bytes.
	if len(packed.Packed) != 3 {
		t.Fatalf("PackQuantizedWeights(3-bit) packed bytes = %d, want 3", len(packed.Packed))
	}
	got, err := DequantizePackedWeights(packed)
	if err != nil {
		t.Fatalf("DequantizePackedWeights(3-bit) error = %v", err)
	}
	assertAutoRoundFloat32SliceClose(t, got, quantized.Dequantized, 1e-6)
}

// TestPackedCov_ValidatePackedWeights_BadBits covers the unsupported-bit-width
// rejection inside validatePackedWeights, reached via DequantizePackedWeights.
func TestPackedCov_ValidatePackedWeights_BadBits(t *testing.T) {
	packed := PackedWeights{
		Bits:       5, // unsupported
		GroupSize:  32,
		Shape:      []int32{1, 4},
		Packed:     []byte{0, 0, 0},
		Scales:     []float32{1},
		ZeroPoints: []float32{0},
	}
	if _, err := DequantizePackedWeights(packed); err == nil || !core.Contains(err.Error(), "bits") {
		t.Fatalf("DequantizePackedWeights(bad bits) error = %v, want bits diagnostic", err)
	}
}

// TestPackedCov_ValidatePackedWeights_BadZeroPointCount covers the
// zero-point-count mismatch branch (distinct from the scale-count branch the
// existing ugly test exercises).
func TestPackedCov_ValidatePackedWeights_BadZeroPointCount(t *testing.T) {
	packed := PackedWeights{
		Bits:       4,
		GroupSize:  32,
		Shape:      []int32{1, 4},
		Packed:     []byte{0, 0},
		Scales:     []float32{1}, // correct: 1 group
		ZeroPoints: nil,          // wrong: expected 1, supplied 0
		QMin:       -8,
		QMax:       7,
	}
	if _, err := DequantizePackedWeights(packed); err == nil || !core.Contains(err.Error(), "zero-point count") {
		t.Fatalf("DequantizePackedWeights(bad zero-point count) error = %v, want zero-point count diagnostic", err)
	}
}

// TestPackedCov_PackedShapeElements_Edges covers packedShapeElements' empty
// shape, non-positive dimension, and overflow guards via the two public callers.
func TestPackedCov_PackedShapeElements_Edges(t *testing.T) {
	t.Run("EmptyShape", func(t *testing.T) {
		// An empty shape has no element count; PackQuantizedWeights surfaces the
		// shape-required diagnostic through validatePackedShape.
		q := QuantizedWeights{Bits: 4, GroupSize: 32, QValues: []int16{0}, Scales: []float32{1}, ZeroPoints: []float32{0}}
		if _, err := PackQuantizedWeights(q, []int32{}); err == nil || !core.Contains(err.Error(), "shape is required") {
			t.Fatalf("PackQuantizedWeights(empty shape) error = %v, want shape-required diagnostic", err)
		}
	})
	t.Run("NonPositiveDim", func(t *testing.T) {
		// A zero dimension cannot describe a tensor; the dequant path validates the
		// packed shape and rejects it.
		packed := PackedWeights{Bits: 4, GroupSize: 32, Shape: []int32{0, 4}, Packed: []byte{0}, Scales: []float32{1}, ZeroPoints: []float32{0}}
		if _, err := DequantizePackedWeights(packed); err == nil || !core.Contains(err.Error(), "dimensions must be positive") {
			t.Fatalf("DequantizePackedWeights(zero dim) error = %v, want positive-dimension diagnostic", err)
		}
	})
	t.Run("Overflow", func(t *testing.T) {
		// Two near-max int32 dimensions overflow the int element product; the
		// guard rejects the shape before allocating.
		packed := PackedWeights{Bits: 4, GroupSize: 32, Shape: []int32{1 << 30, 1 << 30, 1 << 30}, Packed: []byte{0}, Scales: []float32{1}, ZeroPoints: []float32{0}}
		if _, err := DequantizePackedWeights(packed); err == nil || !core.Contains(err.Error(), "too large") {
			t.Fatalf("DequantizePackedWeights(overflow shape) error = %v, want too-large diagnostic", err)
		}
	})
}
