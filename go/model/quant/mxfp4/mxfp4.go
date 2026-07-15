// SPDX-Licence-Identifier: EUPL-1.2

// Package mxfp4 writes OCP Microscaling FP4 (MXFP4) blockwise weights: E2M1
// elements in 32-element blocks that share one E8M0 power-of-two scale. This
// is the AMD-native low-precision lane AMD Quark emits for its MX/FP4 output
// (see model/quant/quantfmt.go's MethodQuark, and model.QuantConfig's
// "mxfp4" mode in model/quant_config.go, which already validates
// group_size==32/bits==4 for this format).
//
// Bit layout is the OCP Microscaling Formats (MX) Specification v1.0
// (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf):
// each element is E2M1 (1 sign + 2 exponent + 1 mantissa bit, exponent bias
// 1), giving the 8 non-negative magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}; each
// block of 32 elements shares one E8M0 scale (an unsigned 8-bit exponent,
// bias 127, 0xFF reserved — https://sw23.github.io/fp-conv/formats/fp4-e2m1.html).
// Both the block size and the magnitude range are cross-checked against a
// real AMD Quark MXFP4 checkpoint
// (huggingface.co/amd/Qwen3.5-397B-A17B-MXFP4/blob/main/config.json:
// quantization_config.global_quant_config.weight = {dtype: "fp4", qscheme:
// "per_group", group_size: 32, scale_format: "e8m0"}). Neither E2M1 nor E8M0
// represent Inf/NaN — the format is finite-only.
//
//	q, _ := mxfp4.Quantize(weights, []int{64})
//	back, _ := mxfp4.Dequantize(q)
package mxfp4

import (
	"math"

	core "dappco.re/go"
)

// BlockSize is the number of elements that share one E8M0 scale — fixed at
// 32 by the OCP MX spec, and confirmed both in model.QuantConfig's mxfp4
// validation (model/quant_config.go) and in AMD Quark's emitted group_size.
const BlockSize = 32

// MaxE2M1 is the largest finite magnitude an E2M1 element can represent.
const MaxE2M1 = float32(6)

// magnitudes is the E2M1 non-negative value table (OCP MX spec v1.0): index
// is the 3-bit unsigned code (2 exponent bits + 1 mantissa bit, bias 1).
var magnitudes = [8]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}

// Tensor is a blockwise-quantised MXFP4 weight: one E2M1 nibble per element,
// two packed per byte (like nf4.Tensor), and one decoded E8M0 scale per
// BlockSize-element block.
type Tensor struct {
	Data  []byte
	Scale []float32
	Shape []int
}

// Quantize packs values into blockwise MXFP4. Each BlockSize-run gets its
// own E8M0 power-of-two scale — the smallest power of two that keeps the
// block's peak magnitude within E2M1's finite range (MaxE2M1) — and each
// element rounds to the nearest E2M1 code.
//
//	q, _ := mxfp4.Quantize([]float32{0, 1, 2, 3}, []int{4})
func Quantize(values []float32, shape []int) (Tensor, error) {
	if len(values) == 0 || product(shape) != len(values) {
		return Tensor{}, core.NewError("mxfp4: values must match a non-empty shape")
	}
	out := Tensor{
		Data:  make([]byte, (len(values)+1)/2),
		Scale: make([]float32, (len(values)+BlockSize-1)/BlockSize),
		Shape: append([]int(nil), shape...),
	}
	for block := range out.Scale {
		start, end := block*BlockSize, min((block+1)*BlockSize, len(values))
		var peak float32
		for _, value := range values[start:end] {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return Tensor{}, core.NewError("mxfp4: values must be finite")
			}
			peak = max(peak, float32(math.Abs(float64(value))))
		}
		scale := blockScale(peak)
		out.Scale[block] = scale
		for i := start; i < end; i++ {
			code := encodeE2M1(values[i] / scale)
			if i&1 == 0 {
				out.Data[i/2] = code << 4
			} else {
				out.Data[i/2] |= code
			}
		}
	}
	return out, nil
}

// Dequantize expands a blockwise MXFP4 Tensor back to float32.
//
//	values, _ := mxfp4.Dequantize(q)
func Dequantize(tensor Tensor) ([]float32, error) {
	count := product(tensor.Shape)
	if count == 0 || len(tensor.Data) != (count+1)/2 || len(tensor.Scale) != (count+BlockSize-1)/BlockSize {
		return nil, core.NewError("mxfp4: invalid tensor")
	}
	out := make([]float32, count)
	for i := range out {
		code := tensor.Data[i/2] & 15
		if i&1 == 0 {
			code = tensor.Data[i/2] >> 4
		}
		out[i] = decodeE2M1(code) * tensor.Scale[i/BlockSize]
	}
	return out, nil
}

// blockScale picks the E8M0 power-of-two scale for a block given its peak
// (largest absolute) value: the smallest power of two large enough that
// peak/scale never exceeds MaxE2M1, so the block's largest element always
// stays representable. peak==0 (an all-zero block) uses scale 1 rather than
// dividing by zero.
//
//	blockScale(6) == 1
func blockScale(peak float32) float32 {
	if peak == 0 {
		return 1
	}
	exp := clampExponent(int(math.Ceil(math.Log2(float64(peak) / float64(MaxE2M1)))))
	return float32(math.Ldexp(1, exp))
}

// clampExponent bounds an E8M0 exponent to its storable range. E8M0 is an
// unsigned 8-bit field biased by 127 with 0xFF reserved (NaN) per the OCP MX
// spec, so the representable signed exponent range is [-127, 127].
//
//	clampExponent(500) == 127
func clampExponent(exp int) int {
	switch {
	case exp < -127:
		return -127
	case exp > 127:
		return 127
	default:
		return exp
	}
}

// encodeE2M1 maps a value already divided by its block scale to the nearest
// E2M1 code: bit 3 is the sign, bits 2:0 index magnitudes. Values outside
// magnitudes' range clamp to the nearest (largest) representable magnitude
// rather than overflowing.
//
//	encodeE2M1(2) == 4
func encodeE2M1(value float32) byte {
	sign := byte(0)
	if value < 0 {
		sign, value = 0x8, -value
	}
	best, distance := byte(0), float32(math.MaxFloat32)
	for i, candidate := range magnitudes {
		if d := float32(math.Abs(float64(value - candidate))); d < distance {
			best, distance = byte(i), d
		}
	}
	return sign | best
}

// decodeE2M1 reverses encodeE2M1: bit 3 is the sign, bits 2:0 index
// magnitudes.
//
//	decodeE2M1(0xC) == -2
func decodeE2M1(code byte) float32 {
	value := magnitudes[code&0x7]
	if code&0x8 != 0 {
		return -value
	}
	return value
}

// product returns the element count a shape describes, 0 for an empty shape.
//
//	product([]int{5, 13}) == 65
func product(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	out := 1
	for _, value := range shape {
		out *= value
	}
	return out
}
