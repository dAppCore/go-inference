// SPDX-Licence-Identifier: EUPL-1.2

// Package awq writes the Hugging Face AWQ tensor layout.
package awq

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/autoround"
)

// Options controls AWQ weight-only quantisation.
type Options struct {
	Bits      int
	GroupSize int
	ZeroPoint bool
}

// Tensor is one quantised rank-two weight in the HF AWQ layout. Shape is the
// original [out_features, in_features] shape; packed shapes follow AutoAWQ:
// qweight [in_features, out_features/(32/bits)], qzeros
// [in_features/group_size, out_features/(32/bits)], and scales
// [in_features/group_size, out_features].
type Tensor struct {
	Shape        [2]int
	Bits         int
	GroupSize    int
	ZeroPoint    bool
	QWeight      []uint32
	QZeros       []uint32
	Scales       []float32
	QWeightShape [2]int
	QZerosShape  [2]int
	ScalesShape  [2]int
	MaxScale     float32
}

// Quantize converts a row-major [out_features, in_features] float matrix to
// the AWQ safetensors representation. The grouped affine parameters come from
// autoround's numeric core; AWQ-specific transposition and word packing live
// here. This data-free entry point uses AutoAWQ zero-point quantisation without
// claiming the activation calibration that a dense checkpoint cannot provide.
func Quantize(values []float32, rows, columns int, opts Options) (Tensor, error) {
	opts = normaliseOptions(opts)
	if rows <= 0 || columns <= 0 || len(values) != rows*columns {
		return Tensor{}, core.NewError("awq: values must match a non-empty matrix shape")
	}
	pack := 32 / opts.Bits
	if 32%opts.Bits != 0 || rows%pack != 0 || columns%pack != 0 {
		return Tensor{}, core.Errorf("awq: rows and columns must be divisible by %d for %d-bit packing", pack, opts.Bits)
	}
	if columns%opts.GroupSize != 0 {
		return Tensor{}, core.NewError("awq: input columns must be divisible by group size")
	}
	groups := columns / opts.GroupSize
	out := Tensor{
		Shape: [2]int{rows, columns}, Bits: opts.Bits, GroupSize: opts.GroupSize, ZeroPoint: opts.ZeroPoint,
		QWeightShape: [2]int{columns, rows / pack}, QZerosShape: [2]int{groups, rows / pack}, ScalesShape: [2]int{groups, rows},
		QWeight: make([]uint32, columns/pack*rows), QZeros: make([]uint32, groups*rows/pack), Scales: make([]float32, groups*rows),
	}
	quantized := make([]uint32, len(values))
	zeros := make([]uint32, groups*rows)
	for group := range groups {
		start := group * opts.GroupSize
		for row := range rows {
			segment := make([]float32, opts.GroupSize)
			copy(segment, values[row*columns+start:row*columns+start+opts.GroupSize])
			// This is AutoAWQ's zero-point pseudo_quantize_tensor layout. True AWQ
			// first derives equivalent per-channel transforms from calibration
			// activations; a checkpoint alone cannot recover those statistics.
			// Reference: casper-hansen/AutoAWQ awq/quantize/quantizer.py
			// (MIT): https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/quantizer.py
			q, err := autoround.QuantizeWeights(segment, autoround.QuantizeConfig{Bits: opts.Bits, GroupSize: opts.GroupSize, Symmetric: !opts.ZeroPoint})
			if err != nil {
				return Tensor{}, core.E("awq.Quantize", "quantise affine group", err)
			}
			scale := q.Scales[0]
			out.Scales[group*rows+row] = scale
			out.MaxScale = max(out.MaxScale, scale)
			zero := int(q.ZeroPoints[0])
			if !opts.ZeroPoint {
				zero = 1 << (opts.Bits - 1)
			}
			zero = max(0, min(zero, (1<<opts.Bits)-1))
			zeros[group*rows+row] = uint32(zero)
			for i, value := range q.QValues {
				code := int(value)
				if !opts.ZeroPoint {
					code += zero
				}
				quantized[row*columns+start+i] = uint32(code)
			}
		}
	}
	mask := uint32((1 << opts.Bits) - 1)
	order := awqOrder(pack)
	for column := range columns {
		for packedRow := range rows / pack {
			var word uint32
			for slot := range pack {
				row := packedRow*pack + order[slot]
				word |= (quantized[row*columns+column] & mask) << (slot * opts.Bits)
			}
			out.QWeight[column*(rows/pack)+packedRow] = word
		}
	}
	for group := range groups {
		for packedRow := range rows / pack {
			var word uint32
			for slot := range pack {
				row := packedRow*pack + order[slot]
				zero := zeros[group*rows+row]
				word |= (zero & mask) << (slot * opts.Bits)
			}
			out.QZeros[group*(rows/pack)+packedRow] = word
		}
	}
	return out, nil
}

// Dequantize reconstructs the row-major float matrix from HF AWQ tensors.
func Dequantize(tensor Tensor) ([]float32, error) {
	rows, columns := tensor.Shape[0], tensor.Shape[1]
	if rows <= 0 || columns <= 0 || tensor.Bits <= 0 || tensor.GroupSize <= 0 {
		return nil, core.NewError("awq: invalid tensor geometry")
	}
	pack := 32 / tensor.Bits
	groups := columns / tensor.GroupSize
	if len(tensor.QWeight) != columns*rows/pack || len(tensor.QZeros) != groups*rows/pack || len(tensor.Scales) != groups*rows {
		return nil, core.NewError("awq: packed tensor lengths do not match geometry")
	}
	mask := uint32((1 << tensor.Bits) - 1)
	out := make([]float32, rows*columns)
	order := awqOrder(pack)
	reverse := make([]int, pack)
	for slot, row := range order {
		reverse[row] = slot
	}
	for row := range rows {
		for column := range columns {
			group := column / tensor.GroupSize
			slot := reverse[row%pack]
			word := tensor.QWeight[column*(rows/pack)+row/pack]
			q := (word >> (slot * tensor.Bits)) & mask
			zeroWord := tensor.QZeros[group*(rows/pack)+row/pack]
			zero := (zeroWord >> (slot * tensor.Bits)) & mask
			out[row*columns+column] = (float32(q) - float32(zero)) * tensor.Scales[group*rows+row]
		}
	}
	return out, nil
}

func awqOrder(pack int) []int {
	if pack == 8 {
		return []int{0, 2, 4, 6, 1, 3, 5, 7}
	}
	order := make([]int, pack)
	for i := range order {
		order[i] = i
	}
	return order
}

func normaliseOptions(opts Options) Options {
	if opts.Bits == 0 {
		opts.Bits = 4
	}
	if opts.GroupSize == 0 {
		opts.GroupSize = 128
	}
	// AutoAWQ's broadly consumed GEMM layout uses asymmetric zero points.
	opts.ZeroPoint = true
	return opts
}
