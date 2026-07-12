// SPDX-Licence-Identifier: EUPL-1.2

// Package gptq writes the Hugging Face GPTQ tensor layout.
package gptq

import (
	"math"
	"sort"

	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/autoround"
)

// Options controls GPTQ weight-only quantisation.
type Options struct {
	Bits      int
	GroupSize int
	Symmetric bool
	DescAct   bool
	// Hessian is the row-major activation Hessian for the input features. When
	// omitted, the writer uses RTN because weights alone cannot recover it.
	Hessian     []float64
	DampPercent float64
}

// Tensor is one quantised rank-two weight in the HF GPTQ layout. Shape is the
// original [out_features, in_features] shape; packed shapes follow AutoGPTQ:
// qweight [in_features/(32/bits), out_features], qzeros
// [in_features/group_size, out_features/(32/bits)], scales
// [in_features/group_size, out_features], and g_idx [in_features].
type Tensor struct {
	Shape        [2]int
	Bits         int
	GroupSize    int
	Symmetric    bool
	QWeight      []uint32
	QZeros       []uint32
	Scales       []float32
	GIdx         []int32
	QWeightShape [2]int
	QZerosShape  [2]int
	ScalesShape  [2]int
	MaxScale     float32
}

// Quantize converts a row-major [out_features, in_features] float matrix to
// the GPTQ safetensors representation. The grouped affine parameters come from
// autoround's numeric core; GPTQ-specific transposition and word packing live
// here. With no calibration Hessian, this is the standard GPTQ RTN fallback.
func Quantize(values []float32, rows, columns int, opts Options) (Tensor, error) {
	opts = normaliseOptions(opts)
	if rows <= 0 || columns <= 0 || len(values) != rows*columns {
		return Tensor{}, core.NewError("gptq: values must match a non-empty matrix shape")
	}
	pack := 32 / opts.Bits
	if 32%opts.Bits != 0 || rows%pack != 0 || columns%pack != 0 {
		return Tensor{}, core.Errorf("gptq: rows and columns must be divisible by %d for %d-bit packing", pack, opts.Bits)
	}
	if columns%opts.GroupSize != 0 {
		return Tensor{}, core.NewError("gptq: input columns must be divisible by group size")
	}
	if len(opts.Hessian) != 0 {
		var err error
		values, err = hessianOrdered(values, rows, columns, opts)
		if err != nil {
			return Tensor{}, err
		}
	}
	groups := columns / opts.GroupSize
	out := Tensor{
		Shape: [2]int{rows, columns}, Bits: opts.Bits, GroupSize: opts.GroupSize, Symmetric: opts.Symmetric,
		QWeightShape: [2]int{columns / pack, rows}, QZerosShape: [2]int{groups, rows / pack}, ScalesShape: [2]int{groups, rows},
		QWeight: make([]uint32, columns/pack*rows), QZeros: make([]uint32, groups*rows/pack), Scales: make([]float32, groups*rows), GIdx: make([]int32, columns),
	}
	quantized := make([]uint32, len(values))
	zeros := make([]uint32, groups*rows)
	for column := range columns {
		out.GIdx[column] = int32(column / opts.GroupSize)
	}
	for group := range groups {
		start := group * opts.GroupSize
		for row := range rows {
			segment := make([]float32, opts.GroupSize)
			copy(segment, values[row*columns+start:row*columns+start+opts.GroupSize])
			q, err := autoround.QuantizeWeights(segment, autoround.QuantizeConfig{Bits: opts.Bits, GroupSize: opts.GroupSize, Symmetric: opts.Symmetric})
			if err != nil {
				return Tensor{}, core.E("gptq.Quantize", "quantise affine group", err)
			}
			scale := q.Scales[0]
			out.Scales[group*rows+row] = scale
			out.MaxScale = max(out.MaxScale, scale)
			zero := int(q.ZeroPoints[0])
			if opts.Symmetric {
				zero = 1 << (opts.Bits - 1)
			}
			zeros[group*rows+row] = uint32(zero)
			for i, value := range q.QValues {
				code := int(value)
				if opts.Symmetric {
					code += zero
				}
				quantized[row*columns+start+i] = uint32(code)
			}
		}
	}
	mask := uint32((1 << opts.Bits) - 1)
	for packedColumn := range columns / pack {
		for row := range rows {
			var word uint32
			for slot := range pack {
				word |= (quantized[row*columns+packedColumn*pack+slot] & mask) << (slot * opts.Bits)
			}
			out.QWeight[packedColumn*rows+row] = word
		}
	}
	for group := range groups {
		for packedRow := range rows / pack {
			var word uint32
			for slot := range pack {
				zero := zeros[group*rows+packedRow*pack+slot]
				word |= ((zero - 1) & mask) << (slot * opts.Bits)
			}
			out.QZeros[group*(rows/pack)+packedRow] = word
		}
	}
	return out, nil
}

// hessianOrdered implements GPTQ's column-wise second-order error update. It
// follows IST-DASLab/gptq quant.py fasterquant (Apache-2.0):
// https://github.com/IST-DASLab/gptq/blob/main/quant.py . The caller must supply
// the activation Hessian; a bf16 checkpoint by itself contains no calibration
// activations, so Quantize deliberately falls back to RTN when Hessian is empty.
func hessianOrdered(values []float32, rows, columns int, opts Options) ([]float32, error) {
	if len(opts.Hessian) != columns*columns {
		return nil, core.NewError("gptq: Hessian shape must be in_features squared")
	}
	h := append([]float64(nil), opts.Hessian...)
	damp := opts.DampPercent
	if damp == 0 {
		damp = 0.01
	}
	var diagonalMean float64
	for i := range columns {
		diagonalMean += h[i*columns+i]
	}
	diagonalMean /= float64(columns)
	for i := range columns {
		h[i*columns+i] += damp * diagonalMean
	}
	order := make([]int, columns)
	for i := range order {
		order[i] = i
	}
	if opts.DescAct {
		sort.Slice(order, func(i, j int) bool { return h[order[i]*columns+order[i]] > h[order[j]*columns+order[j]] })
	}
	permutedH := make([]float64, len(h))
	working := make([]float64, len(values))
	for i, originalI := range order {
		for j, originalJ := range order {
			permutedH[i*columns+j] = h[originalI*columns+originalJ]
		}
		for row := range rows {
			working[row*columns+i] = float64(values[row*columns+originalI])
		}
	}
	inverse, err := invert(permutedH, columns)
	if err != nil {
		return nil, core.E("gptq.Quantize", "invert Hessian", err)
	}
	hinv, err := choleskyUpper(inverse, columns)
	if err != nil {
		return nil, core.E("gptq.Quantize", "factor inverse Hessian", err)
	}
	quantized := make([]float64, len(working))
	qmax := float64((int64(1) << (opts.Bits - 1)) - 1)
	qmin := -float64(int64(1) << (opts.Bits - 1))
	for row := range rows {
		for groupStart := 0; groupStart < columns; groupStart += opts.GroupSize {
			maxAbs := 0.0
			for column := groupStart; column < groupStart+opts.GroupSize; column++ {
				maxAbs = math.Max(maxAbs, math.Abs(working[row*columns+column]))
			}
			scale := maxAbs / qmax
			if scale == 0 {
				scale = 1
			}
			for column := groupStart; column < groupStart+opts.GroupSize; column++ {
				value := working[row*columns+column]
				q := math.Max(qmin, math.Min(qmax, math.Round(value/scale))) * scale
				quantized[row*columns+column] = q
				diagonal := hinv[column*columns+column]
				if diagonal == 0 {
					return nil, core.NewError("gptq: inverse Hessian has zero diagonal")
				}
				errorValue := (value - q) / diagonal
				for later := column + 1; later < columns; later++ {
					working[row*columns+later] -= errorValue * hinv[column*columns+later]
				}
			}
		}
	}
	out := make([]float32, len(values))
	for column, original := range order {
		for row := range rows {
			out[row*columns+original] = float32(quantized[row*columns+column])
		}
	}
	return out, nil
}

func invert(matrix []float64, n int) ([]float64, error) {
	augmented := make([]float64, n*n*2)
	for row := range n {
		copy(augmented[row*2*n:row*2*n+n], matrix[row*n:(row+1)*n])
		augmented[row*2*n+n+row] = 1
	}
	for pivot := range n {
		best := pivot
		for row := pivot + 1; row < n; row++ {
			if math.Abs(augmented[row*2*n+pivot]) > math.Abs(augmented[best*2*n+pivot]) {
				best = row
			}
		}
		if math.Abs(augmented[best*2*n+pivot]) < 1e-15 {
			return nil, core.NewError("gptq: singular Hessian")
		}
		for column := range 2 * n {
			augmented[pivot*2*n+column], augmented[best*2*n+column] = augmented[best*2*n+column], augmented[pivot*2*n+column]
		}
		divisor := augmented[pivot*2*n+pivot]
		for column := range 2 * n {
			augmented[pivot*2*n+column] /= divisor
		}
		for row := range n {
			if row == pivot {
				continue
			}
			factor := augmented[row*2*n+pivot]
			for column := range 2 * n {
				augmented[row*2*n+column] -= factor * augmented[pivot*2*n+column]
			}
		}
	}
	out := make([]float64, n*n)
	for row := range n {
		copy(out[row*n:(row+1)*n], augmented[row*2*n+n:(row+1)*2*n])
	}
	return out, nil
}

func choleskyUpper(matrix []float64, n int) ([]float64, error) {
	lower := make([]float64, n*n)
	for row := range n {
		for column := 0; column <= row; column++ {
			sum := matrix[row*n+column]
			for k := 0; k < column; k++ {
				sum -= lower[row*n+k] * lower[column*n+k]
			}
			if row == column {
				if sum <= 0 {
					return nil, core.NewError("gptq: inverse Hessian is not positive definite")
				}
				lower[row*n+column] = math.Sqrt(sum)
			} else {
				lower[row*n+column] = sum / lower[column*n+column]
			}
		}
	}
	upper := make([]float64, n*n)
	for row := range n {
		for column := row; column < n; column++ {
			upper[row*n+column] = lower[column*n+row]
		}
	}
	return upper, nil
}

// Dequantize reconstructs the row-major float matrix from HF GPTQ tensors.
func Dequantize(tensor Tensor) ([]float32, error) {
	rows, columns := tensor.Shape[0], tensor.Shape[1]
	if rows <= 0 || columns <= 0 || tensor.Bits <= 0 || tensor.GroupSize <= 0 {
		return nil, core.NewError("gptq: invalid tensor geometry")
	}
	pack := 32 / tensor.Bits
	groups := columns / tensor.GroupSize
	if len(tensor.QWeight) != columns/pack*rows || len(tensor.QZeros) != groups*rows/pack || len(tensor.Scales) != groups*rows || len(tensor.GIdx) != columns {
		return nil, core.NewError("gptq: packed tensor lengths do not match geometry")
	}
	mask := uint32((1 << tensor.Bits) - 1)
	out := make([]float32, rows*columns)
	for row := range rows {
		for column := range columns {
			group := int(tensor.GIdx[column])
			word := tensor.QWeight[(column/pack)*rows+row]
			q := (word >> ((column % pack) * tensor.Bits)) & mask
			zeroWord := tensor.QZeros[group*(rows/pack)+row/pack]
			zero := ((zeroWord >> ((row % pack) * tensor.Bits)) & mask) + 1
			out[row*columns+column] = (float32(q) - float32(zero)) * tensor.Scales[group*rows+row]
		}
	}
	return out, nil
}

func normaliseOptions(opts Options) Options {
	if opts.Bits == 0 {
		opts.Bits = 4
	}
	if opts.GroupSize == 0 {
		opts.GroupSize = 128
	}
	if !opts.Symmetric {
		// GPTQ's widely consumed default is symmetric W4 with an unsigned midpoint.
		opts.Symmetric = true
	}
	return opts
}
