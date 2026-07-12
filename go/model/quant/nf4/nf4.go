// SPDX-Licence-Identifier: EUPL-1.2

// Package nf4 writes bitsandbytes NormalFloat4 blockwise weights.
package nf4

import (
	core "dappco.re/go"
	"math"
)

const BlockSize = 64

var Codebook = [16]float32{-1, -0.6961928, -0.52507305, -0.3949175, -0.28444138, -0.18477343, -0.09105004, 0, 0.0795803, 0.1609302, 0.2461123, 0.33791524, 0.44070983, 0.562617, 0.72295684, 1}

type Tensor struct {
	Data   []byte
	Absmax []float32
	Shape  []int
}

func Quantize(values []float32, shape []int) (Tensor, error) {
	if len(values) == 0 || product(shape) != len(values) {
		return Tensor{}, core.NewError("nf4: values must match a non-empty shape")
	}
	out := Tensor{Data: make([]byte, (len(values)+1)/2), Absmax: make([]float32, (len(values)+BlockSize-1)/BlockSize), Shape: append([]int(nil), shape...)}
	for block := range out.Absmax {
		start, end := block*BlockSize, min((block+1)*BlockSize, len(values))
		var maximum float32
		for _, value := range values[start:end] {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return Tensor{}, core.NewError("nf4: values must be finite")
			}
			maximum = max(maximum, float32(math.Abs(float64(value))))
		}
		if maximum == 0 {
			maximum = 1
		}
		out.Absmax[block] = maximum
		for i := start; i < end; i++ {
			code := nearest(values[i] / maximum)
			if i&1 == 0 {
				out.Data[i/2] = code << 4
			} else {
				out.Data[i/2] |= code
			}
		}
	}
	return out, nil
}

func Dequantize(tensor Tensor) ([]float32, error) {
	count := product(tensor.Shape)
	if count == 0 || len(tensor.Data) != (count+1)/2 || len(tensor.Absmax) != (count+BlockSize-1)/BlockSize {
		return nil, core.NewError("nf4: invalid tensor")
	}
	out := make([]float32, count)
	for i := range out {
		code := tensor.Data[i/2] & 15
		if i&1 == 0 {
			code = tensor.Data[i/2] >> 4
		}
		out[i] = Codebook[code] * tensor.Absmax[i/BlockSize]
	}
	return out, nil
}

func nearest(value float32) byte {
	best, distance := byte(0), float32(math.MaxFloat32)
	for i, candidate := range Codebook {
		if d := float32(math.Abs(float64(value - candidate))); d < distance {
			best, distance = byte(i), d
		}
	}
	return best
}
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
