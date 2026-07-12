// SPDX-Licence-Identifier: EUPL-1.2

// Package fp8 writes static per-tensor E4M3 Hugging Face compressed-tensors weights.
package fp8

import (
	"math"

	core "dappco.re/go"
)

const MaxE4M3 = float32(448)

type Tensor struct {
	Data  []byte
	Scale float32
}

func Quantize(values []float32) (Tensor, error) {
	if len(values) == 0 {
		return Tensor{}, core.NewError("fp8: values must not be empty")
	}
	var maximum float32
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return Tensor{}, core.NewError("fp8: values must be finite")
		}
		maximum = max(maximum, float32(math.Abs(float64(value))))
	}
	scale := maximum / MaxE4M3
	if scale == 0 {
		scale = 1
	}
	out := Tensor{Data: make([]byte, len(values)), Scale: scale}
	for i, value := range values {
		out.Data[i] = encodeE4M3(value / scale)
	}
	return out, nil
}

func Dequantize(tensor Tensor) ([]float32, error) {
	if len(tensor.Data) == 0 || tensor.Scale <= 0 {
		return nil, core.NewError("fp8: invalid tensor")
	}
	out := make([]float32, len(tensor.Data))
	for i, value := range tensor.Data {
		out[i] = decodeE4M3(value) * tensor.Scale
	}
	return out, nil
}

func encodeE4M3(value float32) byte {
	if value == 0 {
		return 0
	}
	sign := byte(0)
	if value < 0 {
		sign, value = 0x80, -value
	}
	if value >= MaxE4M3 {
		return sign | 0x7e
	}
	if value < 1.0/64 {
		mantissa := int(math.Round(float64(value * 512)))
		if mantissa < 0 {
			mantissa = 0
		}
		if mantissa > 7 {
			mantissa = 7
		}
		return sign | byte(mantissa)
	}
	exponent := int(math.Floor(math.Log2(float64(value))))
	mantissa := int(math.Round((float64(value)/math.Ldexp(1, exponent) - 1) * 8))
	if mantissa == 8 {
		exponent++
		mantissa = 0
	}
	biased := exponent + 7
	if biased >= 15 && mantissa > 6 {
		mantissa = 6
	}
	return sign | byte(biased<<3|mantissa)
}

func decodeE4M3(value byte) float32 {
	sign := float32(1)
	if value&0x80 != 0 {
		sign = -1
	}
	exponent, mantissa := (value>>3)&15, value&7
	if exponent == 0 {
		return sign * float32(mantissa) / 8 * (1.0 / 64)
	}
	// E4M3FN reserves only 0x7f/0xff for NaN and extends exponent 15 to 448.
	if exponent == 15 && mantissa == 7 {
		return sign * MaxE4M3
	}
	return sign * (1 + float32(mantissa)/8) * float32(math.Ldexp(1, int(exponent)-7))
}
