// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
	"math"
)

// LayerNorm applies affine LayerNorm row-wise. GPT-2 uses this rather than RMSNorm.
func LayerNorm(values, weight, bias []float32, rows, width int, eps float32) error {
	if rows < 0 || width <= 0 || len(values) != rows*width || len(weight) != width || len(bias) != width || eps < 0 {
		return core.NewError("model.LayerNorm: invalid shape or epsilon")
	}
	for row := range rows {
		x := values[row*width : (row+1)*width]
		var mean float64
		for _, v := range x {
			mean += float64(v)
		}
		mean /= float64(width)
		var variance float64
		for _, v := range x {
			d := float64(v) - mean
			variance += d * d
		}
		inv := 1 / math.Sqrt(variance/float64(width)+float64(eps))
		for i := range x {
			x[i] = float32((float64(x[i])-mean)*inv*float64(weight[i]) + float64(bias[i]))
		}
	}
	return nil
}

// GELUNew applies the tanh-approximation used by GPT-2's gelu_new activation.
func GELUNew(values []float32) {
	const c = 0.7978845608028654
	for i, x := range values {
		y := float64(x)
		values[i] = float32(0.5 * y * (1 + math.Tanh(c*(y+0.044715*y*y*y))))
	}
}
