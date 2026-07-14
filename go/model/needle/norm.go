// SPDX-Licence-Identifier: EUPL-1.2

package needle

import "math"

// zcRMSNorm is Needle's "ZCRMSNorm" (zero-centred RMSNorm), copied field-for-field
// from NeedleRMSNorm.forward:
//
//	variance = mean(x²)              # NOTE: no mean subtraction — x is not centred
//	x        = x * rsqrt(var + eps)
//	return x * (1 + weight)          # the weight is stored zero-centred (init 0)
//
// The "zero-centred" in the name refers to the *weight* (the +1 shift, Gemma-style),
// not the activations — the activations are plain RMS-normalised. Getting this
// wrong (subtracting the mean, or dropping the +1) silently corrupts every layer.
//
//	out := zcRMSNorm([]float32{...}, weight, 1e-6) // len(out) == len(x)
func zcRMSNorm(x, weight []float32, eps float64) []float32 {
	n := len(x)
	var sumsq float64
	for _, v := range x {
		sumsq += float64(v) * float64(v)
	}
	variance := sumsq / float64(n)
	inv := float32(1.0 / math.Sqrt(variance+eps))
	out := make([]float32, n)
	for i := range n {
		out[i] = x[i] * inv * (1 + weight[i])
	}
	return out
}

// zcRMSNormRows applies zcRMSNorm independently to each row of a [rows, dim]
// matrix (flattened row-major). Used for the per-token layernorms.
func zcRMSNormRows(x []float32, rows, dim int, weight []float32, eps float64) []float32 {
	out := make([]float32, rows*dim)
	for r := range rows {
		row := zcRMSNorm(x[r*dim:r*dim+dim], weight, eps)
		copy(out[r*dim:r*dim+dim], row)
	}
	return out
}
