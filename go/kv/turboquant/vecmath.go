// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import "math"

// toFloat64 widens a float32 row to float64 for accumulation. Every codec in
// this package does its arithmetic in float64 (RFC: "float64 accumulation in
// the math") even though rows arrive and leave as float32 — the natural K/V
// row representation.
//
//	xs := toFloat64([]float32{1, -2, 0.5})
func toFloat64(x []float32) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = float64(v)
	}
	return out
}

// toFloat32 narrows a float64 row back to float32 for the codec's public
// Decode surface.
//
//	xs := toFloat32([]float64{1, -2, 0.5})
func toFloat32(x []float64) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(v)
	}
	return out
}

// l2Norm returns the Euclidean norm ||x||₂, accumulated in float64.
//
//	l2Norm([]float64{3, 4}) // 5
func l2Norm(x []float64) float64 {
	var sumSq float64
	for _, v := range x {
		sumSq += v * v
	}
	return math.Sqrt(sumSq)
}

// scaled returns x * s as a new slice; x is never mutated.
//
//	scaled([]float64{1, 2}, 0.5) // []float64{0.5, 1}
func scaled(x []float64, s float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = v * s
	}
	return out
}

// subtract returns a - b element-wise as a new slice. a and b must be the
// same length; a mismatched length panics rather than silently truncating —
// a caller passing rows of different dimension is a bug at the call site.
//
//	subtract([]float64{3, 5}, []float64{1, 1}) // []float64{2, 4}
func subtract(a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("turboquant: subtract requires equal-length vectors")
	}
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] - b[i]
	}
	return out
}

// add returns a + b element-wise as a new slice. Same length contract as
// subtract.
//
//	add([]float64{1, 2}, []float64{3, 4}) // []float64{4, 6}
func add(a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("turboquant: add requires equal-length vectors")
	}
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// dot returns the inner product <a, b>, accumulated in float64. Same length
// contract as subtract.
//
//	dot([]float64{1, 2, 3}, []float64{4, 5, 6}) // 32
func dot(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("turboquant: dot requires equal-length vectors")
	}
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// softmax returns a new slice holding the softmax of scores, computed with
// the standard max-subtraction for numerical stability. An empty input
// returns an empty output.
//
//	softmax([]float64{1, 2, 3}) // sums to 1, monotone with input
func softmax(scores []float64) []float64 {
	out := make([]float64, len(scores))
	if len(scores) == 0 {
		return out
	}
	max := scores[0]
	for _, v := range scores[1:] {
		if v > max {
			max = v
		}
	}
	var sum float64
	for i, v := range scores {
		e := math.Exp(v - max)
		out[i] = e
		sum += e
	}
	if sum == 0 {
		return out
	}
	inv := 1 / sum
	for i := range out {
		out[i] *= inv
	}
	return out
}
