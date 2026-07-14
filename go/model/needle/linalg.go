// SPDX-Licence-Identifier: EUPL-1.2

package needle

import "math"

// linearNoBias applies a bias-free linear layer y = x · Wᵀ. The weight is stored
// PyTorch-style, row-major [outDim, inDim], so output o is the dot product of x
// with row o. Every projection in Needle (q/k/v/out, lm_head) is bias-free.
//
//	y := linearNoBias([]float32{1, 2}, w /*[3*2]*/, 3, 2) // len(y) == 3
func linearNoBias(x, weight []float32, outDim, inDim int) []float32 {
	out := make([]float32, outDim)
	for o := range outDim {
		row := weight[o*inDim : o*inDim+inDim]
		var acc float32
		for i, xv := range x {
			acc += xv * row[i]
		}
		out[o] = acc
	}
	return out
}

// linearRows applies linearNoBias to each row of a [rows, inDim] matrix, yielding
// a [rows, outDim] matrix (both flattened row-major).
func linearRows(x []float32, rows int, weight []float32, outDim, inDim int) []float32 {
	out := make([]float32, rows*outDim)
	for r := range rows {
		y := linearNoBias(x[r*inDim:r*inDim+inDim], weight, outDim, inDim)
		copy(out[r*outDim:r*outDim+outDim], y)
	}
	return out
}

// softmaxInPlace turns scores into a probability distribution over the whole
// slice, subtracting the max first for numerical stability.
func softmaxInPlace(scores []float32) {
	maxV := float32(math.Inf(-1))
	for _, v := range scores {
		if v > maxV {
			maxV = v
		}
	}
	var sum float32
	for i, v := range scores {
		e := float32(math.Exp(float64(v - maxV)))
		scores[i] = e
		sum += e
	}
	if sum == 0 {
		return
	}
	for i := range scores {
		scores[i] /= sum
	}
}

// sigmoid is the scalar logistic function used by every residual gate.
func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}

// clipAdd mirrors the reference's _add_clipped: a + b clamped to the bf16 range.
// In f32 the clamp never fires for a 26M model, but keeping it makes the formula
// identical to the source rather than "close enough".
func clipAdd(a, b float32) float32 {
	s := a + b
	if s > 65500 {
		return 65500
	}
	if s < -65500 {
		return -65500
	}
	return s
}
