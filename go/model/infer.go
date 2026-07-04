// SPDX-Licence-Identifier: EUPL-1.2

package model

import "dappco.re/go/inference/safetensors"

// infer.go is the engine's read-the-dimension-FROM-THE-WEIGHT-SHAPE rule, architecture-NEUTRAL: when a
// config omits a dimension, the model reads it from the actual weight rather than guessing. Each arch
// supplies its own weight NAMES + layer pattern (sliding-vs-global head dims, uniform attention, …);
// the shape arithmetic lives here ONCE so no architecture re-rolls it.

// WeightAny returns the first of names present in the tensor set.
func WeightAny(weights map[string]safetensors.Tensor, names ...string) (safetensors.Tensor, bool) {
	for _, n := range names {
		if t, ok := weights[n]; ok {
			return t, true
		}
	}
	return safetensors.Tensor{}, false
}

// InferHeadDim reads a head dim from a q-projection weight: rows ÷ numHeads (a q_proj is
// [numHeads·headDim × hidden], so its row count over the head count is the head dim). Returns 0 when the
// weight is absent or its rows don't divide evenly — the caller then keeps whatever the config declared.
func InferHeadDim(weights map[string]safetensors.Tensor, qProjName string, numHeads int) int {
	if qProj, ok := WeightAny(weights, qProjName); ok {
		shape := qProj.Shape
		if len(shape) > 0 && numHeads > 0 && shape[0]%numHeads == 0 {
			return shape[0] / numHeads
		}
	}
	return 0
}

// InferOutFeaturesPerN reads a projection's flattened out-features ÷ n — e.g. a per-layer projection
// stacked over n layers gives the per-layer width. Returns 0 when absent or it doesn't divide.
func InferOutFeaturesPerN(weights map[string]safetensors.Tensor, projName string, n int) int {
	if n <= 0 {
		return 0
	}
	if w, ok := WeightAny(weights, projName); ok {
		shape := w.Shape
		if len(shape) >= 2 {
			outFeatures := 1
			for _, dim := range shape[:len(shape)-1] {
				outFeatures *= dim
			}
			if outFeatures%n == 0 {
				return outFeatures / n
			}
		}
	}
	return 0
}
