// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import core "dappco.re/go"

func rocmReferenceLoRAProjection(input, baseWeights, loraA, loraB []float32, rows, cols, rank int, alpha float32, bias []float32) ([]float32, error) {
	if rank <= 0 {
		return nil, core.E("rocm.LoRA.ReferenceProjection", "rank must be positive", nil)
	}
	if !hipQ8ScaleIsPositiveFinite(alpha) {
		return nil, core.E("rocm.LoRA.ReferenceProjection", "alpha must be positive and finite", nil)
	}
	if err := validateHIPProjectionShape(len(input), len(baseWeights), len(bias), rows, cols); err != nil {
		return nil, err
	}
	if len(loraA) != rank*cols {
		return nil, core.E("rocm.LoRA.ReferenceProjection", core.Sprintf("LoRA A length %d does not match rank*cols %d", len(loraA), rank*cols), nil)
	}
	if len(loraB) != rows*rank {
		return nil, core.E("rocm.LoRA.ReferenceProjection", core.Sprintf("LoRA B length %d does not match rows*rank %d", len(loraB), rows*rank), nil)
	}

	output, err := hipReferenceFP32Projection(input, baseWeights, rows, cols, bias)
	if err != nil {
		return nil, err
	}
	down := make([]float32, rank)
	for r := 0; r < rank; r++ {
		for c := 0; c < cols; c++ {
			down[r] += loraA[r*cols+c] * input[c]
		}
	}
	scale := alpha / float32(rank)
	for row := 0; row < rows; row++ {
		delta := float32(0)
		for r := 0; r < rank; r++ {
			delta += loraB[row*rank+r] * down[r]
		}
		output[row] += scale * delta
	}
	return output, nil
}

func hipReferenceFP32Projection(input, weights []float32, rows, cols int, bias []float32) ([]float32, error) {
	if err := validateHIPProjectionShape(len(input), len(weights), len(bias), rows, cols); err != nil {
		return nil, err
	}
	output := make([]float32, rows)
	for row := 0; row < rows; row++ {
		sum := float32(0)
		if len(bias) > 0 {
			sum = bias[row]
		}
		for col := 0; col < cols; col++ {
			sum += input[col] * weights[row*cols+col]
		}
		output[row] = sum
	}
	return output, nil
}
