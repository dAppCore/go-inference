// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"math"

	core "dappco.re/go"
)

func hipReferenceFP16Projection(input []float32, weights []uint16, rows, cols int, bias []float32) ([]float32, error) {
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
			sum += input[col] * hipFloat16ToFloat32(weights[row*cols+col])
		}
		output[row] = sum
	}
	return output, nil
}

func hipReferenceBF16Projection(input []float32, weights []uint16, rows, cols int, bias []float32) ([]float32, error) {
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
			sum += input[col] * hipBFloat16ToFloat32(weights[row*cols+col])
		}
		output[row] = sum
	}
	return output, nil
}

func hipReferenceF32Projection(input []float32, weights []float32, rows, cols int, bias []float32) ([]float32, error) {
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

func hipReferenceQ8Projection(input []float32, weights []int8, scale float32, rows, cols int, bias []float32) ([]float32, error) {
	if !hipQ8ScaleIsPositiveFinite(scale) {
		return nil, core.E("rocm.hip.ReferenceQ8Projection", "scale must be positive and finite", nil)
	}
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
			sum += input[col] * float32(weights[row*cols+col]) * scale
		}
		output[row] = sum
	}
	return output, nil
}

func hipReferenceMLXQ4Projection(input []float32, weights []uint32, scales []uint16, biases []uint16, rows, cols, groupSize int) ([]float32, error) {
	return hipReferenceMLXAffineProjection(input, weights, scales, biases, rows, cols, groupSize, hipMLXQ4ProjectionBits)
}

func hipReferenceMLXAffineProjection(input []float32, weights []uint32, scales []uint16, biases []uint16, rows, cols, groupSize, bits int) ([]float32, error) {
	if err := validateHIPMLXAffineProjectionShape(len(input), len(weights), len(scales), len(biases), rows, cols, groupSize, bits); err != nil {
		return nil, err
	}
	packedPerRow, err := hipMLXAffinePackedCols(cols, bits)
	if err != nil {
		return nil, err
	}
	groupsPerRow := cols / groupSize
	output := make([]float32, rows)
	for row := 0; row < rows; row++ {
		sum := float32(0)
		for col := 0; col < cols; col++ {
			quantized, err := hipMLXAffineUnpackValue(weights[row*packedPerRow:], col, bits)
			if err != nil {
				return nil, err
			}
			group := row*groupsPerRow + col/groupSize
			weight := float32(quantized)*hipBFloat16ToFloat32(scales[group]) + hipBFloat16ToFloat32(biases[group])
			sum += input[col] * weight
		}
		output[row] = sum
	}
	return output, nil
}

func validateHIPMLXQ4ProjectionShape(inputLen, weightLen, scaleLen, biasLen, rows, cols, groupSize int) error {
	return validateHIPMLXAffineProjectionShape(inputLen, weightLen, scaleLen, biasLen, rows, cols, groupSize, hipMLXQ4ProjectionBits)
}

func validateHIPMLXAffineProjectionShape(inputLen, weightLen, scaleLen, biasLen, rows, cols, groupSize, bits int) error {
	if rows <= 0 || cols <= 0 || groupSize <= 0 {
		return core.E("rocm.hip.ReferenceMLXQ4Projection", "rows, cols, and group size must be positive", nil)
	}
	packedPerRow, err := hipMLXAffinePackedCols(cols, bits)
	if err != nil {
		return err
	}
	if cols%groupSize != 0 {
		return core.E("rocm.hip.ReferenceMLXQ4Projection", "cols must be divisible by group size", nil)
	}
	if inputLen != cols {
		return core.E("rocm.hip.ReferenceMLXQ4Projection", core.Sprintf("input length %d does not match cols %d", inputLen, cols), nil)
	}
	if weightLen != rows*packedPerRow {
		return core.E("rocm.hip.ReferenceMLXQ4Projection", core.Sprintf("weight length %d does not match rows*packed_cols %d", weightLen, rows*packedPerRow), nil)
	}
	groupCount := rows * (cols / groupSize)
	if scaleLen != groupCount || biasLen != groupCount {
		return core.E("rocm.hip.ReferenceMLXQ4Projection", core.Sprintf("scale/bias length %d/%d does not match row groups %d", scaleLen, biasLen, groupCount), nil)
	}
	return nil
}

func hipMLXQ4ProjectionBitsOrDefault(bits int) int {
	if bits == 0 {
		return hipMLXQ4ProjectionBits
	}
	return bits
}

func hipMLXAffineSupportedBits(bits int) bool {
	switch hipMLXQ4ProjectionBitsOrDefault(bits) {
	case 4, 6, 8:
		return true
	default:
		return false
	}
}

func hipMLXAffinePackedCols(cols, bits int) (int, error) {
	bits = hipMLXQ4ProjectionBitsOrDefault(bits)
	if !hipMLXAffineSupportedBits(bits) {
		return 0, core.E("rocm.hip.ReferenceMLXQ4Projection", "only 4-, 6-, and 8-bit MLX affine projection is supported", nil)
	}
	if cols <= 0 {
		return 0, core.E("rocm.hip.ReferenceMLXQ4Projection", "cols must be positive", nil)
	}
	totalBits := uint64(cols) * uint64(bits)
	if totalBits%32 != 0 {
		return 0, core.E("rocm.hip.ReferenceMLXQ4Projection", "cols*bits must be divisible by 32 for MLX affine packing", nil)
	}
	packed := totalBits / 32
	if packed > uint64(int(^uint(0)>>1)) {
		return 0, core.E("rocm.hip.ReferenceMLXQ4Projection", "packed column count is out of int range", nil)
	}
	return int(packed), nil
}

func hipMLXAffineColsFromPackedCols(packedCols, bits int) (int, error) {
	bits = hipMLXQ4ProjectionBitsOrDefault(bits)
	if !hipMLXAffineSupportedBits(bits) {
		return 0, core.E("rocm.hip.ReferenceMLXQ4Projection", "only 4-, 6-, and 8-bit MLX affine projection is supported", nil)
	}
	if packedCols <= 0 {
		return 0, core.E("rocm.hip.ReferenceMLXQ4Projection", "packed column count must be positive", nil)
	}
	totalBits := uint64(packedCols) * 32
	if totalBits%uint64(bits) != 0 {
		return 0, core.E("rocm.hip.ReferenceMLXQ4Projection", "packed columns do not align with MLX affine bit width", nil)
	}
	cols := totalBits / uint64(bits)
	if cols > uint64(int(^uint(0)>>1)) {
		return 0, core.E("rocm.hip.ReferenceMLXQ4Projection", "logical column count is out of int range", nil)
	}
	return int(cols), nil
}

func hipMLXAffineUnpackValue(rowWeights []uint32, col, bits int) (uint32, error) {
	bits = hipMLXQ4ProjectionBitsOrDefault(bits)
	if !hipMLXAffineSupportedBits(bits) {
		return 0, core.E("rocm.hip.ReferenceMLXQ4Projection", "only 4-, 6-, and 8-bit MLX affine projection is supported", nil)
	}
	if col < 0 {
		return 0, core.E("rocm.hip.ReferenceMLXQ4Projection", "column must be non-negative", nil)
	}
	bitOffset := uint64(col) * uint64(bits)
	wordIndex := int(bitOffset / 32)
	shift := uint(bitOffset % 32)
	if wordIndex < 0 || wordIndex >= len(rowWeights) {
		return 0, core.E("rocm.hip.ReferenceMLXQ4Projection", "packed column is outside row weights", nil)
	}
	value := rowWeights[wordIndex] >> shift
	if shift+uint(bits) > 32 {
		if wordIndex+1 >= len(rowWeights) {
			return 0, core.E("rocm.hip.ReferenceMLXQ4Projection", "packed value crosses row boundary", nil)
		}
		value |= rowWeights[wordIndex+1] << (32 - shift)
	}
	return value & ((uint32(1) << uint(bits)) - 1), nil
}

func validateHIPProjectionShape(inputLen, weightLen, biasLen, rows, cols int) error {
	if rows <= 0 || cols <= 0 {
		return core.E("rocm.hip.ReferenceProjection", "rows and cols must be positive", nil)
	}
	if inputLen != cols {
		return core.E("rocm.hip.ReferenceProjection", core.Sprintf("input length %d does not match cols %d", inputLen, cols), nil)
	}
	if weightLen != rows*cols {
		return core.E("rocm.hip.ReferenceProjection", core.Sprintf("weight length %d does not match rows*cols %d", weightLen, rows*cols), nil)
	}
	if biasLen != 0 && biasLen != rows {
		return core.E("rocm.hip.ReferenceProjection", core.Sprintf("bias length %d does not match rows %d", biasLen, rows), nil)
	}
	return nil
}

func hipFloat16ToFloat32(value uint16) float32 {
	sign := uint32(value&0x8000) << 16
	exponent := int((value >> 10) & 0x1f)
	fraction := uint32(value & 0x03ff)
	switch exponent {
	case 0:
		if fraction == 0 {
			return math.Float32frombits(sign)
		}
		exponent = -14
		for fraction&0x0400 == 0 {
			fraction <<= 1
			exponent--
		}
		fraction &= 0x03ff
		return math.Float32frombits(sign | uint32(exponent+127)<<23 | fraction<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | fraction<<13)
	default:
		return math.Float32frombits(sign | uint32(exponent-15+127)<<23 | fraction<<13)
	}
}

func hipBFloat16ToFloat32(value uint16) float32 {
	return math.Float32frombits(uint32(value) << 16)
}
