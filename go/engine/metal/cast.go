// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "github.com/tmc/apple/metal"

// cast.go is the bf16↔fp32 conversion the dtype scheme (pkg/scheme.DType) implies but can't perform
// itself: the scheme registers bfloat16/float32 + their sizes, these move a tensor between them.
// bf16 is the top 16 bits of fp32 (same 8-bit exponent/range, 7 vs 23 mantissa bits), so widening
// bf16→fp32 is LOSSLESS and narrowing fp32→bf16 rounds once. They wrap MLX's contiguous v_copy cast
// kernels — the primitive a "store bf16, compute fp32" path needs. Verified by TestBF16F32CastRoundtrip.

// encWidenBF16ToF32 encodes a lossless bf16→fp32 widen of n elements (src bf16, dst fp32) into enc.
func encWidenBF16ToF32(enc metal.MTLComputeCommandEncoder, src, dst metal.MTLBuffer, n int) error {
	return encCopyCast(enc, "v_copybfloat16float32", src, dst, n)
}

// encNarrowF32ToBF16 encodes an fp32→bf16 narrow of n elements (round-to-nearest-even), src fp32, dst bf16.
func encNarrowF32ToBF16(enc metal.MTLComputeCommandEncoder, src, dst metal.MTLBuffer, n int) error {
	return encCopyCast(enc, "v_copyfloat32bfloat16", src, dst, n)
}

// encCopyCast dispatches one of MLX's contiguous v_copy cast kernels (src→dst, n elements).
func encCopyCast(enc metal.MTLComputeCommandEncoder, kernel string, src, dst metal.MTLBuffer, n int) error {
	pso, err := pipelineFor(kernel)
	if err != nil {
		return err
	}
	setPSO(enc, pso)
	setBuf(enc, src, 0, 0)
	setBuf(enc, dst, 0, 1)
	group := min(uint(n), uint(256))
	dispatchThreads(enc,
		metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
		metal.MTLSize{Width: group, Height: 1, Depth: 1},
	)
	return nil
}
