// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model/attn"
	"github.com/tmc/apple/metal"
)

// composed_quant_backend.go binds the packed-projection quant matvec seam — attn.ProjQuantMatMulInto
// (the factory host path's projection op: arch_gated_attn.go's projQuantAttn and model/attn's
// gated-delta projections) — to the metallib's MLX affine BF16 kernels. A quant checkpoint keeps its
// 2-D projections PACKED on device; this is the op that serves them without widening a 27B weight to
// f32 (~110 GB). M=1 (decode) dispatches affine_qmv_bfloat16_t (the pooled-scratch decode hot path
// QMVBF16 already drives); M>1 (prompt prefill) dispatches affine_qmm_t_bfloat16_t, one weight pass
// scoring all M rows.
//
// bf16 activations + the checkpoint's own bf16 scales/biases (mlx_lm.convert writes them bf16): the
// serve boundary is bf16, and bf16's 8-bit mantissa is FINER than the 4-bit (or 2-bit) weights, so
// the weight quantisation — not the activation dtype — bounds the error. The composed engine's
// bindings that used to live beside this one were retired with it (#50).
func init() {
	attn.ProjQuantMatMulInto = MatMulQuantF32NTInto
}

// MatMulQuantF32NTInto computes out[M,N] = x[M,K] @ dequant(w)ᵀ for an MLX affine-packed weight with f32
// activations at the seam (converted to bf16 for the kernel). w is the packed uint32 codes + bf16 scales/
// biases (one scale+bias per group per row); N=outDim, K=inDim; groupSize divides K; bits a shipped kernel
// width (2/3/4/5/6/8). M=1 takes QMVBF16 (affine_qmv, pooled scratch); M>1 takes the affine_qmm_t slab. out
// is reused when cap(out) ≥ M*N. Byte-for-byte, this equals the host reference (DequantizeTensor + matNT)
// within bf16 activation tolerance — gated in composed_quant_backend_test.go.
func MatMulQuantF32NTInto(out, x []float32, packed, scales, biases []byte, M, K, N, groupSize, bits int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if M <= 0 || N <= 0 || K <= 0 {
		return nil, core.NewError("native.MatMulQuantF32NTInto: M, N, K must be positive")
	}
	if len(x) != M*K {
		return nil, core.NewError("native.MatMulQuantF32NTInto: len(x) must equal M*K")
	}
	if groupSize <= 0 || bits <= 0 || K%groupSize != 0 {
		return nil, core.NewError("native.MatMulQuantF32NTInto: invalid quant geometry")
	}
	xb := f32sToBF16Bytes(x)
	var ob []byte
	var err error
	if M == 1 {
		ob, err = QMVBF16(xb, packed, scales, biases, N, K, groupSize, bits)
	} else {
		ob, err = qmmTBF16Into(xb, packed, scales, biases, M, K, N, groupSize, bits)
	}
	if err != nil {
		return nil, err
	}
	res := bf16ToF32Slice(ob) // M*N
	if cap(out) >= M*N {
		out = out[:M*N]
		copy(out, res)
		return out, nil
	}
	return res, nil
}

// f32sToBF16Bytes rounds a float32 slice to bf16 bytes (round-to-nearest-even, the model's boundary dtype).
func f32sToBF16Bytes(x []float32) []byte {
	out := make([]byte, len(x)*bf16Size)
	for i, v := range x {
		r := f32ToBF16(v)
		out[2*i], out[2*i+1] = byte(r), byte(r>>8)
	}
	return out
}

// qmmTBF16Into encodes out[M,N] = x[M,K] @ dequant(w[N,K])ᵀ through MLX's affine_qmm_t_bfloat16_t — the
// prompt-prefill fold, one weight pass for all M rows (the per-row qmv would re-read the weights M times).
// x [M,K] bf16 uploads to a shared buffer; the packed weight/scales/biases bind resident (no-copy, cached);
// out [M,N] bf16 reads back from a shared result buffer. Same encoder (encQMMTBF16At) the arch prefill uses.
func qmmTBF16Into(x, packed, scales, biases []byte, M, K, N, groupSize, bits int) ([]byte, error) {
	pso, err := pipelineFor(qmmTKernelName(N, groupSize, bits))
	if err != nil {
		return nil, err
	}
	outBytes := M * N * bf16Size
	out := make([]byte, outBytes)
	withAutoreleasePool(func() {
		wBuf := residentBytes(packed)
		sBuf := residentBytes(scales)
		bBuf := residentBytes(biases)
		xBuf := sharedBytes(x)
		outBuf := device.NewBufferWithLengthOptions(uint(outBytes), metal.MTLResourceStorageModeShared)
		clear(unsafe.Slice((*byte)(outBuf.Contents()), outBytes)) // a partial N-tile (alN_false) leaves edge bytes untouched

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitQMMT(encSink{enc}, pso, wBuf, 0, sBuf, 0, bBuf, 0, xBuf, 0, outBuf, 0, M, N, K)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outBytes))
	})
	return out, nil
}
