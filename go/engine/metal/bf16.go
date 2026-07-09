// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// This file holds the bfloat16 siblings of the float32 native ops — the kernels
// a bf16 attention block actually decodes through (bf16 is the real decode
// dtype). Each one drives the SAME MLX kernel as its float32 counterpart with an
// identical host ABI: only the kernel-name type token swaps (float32 → bfloat16)
// and buffers are 2 bytes/element instead of 4. The dispatch maths (element
// counts, element-strides, tile selection) is dtype-independent, so it is reused
// verbatim. Inputs and outputs are raw bf16 []byte, exactly like SDPA; byte-for-
// byte parity with the matching mlx-c op (on the same bf16 arrays) is gated in
// parity_test.go — anything that isn't bit-identical to mlx-c is a defect, not a
// rounding allowance.

// bf16Size is the byte width of a single bfloat16 element.
const bf16Size = 2

// RMSNormBF16 is the bfloat16 sibling of RMSNorm: it RMS-normalises the rows of
// x (raw bf16 bytes, row-major rows × axisSize) scaled by weight (raw bf16 bytes,
// length axisSize) and returns the result as bf16 bytes of the same shape. It
// drives MLX's rms kernel directly through the no-cgo path with the identical
// buffer ABI — x(0) weight(1) out(2) eps(3) axis_size(4) w_stride(5) — only the
// kernel name (rmsbfloat16) and the 2-byte element width differ. axisSize must
// stay ≤ 4096 so the single-row kernel is used (every gemma hidden size).
// Byte-for-byte parity with pkg/metal.RMSNorm on the same bf16 arrays is gated
// in parity_test.go.
//
//	out, err := native.RMSNormBF16(xBytes, wBytes, 4, 512, 1e-5)
func RMSNormBF16(x, weight []byte, rows, axisSize int, eps float32) ([]byte, error) {
	return RMSNormBF16Into(nil, x, weight, rows, axisSize, eps)
}

func RMSNormBF16Into(out []byte, x, weight []byte, rows, axisSize int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != rows*axisSize*bf16Size {
		return nil, core.NewError("native.RMSNormBF16: len(x) must equal rows*axisSize*2 bytes")
	}
	if len(weight) != axisSize*bf16Size {
		return nil, core.NewError("native.RMSNormBF16: len(weight) must equal axisSize*2 bytes")
	}
	if rows == 0 || axisSize == 0 {
		if cap(out) < len(x) {
			return make([]byte, len(x)), nil
		}
		return out[:len(x)], nil
	}
	pso, err := pipelineFor(rmsKernelBF16(axisSize))
	if err != nil {
		return nil, err
	}

	outLen := rows * axisSize * bf16Size
	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getQMVBF16Scratch(rows*axisSize, rows*axisSize)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVBF16Scratch(scratch)
		xBuf, outBuf, err := scratch.buffers(x)
		if err != nil {
			encErr = err
			return
		}
		wBuf := residentBytes(weight)
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}

		// single-row up to the limit, else the looped kernel (it grid-strides the axis).
		tgSize := rmsThreadgroup(axisSize, pso)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitRMSNormRows(encSink{enc}, pso, xBuf, wBuf, outBuf, 0, 0, 0, axisSize, eps, rows, tgSize)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		if !directOut {
			copy(out, scratch.out.bytes[:outLen])
		}
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

// MatVecBF16 is the bfloat16 sibling of MatVec: out = mat @ vec where mat is a
// row-major (outDim × inDim) matrix and vec has length inDim, all as raw bf16
// bytes, returning bf16 bytes of length outDim. It drives MLX's gemv kernel with
// the identical tile selection (gemvTiles) and buffer ABI as the float32 path —
// mat(0) vec(1) out(3) in_vec_size(4) out_vec_size(5) matrix_ld(6) batch_ndim(9)
// batch_shape(10) vec_stride(11) mat_stride(12) — only the kernel name token
// (gemv_bfloat16_…) and the 2-byte element width differ. Byte-for-byte parity
// with pkg/metal.Matmul of (outDim × inDim) @ (inDim × 1) on the same bf16 arrays
// is gated in parity_test.go.
//
//	out, err := native.MatVecBF16(matBytes, vecBytes, 512, 256)
func MatVecBF16(mat, vec []byte, outDim, inDim int) ([]byte, error) {
	return MatVecBF16Into(nil, mat, vec, outDim, inDim)
}

func MatVecBF16Into(out []byte, mat, vec []byte, outDim, inDim int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(mat) != outDim*inDim*bf16Size {
		return nil, core.NewError("native.MatVecBF16: len(mat) must equal outDim*inDim*2 bytes")
	}
	if len(vec) != inDim*bf16Size {
		return nil, core.NewError("native.MatVecBF16: len(vec) must equal inDim*2 bytes")
	}
	outLen := outDim * bf16Size
	if outDim == 0 || inDim == 0 {
		if cap(out) < outLen {
			return make([]byte, outLen), nil
		}
		return out[:outLen], nil
	}
	return MatVecBF16BufInto(out, bufView{buf: residentBytes(mat)}, vec, outDim, inDim)
}

// ropePSOCacheBF16 memoises the bf16 rope pipeline keyed by the function-constant
// combination (forward/traditional/transpose), mirroring ropePSOCache for the
// float32 path. A name alone doesn't identify the variant — the constants
// specialise the kernel at build time — so the key carries the traditional flag.
var (
	ropePSOBF16Mu    sync.Mutex
	ropePSOBF16Cache = map[string]metal.MTLComputePipelineState{}
)

const (
	ropeBF16Key            = "rope_single_bfloat16|trad=false"
	ropeBF16TraditionalKey = "rope_single_bfloat16|trad=true"
)

func ropePipelineBF16Key(traditional bool) string {
	if traditional {
		return ropeBF16TraditionalKey
	}
	return ropeBF16Key
}

// ropePipelineBF16 is the bfloat16 sibling of ropePipeline: it builds (and
// caches) the rope_single_bfloat16 kernel specialised by MLX's function
// constants — forward (id 1), traditional (id 2), head_seq_transpose (id 3),
// set at pipeline-build time via MTLFunctionConstantValues, identical to the
// float32 path (only the kernel name differs).
func ropePipelineBF16(traditional bool) (metal.MTLComputePipelineState, error) {
	key := ropePipelineBF16Key(traditional)
	ropePSOBF16Mu.Lock()
	defer ropePSOBF16Mu.Unlock()
	if pso, ok := ropePSOBF16Cache[key]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.ropePipelineBF16: library unavailable")
	}
	fc := metal.NewMTLFunctionConstantValues()
	fwd, trad, transpose := uint8(1), uint8(0), uint8(0) // forward, !traditional, !transpose
	if traditional {
		trad = 1
	}
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&fwd), metal.MTLDataTypeBool, 1)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&trad), metal.MTLDataTypeBool, 2)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&transpose), metal.MTLDataTypeBool, 3)

	fn, err := library.NewFunctionWithNameConstantValuesError("rope_single_bfloat16", fc)
	if err != nil {
		return nil, core.E("native.ropePipelineBF16", "rope_single_bfloat16", err)
	}
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.ropePipelineBF16: kernel rope_single_bfloat16 not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.ropePipelineBF16", "pipeline rope_single_bfloat16", err)
	}
	ropePSOBF16Cache[key] = pso
	return pso, nil
}

// RoPEBF16 is the bfloat16 sibling of RoPE: it applies rotary position embedding
// for the single-token (decode) case to x (raw bf16 bytes, row-major
// (b, nHeads, 1, headDim)) at absolute position offset, rotating the full
// headDim, and returns bf16 bytes of the same shape. It drives MLX's
// rope_single_bfloat16 kernel directly with the identical buffer ABI — in(0)
// out(1) offset(2) scale(3) out_strides[0](4) base(10) — and the same
// forward/traditional/transpose function constants and pre-logged (log2) base as
// the float32 path; only the kernel name and 2-byte element width differ.
// Byte-for-byte parity with pkg/metal.RoPE on the same bf16 array is gated in
// parity_test.go.
//
//	out, err := native.RoPEBF16(xBytes, 1, 8, 64, 10000, 1, 5, false)
func RoPEBF16(x []byte, b, nHeads, headDim int, base, scale float32, offset int, traditional bool) ([]byte, error) {
	return RoPEDimsBF16(x, b, nHeads, headDim, headDim, base, scale, offset, traditional)
}

// RoPEDimsBF16 is RoPEBF16 with an explicit rotary dimension: only the first rotaryDim of
// each head's headDim are rotated (gemma4's partial_rotary_factor — full_attention uses 0.25,
// so rotaryDim = headDim/4), and the remaining [rotaryDim:headDim] pass through unchanged. The
// NEOX (non-traditional) pairing is WITHIN the rotated block (dim i with i + rotaryDim/2), and
// the frequencies are normalised over rotaryDim, so it is exactly a full RoPE on the first
// rotaryDim concatenated with the untouched tail. rotaryDim must be even and in (0, headDim];
// rotaryDim == headDim is full RoPE — byte-identical to the prior RoPEBF16 (fresh out buffer,
// the whole head rotated). For partial, the out buffer is seeded with x so the kernel (which
// writes only the rotated dims) leaves the tail as the input.
func RoPEDimsBF16(x []byte, b, nHeads, headDim, rotaryDim int, base, scale float32, offset int, traditional bool) ([]byte, error) {
	return RoPEDimsBF16Into(nil, x, b, nHeads, headDim, rotaryDim, base, scale, offset, traditional)
}

func RoPEDimsBF16Into(out []byte, x []byte, b, nHeads, headDim, rotaryDim int, base, scale float32, offset int, traditional bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != b*nHeads*headDim*bf16Size {
		return nil, core.NewError("native.RoPEDimsBF16: len(x) must equal b*nHeads*headDim*2 bytes (T=1)")
	}
	outLen := len(x)
	if headDim == 0 || nHeads == 0 || b == 0 {
		if cap(out) < outLen {
			return make([]byte, outLen), nil
		}
		return out[:outLen], nil
	}
	if rotaryDim <= 0 || rotaryDim > headDim || rotaryDim%2 != 0 {
		return nil, core.NewError("native.RoPEDimsBF16: rotaryDim must be even and in (0, headDim]")
	}

	pso, err := ropePipelineBF16(traditional)
	if err != nil {
		return nil, err
	}

	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getQMVBF16Scratch(len(x)/bf16Size, len(x)/bf16Size)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVBF16Scratch(scratch)
		xBuf, outBuf, err := scratch.buffers(x)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}
		if rotaryDim < headDim {
			// partial: seed out with x so the non-rotated tail [rotaryDim:headDim] passes through
			// (the kernel writes only the rotated dims).
			if directOut {
				copy(out, x)
			} else {
				copy(scratch.out.bytes[:outLen], x)
			}
		}
		offBuf := scalarI32(int32(offset))
		logBase := float32(math.Log2(float64(base)))

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitRopeAt(encSink{enc}, pso, xBuf, outBuf, 0, 0, offBuf, 0, nil, nHeads, rotaryDim, headDim, scale, logBase)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		if !directOut {
			copy(out, scratch.out.bytes[:outLen])
		}
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

// AddBF16 is the bfloat16 sibling of Add: the element-wise sum a[i]+b[i] over two
// equal-length bf16 byte buffers, returned as bf16 bytes — the residual add used
// twice per decode block, in the dtype decode actually runs. It drives MLX's
// contiguous binary kernel vv_Addbfloat16 with the identical host ABI as the
// float32 path — a(0) b(1) out(2) element-count(3), one GPU thread per element —
// only the kernel name and 2-byte element width differ. Byte-for-byte parity with
// pkg/metal.Add on the same bf16 arrays is gated in parity_test.go.
//
//	out, err := native.AddBF16(aBytes, bBytes)
func AddBF16(a, b []byte) ([]byte, error) {
	return runBinaryBF16("vv_Addbfloat16", a, b)
}

func AddBF16Into(out, a, b []byte) error { return runBinaryBF16Into("vv_Addbfloat16", a, b, out) }
