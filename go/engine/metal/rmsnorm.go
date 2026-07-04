// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// RMS kernel selection constants, mirrored from MLX
// (mlx/backend/metal/kernels/defines.h): n_reads per thread, the axis size above
// which the looped variant is used, and the simd width.
const (
	rmsNReads      = 4
	rmsLoopedLimit = 4096
	rmsSimdSize    = 32
)

// rmsKernelBF16 returns the bf16 rms kernel for an axis: the single-row kernel up to rmsLoopedLimit,
// the LOOPED kernel above it. The single-row kernel needs one threadgroup of ceil(axis/N_READS)
// lanes, which exceeds Metal's 1024-thread cap once axis > rmsLoopedLimit (≈ N_READS·1024 = 4096) —
// that overrun is why a hidden_size of 5376 (gemma4 31B) produced an invalid dispatch. The looped
// kernel uses a fixed threadgroup that grid-strides the whole axis, so it handles any size. Mirrors
// the float32 path in RMSNorm and MLX's normalization.cpp dispatch.
func rmsKernelBF16(axisSize int) string {
	if axisSize > rmsLoopedLimit {
		return "rms_loopedbfloat16"
	}
	return "rmsbfloat16"
}

// rmsThreadgroup is the threadgroup size for an rms dispatch given the chosen pipeline: the looped
// kernel uses its max threads (it grid-strides the axis), the single-row kernel uses
// ceil(axis/N_READS) rounded up to a simd.
func rmsThreadgroup(axisSize int, pso metal.MTLComputePipelineState) uint {
	if axisSize > rmsLoopedLimit {
		return pso.MaxTotalThreadsPerThreadgroup()
	}
	return uint(rmsSimdSize * ((((axisSize + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
}

// RMSNorm computes the RMS-normalised rows of x scaled by weight:
//
//	out[r,i] = x[r,i] * rsqrt(mean_i(x[r,:]²) + eps) * weight[i]
//
// x is row-major (rows × axisSize), weight is length axisSize, and the result is
// the same shape as x. It drives MLX's rms / rms_looped kernel directly through
// the no-cgo path: x(0) weight(1) out(2) eps(3) axis_size(4) w_stride(5), one
// threadgroup per row dispatched as threads. axisSize ≤ 4096 takes the single-row
// kernel (every gemma hidden size); larger takes the looped kernel. float32 only.
// Byte-for-byte parity with pkg/metal.RMSNorm is gated in parity_test.go.
func RMSNorm(x, weight []float32, rows, axisSize int, eps float32) ([]float32, error) {
	out := make([]float32, len(x))
	if err := rmsNormInto(out, x, weight, rows, axisSize, eps, false); err != nil {
		return nil, err
	}
	return out, nil
}

func RMSNormInto(out, x, weight []float32, rows, axisSize int, eps float32) ([]float32, error) {
	callerOut := out != nil && cap(out) >= len(x)
	if !callerOut {
		out = make([]float32, len(x))
	} else {
		out = out[:len(x)]
	}
	if err := rmsNormInto(out, x, weight, rows, axisSize, eps, callerOut); err != nil {
		return nil, err
	}
	return out, nil
}

func rmsNormInto(out, x, weight []float32, rows, axisSize int, eps float32, directOutput bool) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if len(x) != rows*axisSize {
		return core.NewError("native.RMSNorm: len(x) must equal rows*axisSize")
	}
	if len(weight) != axisSize {
		return core.NewError("native.RMSNorm: len(weight) must equal axisSize")
	}
	if len(out) != len(x) {
		return core.NewError("native.RMSNorm: len(out) must equal len(x)")
	}
	if rows == 0 || axisSize == 0 {
		return nil
	}

	name := "rmsfloat32"
	looped := axisSize > rmsLoopedLimit
	if looped {
		name = "rms_loopedfloat32"
	}
	pso, err := pipelineFor(name)
	if err != nil {
		return err
	}

	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getQMVFloatScratch(len(x), len(x))
		if err != nil {
			encErr = err
			return
		}
		defer putQMVFloatScratch(scratch)
		xBuf, outBuf, err := scratch.buffers(x)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if directOutput {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}
		wBuf := residentBytes(float32Bytes(weight))

		tgSize := rmsThreadgroup(axisSize, pso)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitRMSNormRows(encSink{enc}, pso, xBuf, wBuf, outBuf, 0, 0, 0, axisSize, eps, rows, tgSize)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		if !directOut {
			copy(float32Bytes(out), scratch.out.bytes[:len(x)*4])
		}
	})
	if encErr != nil {
		return encErr
	}
	return nil
}

// setEncFloat32 binds a single float32 as an inline constant at a buffer index
// (the rms epsilon).
func setEncFloat32(enc metal.MTLComputeCommandEncoder, v float32, idx uint) {
	setBytesF32(enc, v, idx)
}
