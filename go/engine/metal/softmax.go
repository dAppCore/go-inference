// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

const (
	softmaxNReads      = 4
	softmaxLoopedLimit = 4096
	softmaxSimdSize    = 32
)

// SoftmaxF32 computes the row-wise softmax over the last axis (axisSize) of a row-major [rows,
// axisSize] float32 buffer, driving MLX's block_softmax_float32 or looped_softmax_float32 kernel
// directly — the byte-parity non-cgo equivalent of pkg/metal.Softmax (non-precise) on the same f32
// array. The Conformer audio attention runs in float32 (metal projects with .float()), so its softmax
// over the context axis goes through this. ABI (mlx softmax.cpp): in→0, out→1, axis_size→2; one
// threadgroup per row. Axes up to 4096 use the block kernel; longer axes use the looped kernel.
func SoftmaxF32(in []float32, axisSize int) ([]float32, error) {
	out := make([]float32, len(in))
	if err := softmaxF32Into(out, in, axisSize, false); err != nil {
		return nil, err
	}
	return out, nil
}

// SoftmaxF32Into is SoftmaxF32 with caller-owned output storage when cap(out) >= len(in).
func SoftmaxF32Into(out, in []float32, axisSize int) ([]float32, error) {
	callerOut := out != nil && cap(out) >= len(in)
	if !callerOut {
		out = make([]float32, len(in))
	} else {
		out = out[:len(in)]
	}
	if err := softmaxF32Into(out, in, axisSize, callerOut); err != nil {
		return nil, err
	}
	return out, nil
}

func softmaxF32Into(out, in []float32, axisSize int, directOutput bool) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if axisSize == 0 || len(in)%axisSize != 0 {
		return core.NewError("native.SoftmaxF32: len(in) must be a multiple of axisSize")
	}
	if len(out) != len(in) {
		return core.NewError("native.SoftmaxF32: len(out) must equal len(in)")
	}
	name := "block_softmax_float32"
	if axisSize > softmaxLoopedLimit {
		name = "looped_softmax_float32"
	}
	nRows := len(in) / axisSize
	pso, err := pipelineFor(name)
	if err != nil {
		return err
	}

	tg := softmaxThreadgroup(axisSize, pso)

	if nRows == 0 {
		return nil
	}
	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getQMVFloatScratch(len(in), len(in))
		if err != nil {
			encErr = err
			return
		}
		defer putQMVFloatScratch(scratch)
		inBuf, outBuf, err := scratch.buffers(in)
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
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitSoftmax(encSink{enc}, pso, inBuf, outBuf, axisSize, nRows, tg)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(float32Bytes(out), scratch.out.bytes[:len(in)*4])
		}
	})
	if encErr != nil {
		return encErr
	}
	return nil
}

func softmaxThreadgroup(axisSize int, pso metal.MTLComputePipelineState) uint {
	if axisSize > softmaxLoopedLimit {
		return pso.MaxTotalThreadsPerThreadgroup()
	}
	tgNeeded := (axisSize + softmaxNReads - 1) / softmaxNReads
	simdsNeeded := (tgNeeded + softmaxSimdSize - 1) / softmaxSimdSize
	return uint(softmaxSimdSize * simdsNeeded)
}
