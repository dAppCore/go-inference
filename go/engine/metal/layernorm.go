// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

const (
	layerNormNReads      = 8
	layerNormLoopedLimit = 6656
	layerNormSimdSize    = 32
)

// LayerNormBF16 is the byte-parity bf16 LayerNorm (kernel layer_normbfloat16): per row over the last
// axis it computes (x-mean)/sqrt(var+eps)·weight + bias, equalling pkg/metal.LayerNorm on the same
// bf16 arrays. The gemma4 audio subsampler uses a scale-only LayerNorm (bias = zeros) after each
// strided conv. ABI (mlx normalization.cpp): x→0, w→1, b→2, out→3, eps→4, axis_size→5, w_stride→6,
// b_stride→7; one threadgroup per row. Axes up to 6656 use the block kernel; longer axes use MLX's
// looped kernel. weight/bias are length-axisSize bf16.
func LayerNormBF16(x, weight, bias []byte, rows, axisSize int, eps float32) ([]byte, error) {
	return LayerNormBF16Into(nil, x, weight, bias, rows, axisSize, eps)
}

func LayerNormBF16Into(out []byte, x, weight, bias []byte, rows, axisSize int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if axisSize == 0 || len(x) != rows*axisSize*bf16Size {
		return nil, core.NewError("native.LayerNormBF16: len(x) must equal rows*axisSize*2 bytes")
	}
	if len(weight) != axisSize*bf16Size || len(bias) != axisSize*bf16Size {
		return nil, core.NewError("native.LayerNormBF16: weight/bias must be length axisSize bf16")
	}
	name := "layer_normbfloat16"
	if axisSize > layerNormLoopedLimit {
		name = "layer_norm_loopedbfloat16"
	}
	pso, err := pipelineFor(name)
	if err != nil {
		return nil, err
	}

	tg := layerNormThreadgroup(axisSize, pso)

	outLen := len(x)
	callerOut := out != nil && cap(out) >= outLen
	if !callerOut {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	if rows == 0 {
		return out, nil
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
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}
		wBuf, bBuf := residentBytes(weight), residentBytes(bias)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitLayerNorm(encSink{enc}, pso, xBuf, wBuf, bBuf, outBuf, axisSize, rows, eps, tg)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, scratch.out.bytes[:len(out)])
		}
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

// LayerNormF32 is the fp32 LayerNorm (kernel layer_normfloat32) — the fp32 sibling of LayerNormBF16,
// matching metal.LayerNorm on fp32 arrays (the subsampler's second LayerNorm runs fp32). weight/bias
// are length-axisSize fp32 (the bf16 model weights widened). Axes above 6656 use MLX's looped kernel.
func LayerNormF32(x, weight, bias []float32, rows, axisSize int, eps float32) ([]float32, error) {
	return LayerNormF32Into(nil, x, weight, bias, rows, axisSize, eps)
}

func LayerNormF32Into(out []float32, x, weight, bias []float32, rows, axisSize int, eps float32) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if axisSize == 0 || len(x) != rows*axisSize {
		return nil, core.NewError("native.LayerNormF32: len(x) must equal rows*axisSize")
	}
	if len(weight) != axisSize || len(bias) != axisSize {
		return nil, core.NewError("native.LayerNormF32: weight/bias must be length axisSize")
	}
	name := "layer_normfloat32"
	if axisSize > layerNormLoopedLimit {
		name = "layer_norm_loopedfloat32"
	}
	pso, err := pipelineFor(name)
	if err != nil {
		return nil, err
	}

	tg := layerNormThreadgroup(axisSize, pso)

	outLen := len(x)
	callerOut := out != nil && cap(out) >= outLen
	if !callerOut {
		out = make([]float32, outLen)
	} else {
		out = out[:outLen]
	}
	if rows == 0 {
		return out, nil
	}
	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getQMVFloatScratch(rows*axisSize, rows*axisSize)
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
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}
		wBuf, bBuf := residentFloat32(weight), residentFloat32(bias)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitLayerNorm(encSink{enc}, pso, xBuf, wBuf, bBuf, outBuf, axisSize, rows, eps, tg)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(float32Bytes(out), scratch.out.bytes[:len(x)*4])
		}
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

func layerNormThreadgroup(axisSize int, pso metal.MTLComputePipelineState) uint {
	if axisSize > layerNormLoopedLimit {
		return pso.MaxTotalThreadsPerThreadgroup()
	}
	tgNeeded := (axisSize + layerNormNReads - 1) / layerNormNReads
	simdsNeeded := (tgNeeded + layerNormSimdSize - 1) / layerNormSimdSize
	return uint(layerNormSimdSize * simdsNeeded)
}
