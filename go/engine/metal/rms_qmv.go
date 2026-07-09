// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

var (
	rmsQMVPSOMu    sync.Mutex
	rmsQMVPSOCache = map[string]metal.MTLComputePipelineState{}

	rmsQMVICBPSOMu    sync.Mutex
	rmsQMVICBPSOCache = map[string]metal.MTLComputePipelineState{}

	rmsQMVKernelNames    sync.Map
	rmsQMVICBKernelNames sync.Map
)

type rmsQMVKernelNameKey struct {
	groupSize, bits int
}

func rmsQMVKernelName(groupSize, bits int) string {
	key := rmsQMVKernelNameKey{groupSize: groupSize, bits: bits}
	if v, ok := rmsQMVKernelNames.Load(key); ok {
		return v.(string)
	}
	name := core.Sprintf("lthn_rms_affine_qmv_fast_bfloat16_t_gs_%d_b_%d", groupSize, bits)
	if v, loaded := rmsQMVKernelNames.LoadOrStore(key, name); loaded {
		return v.(string)
	}
	return name
}

func rmsQMVICBKernelKey(groupSize, bits int) string {
	key := rmsQMVKernelNameKey{groupSize: groupSize, bits: bits}
	if v, ok := rmsQMVICBKernelNames.Load(key); ok {
		return v.(string)
	}
	name := rmsQMVKernelName(groupSize, bits) + "|icb"
	if v, loaded := rmsQMVICBKernelNames.LoadOrStore(key, name); loaded {
		return v.(string)
	}
	return name
}

// rmsQMVPipelineICB is rmsQMVFastPipeline with indirect-command-buffer support — the variant the
// decode ICB records (and replays per token). Same fused kernel; the descriptor just opts into ICB.
func rmsQMVPipelineICB(groupSize, bits int) (metal.MTLComputePipelineState, error) {
	name := rmsQMVKernelName(groupSize, bits)
	key := rmsQMVICBKernelKey(groupSize, bits)
	rmsQMVICBPSOMu.Lock()
	defer rmsQMVICBPSOMu.Unlock()
	if pso, ok := rmsQMVICBPSOCache[key]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.rmsQMVPipelineICB: custom library unavailable")
	}
	fn := customLibrary.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.rmsQMVPipelineICB: kernel " + name + " not found")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.rmsQMVPipelineICB", key, err)
	}
	rmsQMVICBPSOCache[key] = pso
	return pso, nil
}

// rmsQMVFastPipeline builds (and caches) the fused rms-norm + affine_qmv_fast pipeline for a group
// size / bits from the custom kernels library. Shares the customLibraryLoaded gate with gelu.
func rmsQMVFastPipeline(groupSize, bits int) (metal.MTLComputePipelineState, error) {
	key := rmsQMVKernelName(groupSize, bits)
	rmsQMVPSOMu.Lock()
	defer rmsQMVPSOMu.Unlock()
	if pso, ok := rmsQMVPSOCache[key]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.rmsQMVFastPipeline: custom library unavailable")
	}
	fn := customLibrary.NewFunctionWithName(key)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.rmsQMVFastPipeline: kernel " + key + " not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.rmsQMVFastPipeline", key, err)
	}
	rmsQMVPSOCache[key] = pso
	return pso, nil
}

// RMSQMVFastBF16 fuses, in ONE dispatch, the per-row RMSNorm(x, normW) + the 4-bit affine_qmv_fast
// projection: out = (W·RMSNorm(x, normW)). x/normW are inDim bf16 bytes; wq/scales/biases are the
// packed 4-bit weight; out is outDim bf16 bytes. Numerically equal to QMVBF16(RMSNormBF16(x, normW)) —
// the qmv arithmetic is byte-identical (bfloat16_t == native bfloat), only the rms reduction differs
// (~1 ULP, cosine ~1.0). Requires the fast-variant geometry (outDim%8==0, inDim%512==0). Guard with
// gpuHasGeluKernel.
func RMSQMVFastBF16(x, normW, wq, scales, biases []byte, outDim, inDim, groupSize, bits int, eps float32) ([]byte, error) {
	return RMSQMVFastBF16Into(nil, x, normW, wq, scales, biases, outDim, inDim, groupSize, bits, eps)
}

func RMSQMVFastBF16Into(out []byte, x, normW, wq, scales, biases []byte, outDim, inDim, groupSize, bits int, eps float32) ([]byte, error) {
	return rmsQMVFastBF16Pooled(out, x, nil, nil, normW, wq, scales, biases, outDim, inDim, groupSize, bits, eps, true, true)
}

func rmsQMVFastBF16WithBufferOutputInPool(x []byte, xBuf, outputBuf metal.MTLBuffer, normW, wq, scales, biases []byte, outDim, inDim, groupSize, bits int, eps float32) error {
	if outputBuf == nil {
		return core.NewError("native.RMSQMVFastBF16: output buffer is nil")
	}
	_, err := rmsQMVFastBF16Pooled(nil, x, xBuf, outputBuf, normW, wq, scales, biases, outDim, inDim, groupSize, bits, eps, false, false)
	return err
}

func rmsQMVFastBF16Pooled(out []byte, x []byte, xBuf, outputBuf metal.MTLBuffer, normW, wq, scales, biases []byte, outDim, inDim, groupSize, bits int, eps float32, useAutoreleasePool bool, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != inDim*bf16Size || len(normW) != inDim*bf16Size {
		return nil, core.NewError("native.RMSQMVFastBF16: x and normW must each be inDim bf16 bytes")
	}
	if outDim%8 != 0 || inDim%512 != 0 {
		return nil, core.NewError("native.RMSQMVFastBF16: needs outDim%8==0 and inDim%512==0 (fast variant)")
	}
	pso, err := rmsQMVFastPipeline(groupSize, bits)
	if err != nil {
		return nil, err
	}

	outLen := outDim * bf16Size
	bufferOut := outputBuf != nil
	callerOut := !bufferOut && useCallerOut && cap(out) >= outLen
	if bufferOut {
		out = nil
	} else if !callerOut {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	var encErr error
	run := func() {
		wBuf, sBuf, bBuf := residentBytes(wq), residentBytes(scales), residentBytes(biases)
		nwBuf := residentBytes(normW)
		scratch, err := getQMVBF16Scratch(outDim, inDim)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVBF16Scratch(scratch)
		inputBuf := xBuf
		finalOutBuf := scratch.out.buf
		if inputBuf == nil {
			var err error
			inputBuf, finalOutBuf, err = scratch.buffers(x)
			if err != nil {
				encErr = err
				return
			}
		}
		directOut := false
		if bufferOut {
			finalOutBuf = outputBuf
			directOut = true
		} else if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				finalOutBuf = tmp
				directOut = true
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitRMSQMV(encSink{enc}, pso, wBuf, 0, sBuf, 0, bBuf, 0, inputBuf, finalOutBuf, 0, nwBuf, 0, inDim, outDim, eps)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, scratch.out.bytes[:outLen])
		}
	}
	if useAutoreleasePool {
		withAutoreleasePool(run)
	} else {
		run()
	}
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}
