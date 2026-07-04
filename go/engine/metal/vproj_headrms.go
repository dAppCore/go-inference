// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

var (
	vprojHeadRMSPSOMu    sync.Mutex
	vprojHeadRMSPSOCache = map[string]metal.MTLComputePipelineState{}
	vprojHeadRMSNames    sync.Map
	vprojHeadRMSICBNames sync.Map
)

type vprojHeadRMSKernelNameKey struct {
	groupSize, bits int
}

func vprojHeadRMSKernelName(groupSize, bits int) string {
	key := vprojHeadRMSKernelNameKey{groupSize: groupSize, bits: bits}
	if v, ok := vprojHeadRMSNames.Load(key); ok {
		return v.(string)
	}
	name := core.Sprintf("lthn_vproj_headrms_bfloat16_t_gs_%d_b_%d", groupSize, bits)
	if v, loaded := vprojHeadRMSNames.LoadOrStore(key, name); loaded {
		return v.(string)
	}
	return name
}

func vprojHeadRMSPipelineKey(groupSize, bits int, icb bool) string {
	if !icb {
		return vprojHeadRMSKernelName(groupSize, bits)
	}
	key := vprojHeadRMSKernelNameKey{groupSize: groupSize, bits: bits}
	if v, ok := vprojHeadRMSICBNames.Load(key); ok {
		return v.(string)
	}
	name := vprojHeadRMSKernelName(groupSize, bits) + "|icb"
	if v, loaded := vprojHeadRMSICBNames.LoadOrStore(key, name); loaded {
		return v.(string)
	}
	return name
}

func vprojHeadRMSPipeline(groupSize, bits int, icb bool) (metal.MTLComputePipelineState, error) {
	name := vprojHeadRMSKernelName(groupSize, bits)
	key := vprojHeadRMSPipelineKey(groupSize, bits, icb)
	vprojHeadRMSPSOMu.Lock()
	defer vprojHeadRMSPSOMu.Unlock()
	if pso, ok := vprojHeadRMSPSOCache[key]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.vprojHeadRMSPipeline: custom library unavailable")
	}
	fn := customLibrary.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.vprojHeadRMSPipeline: kernel " + name + " not found")
	}
	if !icb {
		pso, err := device.NewComputePipelineStateWithFunctionError(fn)
		if err != nil {
			return nil, core.E("native.vprojHeadRMSPipeline", key, err)
		}
		vprojHeadRMSPSOCache[key] = pso
		return pso, nil
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.vprojHeadRMSPipeline", key, err)
	}
	vprojHeadRMSPSOCache[key] = pso
	return pso, nil
}

// VProjHeadRMSBF16 fuses, in ONE dispatch, the whole gemma4 V path: input-RMSNorm(x, inNormW) → 4-bit
// V projection → per-head value-norm (RMS over headDim). One threadgroup per KV head. Equal (cosine ~1.0,
// lockstep) to RMSNormBF16(QMVBF16(RMSNormBF16(x, inNormW)), ones, nKVHeads, headDim). headDim must be a
// power of two ≤ 1024 (the in-kernel tree reductions). x/inNormW are inDim bf16 bytes; out is
// nKVHeads·headDim bf16 bytes.
func VProjHeadRMSBF16(x, inNormW, wq, scales, biases []byte, nKVHeads, headDim, inDim, groupSize, bits int, eps float32) ([]byte, error) {
	return VProjHeadRMSBF16Into(nil, x, inNormW, wq, scales, biases, nKVHeads, headDim, inDim, groupSize, bits, eps)
}

func VProjHeadRMSBF16Into(out []byte, x, inNormW, wq, scales, biases []byte, nKVHeads, headDim, inDim, groupSize, bits int, eps float32) ([]byte, error) {
	return vProjHeadRMSBF16Pooled(out, x, nil, nil, inNormW, wq, scales, biases, nKVHeads, headDim, inDim, groupSize, bits, eps, true, true)
}

func vProjHeadRMSBF16WithBufferOutputInPool(x []byte, xBuf, outputBuf metal.MTLBuffer, inNormW, wq, scales, biases []byte, nKVHeads, headDim, inDim, groupSize, bits int, eps float32) error {
	if outputBuf == nil {
		return core.NewError("native.VProjHeadRMSBF16: output buffer is nil")
	}
	_, err := vProjHeadRMSBF16Pooled(nil, x, xBuf, outputBuf, inNormW, wq, scales, biases, nKVHeads, headDim, inDim, groupSize, bits, eps, false, false)
	return err
}

func vProjHeadRMSBF16Pooled(out []byte, x []byte, xBuf, outputBuf metal.MTLBuffer, inNormW, wq, scales, biases []byte, nKVHeads, headDim, inDim, groupSize, bits int, eps float32, useAutoreleasePool bool, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != inDim*bf16Size || len(inNormW) != inDim*bf16Size {
		return nil, core.NewError("native.VProjHeadRMSBF16: x and inNormW must each be inDim bf16 bytes")
	}
	if headDim <= 0 || headDim > 1024 || headDim&(headDim-1) != 0 {
		return nil, core.NewError("native.VProjHeadRMSBF16: headDim must be a power of two ≤ 1024")
	}
	pso, err := vprojHeadRMSPipeline(groupSize, bits, false)
	if err != nil {
		return nil, err
	}

	outDim := nKVHeads * headDim
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
		nwBuf := residentBytes(inNormW)
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
		emitVProjHeadRMS(encSink{enc}, pso, wBuf, sBuf, bBuf, inputBuf, nwBuf, finalOutBuf, inDim, nKVHeads, headDim, eps)
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
