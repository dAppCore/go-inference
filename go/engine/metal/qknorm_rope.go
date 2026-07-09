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

var (
	qkRopeDummyOnce    sync.Once
	qkRopeDummyPeriods metal.MTLBuffer

	qkRopePeriodsBufMu    sync.Mutex
	qkRopePeriodsBufCache = map[ropePeriodsKey][]ropePeriodsCacheEntry{}
)

// qkRopeDummyBuf is a 1-element float buffer bound at the periods slot when use_freqs == 0 (the kernel
// never reads it on the base-rope path; Metal just wants the declared buffer bound).
func qkRopeDummyBuf() metal.MTLBuffer {
	qkRopeDummyOnce.Do(func() {
		one := float32(1)
		qkRopeDummyPeriods = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&one), 4, metal.MTLResourceStorageModeShared)
	})
	return qkRopeDummyPeriods
}

func cachedQKNormRopePeriodsBuffer(periods []float32) metal.MTLBuffer {
	if len(periods) == 0 {
		return nil
	}
	key := ropePeriodsKeyFor(periods)
	qkRopePeriodsBufMu.Lock()
	for _, entry := range qkRopePeriodsBufCache[key] {
		if sameFloat32Bits(periods, entry.bits) {
			buf := entry.buf
			qkRopePeriodsBufMu.Unlock()
			return buf
		}
	}
	qkRopePeriodsBufMu.Unlock()

	bits := make([]uint32, len(periods))
	for i, f := range periods {
		bits[i] = math.Float32bits(f)
	}
	buf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&periods[0]), uint(len(periods)*4), metal.MTLResourceStorageModeShared)

	qkRopePeriodsBufMu.Lock()
	for _, entry := range qkRopePeriodsBufCache[key] {
		if sameFloat32Bits(periods, entry.bits) {
			existing := entry.buf
			qkRopePeriodsBufMu.Unlock()
			return existing
		}
	}
	qkRopePeriodsBufCache[key] = append(qkRopePeriodsBufCache[key], ropePeriodsCacheEntry{bits: bits, buf: buf})
	qkRopePeriodsBufMu.Unlock()
	return buf
}

var (
	qkRopePSOOnce sync.Once
	qkRopePSO     metal.MTLComputePipelineState
	qkRopePSOErr  error
)

// qkNormRopePipeline builds (once) the fused per-head QK-norm + RoPE pipeline from the custom kernels
// library. Shares the customLibraryLoaded gate with the gelu kernel.
func qkNormRopePipeline() (metal.MTLComputePipelineState, error) {
	qkRopePSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			qkRopePSOErr = core.NewError("native.qkNormRopePipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_qknorm_rope_bf16")
		if fn == nil || fn.GetID() == 0 {
			qkRopePSOErr = core.NewError("native.qkNormRopePipeline: kernel lthn_qknorm_rope_bf16 not found")
			return
		}
		qkRopePSO, qkRopePSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return qkRopePSO, qkRopePSOErr
}

var (
	qkRopeRowsPSOOnce sync.Once
	qkRopeRowsPSO     metal.MTLComputePipelineState
	qkRopeRowsPSOErr  error
)

// qkNormRopeRowsPipeline builds (once) the batched-rows twin of the fused QK-norm + RoPE kernel —
// grid Y carries the row, positions come from the packed per-row offsets buffer.
func qkNormRopeRowsPipeline() (metal.MTLComputePipelineState, error) {
	qkRopeRowsPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			qkRopeRowsPSOErr = core.NewError("native.qkNormRopeRowsPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_qknorm_rope_rows_bf16")
		if fn == nil || fn.GetID() == 0 {
			qkRopeRowsPSOErr = core.NewError("native.qkNormRopeRowsPipeline: kernel lthn_qknorm_rope_rows_bf16 not found")
			return
		}
		qkRopeRowsPSO, qkRopeRowsPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return qkRopeRowsPSO, qkRopeRowsPSOErr
}

// gpuHasQKNormRopeRows reports whether the batched-rows QK-norm+RoPE kernel is loadable — the
// batched pass's gate for folding the K per-row rope dispatches into one.
func gpuHasQKNormRopeRows() bool {
	pso, err := qkNormRopeRowsPipeline()
	return err == nil && pso != nil && pso.GetID() != 0
}

// encQKNormRopeRows encodes the fused per-head QK-norm + RoPE for `rows` rows in ONE dispatch:
// row r reads x at xOff + r·xRowStride elements, writes out at outOff + r·outRowStride elements,
// and ropes at position offBuf[r] (the batched pass's packed per-row positions). periods non-nil
// selects the freqs/YaRN form, exactly as encQKNormRopeAt. Per-(row, head) math is the single-row
// kernel verbatim — byte-identical to `rows` encQKNormRopeAt dispatches at the same offsets.
func encQKNormRopeRows(enc metal.MTLComputeCommandEncoder, x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, xRowStride, outRowStride int, offBuf, periods metal.MTLBuffer, rows, nHeads, headDim, rotaryDim int, base, scale, eps float32) error {
	pso, err := qkNormRopeRowsPipeline()
	if err != nil {
		return err
	}
	rd := headDim
	if rotaryDim > 0 && rotaryDim < headDim {
		rd = rotaryDim
	}
	sink := encSink{enc}
	sink.setPSO(pso)
	sink.setBuf(x, xOff, 0)
	sink.setBuf(w, wOff, 1)
	sink.setBuf(out, outOff, 2)
	sink.setF32(eps, 3)
	sink.setI32(int32(headDim), 4)
	sink.setI32(int32(rd), 5)
	sink.setF32(scale, 6)
	sink.setBuf(offBuf, 0, 7)
	sink.setF32(float32(math.Log2(float64(base))), 8)
	if periods != nil {
		sink.setBuf(periods, 0, 9)
		sink.setI32(1, 10)
	} else {
		sink.setBuf(qkRopeDummyBuf(), 0, 9)
		sink.setI32(0, 10)
	}
	sink.setI32(int32(xRowStride), 11)
	sink.setI32(int32(outRowStride), 12)
	sink.dispatchThreads(
		metal.MTLSize{Width: uint(nHeads * headDim), Height: uint(rows), Depth: 1},
		metal.MTLSize{Width: uint(headDim), Height: 1, Depth: 1},
	)
	return nil
}

// QKNormRopeBF16 fuses, in ONE dispatch, gemma4's per-head QK-norm + RoPE:
//
//	out[head] = RoPE(RMSNorm(x[head], weight), offset)   — rotate the first rotaryDim dims, tail passes through
//
// x is [nHeads*headDim] bf16, weight is [headDim] bf16 (shared per head), out is the same shape. base is
// log2(theta) for the base-rope path; pass periods (1/inv_freq, length rotaryDim/2) for the freqs/YaRN
// path (non-empty ⇒ use_freqs). Numerically equal to RoPE(RMSNormBF16(x,w,nHeads,headDim)) — cosine
// ~1.0, ~1 ULP bf16 rounding (the lockstep fused-kernel gap) — gated in the parity test. headDim ≤ 512.
func QKNormRopeBF16(x, weight []byte, nHeads, headDim, rotaryDim, offset int, scale, eps, base float32, periods []float32) ([]byte, error) {
	return QKNormRopeBF16Into(nil, x, weight, nHeads, headDim, rotaryDim, offset, scale, eps, base, periods)
}

func QKNormRopeBF16Into(out []byte, x, weight []byte, nHeads, headDim, rotaryDim, offset int, scale, eps, base float32, periods []float32) ([]byte, error) {
	return qkNormRopeBF16Pooled(out, x, nil, nil, weight, nHeads, headDim, rotaryDim, offset, scale, eps, base, periods, true, true)
}

func qkNormRopeBF16WithBufferOutputInPool(x []byte, xBuf, outputBuf metal.MTLBuffer, weight []byte, nHeads, headDim, rotaryDim, offset int, scale, eps, base float32, periods []float32) error {
	if outputBuf == nil {
		return core.NewError("native.QKNormRopeBF16: output buffer is nil")
	}
	_, err := qkNormRopeBF16Pooled(nil, x, xBuf, outputBuf, weight, nHeads, headDim, rotaryDim, offset, scale, eps, base, periods, false, false)
	return err
}

func qkNormRopeBF16Pooled(out []byte, x []byte, xBuf, outputBuf metal.MTLBuffer, weight []byte, nHeads, headDim, rotaryDim, offset int, scale, eps, base float32, periods []float32, useAutoreleasePool bool, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != nHeads*headDim*bf16Size {
		return nil, core.NewError("native.QKNormRopeBF16: x must be nHeads*headDim bf16 bytes")
	}
	if len(weight) != headDim*bf16Size {
		return nil, core.NewError("native.QKNormRopeBF16: weight must be headDim bf16 bytes")
	}
	if headDim > 512 {
		return nil, core.NewError("native.QKNormRopeBF16: headDim exceeds the 512 threadgroup cap")
	}
	pso, err := qkNormRopePipeline()
	if err != nil {
		return nil, err
	}
	outLen := len(x)
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
		scratch, err := getQMVBF16Scratch(len(x)/bf16Size, len(x)/bf16Size)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVBF16Scratch(scratch)
		inputBuf := xBuf
		output := scratch.out.buf
		if inputBuf == nil {
			var err error
			inputBuf, output, err = scratch.buffers(x)
			if err != nil {
				encErr = err
				return
			}
		}
		directOut := false
		if bufferOut {
			output = outputBuf
			directOut = true
		} else if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				output = tmp
				directOut = true
			}
		}
		wBuf := residentBytes(weight)
		offBuf := scalarI32(int32(offset))
		var perBuf metal.MTLBuffer
		if len(periods) > 0 {
			perBuf = cachedQKNormRopePeriodsBuffer(periods)
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitQKNormRope(encSink{enc}, pso, inputBuf, wBuf, output, 0, 0, 0, offBuf, perBuf, qkRopeDummyBuf(), nHeads, headDim, rotaryDim, eps, scale, base)
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

// encQKNormRope encodes the fused per-head QK-norm + RoPE (out = RoPE(RMSNorm(x, w))) into enc — the
// re-encode sibling of the ICB's setQKNormRope, using the SAME kernel so the two paths stay byte-equal
// under the lockstep fusion. base is RAW theta (log2'd here, matching encRoPEBF16To); periods non-nil ⇒
// the freqs/YaRN path. x/w/out may carry byte offsets (the K cache row, the qk-norm shard view).
// Caller guards with gpuHasGeluKernel.
func encQKNormRope(enc metal.MTLComputeCommandEncoder, x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, offBuf, periods metal.MTLBuffer, nHeads, headDim, rotaryDim int, base, scale, eps float32) error {
	return encQKNormRopeAt(enc, x, w, out, xOff, wOff, outOff, offBuf, 0, periods, nHeads, headDim, rotaryDim, base, scale, eps)
}

func encQKNormRopeAt(enc metal.MTLComputeCommandEncoder, x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, offBuf metal.MTLBuffer, offOff uint, periods metal.MTLBuffer, nHeads, headDim, rotaryDim int, base, scale, eps float32) error {
	pso, err := qkNormRopePipeline()
	if err != nil {
		return err
	}
	rd := headDim
	if rotaryDim > 0 && rotaryDim < headDim {
		rd = rotaryDim
	}
	// fused per-head QK-norm + RoPE through the SHARED emitQKNormRope body (with the ICB setQKNormRope);
	// periods != nil selects the freqs form, else the base form binds qkRopeDummyBuf() at 9 (unread).
	emitQKNormRopeAt(encSink{enc}, pso, x, w, out, xOff, wOff, outOff, offBuf, offOff, periods, qkRopeDummyBuf(),
		nHeads, headDim, rd, eps, scale, float32(math.Log2(float64(base))))
	return nil
}
