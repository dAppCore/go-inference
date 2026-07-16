// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// composed_attn_core_backend.go — the composed attention core's device kernels (#26 / #18 S6a):
// resolvers + encoder emits for lthn_attn_{qprep,kprep,sdpa,gate_silu}_f32
// (kernels/lthn_attn_core.metal, ported from composed's continueFromQKV), plus the test-facing
// one-CB wrapper the parity gate drives. Integration (the device-resident KV handle inside the
// attention fold) binds the emits directly.

var (
	attnQPrepOnce, attnKPrepOnce, attnSDPAOnce, attnGateOnce sync.Once
	attnQPrepPSO, attnKPrepPSO, attnSDPAPSO, attnGatePSO     metal.MTLComputePipelineState
	attnQPrepErr, attnKPrepErr, attnSDPAErr, attnGateErr     error
)

func attnQPrepPipeline() (metal.MTLComputePipelineState, error) {
	attnQPrepOnce.Do(func() { gdPlainPipeline("lthn_attn_qprep_f32", &attnQPrepPSO, &attnQPrepErr) })
	return attnQPrepPSO, attnQPrepErr
}
func attnKPrepPipeline() (metal.MTLComputePipelineState, error) {
	attnKPrepOnce.Do(func() { gdPlainPipeline("lthn_attn_kprep_f32", &attnKPrepPSO, &attnKPrepErr) })
	return attnKPrepPSO, attnKPrepErr
}
func attnSDPAPipeline() (metal.MTLComputePipelineState, error) {
	attnSDPAOnce.Do(func() { gdPlainPipeline("lthn_attn_sdpa_f32", &attnSDPAPSO, &attnSDPAErr) })
	return attnSDPAPSO, attnSDPAErr
}
func attnGatePipeline() (metal.MTLComputePipelineState, error) {
	attnGateOnce.Do(func() { gdPlainPipeline("lthn_attn_gate_silu_f32", &attnGatePSO, &attnGateErr) })
	return attnGatePSO, attnGateErr
}

// attnCoreUsable reports whether the device attention core serves this geometry: HD within the
// kernels' 256-wide threadgroup staging and a resolvable pipeline set. The customLibrary check
// runs FIRST (the #23 pre-init lesson).
func attnCoreUsable(H, KVH, HD, RD int) bool {
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return false
	}
	if H <= 0 || KVH <= 0 || HD <= 0 || HD > 256 || RD < 0 || RD > HD || RD%2 != 0 || H%KVH != 0 {
		return false
	}
	for _, resolve := range []func() (metal.MTLComputePipelineState, error){
		attnQPrepPipeline, attnKPrepPipeline, attnSDPAPipeline, attnGatePipeline,
	} {
		if pso, err := resolve(); err != nil || pso == nil || pso.GetID() == 0 {
			return false
		}
	}
	return true
}

// encAttnQPrep encodes the q-prep: de-interleave the σ-gate (gated), per-head norm (mode) + rope,
// writing q [L,H,HD] and gate [L,H,HD].
func encAttnQPrep(enc metal.MTLComputeCommandEncoder, qRaw, w, q, gate metal.MTLBuffer, L, H, HD, RD, gated, qkNorm int, eps, theta float32, pos0 int) error {
	pso, err := attnQPrepPipeline()
	if err != nil {
		return err
	}
	setPSO(enc, pso)
	setBuf(enc, qRaw, 0, 0)
	setBuf(enc, w, 0, 1)
	setBuf(enc, q, 0, 2)
	setBuf(enc, gate, 0, 3)
	setEncInt32(enc, int32(H), 4)
	setEncInt32(enc, int32(HD), 5)
	setEncInt32(enc, int32(RD), 6)
	setEncInt32(enc, int32(gated), 7)
	setEncInt32(enc, int32(qkNorm), 8)
	setBytesF32(enc, eps, 9)
	setBytesF32(enc, theta, 10)
	setEncInt32(enc, int32(pos0), 11)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: 1, Height: uint(H), Depth: uint(L)},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1})
	return nil
}

// encAttnKPrep encodes the k-prep: per-head norm + rope on the raw k rows, written into the KV
// cache slots pos0..pos0+L−1.
func encAttnKPrep(enc metal.MTLComputeCommandEncoder, k, w, cacheK metal.MTLBuffer, L, KVH, HD, RD, qkNorm int, eps, theta float32, pos0 int) error {
	pso, err := attnKPrepPipeline()
	if err != nil {
		return err
	}
	setPSO(enc, pso)
	setBuf(enc, k, 0, 0)
	setBuf(enc, w, 0, 1)
	setBuf(enc, cacheK, 0, 2)
	setEncInt32(enc, int32(KVH), 3)
	setEncInt32(enc, int32(HD), 4)
	setEncInt32(enc, int32(RD), 5)
	setEncInt32(enc, int32(qkNorm), 6)
	setBytesF32(enc, eps, 7)
	setBytesF32(enc, theta, 8)
	setEncInt32(enc, int32(pos0), 9)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: 1, Height: uint(KVH), Depth: uint(L)},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1})
	return nil
}

// encAttnSDPA encodes the causal GQA attention over the device cache into out [L,H,HD].
func encAttnSDPA(enc metal.MTLComputeCommandEncoder, q, cacheK, cacheV, out metal.MTLBuffer, L, H, KVH, HD, pos0, window int) error {
	pso, err := attnSDPAPipeline()
	if err != nil {
		return err
	}
	setPSO(enc, pso)
	setBuf(enc, q, 0, 0)
	setBuf(enc, cacheK, 0, 1)
	setBuf(enc, cacheV, 0, 2)
	setBuf(enc, out, 0, 3)
	setEncInt32(enc, int32(H), 4)
	setEncInt32(enc, int32(KVH), 5)
	setEncInt32(enc, int32(HD), 6)
	setEncInt32(enc, int32(pos0), 7)
	setEncInt32(enc, int32(window), 8)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: 1, Height: uint(H), Depth: uint(L)},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1})
	return nil
}

// encAttnGateSilu encodes the σ-gate multiply out[i] *= silu(gate[i]).
func encAttnGateSilu(enc metal.MTLComputeCommandEncoder, out, gate metal.MTLBuffer, total int) error {
	pso, err := attnGatePipeline()
	if err != nil {
		return err
	}
	setPSO(enc, pso)
	setBuf(enc, out, 0, 0)
	setBuf(enc, gate, 0, 1)
	setEncInt32(enc, int32(total), 2)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: uint((total + 255) / 256), Height: 1, Depth: 1},
		metal.MTLSize{Width: 256, Height: 1, Depth: 1})
	return nil
}

// AttnCoreDeviceRun drives the whole device attention core over host slices in ONE command buffer —
// the parity/bench wrapper (integration keeps the cache resident and never reads it back). cacheK/
// cacheV are [cap,KVH,HD] with rows [0,pos0) live; rows [pos0,pos0+L) are written from k/v. qRaw is
// [L, qCols]; out is [L,H,HD]. qNormW/kNormW may be nil unless qkNorm==1.
func AttnCoreDeviceRun(qRaw, k, v, qNormW, kNormW, cacheK, cacheV, out []float32, L, H, KVH, HD, RD, pos0, window, gated, qkNorm int, eps, theta float32) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if !attnCoreUsable(H, KVH, HD, RD) {
		return core.NewError("native.AttnCoreDeviceRun: geometry not servable")
	}
	qCols := H * HD
	if gated != 0 {
		qCols = 2 * H * HD
	}
	cap0 := len(cacheK) / (KVH * HD)
	if len(qRaw) != L*qCols || len(k) != L*KVH*HD || len(v) != L*KVH*HD || len(out) != L*H*HD ||
		len(cacheK) != len(cacheV) || pos0+L > cap0 ||
		(qkNorm == 1 && (len(qNormW) != HD || len(kNormW) != HD)) {
		return core.NewError("native.AttnCoreDeviceRun: size mismatch")
	}
	var encErr error
	withAutoreleasePool(func() {
		alloc := func(src []float32) (*pinnedNoCopyBytes, metal.MTLBuffer) {
			if encErr != nil {
				return nil, nil
			}
			b, err := newPinnedNoCopyBytes(len(src) * 4)
			if err != nil {
				encErr = err
				return nil, nil
			}
			buf, cerr := b.copyBuffer(float32Bytes(src))
			if cerr != nil {
				encErr = cerr
				return nil, nil
			}
			return b, buf
		}
		_, qRawBuf := alloc(qRaw)
		_, kBuf := alloc(k)
		_, vBuf := alloc(v)
		ckPin, ckBuf := alloc(cacheK)
		cvPin, cvBuf := alloc(cacheV)
		qPin, _ := alloc(make([]float32, L*H*HD))
		gatePin, _ := alloc(make([]float32, L*H*HD))
		outPin, _ := alloc(out)
		if encErr != nil {
			return
		}
		normQ, normK := qRawBuf, qRawBuf
		if qkNorm == 1 {
			_, normQ = alloc(qNormW)
			_, normK = alloc(kNormW)
			if encErr != nil {
				return
			}
		}
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		fail := func(err error) { encErr = err; endEncodingFast(enc) }
		if err := encAttnQPrep(enc, qRawBuf, normQ, qPin.buf, gatePin.buf, L, H, HD, RD, gated, qkNorm, eps, theta, pos0); err != nil {
			fail(err)
			return
		}
		if err := encAttnKPrep(enc, kBuf, normK, ckBuf, L, KVH, HD, RD, qkNorm, eps, theta, pos0); err != nil {
			fail(err)
			return
		}
		// v append: plain blit into the cache slot (no transform).
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		endEncodingFast(enc)
		blit := blitCommandEncoderFast(cb)
		blit.CopyFromBufferSourceOffsetToBufferDestinationOffsetSize(vBuf, 0, cvBuf, uint(pos0*KVH*HD*4), uint(L*KVH*HD*4))
		endBlitEncodingFast(blit)
		enc = computeCommandEncoderFast(cb)
		if err := encAttnSDPA(enc, qPin.buf, ckBuf, cvBuf, outPin.buf, L, H, KVH, HD, pos0, window); err != nil {
			fail(err)
			return
		}
		if gated != 0 {
			memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
			if err := encAttnGateSilu(enc, outPin.buf, gatePin.buf, L*H*HD); err != nil {
				fail(err)
				return
			}
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(out, unsafe.Slice((*float32)(unsafe.Pointer(&outPin.bytes[0])), L*H*HD))
		copy(cacheK, unsafe.Slice((*float32)(unsafe.Pointer(&ckPin.bytes[0])), len(cacheK)))
		copy(cacheV, unsafe.Slice((*float32)(unsafe.Pointer(&cvPin.bytes[0])), len(cacheV)))
	})
	return encErr
}
