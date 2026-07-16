// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
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
	attnGateOnce.Do(func() { gdPlainPipeline("lthn_attn_gate_sigmoid_f32", &attnGatePSO, &attnGateErr) })
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

// encAttnGateSilu encodes the σ-gate multiply out[i] *= sigmoid(gate[i]) — the transformers
// qwen3_5 hardcoded-sigmoid convention (see the kernel doc).
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

// --- the device-resident KV integration: one CB per attention layer ---

// attnKVDeviceState is the per-(session, layer) resident KV cache — the attention twin of
// gatedDeltaDeviceState: [capRows, KVH, HD] f32 K and V buffers with cap-doubling growth, primed
// from the host state once and exported only for snapshots/clones.
type attnKVDeviceState struct {
	kBuf, vBuf        *pinnedNoCopyBytes
	n, capRows        int
	KVH, HD           int
}

func (h *attnKVDeviceState) rowBytes() int { return h.KVH * h.HD * 4 }

// ensureCap grows the cache to hold at least rows (cap-doubling; host-visible pinned bytes make
// the old-row carry a plain copy).
func (h *attnKVDeviceState) ensureCap(rows int) error {
	if rows <= h.capRows {
		return nil
	}
	newCap := h.capRows * 2
	if newCap < rows {
		newCap = rows
	}
	if newCap < 128 {
		newCap = 128
	}
	nk, err := newPinnedNoCopyBytes(newCap * h.rowBytes())
	if err != nil {
		return err
	}
	nv, err := newPinnedNoCopyBytes(newCap * h.rowBytes())
	if err != nil {
		return err
	}
	if h.kBuf != nil && h.n > 0 {
		copy(nk.bytes, h.kBuf.bytes[:h.n*h.rowBytes()])
		copy(nv.bytes, h.vBuf.bytes[:h.n*h.rowBytes()])
	}
	h.kBuf, h.vBuf, h.capRows = nk, nv, newCap
	return nil
}

// AttnBF16FullLayerDevice runs one WHOLE dense bf16 attention layer in a single command buffer
// over the resident KV cache: input RMSNorm → q/k/v gemv → qprep/kprep (rope+norm; k lands in the
// cache slot) → v blit into its slot → SDPA → σ-gate → o_proj → the bf16 FFN tail. dev threads the
// resident handle (nil on first call: allocated and primed from priorK/priorV). Only x and y cross
// the host boundary.
func AttnBF16FullLayerDevice(dev any, x, inputNorm []float32, qw, kw, vw, ow *model.BF16Weight, qNormW, kNormW, postNorm []float32, gate, up, down *model.BF16Weight, priorK, priorV []float32, L, D, H, KVH, HD, RD, pos0, window, gated, qkNorm, FF int, eps, theta float32) ([]float32, any, error) {
	if err := ensureInit(); err != nil {
		return nil, dev, err
	}
	if !attnCoreUsable(H, KVH, HD, RD) {
		return nil, dev, core.NewError("native.AttnBF16FullLayerDevice: core not servable")
	}
	qCols := H * HD
	if gated != 0 {
		qCols = 2 * H * HD
	}
	mixCols := H * HD
	if len(x) != L*D || !bf16GeometryOK(qw, qCols, D) || !bf16GeometryOK(kw, KVH*HD, D) ||
		!bf16GeometryOK(vw, KVH*HD, D) || !bf16GeometryOK(ow, D, mixCols) ||
		!bf16GeometryOK(gate, FF, D) || !bf16GeometryOK(up, FF, D) || !bf16GeometryOK(down, D, FF) {
		return nil, dev, core.NewError("native.AttnBF16FullLayerDevice: size/geometry mismatch")
	}
	h, _ := dev.(*attnKVDeviceState)
	if h == nil {
		h = &attnKVDeviceState{KVH: KVH, HD: HD}
		if err := h.ensureCap(pos0 + L); err != nil {
			return nil, dev, err
		}
		if pos0 > 0 { // prime from the host state (a restored snapshot or a handoff)
			if len(priorK) != pos0*KVH*HD || len(priorV) != pos0*KVH*HD {
				return nil, dev, core.NewError("native.AttnBF16FullLayerDevice: prior state size mismatch")
			}
			copy(h.kBuf.bytes, float32Bytes(priorK))
			copy(h.vBuf.bytes, float32Bytes(priorV))
		}
		h.n = pos0
	} else if err := h.ensureCap(pos0 + L); err != nil {
		return nil, h, err
	}
	if h.n != pos0 {
		return nil, h, core.NewError("native.AttnBF16FullLayerDevice: position desync with resident cache")
	}

	y := make([]float32, L*D)
	var encErr error
	withAutoreleasePool(func() {
		alloc := func(nBytes int) *pinnedNoCopyBytes {
			if encErr != nil {
				return nil
			}
			b, err := newPinnedNoCopyBytes(nBytes)
			if err != nil {
				encErr = err
			}
			return b
		}
		xPin := alloc(L * D * 4)
		normed := alloc(L * D * 4)
		nBF := alloc(L * max(D, FF) * bf16Size)
		qRaw := alloc(L * qCols * 4)
		qRawBF := alloc(L * qCols * bf16Size)
		kRaw := alloc(L * KVH * HD * 4)
		kRawBF := alloc(L * KVH * HD * bf16Size)
		vRaw := alloc(L * KVH * HD * 4)
		vRawBF := alloc(L * KVH * HD * bf16Size)
		qPrep := alloc(L * H * HD * 4)
		gateBuf := alloc(L * H * HD * 4)
		attnOut := alloc(L * H * HD * 4)
		attnBF := alloc(L * mixCols * bf16Size)
		mix := alloc(L * D * 4)
		mixBF := alloc(L * D * bf16Size)
		gFF := alloc(L * FF * 4)
		gFFBF := alloc(L * FF * bf16Size)
		uFF := alloc(L * FF * 4)
		uFFBF := alloc(L * FF * bf16Size)
		sFF := alloc(L * FF * 4)
		outP := alloc(L * D * 4)
		if encErr != nil {
			return
		}
		copy(xPin.bytes, float32Bytes(x))
		inNormBuf := residentFloat32(inputNorm)
		postNormBuf := residentFloat32(postNorm)
		normQ, normK := inNormBuf, inNormBuf
		if qkNorm == 1 {
			normQ = residentFloat32(qNormW)
			normK = residentFloat32(kNormW)
		}
		rmsName := "rmsfloat32"
		if D > rmsLoopedLimit {
			rmsName = "rms_loopedfloat32"
		}
		psoRMS, perr := pipelineFor(rmsName)
		if perr != nil {
			encErr = perr
			return
		}
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		fail := func(err error) { encErr = err; endEncodingFast(enc) }

		emitRMSNormRows(encSink{enc}, psoRMS, xPin.buf, inNormBuf, normed.buf, 0, 0, 0, D, eps, L, rmsThreadgroup(D, psoRMS))
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encNarrowF32ToBF16(enc, normed.buf, nBF.buf, L*D); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encProjBF16From(enc, qw, nBF.buf, qRawBF.buf, qRaw.buf, L, qCols, D); err != nil {
			fail(err)
			return
		}
		if err := encProjBF16From(enc, kw, nBF.buf, kRawBF.buf, kRaw.buf, L, KVH*HD, D); err != nil {
			fail(err)
			return
		}
		if err := encProjBF16From(enc, vw, nBF.buf, vRawBF.buf, vRaw.buf, L, KVH*HD, D); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encAttnQPrep(enc, qRaw.buf, normQ, qPrep.buf, gateBuf.buf, L, H, HD, RD, gated, qkNorm, eps, theta, pos0); err != nil {
			fail(err)
			return
		}
		if err := encAttnKPrep(enc, kRaw.buf, normK, h.kBuf.buf, L, KVH, HD, RD, qkNorm, eps, theta, pos0); err != nil {
			fail(err)
			return
		}
		endEncodingFast(enc)
		blit := blitCommandEncoderFast(cb)
		blit.CopyFromBufferSourceOffsetToBufferDestinationOffsetSize(vRaw.buf, 0, h.vBuf.buf, uint(pos0*KVH*HD*4), uint(L*KVH*HD*4))
		endBlitEncodingFast(blit)
		enc = computeCommandEncoderFast(cb)
		if err := encAttnSDPA(enc, qPrep.buf, h.kBuf.buf, h.vBuf.buf, attnOut.buf, L, H, KVH, HD, pos0, window); err != nil {
			fail(err)
			return
		}
		if gated != 0 {
			memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
			if err := encAttnGateSilu(enc, attnOut.buf, gateBuf.buf, L*H*HD); err != nil {
				fail(err)
				return
			}
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encNarrowF32ToBF16(enc, attnOut.buf, attnBF.buf, L*mixCols); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encProjBF16From(enc, ow, attnBF.buf, mixBF.buf, mix.buf, L, D, mixCols); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encResidualNormMLPBF16Tail(enc, quantTailBufs{
			h: xPin.buf, mix: mix.buf, normed: normed.buf, nBF: nBF.buf,
			g: gFF.buf, gBF: gFFBF.buf, u: uFF.buf, uBF: uFFBF.buf, s: sFF.buf, out: outP.buf,
		}, postNormBuf, gate, up, down, L, D, FF, eps); err != nil {
			fail(err)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(y, unsafe.Slice((*float32)(unsafe.Pointer(&outP.bytes[0])), L*D))
	})
	if encErr != nil {
		return nil, h, encErr
	}
	h.n = pos0 + L
	return y, h, nil
}

// attnKVExportHook reads the resident cache back into host slices — the snapshot/clone seam.
func attnKVExportHook(dev any) (k, v []float32, n int, ok bool) {
	h, isH := dev.(*attnKVDeviceState)
	if !isH || h.kBuf == nil {
		return nil, nil, 0, false
	}
	rows := h.n * h.KVH * h.HD
	k = make([]float32, rows)
	v = make([]float32, rows)
	copy(k, unsafe.Slice((*float32)(unsafe.Pointer(&h.kBuf.bytes[0])), rows))
	copy(v, unsafe.Slice((*float32)(unsafe.Pointer(&h.vBuf.bytes[0])), rows))
	return k, v, h.n, true
}
