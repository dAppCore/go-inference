// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/arch/Qwen/qwen3"
	"github.com/tmc/apple/metal"
)

// composed_chain_backend.go — whole-token chaining for the composed lane (#26): every layer's
// encodes land on ONE retained command buffer, the hidden state ping-pongs between two device
// buffers, and the host sees exactly one upload (Begin) and one wait+readback (End) per forward.
// The per-layer folds this chains are the proven single-CB layer bodies; encoder boundaries
// within the shared CB order the layers (a Metal queue executes a CB's encoders serially), so no
// cross-layer barriers are needed. The CB is RETAINED across the hook calls (the lane_set
// pool-boundary trap: an autoreleased cb dies when a step's pool drains, and a later wait hangs).
type composedChainCtx struct {
	cb       metal.MTLCommandBufferObject
	hA, hB   *pinnedNoCopyBytes
	curIsA   bool
	L, D     int
	gdSc     *gatedDeltaQuantLayerScratch
	gdKey    gatedDeltaQuantLayerKey
	attnPins []*pinnedNoCopyBytes
}

func (c *composedChainCtx) cur() *pinnedNoCopyBytes {
	if c.curIsA {
		return c.hA
	}
	return c.hB
}

func (c *composedChainCtx) next() *pinnedNoCopyBytes {
	if c.curIsA {
		return c.hB
	}
	return c.hA
}

// ComposedChainBeginDevice opens a chained forward: uploads h [L,D] once and hands back the
// opaque context every chained layer encodes into.
func ComposedChainBeginDevice(h []float32, L, D int) (any, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if L <= 0 || len(h) != L*D {
		return nil, core.NewError("native.ComposedChainBeginDevice: size mismatch")
	}
	hA, err := newPinnedNoCopyBytes(L * D * 4)
	if err != nil {
		return nil, err
	}
	hB, err := newPinnedNoCopyBytes(L * D * 4)
	if err != nil {
		return nil, err
	}
	copy(hA.bytes, float32Bytes(h))
	var cb metal.MTLCommandBufferObject
	withAutoreleasePool(func() {
		cb = commandBufferFast(queue)
		cb.Retain() // survives the per-step autorelease pools until End releases it
	})
	return &composedChainCtx{cb: cb, hA: hA, hB: hB, curIsA: true, L: L, D: D}, nil
}

// ComposedChainEndDevice commits the chained forward, waits once, and returns the final hidden.
func ComposedChainEndDevice(ctxAny any) ([]float32, error) {
	ctx, ok := ctxAny.(*composedChainCtx)
	if !ok || ctx.cb.GetID() == 0 {
		return nil, core.NewError("native.ComposedChainEndDevice: not a chain context")
	}
	var y []float32
	withAutoreleasePool(func() {
		commitCommandBufferFast(ctx.cb)
		waitUntilCompletedFast(ctx.cb)
		y = make([]float32, ctx.L*ctx.D)
		copy(y, unsafe.Slice((*float32)(unsafe.Pointer(&ctx.cur().bytes[0])), ctx.L*ctx.D))
		ctx.cb.Release()
		ctx.cb = metal.MTLCommandBufferObject{}
	})
	if ctx.gdSc != nil {
		putGatedDeltaQuantLayerScratch(ctx.gdKey, ctx.gdSc)
		ctx.gdSc = nil
	}
	return y, nil
}

// gatedDeltaBF16ChainLayerDevice encodes one dense bf16 gated-delta layer onto the chain — the
// gatedDeltaBF16LayerRun body against the context's ping-pong hidden, sharing one pooled scratch
// across every gd layer of the chain (encoder boundaries serialise their reuse).
func gatedDeltaBF16ChainLayerDevice(ctxAny any, sc *qwen3.GatedDeltaScratch, inputNorm []float32, w *qwen3.GatedDeltaWeights, cfg qwen3.GatedDeltaConfig, postNorm []float32, gate, up, down *model.BF16Weight, priorConv, priorDelta []float32, FF int, eps float32) error {
	ctx, ok := ctxAny.(*composedChainCtx)
	if !ok {
		return core.NewError("native.gatedDeltaBF16ChainLayerDevice: not a chain context")
	}
	h, _ := sc.Device.(*gatedDeltaDeviceState)
	if h == nil {
		if !gatedDeltaBlockUsable(cfg.HeadDim, cfg.HeadDim, cfg.KeyHeads, cfg.ValueHeads, cfg.ConvKernel) {
			return core.NewError("native.gatedDeltaBF16ChainLayerDevice: geometry not servable")
		}
		nh, err := newGatedDeltaDeviceState(cfg.KeyHeads, cfg.ValueHeads, cfg.HeadDim, cfg.HeadDim, cfg.ConvKernel, 1)
		if err != nil {
			return err
		}
		h = nh
	}
	L, D := ctx.L, ctx.D
	convDim, vDim := h.convDim, h.Hv*h.Dv
	if !bf16GeometryOK(w.InProjQKVB, convDim, D) || !bf16GeometryOK(w.InProjZB, vDim, D) ||
		!bf16GeometryOK(w.InProjAB, h.Hv, D) || !bf16GeometryOK(w.InProjBB, h.Hv, D) ||
		!bf16GeometryOK(w.OutProjB, D, vDim) ||
		!bf16GeometryOK(gate, FF, D) || !bf16GeometryOK(up, FF, D) || !bf16GeometryOK(down, D, FF) {
		return core.NewError("native.gatedDeltaBF16ChainLayerDevice: unsupported bf16 geometry")
	}
	if !h.valid {
		h.prime(priorConv, priorDelta)
	}
	key := gatedDeltaQuantLayerKey{L: L, D: D, FF: FF, Hk: h.Hk, Hv: h.Hv, Dk: h.Dk, K: h.K}
	if ctx.gdSc == nil {
		gsc, err := getGatedDeltaQuantLayerScratch(key)
		if err != nil {
			return err
		}
		ctx.gdSc, ctx.gdKey = gsc, key
	} else if ctx.gdKey != key {
		return core.NewError("native.gatedDeltaBF16ChainLayerDevice: mixed gd geometries in one chain")
	}
	gsc := ctx.gdSc
	rmsName := "rmsfloat32"
	if D > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32"
	}
	psoRMS, err := pipelineFor(rmsName)
	if err != nil {
		return err
	}
	var encErr error
	withAutoreleasePool(func() {
		wConv := residentFloat32(w.ConvWeight)
		wBias := wConv
		hasBias := 0
		if w.ConvBias != nil {
			wBias = residentFloat32(w.ConvBias)
			hasBias = 1
		}
		enc := computeCommandEncoderFast(ctx.cb)
		fail := func(err error) { encErr = err; endEncodingFast(enc) }
		emitRMSNormRows(encSink{enc}, psoRMS, ctx.cur().buf, residentFloat32(inputNorm), gsc.normed1.buf, 0, 0, 0, D, eps, L, rmsThreadgroup(D, psoRMS))
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encNarrowF32ToBF16(enc, gsc.normed1.buf, gsc.n1BF.buf, L*D); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encProjBF16From(enc, w.InProjQKVB, gsc.n1BF.buf, gsc.qkvBF.buf, gsc.qkv.buf, L, convDim, D); err != nil {
			fail(err)
			return
		}
		if err := encProjBF16From(enc, w.InProjZB, gsc.n1BF.buf, gsc.zBF.buf, gsc.z.buf, L, vDim, D); err != nil {
			fail(err)
			return
		}
		if err := encProjBF16From(enc, w.InProjAB, gsc.n1BF.buf, gsc.aBF.buf, gsc.a.buf, L, h.Hv, D); err != nil {
			fail(err)
			return
		}
		if err := encProjBF16From(enc, w.InProjBB, gsc.n1BF.buf, gsc.bBF.buf, gsc.b.buf, L, h.Hv, D); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encGatedDeltaBlockStages(enc, h, gdBlockStageBufs{
			qkv: gsc.qkv.buf, z: gsc.z.buf, a: gsc.a.buf, b: gsc.b.buf,
			qN: gsc.qN.buf, kN: gsc.kN.buf, vN: gsc.vN.buf, g: gsc.g.buf, beta: gsc.beta.buf,
			gated: gsc.gated.buf,
		}, wConv, wBias, hasBias, residentFloat32(w.ALog), residentFloat32(w.DtBias), residentFloat32(w.Norm), L); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encProjBF16F32(enc, w.OutProjB, gsc.gated.buf, gsc.gatedBF.buf, gsc.mixBF.buf, gsc.mix.buf, L, D, vDim); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encResidualNormMLPBF16Tail(enc, quantTailBufs{
			h: ctx.cur().buf, mix: gsc.mix.buf, normed: gsc.normed2.buf, nBF: gsc.n2BF.buf,
			g: gsc.gFF.buf, gBF: gsc.gFFBF.buf, u: gsc.uFF.buf, uBF: gsc.uFFBF.buf, s: gsc.sFF.buf, out: ctx.next().buf,
		}, residentFloat32(postNorm), gate, up, down, L, D, FF, eps); err != nil {
			fail(err)
			return
		}
		endEncodingFast(enc)
	})
	if encErr != nil {
		return encErr
	}
	sc.Device = h
	ctx.curIsA = !ctx.curIsA
	return nil
}

// attnBF16ChainLayerDevice encodes one dense bf16 attention layer (device-KV) onto the chain —
// the AttnBF16FullLayerDevice body against the ping-pong hidden.
func attnBF16ChainLayerDevice(ctxAny, dev any, inputNorm []float32, qw, kw, vw, ow *model.BF16Weight, qNormW, kNormW, postNorm []float32, gate, up, down *model.BF16Weight, priorK, priorV []float32, H, KVH, HD, RD, pos0, window, gated, qkNorm, FF int, eps, theta float32) (any, error) {
	ctx, ok := ctxAny.(*composedChainCtx)
	if !ok {
		return dev, core.NewError("native.attnBF16ChainLayerDevice: not a chain context")
	}
	L, D := ctx.L, ctx.D
	if !attnCoreUsable(H, KVH, HD, RD) {
		return dev, core.NewError("native.attnBF16ChainLayerDevice: core not servable")
	}
	qCols := H * HD
	if gated != 0 {
		qCols = 2 * H * HD
	}
	mixCols := H * HD
	if !bf16GeometryOK(qw, qCols, D) || !bf16GeometryOK(kw, KVH*HD, D) || !bf16GeometryOK(vw, KVH*HD, D) ||
		!bf16GeometryOK(ow, D, mixCols) || !bf16GeometryOK(gate, FF, D) || !bf16GeometryOK(up, FF, D) || !bf16GeometryOK(down, D, FF) {
		return dev, core.NewError("native.attnBF16ChainLayerDevice: size/geometry mismatch")
	}
	h, _ := dev.(*attnKVDeviceState)
	if h == nil {
		h = &attnKVDeviceState{KVH: KVH, HD: HD}
		if err := h.ensureCap(pos0 + L); err != nil {
			return dev, err
		}
		if pos0 > 0 {
			if len(priorK) != pos0*KVH*HD || len(priorV) != pos0*KVH*HD {
				return dev, core.NewError("native.attnBF16ChainLayerDevice: prior state size mismatch")
			}
			copy(h.kBuf.bytes, float32Bytes(priorK))
			copy(h.vBuf.bytes, float32Bytes(priorV))
		}
		h.n = pos0
	} else if err := h.ensureCap(pos0 + L); err != nil {
		return h, err
	}
	if h.n != pos0 {
		return h, core.NewError("native.attnBF16ChainLayerDevice: position desync with resident cache")
	}
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
			ctx.attnPins = append(ctx.attnPins, b)
			return b
		}
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
		if encErr != nil {
			return
		}
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
		enc := computeCommandEncoderFast(ctx.cb)
		fail := func(err error) { encErr = err; endEncodingFast(enc) }
		emitRMSNormRows(encSink{enc}, psoRMS, ctx.cur().buf, inNormBuf, normed.buf, 0, 0, 0, D, eps, L, rmsThreadgroup(D, psoRMS))
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
		blit := blitCommandEncoderFast(ctx.cb)
		blit.CopyFromBufferSourceOffsetToBufferDestinationOffsetSize(vRaw.buf, 0, h.vBuf.buf, uint(pos0*KVH*HD*4), uint(L*KVH*HD*4))
		endBlitEncodingFast(blit)
		enc = computeCommandEncoderFast(ctx.cb)
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
			h: ctx.cur().buf, mix: mix.buf, normed: normed.buf, nBF: nBF.buf,
			g: gFF.buf, gBF: gFFBF.buf, u: uFF.buf, uBF: uFFBF.buf, s: sFF.buf, out: ctx.next().buf,
		}, postNormBuf, gate, up, down, L, D, FF, eps); err != nil {
			fail(err)
			return
		}
		endEncodingFast(enc)
	})
	if encErr != nil {
		return h, encErr
	}
	h.n = pos0 + L
	ctx.curIsA = !ctx.curIsA
	return h, nil
}
