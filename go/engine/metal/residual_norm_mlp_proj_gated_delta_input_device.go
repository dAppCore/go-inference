// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// residual_norm_mlp_proj_gated_delta_input_device.go — the input-side mirror of
// residual_norm_mlp_proj_device.go for a gated-delta NEXT layer: where that file folds the CURRENT layer's
// mixer output projection onto the FRONT of its own FFN tail, this one folds the NEXT layer's input
// RMSNorm + gated-delta input projections (in_proj_qkv/z/a/b) onto the BACK of the current layer's tail.
// The current layer's o_proj/out_proj → residual → post-attn RMSNorm → SwiGLU → residual (= y, the next
// layer's hidden) all run exactly as ResidualNormMLPProjDevice, then y feeds STRAIGHT into the next layer's
// RMSNorm and its four input projections without leaving the device — where the unfused boundary pays a
// host RMSNorm pass plus a standalone GatedDeltaInputDevice command buffer per layer. y and the four
// projections all cross back to the host together (y for the next layer's mixer-output residual add; the
// projections feed that layer's causal conv + gated-delta recurrence). Arch-neutral: named for the op
// sequence (proj+tail+gated-delta-input), not for any one model.

// residualNormMLPProjGDInputScratch extends residualNormMLPProjScratch's shape with the next layer's
// RMSNorm output and its four projection outputs/params.
type residualNormMLPProjGDInputScratch struct {
	mh, h, mix, normed, g, u, s, out *pinnedNoCopyBytes
	paramsProj, paramsGU, paramsD    *pinnedNoCopyBytes
	nextNormed, nqkv, nz, na, nb     *pinnedNoCopyBytes
	pNQKV, pNZ, pNA, pNB             *pinnedNoCopyBytes
	paramsFilled                     bool
}

type residualNormMLPProjGDInputKey struct{ L, D, mixCols, FF, nextConvDim, nextVDim, nextVH int }

var residualNormMLPProjGDInputPools sync.Map // residualNormMLPProjGDInputKey -> *sync.Pool

func getResidualNormMLPProjGDInputScratch(L, D, mixCols, FF, nextConvDim, nextVDim, nextVH int) (*residualNormMLPProjGDInputScratch, error) {
	key := residualNormMLPProjGDInputKey{L, D, mixCols, FF, nextConvDim, nextVDim, nextVH}
	poolAny, ok := residualNormMLPProjGDInputPools.Load(key)
	if !ok {
		poolAny, _ = residualNormMLPProjGDInputPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*residualNormMLPProjGDInputScratch), nil
	}
	sc := &residualNormMLPProjGDInputScratch{}
	var err error
	alloc := func(n int) *pinnedNoCopyBytes {
		if err != nil {
			return nil
		}
		var b *pinnedNoCopyBytes
		b, err = newPinnedNoCopyBytes(n)
		return b
	}
	sc.mh = alloc(L * mixCols * 4)
	sc.h = alloc(L * D * 4)
	sc.mix = alloc(L * D * 4)
	sc.normed = alloc(L * D * 4)
	sc.g = alloc(L * FF * 4)
	sc.u = alloc(L * FF * 4)
	sc.s = alloc(L * FF * 4)
	sc.out = alloc(L * D * 4)
	sc.paramsProj = alloc(72)
	sc.paramsGU = alloc(72)
	sc.paramsD = alloc(72)
	sc.nextNormed = alloc(L * D * 4)
	sc.nqkv = alloc(L * nextConvDim * 4)
	sc.nz = alloc(L * nextVDim * 4)
	sc.na = alloc(L * nextVH * 4)
	sc.nb = alloc(L * nextVH * 4)
	sc.pNQKV = alloc(72)
	sc.pNZ = alloc(72)
	sc.pNA = alloc(72)
	sc.pNB = alloc(72)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putResidualNormMLPProjGDInputScratch(L, D, mixCols, FF, nextConvDim, nextVDim, nextVH int, sc *residualNormMLPProjGDInputScratch) {
	if v, ok := residualNormMLPProjGDInputPools.Load(residualNormMLPProjGDInputKey{L, D, mixCols, FF, nextConvDim, nextVDim, nextVH}); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// ResidualNormMLPProjGatedDeltaInputDevice computes the current layer's projection-fused FFN tail —
// identical to ResidualNormMLPProjDevice — AND, in the SAME command buffer, the next layer's input RMSNorm
// plus its four gated-delta input projections:
//
//	y   = ResidualNormMLPProjDevice(...)                    // this layer's output = next layer's hidden
//	n   = RMSNorm(y, nextNormW)                             // next layer's input norm
//	qkv = n @ nextQKVWᵀ, z = n @ nextZWᵀ, a = n @ nextAWᵀ, b = n @ nextBWᵀ
//
// Returns y [L,D] (needed on the host for the next layer's mixer-output residual add) plus qkv
// [L,nextConvDim], z [L,nextVDim], a/b [L,nextVH] — mirrors attn.GatedDeltaInputDevice's own output
// shapes. The caller resumes that layer's gated-delta block from these precomputed projections instead of
// recomputing them (see attn.GatedDeltaForwardScratchFromInputF32).
func ResidualNormMLPProjGatedDeltaInputDevice(
	mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32,
	nextNormW, nextQKVW, nextZW, nextAW, nextBW []float32, nextConvDim, nextVDim, nextVH int,
) (y, qkv, z, a, b []float32, err error) {
	if err := ensureInit(); err != nil {
		return nil, nil, nil, nil, nil, err
	}
	if len(mixerHidden) != L*mixCols || len(projW) != D*mixCols || len(h) != L*D || len(normW) != D ||
		len(gate) != FF*D || len(up) != FF*D || len(down) != D*FF {
		return nil, nil, nil, nil, nil, core.NewError("native.ResidualNormMLPProjGatedDeltaInputDevice: tail size mismatch")
	}
	if len(nextNormW) != D || len(nextQKVW) != nextConvDim*D || len(nextZW) != nextVDim*D ||
		len(nextAW) != nextVH*D || len(nextBW) != nextVH*D {
		return nil, nil, nil, nil, nil, core.NewError("native.ResidualNormMLPProjGatedDeltaInputDevice: next-input size mismatch")
	}
	t := steelNT
	psoProj, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, D%t.bn == 0, mixCols%t.bk == 0)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	psoGU, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, FF%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	psoD, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, D%t.bn == 0, FF%t.bk == 0)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	psoAdd, err := pipelineFor("vv_Addfloat32")
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	rmsName := "rmsfloat32"
	if D > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32"
	}
	psoRMS, err := pipelineFor(rmsName)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	psoSig, err := pipelineFor("v_Sigmoidfloat32float32")
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	psoMul, err := pipelineFor("vv_Multiplyfloat32")
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	psoNQKV, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, nextConvDim%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	psoNZ, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, nextVDim%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	psoNA, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, nextVH%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	psoNB := psoNA // a and b share N=nextVH ⇒ identical alignment ⇒ same pipeline.

	y = make([]float32, L*D)
	qkv = make([]float32, L*nextConvDim)
	z = make([]float32, L*nextVDim)
	a = make([]float32, L*nextVH)
	b = make([]float32, L*nextVH)
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getResidualNormMLPProjGDInputScratch(L, D, mixCols, FF, nextConvDim, nextVDim, nextVH)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putResidualNormMLPProjGDInputScratch(L, D, mixCols, FF, nextConvDim, nextVDim, nextVH, sc)
		mhBuf, cerr := sc.mh.copyBuffer(float32Bytes(mixerHidden))
		if cerr != nil {
			encErr = cerr
			return
		}
		hBuf, cerr := sc.h.copyBuffer(float32Bytes(h))
		if cerr != nil {
			encErr = cerr
			return
		}
		if !sc.paramsFilled {
			tnProj, tmProj := (D+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
			fillMatMulF32SteelParams(sc.paramsProj.bytes, L, mixCols, D, mixCols, tnProj, tmProj, mixCols/t.bk)
			tnGU, tmGU := (FF+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
			fillMatMulF32SteelParams(sc.paramsGU.bytes, L, D, FF, D, tnGU, tmGU, D/t.bk)
			tnD, tmD := (D+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
			fillMatMulF32SteelParams(sc.paramsD.bytes, L, FF, D, FF, tnD, tmD, FF/t.bk)
			tmN := (L + t.bm - 1) / t.bm
			tnNQKV := (nextConvDim + t.bn - 1) / t.bn
			fillMatMulF32SteelParams(sc.pNQKV.bytes, L, D, nextConvDim, D, tnNQKV, tmN, D/t.bk)
			tnNZ := (nextVDim + t.bn - 1) / t.bn
			fillMatMulF32SteelParams(sc.pNZ.bytes, L, D, nextVDim, D, tnNZ, tmN, D/t.bk)
			tnNVH := (nextVH + t.bn - 1) / t.bn
			fillMatMulF32SteelParams(sc.pNA.bytes, L, D, nextVH, D, tnNVH, tmN, D/t.bk)
			fillMatMulF32SteelParams(sc.pNB.bytes, L, D, nextVH, D, tnNVH, tmN, D/t.bk)
			sc.paramsFilled = true
		}
		projBuf := residentFloat32(projW)
		normBuf := residentFloat32(normW)
		gateBuf := residentFloat32(gate)
		upBuf := residentFloat32(up)
		downBuf := residentFloat32(down)
		nextNormBuf := residentFloat32(nextNormW)
		nextQKVBuf := residentFloat32(nextQKVW)
		nextZBuf := residentFloat32(nextZW)
		nextABuf := residentFloat32(nextAW)
		nextBBuf := residentFloat32(nextBW)

		tnProj, tmProj := (D+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
		tnGU, tmGU := (FF+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
		tnD, tmD := (D+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
		tmN := (L + t.bm - 1) / t.bm
		tnNQKV := (nextConvDim + t.bn - 1) / t.bn
		tnNZ := (nextVDim + t.bn - 1) / t.bn
		tnNVH := (nextVH + t.bn - 1) / t.bn
		nD := L * D
		nFF := L * FF
		rmsTG := rmsThreadgroup(D, psoRMS)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		// [tail] identical to ResidualNormMLPProjDevice: out_proj → mixer residual → post-attn RMSNorm →
		// SwiGLU → MLP residual.
		emitSteelGemm(encSink{enc}, psoProj, mhBuf, projBuf, sc.mix.buf, sc.paramsProj.buf, tnProj, tmProj, uint(t.wn), uint(t.wm))
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitBinary(encSink{enc}, psoAdd, hBuf, 0, sc.mix.buf, 0, hBuf, 0, nD)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitRMSNormRows(encSink{enc}, psoRMS, hBuf, normBuf, sc.normed.buf, 0, 0, 0, D, eps, L, rmsTG)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitSteelGemm(encSink{enc}, psoGU, sc.normed.buf, gateBuf, sc.g.buf, sc.paramsGU.buf, tnGU, tmGU, uint(t.wn), uint(t.wm))
		emitSteelGemm(encSink{enc}, psoGU, sc.normed.buf, upBuf, sc.u.buf, sc.paramsGU.buf, tnGU, tmGU, uint(t.wn), uint(t.wm))
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitUnary(encSink{enc}, psoSig, sc.g.buf, sc.s.buf, nFF)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitBinary(encSink{enc}, psoMul, sc.s.buf, 0, sc.g.buf, 0, sc.s.buf, 0, nFF)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitBinary(encSink{enc}, psoMul, sc.s.buf, 0, sc.u.buf, 0, sc.s.buf, 0, nFF)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitSteelGemm(encSink{enc}, psoD, sc.s.buf, downBuf, sc.out.buf, sc.paramsD.buf, tnD, tmD, uint(t.wn), uint(t.wm))
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// y = hplus + mlpOut, in place into out.buf — this layer's output = the next layer's hidden.
		emitBinary(encSink{enc}, psoAdd, hBuf, 0, sc.out.buf, 0, sc.out.buf, 0, nD)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// [next-input] the next layer's RMSNorm reads y (sc.out) straight off the device, then its four
		// gated-delta input projections read the normed result — no host round-trip for either.
		emitRMSNormRows(encSink{enc}, psoRMS, sc.out.buf, nextNormBuf, sc.nextNormed.buf, 0, 0, 0, D, eps, L, rmsTG)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitSteelGemm(encSink{enc}, psoNQKV, sc.nextNormed.buf, nextQKVBuf, sc.nqkv.buf, sc.pNQKV.buf, tnNQKV, tmN, uint(t.wn), uint(t.wm))
		emitSteelGemm(encSink{enc}, psoNZ, sc.nextNormed.buf, nextZBuf, sc.nz.buf, sc.pNZ.buf, tnNZ, tmN, uint(t.wn), uint(t.wm))
		emitSteelGemm(encSink{enc}, psoNA, sc.nextNormed.buf, nextABuf, sc.na.buf, sc.pNA.buf, tnNVH, tmN, uint(t.wn), uint(t.wm))
		emitSteelGemm(encSink{enc}, psoNB, sc.nextNormed.buf, nextBBuf, sc.nb.buf, sc.pNB.buf, tnNVH, tmN, uint(t.wn), uint(t.wm))
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(y, unsafe.Slice((*float32)(unsafe.Pointer(&sc.out.bytes[0])), L*D))
		copy(qkv, unsafe.Slice((*float32)(unsafe.Pointer(&sc.nqkv.bytes[0])), L*nextConvDim))
		copy(z, unsafe.Slice((*float32)(unsafe.Pointer(&sc.nz.bytes[0])), L*nextVDim))
		copy(a, unsafe.Slice((*float32)(unsafe.Pointer(&sc.na.bytes[0])), L*nextVH))
		copy(b, unsafe.Slice((*float32)(unsafe.Pointer(&sc.nb.bytes[0])), L*nextVH))
	})
	if encErr != nil {
		return nil, nil, nil, nil, nil, encErr
	}
	return y, qkv, z, a, b, nil
}
