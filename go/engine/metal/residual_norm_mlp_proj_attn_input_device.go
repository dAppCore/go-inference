// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// residual_norm_mlp_proj_attn_input_device.go — the input-side mirror of residual_norm_mlp_proj_device.go
// for a full-attention NEXT layer: where that file folds the CURRENT layer's mixer output projection onto
// the FRONT of its own FFN tail, this one folds the NEXT layer's input RMSNorm + q/k/v input projections
// onto the BACK of the current layer's tail. The current layer's o_proj → residual → post-attn RMSNorm →
// SwiGLU → residual (= y, the next layer's hidden) all run exactly as ResidualNormMLPProjDevice, then y
// feeds STRAIGHT into the next layer's RMSNorm and its q/k/v projections without leaving the device — where
// the unfused boundary pays a host RMSNorm pass plus a standalone ComposedAttnQKVDevice command buffer per
// layer. y and the three projections all cross back to the host together (y for the next layer's
// mixer-output residual add; q/k/v feed that layer's QK-norm/rotary/attention). Arch-neutral: named for the
// op sequence (proj+tail+attn-input), not for any one model.

// residualNormMLPProjAttnInputScratch extends residualNormMLPProjScratch's shape with the next layer's
// RMSNorm output and its three projection outputs/params.
type residualNormMLPProjAttnInputScratch struct {
	mh, h, mix, normed, g, u, s, out *pinnedNoCopyBytes
	paramsProj, paramsGU, paramsD    *pinnedNoCopyBytes
	nextNormed, nq, nk, nv           *pinnedNoCopyBytes
	pNQ, pNKV                        *pinnedNoCopyBytes
	paramsFilled                     bool
}

type residualNormMLPProjAttnInputKey struct{ L, D, mixCols, FF, nextQCols, nextKVCols int }

var residualNormMLPProjAttnInputPools sync.Map // residualNormMLPProjAttnInputKey -> *sync.Pool

func getResidualNormMLPProjAttnInputScratch(L, D, mixCols, FF, nextQCols, nextKVCols int) (*residualNormMLPProjAttnInputScratch, error) {
	key := residualNormMLPProjAttnInputKey{L, D, mixCols, FF, nextQCols, nextKVCols}
	poolAny, ok := residualNormMLPProjAttnInputPools.Load(key)
	if !ok {
		poolAny, _ = residualNormMLPProjAttnInputPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*residualNormMLPProjAttnInputScratch), nil
	}
	sc := &residualNormMLPProjAttnInputScratch{}
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
	sc.nq = alloc(L * nextQCols * 4)
	sc.nk = alloc(L * nextKVCols * 4)
	sc.nv = alloc(L * nextKVCols * 4)
	sc.pNQ = alloc(72)
	sc.pNKV = alloc(72)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putResidualNormMLPProjAttnInputScratch(L, D, mixCols, FF, nextQCols, nextKVCols int, sc *residualNormMLPProjAttnInputScratch) {
	if v, ok := residualNormMLPProjAttnInputPools.Load(residualNormMLPProjAttnInputKey{L, D, mixCols, FF, nextQCols, nextKVCols}); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// ResidualNormMLPProjAttnInputDevice computes the current layer's projection-fused FFN tail — identical to
// ResidualNormMLPProjDevice — AND, in the SAME command buffer, the next layer's input RMSNorm plus its
// q/k/v projections:
//
//	y  = ResidualNormMLPProjDevice(...)              // this layer's output = next layer's hidden
//	n  = RMSNorm(y, nextNormW)                       // next layer's input norm
//	q  = n @ nextQWᵀ, k = n @ nextKWᵀ, v = n @ nextVWᵀ
//
// Returns y [L,D] (needed on the host for the next layer's mixer-output residual add) plus q
// [L,nextQCols], k/v [L,nextKVCols]. next{QCols,KVCols} mirror ComposedAttnQKVDevice's qCols/kvCols (qCols
// is 2·H·HD when the next layer's attention is gated). The caller resumes that layer's mixer from these
// precomputed projections instead of recomputing them (see composed.attnMixer.forwardFromQKV).
func ResidualNormMLPProjAttnInputDevice(
	mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32,
	nextNormW, nextQW, nextKW, nextVW []float32, nextQCols, nextKVCols int,
) (y, q, k, v []float32, err error) {
	if err := ensureInit(); err != nil {
		return nil, nil, nil, nil, err
	}
	if len(mixerHidden) != L*mixCols || len(projW) != D*mixCols || len(h) != L*D || len(normW) != D ||
		len(gate) != FF*D || len(up) != FF*D || len(down) != D*FF {
		return nil, nil, nil, nil, core.NewError("native.ResidualNormMLPProjAttnInputDevice: tail size mismatch")
	}
	if len(nextNormW) != D || len(nextQW) != nextQCols*D || len(nextKW) != nextKVCols*D || len(nextVW) != nextKVCols*D {
		return nil, nil, nil, nil, core.NewError("native.ResidualNormMLPProjAttnInputDevice: next-input size mismatch")
	}
	t := steelNT
	psoProj, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, D%t.bn == 0, mixCols%t.bk == 0)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	psoGU, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, FF%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	psoD, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, D%t.bn == 0, FF%t.bk == 0)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	psoAdd, err := pipelineFor("vv_Addfloat32")
	if err != nil {
		return nil, nil, nil, nil, err
	}
	rmsName := "rmsfloat32"
	if D > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32"
	}
	psoRMS, err := pipelineFor(rmsName)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	psoSig, err := pipelineFor("v_Sigmoidfloat32float32")
	if err != nil {
		return nil, nil, nil, nil, err
	}
	psoMul, err := pipelineFor("vv_Multiplyfloat32")
	if err != nil {
		return nil, nil, nil, nil, err
	}
	psoNQ, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, nextQCols%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	psoNKV, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, nextKVCols%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	y = make([]float32, L*D)
	q = make([]float32, L*nextQCols)
	k = make([]float32, L*nextKVCols)
	v = make([]float32, L*nextKVCols)
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getResidualNormMLPProjAttnInputScratch(L, D, mixCols, FF, nextQCols, nextKVCols)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putResidualNormMLPProjAttnInputScratch(L, D, mixCols, FF, nextQCols, nextKVCols, sc)
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
			tnNQ := (nextQCols + t.bn - 1) / t.bn
			fillMatMulF32SteelParams(sc.pNQ.bytes, L, D, nextQCols, D, tnNQ, tmN, D/t.bk)
			tnNKV := (nextKVCols + t.bn - 1) / t.bn
			fillMatMulF32SteelParams(sc.pNKV.bytes, L, D, nextKVCols, D, tnNKV, tmN, D/t.bk)
			sc.paramsFilled = true
		}
		projBuf := residentFloat32(projW)
		normBuf := residentFloat32(normW)
		gateBuf := residentFloat32(gate)
		upBuf := residentFloat32(up)
		downBuf := residentFloat32(down)
		nextNormBuf := residentFloat32(nextNormW)
		nextQBuf := residentFloat32(nextQW)
		nextKBuf := residentFloat32(nextKW)
		nextVBuf := residentFloat32(nextVW)

		tnProj, tmProj := (D+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
		tnGU, tmGU := (FF+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
		tnD, tmD := (D+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
		tmN := (L + t.bm - 1) / t.bm
		tnNQ := (nextQCols + t.bn - 1) / t.bn
		tnNKV := (nextKVCols + t.bn - 1) / t.bn
		nD := L * D
		nFF := L * FF
		rmsTG := rmsThreadgroup(D, psoRMS)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		// [tail] identical to ResidualNormMLPProjDevice: o_proj → mixer residual → post-attn RMSNorm →
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
		// [next-input] the next layer's RMSNorm reads y (sc.out) straight off the device, then its three
		// q/k/v projections read the normed result — no host round-trip for either.
		emitRMSNormRows(encSink{enc}, psoRMS, sc.out.buf, nextNormBuf, sc.nextNormed.buf, 0, 0, 0, D, eps, L, rmsTG)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitSteelGemm(encSink{enc}, psoNQ, sc.nextNormed.buf, nextQBuf, sc.nq.buf, sc.pNQ.buf, tnNQ, tmN, uint(t.wn), uint(t.wm))
		emitSteelGemm(encSink{enc}, psoNKV, sc.nextNormed.buf, nextKBuf, sc.nk.buf, sc.pNKV.buf, tnNKV, tmN, uint(t.wn), uint(t.wm))
		emitSteelGemm(encSink{enc}, psoNKV, sc.nextNormed.buf, nextVBuf, sc.nv.buf, sc.pNKV.buf, tnNKV, tmN, uint(t.wn), uint(t.wm))
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(y, unsafe.Slice((*float32)(unsafe.Pointer(&sc.out.bytes[0])), L*D))
		copy(q, unsafe.Slice((*float32)(unsafe.Pointer(&sc.nq.bytes[0])), L*nextQCols))
		copy(k, unsafe.Slice((*float32)(unsafe.Pointer(&sc.nk.bytes[0])), L*nextKVCols))
		copy(v, unsafe.Slice((*float32)(unsafe.Pointer(&sc.nv.bytes[0])), L*nextKVCols))
	})
	if encErr != nil {
		return nil, nil, nil, nil, encErr
	}
	return y, q, k, v, nil
}
