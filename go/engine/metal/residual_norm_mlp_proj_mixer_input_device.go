// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// mixerInputScratch is the pooled pinned staging for the Mamba-2 and RWKV-7 input-side mirrors. Both
// share the same projection-fused tail; only the number and widths of the next layer's input GEMMs vary.
type mixerInputScratch struct {
	mh, h, mix, normed, g, u, s, out, nextNormed *pinnedNoCopyBytes
	paramsProj, paramsGU, paramsD                *pinnedNoCopyBytes
	nextOut, nextParams                          [6]*pinnedNoCopyBytes
	paramsFilled                                 bool
}

type mixerInputKey struct {
	L, D, mixCols, FF, count int
	dims                     [6]int
}

var mixerInputPools sync.Map

func getMixerInputScratch(key mixerInputKey) (*mixerInputScratch, error) {
	poolAny, ok := mixerInputPools.Load(key)
	if !ok {
		poolAny, _ = mixerInputPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*mixerInputScratch), nil
	}
	sc := &mixerInputScratch{}
	var err error
	alloc := func(n int) *pinnedNoCopyBytes {
		if err != nil {
			return nil
		}
		var b *pinnedNoCopyBytes
		b, err = newPinnedNoCopyBytes(n)
		return b
	}
	sc.mh = alloc(key.L * key.mixCols * 4)
	sc.h = alloc(key.L * key.D * 4)
	sc.mix = alloc(key.L * key.D * 4)
	sc.normed = alloc(key.L * key.D * 4)
	sc.g = alloc(key.L * key.FF * 4)
	sc.u = alloc(key.L * key.FF * 4)
	sc.s = alloc(key.L * key.FF * 4)
	sc.out = alloc(key.L * key.D * 4)
	sc.nextNormed = alloc(key.L * key.D * 4)
	sc.paramsProj = alloc(72)
	sc.paramsGU = alloc(72)
	sc.paramsD = alloc(72)
	for i := range key.count {
		sc.nextOut[i] = alloc(key.L * key.dims[i] * 4)
		sc.nextParams[i] = alloc(72)
	}
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putMixerInputScratch(key mixerInputKey, sc *mixerInputScratch) {
	if v, ok := mixerInputPools.Load(key); ok {
		v.(*sync.Pool).Put(sc)
	}
}

func residualNormMLPProjMixerInputDevice(
	mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32,
	nextNormW []float32, weights [6][]float32, dims [6]int, count int,
) (y []float32, projected [6][]float32, err error) {
	if err := ensureInit(); err != nil {
		return nil, projected, err
	}
	if count < 1 || count > len(weights) || len(mixerHidden) != L*mixCols || len(projW) != D*mixCols ||
		len(h) != L*D || len(normW) != D || len(gate) != FF*D || len(up) != FF*D ||
		len(down) != D*FF || len(nextNormW) != D {
		return nil, projected, core.NewError("native.residualNormMLPProjMixerInputDevice: size mismatch")
	}
	for i := range count {
		if dims[i] <= 0 || len(weights[i]) != dims[i]*D {
			return nil, projected, core.NewError("native.residualNormMLPProjMixerInputDevice: next-input size mismatch")
		}
	}
	t := steelNT
	psoProj, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, D%t.bn == 0, mixCols%t.bk == 0)
	if err != nil {
		return nil, projected, err
	}
	psoGU, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, FF%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, projected, err
	}
	psoD, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, D%t.bn == 0, FF%t.bk == 0)
	if err != nil {
		return nil, projected, err
	}
	psoAdd, err := pipelineFor("vv_Addfloat32")
	if err != nil {
		return nil, projected, err
	}
	rmsName := "rmsfloat32"
	if D > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32"
	}
	psoRMS, err := pipelineFor(rmsName)
	if err != nil {
		return nil, projected, err
	}
	psoSig, err := pipelineFor("v_Sigmoidfloat32float32")
	if err != nil {
		return nil, projected, err
	}
	psoMul, err := pipelineFor("vv_Multiplyfloat32")
	if err != nil {
		return nil, projected, err
	}
	var nextPSO [6]metal.MTLComputePipelineState
	for i := range count {
		nextPSO[i], err = steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, dims[i]%t.bn == 0, D%t.bk == 0)
		if err != nil {
			return nil, projected, err
		}
	}
	y = make([]float32, L*D)
	for i := range count {
		projected[i] = make([]float32, L*dims[i])
	}
	key := mixerInputKey{L: L, D: D, mixCols: mixCols, FF: FF, count: count, dims: dims}
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getMixerInputScratch(key)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putMixerInputScratch(key, sc)
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
			fillMatMulF32SteelParams(sc.paramsProj.bytes, L, mixCols, D, mixCols, (D+t.bn-1)/t.bn, (L+t.bm-1)/t.bm, mixCols/t.bk)
			fillMatMulF32SteelParams(sc.paramsGU.bytes, L, D, FF, D, (FF+t.bn-1)/t.bn, (L+t.bm-1)/t.bm, D/t.bk)
			fillMatMulF32SteelParams(sc.paramsD.bytes, L, FF, D, FF, (D+t.bn-1)/t.bn, (L+t.bm-1)/t.bm, FF/t.bk)
			for i := range count {
				fillMatMulF32SteelParams(sc.nextParams[i].bytes, L, D, dims[i], D, (dims[i]+t.bn-1)/t.bn, (L+t.bm-1)/t.bm, D/t.bk)
			}
			sc.paramsFilled = true
		}
		projBuf := residentFloat32(projW)
		normBuf := residentFloat32(normW)
		gateBuf := residentFloat32(gate)
		upBuf := residentFloat32(up)
		downBuf := residentFloat32(down)
		nextNormBuf := residentFloat32(nextNormW)
		var nextBuf [6]metal.MTLBuffer
		for i := range count {
			nextBuf[i] = residentFloat32(weights[i])
		}
		tm := (L + t.bm - 1) / t.bm
		nD, nFF := L*D, L*FF
		rmsTG := rmsThreadgroup(D, psoRMS)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitSteelGemm(encSink{enc}, psoProj, mhBuf, projBuf, sc.mix.buf, sc.paramsProj.buf, (D+t.bn-1)/t.bn, tm, uint(t.wn), uint(t.wm))
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitBinary(encSink{enc}, psoAdd, hBuf, 0, sc.mix.buf, 0, hBuf, 0, nD)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitRMSNormRows(encSink{enc}, psoRMS, hBuf, normBuf, sc.normed.buf, 0, 0, 0, D, eps, L, rmsTG)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitSteelGemm(encSink{enc}, psoGU, sc.normed.buf, gateBuf, sc.g.buf, sc.paramsGU.buf, (FF+t.bn-1)/t.bn, tm, uint(t.wn), uint(t.wm))
		emitSteelGemm(encSink{enc}, psoGU, sc.normed.buf, upBuf, sc.u.buf, sc.paramsGU.buf, (FF+t.bn-1)/t.bn, tm, uint(t.wn), uint(t.wm))
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitUnary(encSink{enc}, psoSig, sc.g.buf, sc.s.buf, nFF)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitBinary(encSink{enc}, psoMul, sc.s.buf, 0, sc.g.buf, 0, sc.s.buf, 0, nFF)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitBinary(encSink{enc}, psoMul, sc.s.buf, 0, sc.u.buf, 0, sc.s.buf, 0, nFF)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitSteelGemm(encSink{enc}, psoD, sc.s.buf, downBuf, sc.out.buf, sc.paramsD.buf, (D+t.bn-1)/t.bn, tm, uint(t.wn), uint(t.wm))
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitBinary(encSink{enc}, psoAdd, hBuf, 0, sc.out.buf, 0, sc.out.buf, 0, nD)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitRMSNormRows(encSink{enc}, psoRMS, sc.out.buf, nextNormBuf, sc.nextNormed.buf, 0, 0, 0, D, eps, L, rmsTG)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		for i := range count {
			emitSteelGemm(encSink{enc}, nextPSO[i], sc.nextNormed.buf, nextBuf[i], sc.nextOut[i].buf, sc.nextParams[i].buf, (dims[i]+t.bn-1)/t.bn, tm, uint(t.wn), uint(t.wm))
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(y, unsafe.Slice((*float32)(unsafe.Pointer(&sc.out.bytes[0])), L*D))
		for i := range count {
			copy(projected[i], unsafe.Slice((*float32)(unsafe.Pointer(&sc.nextOut[i].bytes[0])), L*dims[i]))
		}
	})
	if encErr != nil {
		return nil, projected, encErr
	}
	return y, projected, nil
}

// ResidualNormMLPProjMamba2InputDevice folds the next Mamba-2 input projection onto the current tail.
func ResidualNormMLPProjMamba2InputDevice(
	mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32,
	nextNormW, nextInProjW []float32, nextProjDim int,
) (y, projected []float32, err error) {
	outs := [6][]float32{nextInProjW}
	dims := [6]int{nextProjDim}
	y, got, err := residualNormMLPProjMixerInputDevice(mixerHidden, projW, h, normW, gate, up, down, L, D, mixCols, FF, eps, nextNormW, outs, dims, 1)
	return y, got[0], err
}

// ResidualNormMLPProjRWKV7InputDevice folds all six next RWKV-7 projections onto the current tail.
func ResidualNormMLPProjRWKV7InputDevice(
	mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32,
	nextNormW, nextRW, nextWW, nextKW, nextVW, nextAW, nextBW []float32, nextHK, nextHV int,
) (y, r, w, k, v, a, b []float32, err error) {
	weights := [6][]float32{nextRW, nextWW, nextKW, nextVW, nextAW, nextBW}
	dims := [6]int{nextHK, nextHK, nextHK, nextHV, nextHK, nextHK}
	y, got, err := residualNormMLPProjMixerInputDevice(mixerHidden, projW, h, normW, gate, up, down, L, D, mixCols, FF, eps, nextNormW, weights, dims, 6)
	return y, got[0], got[1], got[2], got[3], got[4], got[5], err
}
