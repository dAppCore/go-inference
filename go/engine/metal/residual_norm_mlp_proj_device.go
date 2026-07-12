// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// residual_norm_mlp_proj_device.go — the projection-fused pre-norm SwiGLU FFN sub-block: the mixer's FINAL
// output projection (attention o_proj / gated-delta out_proj), the mixer-output residual add, the pre-MLP
// RMSNorm, the SwiGLU MLP and the MLP residual add, all encoded into ONE command buffer with every
// intermediate staying device-resident. It is ResidualNormMLPDevice with the mixer's final projection GEMM
// folded onto the front: where the unfused path runs the projection as its own command buffer (a commit+wait
// per layer) then the tail as a second command buffer, here the projection rides the tail's command buffer —
// its output mixOut never crosses the host floor, going straight into the residual add. That collapses one
// command-buffer round-trip per layer. Arch-neutral: every pre-norm SwiGLU stack whose mixer emits its
// output through a single GEMM (o_proj / out_proj) has exactly this shape, so it is named for the op
// sequence, not for any one mixer. Numeric tier: device f32 — the same tier ResidualNormMLPDevice and the
// standalone projection GEMM (MatMulF32NTInto) already serve.

// residualNormMLPProjScratch holds the pinned staging for one (L,D,mixCols,FF) shape. mh is the mixer-hidden
// upload (the projection input [L,mixCols]); mix the projection output mixOut [L,D] (device-only — never
// copied back); h the pre-mixer hidden upload that the residual add rewrites in place into hplus (= h +
// mixOut); normed the RMSNorm output; g/u/s the SwiGLU intermediates; out the down GEMM output that the
// final add rewrites in place. paramsProj/paramsGU/paramsD are the three steel GEMM param blocks.
type residualNormMLPProjScratch struct {
	mh, h, mix, normed, g, u, s, out *pinnedNoCopyBytes
	paramsProj, paramsGU, paramsD    *pinnedNoCopyBytes
	paramsFilled                     bool
}

type residualNormMLPProjKey struct{ L, D, mixCols, FF int }

var residualNormMLPProjPools sync.Map // residualNormMLPProjKey -> *sync.Pool

func getResidualNormMLPProjScratch(L, D, mixCols, FF int) (*residualNormMLPProjScratch, error) {
	key := residualNormMLPProjKey{L, D, mixCols, FF}
	poolAny, ok := residualNormMLPProjPools.Load(key)
	if !ok {
		poolAny, _ = residualNormMLPProjPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*residualNormMLPProjScratch), nil
	}
	sc := &residualNormMLPProjScratch{}
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
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putResidualNormMLPProjScratch(L, D, mixCols, FF int, sc *residualNormMLPProjScratch) {
	if v, ok := residualNormMLPProjPools.Load(residualNormMLPProjKey{L, D, mixCols, FF}); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// ResidualNormMLPProjDevice computes one pre-norm SwiGLU FFN sub-block WITH the mixer's final projection in
// a single command buffer:
//
//	mixOut = mixerHidden @ projWᵀ                   // the mixer's o_proj / out_proj [L,D]
//	hplus  = h + mixOut                             // mixer-output residual [L,D]
//	y      = hplus + (silu(hplus'@gateᵀ) ⊙ hplus'@upᵀ) @ downᵀ,  hplus' = RMSNorm(hplus, normW)
//
// mixerHidden is [L,mixCols]; projW is [D,mixCols]; h is [L,D]; normW is [D] (plain RMSNorm weight); gate/up
// are [FF,D], down is [D,FF] (the steel nt kernel reads each weight transposed); returns y [L,D]. Every
// intermediate stays device-resident — the only host traffic is the mixerHidden/h uploads and the y readback,
// where the unfused path pays a whole projection command buffer plus the tail's three host passes.
func ResidualNormMLPProjDevice(mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(mixerHidden) != L*mixCols || len(projW) != D*mixCols || len(h) != L*D || len(normW) != D || len(gate) != FF*D || len(up) != FF*D || len(down) != D*FF {
		return nil, core.NewError("native.ResidualNormMLPProjDevice: size mismatch")
	}
	t := steelNT
	psoProj, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, D%t.bn == 0, mixCols%t.bk == 0)
	if err != nil {
		return nil, err
	}
	psoGU, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, FF%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, err
	}
	psoD, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, D%t.bn == 0, FF%t.bk == 0)
	if err != nil {
		return nil, err
	}
	psoAdd, err := pipelineFor("vv_Addfloat32")
	if err != nil {
		return nil, err
	}
	rmsName := "rmsfloat32"
	if D > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32"
	}
	psoRMS, err := pipelineFor(rmsName)
	if err != nil {
		return nil, err
	}
	psoSig, err := pipelineFor("v_Sigmoidfloat32float32")
	if err != nil {
		return nil, err
	}
	psoMul, err := pipelineFor("vv_Multiplyfloat32")
	if err != nil {
		return nil, err
	}

	out := make([]float32, L*D)
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getResidualNormMLPProjScratch(L, D, mixCols, FF)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putResidualNormMLPProjScratch(L, D, mixCols, FF, sc)
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
			sc.paramsFilled = true
		}
		projBuf := residentFloat32(projW)
		normBuf := residentFloat32(normW)
		gateBuf := residentFloat32(gate)
		upBuf := residentFloat32(up)
		downBuf := residentFloat32(down)

		tnProj, tmProj := (D+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
		tnGU, tmGU := (FF+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
		tnD, tmD := (D+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
		nD := L * D
		nFF := L * FF
		rmsTG := rmsThreadgroup(D, psoRMS)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		// mixOut = mixerHidden @ projWᵀ — the mixer's final projection, straight into the device-resident
		// mixBuf (never crosses the host floor; the unfused path would commit this as its own CB).
		emitSteelGemm(encSink{enc}, psoProj, mhBuf, projBuf, sc.mix.buf, sc.paramsProj.buf, tnProj, tmProj, uint(t.wn), uint(t.wm))
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// hplus = h + mixOut, in place into hBuf (h is dead after; hplus is the RMSNorm input AND the
		// MLP residual, so hBuf must survive until the final add).
		emitBinary(encSink{enc}, psoAdd, hBuf, 0, sc.mix.buf, 0, hBuf, 0, nD)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// normed = RMSNorm(hplus, normW) — plain rsqrt(mean²+eps)·w, one threadgroup per row.
		emitRMSNormRows(encSink{enc}, psoRMS, hBuf, normBuf, sc.normed.buf, 0, 0, 0, D, eps, L, rmsTG)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// gate and up read the same normed and write disjoint outputs — no barrier between them.
		emitSteelGemm(encSink{enc}, psoGU, sc.normed.buf, gateBuf, sc.g.buf, sc.paramsGU.buf, tnGU, tmGU, uint(t.wn), uint(t.wm))
		emitSteelGemm(encSink{enc}, psoGU, sc.normed.buf, upBuf, sc.u.buf, sc.paramsGU.buf, tnGU, tmGU, uint(t.wn), uint(t.wm))
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitUnary(encSink{enc}, psoSig, sc.g.buf, sc.s.buf, nFF) // s = sigmoid(g)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitBinary(encSink{enc}, psoMul, sc.s.buf, 0, sc.g.buf, 0, sc.s.buf, 0, nFF) // s = silu(g)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitBinary(encSink{enc}, psoMul, sc.s.buf, 0, sc.u.buf, 0, sc.s.buf, 0, nFF) // s = silu(g)·u
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitSteelGemm(encSink{enc}, psoD, sc.s.buf, downBuf, sc.out.buf, sc.paramsD.buf, tnD, tmD, uint(t.wn), uint(t.wm))
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// y = hplus + mlpOut, in place into out.
		emitBinary(encSink{enc}, psoAdd, hBuf, 0, sc.out.buf, 0, sc.out.buf, 0, nD)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(out, unsafe.Slice((*float32)(unsafe.Pointer(&sc.out.bytes[0])), L*D))
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}
