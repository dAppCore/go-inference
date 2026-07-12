// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// residual_norm_mlp_device.go — the fused pre-norm SwiGLU FFN sub-block: the mixer-output residual add,
// the pre-MLP RMSNorm, the SwiGLU MLP (gate/up GEMMs, silu-and-multiply, down GEMM) and the MLP residual
// add, all encoded into ONE command buffer with every intermediate staying device-resident. This is the
// FFN half of EVERY pre-norm SwiGLU transformer layer (llama/qwen/mistral shape), so it is arch-neutral —
// named for the op sequence, not for any one arch's stack. It folds the host glue that otherwise brackets
// the standalone MLP command buffer: currently `h += mixOut`, `RMSNorm(h, w)` and `h += mlpOut` all run on
// the CPU (three passes over [L,D]) with a device round-trip on each side of the MLP; here they ride the
// same command buffer the MLP already pays for. Numeric tier: device f32 — the same tier ComposedMLPDevice
// (the bare MLP fuse) already serves, and the RMSNorm is the plain rsqrt(mean(x²)+eps)·w kernel (no gemma
// 1+w), byte-matching the host rmsNormRowsPlain within f32 tolerance.

// residualNormMLPScratch holds the pinned staging for one (L,D,FF) shape. h is the x upload that the
// residual add rewrites in place into hplus (= h + mixOut) — hplus is both the RMSNorm input and the MLP
// residual, so it must survive until the final add. mix is the mixer-output upload; normed the RMSNorm
// output; g/u/s the SwiGLU intermediates; out the down GEMM output that the final add rewrites in place.
type residualNormMLPScratch struct {
	h, mix, normed, g, u, s, out *pinnedNoCopyBytes
	paramsGU, paramsD            *pinnedNoCopyBytes
	paramsFilled                 bool
}

type residualNormMLPKey struct{ L, D, FF int }

var residualNormMLPPools sync.Map // residualNormMLPKey -> *sync.Pool

func getResidualNormMLPScratch(L, D, FF int) (*residualNormMLPScratch, error) {
	key := residualNormMLPKey{L, D, FF}
	poolAny, ok := residualNormMLPPools.Load(key)
	if !ok {
		poolAny, _ = residualNormMLPPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*residualNormMLPScratch), nil
	}
	sc := &residualNormMLPScratch{}
	var err error
	alloc := func(n int) *pinnedNoCopyBytes {
		if err != nil {
			return nil
		}
		var b *pinnedNoCopyBytes
		b, err = newPinnedNoCopyBytes(n)
		return b
	}
	sc.h = alloc(L * D * 4)
	sc.mix = alloc(L * D * 4)
	sc.normed = alloc(L * D * 4)
	sc.g = alloc(L * FF * 4)
	sc.u = alloc(L * FF * 4)
	sc.s = alloc(L * FF * 4)
	sc.out = alloc(L * D * 4)
	sc.paramsGU = alloc(72)
	sc.paramsD = alloc(72)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putResidualNormMLPScratch(L, D, FF int, sc *residualNormMLPScratch) {
	if v, ok := residualNormMLPPools.Load(residualNormMLPKey{L, D, FF}); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// ResidualNormMLPDevice computes one pre-norm SwiGLU FFN sub-block in a single command buffer:
//
//	hplus = h + mixOut                              // mixer-output residual [L,D]
//	y     = hplus + (silu(hplus'@gateᵀ) ⊙ hplus'@upᵀ) @ downᵀ,  hplus' = RMSNorm(hplus, normW)
//
// h/mixOut are [L,D]; normW is [D] (plain RMSNorm weight); gate/up are [FF,D], down is [D,FF] (the steel
// nt kernel reads each weight transposed); returns y [L,D]. Every intermediate stays device-resident — the
// only host traffic is the h/mixOut uploads and the y readback, where the unfused path pays three extra
// host passes ([L,D] add, RMSNorm, add) plus a round-trip on each side of the MLP command buffer.
func ResidualNormMLPDevice(h, mixOut, normW, gate, up, down []float32, L, D, FF int, eps float32) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(h) != L*D || len(mixOut) != L*D || len(normW) != D || len(gate) != FF*D || len(up) != FF*D || len(down) != D*FF {
		return nil, core.NewError("native.ResidualNormMLPDevice: size mismatch")
	}
	t := steelNT
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
		sc, gerr := getResidualNormMLPScratch(L, D, FF)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putResidualNormMLPScratch(L, D, FF, sc)
		hBuf, cerr := sc.h.copyBuffer(float32Bytes(h))
		if cerr != nil {
			encErr = cerr
			return
		}
		mixBuf, cerr := sc.mix.copyBuffer(float32Bytes(mixOut))
		if cerr != nil {
			encErr = cerr
			return
		}
		if !sc.paramsFilled {
			tnGU, tmGU := (FF+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
			fillMatMulF32SteelParams(sc.paramsGU.bytes, L, D, FF, D, tnGU, tmGU, D/t.bk)
			tnD, tmD := (D+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
			fillMatMulF32SteelParams(sc.paramsD.bytes, L, FF, D, FF, tnD, tmD, FF/t.bk)
			sc.paramsFilled = true
		}
		normBuf := residentFloat32(normW)
		gateBuf := residentFloat32(gate)
		upBuf := residentFloat32(up)
		downBuf := residentFloat32(down)

		tnGU, tmGU := (FF+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
		tnD, tmD := (D+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
		nD := L * D
		nFF := L * FF
		rmsTG := rmsThreadgroup(D, psoRMS)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		// hplus = h + mixOut, in place into hBuf (h is dead after; hplus is the RMSNorm input AND the
		// MLP residual, so hBuf must survive until the final add).
		emitBinary(encSink{enc}, psoAdd, hBuf, 0, mixBuf, 0, hBuf, 0, nD)
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
