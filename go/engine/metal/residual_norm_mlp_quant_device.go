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

// residual_norm_mlp_quant_device.go — the PACKED-weight twin of residual_norm_mlp_device.go (#8-B
// slice 1): the mixer-output residual add, the pre-MLP RMSNorm, the SwiGLU MLP over affine-packed
// gate/up/down (the checkpoint's own 2/3/4/5/6/8-bit codes + bf16 scales/biases — never widened) and
// the MLP residual add, all encoded into ONE command buffer. This is the fold the quant bypass in
// composed.forwardEmb could not take: the f32 fold's steel GEMMs read f32 weights, so a quant 27B ran
// the tail as THREE separate quant-seam command buffers (gate, up, down) bracketed by host glue — the
// round-trip census measured 496 device CBs per decode token with every fused seam at zero. Here the
// three projections ride affine_qmv (L=1, the decode hot path) or affine_qmm_t (L>1, prefill) inside
// the same buffer as the adds, the norm and the silu — one round trip where the bypass paid three.
//
// Numeric tier: activations stay f32 between stages — the adds, the RMSNorm and the silu all run the
// f32 kernels, and each projection casts f32→bf16 at its input and widens bf16→f32 at its output
// (in-encoder v_copy casts, cast.go), which is EXACTLY the tier the unfused per-projection seam
// (MatMulQuantF32NTInto) trades at: bf16 only across the qmv, f32 everywhere else. An earlier
// bf16-end-to-end draft measured ~3.5% drift against the host tail (cancellation in the down
// projection amplifies chained bf16 storage rounding) and was rejected — the fold must not change
// the lane's numerics, only its round trips. Gated against the f64 host reference in
// residual_norm_mlp_quant_device_test.go. Residuals are plain adds — the composed wiring only routes
// here when residualScale == 1 (the Qwen hybrids' case).
type residualNormMLPQuantScratch struct {
	// f32 stage buffers (the unfused tier's activation dtype)
	h, mix, normed, g, u, s, out *pinnedNoCopyBytes
	// bf16 qmv-boundary buffers: nBF holds the cast normed rows (gate/up input, reused as the down
	// output after both consume it), gBF/uBF the projections' bf16 outputs (gBF reused for the cast
	// silu rows the down projection reads).
	nBF, gBF, uBF *pinnedNoCopyBytes
}

type residualNormMLPQuantKey struct{ L, D, FF int }

var residualNormMLPQuantPools sync.Map // residualNormMLPQuantKey -> *sync.Pool

func getResidualNormMLPQuantScratch(L, D, FF int) (*residualNormMLPQuantScratch, error) {
	key := residualNormMLPQuantKey{L, D, FF}
	poolAny, ok := residualNormMLPQuantPools.Load(key)
	if !ok {
		poolAny, _ = residualNormMLPQuantPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*residualNormMLPQuantScratch), nil
	}
	sc := &residualNormMLPQuantScratch{}
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
	sc.nBF = alloc(L * max(D, FF) * bf16Size)
	sc.gBF = alloc(L * FF * bf16Size)
	sc.uBF = alloc(L * FF * bf16Size)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putResidualNormMLPQuantScratch(L, D, FF int, sc *residualNormMLPQuantScratch) {
	if v, ok := residualNormMLPQuantPools.Load(residualNormMLPQuantKey{L, D, FF}); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// quantGeometryOK verifies one packed projection against the fold's expected [outDim, inDim] logical
// shape and a kernel-shipped bit width — the same pre-checks MatMulQuantF32NTInto applies, hoisted so
// a mismatch declines to the host tail instead of half-encoding a command buffer.
func quantGeometryOK(w *model.QuantWeight, outDim, inDim int) bool {
	if w == nil || w.GroupSize <= 0 || inDim%w.GroupSize != 0 {
		return false
	}
	switch w.Bits {
	case 2, 3, 4, 5, 6, 8:
	default:
		return false
	}
	if w.OutDim != 0 && w.OutDim != outDim {
		return false
	}
	if w.InDim != 0 && w.InDim != inDim {
		return false
	}
	groups := inDim / w.GroupSize
	if len(w.Scales) != outDim*groups*bf16Size || len(w.Biases) != outDim*groups*bf16Size {
		return false
	}
	return len(w.Packed) == outDim*(inDim*w.Bits/32)*4
}

// ResidualNormMLPQuantDevice computes one pre-norm SwiGLU FFN sub-block over PACKED weights in a
// single command buffer:
//
//	hplus = h + mixOut                                   // mixer-output residual [L,D]
//	y     = hplus + (silu(hplus'·gateᵀ) ⊙ hplus'·upᵀ)·downᵀ,  hplus' = RMSNorm(hplus, normW)
//
// h/mixOut are [L,D] f32 at the seam; normW is [D] (plain RMSNorm weight); gate/up are packed
// [FF,D], down packed [D,FF]. L==1 rides affine_qmv per projection (the decode hot path), L>1 rides
// affine_qmm_t. Returns y [L,D] f32. The unfused bypass pays three quant-seam command buffers plus
// three host passes over [L,D]; this is one buffer, one upload pair, one readback.
func ResidualNormMLPQuantDevice(h, mixOut, normW []float32, gate, up, down *model.QuantWeight, L, D, FF int, eps float32) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(h) != L*D || len(mixOut) != L*D || len(normW) != D {
		return nil, core.NewError("native.ResidualNormMLPQuantDevice: size mismatch")
	}
	if !quantGeometryOK(gate, FF, D) || !quantGeometryOK(up, FF, D) || !quantGeometryOK(down, D, FF) {
		return nil, core.NewError("native.ResidualNormMLPQuantDevice: unsupported quant geometry")
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
		sc, gerr := getResidualNormMLPQuantScratch(L, D, FF)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putResidualNormMLPQuantScratch(L, D, FF, sc)
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
		normBuf := residentFloat32(normW)
		nD := L * D
		nFF := L * FF
		rmsTG := rmsThreadgroup(D, psoRMS)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		// hplus = h + mixOut, in place into hBuf (hplus is the RMSNorm input AND the MLP residual).
		emitBinary(encSink{enc}, psoAdd, hBuf, 0, mixBuf, 0, hBuf, 0, nD)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// normed = RMSNorm(hplus, normW) — plain rsqrt(mean²+eps)·w, one threadgroup per row, f32.
		emitRMSNormRows(encSink{enc}, psoRMS, hBuf, normBuf, sc.normed.buf, 0, 0, 0, D, eps, L, rmsTG)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// encProj encodes one packed [outDim,inDim] projection — cast the f32 rows to bf16, run
		// affine_qmv per row at L==1 (the decode hot path) or the affine_qmm_t slab at L>1
		// (prefill), and widen the bf16 result back to f32 — the unfused seam's exact dtype dance,
		// inside the SAME encoder as the rest of the tail. xBF/dstBF are the bf16 staging buffers.
		encProj := func(w *model.QuantWeight, x, xBF, dstBF, dst metal.MTLBuffer, outDim, inDim int) error {
			if err := encNarrowF32ToBF16(enc, x, xBF, L*inDim); err != nil {
				return err
			}
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			wq, ws, wb := residentBytes(w.Packed), residentBytes(w.Scales), residentBytes(w.Biases)
			var perr error
			if L == 1 {
				perr = encQMVBF16At(enc, wq, ws, wb, xBF, dstBF, 0, 0, 0, 0, 0, outDim, inDim, w.GroupSize, w.Bits)
			} else {
				perr = encQMMTBF16At(enc, wq, ws, wb, xBF, dstBF, 0, 0, 0, 0, 0, L, outDim, inDim, w.GroupSize, w.Bits)
			}
			if perr != nil {
				return perr
			}
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			return encWidenBF16ToF32(enc, dstBF, dst, L*outDim)
		}
		// gate and up read the same cast normed rows; cast once, project twice.
		if encErr = encNarrowF32ToBF16(enc, sc.normed.buf, sc.nBF.buf, nD); encErr != nil {
			endEncodingFast(enc)
			return
		}
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		encProjPre := func(w *model.QuantWeight, xBF, dstBF, dst metal.MTLBuffer, outDim, inDim int) error {
			wq, ws, wb := residentBytes(w.Packed), residentBytes(w.Scales), residentBytes(w.Biases)
			var perr error
			if L == 1 {
				perr = encQMVBF16At(enc, wq, ws, wb, xBF, dstBF, 0, 0, 0, 0, 0, outDim, inDim, w.GroupSize, w.Bits)
			} else {
				perr = encQMMTBF16At(enc, wq, ws, wb, xBF, dstBF, 0, 0, 0, 0, 0, L, outDim, inDim, w.GroupSize, w.Bits)
			}
			if perr != nil {
				return perr
			}
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			return encWidenBF16ToF32(enc, dstBF, dst, L*outDim)
		}
		if encErr = encProjPre(gate, sc.nBF.buf, sc.gBF.buf, sc.g.buf, FF, D); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encProjPre(up, sc.nBF.buf, sc.uBF.buf, sc.u.buf, FF, D); encErr != nil {
			endEncodingFast(enc)
			return
		}
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitUnary(encSink{enc}, psoSig, sc.g.buf, sc.s.buf, nFF) // s = sigmoid(g)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitBinary(encSink{enc}, psoMul, sc.s.buf, 0, sc.g.buf, 0, sc.s.buf, 0, nFF) // s = silu(g)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitBinary(encSink{enc}, psoMul, sc.s.buf, 0, sc.u.buf, 0, sc.s.buf, 0, nFF) // s = silu(g)·u
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// down: cast the silu rows into gBF (free after its widen) and land the bf16 result in nBF
		// (free after gate/up consumed it), widening into out.
		if encErr = encProj(down, sc.s.buf, sc.gBF.buf, sc.nBF.buf, sc.out.buf, D, FF); encErr != nil {
			endEncodingFast(enc)
			return
		}
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// y = hplus + mlpOut, in place into out.
		emitBinary(encSink{enc}, psoAdd, hBuf, 0, sc.out.buf, 0, sc.out.buf, 0, nD)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(out, unsafe.Slice((*float32)(unsafe.Pointer(&sc.out.bytes[0])), nD))
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}
