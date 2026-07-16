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

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		encErr = encResidualNormMLPQuantTail(enc, quantTailBufs{
			h: hBuf, mix: mixBuf, normed: sc.normed.buf, nBF: sc.nBF.buf,
			g: sc.g.buf, gBF: sc.gBF.buf, u: sc.u.buf, uBF: sc.uBF.buf, s: sc.s.buf, out: sc.out.buf,
		}, normBuf, gate, up, down, L, D, FF, eps)
		endEncodingFast(enc)
		if encErr != nil {
			return
		}
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(out, unsafe.Slice((*float32)(unsafe.Pointer(&sc.out.bytes[0])), nD))
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

// quantTailBufs names the stage buffers encResidualNormMLPQuantTail writes through: h [L,D] f32
// (modified in place — it becomes hplus, the tail's residual), mix [L,D] f32, normed/nBF the f32/
// bf16 normed rows, g/gBF and u/uBF the gate/up stages, s the silu product, out the result [L,D].
type quantTailBufs struct {
	h, mix, normed, nBF, g, gBF, u, uBF, s, out metal.MTLBuffer
}

// encProjQuantF32 encodes one packed [outDim,inDim] projection over f32 rows inside a live
// encoder: cast x to bf16, affine_qmv per row at L==1 (the decode hot path) or the affine_qmm_t
// slab at L>1, widen back to f32 — the unfused quant seam's exact dtype dance as encoder stages.
// xBF/dstBF are bf16 staging; barriers separate the dependent stages.
func encProjQuantF32(enc metal.MTLComputeCommandEncoder, w *model.QuantWeight, x, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
	if err := encNarrowF32ToBF16(enc, x, xBF, L*inDim); err != nil {
		return err
	}
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
	return encProjQuantBF16In(enc, w, xBF, dstBF, dst, L, outDim, inDim)
}

// encProjQuantBF16In is encProjQuantF32 from ALREADY-CAST bf16 rows (cast once, project many —
// the gate/up pattern).
func encProjQuantBF16In(enc metal.MTLComputeCommandEncoder, w *model.QuantWeight, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
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
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
	return encWidenBF16ToF32(enc, dstBF, dst, L*outDim)
}

// tailProjFromBF16 encodes one projection from ALREADY-CAST bf16 rows (xBF) into dst f32 via the
// dstBF staging — the shape both weight forms (packed codes, raw bf16) provide, so one tail body
// serves both (#26).
type tailProjFromBF16 func(enc metal.MTLComputeCommandEncoder, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error

// tailProjFromF32 is the same projection from f32 rows (the down projection's shape: cast + project).
type tailProjFromF32 func(enc metal.MTLComputeCommandEncoder, x, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error

// encResidualNormMLPQuantTail encodes the whole packed FFN tail into a live encoder:
//
//	hplus = h + mix (in place over h) → normed = RMSNorm(hplus, normW) → silu(normed·gateᵀ) ⊙
//	(normed·upᵀ) → ·downᵀ → out = hplus + mlpOut
//
// — the body ResidualNormMLPQuantDevice runs as its own command buffer, split out so the fused
// gated-delta LAYER command buffer (#18 S3) stacks the identical tail behind the mixer stages.
func encResidualNormMLPQuantTail(enc metal.MTLComputeCommandEncoder, tb quantTailBufs, normW metal.MTLBuffer, gate, up, down *model.QuantWeight, L, D, FF int, eps float32) error {
	return encResidualNormMLPTailCore(enc, tb, normW,
		func(enc metal.MTLComputeCommandEncoder, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjQuantBF16In(enc, gate, xBF, dstBF, dst, L, outDim, inDim)
		},
		func(enc metal.MTLComputeCommandEncoder, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjQuantBF16In(enc, up, xBF, dstBF, dst, L, outDim, inDim)
		},
		func(enc metal.MTLComputeCommandEncoder, x, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjQuantF32(enc, down, x, xBF, dstBF, dst, L, outDim, inDim)
		}, L, D, FF, eps)
}

// encResidualNormMLPTailCore is the weight-form-agnostic tail body — the projections arrive as
// closures (packed codes via the affine qmv, raw bf16 via the gemv), everything else is shared.
func encResidualNormMLPTailCore(enc metal.MTLComputeCommandEncoder, tb quantTailBufs, normW metal.MTLBuffer, gateProj, upProj tailProjFromBF16, downProj tailProjFromF32, L, D, FF int, eps float32) error {
	psoAdd, err := pipelineFor("vv_Addfloat32")
	if err != nil {
		return err
	}
	rmsName := "rmsfloat32"
	if D > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32"
	}
	psoRMS, err := pipelineFor(rmsName)
	if err != nil {
		return err
	}
	psoSig, err := pipelineFor("v_Sigmoidfloat32float32")
	if err != nil {
		return err
	}
	psoMul, err := pipelineFor("vv_Multiplyfloat32")
	if err != nil {
		return err
	}
	nD, nFF := L*D, L*FF
	rmsTG := rmsThreadgroup(D, psoRMS)
	// hplus = h + mix, in place into h (hplus is the RMSNorm input AND the MLP residual).
	emitBinary(encSink{enc}, psoAdd, tb.h, 0, tb.mix, 0, tb.h, 0, nD)
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
	emitRMSNormRows(encSink{enc}, psoRMS, tb.h, normW, tb.normed, 0, 0, 0, D, eps, L, rmsTG)
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
	// gate and up read the same cast normed rows; cast once, project twice.
	if err := encNarrowF32ToBF16(enc, tb.normed, tb.nBF, nD); err != nil {
		return err
	}
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
	if err := gateProj(enc, tb.nBF, tb.gBF, tb.g, L, FF, D); err != nil {
		return err
	}
	if err := upProj(enc, tb.nBF, tb.uBF, tb.u, L, FF, D); err != nil {
		return err
	}
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
	emitUnary(encSink{enc}, psoSig, tb.g, tb.s, nFF) // s = sigmoid(g)
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
	emitBinary(encSink{enc}, psoMul, tb.s, 0, tb.g, 0, tb.s, 0, nFF) // s = silu(g)
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
	emitBinary(encSink{enc}, psoMul, tb.s, 0, tb.u, 0, tb.s, 0, nFF) // s = silu(g)·u
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
	// down: cast the silu rows into gBF (free after its widen) and land the bf16 result in nBF
	// (free after gate/up consumed it), widening into out.
	if err := downProj(enc, tb.s, tb.gBF, tb.nBF, tb.out, L, D, FF); err != nil {
		return err
	}
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
	// y = hplus + mlpOut, in place into out.
	emitBinary(encSink{enc}, psoAdd, tb.h, 0, tb.out, 0, tb.out, 0, nD)
	return nil
}
