// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// residual_norm_mlp_proj_head_device.go — the OUTPUT-side mirror of residual_norm_mlp_proj_attn_input_device.go
// and residual_norm_mlp_proj_gated_delta_input_device.go: where those fold the NEXT layer's input RMSNorm +
// input projections onto the back of a proj-fused tail, this folds the MODEL's OWN terminal stage on
// instead — the final RMSNorm (NormF) and the LM head GEMM (tied Embed or untied Output) — because there is
// no next layer past the LAST one. It is ResidualNormMLPProjDevice's tail PLUS: normed = RMSNorm(y[last
// row], NormF); logits = normed @ headᵟ. The head's own separate command buffer (today's
// ComposedSession.headLogits, a host RMSNorm followed by a standalone device GEMM) disappears — the
// terminal N+1 → N command-buffer collapse. Only the LAST row of y ever needs the head (mirrors
// headLogits' single-row contract exactly for L>1 prefill); y itself stays full [L,D] — the caller still
// needs every row for the DecodeStepper/DecodeForward contract. Numeric tier: device f32 throughout
// (RMSNorm included) — the same tier every fused tail in this family already serves.

// residualNormMLPProjHeadScratch is residualNormMLPProjScratch's shape plus the final-row RMSNorm output
// and the head GEMM's output/params.
type residualNormMLPProjHeadScratch struct {
	mh, h, mix, normed, g, u, s, out *pinnedNoCopyBytes
	paramsProj, paramsGU, paramsD    *pinnedNoCopyBytes
	finalNormed, logits, paramsHead  *pinnedNoCopyBytes
	paramsFilled                     bool
}

type residualNormMLPProjHeadKey struct{ L, D, mixCols, FF, Vocab int }

var residualNormMLPProjHeadPools sync.Map // residualNormMLPProjHeadKey -> *sync.Pool

func getResidualNormMLPProjHeadScratch(L, D, mixCols, FF, Vocab int) (*residualNormMLPProjHeadScratch, error) {
	key := residualNormMLPProjHeadKey{L, D, mixCols, FF, Vocab}
	poolAny, ok := residualNormMLPProjHeadPools.Load(key)
	if !ok {
		poolAny, _ = residualNormMLPProjHeadPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*residualNormMLPProjHeadScratch), nil
	}
	sc := &residualNormMLPProjHeadScratch{}
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
	sc.finalNormed = alloc(D * 4)
	sc.logits = alloc(Vocab * 4)
	sc.paramsHead = alloc(72)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putResidualNormMLPProjHeadScratch(L, D, mixCols, FF, Vocab int, sc *residualNormMLPProjHeadScratch) {
	if v, ok := residualNormMLPProjHeadPools.Load(residualNormMLPProjHeadKey{L, D, mixCols, FF, Vocab}); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// ResidualNormMLPProjHeadDevice computes the LAST layer's projection-fused FFN tail — identical to
// ResidualNormMLPProjDevice — AND, in the SAME command buffer, the model's terminal stage over the tail's
// LAST row:
//
//	y      = ResidualNormMLPProjDevice(...)             // this (last) layer's output
//	normed = RMSNorm(y[L-1,:], normF)                    // the model's final norm, LAST row only
//	logits = normed @ headᵀ                              // headW is Output (untied) or Embed (tied)
//
// mixerHidden is [L,mixCols]; projW is [D,mixCols]; h is [L,D]; normW/normF are [D]; gate/up are [FF,D],
// down is [D,FF]; head is [Vocab,D]. Returns y [L,D] (still needed — the caller's DecodeStepper/
// DecodeForward contract returns hidden, not logits) and logits [Vocab] for the last row. Every
// intermediate stays device-resident, including the terminal RMSNorm — where the unfused path pays a host
// RMSNorm plus a standalone head command buffer after this one returns.
func ResidualNormMLPProjHeadDevice(
	mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32,
	normF, head []float32, Vocab int,
) (y, logits []float32, err error) {
	if err := ensureInit(); err != nil {
		return nil, nil, err
	}
	if len(mixerHidden) != L*mixCols || len(projW) != D*mixCols || len(h) != L*D || len(normW) != D ||
		len(gate) != FF*D || len(up) != FF*D || len(down) != D*FF {
		return nil, nil, core.NewError("native.ResidualNormMLPProjHeadDevice: tail size mismatch")
	}
	if len(normF) != D || len(head) != Vocab*D {
		return nil, nil, core.NewError("native.ResidualNormMLPProjHeadDevice: head size mismatch")
	}
	t := steelNT
	psoProj, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, D%t.bn == 0, mixCols%t.bk == 0)
	if err != nil {
		return nil, nil, err
	}
	psoGU, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, FF%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, nil, err
	}
	psoD, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, D%t.bn == 0, FF%t.bk == 0)
	if err != nil {
		return nil, nil, err
	}
	psoAdd, err := pipelineFor("vv_Addfloat32")
	if err != nil {
		return nil, nil, err
	}
	rmsName := "rmsfloat32"
	if D > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32"
	}
	psoRMS, err := pipelineFor(rmsName)
	if err != nil {
		return nil, nil, err
	}
	psoSig, err := pipelineFor("v_Sigmoidfloat32float32")
	if err != nil {
		return nil, nil, err
	}
	psoMul, err := pipelineFor("vv_Multiplyfloat32")
	if err != nil {
		return nil, nil, err
	}
	// The head GEMM: M=1 (last row only), K=D, N=Vocab, B=head transposed [Vocab,D] — the same shape
	// matNTInto's device hook (MatMulF32NTInto) already dispatches today via the fused nt kernel
	// (dtm·dtn far exceeds the split-K threshold at any real vocab size, so matMulF32NTInto's own
	// dispatch always lands here too — see matmul_steel.go).
	psoHead, err := steelGemmPipeline(t.name, false, false, false, 1%t.bm == 0, Vocab%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, nil, err
	}

	y = make([]float32, L*D)
	logits = make([]float32, Vocab)
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getResidualNormMLPProjHeadScratch(L, D, mixCols, FF, Vocab)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putResidualNormMLPProjHeadScratch(L, D, mixCols, FF, Vocab, sc)
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
			tnHead, tmHead := (Vocab+t.bn-1)/t.bn, (1+t.bm-1)/t.bm
			fillMatMulF32SteelParams(sc.paramsHead.bytes, 1, D, Vocab, D, tnHead, tmHead, D/t.bk)
			sc.paramsFilled = true
		}
		projBuf := residentFloat32(projW)
		normBuf := residentFloat32(normW)
		gateBuf := residentFloat32(gate)
		upBuf := residentFloat32(up)
		downBuf := residentFloat32(down)
		normFBuf := residentFloat32(normF)
		headBuf := residentFloat32(head)

		tnProj, tmProj := (D+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
		tnGU, tmGU := (FF+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
		tnD, tmD := (D+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
		tnHead, tmHead := (Vocab+t.bn-1)/t.bn, (1+t.bm-1)/t.bm
		nD := L * D
		nFF := L * FF
		rmsTG := rmsThreadgroup(D, psoRMS)
		lastRowOff := uint((L - 1) * D * 4)

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
		// y = hplus + mlpOut, in place into out.buf — this (last) layer's output.
		emitBinary(encSink{enc}, psoAdd, hBuf, 0, sc.out.buf, 0, sc.out.buf, 0, nD)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// [head] the model's terminal stage, LAST row only: RMSNorm(y[L-1], normF) then the head GEMM
		// against it — the separate command buffer ComposedSession.headLogits pays today disappears.
		emitRMSNormAt(encSink{enc}, psoRMS, sc.out.buf, normFBuf, sc.finalNormed.buf, lastRowOff, 0, 0, D, eps, rmsTG)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitSteelGemm(encSink{enc}, psoHead, sc.finalNormed.buf, headBuf, sc.logits.buf, sc.paramsHead.buf, tnHead, tmHead, uint(t.wn), uint(t.wm))
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(y, unsafe.Slice((*float32)(unsafe.Pointer(&sc.out.bytes[0])), L*D))
		copy(logits, unsafe.Slice((*float32)(unsafe.Pointer(&sc.logits.bytes[0])), Vocab))
	})
	if encErr != nil {
		return nil, nil, encErr
	}
	return y, logits, nil
}
