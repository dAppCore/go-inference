// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/attn"
	"github.com/tmc/apple/metal"
)

// composed_bf16_backend.go — the dense bf16 matvec seam (#26): the exact quant-seam shape
// (composed_quant_backend.go) for a checkpoint's UNQUANTISED bf16 projections. The weight bytes
// stay the mmap view (residentBytes caches the device buffer per slice); activations cross the
// seam f32 exactly like the quant seam, cast to bf16 for the kernel. This is what stops the dense
// serving lane widening every projection to f32 — the ×2 on bytes streamed per token AND on the
// resident set that put the official bf16 Qwen exports at ×4-6 behind mlx-lm.
// bf16SeamEnabled gates the device bf16 matvec seam. Default on; LTHN_BF16_SEAM=0 leaves the hooks
// unbound so the row-widen host floor serves — the same-binary A/B arm (the LTHN_GD_BLOCK shape).
var bf16SeamEnabled = os.Getenv("LTHN_BF16_SEAM") != "0"

func init() {
	if bf16SeamEnabled {
		attn.ProjBF16MatMulInto = MatMulBF16WeightF32NTInto
		attn.GatedDeltaBF16LayerDevice = gatedDeltaBF16LayerDeviceHook // the WHOLE dense bf16 layer in one CB (#26)
	}
}

// MatMulBF16WF32NTInto computes out[M,N] = x[M,K] @ wᵀ for a dense bf16 weight (w = raw row-major
// bf16 bytes [N,K]) with f32 activations at the seam. M=1 (decode) rides the MLX bf16 gemv over the
// resident weight bytes; M>1 currently loops rows through the same gemv (correct; the batched
// prefill slab is the follow-up). out is reused when cap(out) >= M*N.
func MatMulBF16WF32NTInto(out, x []float32, w []byte, M, K, N int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if M <= 0 || N <= 0 || K <= 0 {
		return nil, core.NewError("native.MatMulBF16WF32NTInto: M, N, K must be positive")
	}
	if len(x) != M*K {
		return nil, core.NewError("native.MatMulBF16WF32NTInto: len(x) must equal M*K")
	}
	if len(w) != N*K*bf16Size {
		return nil, core.NewError("native.MatMulBF16WF32NTInto: len(w) must equal N*K*2 bytes")
	}
	if cap(out) < M*N {
		out = make([]float32, M*N)
	} else {
		out = out[:M*N]
	}
	xb := f32sToBF16Bytes(x)
	var ob []byte
	for m := 0; m < M; m++ {
		row, err := MatVecBF16Into(ob, w, xb[m*K*bf16Size:(m+1)*K*bf16Size], N, K)
		if err != nil {
			return nil, err
		}
		ob = row
		for n := 0; n < N; n++ {
			out[m*N+n] = bf16ToF32(row[n*bf16Size], row[n*bf16Size+1])
		}
	}
	return out, nil
}

// MatMulBF16WeightF32NTInto is MatMulBF16WF32NTInto over the lib's BF16Weight form — the signature
// the composed/qwen3 hooks bind.
func MatMulBF16WeightF32NTInto(out, x []float32, w *model.BF16Weight, M, K, N int) ([]float32, error) {
	if w == nil || w.OutDim != N || w.InDim != K {
		return nil, core.NewError("native.MatMulBF16WeightF32NTInto: weight geometry mismatch")
	}
	return MatMulBF16WF32NTInto(out, x, w.Data, M, K, N)
}

// --- the dense bf16 whole-layer CB (#26 speed half): the S3 fold with gemv emitters in the qmv slots ---

// bf16GeometryOK verifies a raw bf16 projection against the fold's expected [outDim, inDim] shape.
func bf16GeometryOK(w *model.BF16Weight, outDim, inDim int) bool {
	return w != nil && w.OutDim == outDim && w.InDim == inDim && len(w.Data) == outDim*inDim*bf16Size
}

// encProjBF16From encodes one raw-bf16 projection from ALREADY-CAST bf16 rows: the MLX bf16 gemv
// (L=1) or the batched grid-Z gemv / steel GEMM (L>1) over the resident weight bytes, widened back
// to f32 — the gemv twin of encProjQuantBF16In.
func encProjBF16From(enc metal.MTLComputeCommandEncoder, w *model.BF16Weight, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
	mat := residentBytes(w.Data)
	var perr error
	if L == 1 {
		perr = encGemvBF16VecAt(enc, mat, xBF, dstBF, 0, 0, 0, outDim, inDim)
	} else {
		perr = encGemvBF16BatchedAt(enc, mat, xBF, dstBF, 0, 0, 0, outDim, inDim, L)
	}
	if perr != nil {
		return perr
	}
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
	return encWidenBF16ToF32(enc, dstBF, dst, L*outDim)
}

// encProjBF16F32 is encProjBF16From with the f32→bf16 input cast included — the gemv twin of
// encProjQuantF32.
func encProjBF16F32(enc metal.MTLComputeCommandEncoder, w *model.BF16Weight, x, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
	if err := encNarrowF32ToBF16(enc, x, xBF, L*inDim); err != nil {
		return err
	}
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
	return encProjBF16From(enc, w, xBF, dstBF, dst, L, outDim, inDim)
}

// encResidualNormMLPBF16Tail is the packed tail's raw-bf16 twin: the shared tail core with gemv
// projections over the checkpoint's own gate/up/down bytes.
func encResidualNormMLPBF16Tail(enc metal.MTLComputeCommandEncoder, tb quantTailBufs, normW metal.MTLBuffer, gate, up, down *model.BF16Weight, L, D, FF int, eps float32) error {
	return encResidualNormMLPTailCore(enc, tb, normW,
		func(enc metal.MTLComputeCommandEncoder, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjBF16From(enc, gate, xBF, dstBF, dst, L, outDim, inDim)
		},
		func(enc metal.MTLComputeCommandEncoder, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjBF16From(enc, up, xBF, dstBF, dst, L, outDim, inDim)
		},
		func(enc metal.MTLComputeCommandEncoder, x, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjBF16F32(enc, down, x, xBF, dstBF, dst, L, outDim, inDim)
		}, L, D, FF, eps)
}

// gatedDeltaBF16LayerRun runs one WHOLE dense bf16 gated-delta layer in a single command buffer —
// gatedDeltaQuantLayerRun with the checkpoint's raw bf16 bytes riding the gemv where the packed
// codes ride the affine qmv: input RMSNorm → the five bf16 projections (cast-once/project-many) →
// the gated-delta block stages (state resident) → out_proj → the bf16 FFN tail. x [L,D] is the
// only upload, y [L,D] the only readback; the per-stage bf16 path pays seven CB round-trips.
func gatedDeltaBF16LayerRun(
	h *gatedDeltaDeviceState,
	x, inputNorm []float32,
	w *model.GatedDeltaWeights,
	postNorm []float32,
	gate, up, down *model.BF16Weight,
	L, D, FF int, eps float32,
	priorConv, priorDelta []float32,
	y []float32,
) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if h == nil || !gatedDeltaBlockUsable(h.Dk, h.Dv, h.Hk, h.Hv, h.K) {
		return core.NewError("native.gatedDeltaBF16LayerRun: geometry not servable")
	}
	convDim, vDim := h.convDim, h.Hv*h.Dv
	if L <= 0 || len(x) != L*D || len(y) != L*D || len(inputNorm) != D || len(postNorm) != D ||
		len(w.ALog) != h.Hv || len(w.DtBias) != h.Hv || len(w.Norm) != h.Dv ||
		(w.ConvBias != nil && len(w.ConvBias) != convDim) || len(w.ConvWeight) != convDim*h.K {
		return core.NewError("native.gatedDeltaBF16LayerRun: size mismatch")
	}
	if !bf16GeometryOK(w.InProjQKVB, convDim, D) || !bf16GeometryOK(w.InProjZB, vDim, D) ||
		!bf16GeometryOK(w.InProjAB, h.Hv, D) || !bf16GeometryOK(w.InProjBB, h.Hv, D) ||
		!bf16GeometryOK(w.OutProjB, D, vDim) ||
		!bf16GeometryOK(gate, FF, D) || !bf16GeometryOK(up, FF, D) || !bf16GeometryOK(down, D, FF) {
		return core.NewError("native.gatedDeltaBF16LayerRun: unsupported bf16 geometry")
	}
	if !h.valid {
		h.prime(priorConv, priorDelta)
	}
	rmsName := "rmsfloat32"
	if D > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32"
	}
	psoRMS, err := pipelineFor(rmsName)
	if err != nil {
		return err
	}
	key := gatedDeltaQuantLayerKey{L: L, D: D, FF: FF, Hk: h.Hk, Hv: h.Hv, Dk: h.Dk, K: h.K}
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getGatedDeltaQuantLayerScratch(key)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putGatedDeltaQuantLayerScratch(key, sc)
		xBuf, cerr := sc.x.copyBuffer(float32Bytes(x))
		if cerr != nil {
			encErr = cerr
			return
		}
		wConv := residentFloat32(w.ConvWeight)
		wBias := wConv
		hasBias := 0
		if w.ConvBias != nil {
			wBias = residentFloat32(w.ConvBias)
			hasBias = 1
		}
		wALog := residentFloat32(w.ALog)
		wDt := residentFloat32(w.DtBias)
		wNorm := residentFloat32(w.Norm)
		inNormBuf := residentFloat32(inputNorm)
		postNormBuf := residentFloat32(postNorm)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		fail := func(err error) {
			encErr = err
			endEncodingFast(enc)
		}

		emitRMSNormRows(encSink{enc}, psoRMS, xBuf, inNormBuf, sc.normed1.buf, 0, 0, 0, D, eps, L, rmsThreadgroup(D, psoRMS))
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encNarrowF32ToBF16(enc, sc.normed1.buf, sc.n1BF.buf, L*D); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encProjBF16From(enc, w.InProjQKVB, sc.n1BF.buf, sc.qkvBF.buf, sc.qkv.buf, L, convDim, D); err != nil {
			fail(err)
			return
		}
		if err := encProjBF16From(enc, w.InProjZB, sc.n1BF.buf, sc.zBF.buf, sc.z.buf, L, vDim, D); err != nil {
			fail(err)
			return
		}
		if err := encProjBF16From(enc, w.InProjAB, sc.n1BF.buf, sc.aBF.buf, sc.a.buf, L, h.Hv, D); err != nil {
			fail(err)
			return
		}
		if err := encProjBF16From(enc, w.InProjBB, sc.n1BF.buf, sc.bBF.buf, sc.b.buf, L, h.Hv, D); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)

		if err := encGatedDeltaBlockStages(enc, h, gdBlockStageBufs{
			qkv: sc.qkv.buf, z: sc.z.buf, a: sc.a.buf, b: sc.b.buf,
			qN: sc.qN.buf, kN: sc.kN.buf, vN: sc.vN.buf, g: sc.g.buf, beta: sc.beta.buf,
			gated: sc.gated.buf,
		}, wConv, wBias, hasBias, wALog, wDt, wNorm, L); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)

		if err := encProjBF16F32(enc, w.OutProjB, sc.gated.buf, sc.gatedBF.buf, sc.mixBF.buf, sc.mix.buf, L, D, vDim); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)

		if err := encResidualNormMLPBF16Tail(enc, quantTailBufs{
			h: xBuf, mix: sc.mix.buf, normed: sc.normed2.buf, nBF: sc.n2BF.buf,
			g: sc.gFF.buf, gBF: sc.gFFBF.buf, u: sc.uFF.buf, uBF: sc.uFFBF.buf, s: sc.sFF.buf, out: sc.out.buf,
		}, postNormBuf, gate, up, down, L, D, FF, eps); err != nil {
			fail(err)
			return
		}

		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(y, unsafe.Slice((*float32)(unsafe.Pointer(&sc.out.bytes[0])), L*D))
	})
	return encErr
}

// gatedDeltaBF16LayerDeviceHook adapts gatedDeltaBF16LayerRun to qwen3's declared bf16 layer seam —
// the same handle discipline as the quant layer hook.
func gatedDeltaBF16LayerDeviceHook(sc *attn.GatedDeltaScratch, x, inputNorm []float32, w *model.GatedDeltaWeights, cfg model.GatedDeltaConfig, postNorm []float32, gate, up, down *model.BF16Weight, L, D, FF int, eps float32, priorConv, priorDelta []float32) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	h, _ := sc.Device.(*gatedDeltaDeviceState)
	if h == nil {
		if !gatedDeltaBlockUsable(cfg.HeadDim, cfg.HeadDim, cfg.KeyHeads, cfg.ValueHeads, cfg.ConvKernel) {
			return nil, core.NewError("native.gatedDeltaBF16LayerDeviceHook: geometry not servable")
		}
		nh, err := newGatedDeltaDeviceState(cfg.KeyHeads, cfg.ValueHeads, cfg.HeadDim, cfg.HeadDim, cfg.ConvKernel, 1)
		if err != nil {
			return nil, err
		}
		h = nh
	}
	y := make([]float32, L*D)
	if err := gatedDeltaBF16LayerRun(h, x, inputNorm, w, postNorm, gate, up, down, L, D, FF, eps, priorConv, priorDelta, y); err != nil {
		return nil, err
	}
	sc.Device = h
	return y, nil
}

// --- the attention fold (#26 / #18 S6a step 1): two CBs around the host attention core ---
