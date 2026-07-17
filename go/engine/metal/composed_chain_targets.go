// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// composed_chain_targets.go — the composed chain's op emitters in TARGET form (see chainTarget
// in composed_chain_icb.go): each op is written once against the target and lands either on the
// live per-layer encoder (the re-encode path, byte-identical to the pre-ICB emitters) or into
// the ICB recording. The legacy enc* wrappers used outside the chain delegate here with a live
// target, so there is exactly one body per op — the dispatch_sink drift lesson applied at the
// wrapper layer.

// liveChainTarget wraps a live encoder as a chain target.
func liveChainTarget(enc metal.MTLComputeCommandEncoderObject) *chainTarget {
	return &chainTarget{enc: enc}
}

// chainCopyCast dispatches one of MLX's contiguous v_copy cast kernels (src→dst, n elements).
func chainCopyCast(t *chainTarget, kernel string, src, dst metal.MTLBuffer, n int) error {
	pso, err := t.pso(kernel)
	if err != nil {
		return err
	}
	emitUnary(t.cmd(), pso, src, dst, n)
	return nil
}

func chainNarrowF32ToBF16(t *chainTarget, src, dst metal.MTLBuffer, n int) error {
	return chainCopyCast(t, "v_copyfloat32bfloat16", src, dst, n)
}

func chainWidenBF16ToF32(t *chainTarget, src, dst metal.MTLBuffer, n int) error {
	return chainCopyCast(t, "v_copybfloat16float32", src, dst, n)
}

// chainProjQuantBF16In projects ALREADY-CAST bf16 rows through a packed weight (qmv at L==1,
// the qmm_t slab at L>1), widening the bf16 result to f32 — encProjQuantBF16In in target form.
func chainProjQuantBF16In(t *chainTarget, w *model.QuantWeight, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
	wq, ws, wb := residentBytes(w.Packed), residentBytes(w.Scales), residentBytes(w.Biases)
	var name string
	if L == 1 {
		name = qmvBF16KernelName(outDim, inDim, w.GroupSize, w.Bits)
	} else {
		name = qmmTKernelName(outDim, w.GroupSize, w.Bits)
	}
	pso, err := t.pso(name)
	if err != nil {
		return err
	}
	if L == 1 {
		emitQMVAt(t.cmd(), pso, wq, 0, ws, 0, wb, 0, xBF, 0, dstBF, 0, inDim, outDim)
	} else {
		emitQMMT(t.cmd(), pso, wq, 0, ws, 0, wb, 0, xBF, 0, dstBF, 0, L, outDim, inDim)
	}
	t.barrier()
	return chainWidenBF16ToF32(t, dstBF, dst, L*outDim)
}

// chainProjQuantF32 is chainProjQuantBF16In from f32 rows: cast, project, widen.
func chainProjQuantF32(t *chainTarget, w *model.QuantWeight, x, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
	if err := chainNarrowF32ToBF16(t, x, xBF, L*inDim); err != nil {
		return err
	}
	t.barrier()
	return chainProjQuantBF16In(t, w, xBF, dstBF, dst, L, outDim, inDim)
}

// chainProjBF16F32 projects f32 rows through a RAW bf16 weight (the dense-resident form):
// cast, gemv per row, widen — encProjBF16F32 in target form. L==1 on the chain today.
func chainProjBF16F32(t *chainTarget, w *model.BF16Weight, x, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
	if err := chainNarrowF32ToBF16(t, x, xBF, L*inDim); err != nil {
		return err
	}
	t.barrier()
	wBuf := residentBytes(w.Data)
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	pso, err := t.pso(gemvKernelName("bfloat16", bm, bn, sm, sn, tm, tn))
	if err != nil {
		return err
	}
	for row := 0; row < L; row++ {
		emitGemvVecAt(t.cmd(), pso, wBuf, 0, xBF, uint(row*inDim*bf16Size), dstBF, uint(row*outDim*bf16Size), inDim, outDim, bm, bn, sm, tm)
	}
	t.barrier()
	return chainWidenBF16ToF32(t, dstBF, dst, L*outDim)
}

// chainTailProjFromBF16 / chainTailProjFromF32 are the tail core's projection closures in
// target form (packed codes or raw bf16 — one tail body serves both).
type chainTailProjFromBF16 func(t *chainTarget, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error
type chainTailProjFromF32 func(t *chainTarget, x, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error

// chainResidualNormMLPTailCore is encResidualNormMLPTailCore in target form: the FFN tail
// (mix residual → RMSNorm → SwiGLU → down → MLP residual) between a chain layer's mixer and
// its output hidden.
func chainResidualNormMLPTailCore(t *chainTarget, tb quantTailBufs, normW metal.MTLBuffer, gateProj, upProj chainTailProjFromBF16, downProj chainTailProjFromF32, L, D, FF int, eps float32) error {
	psoAdd, err := t.pso("vv_Addfloat32")
	if err != nil {
		return err
	}
	rmsName := "rmsfloat32"
	if D > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32"
	}
	psoRMS, err := t.pso(rmsName)
	if err != nil {
		return err
	}
	psoSig, err := t.pso("v_Sigmoidfloat32float32")
	if err != nil {
		return err
	}
	psoMul, err := t.pso("vv_Multiplyfloat32")
	if err != nil {
		return err
	}
	nD, nFF := L*D, L*FF
	rmsTG := rmsThreadgroup(D, psoRMS)
	emitBinary(t.cmd(), psoAdd, tb.h, 0, tb.mix, 0, tb.h, 0, nD)
	t.barrier()
	emitRMSNormRows(t.cmd(), psoRMS, tb.h, normW, tb.normed, 0, 0, 0, D, eps, L, rmsTG)
	t.barrier()
	if err := chainNarrowF32ToBF16(t, tb.normed, tb.nBF, nD); err != nil {
		return err
	}
	t.barrier()
	if err := gateProj(t, tb.nBF, tb.gBF, tb.g, L, FF, D); err != nil {
		return err
	}
	if err := upProj(t, tb.nBF, tb.uBF, tb.u, L, FF, D); err != nil {
		return err
	}
	t.barrier()
	emitUnary(t.cmd(), psoSig, tb.g, tb.s, nFF)
	t.barrier()
	emitBinary(t.cmd(), psoMul, tb.s, 0, tb.g, 0, tb.s, 0, nFF)
	t.barrier()
	emitBinary(t.cmd(), psoMul, tb.s, 0, tb.u, 0, tb.s, 0, nFF)
	t.barrier()
	if err := downProj(t, tb.s, tb.gBF, tb.nBF, tb.out, L, D, FF); err != nil {
		return err
	}
	t.barrier()
	emitBinary(t.cmd(), psoAdd, tb.h, 0, tb.out, 0, tb.out, 0, nD)
	return nil
}

// chainResidualNormMLPQuantTail is the packed-weight tail (both projections through the qmv).
func chainResidualNormMLPQuantTail(t *chainTarget, tb quantTailBufs, normW metal.MTLBuffer, gate, up, down *model.QuantWeight, L, D, FF int, eps float32) error {
	return chainResidualNormMLPTailCore(t, tb, normW,
		func(t *chainTarget, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return chainProjQuantBF16In(t, gate, xBF, dstBF, dst, L, outDim, inDim)
		},
		func(t *chainTarget, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return chainProjQuantBF16In(t, up, xBF, dstBF, dst, L, outDim, inDim)
		},
		func(t *chainTarget, x, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return chainProjQuantF32(t, down, x, xBF, dstBF, dst, L, outDim, inDim)
		}, L, D, FF, eps)
}

// --- the attention core in target form (position through posBind, so a recorded command
// replays at the host-bumped position) ---

func chainAttnQPrep(t *chainTarget, qRaw, w, q, gate metal.MTLBuffer, posBuf metal.MTLBuffer, L, H, HD, RD, gated, qkNorm int, eps, theta float32, pos0 int) error {
	pso, err := t.customPSO(attnQPrepPipeline, "lthn_attn_qprep_f32")
	if err != nil {
		return err
	}
	s := t.cmd()
	s.setPSO(pso)
	s.setBuf(qRaw, 0, 0)
	s.setBuf(w, 0, 1)
	s.setBuf(q, 0, 2)
	s.setBuf(gate, 0, 3)
	s.setI32(int32(H), 4)
	s.setI32(int32(HD), 5)
	s.setI32(int32(RD), 6)
	s.setI32(int32(gated), 7)
	s.setI32(int32(qkNorm), 8)
	s.setF32(eps, 9)
	s.setF32(theta, 10)
	t.posBind(s, posBuf, pos0, 11)
	s.dispatchThreadgroups(
		metal.MTLSize{Width: 1, Height: uint(H), Depth: uint(L)},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1})
	return nil
}

func chainAttnKPrep(t *chainTarget, k, w, cacheK metal.MTLBuffer, posBuf metal.MTLBuffer, L, KVH, HD, RD, qkNorm int, eps, theta float32, pos0 int) error {
	pso, err := t.customPSO(attnKPrepPipeline, "lthn_attn_kprep_f32")
	if err != nil {
		return err
	}
	s := t.cmd()
	s.setPSO(pso)
	s.setBuf(k, 0, 0)
	s.setBuf(w, 0, 1)
	s.setBuf(cacheK, 0, 2)
	s.setI32(int32(KVH), 3)
	s.setI32(int32(HD), 4)
	s.setI32(int32(RD), 5)
	s.setI32(int32(qkNorm), 6)
	s.setF32(eps, 7)
	s.setF32(theta, 8)
	t.posBind(s, posBuf, pos0, 9)
	s.dispatchThreadgroups(
		metal.MTLSize{Width: 1, Height: uint(KVH), Depth: uint(L)},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1})
	return nil
}

func chainAttnSDPA(t *chainTarget, q, cacheK, cacheV, out metal.MTLBuffer, posBuf metal.MTLBuffer, L, H, KVH, HD, pos0, window int) error {
	pso, err := t.customPSO(attnSDPAPipeline, "lthn_attn_sdpa_f32")
	if err != nil {
		return err
	}
	s := t.cmd()
	s.setPSO(pso)
	s.setBuf(q, 0, 0)
	s.setBuf(cacheK, 0, 1)
	s.setBuf(cacheV, 0, 2)
	s.setBuf(out, 0, 3)
	s.setI32(int32(H), 4)
	s.setI32(int32(KVH), 5)
	s.setI32(int32(HD), 6)
	t.posBind(s, posBuf, pos0, 7)
	s.setI32(int32(window), 8)
	s.dispatchThreadgroups(
		metal.MTLSize{Width: 1, Height: uint(H), Depth: uint(L)},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1})
	return nil
}

func chainAttnGateSilu(t *chainTarget, out, gate metal.MTLBuffer, total int) error {
	pso, err := t.customPSO(attnGatePipeline, "lthn_attn_gate_sigmoid_f32")
	if err != nil {
		return err
	}
	s := t.cmd()
	s.setPSO(pso)
	s.setBuf(out, 0, 0)
	s.setBuf(gate, 0, 1)
	s.setI32(int32(total), 2)
	s.dispatchThreadgroups(
		metal.MTLSize{Width: uint((total + 255) / 256), Height: 1, Depth: 1},
		metal.MTLSize{Width: 256, Height: 1, Depth: 1})
	return nil
}

// chainAttnVAppend lands the L freshly-projected V rows into the cache at the position row —
// the compute twin of the blit copy (an ICB carries compute commands only, and the position
// must come from the bumpable buffer). See kernels/lthn_attn_vappend_f32.metal.
func chainAttnVAppend(t *chainTarget, v, cacheV metal.MTLBuffer, posBuf metal.MTLBuffer, L, rowDim, pos0 int) error {
	pso, err := t.customPSO(attnVAppendPipeline, "lthn_attn_vappend_f32")
	if err != nil {
		return err
	}
	total := L * rowDim
	s := t.cmd()
	s.setPSO(pso)
	s.setBuf(v, 0, 0)
	s.setBuf(cacheV, 0, 1)
	s.setI32(int32(rowDim), 2)
	t.posBind(s, posBuf, pos0, 3)
	s.setI32(int32(total), 4)
	s.dispatchThreadgroups(
		metal.MTLSize{Width: uint((total + 255) / 256), Height: 1, Depth: 1},
		metal.MTLSize{Width: 256, Height: 1, Depth: 1})
	return nil
}

// --- the gated-delta block in target form ---

func chainGatedDeltaStepF32(t *chainTarget, q, k, v, g, beta, state, y metal.MTLBuffer, T, kSlots, Hk, Hv, Dk, Dv int) error {
	if T <= 0 || kSlots < 1 || Dv <= 0 || Hk <= 0 || Hv <= 0 || Hv%Hk != 0 {
		return core.NewError("native.chainGatedDeltaStepF32: invalid geometry")
	}
	name := "lthn_gated_delta_step_f32_dk64"
	if Dk == 128 {
		name = "lthn_gated_delta_step_f32_dk128"
	} else if Dk != 64 {
		return core.NewError("native.chainGatedDeltaStepF32: unsupported key head dim")
	}
	pso, err := t.customPSO(func() (metal.MTLComputePipelineState, error) { return gatedDeltaStepPipeline(Dk) }, name)
	if err != nil {
		return err
	}
	s := t.cmd()
	s.setPSO(pso)
	s.setBuf(q, 0, 0)
	s.setBuf(k, 0, 1)
	s.setBuf(v, 0, 2)
	s.setBuf(g, 0, 3)
	s.setBuf(beta, 0, 4)
	s.setBuf(state, 0, 5)
	s.setBuf(y, 0, 6)
	s.setI32(int32(T), 7)
	s.setI32(int32(kSlots), 8)
	s.setI32(int32(Hk), 9)
	s.setI32(int32(Hv), 10)
	s.setI32(int32(Dv), 11)
	const dvPerGroup = 4
	s.dispatchThreadgroups(
		metal.MTLSize{Width: 1, Height: uint((Dv + dvPerGroup - 1) / dvPerGroup), Depth: uint(Hv)},
		metal.MTLSize{Width: 32, Height: dvPerGroup, Depth: 1})
	return nil
}

// chainGatedDeltaBlockStages is encGatedDeltaBlockStages in target form: conv+SiLU+split+norm →
// ring advance ∥ gates → recurrence → gated RMSNorm·SiLU(z). Positions never appear — the conv
// ring and the recurrence state advance on device, so the recorded commands replay unchanged.
func chainGatedDeltaBlockStages(t *chainTarget, st *gatedDeltaDeviceState, b gdBlockStageBufs, wConv, wBias metal.MTLBuffer, hasBias int, wALog, wDt, wNorm metal.MTLBuffer, L int) error {
	convName := "lthn_gd_conv_silu_split_norm_dk64"
	if st.Dk == 128 {
		convName = "lthn_gd_conv_silu_split_norm_dk128"
	}
	convPSO, err := t.customPSO(func() (metal.MTLComputePipelineState, error) { return gdConvPipeline(st.Dk) }, convName)
	if err != nil {
		return err
	}
	ringPSO, err := t.customPSO(gdRingPipeline, "lthn_gd_ring_advance")
	if err != nil {
		return err
	}
	gatesPSO, err := t.customPSO(gdGatesPipeline, "lthn_gd_gates")
	if err != nil {
		return err
	}
	normName := "lthn_gd_gated_rmsnorm_silu_dv64"
	if st.Dv == 128 {
		normName = "lthn_gd_gated_rmsnorm_silu_dv128"
	}
	normPSO, err := t.customPSO(func() (metal.MTLComputePipelineState, error) { return gdNormPipeline(st.Dv) }, normName)
	if err != nil {
		return err
	}

	s := t.cmd()
	s.setPSO(convPSO)
	s.setBuf(st.ring.buf, 0, 0)
	s.setBuf(b.qkv, 0, 1)
	s.setBuf(wConv, 0, 2)
	s.setBuf(wBias, 0, 3)
	s.setBuf(b.qN, 0, 4)
	s.setBuf(b.kN, 0, 5)
	s.setBuf(b.vN, 0, 6)
	s.setI32(int32(L), 7)
	s.setI32(int32(st.K), 8)
	s.setI32(int32(st.Hk), 9)
	s.setI32(int32(st.Hv), 10)
	s.setI32(int32(hasBias), 11)
	s.dispatchThreadgroups(
		metal.MTLSize{Width: 1, Height: uint(2*st.Hk + st.Hv), Depth: uint(L)},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1})
	t.barrier()

	s = t.cmd()
	s.setPSO(ringPSO)
	s.setBuf(st.ring.buf, 0, 0)
	s.setBuf(b.qkv, 0, 1)
	s.setI32(int32(L), 2)
	s.setI32(int32(st.K), 3)
	s.setI32(int32(st.convDim), 4)
	s.dispatchThreadgroups(
		metal.MTLSize{Width: uint((st.convDim + 255) / 256), Height: 1, Depth: 1},
		metal.MTLSize{Width: 256, Height: 1, Depth: 1})

	s = t.cmd()
	s.setPSO(gatesPSO)
	s.setBuf(b.a, 0, 0)
	s.setBuf(b.b, 0, 1)
	s.setBuf(wALog, 0, 2)
	s.setBuf(wDt, 0, 3)
	s.setBuf(b.g, 0, 4)
	s.setBuf(b.beta, 0, 5)
	s.setI32(int32(L*st.Hv), 6)
	s.setI32(int32(st.Hv), 7)
	s.dispatchThreadgroups(
		metal.MTLSize{Width: uint((L*st.Hv + 255) / 256), Height: 1, Depth: 1},
		metal.MTLSize{Width: 256, Height: 1, Depth: 1})
	t.barrier()

	if err := chainGatedDeltaStepF32(t, b.qN, b.kN, b.vN, b.g, b.beta, st.state.buf, b.gated, L, st.kSlots, st.Hk, st.Hv, st.Dk, st.Dv); err != nil {
		return err
	}
	t.barrier()

	s = t.cmd()
	s.setPSO(normPSO)
	s.setBuf(b.gated, 0, 0)
	s.setBuf(b.z, 0, 1)
	s.setBuf(wNorm, 0, 2)
	s.setBuf(b.gated, 0, 3)
	s.setI32(int32(L*st.Hv), 4)
	s.setF32(1e-6, 5)
	s.dispatchThreadgroups(
		metal.MTLSize{Width: 1, Height: uint(L * st.Hv), Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1})
	return nil
}
