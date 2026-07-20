// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// composed_chain_moe.go — the Qwen hybrid's Mixture-of-Experts FFN tail in chainTarget form,
// so a qwen3_5_moe layer rides the factory's whole-token chain instead of firing its own
// per-expert command buffers. The dense SwiGLU tail (chainResidualNormMLPQuantTail) already rides
// the chain; this is its MoE twin — residual add → RMSNorm → cast bf16 → [router → gather experts →
// SiLU SwiGLU → weighted combine → shared expert] → residual add — every op emitted through the
// chainTarget (t.cmd()/t.pso()/t.barrier()) so the whole layer lands on ONE command buffer.
//
// The routing is a DEVICE decision: the router gemv + lthn_moe_router_topk_bf16 write the selected
// expert ids and their softmax-over-topK weights to device buffers, and lthn_gather_qmv reads that
// idx buffer to project each routed expert — so the recorded/re-encoded stream is fixed across
// tokens while the experts it touches change per token. This reuses the EXISTING kernels the
// batched-MoE block (encMoEBlockQuantBatched) drives: lthn_gather_qmv is byte-identical to MLX's
// affine_gather_qmv (same qmv_fast_impl/qmv_impl), and lthn_moe_weighted_sum_bf16 is byte-identical
// to the per-route scale+add chain — so the chain body equals MoEExpertsQuantSiLU + shared to ≤1 bf16
// ULP (gated in composed_chain_moe_test.go). The activation is SiLU (encSiLUGateMulBF16's ops), never
// gemma's GELU — a GELU variant would DIFFER, which is the one fact a greedy model A/B cannot show.
//
// LIVE-only: the MoE tail declines the ICB recording pass (the custom router/combine/scale kernels
// have no ICB-specialised pipeline yet) — the RE-ENCODE chain still collapses the ~378 per-token
// command buffers into ONE, which is the bulk of the win; the recorded replay is a later increment.

// chainMoEShared is the shared expert's packed SwiGLU trio — nil projections decline the chain
// (a fused GateUpQ or dense shared expert is not chain-servable).
type chainMoEShared struct {
	GateQ, UpQ, DownQ *model.QuantWeight
}

// chainMoE is the minimal MoE view the fused whole-token chain resolves its weights from — the
// native port (#50) of the retired composed engine's MoEMLP, carrying exactly the fields the chain
// emitters read. qwenChainMoE (arch_qwen_fused.go) builds it from the factory's native MoE holder:
// batched switch_mlp experts, a softmax top-k router dequantised to f32 once, the shared expert
// trio and its optional sigmoid gate.
type chainMoE struct {
	Router       []float32 // [NumExperts, D] dequantised router rows
	NumExperts   int
	TopK         int
	NormTopKProb bool // renormalise the top-k router weights over the selection
	Gating       model.MoEGating

	// GateBatchedQ/UpBatchedQ/DownBatchedQ are the WHOLE switch_mlp.{gate,up,down}_proj tensors as
	// one packed [numExperts, …] quant weight each — the batched form the gather kernels consume.
	GateBatchedQ, UpBatchedQ, DownBatchedQ *model.QuantWeight

	Shared     *chainMoEShared // nil ⇒ no shared expert
	SharedGate []float32       // [D] shared_expert_gate weight; nil ⇒ shared added ungated
}

// moeChainWeights are one MoE layer's weights resolved to the views the chain emitters bind: the
// router narrowed to a bf16 device buffer, the batched routed experts as native QuantWeights (every
// expert concatenated — the gather offsets by the device idx), the shared expert's packed SwiGLU,
// and the optional shared-expert sigmoid gate narrowed to bf16. resolveMoEChainWeights builds it
// from the *chainMoE view; the byte-identity test builds it directly from a fixture.
type moeChainWeights struct {
	routerBF     metal.MTLBuffer // [numExperts, D] bf16 (the dequant f32 router, narrowed once)
	routerBacker []byte          // holds routerBF's bytes alive (residentBytes is no-copy)

	numExperts, topK, expertDFF, groupSize, bits int
	gate, up, down                               QuantWeight // batched [numExperts, …] routed experts

	hasShared         bool
	sGate, sUp, sDown QuantWeight // the shared expert's packed SwiGLU (unfused)
	sharedFF          int
	sharedGateBF      metal.MTLBuffer // [D] bf16 shared_expert_gate; nil ⇒ shared added ungated
	sharedGateBacker  []byte
}

// moeChainScratch is the MoE tail's device staging — the router/gather/combine slabs plus the
// shared-expert working buffers. Sized by (D, numExperts, topK, expertDFF, sharedFF); every buffer
// is a fresh shared allocation (stable across the encode, unlike a pooled scratch).
type moeChainScratch struct {
	scores       metal.MTLBuffer // [numExperts] bf16 router scores
	idxBuf       metal.MTLBuffer // [topK] int32 selected expert ids
	weightBuf    metal.MTLBuffer // [topK] bf16 softmax-over-topK route weights
	routeZeros   metal.MTLBuffer // [topK] int32 all-zero (gate/up shared-x lhs; unread)
	routeIota    metal.MTLBuffer // [topK] int32 identity (down per-route lhs)
	gateAll      metal.MTLBuffer // [topK × expertDFF] bf16
	upAll        metal.MTLBuffer // [topK × expertDFF] bf16
	gatedAll     metal.MTLBuffer // [topK × expertDFF] bf16
	downAll      metal.MTLBuffer // [topK × D] bf16
	sGate        metal.MTLBuffer // [sharedFF] bf16 shared gate
	sUp          metal.MTLBuffer // [sharedFF] bf16 shared up
	sGated       metal.MTLBuffer // [sharedFF] bf16 shared silu(gate)·up
	sharedOut    metal.MTLBuffer // [D] bf16 shared down
	gScore       metal.MTLBuffer // [1] bf16 shared-gate dot
	gVal         metal.MTLBuffer // [1] bf16 σ(shared-gate dot)
	sharedScaled metal.MTLBuffer // [D] bf16 g·shared
	bodyBF       metal.MTLBuffer // [D] bf16 the combined MoE body (before the outer residual)
	bodyF32      metal.MTLBuffer // [D] f32 the widened body (for the residual add)
}

// newMoEChainScratch allocates the MoE tail staging for one geometry. sharedFF may equal expertDFF
// or differ (a distinct shared_expert_intermediate_size); pass sharedFF=0 for a shared-less layer.
func newMoEChainScratch(dModel, numExperts, topK, expertDFF, sharedFF int) (*moeChainScratch, error) {
	if dModel <= 0 || numExperts <= 0 || topK <= 0 || expertDFF <= 0 {
		return nil, core.NewError("native.newMoEChainScratch: invalid geometry")
	}
	zeros := make([]int32, topK)
	iota := make([]int32, topK)
	for i := range iota {
		iota[i] = int32(i)
	}
	sff := max(sharedFF, 1)
	sc := &moeChainScratch{
		scores:       scratchBF16(numExperts),
		idxBuf:       device.NewBufferWithLengthOptions(uint(topK*4), metal.MTLResourceStorageModeShared),
		weightBuf:    scratchBF16(topK),
		routeZeros:   device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&zeros[0]), uint(topK*4), metal.MTLResourceStorageModeShared),
		routeIota:    device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&iota[0]), uint(topK*4), metal.MTLResourceStorageModeShared),
		gateAll:      scratchBF16(topK * expertDFF),
		upAll:        scratchBF16(topK * expertDFF),
		gatedAll:     scratchBF16(topK * expertDFF),
		downAll:      scratchBF16(topK * dModel),
		sGate:        scratchBF16(sff),
		sUp:          scratchBF16(sff),
		sGated:       scratchBF16(sff),
		sharedOut:    scratchBF16(dModel),
		gScore:       scratchBF16(1),
		gVal:         scratchBF16(1),
		sharedScaled: scratchBF16(dModel),
		bodyBF:       scratchBF16(dModel),
		bodyF32:      scratchF32(dModel),
	}
	if sc.scores == nil || sc.idxBuf == nil || sc.weightBuf == nil || sc.routeZeros == nil || sc.routeIota == nil ||
		sc.gateAll == nil || sc.upAll == nil || sc.gatedAll == nil || sc.downAll == nil || sc.sharedOut == nil ||
		sc.sharedScaled == nil || sc.bodyBF == nil || sc.bodyF32 == nil {
		return nil, core.NewError("native.newMoEChainScratch: buffer allocation failed")
	}
	return sc, nil
}

// chainLthnGatherPSO resolves the lean lthn_gather_qmv pipeline for a target: the ICB-capable
// build in recording mode, the live pipeline otherwise. Both bake expertRows/batchedX as function
// constants on the gs/bits template — customPipelineForICB cannot (it resolves names only), which
// is why the MoE tail declines recording when only the live pipeline exists.
func chainLthnGatherPSO(t *chainTarget, key lthnGatherQMVKey) (metal.MTLComputePipelineState, bool) {
	if t.recording() {
		return lthnGatherQMVPipelineICB(key)
	}
	return lthnGatherQMVPipeline(key)
}

// chainQMVBF16To projects an already-cast bf16 vector through a packed weight into a bf16 result —
// encQMVBF16 in target form, WITHOUT the f32 widen chainProjQuantBF16In appends (the SwiGLU
// intermediate stays bf16). Byte-identical to encQMVBF16 (same qmvBF16KernelName + emitQMVAt).
func chainQMVBF16To(t *chainTarget, w QuantWeight, xBF, dstBF metal.MTLBuffer, outDim, inDim int) error {
	pso, err := t.pso(qmvBF16KernelName(outDim, inDim, w.GroupSize, w.Bits))
	if err != nil {
		return err
	}
	emitQMVAt(t.cmd(), pso, residentBytes(w.Packed), 0, residentBytes(w.Scales), 0, residentBytes(w.Biases), 0, xBF, 0, dstBF, 0, inDim, outDim)
	t.barrier()
	return nil
}

// chainSiLUGateMulBF16 is encSiLUGateMulBF16 in target form: out = silu(gate)·up = gate·σ(gate)·up,
// the SwiGLU gate (llama/mistral/qwen), composed from the bf16 sigmoid + two muls. out must NOT alias
// gate (σ(gate) overwrites out first). n contiguous bf16 elements.
func chainSiLUGateMulBF16(t *chainTarget, gate, up, out metal.MTLBuffer, n int) error {
	psoSig, err := t.pso("v_Sigmoidbfloat16bfloat16")
	if err != nil {
		return err
	}
	psoMul, err := t.pso("vv_Multiplybfloat16")
	if err != nil {
		return err
	}
	emitUnary(t.cmd(), psoSig, gate, out, n) // out = σ(gate)
	t.barrier()
	emitBinary(t.cmd(), psoMul, gate, 0, out, 0, out, 0, n) // out = gate·σ(gate) = silu(gate)
	t.barrier()
	emitBinary(t.cmd(), psoMul, out, 0, up, 0, out, 0, n) // out = silu(gate)·up
	t.barrier()
	return nil
}

// chainMoEWeightedSum records lthn_moe_weighted_sum_bf16: out[d] = Σ_r weights[r]·rows[r,d] over the
// topK routes, one dispatch, byte-identical rounding to the per-route scale+add chain. K=1 decode
// (grid height 1).
func chainMoEWeightedSum(t *chainTarget, rows, weights, out metal.MTLBuffer, n, topK int) error {
	pso, err := t.customPSO(func() (metal.MTLComputePipelineState, error) { return moeWeightedSumPipeline() }, "lthn_moe_weighted_sum_bf16")
	if err != nil {
		return err
	}
	s := t.cmd()
	s.setPSO(pso)
	s.setBuf(rows, 0, 0)
	s.setBuf(weights, 0, 1)
	s.setBuf(out, 0, 2)
	s.setI32(int32(n), 3)
	s.setI32(int32(topK), 4)
	group := min(uint(n), uint(256))
	s.dispatchThreads(
		metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
		metal.MTLSize{Width: group, Height: 1, Depth: 1},
	)
	return nil
}

// chainMoEScaleBF16 records lthn_bf16_mul_scalar: out[i] = in[i]·scalar[0] — the shared expert's
// sigmoid-gate scale. Byte-identical to encScaleBF16 (and to vv_Multiply against a broadcast scalar,
// the moe_quant test reference).
func chainMoEScaleBF16(t *chainTarget, in, scalar, out metal.MTLBuffer, n int) error {
	pso, err := t.customPSO(func() (metal.MTLComputePipelineState, error) { return bf16MulScalarPipeline() }, "lthn_bf16_mul_scalar")
	if err != nil {
		return err
	}
	s := t.cmd()
	s.setPSO(pso)
	s.setBuf(in, 0, 0)
	s.setBuf(scalar, 0, 1)
	s.setBuf(out, 0, 2)
	s.setI32(int32(n), 3)
	group := min(uint(n), uint(256))
	s.dispatchThreads(
		metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
		metal.MTLSize{Width: group, Height: 1, Depth: 1},
	)
	return nil
}

// chainMoERouterTopK records the router selection: scores = routerBF·normed (gemv), then
// lthn_moe_router_topk_bf16 writes the topK expert ids (idxBuf) and their softmax-over-topK weights
// (weightBuf) — the same device router encMoEBlockQuantBatched drives, but reading the tail's ONE
// norm (no router-private RMSNorm). No per-expert scale on the composed lane.
func chainMoERouterTopK(t *chainTarget, sc *moeChainScratch, nBF metal.MTLBuffer, w *moeChainWeights, D int) error {
	bm, bn, sm, sn, tm, tn := gemvTiles(D, w.numExperts)
	gpso, err := t.pso(gemvKernelName("bfloat16", bm, bn, sm, sn, tm, tn))
	if err != nil {
		return err
	}
	emitGemv(t.cmd(), gpso, w.routerBF, 0, nBF, sc.scores, 0, D, w.numExperts, bm, bn, sm, tm)
	t.barrier()
	tkPSO, err := t.customPSO(func() (metal.MTLComputePipelineState, error) { return routerTopKPipelineK(w.topK) }, "lthn_moe_router_topk_bf16")
	if err != nil {
		return err
	}
	s := t.cmd()
	s.setPSO(tkPSO)
	s.setBuf(sc.scores, 0, 0)
	s.setBuf(sc.scores, 0, 1) // per_expert_scale unused (has_scale=0)
	s.setBuf(sc.idxBuf, 0, 2)
	s.setBuf(sc.weightBuf, 0, 3)
	s.setI32(int32(w.numExperts), 4)
	s.setI32(int32(w.topK), 5)
	s.setI32(0, 6) // has_scale=0
	s.dispatchThreads(
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
	t.barrier()
	return nil
}

// chainMoEBody records the whole MoE body over an ALREADY-NORMED bf16 input row nBF, writing the
// combined bf16 result into sc.bodyBF (BEFORE the outer residual add): router → gather routed
// experts (gate+up via lthn_gather_qmv, SiLU, down) → weighted combine → shared expert (SiLU
// SwiGLU × σ(gate)). This is the unit the byte-identity gate drives directly.
func chainMoEBody(t *chainTarget, sc *moeChainScratch, nBF metal.MTLBuffer, w *moeChainWeights, D int) error {
	ff, topK := w.expertDFF, w.topK
	if err := chainMoERouterTopK(t, sc, nBF, w, D); err != nil {
		return err
	}
	// routed experts, gate+up: one shared-x gather per projection reading the device idx buffer.
	gateKey := lthnGatherQMVKey{groupSize: w.groupSize, bits: w.bits, expertRows: ff, fast: ff%8 == 0 && D%512 == 0, batchedX: false}
	gPSO, ok := chainLthnGatherPSO(t, gateKey)
	if !ok {
		return core.NewError("native.chainMoEBody: gate/up gather pipeline unavailable")
	}
	emitLthnGatherQMVRoutes(t.cmd(), gPSO, nBF, 0,
		residentBytes(w.gate.Packed), 0, residentBytes(w.gate.Scales), 0, residentBytes(w.gate.Biases), 0,
		sc.routeZeros, sc.idxBuf, 0, sc.gateAll, 0, ff, D, w.groupSize, w.bits, 0, topK)
	emitLthnGatherQMVRoutes(t.cmd(), gPSO, nBF, 0,
		residentBytes(w.up.Packed), 0, residentBytes(w.up.Scales), 0, residentBytes(w.up.Biases), 0,
		sc.routeZeros, sc.idxBuf, 0, sc.upAll, 0, ff, D, w.groupSize, w.bits, 0, topK)
	t.barrier()
	if err := chainSiLUGateMulBF16(t, sc.gateAll, sc.upAll, sc.gatedAll, topK*ff); err != nil {
		return err
	}
	// routed down: per-route (batched-x) gather over the gated rows.
	downKey := lthnGatherQMVKey{groupSize: w.groupSize, bits: w.bits, expertRows: D, fast: D%8 == 0 && ff%512 == 0, batchedX: true}
	dPSO, ok := chainLthnGatherPSO(t, downKey)
	if !ok {
		return core.NewError("native.chainMoEBody: down gather pipeline unavailable")
	}
	emitLthnGatherQMVRoutes(t.cmd(), dPSO, sc.gatedAll, 0,
		residentBytes(w.down.Packed), 0, residentBytes(w.down.Scales), 0, residentBytes(w.down.Biases), 0,
		sc.routeIota, sc.idxBuf, 0, sc.downAll, 0, D, ff, w.groupSize, w.bits, 0, topK)
	t.barrier()
	if err := chainMoEWeightedSum(t, sc.downAll, sc.weightBuf, sc.bodyBF, D, topK); err != nil {
		return err
	}
	t.barrier()
	if !w.hasShared {
		return t.err
	}
	// shared expert: SiLU SwiGLU on the SAME normed input, added (σ(gate)-scaled) to the routed sum.
	if err := chainQMVBF16To(t, w.sGate, nBF, sc.sGate, w.sharedFF, D); err != nil {
		return err
	}
	if err := chainQMVBF16To(t, w.sUp, nBF, sc.sUp, w.sharedFF, D); err != nil {
		return err
	}
	if err := chainSiLUGateMulBF16(t, sc.sGate, sc.sUp, sc.sGated, w.sharedFF); err != nil {
		return err
	}
	if err := chainQMVBF16To(t, w.sDown, sc.sGated, sc.sharedOut, D, w.sharedFF); err != nil {
		return err
	}
	addPSO, err := t.pso("vv_Addbfloat16")
	if err != nil {
		return err
	}
	if w.sharedGateBF != nil {
		gbm, gbn, gsm, gsn, gtm, gtn := gemvTiles(D, 1)
		ggPSO, gerr := t.pso(gemvKernelName("bfloat16", gbm, gbn, gsm, gsn, gtm, gtn))
		if gerr != nil {
			return gerr
		}
		emitGemv(t.cmd(), ggPSO, w.sharedGateBF, 0, nBF, sc.gScore, 0, D, 1, gbm, gbn, gsm, gtm)
		t.barrier()
		sigPSO, serr := t.pso("v_Sigmoidbfloat16bfloat16")
		if serr != nil {
			return serr
		}
		emitUnary(t.cmd(), sigPSO, sc.gScore, sc.gVal, 1) // g = σ(sharedGate·normed)
		t.barrier()
		if err := chainMoEScaleBF16(t, sc.sharedOut, sc.gVal, sc.sharedScaled, D); err != nil {
			return err
		}
		t.barrier()
		emitBinary(t.cmd(), addPSO, sc.bodyBF, 0, sc.sharedScaled, 0, sc.bodyBF, 0, D) // body += g·shared
	} else {
		emitBinary(t.cmd(), addPSO, sc.bodyBF, 0, sc.sharedOut, 0, sc.bodyBF, 0, D) // body += shared (ungated)
	}
	t.barrier()
	return t.err
}

// chainResidualNormMoEQuantTail is the MoE twin of chainResidualNormMLPQuantTail: the FFN tail (mix
// residual → RMSNorm → cast bf16 → MoE body → MLP residual) between a chain layer's mixer output and
// its output hidden. tb.h holds the pre-tail hidden (residual, mutated in place with += mix, as the
// dense tail does); tb.out receives h + body. L must be 1 (the decode chain).
func chainResidualNormMoEQuantTail(t *chainTarget, tb moeTailBufs, normW metal.MTLBuffer, w *moeChainWeights, L, D int, eps float32) error {
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
	nD := L * D
	emitBinary(t.cmd(), psoAdd, tb.h, 0, tb.mix, 0, tb.h, 0, nD) // mixer residual: h += mix
	t.barrier()
	emitRMSNormRows(t.cmd(), psoRMS, tb.h, normW, tb.normed, 0, 0, 0, D, eps, L, rmsThreadgroup(D, psoRMS))
	t.barrier()
	if err := chainNarrowF32ToBF16(t, tb.normed, tb.nBF, nD); err != nil {
		return err
	}
	t.barrier()
	if err := chainMoEBody(t, tb.sc, tb.nBF, w, D); err != nil {
		return err
	}
	t.barrier()
	if err := chainWidenBF16ToF32(t, tb.sc.bodyF32src(), tb.sc.bodyF32, nD); err != nil {
		return err
	}
	t.barrier()
	emitBinary(t.cmd(), psoAdd, tb.h, 0, tb.sc.bodyF32, 0, tb.out, 0, nD) // MLP residual: out = h + body
	return nil
}

// bodyF32src returns the bf16 body buffer chainWidenBF16ToF32 widens into bodyF32.
func (sc *moeChainScratch) bodyF32src() metal.MTLBuffer { return sc.bodyBF }

// moeTailBufs is the MoE tail's buffer set: the mixer-supplied staging (h/mix/normed/nBF/out, reused
// from the layer's attn/gd scratch exactly as quantTailBufs is) plus the MoE body slabs (sc).
type moeTailBufs struct {
	h, mix, normed, nBF, out metal.MTLBuffer
	sc                       *moeChainScratch
}

// resolveMoEChainWeights bridges a *chainMoE view to the chain emitters' native views, returning
// ok=false when the layer cannot ride the MoE chain (missing batched experts, an unsupported gating
// or top-k policy, a fused shared expert, or an absent lean gather kernel for its geometry). The
// router + shared gate are narrowed to bf16 ONCE here (owned backers keep them resident).
func resolveMoEChainWeights(moe *chainMoE) (*moeChainWeights, bool) {
	if moe == nil || moe.GateBatchedQ == nil || moe.UpBatchedQ == nil || moe.DownBatchedQ == nil {
		return nil, false
	}
	if moe.Gating != model.MoEGatingSoftmax || !moe.NormTopKProb {
		return nil, false // the router topk kernel is softmax-over-topK, renormalised
	}
	if moe.Router == nil {
		return nil, false
	}
	gb, ub, db := moe.GateBatchedQ, moe.UpBatchedQ, moe.DownBatchedQ
	expertDFF, D := gb.OutDim, gb.InDim
	gs, bits := gb.GroupSize, gb.Bits
	if expertDFF <= 0 || D <= 0 || gs <= 0 || D%gs != 0 || moe.TopK <= 0 || moe.TopK > moe.NumExperts {
		return nil, false
	}
	if len(moe.Router) != moe.NumExperts*D {
		return nil, false
	}
	// the lean gather must exist for both projection geometries (else the chain can neither record
	// nor re-encode the routed experts — the MLX gather path binds a concrete encSink, incompatible
	// with the chain's object encoder).
	gateKey := lthnGatherQMVKey{groupSize: gs, bits: bits, expertRows: expertDFF, fast: expertDFF%8 == 0 && D%512 == 0, batchedX: false}
	downKey := lthnGatherQMVKey{groupSize: gs, bits: bits, expertRows: D, fast: D%8 == 0 && expertDFF%512 == 0, batchedX: true}
	if _, ok := lthnGatherQMVPipeline(gateKey); !ok {
		return nil, false
	}
	if _, ok := lthnGatherQMVPipeline(downKey); !ok {
		return nil, false
	}
	w := &moeChainWeights{
		numExperts: moe.NumExperts, topK: moe.TopK, expertDFF: expertDFF, groupSize: gs, bits: bits,
		gate: nativeQuant(gb), up: nativeQuant(ub), down: nativeQuant(db),
	}
	w.routerBacker = f32sToBF16Bytes(moe.Router)
	w.routerBF = residentBytes(w.routerBacker)
	if moe.Shared != nil {
		sh := moe.Shared
		if sh.GateQ == nil || sh.UpQ == nil || sh.DownQ == nil { // fused (GateUpQ) or dense shared not supported on the chain
			return nil, false
		}
		w.hasShared = true
		w.sGate, w.sUp, w.sDown = nativeQuant(sh.GateQ), nativeQuant(sh.UpQ), nativeQuant(sh.DownQ)
		w.sharedFF = sh.GateQ.OutDim
		if moe.SharedGate != nil {
			if len(moe.SharedGate) != D {
				return nil, false
			}
			w.sharedGateBacker = f32sToBF16Bytes(moe.SharedGate)
			w.sharedGateBF = residentBytes(w.sharedGateBacker)
		}
	}
	return w, true
}

// nativeQuant lifts a model.QuantWeight (the loader's packed form) to the engine's QuantWeight view.
func nativeQuant(q *model.QuantWeight) QuantWeight {
	return QuantWeight{Packed: q.Packed, Scales: q.Scales, Biases: q.Biases, GroupSize: q.GroupSize, Bits: q.Bits}
}

// moeChainRecordable reports whether this MoE layer can ride the whole-token chain —
// resolveMoEChainWeights succeeds. qwenChainReady (arch_qwen_fused.go) consults it per layer.
func moeChainRecordable(moe *chainMoE) bool {
	_, ok := resolveMoEChainWeights(moe)
	return ok
}
