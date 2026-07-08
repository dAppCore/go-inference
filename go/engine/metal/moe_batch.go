// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// moe_batch.go — the K-token BATCHED MoE block for the prompt prefill (the #338 P-A lane).
// The per-token MoE stepping measured 14.2 ms/prompt-token (49s for a 3.5k prompt); this
// encodes the whole block for a K-row chunk: rows-rms + one qmm for the router scores, the
// row-dimensioned topk, the local MLP as three qmm_t sweeps, the expert projections as
// ALL-PAIRS gathers (routes = K·topK — the decode's all-routes gather with a pair→token lhs
// map), and the row-dimensioned weighted-sum + fused norm/combine tails. Token-identity tier
// (the prompt-scale qmm fold's established bar, decode_batched_session.go).

// moeBatchScratch holds the K-row MoE fold slabs, grown as the chunk geometry grows. Owned
// by the state's denseBatch scratch — single-flight like every batch slab.
type moeBatchScratch struct {
	routerNorm metal.MTLBuffer // K × dModel
	scores     metal.MTLBuffer // K × numExperts
	idx        metal.MTLBuffer // K·topK int32 selected experts
	weights    metal.MTLBuffer // K·topK bf16 route weights
	localIn    metal.MTLBuffer // K × dModel
	localGate  metal.MTLBuffer // K × dFF
	localUp    metal.MTLBuffer // K × dFF
	localGated metal.MTLBuffer // K × dFF
	localOut   metal.MTLBuffer // K × dModel
	expertIn   metal.MTLBuffer // K × dModel
	gateAll    metal.MTLBuffer // K·topK × expertDFF
	upAll      metal.MTLBuffer // K·topK × expertDFF
	gatedAll   metal.MTLBuffer // K·topK × expertDFF
	downAll    metal.MTLBuffer // K·topK × dModel
	expertAcc  metal.MTLBuffer // K × dModel

	pairToToken metal.MTLBuffer // K·topK int32: pair → its token row (iota / topK)
	pairIota    metal.MTLBuffer // K·topK int32: identity

	rowCap, dModelCap, dFFCap, expertDFFCap, topKCap, numExpertsCap int
}

func (m *moeBatchScratch) ensure(k, dModel, dFF, expertDFF, topK, numExperts int) error {
	if m.rowCap >= k && m.dModelCap == dModel && m.dFFCap >= dFF && m.expertDFFCap >= expertDFF && m.topKCap >= topK && m.numExpertsCap >= numExperts {
		return nil
	}
	pairs := k * topK
	m.routerNorm = scratchBF16(k * dModel)
	m.scores = scratchBF16(k * numExperts)
	m.idx = device.NewBufferWithLengthOptions(uint(pairs*4), metal.MTLResourceStorageModeShared)
	m.weights = scratchBF16(pairs)
	m.localIn = scratchBF16(k * dModel)
	m.localGate = scratchBF16(k * dFF)
	m.localUp = scratchBF16(k * dFF)
	m.localGated = scratchBF16(k * dFF)
	m.localOut = scratchBF16(k * dModel)
	m.expertIn = scratchBF16(k * dModel)
	m.gateAll = scratchBF16(pairs * expertDFF)
	m.upAll = scratchBF16(pairs * expertDFF)
	m.gatedAll = scratchBF16(pairs * expertDFF)
	m.downAll = scratchBF16(pairs * dModel)
	m.expertAcc = scratchBF16(k * dModel)
	p2t := make([]int32, pairs)
	iota := make([]int32, pairs)
	for i := range pairs {
		p2t[i] = int32(i / topK)
		iota[i] = int32(i)
	}
	m.pairToToken = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&p2t[0]), uint(pairs*4), metal.MTLResourceStorageModeShared)
	m.pairIota = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&iota[0]), uint(pairs*4), metal.MTLResourceStorageModeShared)
	if m.routerNorm == nil || m.idx == nil || m.pairToToken == nil || m.pairIota == nil {
		return core.NewError("native.moeBatchScratch: buffer allocation failed")
	}
	m.rowCap, m.dModelCap, m.dFFCap, m.expertDFFCap, m.topKCap, m.numExpertsCap = k, dModel, dFF, expertDFF, topK, numExperts
	return nil
}

// moeQuantAt returns the layer's quant MoE weights, nil-safe on the slice bound.
func moeQuantAt(moeQuant []*MoEQuantLayerWeights, li int) *MoEQuantLayerWeights {
	if li < 0 || li >= len(moeQuant) {
		return nil
	}
	return moeQuant[li]
}

// batchedMoEUsable reports whether the batched MoE block can encode this layer's weights —
// checked in the batch admission pass so the block itself never declines mid-chunk.
func (s *archDecodeState) batchedMoEUsable(w *MoEQuantLayerWeights) bool {
	if w == nil || !gpuHasGeluKernel() {
		return false
	}
	if !quantMoEDeviceRouterBuffersUsable(*w, s.dModel) || !routerTopKUsable(w.NumExperts, w.TopK) {
		return false
	}
	if s.dModel > rmsLoopedLimit { // the fused combine tail is single-row-rms shaped
		return false
	}
	if _, err := moeWeightedSumPipeline(); err != nil {
		return false
	}
	if _, err := moeCombineNormsPipeline(); err != nil {
		return false
	}
	return true
}

// encMoEBlockQuantBatched encodes the whole MoE block for K token rows: hSlab (K × dModel,
// the attention-half outputs) in, the combined residual rows out. outContig writes one
// contiguous K-row slab at outRows[0]+rowOff[0]; otherwise each row lands at its own
// buffer+offset. The caller pre-validated admission via batchedMoEUsable.
func (s *archDecodeState) encMoEBlockQuantBatched(enc metal.MTLComputeCommandEncoderObject, w MoEQuantLayerWeights, hSlab metal.MTLBuffer, outRows []metal.MTLBuffer, rowOff []uint, outContig bool, K int) error {
	dModel, dFF := s.dModel, s.dFF
	numExperts, topK, expertDFF := w.NumExperts, w.TopK, w.ExpertDFF
	mb := s.denseBatch.moeBatch
	if mb == nil {
		mb = &moeBatchScratch{}
		s.denseBatch.moeBatch = mb
	}
	if err := mb.ensure(K, dModel, dFF, expertDFF, topK, numExperts); err != nil {
		return err
	}

	// weight views — the same resolution the decode block performs.
	localGatePacked, localGateScales, localGateBiases, localGateGS, localGateBits, err := quantWeightViewsForShape("native.encMoEBlockQuantBatched: local gate", w.LocalGate, dFF, dModel, w.LocalGroupSize, w.LocalBits)
	if err != nil {
		return err
	}
	localUpPacked, localUpScales, localUpBiases, localUpGS, localUpBits, err := quantWeightViewsForShape("native.encMoEBlockQuantBatched: local up", w.LocalUp, dFF, dModel, w.LocalGroupSize, w.LocalBits)
	if err != nil {
		return err
	}
	localDownPacked, localDownScales, localDownBiases, localDownGS, localDownBits, err := quantWeightViewsForShape("native.encMoEBlockQuantBatched: local down", w.LocalDown, dModel, dFF, w.LocalGroupSize, w.LocalBits)
	if err != nil {
		return err
	}
	fusedExperts := len(w.ExpGateUp.Packed) > 0
	var expGatePacked, expGateScales, expGateBiases bufView
	var expUpPacked, expUpScales, expUpBiases bufView
	var expGateUpPacked, expGateUpScales, expGateUpBiases bufView
	var inGS, inBits, inRows int
	if fusedExperts {
		expGateUpPacked, expGateUpScales, expGateUpBiases, inGS, inBits, err = quantWeightViewsForShape("native.encMoEBlockQuantBatched: expert gate_up", w.ExpGateUp, numExperts*2*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return err
		}
		inRows = 2 * expertDFF
	} else {
		var gGS, gBits, uGS, uBits int
		expGatePacked, expGateScales, expGateBiases, gGS, gBits, err = quantWeightViewsForShape("native.encMoEBlockQuantBatched: expert gate", w.ExpGate, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return err
		}
		expUpPacked, expUpScales, expUpBiases, uGS, uBits, err = quantWeightViewsForShape("native.encMoEBlockQuantBatched: expert up", w.ExpUp, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return err
		}
		if gBits != uBits || gGS != uGS {
			return core.NewError("native.encMoEBlockQuantBatched: expert gate/up geometry mismatch")
		}
		inGS, inBits, inRows = gGS, gBits, expertDFF
	}
	expDownPacked, expDownScales, expDownBiases, expDownGS, expDownBits, err := quantWeightViewsForShape("native.encMoEBlockQuantBatched: expert down", w.ExpDown, numExperts*dModel, expertDFF, w.ExpertGroupSize, w.ExpertBits)
	if err != nil {
		return err
	}
	gatherInPSO, err := gatherQMVBF16SteelPipeline(expertDFF, dModel, inGS, inBits)
	if err != nil {
		return err
	}
	gatherDownPSO, err := gatherQMVBF16SteelPipeline(dModel, expertDFF, expDownGS, expDownBits)
	if err != nil {
		return err
	}
	pairs := K * topK
	inKey := gatherQMVAllRoutesMetaKey{numExperts: numExperts, outDim: expertDFF, inDim: dModel, groupSize: inGS, bits: inBits, expertRows: inRows, routes: pairs, xRows: K, batchedX: true}
	inMeta, err := gatherQMVAllRoutesMetadata(numExperts, expertDFF, dModel, inGS, inBits, inRows, pairs, K, true)
	if err != nil {
		return err
	}
	downKey := gatherQMVAllRoutesMetaKey{numExperts: numExperts, outDim: dModel, inDim: expertDFF, groupSize: expDownGS, bits: expDownBits, expertRows: dModel, routes: pairs, xRows: pairs, batchedX: true}
	downMeta, err := gatherQMVAllRoutesMetadata(numExperts, dModel, expertDFF, expDownGS, expDownBits, dModel, pairs, pairs, true)
	if err != nil {
		return err
	}
	wsumPSO, err := moeWeightedSumPipeline()
	if err != nil {
		return err
	}
	combinePSO, err := moeCombineNormsPipeline()
	if err != nil {
		return err
	}
	combineTG := uint(rmsSimdSize * ((((dModel + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
	if combineTG > combinePSO.MaxTotalThreadsPerThreadgroup() {
		return core.NewError("native.encMoEBlockQuantBatched: combine threadgroup exceeds pipeline max")
	}
	topkPSO, err := routerTopKPipelineK(topK)
	if err != nil {
		return err
	}
	routerPacked, routerScales, routerBiases, routerGS, routerBits, err := quantWeightViewsForShape("native.encMoEBlockQuantBatched: router", w.Router, numExperts, dModel, w.RouterGroupSize, w.RouterBits)
	if err != nil {
		return err
	}
	routerNormView := bf16WeightView(w.RouterNormWScaled, w.routerNormView)
	pre1 := bf16WeightView(w.PreFFNormW, w.preFFNormView)
	pre2 := bf16WeightView(w.PreFFNorm2W, w.preFFNorm2View)
	post1 := bf16WeightView(w.PostFFNorm1W, w.postFFNorm1View)
	post2 := bf16WeightView(w.PostFFNorm2W, w.postFFNorm2View)
	post := bf16WeightView(w.PostFFNormW, w.postFFNormView)
	var scaleBuf metal.MTLBuffer
	var scaleOff uint
	scaleFlag := int32(0)
	if w.PerExpertScale != nil {
		sv := bf16WeightView(w.PerExpertScale, w.perExpertScaleView)
		scaleBuf, scaleOff, scaleFlag = sv.buf, sv.off, 1
	} else {
		scaleBuf = mb.scores
	}

	sink := encSink{enc}

	// router: rms rows → scores qmm → row-dimensioned topk (idx/weight slabs).
	if err := encRMSNormRowsBF16Object(enc, hSlab, routerNormView.buf, mb.routerNorm, 0, routerNormView.off, 0, K, dModel, s.eps); err != nil {
		return err
	}
	if err := encQMMTBF16At(enc, routerPacked.buf, routerScales.buf, routerBiases.buf, mb.routerNorm, mb.scores, routerPacked.off, routerScales.off, routerBiases.off, 0, 0, K, numExperts, dModel, routerGS, routerBits); err != nil {
		return err
	}
	sink.setPSO(topkPSO)
	sink.setBuf(mb.scores, 0, 0)
	sink.setBuf(scaleBuf, scaleOff, 1)
	sink.setBuf(mb.idx, 0, 2)
	sink.setBuf(mb.weights, 0, 3)
	sink.setI32(int32(numExperts), 4)
	sink.setI32(int32(topK), 5)
	sink.setI32(scaleFlag, 6)
	sink.dispatchThreads(
		metal.MTLSize{Width: uint(32 * K), Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)

	// local MLP: rms rows → gate/up qmm → fused gelu → down qmm.
	if err := encRMSNormRowsBF16Object(enc, hSlab, pre1.buf, mb.localIn, 0, pre1.off, 0, K, dModel, s.eps); err != nil {
		return err
	}
	if err := encQMMTBF16At(enc, localGatePacked.buf, localGateScales.buf, localGateBiases.buf, mb.localIn, mb.localGate, localGatePacked.off, localGateScales.off, localGateBiases.off, 0, 0, K, dFF, dModel, localGateGS, localGateBits); err != nil {
		return err
	}
	if err := encQMMTBF16At(enc, localUpPacked.buf, localUpScales.buf, localUpBiases.buf, mb.localIn, mb.localUp, localUpPacked.off, localUpScales.off, localUpBiases.off, 0, 0, K, dFF, dModel, localUpGS, localUpBits); err != nil {
		return err
	}
	if err := encGeluGateMulFused(enc, mb.localGate, mb.localUp, mb.localGated, K*dFF); err != nil {
		return err
	}
	if err := encQMMTBF16At(enc, localDownPacked.buf, localDownScales.buf, localDownBiases.buf, mb.localGated, mb.localOut, localDownPacked.off, localDownScales.off, localDownBiases.off, 0, 0, K, dModel, dFF, localDownGS, localDownBits); err != nil {
		return err
	}

	// experts, all pairs: rms rows, then the pair projections. The grouped-GEMM lane
	// (moe_grouped.go) sorts the pairs by expert and sweeps each expert's weights once per
	// 64-row block; the all-routes gathers (one GEMV per pair, weights re-read per pair)
	// remain the fallback for missing kernels / LTHN_MOE_GEMM=0.
	if err := encRMSNormRowsBF16Object(enc, hSlab, pre2.buf, mb.expertIn, 0, pre2.off, 0, K, dModel, s.eps); err != nil {
		return err
	}
	if moeGroupedUsable(w, inBits, inGS) {
		if err := s.encMoEExpertsGrouped(enc, mb, fusedExperts,
			expGateUpPacked, expGateUpScales, expGateUpBiases,
			expGatePacked, expGateScales, expGateBiases,
			expUpPacked, expUpScales, expUpBiases,
			expDownPacked, expDownScales, expDownBiases,
			inGS, inBits, expDownGS, expDownBits,
			numExperts, topK, expertDFF, dModel, pairs); err != nil {
			return err
		}
	} else {
		if fusedExperts {
			emitGatherQMVAllRoutes(sink, gatherInPSO, inMeta, inKey, mb.expertIn, 0, expGateUpPacked.buf, expGateUpPacked.off, expGateUpScales.buf, expGateUpScales.off, expGateUpBiases.buf, expGateUpBiases.off, mb.pairToToken, mb.idx, 0, mb.gateAll, 0, expertDFF, dModel, inGS, inBits, 0, pairs)
			emitGatherQMVAllRoutes(sink, gatherInPSO, inMeta, inKey, mb.expertIn, 0, expGateUpPacked.buf, expGateUpPacked.off, expGateUpScales.buf, expGateUpScales.off, expGateUpBiases.buf, expGateUpBiases.off, mb.pairToToken, mb.idx, 0, mb.upAll, 0, expertDFF, dModel, inGS, inBits, expertDFF, pairs)
		} else {
			emitGatherQMVAllRoutes(sink, gatherInPSO, inMeta, inKey, mb.expertIn, 0, expGatePacked.buf, expGatePacked.off, expGateScales.buf, expGateScales.off, expGateBiases.buf, expGateBiases.off, mb.pairToToken, mb.idx, 0, mb.gateAll, 0, expertDFF, dModel, inGS, inBits, 0, pairs)
			emitGatherQMVAllRoutes(sink, gatherInPSO, inMeta, inKey, mb.expertIn, 0, expUpPacked.buf, expUpPacked.off, expUpScales.buf, expUpScales.off, expUpBiases.buf, expUpBiases.off, mb.pairToToken, mb.idx, 0, mb.upAll, 0, expertDFF, dModel, inGS, inBits, 0, pairs)
		}
		if err := encGeluGateMulFused(enc, mb.gateAll, mb.upAll, mb.gatedAll, pairs*expertDFF); err != nil {
			return err
		}
		emitGatherQMVAllRoutes(sink, gatherDownPSO, downMeta, downKey, mb.gatedAll, 0, expDownPacked.buf, expDownPacked.off, expDownScales.buf, expDownScales.off, expDownBiases.buf, expDownBiases.off, mb.pairIota, mb.idx, 0, mb.downAll, 0, dModel, expertDFF, expDownGS, expDownBits, 0, pairs)
	}

	// route combine: acc[t] = Σ_r w[t,r] · down[t,r] — one row-dimensioned dispatch.
	sink.setPSO(wsumPSO)
	sink.setBuf(mb.downAll, 0, 0)
	sink.setBuf(mb.weights, 0, 1)
	sink.setBuf(mb.expertAcc, 0, 2)
	sink.setI32(int32(dModel), 3)
	sink.setI32(int32(topK), 4)
	group := min(uint(dModel), uint(256))
	sink.dispatchThreads(
		metal.MTLSize{Width: uint(dModel), Height: uint(K), Depth: 1},
		metal.MTLSize{Width: group, Height: 1, Depth: 1},
	)

	// norm/combine tail: out = h + rms(rms(local)·w1 + rms(expert)·w2)·w3, per row.
	emitCombine := func(xLOff, xEOff, hOff uint, outBuf metal.MTLBuffer, outOff uint, rows int) {
		sink.setPSO(combinePSO)
		sink.setBuf(mb.localOut, xLOff, 0)
		sink.setBuf(post1.buf, post1.off, 1)
		sink.setBuf(mb.expertAcc, xEOff, 2)
		sink.setBuf(post2.buf, post2.off, 3)
		sink.setBuf(post.buf, post.off, 4)
		sink.setBuf(hSlab, hOff, 5)
		sink.setBuf(outBuf, outOff, 6)
		sink.setF32(s.eps, 7)
		sink.setI32(int32(dModel), 8)
		sink.dispatchThreads(
			metal.MTLSize{Width: combineTG * uint(rows), Height: 1, Depth: 1},
			metal.MTLSize{Width: combineTG, Height: 1, Depth: 1},
		)
	}
	rowBytes := uint(dModel * bf16Size)
	if outContig {
		emitCombine(0, 0, 0, outRows[0], rowOff[0], K)
	} else {
		for i := range K {
			emitCombine(uint(i)*rowBytes, uint(i)*rowBytes, uint(i)*rowBytes, outRows[i], rowOff[i], 1)
		}
	}
	return nil
}
