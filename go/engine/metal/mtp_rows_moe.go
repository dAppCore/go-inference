// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"sync/atomic"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// mtp_rows_moe.go — byte-exact multi-row batching for the MoE expert projections (#53). The
// greedy-exact MTP verify lane steps K drafted rows sequentially (verifyAssistantDraftHiddens's
// per-row fallback in assistant_load.go): each row pays its OWN full MoE block — router, local
// dense MLP, and every routed expert's gate/up/down weight read fresh. On a MoE target that is K
// separate weight sweeps per layer where one would do, because the sequential per-row driver
// (stepTokenEncode, decode_forward_arch.go) processes ALL layers for one row before starting the
// next — there is no seam in that path to defer or batch a later stage across rows.
//
// mtpRowsMoEBatched is the reusable, byte-identical alternative to K per-row block calls, GIVEN
// the K rows' post-attention hidden states already computed (hSlab): it groups the K·topK routed
// (row, slot) pairs BY EXPERT and runs the row-batched qmv_rows dot-body (qmv_rows.go — proven
// row-for-row identical to the per-row decode qmv) ONCE per touched expert, instead of once per
// pair. The router stays a per-row call to the SAME single-row device primitive the sequential
// path itself calls (moeRouterQuantDeviceTopKNoCopyWithBufferInPool, router.go) — never batched,
// per the hard #53 contract: batching the gate's own reduction is out of scope even where it might
// incidentally be safe, because the risk of a batching-order regression in the selection stage is
// not worth the (small) router cost.
//
// LIVE WIRING: mtp_rows_driver.go consults mtpRowsMoEArmed from the byte-exact greedy verify
// lane (verifyAssistantDraftHiddens, assistant_load.go) via a layer-major driver that interleaves
// K rows' attention (sequential — each row's SDPA depends on the K/V the PRECEDING rows in the
// SAME block just wrote) with this batched MoE step BETWEEN layers — reusing the SAME single-row
// attention kernels stepTokenEncode itself calls (encAttnHalfKV's offset-capable twin for the
// plain linear cache, encAttnHalfKVPaged for the device-paged cache, which is the LIVE DEFAULT
// for a quant MoE session — see mtp_rows_driver.go's header). moe_batch.go's own K-row
// prompt-prefill MoE fold (encMoEBlockQuantBatched) is a DIFFERENT, NOT byte-identical mechanism
// (an all-pairs gather, not grouped-by-expert, used only by the sampled verify tier) and stays
// untouched.
//
// Both expert gate/up layouts are supported (mtpRowsMoEEligible/mtpRowsMoEBatchedInPool): split
// (ExpGate/ExpUp) and fused (ExpGateUp). Fused is not an edge case — gemma4 declares
// Arch.FuseExpertGateUp, so loadedToQuant ALWAYS synthesises ExpGateUp and drops the split halves
// for every checkpoint (TestLoadGemma4QuantMoEFusedGateUpMatchesSplitExperts); a lane that only
// served split geometry would never engage on any real loaded gemma4 26B-A4B session.

// mtpRowsMoEArmed arms the grouped-by-expert MoE lane — default ON (the byte-identity receipts
// make it the exact lane's implementation, not an experiment; MTP's pair is already opt-in via
// -draft, so there is no second opt-in to stack). LTHN_MTP_ROWS_MOE=0 is the escape hatch for
// A/B against the row-major path and for rollback. Read once at package init like its siblings;
// tests save/restore the var directly (no env re-read needed mid-run).
var mtpRowsMoEArmed = os.Getenv("LTHN_MTP_ROWS_MOE") != "0"

// mtpRowsMoEMaxGroupSize is the largest single-expert pair-group the most recent
// mtpRowsMoEBatched call processed — the engagement counter (mirrors routerFusedDispatches):
// a compare where every group is size 1 never exercises the multi-row tiled kernel and proves
// nothing about the grouping. Diagnostic only.
var mtpRowsMoEMaxGroupSize atomic.Int64

// mtpRowsMoEGroupHist1..mtpRowsMoEGroupHist4Plus tally ONE verify round's expert-group sizes for
// the #53 diagnostic (mtp_rows_driver.go's "mtp-diag rows-moe" per-round trace line): how many of
// the touched experts this MoE batch grouped got exactly 1, 2, 3, or 4-or-more (row, slot) pairs.
// A histogram dominated by size-1 groups means real routing scattered the K·topK pairs across
// mostly distinct experts — the batched gather/scatter pays its overhead without the matching
// per-expert weight-read saving that made the unit fixtures' tiny-expert-count grouping a clean
// win (hypothesis B in #53's write-up: mtp_rows_driver.go's header). NOT reset per layer —
// mtp_rows_driver.go's stepRowsMoEBatched resets them ONCE at the start of a verify round
// (mtpRowsMoEGroupHistReset), before its per-layer loop, so by the round's last layer they hold
// the SUM across every layer processed so far; verifyRowsMoEBatchedHiddens reads them into that
// round's diagnostic snapshot the moment the round completes. Package-level like
// mtpRowsMoEMaxGroupSize: a decode session drives one goroutine (mtp.go's own contract), so
// there is no concurrent-round hazard.
var (
	mtpRowsMoEGroupHist1     atomic.Int64
	mtpRowsMoEGroupHist2     atomic.Int64
	mtpRowsMoEGroupHist3     atomic.Int64
	mtpRowsMoEGroupHist4Plus atomic.Int64
)

// mtpRowsMoEGroupHistReset zeroes the round histogram — called once per verify round
// (mtp_rows_driver.go's stepRowsMoEBatched, before the per-layer loop starts) so sums from a
// PRIOR round never leak into the next one.
func mtpRowsMoEGroupHistReset() {
	mtpRowsMoEGroupHist1.Store(0)
	mtpRowsMoEGroupHist2.Store(0)
	mtpRowsMoEGroupHist3.Store(0)
	mtpRowsMoEGroupHist4Plus.Store(0)
}

// mtpRowsMoEGroupHistBump tallies one touched expert group of size m (row,slot) pairs into the
// round histogram — called once per group from mtpRowsMoEBatchedInPool's grouping loop, beside
// its existing maxGroup tracking.
func mtpRowsMoEGroupHistBump(m int) {
	switch {
	case m <= 1:
		mtpRowsMoEGroupHist1.Add(1)
	case m == 2:
		mtpRowsMoEGroupHist2.Add(1)
	case m == 3:
		mtpRowsMoEGroupHist3.Add(1)
	default:
		mtpRowsMoEGroupHist4Plus.Add(1)
	}
}

// mtpRowsMoEEligible reports whether mtpRowsMoEBatched can serve this MoE layer's geometry at
// all — mirrors batchedMoEUsable's discriminators (moe_batch.go): gpt_oss (ClampedSwiGLU) and
// qwen (a bound SharedGate) decode on entirely different host paths and never reach here. Both
// expert gate/up layouts are supported: split (ExpGate/ExpUp) and fused (ExpGateUp) — the LATTER
// is not an edge case: gemma4 declares Arch.FuseExpertGateUp, so loadedToQuant (load_shared.go's
// moeToQuant) ALWAYS synthesises ExpGateUp and drops the split halves, for every checkpoint,
// split-shipped or not (TestLoadGemma4QuantMoEFusedGateUpMatchesSplitExperts pins this) — a lane
// that declined fused ExpGateUp would never engage on any real loaded gemma4 26B-A4B session, only
// on a hand-built split fixture. false always means "the caller keeps the per-row path" — never a
// wrong answer forced through.
// qmvByteExactServable is encQMVByteExactAt's route decision WITHOUT the encode — the pure
// probe eligibility uses so the driver can never start a verify it cannot finish. It MUST
// mirror encQMVByteExactAt exactly (same helpers, same order); the driver's in-flight tripwire
// error fires if they ever drift. Real-26B lesson: expert dFF not %512 had NO
// multi-row byte-exact route at any rows>=2 (tiled and chunked both required outDim%8 &&
// inDim%512), and the coarse eligibility missed it — the pair hard-errored mid-verify on the
// real checkpoint while every synthetic fixture chose aligned dims. The general tiled tier
// (lthn_qmv_rows_general — qmv_impl's M-variant, qmvRowsTiledKeyFor) closes exactly that hole:
// unaligned dims now serve rows 2..4 tiled and larger groups by chunked composition.
func qmvByteExactServable(rows, outDim, inDim, gs, bits int) bool {
	if rows <= 0 {
		return false
	}
	if rows == 1 {
		_, err := pipelineFor(qmvBF16KernelName(outDim, inDim, gs, bits))
		return err == nil
	}
	if rows <= qmvRowsTiledCap() {
		if plan, ok := qmvRowsPlanFor(rows, outDim, inDim, gs, bits); ok && plan.tiled {
			_, ok = lthnQMVRowsPipeline(plan.tiledKey)
			return ok
		}
		// No tiled plan at these rows (e.g. unaligned dims past the general
		// flat cap under an armed wide lane) — fall through: the chunked
		// composition below may still serve byte-exactly.
	}
	if rows <= lthnQMVRowsMaxM || rows > qmvRowsMax || !qmvChunksEnabled() {
		return false
	}
	for _, m := range qmvRowsChunks(rows) {
		key, ok := qmvRowsTiledKeyFor(m, outDim, inDim, gs, bits)
		if !ok {
			return false
		}
		if _, ok := lthnQMVRowsPipeline(key); !ok {
			return false
		}
	}
	return true
}

func mtpRowsMoEEligible(w MoEQuantLayerWeights, dModel, dFF, maxRows int) bool {
	if maxRows < 1 {
		return false
	}
	if w.ClampedSwiGLU || len(w.SharedGate.Packed) > 0 {
		return false
	}
	if w.NumExperts <= 0 || w.TopK <= 0 || w.ExpertDFF <= 0 || dModel <= 0 || dFF <= 0 {
		return false
	}
	size := dModel * bf16Size
	if len(w.PreFFNormW) != size || len(w.PreFFNorm2W) != size ||
		len(w.PostFFNorm1W) != size || len(w.PostFFNorm2W) != size || len(w.PostFFNormW) != size {
		return false
	}
	// the fused combine kernel is single-row-rms shaped (moeCombineNormsPipeline's own contract).
	if dModel > rmsLoopedLimit {
		return false
	}
	if !gpuHasGeluKernel() {
		return false
	}
	if !routerTopKUsable(w.NumExperts, w.TopK) {
		return false
	}
	if _, err := moeWeightedSumPipeline(); err != nil {
		return false
	}
	if _, err := moeCombineNormsPipeline(); err != nil {
		return false
	}
	if _, err := pipelineFor(rmsKernelBF16(dModel)); err != nil {
		return false
	}
	if _, _, _, _, _, err := quantWeightViewsForShape("native.mtpRowsMoEEligible: local gate", w.LocalGate, dFF, dModel, w.LocalGroupSize, w.LocalBits); err != nil {
		return false
	}
	if _, _, _, _, _, err := quantWeightViewsForShape("native.mtpRowsMoEEligible: local up", w.LocalUp, dFF, dModel, w.LocalGroupSize, w.LocalBits); err != nil {
		return false
	}
	if _, _, _, _, _, err := quantWeightViewsForShape("native.mtpRowsMoEEligible: local down", w.LocalDown, dModel, dFF, w.LocalGroupSize, w.LocalBits); err != nil {
		return false
	}
	if len(w.ExpGateUp.Packed) > 0 {
		// fused: [numExperts × 2·expertDFF, dModel] — expert e's gate rows at [e·2·expertDFF,
		// e·2·expertDFF+expertDFF), up rows immediately after (fuseExpertGateUpQuant's own layout,
		// load_shared.go — "gate's packed/scales/biases ahead of up's").
		if _, _, _, _, _, err := quantWeightViewsForShape("native.mtpRowsMoEEligible: expert gate_up", w.ExpGateUp, w.NumExperts*2*w.ExpertDFF, dModel, w.ExpertGroupSize, w.ExpertBits); err != nil {
			return false
		}
	} else {
		if _, _, _, _, _, err := quantWeightViewsForShape("native.mtpRowsMoEEligible: expert gate", w.ExpGate, w.NumExperts*w.ExpertDFF, dModel, w.ExpertGroupSize, w.ExpertBits); err != nil {
			return false
		}
		if _, _, _, _, _, err := quantWeightViewsForShape("native.mtpRowsMoEEligible: expert up", w.ExpUp, w.NumExperts*w.ExpertDFF, dModel, w.ExpertGroupSize, w.ExpertBits); err != nil {
			return false
		}
	}
	if _, _, _, _, _, err := quantWeightViewsForShape("native.mtpRowsMoEEligible: expert down", w.ExpDown, w.NumExperts*dModel, w.ExpertDFF, w.ExpertGroupSize, w.ExpertBits); err != nil {
		return false
	}
	// Every dispatch shape the batched path pushes through encQMVRowsGroupAt must have a
	// byte-exact route at EVERY row count a K-row verify can produce (expert groups run 1..K,
	// group chunking caps at qmvRowsMax): expert gate/up (expertDFF x dModel) + expert down
	// (dModel x expertDFF) at the expert quant, local gate/up (dFF x dModel) + local down
	// (dModel x dFF) at the local quant. Probing up front is what makes the driver's
	// mid-verify decline impossible.
	for r := 1; r <= min(maxRows, qmvRowsMax); r++ {
		if !qmvByteExactServable(r, w.ExpertDFF, dModel, w.ExpertGroupSize, w.ExpertBits) ||
			!qmvByteExactServable(r, dModel, w.ExpertDFF, w.ExpertGroupSize, w.ExpertBits) ||
			!qmvByteExactServable(r, dFF, dModel, w.LocalGroupSize, w.LocalBits) ||
			!qmvByteExactServable(r, dModel, dFF, w.LocalGroupSize, w.LocalBits) {
			return false
		}
	}
	return true
}

// mtpRowsMoEBatched computes K rows' MoE block output — router, local dense MLP + routed
// experts, weighted-sum, norm/combine, residual — byte-identical to calling the per-row block
// (MoEBlockQuantInto) once per row, EXCEPT the expert gate/up/down projections read each touched
// expert's weight ONCE across every (row, slot) pair that selected it, instead of once per pair:
// rows are grouped by their routed expert id and each group runs through encQMVRowsGroupAt (the
// qmv_rows tiled/chunked byte-tier — qmv_fast_impl's M-variant, proven row-for-row identical to
// the per-row qmv), scattered back to (row, slot) order for the combine tail.
//
// hSlab is K contiguous rows of dModel bf16 bytes (the layer's post-attention hidden — the same
// input the per-row block's h/hBuf takes); the returned outSlab is K contiguous rows of the same
// shape (the residual-added layer output). ok=false means the geometry declined — the caller
// falls back to the per-row path; this never returns a wrong-but-silent answer. LTHN_MTP_ROWS_MOE
// is NOT consulted here — callers gate on mtpRowsMoEArmed themselves (see the file header for
// why no call site does that yet).
func mtpRowsMoEBatched(hSlab []byte, w MoEQuantLayerWeights, dModel, dFF, K int, eps float32) (outSlab []byte, ok bool, err error) {
	if err := ensureInit(); err != nil {
		return nil, false, err
	}
	if K < 1 {
		return nil, false, nil
	}
	rowBytes := dModel * bf16Size
	if len(hSlab) != K*rowBytes {
		return nil, false, core.NewError("native.mtpRowsMoEBatched: hSlab must be K*dModel bf16 bytes")
	}
	if !mtpRowsMoEEligible(w, dModel, dFF, K) {
		return nil, false, nil
	}
	withAutoreleasePool(func() {
		outSlab, ok, err = mtpRowsMoEBatchedInPool(hSlab, w, dModel, dFF, K, eps)
	})
	return outSlab, ok, err
}

func mtpRowsMoEBatchedInPool(hSlab []byte, w MoEQuantLayerWeights, dModel, dFF, K int, eps float32) ([]byte, bool, error) {
	numExperts, topK, expertDFF := w.NumExperts, w.TopK, w.ExpertDFF
	rowBytes := dModel * bf16Size

	// 1. ROUTER — K independent instances of the sequential path's own single-row
	// dispatch sequence (encMoERouterQuantTopKRow, router.go). Never batched
	// (#53): the gate's reduction stays the per-row shape unconditionally — the
	// K rows share ONE command buffer purely as a submission seam (the per-row
	// private commit+wait cost K×layers GPU round-trips per verify round, the
	// dominant moe wall once the expert stage batched).
	idxAll := make([]int32, K*topK)
	weightsAll := make([]byte, K*topK*bf16Size)
	{
		if !routerTopKUsable(numExperts, topK) {
			return nil, false, nil
		}
		rGroupSize, rBits := quantWeightGeometryForShape(w.Router, numExperts, dModel, w.RouterGroupSize, w.RouterBits)
		if rGroupSize <= 0 || dModel%rGroupSize != 0 {
			return nil, false, core.NewError("native.mtpRowsMoEBatched: router groupSize must divide dModel")
		}
		rmsPSO, rerr := pipelineFor(rmsKernelBF16(dModel))
		if rerr != nil {
			return nil, false, rerr
		}
		rmsTG := rmsThreadgroup(dModel, rmsPSO)
		qmvPSO, rerr := pipelineFor(qmvBF16KernelName(numExperts, dModel, rGroupSize, rBits))
		if rerr != nil {
			return nil, false, rerr
		}
		routerPSO, rerr := routerTopKPipelineK(topK)
		if rerr != nil {
			return nil, false, rerr
		}
		normBuf := bf16WeightView(w.RouterNormWScaled, w.routerNormView)
		wBuf, scalesBuf, biasesBuf := quantWeightViews(w.Router)
		var scaleBuf metal.MTLBuffer
		var scaleOff uint
		if w.PerExpertScale != nil {
			scaleView := bf16WeightView(w.PerExpertScale, w.perExpertScaleView)
			scaleBuf, scaleOff = scaleView.buf, scaleView.off
		}
		scratches := make([]*routerDeviceScratch, K)
		defer func() {
			for _, sc := range scratches {
				if sc != nil {
					putRouterDeviceScratch(sc)
				}
			}
		}()
		rcb := commandBufferFast(queue)
		renc := computeCommandEncoderFast(rcb)
		for r := range K {
			scratch, serr := getRouterDeviceScratch(dModel, numExperts, topK)
			if serr != nil {
				endEncodingFast(renc)
				return nil, false, serr
			}
			scratches[r] = scratch
			row := hSlab[r*rowBytes : (r+1)*rowBytes]
			inputBuf, iok := scratch.inputView(row)
			if !iok {
				var cerr error
				if inputBuf, cerr = scratch.x.copyPrefixBuffer(row); cerr != nil {
					endEncodingFast(renc)
					return nil, false, cerr
				}
			}
			encMoERouterQuantTopKRow(renc, scratch, inputBuf, normBuf, wBuf, scalesBuf, biasesBuf, scaleBuf, scaleOff, w.PerExpertScale != nil, rmsPSO, qmvPSO, routerPSO, rmsTG, dModel, numExperts, topK, eps)
		}
		endEncodingFast(renc)
		commitCommandBufferFast(rcb)
		waitUntilCompletedFast(rcb)
		for r, scratch := range scratches {
			copy(idxAll[r*topK:], unsafe.Slice(scratch.idxPtr, topK))
			copy(weightsAll[r*topK*bf16Size:], unsafe.Slice(scratch.weightPtr, topK*bf16Size))
		}
	}

	// 2. LOCAL MLP + the expert branch's pre-norm — batched across all K rows (shared weight per
	// projection, the established qmv_rows dense-verify pattern; #53 only forbids batching the
	// router). One command buffer for the whole stage.
	localGateView, localGateScales, localGateBiases, localGateGS, localGateBits, err := quantWeightViewsForShape("native.mtpRowsMoEBatched: local gate", w.LocalGate, dFF, dModel, w.LocalGroupSize, w.LocalBits)
	if err != nil {
		return nil, false, err
	}
	localUpView, localUpScales, localUpBiases, localUpGS, localUpBits, err := quantWeightViewsForShape("native.mtpRowsMoEBatched: local up", w.LocalUp, dFF, dModel, w.LocalGroupSize, w.LocalBits)
	if err != nil {
		return nil, false, err
	}
	localDownView, localDownScales, localDownBiases, localDownGS, localDownBits, err := quantWeightViewsForShape("native.mtpRowsMoEBatched: local down", w.LocalDown, dModel, dFF, w.LocalGroupSize, w.LocalBits)
	if err != nil {
		return nil, false, err
	}
	rmsPSO, err := pipelineFor(rmsKernelBF16(dModel))
	if err != nil {
		return nil, false, err
	}
	rmsTG := rmsThreadgroup(dModel, rmsPSO)
	pre1 := bf16WeightView(w.PreFFNormW, w.preFFNormView)
	pre2 := bf16WeightView(w.PreFFNorm2W, w.preFFNorm2View)

	hBuf := sharedBytes(hSlab)
	localNormBuf := scratchBF16(K * dModel)
	expertNormBuf := scratchBF16(K * dModel)
	localGateBuf := scratchBF16(K * dFF)
	localUpBuf := scratchBF16(K * dFF)
	localGatedBuf := scratchBF16(K * dFF)
	localOutBuf := scratchBF16(K * dModel)
	if hBuf == nil || localNormBuf == nil || expertNormBuf == nil || localGateBuf == nil || localUpBuf == nil || localGatedBuf == nil || localOutBuf == nil {
		return nil, false, core.NewError("native.mtpRowsMoEBatched: local-stage scratch allocation failed")
	}

	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	sink := encSink{enc}
	emitRMSNormRows(sink, rmsPSO, hBuf, pre1.buf, localNormBuf, 0, pre1.off, 0, dModel, eps, K, rmsTG)
	emitRMSNormRows(sink, rmsPSO, hBuf, pre2.buf, expertNormBuf, 0, pre2.off, 0, dModel, eps, K, rmsTG)
	if hOK, herr := encQMVRowsGroupAt(enc, localGateView, localGateScales, localGateBiases, localNormBuf, localGateBuf, 0, 0, K, dFF, dModel, localGateGS, localGateBits); herr != nil || !hOK {
		endEncodingFast(enc)
		return nil, false, herr
	}
	if hOK, herr := encQMVRowsGroupAt(enc, localUpView, localUpScales, localUpBiases, localNormBuf, localUpBuf, 0, 0, K, dFF, dModel, localUpGS, localUpBits); herr != nil || !hOK {
		endEncodingFast(enc)
		return nil, false, herr
	}
	if err := encGeluGateMulFused(enc, localGateBuf, localUpBuf, localGatedBuf, K*dFF); err != nil {
		endEncodingFast(enc)
		return nil, false, err
	}
	if hOK, herr := encQMVRowsGroupAt(enc, localDownView, localDownScales, localDownBiases, localGatedBuf, localOutBuf, 0, 0, K, dModel, dFF, localDownGS, localDownBits); herr != nil || !hOK {
		endEncodingFast(enc)
		return nil, false, herr
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)

	expertNormHost := make([]byte, K*rowBytes)
	copy(expertNormHost, unsafe.Slice((*byte)(expertNormBuf.Contents()), K*rowBytes))

	// 3. EXPERT MLP — group (row, slot) pairs by routed expert; each touched expert's gate/up/down
	// weight is read ONCE across every pair sharing it, not once per pair. fusedExperts mirrors
	// encMoEBlockQuantDevice's OWN fused handling (moe_block.go, emitGatherInAll): two separate
	// GEMV dispatches against the SAME ExpGateUp tensor, gate at each expert's block start and up
	// expertDFF rows further in — never a single "gemv-then-split" kernel.
	fusedExperts := len(w.ExpGateUp.Packed) > 0
	var expGateView, expGateScales, expGateBiases bufView
	var expUpView, expUpScales, expUpBiases bufView
	var expGateUpView, expGateUpScales, expGateUpBiases bufView
	var expGateGS, expGateBits, expUpGS, expUpBits, expGateUpGS, expGateUpBits int
	if fusedExperts {
		expGateUpView, expGateUpScales, expGateUpBiases, expGateUpGS, expGateUpBits, err = quantWeightViewsForShape("native.mtpRowsMoEBatched: expert gate_up", w.ExpGateUp, numExperts*2*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return nil, false, err
		}
	} else {
		expGateView, expGateScales, expGateBiases, expGateGS, expGateBits, err = quantWeightViewsForShape("native.mtpRowsMoEBatched: expert gate", w.ExpGate, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return nil, false, err
		}
		expUpView, expUpScales, expUpBiases, expUpGS, expUpBits, err = quantWeightViewsForShape("native.mtpRowsMoEBatched: expert up", w.ExpUp, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return nil, false, err
		}
	}
	expDownView, expDownScales, expDownBiases, expDownGS, expDownBits, err := quantWeightViewsForShape("native.mtpRowsMoEBatched: expert down", w.ExpDown, numExperts*dModel, expertDFF, w.ExpertGroupSize, w.ExpertBits)
	if err != nil {
		return nil, false, err
	}

	pairs := K * topK
	groups := make(map[int32][]int, pairs)
	order := make([]int32, 0, pairs)
	for p := range pairs {
		e := idxAll[p]
		if _, seen := groups[e]; !seen {
			order = append(order, e)
		}
		groups[e] = append(groups[e], p)
	}

	// ONE gather slab + ONE command buffer for every group (#53): the per-group
	// commit+wait paid a full GPU submit round-trip per touched expert — ~550
	// per verify round across the block, launch-bound at ~0.4ms each (the
	// measured moe=221ms wall against ~20ms of arithmetic). Every (row, slot)
	// pair belongs to exactly one group, so the groups pack contiguously into
	// pair-sized slabs; each group's dispatches read/write its own slab span
	// (encQMVRowsGroupAt's byte offsets), the elementwise gelu runs ONCE over
	// the whole slab, and the single wait replaces ~550. Kernels, per-group
	// shapes, and rounding are unchanged — byte-identical rows, one submit.
	maxGroup := 0
	type groupSpan struct {
		e      int32
		off, m int // pair-slot cursor + group size
	}
	spans := make([]groupSpan, 0, len(order))
	gatherAll := make([]byte, pairs*rowBytes)
	cursor := 0
	for _, e := range order {
		group := groups[e]
		m := len(group)
		if m > maxGroup {
			maxGroup = m
		}
		mtpRowsMoEGroupHistBump(m)
		for i, p := range group {
			r := p / topK
			copy(gatherAll[(cursor+i)*rowBytes:(cursor+i+1)*rowBytes], expertNormHost[r*rowBytes:(r+1)*rowBytes])
		}
		spans = append(spans, groupSpan{e: e, off: cursor, m: m})
		cursor += m
	}

	inAllBuf := sharedBytes(gatherAll)
	gateAllBuf := scratchBF16(pairs * expertDFF)
	upAllBuf := scratchBF16(pairs * expertDFF)
	gatedAllBuf := scratchBF16(pairs * expertDFF)
	downSlabBuf := scratchBF16(pairs * dModel)
	if inAllBuf == nil || gateAllBuf == nil || upAllBuf == nil || gatedAllBuf == nil || downSlabBuf == nil {
		return nil, false, core.NewError("native.mtpRowsMoEBatched: expert-group scratch allocation failed")
	}

	gcb := commandBufferFast(queue)
	// CONCURRENT dispatch (#53): the groups' gate/up (and later down) GEMVs are
	// mutually independent — disjoint slab spans — so the encoder must not
	// serialise them on the shared scratch buffers. Two explicit barriers mark
	// the only true edges: gate+up → gelu → down.
	genc := concurrentComputeEncoderFast(gcb)
	for _, g := range spans {
		var gPacked, gScales, gBiases, uPacked, uScales, uBiases bufView
		var gGS, gBits, uGS, uBits int
		if fusedExperts {
			// expert e's fused block is rows [e·2·expertDFF, (e+1)·2·expertDFF) of ExpGateUp: gate
			// is the block's own start (the "2e"th expertDFF-sized slice), up is the very next
			// expertDFF-sized slice ("2e+1"th) — moeExpertQuantOffsets' rowOff=e·rowsPerExpert
			// formula gives exactly that when fed the DOUBLED expert index at expertDFF-row stride.
			gPacked, gScales, gBiases = moeExpertQuantOffsets(expGateUpView, expGateUpScales, expGateUpBiases, int(g.e)*2, expertDFF, dModel, expGateUpGS, expGateUpBits)
			uPacked, uScales, uBiases = moeExpertQuantOffsets(expGateUpView, expGateUpScales, expGateUpBiases, int(g.e)*2+1, expertDFF, dModel, expGateUpGS, expGateUpBits)
			gGS, gBits, uGS, uBits = expGateUpGS, expGateUpBits, expGateUpGS, expGateUpBits
		} else {
			gPacked, gScales, gBiases = moeExpertQuantOffsets(expGateView, expGateScales, expGateBiases, int(g.e), expertDFF, dModel, expGateGS, expGateBits)
			uPacked, uScales, uBiases = moeExpertQuantOffsets(expUpView, expUpScales, expUpBiases, int(g.e), expertDFF, dModel, expUpGS, expUpBits)
			gGS, gBits, uGS, uBits = expGateGS, expGateBits, expUpGS, expUpBits
		}
		inOff := uint(g.off * rowBytes)
		ffOff := uint(g.off * expertDFF * bf16Size)
		if hOK, herr := encQMVRowsGroupAt(genc, gPacked, gScales, gBiases, inAllBuf, gateAllBuf, inOff, ffOff, g.m, expertDFF, dModel, gGS, gBits); herr != nil || !hOK {
			endEncodingFast(genc)
			return nil, false, herr
		}
		if hOK, herr := encQMVRowsGroupAt(genc, uPacked, uScales, uBiases, inAllBuf, upAllBuf, inOff, ffOff, g.m, expertDFF, dModel, uGS, uBits); herr != nil || !hOK {
			endEncodingFast(genc)
			return nil, false, herr
		}
	}
	memoryBarrierObject(genc, metal.MTLBarrierScopeBuffers)
	if err := encGeluGateMulFused(genc, gateAllBuf, upAllBuf, gatedAllBuf, pairs*expertDFF); err != nil {
		endEncodingFast(genc)
		return nil, false, err
	}
	memoryBarrierObject(genc, metal.MTLBarrierScopeBuffers)
	for _, g := range spans {
		dPacked, dScales, dBiases := moeExpertQuantOffsets(expDownView, expDownScales, expDownBiases, int(g.e), dModel, expertDFF, expDownGS, expDownBits)
		ffOff := uint(g.off * expertDFF * bf16Size)
		dnOff := uint(g.off * rowBytes)
		if hOK, herr := encQMVRowsGroupAt(genc, dPacked, dScales, dBiases, gatedAllBuf, downSlabBuf, ffOff, dnOff, g.m, dModel, expertDFF, expDownGS, expDownBits); herr != nil || !hOK {
			endEncodingFast(genc)
			return nil, false, herr
		}
	}
	endEncodingFast(genc)
	commitCommandBufferFast(gcb)
	waitUntilCompletedFast(gcb)

	// scatter the group-contiguous down rows back to PAIR order for the combine
	// (K·topK rows — host-µs, not a dispatch).
	downAllHost := make([]byte, pairs*rowBytes)
	downSlabHost := unsafe.Slice((*byte)(downSlabBuf.Contents()), pairs*rowBytes)
	for _, g := range spans {
		for i, p := range groups[g.e] {
			copy(downAllHost[p*rowBytes:(p+1)*rowBytes], downSlabHost[(g.off+i)*rowBytes:(g.off+i+1)*rowBytes])
		}
	}
	mtpRowsMoEMaxGroupSize.Store(int64(maxGroup))

	// 4. combine — the SAME fused kernels the per-row block uses (moeWeightedSumPipeline,
	// moeCombineNormsPipeline: "byte-identical rounding" per their own doc contract, whether
	// dispatched for one row or K — each threadgroup owns exactly one row, never mixed).
	wsumPSO, err := moeWeightedSumPipeline()
	if err != nil {
		return nil, false, err
	}
	combinePSO, err := moeCombineNormsPipeline()
	if err != nil {
		return nil, false, err
	}
	combineTG := uint(rmsSimdSize * ((((dModel + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
	if combineTG > combinePSO.MaxTotalThreadsPerThreadgroup() {
		return nil, false, nil
	}
	post1 := bf16WeightView(w.PostFFNorm1W, w.postFFNorm1View)
	post2 := bf16WeightView(w.PostFFNorm2W, w.postFFNorm2View)
	post := bf16WeightView(w.PostFFNormW, w.postFFNormView)

	downAllBuf := sharedBytes(downAllHost)
	weightsAllBuf := sharedBytes(weightsAll)
	expertAccBuf := scratchBF16(K * dModel)
	outBuf := scratchBF16(K * dModel)
	if downAllBuf == nil || weightsAllBuf == nil || expertAccBuf == nil || outBuf == nil {
		return nil, false, core.NewError("native.mtpRowsMoEBatched: combine scratch allocation failed")
	}

	ccb := commandBufferFast(queue)
	cenc := computeCommandEncoderFast(ccb)
	csink := encSink{cenc}
	csink.setPSO(wsumPSO)
	csink.setBuf(downAllBuf, 0, 0)
	csink.setBuf(weightsAllBuf, 0, 1)
	csink.setBuf(expertAccBuf, 0, 2)
	csink.setI32(int32(dModel), 3)
	csink.setI32(int32(topK), 4)
	wsumGroup := min(uint(dModel), uint(256))
	csink.dispatchThreads(metal.MTLSize{Width: uint(dModel), Height: uint(K), Depth: 1}, metal.MTLSize{Width: wsumGroup, Height: 1, Depth: 1})

	csink.setPSO(combinePSO)
	csink.setBuf(localOutBuf, 0, 0)
	csink.setBuf(post1.buf, post1.off, 1)
	csink.setBuf(expertAccBuf, 0, 2)
	csink.setBuf(post2.buf, post2.off, 3)
	csink.setBuf(post.buf, post.off, 4)
	csink.setBuf(hBuf, 0, 5)
	csink.setBuf(outBuf, 0, 6)
	csink.setF32(eps, 7)
	csink.setI32(int32(dModel), 8)
	csink.dispatchThreads(metal.MTLSize{Width: combineTG * uint(K), Height: 1, Depth: 1}, metal.MTLSize{Width: combineTG, Height: 1, Depth: 1})

	endEncodingFast(cenc)
	commitCommandBufferFast(ccb)
	waitUntilCompletedFast(ccb)

	out := make([]byte, K*rowBytes)
	copy(out, unsafe.Slice((*byte)(outBuf.Contents()), K*rowBytes))
	return out, true, nil
}

// moeExpertQuantOffsets returns expert e's row-contiguous slice of a packed [numExperts×rowsPerExpert,
// inDim] quant tensor: rowsPerExpert=expertDFF for gate/up (inDim=dModel), rowsPerExpert=dModel for
// down (inDim=expertDFF). packed/scales/biases are the WHOLE tensor's views (quantWeightViewsForShape);
// the returned views add expert e's byte offset on top — the same row-byte-width arithmetic
// quantWeightViewsForShape itself validates (inDim*bits/8 per packed row, (inDim/groupSize)*2 per
// scale/bias row).
func moeExpertQuantOffsets(packed, scales, biases bufView, e, rowsPerExpert, inDim, groupSize, bits int) (bufView, bufView, bufView) {
	packedRowBytes := uint(inDim * bits / 8)
	sbRowBytes := uint((inDim / groupSize) * bf16Size)
	rowOff := uint(e * rowsPerExpert)
	return bufView{buf: packed.buf, off: packed.off + rowOff*packedRowBytes},
		bufView{buf: scales.buf, off: scales.off + rowOff*sbRowBytes},
		bufView{buf: biases.buf, off: biases.off + rowOff*sbRowBytes}
}

// encQMVRowsGroupAt projects `rows` contiguous activation rows (at inOff) through one quant
// weight (wq/scales/biases) into `rows` contiguous output rows (at outOff), splitting into
// ≤qmvRowsMax sub-chunks so an oversized expert group (more pairs than one dispatch admits)
// still lands entirely on the byte-exact route rather than silently falling to a throughput tier.
// ok=false (from any sub-chunk) means the WHOLE group declines — the caller falls back.
func encQMVRowsGroupAt(enc metal.MTLComputeCommandEncoderObject, wq, scales, biases bufView, in, out metal.MTLBuffer, inOff, outOff uint, rows, outDim, inDim, gs, bits int) (bool, error) {
	row := 0
	for row < rows {
		chunk := min(rows-row, qmvRowsMax)
		hOK, herr := encQMVByteExactAt(enc, wq, scales, biases, in, out,
			inOff+uint(row*inDim*bf16Size), outOff+uint(row*outDim*bf16Size), chunk, outDim, inDim, gs, bits)
		if herr != nil || !hOK {
			return false, herr
		}
		row += chunk
	}
	return true, nil
}

// encQMVByteExactAt encodes ONE dispatch, choosing the byte-exact route for `rows`: the plain
// per-row qmv kernel at rows==1 (qmvBF16KernelName — the exact PSO the sequential per-row block
// itself resolves for this projection), the register-tiled lthn_qmv_rows[_general] where
// qmvRowsPlanFor reports a tiled plan (qmv_rows.go — qmvRowsTiledKeyFor matches the fast or
// general twin to the envelope; the gather fallback that function can also report is a
// THROUGHPUT route and is deliberately never taken here), or the chunked byte-tier composition
// otherwise (encQMVRowsBF16ChunkedAt carries the full gate — rows band, kill switch, per-chunk
// keys). ok=false on any geometry qmv_rows itself would not serve byte-exactly.
func encQMVByteExactAt(enc metal.MTLComputeCommandEncoderObject, wq, scales, biases bufView, in, out metal.MTLBuffer, inOff, outOff uint, rows, outDim, inDim, gs, bits int) (bool, error) {
	if rows <= 0 {
		return false, nil
	}
	if rows == 1 {
		pso, err := pipelineFor(qmvBF16KernelName(outDim, inDim, gs, bits))
		if err != nil {
			return false, nil
		}
		emitQMVAt(encSink{enc}, pso, wq.buf, wq.off, scales.buf, scales.off, biases.buf, biases.off, in, inOff, out, outOff, inDim, outDim)
		return true, nil
	}
	if rows <= qmvRowsTiledCap() {
		if plan, ok := qmvRowsPlanFor(rows, outDim, inDim, gs, bits); ok && plan.tiled {
			pso, ok := lthnQMVRowsPipeline(plan.tiledKey)
			if !ok {
				return false, nil
			}
			emitQMVRowsTiled(encSink{enc}, pso, wq.buf, wq.off, scales.buf, scales.off, biases.buf, biases.off, in, inOff, out, outOff, inDim, outDim)
			return true, nil
		}
		// No tiled plan at these rows — fall through to the chunked
		// composition, mirroring qmvByteExactServable exactly.
	}
	return encQMVRowsBF16ChunkedAt(enc, wq.buf, scales.buf, biases.buf, in, out, wq.off, scales.off, biases.off, inOff, outOff, rows, outDim, inDim, gs, bits)
}
