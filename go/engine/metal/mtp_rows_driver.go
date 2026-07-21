// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync/atomic"
	"time"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// mtp_rows_driver.go — the #53 WIRING lane: a layer-major verify driver that finally consults
// mtpRowsMoEArmed (mtp_rows_moe.go), which every prior lane left "recognised but not consulted".
//
// THE PROBLEM (written up in mtp_rows_moe.go's header and commit 7946dacc): the byte-exact greedy
// verify block steps K drafted rows ROW-MAJOR (verifyAssistantDraftHiddens's per-row stepID loop,
// which is stepTokenEncode in decode_forward_arch.go — ALL layers for row 0, then ALL layers for
// row 1, …). stepTokenEncode has no seam to defer or batch a later stage across rows: each row pays
// its own full MoE block, K separate weight sweeps per layer where one would do.
//
// THE FIX: this file drives the SAME K rows LAYER-MAJOR instead — for layer L, all K rows'
// attention runs SEQUENTIALLY in row order (row r's SDPA depends on the K/V rows 0..r-1 THIS block
// just wrote at layer L), then mtpRowsMoEBatched runs layer L's WHOLE MoE block for all K rows at
// once, grouped by routed expert. The reorder is safe because attention and MoE have no cross-row
// dependency once attention's KV writes stay ordered — layer-major is a valid topological
// reordering of the exact same per-row DAG the row-major driver walks, so the maths is identical;
// only the SCHEDULE differs. Per-row arithmetic is untouched: this file reuses the SAME byte-exact
// single-row attention kernels stepTokenEncode's plain per-row lane itself calls — encAttnHalfKV
// (decode_step.go, via its offset-capable encAttnHalfKVInputAt twin) for the plain linear KV
// cache, encAttnHalfKVPaged (paged_kv_device.go) for the device-paged cache — merely addressed at
// row offsets (linear) or K separate 1-row buffers (paged, which has no offset parameters) instead
// of the state's single-row s.xA/s.xB/s.hBuf scratch. mtpRowsMoEBatched is already proven
// byte-identical to K sequential MoEBlockQuantInto calls (mtp_rows_moe_test.go).
//
// PAGED KV IS THE LIVE DEFAULT, NOT AN EDGE CASE: initDevicePagedKVWithPrealloc
// (decode_forward_arch.go/arch_session.go) builds device-paged caches for every cache-owning layer
// of every quant-MoE ArchSession UNCONDITIONALLY (newDevicePagedKVCache substitutes
// defaultPagedKVPageSize whenever pageSize<=0 — there is no "plain linear" opt-out for a live
// session); the linear lb caches this file ALSO supports only get exercised via
// s.hasDevicePagedKV()==false, which happens for a TurboQuant or attention-sinks session, not
// gemma4 26B-A4B. Both attention halves must therefore actually work for this lane to matter on
// the model #53 names.
//
// SCOPE (mtpRowsDriverEligible): a plain, cache-OWNING, non-shared, non-TurboQuant attention layer
// (linear OR device-paged — both implemented), feeding a UNIFORM quant MoE FFN (every layer,
// mtpRowsMoEEligible) that is what gemma4 26B-A4B actually is (gemma4.Config.Arch: "gemma4 applies
// MoE uniformly across layers"; every attention layer owns its cache unless a checkpoint sets
// num_kv_shared_layers > 0). Any model that departs from that shape — a qwen hybrid mixer,
// KV-shared layers, a TurboQuant KV state lane, a dense (non-MoE) layer mixed in, PLE (gemma4
// E2B/E4B only, never this MoE checkpoint) — declines the WHOLE block up front, before any GPU
// work starts, and the caller keeps today's row-major lane unchanged. Never a wrong-but-silent
// answer.

// mtpRowsDriverEngaged counts successful whole-block completions of stepRowsMoEBatched — the
// wiring-level engagement proof (mirrors mtpRowsMoEMaxGroupSize at the primitive level, one layer
// down): a test that never sees this counter move proves nothing about the layer-major driver
// itself, even if the primitive's own counter moved on some OTHER, unrelated call.
var mtpRowsDriverEngaged atomic.Int64

// mtpRowsDiagAttnWall / mtpRowsDiagMoEWall accumulate ONE verify round's layer-major wall-clock
// split between the attention half (encRowsAttnLinear/encRowsAttnPaged) and the batched MoE half
// (mtpRowsMoEBatched, including the per-layer output scalar) — the #53 diagnostic's per-stage
// breakdown ("mtp-diag rows-moe", mtpRowsDiagEmitRound). stepRowsMoEBatched resets both to zero
// at the start of every round and sums across all layers in the block; verifyRowsMoEBatchedHiddens
// reads them the moment the round completes. Package-level and NOT atomic, like
// mtpDiagDraftCalls/mtpDiagVerifyRowsCalls (mtp.go): a decode session drives one goroutine (mtp.go's
// own contract), so there is no concurrent-round hazard.
var (
	mtpRowsDiagAttnWall time.Duration
	mtpRowsDiagMoEWall  time.Duration
)

// mtpRowsDiagRoundsSeen counts verify rounds that reached verifyRowsMoEBatchedHiddens (engaged or
// declined) — the #53 diagnostic's round number and its "first round" gate: round 1 always logs
// (mtp-diag rows-moe), every later round only under LTHN_MTP_DIAG (mtpDiagForTest, mtp.go) —
// matching the cache-plan/load-summary precedent of an always-on trace-class log kept off the
// serve hot path's per-round rate without a NEW env var.
var mtpRowsDiagRoundsSeen int

// mtpRowsDiagSnapshot is one verify round's #53 instrument state, populated by
// verifyRowsMoEBatchedHiddens (both the decline and the engage path) and consumed once by
// mtpRowsDiagEmitRound — called from the caller one frame up (verifyAssistantDraftRows,
// assistant_load.go) after it has ALSO measured the head/greedy stage, which this driver has no
// visibility into (the head runs on the hiddens THIS function returns, after it has returned
// them).
type mtpRowsDiagSnapshot struct {
	Engaged                        bool
	Reason                         string // decline condition (mtpRowsDriverDeclineReason); "" when Engaged
	K                              int
	Hist1, Hist2, Hist3, Hist4Plus int64 // expert-group sizes seen this round, summed over layers
	MaxGroup                       int64
	AttnWall, MoEWall              time.Duration
}

// mtpRowsDiagLast is the most recently completed round's snapshot — single-goroutine, like every
// other #53/#352 diag var in this package.
var mtpRowsDiagLast mtpRowsDiagSnapshot

// mtpRowsDriverEligible reports whether the WHOLE verify block — every layer, not just the MoE
// weights — can take the layer-major batched driver. false always means "the caller keeps the
// row-major per-row lane" — this function is the ONLY gate; stepRowsMoEBatched trusts it and
// treats any mid-block inconsistency as an error, not a fallback (see stepRowsMoEBatched's doc).
// A thin bool wrapper over mtpRowsDriverDeclineReason so existing callers/tests keep the plain
// contract; the #53 diagnostic (verifyRowsMoEBatchedHiddens, "mtp-diag rows-moe") wants the
// SPECIFIC reason a real serve session declined, which a bare bool can never carry.
func mtpRowsDriverEligible(s *archDecodeState, maxRows int) bool {
	return mtpRowsDriverDeclineReason(s, maxRows) == ""
}

// mtpRowsDriverDeclineReason walks the SAME eligibility checks as mtpRowsDriverEligible, in the
// SAME order, but returns the specific condition that declined instead of a bare false. The #53
// diagnostic instrument (verifyRowsMoEBatchedHiddens, mtp-diag rows-moe) names this in its
// per-round trace line so a live serve decline (hypothesis A: the driver never engages on the
// real 26B geometry) is distinguishable from an engage-but-scatter decline (hypothesis B: the
// driver runs but the expert grouping degenerates) without re-deriving the walk by hand. ""
// means eligible (mtpRowsDriverEligible's true). Per-layer reasons carry the layer index (@L<n>)
// since a real checkpoint's layers are not uniform BY ASSUMPTION, only uniform by construction
// for gemma4 — the index is the fastest way to confirm that construction actually held.
func mtpRowsDriverDeclineReason(s *archDecodeState, maxRows int) string {
	if s == nil {
		return "nil-state"
	}
	if len(s.specs) == 0 {
		return "no-layer-specs"
	}
	if len(s.lb) != len(s.specs) {
		return "lb-specs-length-mismatch"
	}
	if s.dModel <= 0 || s.dFF <= 0 {
		return "dmodel-or-dff-invalid"
	}
	// state-level: no diagnostic/test hook, no PLE tower, no chained live-decode tail, no resident
	// ICB replay. icb is nil by construction whenever a session loads quant MoE (session build's
	// icbEligible excludes MoE) — checked here anyway as a hard guarantee, not an assumption.
	if s.trace {
		return "trace-hook-active"
	}
	if s.gpuProf != nil {
		return "gpu-profile-active"
	}
	if s.chainTail != nil {
		return "chain-tail-active"
	}
	if len(s.ple) > 0 {
		return "ple-tower-present"
	}
	if s.icb != nil {
		return "icb-resident"
	}
	if layerSpanProbeForTest != nil {
		return "layer-span-test-probe-active"
	}
	if captureLayerHiddens {
		return "capture-layer-hiddens-active"
	}
	for li := range s.specs {
		spec := s.specs[li]
		if spec.Mixer == model.MixerGatedDelta {
			return core.Sprintf("gated-delta-mixer@L%d", li) // Qwen3.5 gated-delta recurrence: no KV cache, a different mixer entirely
		}
		if s.gatedAttn != nil && li < len(s.gatedAttn) && s.gatedAttn[li] != nil {
			return core.Sprintf("gated-attention-gate@L%d", li) // Qwen3.5 gated full-attention (attn_output_gate): a different fused lane
		}
		if !spec.OwnsCache() {
			return core.Sprintf("shared-kv-cache@L%d", li) // cross-layer KV sharing: not implemented (encAttnHalfShared[Paged]'s shape)
		}
		if s.kvTQState.on(li) {
			return core.Sprintf("turboquant-kv-state@L%d", li) // TurboQuant KV state lane: not implemented by this driver
		}
		moeQ := moeQuantAt(s.moeQuant, li)
		if moeQ == nil {
			return core.Sprintf("non-uniform-moe-layer@L%d", li) // uniform-MoE only: a dense or bf16-MoE layer is out of scope
		}
		if !mtpRowsMoEEligible(*moeQ, s.dModel, s.dFF, maxRows) {
			return core.Sprintf("moe-geometry-ineligible@L%d", li)
		}
	}
	return ""
}

// stepRowsMoEBatched is the layer-major verify driver. embs holds K rows' input embeddings
// packed contiguously (row r at embs[r*rowBytes:(r+1)*rowBytes], rowBytes = dModel bf16 bytes);
// startPos is the FIRST row's absolute position — row r sits at startPos+r, exactly matching what
// K sequential ArchSession.stepID calls would advance s.pos through. Returns the K rows' final
// hidden states (post-last-layer, pre-final-norm — the same value stepTokenEncode's per-row
// res/hidden is), packed the same way.
//
// Per layer: all K rows' attention halves encode into ONE command buffer in row order (either
// encRowsAttnLinear or encRowsAttnPaged, chosen per layer by s.layerPagedKV — see their own docs
// for the ordering argument), committed and waited once; row r's SDPA reads K/V rows 0..r written
// earlier in THIS SAME pass, so the row order must hold and does. Once all K rows' post-attention
// hidden lands in one packed hSlab, mtpRowsMoEBatched (mtp_rows_moe.go) runs the WHOLE MoE block
// for all K rows grouped by routed expert; the per-layer output scalar (if the arch declares one)
// then applies per row, and the result becomes layer L+1's input.
//
// Callers MUST have already confirmed mtpRowsDriverEligible(s) — this function trusts that and
// surfaces an unexpected mtpRowsMoEBatched decline as an ERROR, never a silent per-layer fallback:
// by the time layer L's MoE runs, this block has already committed KV writes for layer L (and
// every earlier layer) under the layer-major schedule, so falling back to the row-major lane
// mid-block would mean recomputing from scratch to stay safe — treating the inconsistency as
// exceptional (it should never fire: mtpRowsDriverEligible checks the SAME mtpRowsMoEEligible
// predicate against the SAME weights) is simpler and equally safe, since the lever defaults off
// and only affects callers who opted in.
func (s *archDecodeState) stepRowsMoEBatched(embs []byte, startPos, K int) ([]byte, error) {
	if s == nil {
		return nil, core.NewError("native.archDecodeState.stepRowsMoEBatched: nil state")
	}
	rowBytes := s.dModel * bf16Size
	if K < 1 {
		return nil, core.NewError("native.archDecodeState.stepRowsMoEBatched: K must be >= 1")
	}
	if len(embs) != K*rowBytes {
		return nil, core.NewError("native.archDecodeState.stepRowsMoEBatched: embs must be K*dModel bf16 bytes")
	}
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if !s.hasDevicePagedKV() { // mirrors stepTokenEncode's own entry guard
		if err := s.ensureLBKVCaches(); err != nil {
			return nil, err
		}
	}

	// One packed K-slot position buffer, written ONCE (positions don't change across layers within
	// one verify block) — deliberately NOT s.offBuf/s.offRing: those belong to the single-row step
	// machinery (the submit-ahead chain's ring), and this driver must never perturb state the
	// row-major fallback (or a later live-decode step on the same session) still relies on.
	offPacked := device.NewBufferWithLengthOptions(uint(K*4), metal.MTLResourceStorageModeShared)
	if offPacked == nil {
		return nil, core.NewError("native.archDecodeState.stepRowsMoEBatched: position buffer allocation failed")
	}
	offSlots := unsafe.Slice((*int32)(offPacked.Contents()), K)
	for r := range K {
		offSlots[r] = int32(startPos + r)
	}

	// #53 diag (mtpRowsDiagEmitRound, assistant_load.go): reset the round-scoped attention/MoE
	// wall-clock split and expert-group histogram HERE, once per verify round, so a multi-layer
	// block's per-layer contributions sum cleanly and never leak into the NEXT round's numbers.
	mtpRowsDiagAttnWall = 0
	mtpRowsDiagMoEWall = 0
	mtpRowsMoEGroupHistReset()

	// one-submission round (#53): every layer chains into a single command
	// buffer when the device-routed gather serves the whole block; the
	// per-layer loop below is the fallback.
	if out, chained, cerr := s.stepRowsMoEBatchedChained(embs, startPos, K); cerr != nil {
		return nil, cerr
	} else if chained {
		return out, nil
	}

	curInHost := embs // layer 0 input: the K embedded rows
	for li := range s.specs {
		lhd, lkv := headDimOf(s.specs[li], s.headDim), kvHeadsOf(s.specs[li], s.nKVHeads)
		slideW, rbase, rotDim := 0, s.base, s.rotaryDim
		layerRopeFreqs := s.ropeFreqs
		if s.specs[li].Attention == model.SlidingAttention {
			slideW, rbase, rotDim = s.slidingWindow, s.localBase, s.rotaryDimLocal
		} else if s.globalRopeFreqs != nil {
			layerRopeFreqs, rotDim = s.globalRopeFreqs, lhd
		}

		var (
			hSlabHost []byte
			err       error
		)
		attnT0 := time.Now()
		if cache := s.layerPagedKV(li); cache != nil {
			hSlabHost, err = s.encRowsAttnPaged(cache, curInHost, offPacked, li, K, rowBytes, startPos, lhd, lkv, slideW, rotDim, rbase, layerRopeFreqs)
		} else {
			hSlabHost, err = s.encRowsAttnLinear(curInHost, offPacked, li, K, rowBytes, startPos, lhd, lkv, slideW, rotDim, rbase, layerRopeFreqs)
		}
		mtpRowsDiagAttnWall += time.Since(attnT0)
		if err != nil {
			return nil, err
		}

		moeQ := moeQuantAt(s.moeQuant, li)
		if moeQ == nil {
			return nil, core.NewError("native.archDecodeState.stepRowsMoEBatched: layer lost its quant MoE weights mid-block")
		}
		moeT0 := time.Now()
		outHost, ok, err := mtpRowsMoEBatched(hSlabHost, *moeQ, s.dModel, s.dFF, K, s.eps)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, core.NewError("native.archDecodeState.stepRowsMoEBatched: mtpRowsMoEBatched declined a layer after mtpRowsDriverEligible passed — qmvByteExactServable is out of sync with encQMVByteExactAt (fix the probe, they must mirror)")
		}

		if s.lb[li].layerScalar != nil {
			outHost, err = s.applyLayerScalarRows(outHost, s.lb[li].layerScalar, K, rowBytes)
			if err != nil {
				return nil, err
			}
		}
		// Timed through the (optional) layer-scalar application too — it is the MoE output's own
		// per-layer scale, applied before the residual feeds the next layer's attention.
		mtpRowsDiagMoEWall += time.Since(moeT0)
		curInHost = outHost
	}

	mtpRowsDriverEngaged.Add(1)
	return curInHost, nil
}

// encRowsAttnLinear runs layer li's attention half for K rows over the plain (non-paged) linear
// KV cache: K sequential encAttnHalfKVInputAt calls (decode_step.go's byte-exact single-row
// kernel — encAttnHalfKV's own offset-capable twin, addressed at row offsets into ONE packed
// K-row buffer instead of the state's single-row s.xA/s.xB/s.hBuf), one command buffer. Row r's
// SDPA reads K/V rows 0..r-1 written earlier in THIS SAME pass — the row order must hold and does,
// because Metal's compute encoder hazard-tracks buffer writes/reads between dispatches in
// submission order, the identical guarantee stepTokenEncode's OWN cross-LAYER scratch reuse
// (s.asc reused layer to layer) already depends on; this is the same mechanism, reused across
// rows within one layer instead of across layers within one row.
func (s *archDecodeState) encRowsAttnLinear(curInHost []byte, offPacked metal.MTLBuffer, li, K, rowBytes, startPos, lhd, lkv, slideW, rotDim int, rbase float32, layerRopeFreqs metal.MTLBuffer) ([]byte, error) {
	curIn := sharedBytes(curInHost)
	hSlab := scratchBF16(K * s.dModel)
	if hSlab == nil {
		return nil, core.NewError("native.archDecodeState.encRowsAttnLinear: attention-output scratch allocation failed")
	}
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	for r := range K {
		rOff := uint(r * rowBytes)
		if err := encAttnHalfKVInputAt(enc, curIn, rOff, s.lb[li].kCache, s.lb[li].vCache, offPacked, hSlab, rOff, uint(r*4),
			s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.lb[li].kNorm, s.valueNormOnes, s.asc, s.lb[li].proj,
			s.dModel, s.nHeads, lkv, lhd, startPos+r, slideW, rotDim, rbase, s.scale, s.ropeScale, s.eps, layerRopeFreqs, s.lb[li].sinks); err != nil {
			endEncodingFast(enc)
			return nil, err
		}
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)

	hSlabHost := make([]byte, K*rowBytes)
	copy(hSlabHost, unsafe.Slice((*byte)(hSlab.Contents()), K*rowBytes))
	return hSlabHost, nil
}

// encRowsAttnPaged is encRowsAttnLinear's device-paged-KV twin — the LIVE DEFAULT cache mode for
// a quant MoE session (see the file header: initDevicePagedKVWithPrealloc builds paged caches for
// every cache-owning layer unconditionally). encAttnHalfKVPaged (paged_kv_device.go) has no
// row-offset parameters — its x/h are always the given buffer's own byte 0 — so each row gets its
// OWN 1-row input/output buffer rather than an offset into one packed slab.
//
// The SAME K-sequential-calls-into-one-command-buffer shape still holds, and so does its ordering
// argument: enc/encConc thread through every call exactly as they already thread ACROSS LAYERS in
// stepTokenEncode's own loop (encAttnHalfKVPaged's fused-rope fast path leaves the encoder open in
// CONCURRENT mode and carries it into the next call behind a single buffer barrier — "the next
// pass's entry barrier orders ITS writes in turn", per that loop's own comment on the identical
// carry). This driver's cross-ROW carry is the SAME contract, one level down: row r+1's call opens
// with that barrier (or, on the non-fused-rope serial fallback, ordinary hazard tracking) ordering
// EVERYTHING row r encoded — including its page write — ahead of EVERYTHING row r+1 encodes,
// including its SDPA read of that page. Metal has no notion of "layer" vs "row" here; the contract
// is per-call, not per-caller.
func (s *archDecodeState) encRowsAttnPaged(cache *devicePagedKVCache, curInHost []byte, offPacked metal.MTLBuffer, li, K, rowBytes, startPos, lhd, lkv, slideW, rotDim int, rbase float32, layerRopeFreqs metal.MTLBuffer) ([]byte, error) {
	hRows := make([]metal.MTLBuffer, K)
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	encConc := false
	for r := range K {
		xRow := sharedBytes(curInHost[r*rowBytes : (r+1)*rowBytes])
		hRow := scratchBF16(s.dModel)
		if hRow == nil {
			endEncodingFast(enc)
			return nil, core.NewError("native.archDecodeState.encRowsAttnPaged: attention-output scratch allocation failed")
		}
		var perr error
		enc, encConc, perr = encAttnHalfKVPaged(enc, cb, s.gpuProf, encConc, xRow, cache, offPacked, hRow, uint(r*4),
			s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.lb[li].kNorm, s.valueNormOnes, s.asc, s.lb[li].proj,
			s.dModel, s.nHeads, lkv, lhd, startPos+r, slideW, rotDim, rbase, s.scale, s.ropeScale, s.eps, layerRopeFreqs)
		if perr != nil {
			endEncodingFast(enc)
			return nil, perr
		}
		hRows[r] = hRow
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)

	hSlabHost := make([]byte, K*rowBytes)
	for r := range K {
		copy(hSlabHost[r*rowBytes:(r+1)*rowBytes], unsafe.Slice((*byte)(hRows[r].Contents()), rowBytes))
	}
	return hSlabHost, nil
}

// applyLayerScalarRows multiplies each of K packed rows by the SAME broadcast dModel-wide
// layerScalar vector (gemma4's per-layer output scalar) — K single-row encMulBF16To dispatches
// (elementwise, no cross-row interaction, so row order is irrelevant) into one command buffer.
func (s *archDecodeState) applyLayerScalarRows(rowsHost []byte, layerScalar metal.MTLBuffer, K, rowBytes int) ([]byte, error) {
	outBuf := sharedBytes(rowsHost)
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	for r := range K {
		rOff := uint(r * rowBytes)
		if err := encMulBF16To(enc, outBuf, layerScalar, outBuf, rOff, 0, rOff, s.dModel); err != nil {
			endEncodingFast(enc)
			return nil, err
		}
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	out := make([]byte, K*rowBytes)
	copy(out, unsafe.Slice((*byte)(outBuf.Contents()), K*rowBytes))
	return out, nil
}

// verifyRowsMoEBatchedHiddens is the ArchSession-level #53 entry point, called only from the
// byte-exact greedy verify lane (verifyAssistantDraftHiddens, exact=true) while the lane is armed
// (mtpRowsMoEArmed — default ON; LTHN_MTP_ROWS_MOE=0 opts out). ok=false (nil error) means
// declined — the caller falls back to the existing
// per-row stepID loop UNCHANGED; nothing here has side-effected the session (no embedding, no GPU
// work) unless mtpRowsDriverEligible already passed.
func (s *ArchSession) verifyRowsMoEBatchedHiddens(draftTokens []int32, rowBytes int) ([][]byte, bool, error) {
	K := len(draftTokens)
	reason := ""
	switch {
	case K < 1:
		reason = "k-lt-1"
	case s.perLayerInput != nil:
		reason = "per-layer-input-active"
	default:
		reason = mtpRowsDriverDeclineReason(&s.state, K)
	}
	if reason != "" {
		mtpRowsDiagLast = mtpRowsDiagSnapshot{K: K, Reason: reason}
		return nil, false, nil
	}
	embs := make([]byte, K*rowBytes)
	for i, tok := range draftTokens {
		emb, err := s.embedID(tok)
		if err != nil {
			return nil, false, err
		}
		if len(emb) != rowBytes {
			return nil, false, core.NewError("native.assistant verify rows-moe embed size mismatch")
		}
		copy(embs[i*rowBytes:(i+1)*rowBytes], emb)
	}
	startPos := s.pos
	var (
		out []byte
		err error
	)
	withAutoreleasePool(func() {
		out, err = s.state.stepRowsMoEBatched(embs, startPos, K)
	})
	if err != nil {
		return nil, false, err
	}
	s.pos = startPos + K
	rows := make([][]byte, K)
	for i := range rows {
		rows[i] = append([]byte(nil), out[i*rowBytes:(i+1)*rowBytes]...)
	}
	mtpRowsDiagLast = mtpRowsDiagSnapshot{
		Engaged:   true,
		K:         K,
		Hist1:     mtpRowsMoEGroupHist1.Load(),
		Hist2:     mtpRowsMoEGroupHist2.Load(),
		Hist3:     mtpRowsMoEGroupHist3.Load(),
		Hist4Plus: mtpRowsMoEGroupHist4Plus.Load(),
		MaxGroup:  mtpRowsMoEMaxGroupSize.Load(),
		AttnWall:  mtpRowsDiagAttnWall,
		MoEWall:   mtpRowsDiagMoEWall,
	}
	return rows, true, nil
}

// mtpRowsDiagEmitRound closes out the #53 per-round instrument ("mtp-diag rows-moe"): combines
// mtpRowsDiagLast — the attention/MoE wall split, expert-group histogram, and engaged-or-declined
// reason verifyRowsMoEBatchedHiddens captured THIS round — with headWall, the LM-head/greedy
// stage's wall time, which only the caller can measure (verifyAssistantDraftRows,
// assistant_load.go: the head runs one call-frame up, on the hiddens this driver returns, so it
// is invisible from here — the reason that file is in the #53 file fence at all). Always logs the
// session's first round (so a serve log always carries at least one #53 receipt even with
// LTHN_MTP_DIAG unset); every later round only under LTHN_MTP_DIAG (mtpDiagForTest) — the
// per-round rate is serve noise otherwise, matching the cache-plan/load-summary precedent
// (cache_plan.go, device.go) of an always-on trace-class log with no NEW env gate.
func mtpRowsDiagEmitRound(headWall time.Duration) {
	mtpRowsDiagRoundsSeen++
	if mtpRowsDiagRoundsSeen != 1 && !mtpDiagForTest {
		return
	}
	d := mtpRowsDiagLast
	headMs := float64(headWall.Microseconds()) / 1000
	if !d.Engaged {
		nativeTraceLog(core.Sprintf("mtp-diag rows-moe round=%d engaged=false reason=%s K=%d wall{head=%.2fms}\n",
			mtpRowsDiagRoundsSeen, d.Reason, d.K, headMs))
		return
	}
	nativeTraceLog(core.Sprintf(
		"mtp-diag rows-moe round=%d engaged=true K=%d hist{1=%d 2=%d 3=%d 4+=%d} maxGroup=%d wall{attn=%.2fms moe=%.2fms head=%.2fms}\n",
		mtpRowsDiagRoundsSeen, d.K, d.Hist1, d.Hist2, d.Hist3, d.Hist4Plus, d.MaxGroup,
		float64(d.AttnWall.Microseconds())/1000, float64(d.MoEWall.Microseconds())/1000, headMs))
}

// mtpRowsMoEGatherServable mirrors mtpRowsMoEBatchedGatherEnc's pipeline and
// geometry prologue as a pure probe — the chained round driver confirms EVERY
// layer before encoding anything, so a mid-round decline (which would strand
// committed KV writes) is structurally impossible rather than merely unlikely.
func mtpRowsMoEGatherServable(w MoEQuantLayerWeights, dModel, dFF int) bool {
	numExperts, topK, expertDFF := w.NumExperts, w.TopK, w.ExpertDFF
	if !routerTopKUsable(numExperts, topK) {
		return false
	}
	rGroupSize, rBits := quantWeightGeometryForShape(w.Router, numExperts, dModel, w.RouterGroupSize, w.RouterBits)
	if rGroupSize <= 0 || dModel%rGroupSize != 0 {
		return false
	}
	if _, err := pipelineFor(rmsKernelBF16(dModel)); err != nil {
		return false
	}
	if _, err := pipelineFor(qmvBF16KernelName(numExperts, dModel, rGroupSize, rBits)); err != nil {
		return false
	}
	if _, err := routerTopKPipelineK(topK); err != nil {
		return false
	}
	fusedExperts := len(w.ExpGateUp.Packed) > 0
	var expGS, expBits, gateRows int
	if fusedExperts {
		_, _, _, gs, bits, err := quantWeightViewsForShape("native.mtpRowsMoEGatherServable: expert gate_up", w.ExpGateUp, numExperts*2*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return false
		}
		expGS, expBits, gateRows = gs, bits, 2*expertDFF
	} else {
		_, _, _, gs, bits, err := quantWeightViewsForShape("native.mtpRowsMoEGatherServable: expert gate", w.ExpGate, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return false
		}
		if _, _, _, _, _, uerr := quantWeightViewsForShape("native.mtpRowsMoEGatherServable: expert up", w.ExpUp, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits); uerr != nil {
			return false
		}
		expGS, expBits, gateRows = gs, bits, expertDFF
	}
	_, _, _, downGS, downBits, derr := quantWeightViewsForShape("native.mtpRowsMoEGatherServable: expert down", w.ExpDown, numExperts*dModel, expertDFF, w.ExpertGroupSize, w.ExpertBits)
	if derr != nil {
		return false
	}
	if _, ok := lthnGatherQMVPipeline(lthnGatherQMVKey{groupSize: expGS, bits: expBits, expertRows: gateRows, fast: expertDFF%8 == 0 && dModel%512 == 0, batchedX: true}); !ok {
		return false
	}
	if _, ok := lthnGatherQMVPipeline(lthnGatherQMVKey{groupSize: downGS, bits: downBits, expertRows: dModel, batchedX: true, gelu: true}); !ok {
		return false
	}
	if _, err := moeWeightedSumPipeline(); err != nil {
		return false
	}
	combinePSO, err := moeCombineNormsPipeline()
	if err != nil {
		return false
	}
	combineTG := uint(rmsSimdSize * ((((dModel + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
	if combineTG > combinePSO.MaxTotalThreadsPerThreadgroup() {
		return false
	}
	if _, _, _, _, _, err := quantWeightViewsForShape("native.mtpRowsMoEGatherServable: local gate", w.LocalGate, dFF, dModel, w.LocalGroupSize, w.LocalBits); err != nil {
		return false
	}
	if _, _, _, _, _, err := quantWeightViewsForShape("native.mtpRowsMoEGatherServable: local up", w.LocalUp, dFF, dModel, w.LocalGroupSize, w.LocalBits); err != nil {
		return false
	}
	if _, _, _, _, _, err := quantWeightViewsForShape("native.mtpRowsMoEGatherServable: local down", w.LocalDown, dModel, dFF, w.LocalGroupSize, w.LocalBits); err != nil {
		return false
	}
	return true
}

// stepRowsMoEBatchedChained is the ONE-SUBMISSION round driver (#53): every
// layer's attention rows and device-routed MoE block encode back-to-back into
// a single command buffer — attention on its own serial encoder (row r+1's
// SDPA reads row r's KV write through in-buffer hazard order, the same
// argument as the per-layer form), the MoE block on a fresh serial encoder —
// and the round pays ONE commit+wait where the per-layer form paid two per
// layer. Rows live on-device across the whole round in three rotating K-row
// slabs; the host touches bytes exactly twice (embs in, final hiddens out).
// ok=false means a layer's gather block is not servable and the caller keeps
// the per-layer driver.
func (s *archDecodeState) stepRowsMoEBatchedChained(embs []byte, startPos, K int) ([]byte, bool, error) {
	rowBytes := s.dModel * bf16Size
	for li := range s.specs {
		moeQ := moeQuantAt(s.moeQuant, li)
		if moeQ == nil || !mtpRowsMoEGatherServable(*moeQ, s.dModel, s.dFF) {
			return nil, false, nil
		}
	}
	offPacked := device.NewBufferWithLengthOptions(uint(K*4), metal.MTLResourceStorageModeShared)
	if offPacked == nil {
		return nil, false, core.NewError("native.stepRowsMoEBatchedChained: position buffer allocation failed")
	}
	offSlots := unsafe.Slice((*int32)(offPacked.Contents()), K)
	for r := range K {
		offSlots[r] = int32(startPos + r)
	}
	mtpRowsDiagAttnWall = 0
	mtpRowsDiagMoEWall = 0
	mtpRowsMoEGroupHistReset()

	slabIn := scratchBF16(K * s.dModel)
	slabH := scratchBF16(K * s.dModel)
	slabOut := scratchBF16(K * s.dModel)
	if slabIn == nil || slabH == nil || slabOut == nil {
		return nil, false, core.NewError("native.stepRowsMoEBatchedChained: slab allocation failed")
	}
	copy(unsafe.Slice((*byte)(slabIn.Contents()), K*rowBytes), embs)

	firstMoE := moeQuantAt(s.moeQuant, 0)
	scratches := make([]*routerDeviceScratch, K)
	defer func() {
		for _, sc := range scratches {
			if sc != nil {
				putRouterDeviceScratch(sc)
			}
		}
	}()
	for r := range K {
		sc, serr := getRouterDeviceScratch(s.dModel, firstMoE.NumExperts, firstMoE.TopK)
		if serr != nil {
			return nil, false, serr
		}
		scratches[r] = sc
	}

	idxDevs := make([]metal.MTLBuffer, 0, len(s.specs))
	topKs := make([]int, 0, len(s.specs))
	foldScr := s.rowsAttnScratchSets(K)
	cb := commandBufferFast(queue)
	for li := range s.specs {
		lhd, lkv := headDimOf(s.specs[li], s.headDim), kvHeadsOf(s.specs[li], s.nKVHeads)
		slideW, rbase, rotDim := 0, s.base, s.rotaryDim
		layerRopeFreqs := s.ropeFreqs
		if s.specs[li].Attention == model.SlidingAttention {
			slideW, rbase, rotDim = s.slidingWindow, s.localBase, s.rotaryDimLocal
		} else if s.globalRopeFreqs != nil {
			layerRopeFreqs, rotDim = s.globalRopeFreqs, lhd
		}
		attnT0 := time.Now()
		if cache := s.layerPagedKV(li); cache != nil && s.gpuProf == nil && !attnConcurrentDisabled && gpuHasGeluKernel() && s.lb[li].qNorm.buf != nil { // BISECT arm 2
			if err := s.encRowsAttnPagedFold(cb, cache, slabIn, offPacked, li, K, rowBytes, startPos, lhd, lkv, slideW, rotDim, rbase, layerRopeFreqs, slabH, foldScr); err != nil {
				return nil, false, err
			}
			mtpRowsDiagAttnWall += time.Since(attnT0)
			moeQ := moeQuantAt(s.moeQuant, li)
			moeT0 := time.Now()
			enc2 := computeCommandEncoderFast(cb)
			idxDev, ok, err := mtpRowsMoEBatchedGatherEnc(enc2, slabH, slabOut, scratches, *moeQ, s.dModel, s.dFF, K, s.eps)
			if err != nil {
				endEncodingFast(enc2)
				return nil, false, err
			}
			if !ok {
				endEncodingFast(enc2)
				return nil, false, core.NewError("native.stepRowsMoEBatchedChained: gather block declined after the probe passed — mtpRowsMoEGatherServable is out of sync with mtpRowsMoEBatchedGatherEnc")
			}
			if s.lb[li].layerScalar != nil {
				for r := range K {
					rOff := uint(r * rowBytes)
					if err := encMulBF16To(enc2, slabOut, s.lb[li].layerScalar, slabOut, rOff, 0, rOff, s.dModel); err != nil {
						endEncodingFast(enc2)
						return nil, false, err
					}
				}
			}
			endEncodingFast(enc2)
			mtpRowsDiagMoEWall += time.Since(moeT0)
			idxDevs = append(idxDevs, idxDev)
			topKs = append(topKs, moeQ.TopK)
			slabIn, slabOut = slabOut, slabIn
			continue
		}
		enc := computeCommandEncoderFast(cb)
		if cache := s.layerPagedKV(li); cache != nil {
			encConc := false
			var perr error
			for r := range K {
				rOff := uint(r * rowBytes)
				enc, encConc, perr = encAttnHalfKVPagedInputAt(enc, cb, s.gpuProf, encConc, slabIn, rOff, cache, offPacked, slabH, rOff, uint(r*4),
					s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.lb[li].kNorm, s.valueNormOnes, s.asc, s.lb[li].proj,
					s.dModel, s.nHeads, lkv, lhd, startPos+r, slideW, rotDim, rbase, s.scale, s.ropeScale, s.eps, layerRopeFreqs)
				if perr != nil {
					endEncodingFast(enc)
					return nil, false, perr
				}
			}
		} else {
			for r := range K {
				rOff := uint(r * rowBytes)
				if err := encAttnHalfKVInputAt(enc, slabIn, rOff, s.lb[li].kCache, s.lb[li].vCache, offPacked, slabH, rOff, uint(r*4),
					s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.lb[li].kNorm, s.valueNormOnes, s.asc, s.lb[li].proj,
					s.dModel, s.nHeads, lkv, lhd, startPos+r, slideW, rotDim, rbase, s.scale, s.ropeScale, s.eps, layerRopeFreqs, s.lb[li].sinks); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
			}
		}
		endEncodingFast(enc)
		mtpRowsDiagAttnWall += time.Since(attnT0)

		moeQ := moeQuantAt(s.moeQuant, li)
		moeT0 := time.Now()
		enc2 := computeCommandEncoderFast(cb)
		idxDev, ok, err := mtpRowsMoEBatchedGatherEnc(enc2, slabH, slabOut, scratches, *moeQ, s.dModel, s.dFF, K, s.eps)
		if err != nil {
			endEncodingFast(enc2)
			return nil, false, err
		}
		if !ok {
			endEncodingFast(enc2)
			return nil, false, core.NewError("native.stepRowsMoEBatchedChained: gather block declined after the probe passed — mtpRowsMoEGatherServable is out of sync with mtpRowsMoEBatchedGatherEnc")
		}
		if s.lb[li].layerScalar != nil {
			for r := range K {
				rOff := uint(r * rowBytes)
				if err := encMulBF16To(enc2, slabOut, s.lb[li].layerScalar, slabOut, rOff, 0, rOff, s.dModel); err != nil {
					endEncodingFast(enc2)
					return nil, false, err
				}
			}
		}
		endEncodingFast(enc2)
		mtpRowsDiagMoEWall += time.Since(moeT0)
		idxDevs = append(idxDevs, idxDev)
		topKs = append(topKs, moeQ.TopK)
		slabIn, slabOut = slabOut, slabIn
	}
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	for i, idx := range idxDevs {
		mtpRowsMoEHistFromIdx(idx, K*topKs[i])
	}
	out := make([]byte, K*rowBytes)
	copy(out, unsafe.Slice((*byte)(slabIn.Contents()), K*rowBytes))
	mtpRowsDriverEngaged.Add(1)
	return out, true, nil
}

// rowsAttnScratchSet is the chained round's per-row attention scratch — K
// disjoint sets let the K rows' projection/rope/output stages overlap on a
// concurrent encoder (the single shared s.asc is exactly what forced row
// order on every stage, dependent or not). Sized once per round at the
// largest layer geometry and reused across layers: encoder boundaries on one
// command buffer are full barriers, so layer L+1's writes never race layer
// L's reads.
func (s *archDecodeState) rowsAttnScratchSets(K int) []attnScratch {
	maxQ, maxKV := 0, 0
	for li := range s.specs {
		lhd, lkv := headDimOf(s.specs[li], s.headDim), kvHeadsOf(s.specs[li], s.nKVHeads)
		if q := s.nHeads * lhd; q > maxQ {
			maxQ = q
		}
		if kv := lkv * lhd; kv > maxKV {
			maxKV = kv
		}
	}
	sets := make([]attnScratch, K)
	for r := range K {
		sets[r] = newAttnScratch(s.dModel, maxQ, maxKV, s.nHeads, 0)
	}
	return sets
}

// encRowsAttnPagedFold encodes layer li's K attention rows as STAGE WAVES on
// one concurrent encoder (#53): every row's RMS, q/k/v projections, ropes,
// value norm, output projection and residual overlap freely across rows
// (disjoint per-row scratch, read-only weights), with explicit barriers at
// the true stage edges — the monolith's own concurrent-pass structure,
// widened from one row to K. The landing + SDPA stage is two-mode:
//
//   - GLOBAL layers land ALL K rows' pages first, then run the K SDPAs
//     barrier-free (pure page reads, disjoint outputs). Byte-exact vs the
//     sequential order because row r's plan is built at watermark pos_r —
//     rows r+1..K-1's pages exist but sit past every lens the plan baked.
//   - RING (sliding) layers keep the per-row land->scan chain: a later
//     row's landing EVICTS a position inside an earlier row's window, so
//     land-before-read would change bytes (the deferred-ring lesson, paged
//     form). Their projections still ride the waves.
//
// The q8 paged cache stages K/V in the row's scratch and quantise-stores at
// the land point; a bf16 page copies the staged row instead — either way the
// LANDING time is the fold's to schedule, never the projection's.
func (s *archDecodeState) encRowsAttnPagedFold(cb metal.MTLCommandBufferObject, cache *devicePagedKVCache, slabIn metal.MTLBuffer, offPacked metal.MTLBuffer, li, K, rowBytes, startPos, lhd, lkv, slideW, rotDim int, rbase float32, layerRopeFreqs metal.MTLBuffer, slabH metal.MTLBuffer, scr []attnScratch) error {
	if slideW > 0 {
		if cache == nil {
			return core.NewError("native.encRowsAttnPagedFold: sliding window requires ring pages")
		}
		if !cache.ring {
			if cache.maxSize > slideW {
				return core.NewError("native.encRowsAttnPagedFold: sliding window requires ring pages")
			}
			slideW = 0
		}
	}
	ringMode := slideW > 0 && cache.ring
	kvDim := lkv * lhd

	// host prologue, sequenced per row: slot -> state -> plan, so each plan
	// bakes lens at ITS row's watermark (the causal bound; pageLens is a host
	// slice consumed at build time).
	type rowCtx struct {
		kPage, vPage           metal.MTLBuffer
		rowOff                 uint
		kScalePage, vScalePage metal.MTLBuffer
		scaleOff               uint
		plan                   sdpaPagedDecodePlan
	}
	rows := make([]rowCtx, K)
	for r := range K {
		kPage, vPage, rowOff, err := cache.slot(startPos + r)
		if err != nil {
			return err
		}
		keyPages, valuePages, pageLens, kHead, kSeq, vHead, vSeq, err := cache.state()
		if err != nil {
			return err
		}
		// state() refills SHARED per-cache scratch slices and the plan BORROWS
		// them — every later slot()/state() in this prologue would mutate what
		// an earlier row's plan reads at emit time (row 0 attending rows 1..K-1:
		// the causality break the parity gate caught). Each row's plan gets its
		// own copies; the monolith never needed them only because it emits each
		// plan before the next state().
		keyPages = append([]metal.MTLBuffer(nil), keyPages...)
		valuePages = append([]metal.MTLBuffer(nil), valuePages...)
		pageLens = append([]int(nil), pageLens...)
		kHead = append([]int(nil), kHead...)
		kSeq = append([]int(nil), kSeq...)
		vHead = append([]int(nil), vHead...)
		vSeq = append([]int(nil), vSeq...)
		pagedScratch, err := cache.attentionScratch(s.nHeads)
		if err != nil {
			return err
		}
		plan, err := buildSDPAPagedDecodePlan(scr[r].q, keyPages, valuePages, pageLens, kHead, kSeq, vHead, vSeq, scr[r].attn, pagedScratch, s.nHeads, lkv, lhd, s.scale)
		if err != nil {
			return err
		}
		rc := rowCtx{kPage: kPage, vPage: vPage, rowOff: rowOff, plan: plan}
		if cache.quantQ8 {
			kSc, vSc := cache.scaleState()
			if err := rc.plan.attachQ8(kSc, vSc); err != nil {
				return err
			}
			rc.kScalePage, rc.vScalePage, rc.scaleOff = cache.scaleSlot(startPos + r)
			if rc.kScalePage == nil || rc.vScalePage == nil {
				return core.NewError("native.encRowsAttnPagedFold: q8 cache missing scale pages")
			}
		}
		rows[r] = rc
	}

	enc := concurrentComputeEncoderFast(cb)
	encI := metal.MTLComputeCommandEncoder(enc)
	fail := func(err error) error {
		endEncodingFast(enc)
		return err
	}
	// W1: the shared input norms
	for r := range K {
		if err := encRMSNormBF16At(encI, slabIn, s.lb[li].anw.buf, scr[r].normed, uint(r*rowBytes), s.lb[li].anw.off, 0, s.dModel, s.eps); err != nil {
			return fail(err)
		}
	}
	memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
	// W2: q ∥ k ∥ v projections, all rows — K/V into the row's staging (the
	// landing schedule belongs to the fold)
	vIdx := projV
	if !s.lb[li].proj.hasV() {
		vIdx = projK
	}
	for r := range K {
		if err := s.lb[li].proj.project(encI, scr[r].normed, scr[r].q, 0, projQ); err != nil {
			return fail(err)
		}
		if err := s.lb[li].proj.project(encI, scr[r].normed, scr[r].kProj, 0, projK); err != nil {
			return fail(err)
		}
		if err := s.lb[li].proj.project(encI, scr[r].normed, scr[r].vProj, 0, vIdx); err != nil {
			return fail(err)
		}
	}
	memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
	// W3: q rope ∥ k rope ∥ v norm, all rows
	for r := range K {
		if err := encQKNormRopeAt(encI, scr[r].q, s.lb[li].qNorm.buf, scr[r].q, 0, s.lb[li].qNorm.off, 0, offPacked, uint(r*4), layerRopeFreqs, s.nHeads, lhd, rotDim, rbase, s.ropeScale, s.eps); err != nil {
			return fail(err)
		}
		if s.lb[li].kNorm.buf != nil {
			if err := encQKNormRopeAt(encI, scr[r].kProj, s.lb[li].kNorm.buf, scr[r].kProj, 0, s.lb[li].kNorm.off, 0, offPacked, uint(r*4), layerRopeFreqs, lkv, lhd, rotDim, rbase, s.ropeScale, s.eps); err != nil {
				return fail(err)
			}
		} else {
			if err := encRopeDecodeAt(encI, scr[r].kProj, scr[r].kProj, 0, 0, offPacked, uint(r*4), layerRopeFreqs, lkv, lhd, rotDim, rbase, s.ropeScale); err != nil {
				return fail(err)
			}
		}
		if s.valueNormOnes != nil {
			if err := encRMSNormRowsBF16(encI, scr[r].vProj, s.valueNormOnes, scr[r].vProj, 0, 0, 0, lkv, lhd, s.eps); err != nil {
				return fail(err)
			}
		}
	}
	memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
	land := func(r int) error {
		if cache.quantQ8 {
			if err := encKVQ8Store(encI, scr[r].kProj, rows[r].kPage, rows[r].rowOff, rows[r].kScalePage, rows[r].scaleOff, kvDim); err != nil {
				return err
			}
			return encKVQ8Store(encI, scr[r].vProj, rows[r].vPage, rows[r].rowOff, rows[r].vScalePage, rows[r].scaleOff, kvDim)
		}
		if err := encCopyBF16Contig(encI, scr[r].kProj, rows[r].kPage, 0, rows[r].rowOff, kvDim); err != nil {
			return err
		}
		return encCopyBF16Contig(encI, scr[r].vProj, rows[r].vPage, 0, rows[r].rowOff, kvDim)
	}
	if ringMode {
		// per-row land -> scan: eviction order IS the byte contract on a ring
		for r := range K {
			if err := land(r); err != nil {
				return fail(err)
			}
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			rows[r].plan.emitP1s(encI)
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			rows[r].plan.emitP2(encI)
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		}
	} else {
		// W4: land ALL rows, W5/W6: every row's SDPA passes — barrier-free
		// across rows within each wave
		for r := range K {
			if err := land(r); err != nil {
				return fail(err)
			}
		}
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		for r := range K {
			rows[r].plan.emitP1s(encI)
		}
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		for r := range K {
			rows[r].plan.emitP2(encI)
		}
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
	}
	// W7: output projections
	for r := range K {
		if err := s.lb[li].proj.project(encI, scr[r].attn, scr[r].attnOut, 0, projO); err != nil {
			return fail(err)
		}
	}
	memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
	// W8: residual + post-attention norm into the H slab
	for r := range K {
		if err := encResidualMaybeNormAt(encI, slabIn, uint(r*rowBytes), scr[r].attnOut, 0, scr[r].normed, slabH, uint(r*rowBytes), s.lb[li].postAttnNorm, s.dModel, s.eps); err != nil {
			return fail(err)
		}
	}
	endEncodingFast(enc)
	return nil
}
