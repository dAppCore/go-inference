// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync/atomic"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// mtp_rows_driver.go — the #53 WIRING lane: a layer-major verify driver that finally consults
// mtpRowsMoEForced (mtp_rows_moe.go), which every prior lane left "recognised but not consulted".
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

// mtpRowsDriverEligible reports whether the WHOLE verify block — every layer, not just the MoE
// weights — can take the layer-major batched driver. false always means "the caller keeps the
// row-major per-row lane" — this function is the ONLY gate; stepRowsMoEBatched trusts it and
// treats any mid-block inconsistency as an error, not a fallback (see stepRowsMoEBatched's doc).
func mtpRowsDriverEligible(s *archDecodeState) bool {
	if s == nil || len(s.specs) == 0 || len(s.lb) != len(s.specs) || s.dModel <= 0 || s.dFF <= 0 {
		return false
	}
	// state-level: no diagnostic/test hook, no PLE tower, no chained live-decode tail, no resident
	// ICB replay. icb is nil by construction whenever a session loads quant MoE (session build's
	// icbEligible excludes MoE) — checked here anyway as a hard guarantee, not an assumption.
	if s.trace || s.gpuProf != nil || s.chainTail != nil || len(s.ple) > 0 || s.icb != nil {
		return false
	}
	if layerSpanProbeForTest != nil || captureLayerHiddens {
		return false
	}
	for li := range s.specs {
		spec := s.specs[li]
		if spec.Mixer == model.MixerGatedDelta {
			return false // Qwen3.5 gated-delta recurrence: no KV cache, a different mixer entirely
		}
		if s.gatedAttn != nil && li < len(s.gatedAttn) && s.gatedAttn[li] != nil {
			return false // Qwen3.5 gated full-attention (attn_output_gate): a different fused lane
		}
		if !spec.OwnsCache() {
			return false // cross-layer KV sharing: not implemented (encAttnHalfShared[Paged]'s shape)
		}
		if s.kvTQState.on(li) {
			return false // TurboQuant KV state lane: not implemented by this driver
		}
		moeQ := moeQuantAt(s.moeQuant, li)
		if moeQ == nil {
			return false // uniform-MoE only: a dense or bf16-MoE layer is out of scope
		}
		if !mtpRowsMoEEligible(*moeQ, s.dModel, s.dFF) {
			return false
		}
	}
	return true
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
		if cache := s.layerPagedKV(li); cache != nil {
			hSlabHost, err = s.encRowsAttnPaged(cache, curInHost, offPacked, li, K, rowBytes, startPos, lhd, lkv, slideW, rotDim, rbase, layerRopeFreqs)
		} else {
			hSlabHost, err = s.encRowsAttnLinear(curInHost, offPacked, li, K, rowBytes, startPos, lhd, lkv, slideW, rotDim, rbase, layerRopeFreqs)
		}
		if err != nil {
			return nil, err
		}

		moeQ := moeQuantAt(s.moeQuant, li)
		if moeQ == nil {
			return nil, core.NewError("native.archDecodeState.stepRowsMoEBatched: layer lost its quant MoE weights mid-block")
		}
		outHost, ok, err := mtpRowsMoEBatched(hSlabHost, *moeQ, s.dModel, s.dFF, K, s.eps)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, core.NewError("native.archDecodeState.stepRowsMoEBatched: mtpRowsMoEBatched declined a layer after mtpRowsDriverEligible passed")
		}

		if s.lb[li].layerScalar != nil {
			outHost, err = s.applyLayerScalarRows(outHost, s.lb[li].layerScalar, K, rowBytes)
			if err != nil {
				return nil, err
			}
		}
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
// byte-exact greedy verify lane (verifyAssistantDraftHiddens, exact=true) when LTHN_MTP_ROWS_MOE=1
// (mtpRowsMoEForced). ok=false (nil error) means declined — the caller falls back to the existing
// per-row stepID loop UNCHANGED; nothing here has side-effected the session (no embedding, no GPU
// work) unless mtpRowsDriverEligible already passed.
func (s *ArchSession) verifyRowsMoEBatchedHiddens(draftTokens []int32, rowBytes int) ([][]byte, bool, error) {
	K := len(draftTokens)
	if K < 1 || s.perLayerInput != nil || !mtpRowsDriverEligible(&s.state) {
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
	return rows, true, nil
}
