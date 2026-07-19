// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// decode_batched_tq.go — the BATCHED/chunked TurboQuant prefill (#48 slice):
// the batched dense pass made TQ-aware so a `-kv-cache turboquant[:N]` session
// prefills at batched speed instead of falling back to the per-token replay.
//
// The store side reuses the per-token lthn_tq_kv_store kernel by grid width
// (kvHeads·numRows landings in one dispatch — encTQKVStoreRows). The read side
// is the design's core: a qualifying GLOBAL owner's PRIOR history codes are
// reconstructed (Πᵀ·γ·centroid) into a reusable per-layer bf16 SCRATCH, and the
// existing bf16 chunk-attention SDPA (multiQ / prompt-GEMM / flash) scores
// against the scratch UNCHANGED — no TQ-aware batched SDPA kernel. The scratch
// is INCREMENTAL across chunks within one prefill: chunk c appends only the rows
// produced since the scratch's high-water mark (never re-dequantising the whole
// history per chunk — that would be the O(n²) trap). The persistent decode cache
// stays codes-only; the scratch is prefill-transient and freed when decode
// begins (releaseTQPrefillScratch), so the deep-context residency win is kept
// during generation. Numerically the scratch read equals what the decode SDPA
// scores against the same codes (q·k̃ = q·γΠᵀc = γ·(Πq)·c) — prefill-then-decode
// stays coherent with the all-sequential path by construction.

// batchedTQPrefillDisabledForTest forces the TQ batched prefill to DECLINE so the
// session falls through to the per-token replay — the A/B lever the parity test
// flips to compare the two prefill paths' decode logits (the honest way to
// exercise the sequential fallback now the batched pass is the default route).
var batchedTQPrefillDisabledForTest bool

// tqBatchedPrefillEnvOff is the production A/B lever (LTHN_TQ_BATCHED_PREFILL=0):
// pin every TQ prefill onto the per-token replay, the pre-#48 behaviour, for a
// clean before/after receipt on a live box.
var tqBatchedPrefillEnvOff = os.Getenv("LTHN_TQ_BATCHED_PREFILL") == "0"

// tqPrefillScratch holds each TurboQuant global owner layer's reusable bf16
// reconstruction of its code history — one K plane and one V plane per layer,
// laid out [rows × kvHeads × headDim] bf16 exactly as a bf16 KV cache, so the
// batched SDPA reads them with the ordinary bf16 strides. hwm[li] is the number
// of rows already reconstructed into that layer's scratch (the incremental
// high-water mark). Buffers are sized to maxLen once (prefill never re-allocs
// mid-run, which would drop the accumulated history) and released at the
// prefill→decode transition.
type tqPrefillScratch struct {
	k, v    []metal.MTLBuffer
	hwm     []int
	rowsCap []int
}

// ensureTQPrefill lazily builds the per-session TQ prefill scratch keyed by
// layer count. The buffers themselves are allocated on first touch per layer
// (ensureLayer), so a session whose prefill never reaches the batched TQ lane
// pays nothing.
func (s *archDecodeState) ensureTQPrefill(nLayers int) *tqPrefillScratch {
	if s.tqPrefill == nil {
		s.tqPrefill = &tqPrefillScratch{
			k:       make([]metal.MTLBuffer, nLayers),
			v:       make([]metal.MTLBuffer, nLayers),
			hwm:     make([]int, nLayers),
			rowsCap: make([]int, nLayers),
		}
	}
	return s.tqPrefill
}

// ensureLayer allocates layer li's K/V scratch planes at maxLen rows (the code
// cache's own row capacity for a global owner) the first time the layer lands a
// batched TQ chunk. maxLen is the hard cap on any chunk's basePos+K, so the
// plane never grows mid-prefill.
func (sc *tqPrefillScratch) ensureLayer(li, maxLen, kvDim int) {
	if sc.k[li] != nil {
		return
	}
	sc.k[li] = scratchBF16(maxLen * kvDim)
	sc.v[li] = scratchBF16(maxLen * kvDim)
	sc.rowsCap[li] = maxLen
	sc.hwm[li] = 0
}

// tqBatchedLanding lands the chunk's staged (roped/normed) K/V rows into layer
// li's code caches and incrementally reconstructs the [lo, basePos+K) history
// into the bf16 scratch, returning the scratch planes the SDPA reads. Store,
// dequant and the downstream SDPA share ONE encoder — Metal's in-encoder buffer
// hazard tracking orders store→dequant (both touch the codes) and dequant→SDPA
// (both touch the scratch), so the chunk attends its own just-landed rows
// exactly as the sequential replay does.
//
// lo = min(hwm, basePos): a contiguous prefill dequants only the current chunk
// (hwm == basePos, O(K)); a fresh prefill at basePos 0 covers [0, K); a
// turn-append after decode fills the gap the decode step's per-token stores left
// (their codes are in the cache, never in this scratch).
func (s *archDecodeState) tqBatchedLanding(enc metal.MTLComputeCommandEncoder, li, basePos, K int, kStage, vStage metal.MTLBuffer) (kScratch, vScratch metal.MTLBuffer, err error) {
	tq := s.icb.kvTQ
	lkv := kvHeadsOf(s.specs[li], s.nKVHeads)
	lhd := headDimOf(s.specs[li], s.headDim)
	kvDim := lkv * lhd

	// batched store: the whole chunk's staged rows → code cache + γ at basePos.
	if err = encTQKVStoreRows(enc, kStage, s.icb.kCaches[li], uint(basePos*tq.kRowBytes[li]),
		tq.kGammas[li], uint(basePos*tq.gammaRowBytes[li]), K, lkv, lhd, tq.kBits); err != nil {
		return nil, nil, err
	}
	if err = encTQKVStoreRows(enc, vStage, s.icb.vCaches[li], uint(basePos*tq.vRowBytes[li]),
		tq.vGammas[li], uint(basePos*tq.gammaRowBytes[li]), K, lkv, lhd, tq.vBits); err != nil {
		return nil, nil, err
	}

	sc := s.ensureTQPrefill(len(s.specs))
	sc.ensureLayer(li, s.maxLen, kvDim)
	end := basePos + K
	lo := sc.hwm[li]
	if basePos < lo {
		lo = basePos // re-visit: re-cover the current chunk's fresh codes
	}
	if lo < end {
		n := end - lo
		if err = encTQKVDequantRows(enc, s.icb.kCaches[li], uint(lo*tq.kRowBytes[li]),
			tq.kGammas[li], uint(lo*tq.gammaRowBytes[li]), sc.k[li], uint(lo*kvDim*bf16Size), n, lkv, lhd, tq.kBits); err != nil {
			return nil, nil, err
		}
		if err = encTQKVDequantRows(enc, s.icb.vCaches[li], uint(lo*tq.vRowBytes[li]),
			tq.vGammas[li], uint(lo*tq.gammaRowBytes[li]), sc.v[li], uint(lo*kvDim*bf16Size), n, lkv, lhd, tq.vBits); err != nil {
			return nil, nil, err
		}
	}
	sc.hwm[li] = end
	return sc.k[li], sc.v[li], nil
}

// tqBatchedPrefillUsable reports whether a TQ owner layer's bit widths and head
// dim are servable by the batched store+dequant pair — the per-layer gate the
// batched pass checks before committing to the TQ lane (a miss declines the
// whole chunk to the per-token replay, which is TQ-aware by recording).
func (s *archDecodeState) tqBatchedPrefillUsable() bool {
	tq := s.icb.kvTQ
	if tq == nil {
		return false
	}
	if !gpuHasTQKVStore(tq.kBits) || !gpuHasTQKVStore(tq.vBits) ||
		!gpuHasTQKVDequant(tq.kBits) || !gpuHasTQKVDequant(tq.vBits) {
		return false
	}
	for li := range s.specs {
		if !tq.on(li) {
			continue
		}
		lhd := headDimOf(s.specs[li], s.headDim)
		if !tqKVHeadDimOK(lhd) || !gpuHasSDPAMultiQ(lhd) {
			return false
		}
		// the batched read scores over the ordinary bf16 SDPA — a TQ owner is
		// always GLOBAL (allocArchICBCachesTQ), so a sliding-window batch shape
		// never reaches here.
		if s.specs[li].Attention != model.GlobalAttention {
			return false
		}
		// the whole-chunk batched-rope landing (foldAttn) needs the fused
		// qk-norm+rope shape: rows-capable projections and both norm weights on
		// the owner. A miss keeps the layer on the per-token replay.
		if li >= len(s.lb) || s.lb[li].proj == nil || !s.lb[li].proj.rowsCapable() ||
			s.lb[li].qNorm.buf == nil || s.lb[li].kNorm.buf == nil || s.lb[li].sinks.buf != nil {
			return false
		}
	}
	return true
}

// releaseTQPrefillScratch frees the prefill scratch planes and resets the
// high-water marks. Called at the prefill→decode transition (the codes-reading
// decode SDPA needs no bf16 mirror, so the deep-context residency win returns)
// and at session teardown. A later prefill re-allocs and re-fills from the
// codes, so the reset stays correct across multi-turn.
func (s *archDecodeState) releaseTQPrefillScratch() {
	if s == nil || s.tqPrefill == nil {
		return
	}
	sc := s.tqPrefill
	releaseDeviceBuffers(sc.k...)
	releaseDeviceBuffers(sc.v...)
	for i := range sc.k {
		sc.k[i], sc.v[i], sc.hwm[i], sc.rowsCap[i] = nil, nil, 0, 0
	}
}
